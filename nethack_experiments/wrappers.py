import argparse
import cv2
import gym
import json
import jsonlines
import numpy as np
import os
import pandas as pd
import pdb
import time
import torch
import tqdm

from accelerate import Accelerator
from datasets import load_dataset
from gym import Wrapper
from gym import spaces
from nle.env import NLE
from nle.env import tasks
from nle.nethack.actions import ACTIONS
from transformers import GenerationConfig
from transformers import StoppingCriteria

import pathlib
import sys


base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, PROJECT_PATH)
from instruction_encode_templates import *
from utils import load_hf_lm_and_tokenizer
from utils import set_seed_everywhere
from action_textmap import (
    nle_action_textmap,
    nle_comp_preqs,
    nle_obs_preqs,
    special_tokens_interaction_history,
)

sys.path.insert(0, os.path.join(PROJECT_PATH, "external/nle-language-wrapper"))
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from nle import nethack
from nle_language_wrapper import NLELanguageWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv
from numba import njit

import render_utils

SMALL_FONT_PATH = os.path.abspath(
    os.path.join(
        PROJECT_PATH,
        "external/dungeonsdata-neurips2022/experiment_code/render_utils/Hack-Regular.ttf",
    )
)

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right
# https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


class NLELMWrapper(Wrapper):
    def __init__(
        self,
        env,
        prediction_target="action",
        observation=False,
        kphist=False,
        random_template=True,
        include_interleave_in_prompt=False,
    ):
        super().__init__(env)
        assert isinstance(env, NLE), "Only NLE environments are supported"

        self._kphist = kphist
        self._observation = observation

        self.action_space = spaces.Space()
        self.observation_space = spaces.Dict(
            {
                "text_glyphs": spaces.Space(),
                "text_message": spaces.Space(),
                "text_blstats": spaces.Space(),
                "text_inventory": spaces.Space(),
                "text_cursor": spaces.Space(),
            }
        )
        self._instruction = (
            "You are an agent playing NetHack. Predict the next keypresses."
        )
        self._interleaving_token = special_tokens_interaction_history["action"]
        self._final_completion_prefix = None
        self._hist_preq = nle_obs_preqs["prev_action_seq"]
        self.random_template = random_template

        self._action_map = {
            nle_action_map[str(action)]: i for i, action in enumerate(ACTIONS)
        }
        self._reverse_action_map = {v: k for (k, v) in self._action_map.items()}
        self._nle_language = NLELanguageObsv()
        self._obs_preqs = nle_obs_preqs
        self._comp_preqs = nle_comp_preqs

        self.include_interleave_in_prompt = include_interleave_in_prompt

    @property
    def action_map(self):
        return self._action_map

    @property
    def reverse_action_map(self):
        return self._reverse_action_map

    @property
    def nle_language(self):
        return self._nle_language

    @property
    def obs_preqs(self):
        return self._obs_preqs

    @property
    def comp_preqs(self):
        return self._comp_preqs

    @property
    def instruction(self):
        return self._instruction

    @property
    def interleaving_token(self):
        return self._interleaving_token

    @property
    def spec(self):
        return self.env.spec

    def action_map_fn(self, action):
        if isinstance(action, str):
            return self.action_map[action]
        else:
            return self._reverse_action_map[action]

    def strategy_map_fn(self, strategy):
        return (
            strategy.replace("visit and search", "open_visit_search")
            .replace(" ", "_")
            .replace("fight", "fight2")
            .replace("explore", "explore_gather_identify")
        )

    def nle_obsv_to_language(self, nle_obsv):
        """Translate NLE Observation into a language observation.
        Args:
            nle_obsv (dict): NLE observation from the base environment
        Returns:
            (dict): language observation
        """
        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]
        tty_chars = nle_obsv["tty_chars"]
        return {
            "txt_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode(
                "latin-1"
            ),
            "txt_message": self.nle_language.text_message(tty_chars).decode("latin-1"),
            "txt_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "txt_inventory": self.nle_language.text_inventory(
                inv_strs, inv_letters
            ).decode("latin-1"),
            "txt_cursor": self.nle_language.text_cursor(
                glyphs, blstats, tty_cursor
            ).decode("latin-1"),
        }

    def promptify_nle_obsv(self, nle_obsv, history=None):
        obs = self.nle_obsv_to_language(nle_obsv)

        if self._observation:
            query = "\n".join(
                [
                    "%s[\n%s\n]" % (self.obs_preqs[key], obs[key])
                    for key in (
                        "txt_blstats",
                        "txt_glyphs",
                        "txt_message",
                        "txt_inventory",
                        "txt_cursor",
                    )
                ]
            )
        else:
            inter = "\n"
            query = inter.join(
                [
                    "%s\n%s" % (self.obs_preqs[key], obs[key])
                    for key in (
                        "txt_blstats",
                        "txt_glyphs",
                        "txt_message",
                        "txt_inventory",
                        "txt_cursor",
                    )
                ]
            )

        if not self._kphist:
            if history == " ":
                query = "%s\n" % (self._hist_preq,) + "\n\n" + query
            elif not history is None:
                query = "%s\n%s" % (self._hist_preq, history) + "\n\n" + query
        else:
            if history == []:
                query = "%s\n" % (self._hist_preq,) + "\n\n" + query
            elif not history is None:
                prev_action_seqs = "".join(
                    [
                        "%s%s" % (self._comp_preqs["action"], prev_action)
                        for prev_action in history
                    ]
                )
                query = "%s\n%s" % (self._hist_preq, prev_action_seqs) + "\n\n" + query

        query = "\n" + query

        out = encode_instruction_example(
            self.instruction,
            query,
            " ",
            random_template=self.random_template,
            eos_token=None,
        )["prompt"]

        if not self._final_completion_prefix is None:
            out = out.strip() + self._final_completion_prefix

        if self.include_interleave_in_prompt:
            out += self._interleaving_token

        return out, obs["txt_glyphs"], obs["txt_message"]

    def reset(self, history=None, **kwargs):
        nle_obsv = self.env.reset(**kwargs)
        prompt, txt_glyphs, txt_message = self.promptify_nle_obsv(
            nle_obsv, history=history
        )
        nle_obsv["prompt"] = prompt
        nle_obsv["txt_glyphs"] = txt_glyphs
        nle_obsv["txt_message"] = txt_message
        return nle_obsv

    def step(self, action, output_caction=False, history=None):
        c_action = self.action_map[action]
        nle_obsv, reward, done, info = self.env.step(c_action)
        prompt, txt_glyphs, txt_message = self.promptify_nle_obsv(
            nle_obsv, history=history
        )
        nle_obsv["prompt"] = prompt
        nle_obsv["txt_glyphs"] = txt_glyphs
        nle_obsv["txt_message"] = txt_message
        if output_caction:
            return nle_obsv, reward, done, info, c_action
        return nle_obsv, reward, done, info


@njit
def _tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w,
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[
                :, h_pixel : h_pixel + char_height, w_pixel : w_pixel + char_width
            ] = char_array[char, color]


def _initialize_char_array(font_size, rescale_font_size):
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    font = ImageFont.truetype(SMALL_FONT_PATH, font_size)
    dummy_text = "".join(
        [(chr(i) if chr(i).isprintable() else " ") for i in range(256)]
    )
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    char_width = rescale_font_size[0]
    char_height = rescale_font_size[1]

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
        blstats_cursor=False,
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.blstats_cursor = blstats_cursor

        self.half_crop_size = crop_size // 2
        self.output_height_chars = crop_size
        self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width,
        )

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            if self.blstats_cursor:
                center_x, center_y = obs["blstats"][:2]
            else:
                center_y, center_x = obs["tty_cursor"]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        _tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )

        obs["screen_image"] = out_image
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._render_text_to_image(obs)
        return obs


class RenderCharImagesWithNumpyWrapperV2(gym.Wrapper):
    """
    Same as V1, but simpler and faster.
    """

    def __init__(
        self,
        env,
        font_size=9,
        crop_size=12,
        rescale_font_size=(6, 6),
    ):
        super().__init__(env)
        self.char_array = _initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)
        self.char_array = np.ascontiguousarray(self.char_array)
        self.crop_size = crop_size

        crop_rows = crop_size or nethack.nethack.TERMINAL_SHAPE[0]
        crop_cols = crop_size or nethack.nethack.TERMINAL_SHAPE[1]

        self.chw_image_shape = (
            3,
            crop_rows * self.char_height,
            crop_cols * self.char_width,
        )

        obs_spaces = {
            "screen_image": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k not in ["tty_chars", "tty_colors"]
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _populate_obs(self, obs):
        screen = np.zeros(self.chw_image_shape, order="C", dtype=np.uint8)
        render_utils.render_crop(
            obs["tty_chars"],
            obs["tty_colors"],
            obs["tty_cursor"],
            self.char_array,
            screen,
            crop_size=self.crop_size,
        )
        obs["screen_image"] = screen

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._populate_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._populate_obs(obs)
        return obs
