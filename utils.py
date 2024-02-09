import asyncio
import csv
import difflib
import gym
import json
import logging
import numpy as np
import os
import pathlib
import pdb
import random
import sys
import time
import torch
import tqdm

from nle.nethack.actions import ACTIONS
from transformers import GenerationConfig
from transformers import StoppingCriteria

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, os.path.join(PROJECT_PATH, "external/nle-language-wrapper"))
from nle_language_wrapper import NLELanguageWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv


NH_ACTION_STR_TO_IDX = {str(ACTIONS[i]): i for i in range(len(ACTIONS))}
NH_ACTION_IDX_TO_STR = {v: k for (k, v) in NH_ACTION_STR_TO_IDX.items()}


class UnrollLengthCriteria(StoppingCriteria):
    def __init__(self, unroll_length, stop_token_id, num_return_sequences):
        assert isinstance(unroll_length, int)
        self.unroll_length = unroll_length
        self.stop_token_id = stop_token_id
        self.counts_per_sequence = torch.zeros((num_return_sequences,))

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            if input_ids[i][-1] == self.stop_token_id:
                self.counts_per_sequence[i] += 1
                if self.counts_per_sequence[i] >= self.unroll_length:
                    sequences_should_be_stopped.append(True)
                    continue
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


def get_diff(prompt_uno, prompt_dos, n=0):
    proc_prompt_uno = prompt_uno.strip().splitlines()
    proc_prompt_dos = prompt_dos.strip().splitlines()

    out = "\n".join(
        [
            line
            for i, line in enumerate(
                difflib.unified_diff(
                    proc_prompt_uno,
                    proc_prompt_dos,
                    n=n,
                    fromfile="file1",
                    tofile="file2",
                    lineterm="",
                )
            )
            if i > 1
        ]
    )
    return out


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def log(stats, step, is_global=False, wandb=False):
    stats_values = {}
    prefix = "global/" if is_global else "local/"
    for k, v in stats.items():
        stats_values[prefix + k] = v.result()
        v.reset()

    logging.info(stats_values)
    if not is_global:
        record.log_to_file(**stats_values)

    if FLAGS.wandb:
        wandb.log(stats_values, step=step)


def pretty_print_ttyrec(observation):
    nrows, ncols = observation["tty_chars"].shape
    ob_as_array = np.array([chr(oo) for oo in observation["tty_chars"].flatten()])
    ob_as_array = ob_as_array.reshape(nrows, ncols)  ## tty_chars default shape
    rows = []
    for row in range(nrows):
        ob_row = ob_as_array[row]
        ob_row_as_str = "".join([oo for oo in ob_row])
        rows += [ob_row_as_str]
    ob_as_str = "\n".join(rows)
    print(ob_as_str)
    return ob_as_str


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    load_in_half=False,
    gptq_model=False,
    use_fast_tokenizer=False,
    padding_side="left",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    if "longformer" in tokenizer_name_or_path:
        tokenizer = LongformerTokenizerFast.from_pretrained(tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, use_fast=use_fast_tokenizer
        )

    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

            if torch.cuda.is_available():
                model = model.cuda()
        if load_in_half:
            print("loading in half")
            model = model.half()

    model.eval()

    model = model.cuda()

    return model, tokenizer
