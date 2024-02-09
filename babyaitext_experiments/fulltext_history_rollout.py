import argparse
import babyai_text
import difflib
import functools
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
from transformers import GenerationConfig
from transformers import StoppingCriteria

import pathlib
import sys

from lm_wrappers import BABYAI_ACTION_SPACE
from lm_wrappers import BabyAITextLMWrapper

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, os.path.join(PROJECT_PATH, "external/Grounding_LLMs_with_online_RL"))

from experiments.agents.bot import *

from babyai.paral_env_simple import ParallelEnv
from babyai.rl.utils import DictList


sys.path.insert(0, PROJECT_PATH)
from action_textmap import special_tokens_interaction_history
from utils import load_hf_lm_and_tokenizer
from utils import pretty_print_ttyrec
from utils import set_seed_everywhere

ACTION_TOKEN = special_tokens_interaction_history["action"]
OBSERVATION_TOKEN = special_tokens_interaction_history["observation"]


def history_rollout(
    seed,
    model,
    tokenizer,
    action_generation_config,
    args,
    scratchpad_token=OBSERVATION_TOKEN,
    max_tries=1,
    history=4,
    max_ctx_tokens=4000,
    history_max=True,
    env_name="BabyAI-MixedTrainLocal-v0",
    timeout=80,
):
    # config envs
    env = gym.make(env_name, num_dists=0)
    seed = int(seed)

    env = BabyAILlamaWrapper(env)
    env.env.seed(seed)

    interleaving_token = env.interleaving_token
    interleaving_token_id = tokenizer.encode_plus(env.interleaving_token)["input_ids"][
        -1
    ]
    scratchpad_token_id = tokenizer.encode_plus(scratchpad_token)["input_ids"][-1]

    def _query_model(
        prompt, unroll_length=1, stop_token_id=scratchpad_token_id, max_tokens=1024
    ):
        stopping_criteria = UnrollLengthCriteria(
            unroll_length=unroll_length,
            stop_token_id=stop_token_id,
            num_return_sequences=1,
        )

        tokenized_prompt = tokenizer(
            prompt, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        tokenized_prompt = {k: v.cuda() for (k, v) in tokenized_prompt.items()}
        tries = 0
        while 1:
            tries += 1
            out = model.generate(
                **tokenized_prompt,
                generation_config=action_generation_config,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(out)
            suffix = decoded[0][prompt.rfind(ACTION_TOKEN) :]

            if suffix.count(ACTION_TOKEN) > 0:
                break
            if tries > max_tries:
                return None, None, None

        keypresses = []
        diffs = []
        while len(keypresses) < unroll_length:
            skeypress = suffix[suffix.find(ACTION_TOKEN) + len(ACTION_TOKEN) :]
            keypress = (
                skeypress[: skeypress.find(OBSERVATION_TOKEN)]
                .replace(ACTION_TOKEN, "")
                .strip()
            )
            if "<" in keypress:
                keypress = keypress[: keypress.find("<")]

            keypresses += [keypress]
            suffix = suffix[suffix.find(OBSERVATION_TOKEN) :]
            diff = suffix[len(OBSERVATION_TOKEN) : suffix.find(ACTION_TOKEN)]
            diffs += [diff]
            suffix = suffix[suffix.find(ACTION_TOKEN) :]

        return [keypresses[-1]], [diffs[-1]], decoded[0]

    done = False
    counter = 0

    obs, infos = env.reset()

    mission = obs["mission"]

    query = obs["prompt"]
    ctx = query[:].strip() + "\n"
    ctx_idx = 0
    prompt_history = [obs["prompt"]]

    diffs_since_reset = []
    actions_since_reset = []
    steps_since_menu_open = 0

    imagined_diffs = []
    gt_diffs = []
    all_actions = []
    diff_idx = 0
    reward = 0

    while not done:
        if history_max:
            passed = False
            candidate_history = min(history, len(all_actions))

            while not passed and candidate_history >= 0:
                query = prompt_history[-(candidate_history + 1)].strip()
                ctx = obs["instruction"].strip() + "\n\n" + query.strip() + "\n"
                for i in range(-candidate_history, 0):
                    ctx += "\n%s%s" % (interleaving_token, all_actions[i])
                    ctx += "\n%s\n%s" % (
                        scratchpad_token,
                        prompt_history[i],
                    )
                tokenized_prompt = tokenizer(
                    ctx + "\n%s" % (interleaving_token),
                    padding="longest",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if tokenized_prompt["input_ids"].shape[1] < max_ctx_tokens:
                    passed = True
                else:
                    candidate_history -= 1

        else:
            candidate_history = min(history, len(all_actions))
            query = prompt_history[-(candidate_history + 1)].strip()
            ctx = obs["instruction"].strip() + "\n\n" + query.strip() + "\n"

            for i in range(-candidate_history, 0):
                ctx += "\n%s%s" % (interleaving_token, all_actions[i])
                ctx += "\n%s\n%s" % (
                    scratchpad_token,
                    prompt_history[i],
                )

        ctx += "\n%s" % (interleaving_token)

        actions, diffs, decoded_output = _query_model(ctx, unroll_length=1)

        imagined_diffs += [diffs]

        for action in actions:
            try:
                obs, reward, done, infos = env.step(action)
                true_diff = get_diff(prompt_history[diff_idx], obs["prompt"], n=0)

                gt_diffs += [true_diff]
                all_actions += [action]
                prompt_history += [obs["prompt"]]

            except:
                done = True

        if max(len(all_actions) - history, 0) != 0:
            ctx_idx += 1
        diff_idx += 1
        query = prompt_history[ctx_idx].strip()

        if len(all_actions) > timeout:
            done = True

    return mission, reward


def main():
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2_1k_chunk4_raw_masked/",
    )
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--pred_horizon", type=int, default=1)
    parser.add_argument("--n_rollouts", type=int, default=256)
    parser.add_argument("--beam_decoding", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--greedy_decoding", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--history", default=4)
    parser.add_argument(
        "--history_max", action="store_true", info="max-context history back-tracking"
    )
    parser.add_argument("--env_name", default="BabyAI-MixedTrainLocal-v0", type=str)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="babyaitext_chunked_raw/",
    )

    # parse args
    args = parser.parse_args()

    assert not (args.beam_decoding and args.greedy_decoding)

    try:
        args.history = int(args.history)
    except:
        pass

    # fill in fillable args
    model_rollout_dump_dir = os.path.join(
        args.base_dir,
        "-".join(
            args.model_name_or_path.split("/")[-1:]
            + [args.env_name]
            + ["beam"] * int(args.beam_decoding)
            + [f"nbeams{args.num_beams}"] * int(args.beam_decoding)
            + ["greedy"] * int(args.greedy_decoding)
        ),
    )

    if args.history > 0:
        model_rollout_dump_dir += f"-hist{args.history}"
    if args.history_max:
        model_rollout_dump_dir += "-histmax"

    if not os.path.exists(model_rollout_dump_dir):
        os.makedirs(model_rollout_dump_dir, exist_ok=True)

    out_fn = os.path.join(
        model_rollout_dump_dir, f"{args.n_rollouts}_{args.seed}.jsonl"
    )
    print(f"Saving eval summary to: {out_fn}")

    # set seed everywhere
    set_seed_everywhere(args.seed)

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
    )

    if args.beam_decoding:
        action_generation_config = GenerationConfig(
            max_new_tokens=24,
            decoder_start_token_id=0,
            eos_token_id=model.config.eos_token_id,
            pad_token=model.config.pad_token_id,
            num_beams=args.num_beams,
        )
    else:
        action_generation_config = GenerationConfig(
            max_new_tokens=24,
            decoder_start_token_id=0,
            eos_token_id=model.config.eos_token_id,
            pad_token=model.config.pad_token_id,
        )

    rollout_fn = functools.partial(
        history_rollout,
        history=args.history,
        model=model,
        tokenizer=tokenizer,
        action_generation_config=action_generation_config,
        args=args,
        history_max=args.history_max,
        env_name=args.env_name,
    )

    seed = args.seed
    rewards_by_seed = {}
    missions_by_seed = {}
    rewards = []
    for _ in range(args.n_rollouts):
        mission, reward = rollout_fn(seed)

        rewards_by_seed[seed] = reward
        missions_by_seed[seed] = mission
        rewards += [reward]

        seed += 1
        print(mission, reward, sum(rewards) / len(rewards))

    print("-" * 100)
    rewards = np.array(rewards)
    print(
        "mean: %.4f, median: %.4f, max: %.4f, min: %.4f"
        % (np.mean(rewards), np.median(rewards), np.max(rewards), np.min(rewards))
    )

    with jsonlines.open(out_fn, "w") as writer:
        writer.write_all([rewards_by_seed, missions_by_seed])


if __name__ == "__main__":
    main()
