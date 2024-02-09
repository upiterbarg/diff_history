import functools
import h5py
import json
import jsonlines
import multiprocessing
import pathlib
import pdb
import shutil
import subprocess
import sys
import termios
import time
import traceback
import tty
import warnings

from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
from nle.nethack.actions import ACTIONS
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

import gym
import nle.nethack as nh
import numpy as np
import os
import pathlib
import sys

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, PROJECT_PATH)
from action_textmap import nle_action_textmap

sys.path.insert(0, os.path.join(PROJECT_PATH, "external/autoascend"))
from autoascend import agent as agent_lib
from autoascend.env_wrapper import EnvWrapper

sys.path.insert(0, os.path.join(PROJECT_PATH, "external/nle-language-wrapper"))
from nle_language_wrapper import NLELanguageWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv


NH_ACTION_STR_TO_IDX = {str(ACTIONS[i]): i for i in range(len(ACTIONS))}
NH_ACTION_IDX_TO_STR = {v: k for (k, v) in NH_ACTION_STR_TO_IDX.items()}


def get_seeds(
    n,
    target_role,
    start_seed,
):
    if target_role is None:
        return np.array([s for s in range(start_seed, start_seed + n)])

    seed = start_seed
    with tqdm(total=n) as pbar:
        while not len(relevant_seeds) == n:
            candidate_seed = seed
            while 1:
                env = gym.make("NetHackChallenge-v0")
                env.seed(candidate_seed, candidate_seed)
                obs = env.reset()
                blstats = agent_lib.BLStats(*obs["blstats"])
                character_glyph = obs["glyphs"][blstats.y, blstats.x]
                if any(
                    [
                        nh.permonst(nh.glyph_to_mon(character_glyph)).mname.startswith(
                            role
                        )
                        for role in target_role
                    ]
                ):
                    break
                candidate_seed += 10**5
                candidate_seed = candidate_seed % sys.maxsize
                env.close()
            if (
                not candidate_seed in relevant_seeds
                and not candidate_seed in restricted_seeds
            ):
                relevant_seeds += [candidate_seed]
                pbar.update(1)
            seed += 1
    return np.array(relevant_seeds).astype(int)


def gen_and_write_episode(
    idx, start_i, total_rollouts, data_dir, seeds, vision_version=False
):
    nle_language = NLELanguageObsv()

    with tqdm(total=total_rollouts, position=idx, desc=str(os.getpid())) as pbar:
        for game_id in range(start_i, start_i + total_rollouts):
            # unpack game seed
            if game_id >= seeds.shape[0]:
                break
            game_seed = seeds[game_id]

            env = EnvWrapper(
                gym.make("NetHackChallenge-v0", no_progress_timeout=100),
                agent_args=dict(panic_on_errors=True, verbose=False),
                step_limit=10000000000,
            )

            env.env.seed(game_seed, game_seed)
            try:
                env.main()
            except BaseException:
                pass
            summary = env.get_summary()

            json_safe_summary = {}
            for key, val in summary.items():
                if (
                    isinstance(val, int)
                    or isinstance(val, str)
                    or isinstance(val, float)
                    or isinstance(val, tuple)
                ):
                    json_safe_summary[key] = val
                else:
                    json_safe_summary[key] = val.item()

            text_data = [json_safe_summary]

            data = env.get_data()

            for ts in range(len(data)):
                datum = data[ts]

                txt_blstats = nle_language.text_blstats(datum["blstats"]).decode(
                    "latin-1"
                )
                txt_glyphs = nle_language.text_glyphs(
                    datum["glyphs"], datum["blstats"]
                ).decode("latin-1")
                txt_message = nle_language.text_message(datum["tty_chars"]).decode(
                    "latin-1"
                )
                txt_inventory = nle_language.text_inventory(
                    datum["inv_strs"], datum["inv_letters"]
                ).decode("latin-1")
                txt_cursor = (
                    nle_language.text_cursor(
                        datum["glyphs"], datum["blstats"], datum["tty_chars"]
                    ).decode("latin-1"),
                )
                if ts < len(data) - 1:
                    txt_action = nle_action_textmap[data[ts + 1]["action"]]
                else:
                    txt_action = "esc"

                text_datum = {
                    "txt_blstats": txt_blstats,
                    "txt_glyphs": txt_glyphs,
                    "txt_message": txt_message,
                    "txt_inventory": txt_inventory,
                    "txt_cursor": txt_cursor,
                    "txt_action": txt_action,
                }

                if vision_version:
                    vision_datum = {
                        "tty_chars": datum["tty_chars"].tolist(),
                        "tty_colors": datum["tty_colors"].tolist(),
                        "tty_cursor": datum["tty_cursor"].tolist(),
                    }
                    if ts < len(data) - 1:
                        action = NH_ACTION_STR_TO_IDX[data[ts + 1]["action"]]
                    else:
                        action = NH_ACTION_STR_TO_IDX["Command.ESC"]
                    vision_datum["int_action"] = action

                    text_datum = {**text_datum, **vision_datum}

                text_data += [text_datum]

            fn = f"{game_seed}_{len(data)}.jsonl"

            with jsonlines.open(os.path.join(data_dir, fn), "w") as writer:
                writer.write_all(text_data)

            pbar.update(1)

    return 1


def create_dataset(args):
    # configure data directory
    role = "-".join(args.role) if str(args.role) != "None" else "all"
    data_dir = os.path.join(args.base_dir, f"{role}-{args.episodes}")
    if args.vision_version:
        data_dir += f"-vision"

    relevant_seeds = get_seeds(
        args.episodes,
        args.role,
        args.seed,
    )

    if os.path.isdir(data_dir):
        seeds_done = [int(s.split("_")[0]) for s in os.listdir(data_dir)]

        relevant_seeds = np.array(
            [seed for seed in relevant_seeds if not seed in seeds_done]
        )

        print("seeds unpacked!")
    else:
        os.makedirs(data_dir, exist_ok=True)

        # first determine n unique seeds
        relevant_seeds = get_seeds(
            args.episodes,
            args.role,
            args.seed,
        )

        print("seeds generated!")

    ## parallelize dataset generation + saving
    # configure process info

    cores_to_reserve = args.cores_to_reserve
    num_proc = min(
        multiprocessing.cpu_count() - cores_to_reserve, relevant_seeds.shape[0]
    )
    num_rollouts_per_proc = relevant_seeds.shape[0] // num_proc
    # configure generator w/ syntactic sugar
    gen_helper_fn = functools.partial(
        gen_and_write_episode,
        data_dir=data_dir,
        seeds=relevant_seeds,
        vision_version=args.vision_version,
    )
    # generate remaining args
    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(num_proc - 1)):
        gen_args += [[j, start_i, num_rollouts_per_proc]]
        start_i += num_rollouts_per_proc
    if relevant_seeds.shape[0] - start_i > 0:
        gen_args += [[num_proc - 1, start_i, relevant_seeds.shape[0] - start_i]]

    # run pool
    pool = multiprocessing.Pool(num_proc)
    runs = [
        pool.apply_async(gen_helper_fn, args=gen_args[k])
        for k in range(num_proc)
        if len(gen_args) > k
    ]
    results = [p.get() for p in runs]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir", default="data", type=str, help="dir where to store data"
    )
    parser.add_argument("--seed", default=0, type=int, help="Starting random seed")
    parser.add_argument("--vision_version", default=0, type=int)
    parser.add_argument("-n", "--episodes", type=int, default=1000)
    parser.add_argument(
        "--role",
        default=None,
        choices=(
            "arc",
            "bar",
            "cav",
            "hea",
            "kni",
            "mon",
            "pri",
            "ran",
            "rog",
            "sam",
            "tou",
            "val",
            "wiz",
        ),
        action="append",
    )
    parser.add_argument("--panic-on-errors", default=True, action="store_true")
    parser.add_argument(
        "--nonoverlapping_seeds_with",
        default="null",
        help="make sure the new dataset has non-overlapping seeds with the dataset at this path",
    )
    parser.add_argument("--cores_to_reserve", type=int, default=32)

    args = parser.parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, sys.maxsize)

    print("ARGS:", args)
    return args


def main():
    args = parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
