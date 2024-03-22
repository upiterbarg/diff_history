import difflib
import functools
import gym
import h5py
import json
import jsonlines
import multiprocessing
import numpy as np
import os
import pdb
import threading
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
from nle.nethack.actions import ACTIONS
from tqdm import tqdm


import pathlib
import sys

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, PROJECT_PATH)
from action_textmap import (
    nle_action_textmap,
    nle_comp_preqs,
    nle_obs_preqs,
    special_tokens_interaction_history,
)
from instruction_encode_templates import *
from utils import set_seed_everywhere, get_diff


def form_prompt(data, obs_preqs):
    return "\n".join(
        [
            "%s[\n%s\n]" % (obs_preqs[key], data[key])
            for key in (
                "txt_blstats",
                "txt_glyphs",
                "txt_message",
                "txt_inventory",
                "txt_cursor",
            )
        ]
    )


def get_samples(source_dir, seq, nsamples):
    ##  SAMPLE ##
    ds_metadata = []
    for fn in os.listdir(source_dir):
        _, ep_len = fn[: fn.find(".")].split("_")
        ep_len = int(ep_len)
        ds_metadata += [(fn, ep_len)]

    ds_metadata = sorted(ds_metadata, key=lambda tup: tup[1])  ### sort by game seeds
    ep_lens = [d[-1] for d in ds_metadata]

    ep_lens = np.array(ep_lens)

    partial_sums = [0]
    for i in range(ep_lens.shape[0]):
        partial_sums += [partial_sums[-1] + ep_lens[i] - seq - 1]
    partial_sums = np.array(partial_sums)
    lsamples = np.random.randint(low=0, high=partial_sums[-1], size=nsamples)
    eps_ids = np.array([i for i in range(ep_lens.shape[0] + 1)])
    samples = []
    for lsample in lsamples:
        eps_id = eps_ids[partial_sums > lsample][0] - 1
        ts_id = lsample - partial_sums[eps_id] + 1
        samples += [[ds_metadata[eps_id][0], ts_id.item()]]  ## unpack filename, ts id

    return samples


def load_and_process_chunks(
    idx,
    samples,
    diff_fn,
    fulltext_fn,
    source_dir,
    seq=64,
    observation_tok=special_tokens_interaction_history["observation"],
    obs_preqs=nle_obs_preqs,
    action_map=nle_action_textmap,
    comp_preqs=nle_comp_preqs,
    instruction="You are an agent playing NetHack. Predict the next actions.",
):
    def _process_helper_diff_observation(
        data,
    ):
        query = form_prompt(data[0], obs_preqs)
        prompts = []
        actions = []
        for i in range(seq):
            if i < seq - 1:
                prompts += [form_prompt(data[i + 1], obs_preqs=obs_preqs)]
            actions += [data[i]["txt_action"]]

        diffs = []
        for i in range(seq - 1):
            if i == 0:
                diffs += [get_diff(query, prompts[i], n=0)]
            else:
                diffs += [get_diff(prompts[i - 1], prompts[i], n=0)]

        completion = ""
        for j in range(seq):
            completion += "\n%s%s" % (
                comp_preqs["action"],
                actions[j],
            )
            if j < seq - 1:
                completion += "\n%s\n%s" % (observation_tok, diffs[j])

        return encode_instruction_example(
            instruction,
            query,
            completion,
            random_template=False,
            eos_token=None,
        )

    def _process_helper_raw_observation(
        data,
    ):
        query = form_prompt(data[0], obs_preqs)
        prompts = []
        actions = []
        for i in range(seq):
            if i < seq - 1:
                prompts += [form_prompt(data[i + 1], obs_preqs=obs_preqs)]
            actions += [data[i]["txt_action"]]

        obs = []
        for i in range(seq - 1):
            obs += [prompts[i]]

        completion = ""
        for j in range(seq):
            completion += "\n%s%s" % (
                comp_preqs["action"],
                actions[j],
            )
            if j < seq - 1:
                completion += "\n%s\n%s" % (observation_tok, obs[j])

        return encode_instruction_example(
            instruction,
            query,
            completion,
            random_template=False,
            eos_token=None,
        )

    with tqdm(total=len(samples), position=idx, desc=str(os.getpid())) as pbar:
        for eps_fn, start_ts_id in samples:
            ## only load relevant part of data
            chunk = []
            with jsonlines.open(os.path.join(source_dir, eps_fn), "r") as reader:
                for i, datum in enumerate(reader):
                    #### first item in file is metadata string --> skip
                    if (i - 1) < start_ts_id:
                        continue
                    elif (i - 1) >= start_ts_id + seq:
                        break
                    chunk += [datum]

            diff_history = _process_helper_diff_observation(chunk)
            raw_history = _process_helper_raw_observation(chunk)

            with jsonlines.open(diff_fn, "a") as writer:
                writer.write_all([diff_history])

            with jsonlines.open(fulltext_fn, "a") as writer:
                writer.write_all([raw_history])

            pbar.update(1)

    return 1


def main(args):
    ## get samples
    samples = get_samples(args.source_dir, args.seq, args.nsamples)

    ## configure dump dirs
    os.makedirs(args.dump_dir, exist_ok=True)

    prefix = (
        args.source_dir[args.source_dir.rfind("/") + 1 :]
        .replace("-", "")
        .replace("_", "")
    )

    diff_fn = prefix + f"-n{len(samples)}-k{args.seq}-diff.jsonl"
    metadata_fn = prefix + f"-n{len(samples)}-k{args.seq}-metadata.jsonl"
    fulltext_fn = prefix + f"-n{len(samples)}-k{args.seq}-fulltext.jsonl"

    diff_fn = os.path.join(args.dump_dir, diff_fn)
    fulltext_fn = os.path.join(args.dump_dir, metadata_fn)
    metadata_fn = os.path.join(args.dump_dir, fulltext_fn)

    ## dump sample metadata
    with jsonlines.open(metadata_fn, "w") as writer:
        writer.write_all(samples)

    ## launch pool
    pool = multiprocessing.Pool(args.num_workers)

    gen_helper_fn = functools.partial(
        load_and_process_chunks,
        diff_fn=diff_fn,
        fulltext_fn=fulltext_fn,
        source_dir=args.source_dir,
        seq=args.seq,
    )

    num_samples_per_worker = len(samples) // args.num_workers + 1

    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(args.num_workers - 1)):
        gen_args += [[j, samples[start_i : start_i + num_samples_per_worker]]]
        start_i += num_samples_per_worker
    if len(samples) - start_i > 0:
        gen_args += [[args.num_workers - 1, samples[start_i:]]]

    pool = multiprocessing.Pool(args.num_workers)
    runs = [
        pool.apply_async(gen_helper_fn, args=gen_args[k])
        for k in range(args.num_workers)
        if len(gen_args) > k
    ]
    results = [p.get() for p in runs]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default=os.path.join(PROJECT_PATH, "nethack_experiments", "data", "all-10000"),
        type=str,
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=os.path.join(PROJECT_PATH, "nethack_experiments", "data", "processed"),
    )
    parser.add_argument("--nsamples", default=500000, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--seq", default=128, type=int)

    args = parser.parse_args()

    print("ARGS:", args)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
