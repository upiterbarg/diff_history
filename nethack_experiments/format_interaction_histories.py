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


def faster_load_and_process_chunks(
    idx,
    files,
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

    relevant_samples_by_file = {file: [] for file in files}

    total_samples = 0
    for sample in samples:
        eps_fn, start_ts_id = sample
        if not eps_fn in files:
            continue
        relevant_samples_by_file[eps_fn] += [start_ts_id]
        total_samples += 1

    with tqdm(total=total_samples, position=idx, desc=str(os.getpid())) as pbar:
        for eps_fn in relevant_samples_by_file:
            ## only load relevant part of data
            chunks = {
                start_ts_id: [] for start_ts_id in relevant_samples_by_file[eps_fn]
            }

            relevant_ts_ids = []
            chunk_lookup = {}
            for start_ts_id in relevant_samples_by_file[eps_fn]:
                ts_ids = list(range(start_ts_id, start_ts_id + seq + 1))
                for ts_id in ts_ids:
                    if ts_id in chunk_lookup:
                        chunk_lookup[ts_id] += [start_ts_id]
                    else:
                        chunk_lookup[ts_id] = [start_ts_id]
                relevant_ts_ids += ts_ids

            with jsonlines.open(os.path.join(source_dir, eps_fn), "r") as reader:
                for i, datum in enumerate(reader):
                    #### first item in file is metadata string --> skip
                    if (i - 1) in relevant_ts_ids:
                        for chunk_id in chunk_lookup[(i - 1)]:
                            chunks[chunk_id] += [datum]
                            if len(chunks[chunk_id]) == (seq + 1):
                                pbar.update(1)
                    if (i - 1) > max(relevant_ts_ids):
                        break

            diff_histories = []
            raw_histories = []
            for start_ts_id in relevant_samples_by_file[eps_fn]:
                chunk = chunks[start_ts_id]
                diff_histories += [_process_helper_diff_observation(chunk)]
                raw_histories += [_process_helper_raw_observation(chunk)]

            with jsonlines.open(diff_fn, "a") as writer:
                writer.write_all(diff_histories)

            with jsonlines.open(fulltext_fn, "a") as writer:
                writer.write_all(raw_histories)

    return 1


def main(args):
    ## get samples
    samples = get_samples(args.source_dir, args.seq, args.nsamples)
    samples.sort(key=lambda x: x[1])
    samples.sort(key=lambda x: x[0])

    sampled_files = list(set([x[0] for x in samples]))

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
    fulltext_fn = os.path.join(args.dump_dir, fulltext_fn)
    metadata_fn = os.path.join(args.dump_dir, metadata_fn)

    # check whether files exist --> raise errors
    assert not os.path.exists(
        fulltext_fn
    ), f"A file already exists as {fulltext_fn}! Please remove this file and relaunch this script."
    assert not os.path.exists(
        diff_fn
    ), f"A file already exists as {diff_fn}! Please remove this file and relaunch this script."
    assert not os.path.exists(
        metadata_fn
    ), f"A file already exists as {metadata_fn}! Please remove this file and relaunch this script."

    ## dump sample metadata
    with jsonlines.open(metadata_fn, "w") as writer:
        writer.write_all(samples)

    args.num_workers = min(args.num_workers, len(sampled_files))

    ## launch pool
    pool = multiprocessing.Pool(args.num_workers)

    gen_helper_fn = functools.partial(
        faster_load_and_process_chunks,
        diff_fn=diff_fn,
        fulltext_fn=fulltext_fn,
        source_dir=args.source_dir,
        seq=args.seq,
        samples=samples,
    )

    num_files_per_worker = len(sampled_files) // args.num_workers + 1

    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(args.num_workers - 1)):
        gen_args += [[j, sampled_files[start_i : start_i + num_files_per_worker]]]
        start_i += num_files_per_worker
    if len(sampled_files) - start_i > 0:
        gen_args += [[args.num_workers - 1, sampled_files[start_i:]]]

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
