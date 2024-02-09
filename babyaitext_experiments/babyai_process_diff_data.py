import difflib
import jsonlines
import numpy as np
import os
import pathlib
import pdb
import sys

from argparse import ArgumentParser
from tqdm import tqdm

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, PROJECT_PATH)
from instruction_encode_templates import *
from utils import (
    set_seed_everywhere,
    get_diff,
)

from action_textmap import special_tokens_interaction_history

ACTION_TOKEN = special_tokens_interaction_history["action"]
OBSERVATION_TOKEN = special_tokens_interaction_history["observation"]


N_EPISODES = 5000


def get_fulltext_interaction_history(
    example_episode,
    base_instruction="You are an agent playing BabyAI.",
    observation_tok=OBSERVATION_TOKEN,
    action_tok=ACTION_TOKEN,
    aspace=["turn left", "turn right", "go forward", "pick up", "drop", "toggle"],
):
    def _form_prompt(description):
        return "\n".join([d.replace("You see ", "") for d in description])

    mission = example_episode["mission"]
    aspace_str = ", ".join(aspace[:-1]) + ", and " + aspace[-1]

    instruction = "\n".join(
        [
            f"Your task is to {mission}.",
            f"You can take {len(aspace)} different actions: {aspace_str}.",
            f"Predict the next actions.",
        ]
    )

    query = _form_prompt(example_episode["descriptions"][0])
    seq = len(example_episode["descriptions"]) - 1

    prompts = []
    for i in range(seq):
        prompts += [_form_prompt(example_episode["descriptions"][i + 1])]

    actions = example_episode["actions"]

    completion = ""
    for j in range(seq):
        completion += "\n%s%s" % (
            action_tok,
            actions[j],
        )
        if j < seq - 1:
            completion += "\n%s\n%s" % (observation_tok, prompts[j])

    return encode_instruction_example(
        instruction,
        query,
        completion,
        random_template=False,
        eos_token=None,
    )


def get_diff_interaction_history(
    example_episode,
    base_instruction="You are an agent playing BabyAI.",
    observation_tok=OBSERVATION_TOKEN,
    action_tok=ACTION_TOKEN,
    aspace=["turn left", "turn right", "go forward", "pick up", "drop", "toggle"],
):
    def _form_prompt(description):
        return "\n".join([d.replace("You see ", "") for d in description])

    mission = example_episode["mission"]
    aspace_str = ", ".join(aspace[:-1]) + ", and " + aspace[-1]

    instruction = "\n".join(
        [
            f"Your task is to {mission}.",
            f"You can take {len(aspace)} different actions: {aspace_str}.",
        ]
    )

    query = _form_prompt(example_episode["descriptions"][0])
    seq = len(example_episode["descriptions"]) - 1

    prompts = []
    for i in range(seq):
        prompts += [_form_prompt(example_episode["descriptions"][i + 1])]

    diffs = []
    for i in range(seq):
        if i == 0:
            diffs += [get_diff(query, prompts[i], n=0)]
        else:
            diffs += [get_diff(prompts[i - 1], prompts[i], n=0)]

    actions = example_episode["actions"]

    completion = ""
    for j in range(seq):
        completion += "\n%s%s" % (
            action_tok,
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


def process_full_episodes(args):
    # read file
    with tqdm(
        total=N_EPISODES, position=1, desc="Loading source data", leave=False
    ) as pbar:
        with jsonlines.open(args.source_fn, "r") as reader:
            out = []
            for obj in reader:
                out += [obj]
                pbar.update(1)

    # configure target filenames
    source_config = args.source_fn[
        args.source_fn.find("BabyAI") : args.source_fn.rfind("/")
    ]

    os.makedirs(args.data_dir, exist_ok=True)

    diff_fn = os.path.join(args.data_dir, source_config + "-diff.jsonl")
    fulltext_fn = os.path.join(args.data_dir, source_config + "-fulltext.jsonl")
    print(f"Diff data will be saved to {diff_fn}")
    print(f"Full-text data will be saved to {fulltext_fn}")

    # process diffs
    histories = []
    with tqdm(
        total=N_EPISODES, position=1, desc="Computing diff histories", leave=False
    ) as pbar:
        for episode in out:
            hist = get_diff_interaction_history(episode)
            histories += [hist]
            pbar.update(1)

    print("Saving diff data")
    # save diffs
    with jsonlines.open(diff_fn, "w") as writer:
        writer.write_all(histories)

    # process fulltexts
    histories = []
    with tqdm(
        total=N_EPISODES, position=1, desc="Computing fulltext histories", leave=False
    ) as pbar:
        for episode in out:
            hist = get_fulltext_interaction_history(episode)
            histories += [hist]
            pbar.update(1)

    print("\nSaving fulltext data\n")
    # save diffs
    with jsonlines.open(fulltext_fn, "w") as writer:
        writer.write_all(histories)


def process_episode_chunks(args):
    all_idxs = []
    j = 0
    # read file
    with tqdm(
        total=N_EPISODES, position=1, desc="Loading source data", leave=True
    ) as pbar:
        with jsonlines.open(args.source_fn, "r") as reader:
            out = []
            for obj in reader:
                out += [obj]

                seq = max(1, len(obj["actions"]))
                idxs = np.stack([np.zeros((seq,)) + j, np.arange(0, seq, 1)], axis=0)
                all_idxs += [idxs]

                j += 1

                pbar.update(1)

    all_idxs = np.hstack(all_idxs)
    print("\n", all_idxs.shape, "\n")

    i = np.random.choice(
        np.arange(all_idxs.shape[1]), size=(args.n_samples,), replace=False
    )
    selected_idxs = all_idxs[:, i].astype(int)

    # unpack sampled episode chunks
    episode_chunks = []
    for j in range(args.n_samples):
        i, ts_start = selected_idxs[:, j]
        episode = out[i]
        chunk = {}
        for k, v in episode.items():
            if not isinstance(v, list):
                chunk[k] = v
            else:
                chunk[k] = v[ts_start : min(len(v), ts_start + args.chunk_length)]
        episode_chunks += [chunk]

    # configure target filenames
    source_config = args.source_fn[
        args.source_fn.find("BabyAI") : args.source_fn.rfind("/")
    ]
    diff_fn = os.path.join(
        args.data_dir,
        source_config + f"-diff-{args.n_samples}-k{args.chunk_length}.jsonl",
    )
    fulltext_fn = os.path.join(
        args.data_dir,
        source_config + f"-fulltext-{args.n_samples}-k{args.chunk_length}.jsonl",
    )

    os.makedirs(args.data_dir, exist_ok=True)

    print(f"\nDiff data will be saved to {diff_fn}\n")
    print(f"\nFull-text data will be saved to {fulltext_fn}\n")

    # process diffs
    histories = []
    with tqdm(
        total=args.n_samples, position=1, desc="Computing diff histories", leave=True
    ) as pbar:
        for episode in episode_chunks:
            hist = get_diff_interaction_history(episode)
            histories += [hist]
            pbar.update(1)

    print("\nSaving diff data\n")
    # save diffs
    with jsonlines.open(diff_fn, "w") as writer:
        writer.write_all(histories)

    # process fulltexts
    histories = []
    with tqdm(
        total=args.n_samples,
        position=1,
        desc="Computing fulltext histories",
        leave=True,
    ) as pbar:
        for episode in episode_chunks:
            hist = get_fulltext_interaction_history(episode)
            histories += [hist]
            pbar.update(1)

    print("\nSaving fulltext data\n")
    # save diffs
    with jsonlines.open(fulltext_fn, "w") as writer:
        writer.write_all(histories)


def main(args):
    process_episode_chunks(args)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_fn",
        default=(
            f"data/test-BabyAI-MixedTrainLocal-v0-0-%i-defaultbot/data.jsonl"
            % N_EPISODES
        ),
        type=str,
        help="dir where to store data",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            PROJECT_PATH, "babyaitext_experiments", "data", "processed"
        ),
        type=str,
    )
    parser.add_argument("--n_samples", default=1000, type=int)
    parser.add_argument("--chunk_length", default=4)
    args = parser.parse_args()

    print("ARGS:", args)
    return args


if __name__ == "__main__":
    args = parse_args()

    set_seed_everywhere(0)
    main(args)
