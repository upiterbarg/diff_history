import babyai_text
import gym
import jsonlines
import numpy as np
import os
import pathlib
import pdb
import sys

from argparse import ArgumentParser
from babyai_text_bot import BotAgent
from collections import deque
from tqdm import tqdm

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(base_path[: base_path.find("diff_history")], "diff_history")
sys.path.insert(0, PROJECT_PATH)
from utils import set_seed_everywhere

ACTION_SPACE = ["turn_left", "turn_right", "go_forward", "pick_up", "drop", "toggle"]


def main(args):
    if not args.bot_type == "default":
        raise NotImplementedError

    # set saving paths
    gen_dir = f"test-{args.env_name}-{args.seed}-{args.episodes}-{args.bot_type}bot"
    gen_dir = os.path.join(args.base_dir, gen_dir)
    os.makedirs(gen_dir, exist_ok=True)
    fn = os.path.join(gen_dir, "data.jsonl")

    set_seed_everywhere(args.seed)

    # config actions
    list_actions = [a.replace("_", " ") for a in ACTION_SPACE]

    # config envs
    env = gym.make(args.env_name, num_dists=0)
    seed = int(args.seed)
    env.seed(seed)

    # init bot
    algo = BotAgent(env=env, subgoals=list_actions)

    # gen data
    episodes = []
    print("generating")
    with tqdm(total=args.episodes, position=1) as pbar:
        for i in range(args.episodes):
            all_obs, prompts, actions, all_infos, reward = algo.generate_trajectories()
            imgs = [obs["image"].tolist() for obs in all_obs]
            directions = [int(obs["direction"]) for obs in all_obs]
            mission = all_obs[0]["mission"]
            descriptions = [info["descriptions"] for info in all_infos]

            episode_log = {
                "imgs": imgs,
                "directions": directions,
                "descriptions": descriptions,
                "default_prompts": prompts,
                "actions": actions,
                "reward": reward,
                "seed": seed,
                "mission": mission,
            }

            seed += 1

            env = gym.make(args.env_name, num_dists=0)
            env.seed(seed)

            algo.set_env(env)
            algo.reset()

            episodes += [episode_log]
            pbar.update(1)

    print(f"writing data to file @: {fn}")
    with jsonlines.open(fn, "w") as writer:
        writer.write_all(episodes)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir", default="data", type=str, help="dir where to store data"
    )
    parser.add_argument("--seed", default=0, type=int, help="Starting random seed")
    parser.add_argument("-n", "--episodes", type=int, default=5000)
    parser.add_argument("--bot_type", default="default", type=str)
    parser.add_argument("--env_name", default="BabyAI-MixedTrainLocal-v0")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, sys.maxsize)

    print("ARGS:", args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
