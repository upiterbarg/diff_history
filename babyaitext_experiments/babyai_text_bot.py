import numpy as np

from babyai.bot import Bot
from babyai.rl.utils import DictList
from collections import deque
from tqdm import tqdm


import os
import pathlib
import sys

base_path = str(pathlib.Path().resolve())
PROJECT_PATH = os.path.join(
    base_path[: base_path.find("diff_history")],
    "diff_history",
)
sys.path.insert(0, os.path.join(PROJECT_PATH, "external/Grounding_LLMs_with_online_RL"))
from experiments.agents.base_agent import BaseAgent


class BotAgent(BaseAgent):
    def __init__(self, env, subgoals):
        """An agent based on BabyAI's GOFAI bot."""
        self.env = env
        self.subgoals = subgoals
        self.obs, self.infos = self.env.reset()
        self.bot = Bot(self.env)

        self.obs_queue = deque([], maxlen=3)
        self.acts_queue = deque([], maxlen=2)

        self.obs_queue.append(self.infos["descriptions"])

        self.prompts = []
        self.actions = []
        self.all_obs = []
        self.all_infos = []

        self.log_done_counter = 0

    def act(self, action_choosen=None):
        actions = self.bot.replan(action_choosen)
        return actions

    def reset(self):
        self.obs_queue = deque([], maxlen=3)
        self.acts_queue = deque([], maxlen=2)

        self.log_done_counter = 0
        self.obs, self.infos = self.env.reset()
        self.bot = Bot(self.env)

        self.obs_queue.append(self.infos["descriptions"])

        self.prompts = []
        self.actions = []
        self.all_obs = []
        self.all_infos = []

    def set_env(self, env):
        self.env = env

    def generate_trajectories(self, n_episodes=1, dict_modifier={}, language="english"):
        assert language == "english"

        nbr_frames = 1
        previous_action = None

        while self.log_done_counter < n_episodes:
            nbr_frames += 1
            prompt = self.prompt_modifier(
                self.generate_prompt(
                    goal=self.obs["mission"],
                    subgoals=self.subgoals,
                    deque_obs=self.obs_queue,
                    deque_actions=self.acts_queue,
                ),
                dict_modifier,
            )

            action = self.act(previous_action)
            # previous_action = action
            self.actions.append(self.subgoals[int(action)])
            self.acts_queue.append(self.subgoals[int(action)])
            self.prompts.append(prompt)
            self.all_obs.append({k: v for k, v in self.obs.items()})
            self.all_infos.append({k: v for k, v in self.infos.items()})

            self.obs, reward, done, self.infos = self.env.step(action)

            if done:
                self.all_obs.append({k: v for k, v in self.obs.items()})
                self.all_infos.append({k: v for k, v in self.infos.items()})

                self.log_done_counter += 1
                # pbar.update(1)
                self.obs_queue.clear()
                self.acts_queue.clear()
                self.obs, infos = self.env.reset()
                self.bot = Bot(self.env)

            self.obs_queue.append(self.infos["descriptions"])

        self.log_done_counter = 0
        return self.all_obs, self.prompts, self.actions, self.all_infos, reward

    def update_parameters(self):
        pass
