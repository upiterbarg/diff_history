from gym import Wrapper

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]


class BabyAITextLMWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._aspace = BABYAI_ACTION_SPACE[:]
        self._aspace_str = ", ".join(self._aspace[:-1]) + ", and " + self._aspace[-1]
        self._interleaving_token = "<|action|>"

    @property
    def interleaving_token(self):
        return self._interleaving_token

    def get_instruction_and_prompt(self, obs, infos):
        def _form_prompt(description):
            return "\n".join([d.replace("You see ", "") for d in description])

        mission = obs["mission"]
        instruction = "\n".join(
            [
                f"Your task is to {mission}.",
                f"You can take {len(self._aspace)} different actions: {self._aspace_str}.",
                f"Predict the next actions.",
            ]
        )

        prompt = _form_prompt(infos["descriptions"])
        return instruction, prompt

    def reset(self):
        obs, infos = self.env.reset()
        instruction, prompt = self.get_instruction_and_prompt(obs, infos)
        obs["instruction"] = instruction
        obs["prompt"] = prompt
        return obs, infos

    def step(self, action):
        action_int = self._aspace.index(action)
        obs, reward, done, infos = self.env.step(action_int)
        instruction, prompt = self.get_instruction_and_prompt(obs, infos)
        obs["instruction"] = instruction
        obs["prompt"] = prompt
        return obs, reward, done, infos
