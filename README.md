
# `diff` History for Neural Language Agents

This is the official code release accompanying the paper [***`diff` History for Neural Language Agents***](https://upiterbarg.github.io/projects) by **Piterbarg**, **Pinto**, and **Fergus** (arXiv preprint, 2024).

--- 
**Tldr**:  `diff` history is a method for improving the quality of LM generations for decision-making settings through low-resource instruction tuning. We show that small LMs can be data efficiently tuned into highly competitive neural agents just by: (1) treating the LM as a policy; (2) extending model context lengths; (3) increasing the length of the history used to train/tune and prompt models; and (4) preprocessing observations in history with the Unix `diff` command.

[:paperclips: **Project Page** :paperclips:](https://diffhistory.github.io) | 
[:bulb:  **Abstract** :bulb:](https://arxiv.org/abs/2312.07540 ) | 
[:memo: **Paper PDF** :pencil:](https://arxiv.org/pdf/2312.07540.pdf ) | 
[:inbox_tray: **Dataset Coming Soon!** :inbox_tray:]()


---

## Installation :electric_plug:

Start by cloning our repo recursively.
```
git clone --recursive git@github.com:upiterbarg/diff_history.git
```

### Install core dependencies
```
conda env create --file=conda_env.yaml
conda activate test
```

### Install external dependencies

#### BabyAI-Text
```
cd external/Grounding_LLMs_with_online_RL/babyai-text/babyai; pip install -e .; cd ..
cd gym-minigrid; pip install -e.; cd ..
pip install -e .; cd ../../..
```

#### NLE (**with seed changes**)
```
sed -i '344,349d' external/nle/nle/env/tasks.py
sed -i '365,366d' external/nle/nle/env/tasks.py
cd external/nle; python setup.py build; python setup.py install; cd ../..
```

#### NetHack language wrapper
```
cd external/nle-language-wrapper; python setup.py build; python setup.py install; python -m setup develop; cd ../..
```

#### Vision NLE utilities
```
pip install git+ssh://git@github.com/facebookresearch/moolib
cd external/dungeonsdata-neurips2022/experiment_code
pip install -e . && cd ../../..
```

---

## Navigating the Repo :world_map:

```
--> conda_env.yaml         # Conda config.
--> finetune.py            # Finetuning script. Copies https://github.com/allenai/open-instruct/open_instruct/finetune.py, with token additions + masking.
--> action_textmap.py      # Interaction history tokens.
--> gpt2_resize.py         # Resize GPT-2 context length
--> utils.py               # Various utilities: configuring custom stop generation, computing diffs, setting seeds everywhere.
```
```
--> scripts                # Bash scripts
----- / launch.sh                    # Sample instruction tuning launch script
```
```
--> ds_configs             # Distributed training configs
----- / stage3_offloading_accelerate.conf    #  ZeRO Stage 3
```
```
--> nethack_experiments    # NetHack experiment code
----- / diff_history_rollout.py       # Test LMs with diff history in NetHack.
----- / fulltext_history_rollout.py   # Test LMs with full text history in NetHack.
----- / generate_aa_dataset.py        # Generate a dataset with full games using AutoAscend.
----- / format_interaction_histories.py      # Format interaction histories
----- / wrappers.py                   # Define pixel and LM interaction history wrappers for NetHack.
```
```
--> babyaitext_experiments
----- / diff_history_rollout.py       # Test LMs with diff history in BabyAI-Text
----- / fulltext_history_rollout.py   # Test LMs with full text history in BabyAI-Text
----- / generate_babyai_dataset.py    # Generate BabyAI-Text dataset with full games from the BabyAI bot
----- / format_interaction_histories.py      # Format interaction histories
----- / babyai_text_bot.py            # BabyAI Text bot.
----- / lm_wrappers.py                # Wrapper over BabyAI-Text for formatting interaction histories.
```
```
--> external               # Various external dependencies
```
