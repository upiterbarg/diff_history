# `diff` History for Neural Language Agents

This is the official code release accompanying the paper **[*`diff` History for Neural Language Agents***](https://upiterbarg.github.io/projects) by **Piterbarg**, **Pinto**, and **Fergus**.

**Tldr**:  `diff` history is a method for improving the quality of LM generations through instruction tuning for decision-making settings. We show that small LMs can be data efficiently tuned into highly competitive agents just by: (1) treating the LM as a policy; (2) extending model context lengths; (3) increasing the length of the history used to train/tune and prompt models; and (4) preprocessing observations in history with the Unix `diff` command.

[:paperclips: **Project Page** :paperclips:](https://diffhistory.github.io)
[:bulb:  **Abstract** :bulb:](https://arxiv.org/abs/2312.07540 ) 
[ :memo: **Paper PDF** :pencil:](https://arxiv.org/pdf/2312.07540.pdf )




## Installation

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
