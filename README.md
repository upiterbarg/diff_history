# diff history for neural language agents

## install core dependencies
```
conda env create --file=conda_env.yaml
conda activate test
```

## install external dependencies

### BabyAI-Text
```
cd external/Grounding_LLMs_with_online_RL/babyai-text/babyai; pip install -e .; cd ..
cd gym-minigrid; pip install -e.; cd ..
pip install -e .; cd ../../..
```

### NLE (**with seed changes**)
```
sed -i '344,349d' external/nle/nle/env/tasks.py
sed -i '365,366d' external/nle/nle/env/tasks.py
cd external/nle; python setup.py build; python setup.py install; cd ../..
```

### NetHack language wrapper
```
cd external/nle-language-wrapper; python setup.py build; python setup.py install; python -m setup develop; cd ../..
```

### vision NLE utilities
```
pip install git+ssh://git@github.com/facebookresearch/moolib
cd external/dungeonsdata-neurips2022/experiment_code
pip install -e . && cd ../../..
```
