# interpreting-prompt-injection
This is my research work on interpreting what makes a model follow another an injected task

It builds on this [repository](https://github.com/paulphilip-louis/attention-tracker) which is a reimplementation of the [Attention Tracker](https://arxiv.org/abs/2411.00348) paper (Hung et al., 2024) as well as an extension and some extra experiments.

The core idea of this research is that the [trigger](https://arxiv.org/abs/2403.03792) (a string surrounding the actual injected task, that aims at incentivizing the model to obey it, e.g "Ignore previous instructions") is a key element to understand *what* makes a model fall for prompt injection.

The reason behind is that the attack success rate of an injection can go from 0% to almost 100% simply by changing the trigger.

In this work, I adopted a mechanistic interpretability-grounded approach to uncover possible internal mechanisms in an LLM that could explain what makes a model follow an injected instruction.

## 1. Experiments and results

## 2. Structure of the repository

```
interpreting-prompt-injection/
│
├── src/
│   ├── data/            # dataloaders, préprocessing
│   └── utils/           # utilitary functions
│
├── experiments/
│   ├── distraction_effect.md
│   ├── activation_patching.md
│   └── steering.md
│
├── notebooks/
│   └── distraction_effect.ipynb       # measuring distraction effect
│   └── activation_patching.ipynb      # activation patching experiments
│   └── steering.ipynb                 # steering vector study
│
├── results/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## 3. Installation
First make sure you have installed ```uv```.

```uv sync```.

## 4. Utilisation

First ensure your environment is activated : 
```source .venv/bin/activate```

