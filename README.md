# interpreting-prompt-injection
This is my research work on interpreting what makes a model follow another an injected task

This repo builds on [attention-tracker](https://github.com/paulphilip-louis/attention-tracker) (re-implementation of Hung et al.) and extends it with mechanistic interpretability methods

The core idea of this research is that the [trigger](https://arxiv.org/abs/2403.03792) (a string surrounding the actual injected task, that aims at incentivizing the model to obey it, e.g "Ignore previous instructions") is a key element to understand *what* makes a model fall for prompt injection.

The reason behind is that the attack success rate of an injection can go from 0% to almost 100% simply by changing the trigger.

In this work, I adopted a mechanistic interpretability-grounded approach to uncover possible internal mechanisms in an LLM that could explain what makes a model follow an injected instruction.

## 1. Experiments and results

First, I defined the following hypothesis:
1. Appending an instruction-like string at the end of a prompt decreases significantly the focus score of that prompt
2. Adding a trigger further decreases the focus score, with a stronger decrease being associated with a more complex trigger
3. The effect of the trigger on the model's behavior can be traced back to a subset of heads in the model.

```notebooks/distraction_effect.ipynb``` verifies the 1. and 2. hypotheses.

```activation_patching.ipynb``` starts exploring a potential circuit involved in the success of the injection. Results point consistently across tasks towards a given set of heads. The mechanistic roles of the specific heads, as well as the existence of several subcircuits has not been investigated yet.

```steering_vectors.ipynb``` investigates whether steering vectors could push a model towards/against following an injection, building on the strong difference in answers between absence of trigger and complex trigger. Early results point towards a shared direction that plays some role in the following of an injection.


## 2. Structure of the repository

```
interpreting-prompt-injection/
│
├── src/
│   ├── data/            # dataloaders, préprocessing
│   └── utils/           # utilitary functions
│
├── notebooks/
│   └── distraction_effect.ipynb       # measuring distraction effect
│   └── activation_patching.ipynb      # activation patching experiments
│   └── steering_vectors.ipynb         # steering vector study
│
├── results/                           # results of some experiments
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

At this stage, you can just run the notebooks and see the results for yourself.

Once I will have developed a method to mechanistically limit prompt injection, I will add a script to run.

