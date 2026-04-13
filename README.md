# Interpreting Prompt Injection

Research into understanding what makes language models follow injected instructions — probing the internal mechanisms behind prompt injection susceptibility.

## Overview

This project investigates prompt injection from an interpretability angle: rather than simply demonstrating that injection works, we aim to understand *why* models follow injected tasks, what internal representations are involved, and what factors modulate susceptibility.

## Repository Structure

```
.
├── notebooks/       # Exploratory analysis and experiment notebooks
├── src/             # Reusable modules and utilities
├── experiments/     # Experiment scripts and configuration files
├── data/            # Datasets (not tracked by git)
└── results/         # Outputs, figures, and metrics (not tracked by git)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Research Questions

- What internal model features distinguish instruction-following from injection-following?
- How does context framing influence a model's decision to follow injected content?
- Are there universal or model-specific mechanisms behind prompt injection?
