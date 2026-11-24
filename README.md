# Can LLMs Reason? Predictive and Diagnostic Causal Reasoning in Large Language Models

Code and preliminary pilot outputs for the UNSW Bachelor of Software Engineering (Honours) Thesis A report.

## Overview

This repository contains the Thesis A pilot pipeline evaluating `Qwen/Qwen3-0.6B` (non-thinking mode) on all 500 labelled items of SuperGLUE COPA (`train` = 400, `validation` = 100), testing each item in both option orders (`swap=False` and `swap=True`, 1,000 generations total).

## Files

- `run_pilot.py` — Runs `Qwen/Qwen3-0.6B` via HuggingFace `transformers` on SuperGLUE COPA in both option orders and writes raw outputs and parsed predictions to JSONL.
- `analyze.py` — Computes directional accuracy, order-invariant accuracy, order consistency, and a 10,000-resample bootstrap 95% confidence interval for the `cause − effect` gap.
- `copa_text_Qwen3-0.6B_nothink.jsonl` — Full pilot output log (1,000 rows).

## Environment

Verified locally with:

- Python 3.12.10
- PyTorch `2.14.1+cu126`
- `transformers 5.24.0`
- `datasets 5.2.0`
- GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```
