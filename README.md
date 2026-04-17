<h1 align="center"><b>AgentV-RL: Scaling Reward Modeling with Agentic Verifier</b></h1>

<p align="center">
  <a href="#">
    <img alt="ACL 2026" src="https://img.shields.io/badge/ACL-2026-EE4C2C?style=flat&labelColor=1F1F1F" height="20">
  </a>
  &nbsp;
  <a href="#citation">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Read-blue?logo=readthedocs&logoColor=white" height="20">
  &nbsp;
  <a href="https://github.com/volcengine/verl">
    <img alt="Built on Verl" src="https://img.shields.io/badge/Built%20on-verl-ff6f00?logo=pytorch&logoColor=white" height="20">
  </a>
  &nbsp;
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg" height="20">
  </a>
  &nbsp;
  <a href="#">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" height="20">
  </a>
</p>


<!-- <p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-main-results">Results</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-data-format">Data Format</a> •
  <a href="#-citation">Citation</a>
</p> -->

<p align="center">
  <img src="figures/framework-full.png" alt="AgentV-RL framework" width="100%">
</p>

**AgentV-RL** is an open-source recipe for scaling reward modeling with an **agentic verifier**. We turn verification into a multi-turn process with explicit *planning*, *stepwise validation*, *final verdict aggregation*, and *tool use*, and apply it to both **Best-of-N reranking** and **iterative refinement**.

> This work is done by collaborators from **Fudan University**, **Huazhong University of Science and Technology**, **The University of Hong Kong**, and **ByteDance Seed**. Our training and evaluation codebase is built on [Verl](https://github.com/volcengine/verl). This repository includes the core inference, refinement, SFT, and GRPO training code for the AgentV-RL pipeline.

---

## 🔎 Overview

AgentV-RL reframes reward modeling as an **agentic verification process** instead of a single-pass scoring call:

- 🧭 **Planning** — decompose the candidate solution into checkable sub-claims.
- 🔬 **Stepwise validation** — validate each sub-claim with targeted reasoning and tool use.
- 🧮 **Final verdict** — aggregate per-step judgments into a calibrated correctness signal.
- 🛠️ **Tool use** — invoke symbolic/numeric tools when the textual trace is insufficient.

The same verifier is then used at test time for **parallel TTS** (Best-of-N reranking) and **sequential TTS** (iterative refinement).

---

## 📊 Main Results

We evaluate AgentV-RL under two test-time scaling settings: **parallel TTS** with Best-of-N selection, and **sequential TTS** with iterative refinement.

### 🧩 Parallel TTS: Best-of-N

Accuracy of the AgentV-RL 4B verifier under different BoN budgets:

| Benchmark | BoN@32 | BoN@64 | BoN@128 |
| --- | ---: | ---: | ---: |
| MATH500 | 73.8 | 76.2 | **79.0** |
| GSM8K | 93.0 | 92.6 | **93.3** |
| Gaokao2023 | 54.5 | 55.1 | **57.4** |
| AIME24 | 46.7 | 50.0 | **53.3** |

> 🚀 On MATH500, the paper reports up to **+25.2** absolute points improvement over prior outcome-level reward model baselines.

### 🔁 Sequential TTS: Iterative Refinement

Accuracy of the AgentV-RL 4B verifier when used as the critique module in multi-round refinement:

| Benchmark | Turn 1 | Turn 2 | Turn 3 |
| --- | ---: | ---: | ---: |
| MATH500 | 84.2 | 89.2 | **89.8** |
| GSM8K | 94.6 | 94.1 | 94.1 |
| Gaokao2023 | 75.6 | **76.6** | 76.4 |
| AIME24 | 40.0 | 33.3 | 33.0 |

> 💡 Most of the gain is obtained in the first one or two refinement rounds, with later rounds mainly stabilizing performance on the easier benchmarks.

---

## 📁 Repository Structure

```text
AgentV-RL/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yml
│   └── score_vanilla.yml
├── examples/
│   ├── run_verify.sh
│   ├── run_verify_entry.sh
│   ├── run_refine.sh
│   ├── run_refine_entry.sh
│   ├── score_vanilla_infer.sh
│   ├── train_sft_multiturn.sh
│   └── train_grpo.sh
└── src/
    ├── run_verify_multihead.py
    ├── score_vanilla_infer.py
    ├── refine/
    ├── agentflow/
    └── verl/
```

**Important entrypoints:**

| Path | Purpose |
| --- | --- |
| `src/run_verify_multihead.py` | Agentic Best-of-N verification |
| `src/refine/main_refine.py` | Iterative refinement |
| `src/score_vanilla_infer.py` | Vanilla single-pass verifier baseline |
| `examples/train_sft_multiturn.sh` | Multiturn SFT |
| `examples/train_grpo.sh` | GRPO training |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

---

## 🚀 Getting Started

### 🧪 Best-of-N Verification

```bash
bash examples/run_verify_entry.sh \
  --task-name math500 \
  --exp-name qwen3_4b_agentic \
  --config config/default.yml \
  --model-path /path/to/verifier-model \
  --input /path/to/bon_input.jsonl \
  --output-dir /path/to/output \
  --log-dir /path/to/logs \
  --num-workers 4 \
  --enable-thinking
```

### 📏 Vanilla Verifier Baseline

```bash
python src/score_vanilla_infer.py \
  --config config/score_vanilla.yml \
  --input /path/to/bon_input.jsonl \
  --output /path/to/bon_result.jsonl \
  --record-batch-size 1 \
  --append
```

### 🔁 Iterative Refinement

```bash
bash examples/run_refine_entry.sh \
  --candidate-config config/default.yml \
  --verifier-config config/default.yml \
  --input /path/to/refine_input.jsonl \
  --output /path/to/final.jsonl \
  --round-output-dir /path/to/round_outputs \
  --metrics-output-dir /path/to/metrics \
  --exp-name refine_qwen3 \
  --verifier-type forward \
  --candidate-model-path /path/to/candidate-model \
  --verifier-model-path /path/to/verifier-model \
  --num-candidate-workers 2 \
  --num-verifier-workers 2 \
  --batch-size 16 \
  --max-refine-rounds 3 \
  --thinking-candidate \
  --thinking-verifier
```

### 🏋️ Training

#### Multiturn SFT

```bash
bash examples/train_sft_multiturn.sh
```

#### GRPO

```bash
bash examples/train_grpo.sh
```

---

## 📦 Data Format

### 🧪 Best-of-N Verification Input

Each JSONL record contains one problem and a fixed candidate pool.

```json
{
  "idx": 0,
  "input": "<prompt text for the candidate model>",
  "question": "<raw question>",
  "answer": "optional reference solution",
  "ground_truth": "(3,\\frac{\\pi}{2})",
  "samples": [
    "candidate answer 0",
    "candidate answer 1"
  ],
  "evaluations": [
    {
      "correct": true,
      "parsed_gt": "(3,\\frac{\\pi}{2})",
      "parsed_pred": "(3,\\frac{\\pi}{2})",
      "mathd_equal": true,
      "sympy_equal": false,
      "sampling_id": 0
    }
  ]
}
```

> ℹ️ `samples` and `evaluations` must be aligned by index — each evaluation corresponds to exactly one candidate answer.

### 🔁 Refinement Input

Each JSONL record contains one problem and, optionally, an initial answer.

```json
{
  "idx": 0,
  "input": "<prompt text for the candidate model>",
  "question": "<raw question>",
  "answer": "optional reference solution",
  "ground_truth": "33",
  "refine_rounds": [
    {
      "answer": "<candidate reasoning trace>",
      "cand_correct": false,
      "cand_grade": {
        "parsed_gt": "33",
        "parsed_pred": "13",
        "correct": false,
        "mathd_equal": false,
        "sympy_equal": false
      }
    }
  ]
}
```

> ℹ️ If no initial answer is provided, the candidate model generates it in round 0.

### 🏋️ RL Training Input

The GRPO stage expects boolean verifier supervision:

```json
{
  "idx": 708,
  "data_source": "rm_bool",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reward_model": {
    "ground_truth": true,
    "style": "rule"
  },
  "extra_info": {
    "problem": "<original question>",
    "solution": "<candidate solution>"
  }
}
```

> ℹ️ `data_source` must be `rm_bool`. The effective training input is built from `extra_info.problem` and `extra_info.solution`.

---

## 📝 Citation

If you find AgentV-RL useful for your research, please consider citing our paper:

```bibtex
@misc{zhang2025agentvrl,
  title={AgentV-RL: Scaling Reward Modeling with Agentic Verifier},
  author={Jiazheng Zhang and Ziche Fu and Zhiheng Xi and Wenqing Jing and Mingxu Chai and Wei He and Guoqiang Zhang and Chenghao Fan and Chenxin An and Wenxiang Chen and Zhicheng Liu and Haojie Pan and Dingwei Zhu and Tao Gui and Qi Zhang and Xuanjing Huang},
  year={2025},
  note={Paper URL to be added}
}
```

---

## 🙏 Acknowledgements

This repository is built on top of [**Verl**](https://github.com/volcengine/verl). We gratefully acknowledge the collaboration and compute support from [**ByteDance Seed**](https://seed.bytedance.com/), and thank the open-source community for datasets such as **MATH500**, **GSM8K**, **Gaokao2023**, and **AIME24** that make this research possible.

---

## 📄 License

This project is released under the license in [`LICENSE`](LICENSE).
