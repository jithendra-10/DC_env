# DataClean-Env

**An OpenEnv-compliant reinforcement learning environment for data cleaning agents.**

DataClean-Env challenges LLM agents to clean realistic tabular datasets through a sequence of structured operations — null imputation, dtype correction, outlier clipping, deduplication, and more. Every episode is reproducible via a seeded generator. Every grader is a deterministic pandas assertion.

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.dev)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange)](https://huggingface.co/spaces/jithendra/dataclean-env)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](./Dockerfile)

---

## Overview & Motivation

Data cleaning consumes 60–80% of a data scientist's working time. It is a universal, expensive, and well-understood real-world problem — but no existing OpenEnv environment tests it.

DataClean-Env fills that gap. It provides:
- **3 tasks** of escalating difficulty (easy → medium → hard)
- **7 action types** covering the full cleaning workflow
- **Rich partial rewards** on every operation — agents get signal every step, not just at the end
- **Provenance tracking** — an immutable ops log that must replay cleanly on the raw data
- **Confidence calibration** — agents are rewarded for knowing what they know and penalised for overconfidence on wrong actions
- **Gradio web UI** at `/web` so judges can manually step through a cleaning episode in their browser

---

## Section 1 — Action & Observation Spaces

### Actions

Every action is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `action_type` | string | One of 7 operations (see table below) |
| `column` | string \| null | Target column. Required for all ops except `remove_duplicates` and `done` |
| `params` | object | Operation-specific parameters |
| `confidence` | float 0–1 | Agent's self-reported confidence. Calibration is rewarded. |

**Action types:**

| Action | Params | Effect |
|---|---|---|
| `fill_nulls` | `strategy`: mean\|median\|mode\|constant\|ffill, `value` (for constant) | Impute missing values |
| `remove_duplicates` | `subset`: list of columns (optional) | Drop duplicate rows |
| `fix_dtype` | `target_dtype`: int64\|float64\|str\|datetime64\|bool | Cast column to correct type |
| `clip_outliers` | `method`: iqr\|zscore\|percentile, `threshold` | Clip extreme values |
| `rename_column` | `new_name`: string | Rename column to canonical name |
| `drop_column` | — | Drop an irrelevant column |
| `done` | — | Signal that cleaning is complete |

### Observation

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Active task |
| `episode_id` | string | Unique episode UUID |
| `step` | int | Current step number |
| `budget_remaining` | int | Steps left before forced termination |
| `n_rows` | int | Current row count |
| `n_cols` | int | Current column count |
| `duplicate_rate` | float | Fraction of duplicate rows |
| `columns` | list[ColumnProfile] | Per-column stats (see below) |
| `ops_log` | list | All operations applied so far |
| `quality_scores` | dict | Per-dimension quality preview: null_score, type_score, outlier_score, dup_score, overall |
| `last_action_result` | string | Human-readable feedback on the previous action |

**ColumnProfile fields:** `name`, `dtype`, `null_rate`, `n_unique`, `mean`, `std`, `min`, `max`, `sample_values`, `corruption_flags`

**Corruption flags:** `heavy_nulls` (>20% missing), `has_nulls` (>1%), `heavy_outliers` (>5% IQR outliers), `type_chaos` (numeric data stored as strings)

---

## Section 2 — Tasks & Difficulty

### Task 1 — Employee Dataset (Easy)

300-row employee records. Seeded, reproducible.

| Issue | Detail |
|---|---|
| Null ages | 22% of the `age` column is missing |
| Salary outliers | 12 rows with physiologically impossible salaries (−99k, 500k, 999k) |
| Type chaos | `years_at_company` stored as string ("N/A" mixed with numbers) |
| Duplicates | 30 duplicate rows |

**Grader checks:** `null_rate(age) ≤ 0.01`, `dtype(years_at_company)` is numeric, `salary` IQR-clean, `duplicate_rate < 0.005`

**Max steps:** 15 | **Expected baseline (GPT-4o-mini):** ≥ 0.90

---

### Task 2 — E-Commerce Orders (Medium)

500-row order records with an irrelevant column to discover and drop.

| Issue | Detail |
|---|---|
| Heavy null quantities | 28% of `qty` column missing |
| Null ratings | 15% of `customer_rating` missing |
| Amount outliers | 25 rows with impossible order amounts (−500, 50k, 999k) |
| Irrelevant column | `internal_hash` — should be dropped |
| Duplicates | 40 duplicate rows |

**Grader checks:** all null rates ≤ 1%, numeric dtypes, amount IQR-clean, no duplicates, `internal_hash` absent

**Max steps:** 18 | **Expected baseline (GPT-4o-mini):** ≥ 0.88

---

### Task 3 — Healthcare Patient Records (Hard)

800-row patient dataset with **mixed corruption profiles per column** — the agent cannot apply one fix globally. Each column needs individual diagnosis.

| Column | Corruption profile |
|---|---|
| `patient_age` | heavy_nulls (25%) |
| `weight_kg` | heavy_outliers (physiologically impossible: −10, 5, 500, 999 kg) |
| `glucose_mgdl` | type_chaos — mix of floats and strings ("N/A", "pending", "HIGH", "???") |
| `cholesterol` | nulls (12%) + moderate outliers |
| `systolic_bp` | nulls (8%) |
| `admin_notes` | irrelevant — should be dropped |
| All | 60 duplicate rows |

This task is designed to challenge GPT-4-class models. An agent that applies one strategy globally will score poorly on the multi-profile columns.

**Grader checks:** null_score (weighted), dtype of glucose numeric, outlier_score on weight + cholesterol, dup_score, admin_notes absent, glucose numeric fraction

**Max steps:** 20 | **Expected baseline (GPT-4o-mini):** ≥ 0.82

---

## Section 3 — Reward Function

Every step: **−0.01** (efficiency pressure — don't waste budget on clean columns)

| Action | Reward |
|---|---|
| `fill_nulls` correct column | +0.10 × (1 − remaining_null_rate) |
| `remove_duplicates` with real dupes | +0.12 |
| `fix_dtype` mistyped column | +0.10 |
| `clip_outliers` column with outliers | +0.08 × (1 + std_reduction) |
| `rename_column` to canonical name | +0.05 |
| `drop_column` irrelevant column | +0.04 |
| Any action on already-clean column | −0.05 (wrong column penalty) |
| Malformed params / unsupported op | −0.03 |
| `done` (quality ≥ 0.80) | +0.15 |
| `done` (quality < 0.80) | +0.15 × quality |
| **Provenance bonus** | **+0.05** if ops log fully replays on raw data |
| **Confidence bonus** | **+0.04** if confidence ≥ 0.75 and action was correct |
| **Confidence penalty** | **−0.06** if confidence ≥ 0.75 and action was wrong |

**Step-efficiency mechanic:** The −0.01 step penalty means that an agent which applies operations to already-clean columns loses points. The optimal agent does only necessary work and calls `done` early.

**Provenance mechanic:** At episode end, the environment replays every successful operation from the raw dataframe and checks whether the result is byte-identical to the current dataframe. If reproducible, the agent earns +0.05. This tests systematic planning — trial-and-error agents that accidentally fix things without intent will fail the replay check.

---

## Section 4 — Setup & Usage

### Requirements

```
Python 3.11+
```

### Install locally

```bash
git clone https://huggingface.co/spaces/jithendra/dataclean-env
cd dataclean-env
pip install -r requirements.txt
```

### Run server locally

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
# Gradio UI available at http://localhost:7860/web
```

### Run with Docker

```bash
docker build -t dataclean-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e ENABLE_WEB_INTERFACE=true \
  dataclean-env
```

### Run baseline agent (LLM)

```bash
export OPENAI_API_KEY=sk-...

# Against local server
python baseline/agent.py

# Against HuggingFace Space
python baseline/agent.py --url https://huggingface.co/spaces/jithendra/dataclean-env

# Single task
python baseline/agent.py --task task_3 --seed 42
```

### Run OpenEnv validation

```bash
openenv validate --url https://huggingface.co/spaces/jithendra/dataclean-env
```

### Quick API test

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "<id>", "action": {"action_type": "remove_duplicates", "confidence": 0.9}}'

# Grade
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "<id>"}'
```

---

## Section 5 — Baseline Scores

Reproducible scores from two agents on seed=42.

### Heuristic agent (no LLM — deterministic lower bound)

| Task | Final Score | Total Reward | Steps Used | Provenance |
|---|---|---|---|---|
| task_1 | 0.9167 | 0.4700 | 3 | ✓ |
| task_2 | 0.9500 | 0.7899 | 5 | ✓ |
| task_3 | 0.9500 | 0.9900 | 7 | ✓ |

To reproduce:
```bash
curl http://localhost:7860/baseline?seed=42
```

### GPT-4o-mini agent (LLM baseline)

Scores will be updated after official run. To reproduce:
```bash
export OPENAI_API_KEY=sk-...
python baseline/agent.py --seed 42 --quiet
```

---

## Open-Source Baseline — Meta Llama-3-70B-Instruct

DataClean-Env is **fully model-agnostic**. In addition to the GPT-4o-mini baseline, a second agent powered by Meta's **Llama-3-70B-Instruct** via the HuggingFace Inference API is included.

This directly leverages the infrastructure of the hackathon co-organisers:
- **Meta** built Llama-3-70B-Instruct
- **HuggingFace** hosts it via their Inference API
- **DataClean-Env** runs on HuggingFace Spaces

```bash
export HF_TOKEN=hf_...   # huggingface.co/settings/tokens

# Against local server
python baseline/llama_agent.py

# Against HuggingFace Space
python baseline/llama_agent.py --url https://huggingface.co/spaces/jithendra/dataclean-env

# Single task
python baseline/llama_agent.py --task task_3 --seed 42
```

### Llama-3-70B-Instruct baseline scores (seed=42)

| Task | Final Score | Total Reward | Steps Used | Provenance |
|---|---|---|---|---|
| task_1 | TBD | TBD | TBD | TBD |
| task_2 | TBD | TBD | TBD | TBD |
| task_3 | TBD | TBD | TBD | TBD |

*Run `python baseline/llama_agent.py --quiet` to reproduce.*

### Model comparison

| Agent | Model | Provider | task_1 | task_2 | task_3 |
|---|---|---|---|---|---|
| Heuristic | None (rule-based) | — | 0.9167 | 0.9500 | 0.9500 |
| GPT-4o-mini | gpt-4o-mini | OpenAI | TBD | TBD | TBD |
| **Llama-3-70B** | **Meta-Llama-3-70B-Instruct** | **HuggingFace** | **TBD** | **TBD** | **TBD** |

---

## Nemotron-Compatible Agent Wrapper

Both `NemotronAgentWrapper` (GPT-4o-mini) and `LlamaNemotronWrapper` (Llama-3-70B) expose the exact 3-method interface used by the Phase 2 judge:

```python
from baseline.agent import NemotronAgentWrapper

agent = NemotronAgentWrapper(server_url="https://huggingface.co/spaces/jithendra/dataclean-env")
obs   = agent.reset(task_id="task_1", seed=42)

while True:
    action = agent.step(obs)
    # env drives the loop server-side
    break  # judge controls the loop externally

score = agent.score()
agent.close()
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for GPT-4o-mini baseline agent |
| `HF_TOKEN` | — | Required for Llama-3-70B baseline agent |
| `ENABLE_WEB_INTERFACE` | `true` | Mount Gradio UI at `/web` |
| `PORT` | `7860` | Server port |

---

## License

MIT License. Built for AiGenX Hackathon 2026 — OpenEnv Round 1.