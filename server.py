"""
DataClean-Env — FastAPI Server
OpenEnv-compliant: all 6 required endpoints + Gradio web UI mount.

Endpoints
---------
POST /reset          → start episode, returns Observation
POST /step           → apply action, returns StepResult
GET  /state          → current EpisodeState snapshot
GET  /tasks          → task list + action schema
POST /grader         → score current dataframe 0.0–1.0
GET  /baseline       → run GPT-4o-mini agent on all 3 tasks
GET  /health         → liveness probe
"""
from __future__ import annotations

import os
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dataclean.env import DataCleanEnv
from dataclean.models import DataCleanAction, Observation, StepResult
from dataclean.tasks import TASK_REGISTRY


# ── Session store (episode_id → env instance) ────────────────────────────────
# Supports concurrent sessions — each gets its own DataCleanEnv.

_sessions: dict[str, DataCleanEnv] = {}
_MAX_SESSIONS = 50   # evict oldest when limit hit


def _get_env(episode_id: str) -> DataCleanEnv:
    if episode_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found. Call /reset first.")
    return _sessions[episode_id]


def _new_session() -> tuple[str, DataCleanEnv]:
    episode_id = str(uuid.uuid4())
    if len(_sessions) >= _MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
    env = DataCleanEnv()
    _sessions[episode_id] = env
    return episode_id, env


# ── Request / response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1"
    seed: int = 42


class StepRequest(BaseModel):
    episode_id: str
    action: DataCleanAction


class GraderRequest(BaseModel):
    episode_id: str


class BaselineRequest(BaseModel):
    task_ids: list[str] = ["task_1", "task_2", "task_3"]
    seed: int = 42
    max_steps: int = 20


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DataClean-Env server starting up...")
    yield
    print("DataClean-Env server shutting down.")


app = FastAPI(
    title="DataClean-Env",
    description=(
        "OpenEnv-compliant RL environment for data cleaning agents. "
        "Tests null imputation, dtype fixing, outlier clipping, deduplication, "
        "provenance tracking, and confidence calibration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 1. GET /health ────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — openenv validate hits this first."""
    return {"status": "ok", "environment": "DataClean-Env", "version": "1.0.0"}


# ── 2. POST /reset ────────────────────────────────────────────────────────────

@app.post("/reset", tags=["openenv"], response_model=dict)
def reset(req: ResetRequest):
    """
    Start a new episode. Returns the initial Observation plus the episode_id
    you must pass to /step and /grader.
    """
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_REGISTRY)}"
        )

    episode_id, env = _new_session()
    obs: Observation = env.reset(req.task_id, seed=req.seed)

    # Overwrite episode_id so it matches our session key
    obs_dict = obs.model_dump()
    obs_dict["episode_id"] = episode_id
    _sessions[episode_id]._state.episode_id = episode_id

    return {
        "episode_id": episode_id,
        "observation": obs_dict,
    }


# ── 3. POST /step ─────────────────────────────────────────────────────────────

@app.post("/step", tags=["openenv"], response_model=dict)
def step(req: StepRequest):
    """
    Apply one DataCleanAction to the episode. Returns observation, reward, done, info.
    """
    env = _get_env(req.episode_id)

    try:
        result: StepResult = env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation":       result.observation.model_dump(),
        "reward":            result.reward,
        "done":              result.done,
        "info":              result.info,
        "reward_breakdown":  result.reward_breakdown,
    }


# ── 4. GET /state ─────────────────────────────────────────────────────────────

@app.get("/state", tags=["openenv"])
def state(episode_id: str):
    """Return the current EpisodeState snapshot (step count, total reward, ops log, done flag)."""
    env = _get_env(episode_id)
    if env._state is None:
        raise HTTPException(status_code=400, detail="Episode not initialised.")
    return env._state.model_dump()


# ── 5. GET /tasks ─────────────────────────────────────────────────────────────

@app.get("/tasks", tags=["openenv"])
def tasks():
    """
    Return all available tasks + the full action schema.
    The agent uses this to understand what actions are valid.
    """
    task_list = []
    for tid, task in TASK_REGISTRY.items():
        task_list.append({
            "task_id":     tid,
            "description": task.description,
            "max_steps":   task.max_steps,
            "canonical_column_names": task.canonical_column_names,
            "irrelevant_columns":     task.irrelevant_columns,
        })

    action_schema = {
        "action_type": {
            "type": "string",
            "enum": [
                "fill_nulls",
                "remove_duplicates",
                "fix_dtype",
                "clip_outliers",
                "rename_column",
                "drop_column",
                "done",
            ],
        },
        "column": {
            "type": "string | null",
            "description": "Target column name. Required for all actions except remove_duplicates and done.",
        },
        "params": {
            "type": "object",
            "description": "Action-specific parameters.",
            "examples": {
                "fill_nulls":        {"strategy": "mean | median | mode | constant | ffill", "value": "<for constant>"},
                "fix_dtype":         {"target_dtype": "int64 | float64 | str | datetime64 | bool"},
                "clip_outliers":     {"method": "iqr | zscore | percentile", "threshold": 3.0},
                "rename_column":     {"new_name": "<string>"},
                "remove_duplicates": {"subset": "<list of columns or null for all>"},
            },
        },
        "confidence": {
            "type": "float",
            "range": [0.0, 1.0],
            "description": (
                "Agent's self-reported confidence in this action. "
                "High confidence (≥0.75) + correct action → +0.04 bonus. "
                "High confidence + wrong action → −0.06 penalty. "
                "Calibrated agents score significantly higher."
            ),
        },
    }

    return {
        "tasks":         task_list,
        "action_schema": action_schema,
        "reward_info": {
            "fill_nulls":        "+0.10 per column (scaled by remaining null rate)",
            "remove_duplicates": "+0.12",
            "fix_dtype":         "+0.10",
            "clip_outliers":     "+0.08 (scaled by std reduction)",
            "rename_column":     "+0.05 (canonical) / +0.025 (non-canonical)",
            "drop_column":       "+0.04 (irrelevant) / −0.025 (relevant)",
            "done_bonus":        "+0.15 if quality ≥ 0.80, else +0.15 × quality",
            "provenance_bonus":  "+0.05 if ops log is fully reproducible",
            "step_penalty":      "−0.01 per step",
            "confidence_bonus":  "+0.04 for high-confidence correct action",
            "confidence_penalty": "−0.06 for high-confidence wrong action",
        },
    }


# ── 6. POST /grader ───────────────────────────────────────────────────────────

@app.post("/grader", tags=["openenv"])
def grader(req: GraderRequest):
    """
    Run the task grader on the current dataframe state.
    Returns a score in [0.0, 1.0] plus dimension breakdown.
    """
    env = _get_env(req.episode_id)

    if env._df is None or env._task is None:
        raise HTTPException(status_code=400, detail="No active episode dataframe.")

    score = env.grade()
    quality_scores = env._compute_quality_scores()
    provenance_ok  = env.verify_provenance()

    return {
        "score":          score,
        "quality_scores": quality_scores,
        "provenance_reproducible": provenance_ok,
        "ops_log_length": len(env.get_ops_log()),
    }


# ── 7. GET /baseline ─────────────────────────────────────────────────────────

@app.get("/baseline", tags=["openenv"])
def baseline(seed: int = 42):
    """
    Run a simple heuristic baseline agent on all 3 tasks and return scores.
    The heuristic agent applies a fixed cleaning sequence without an LLM —
    this establishes the lower-bound score judges use for comparison.

    For the GPT-4o-mini agent baseline, see baseline/agent.py.
    """
    results = {}

    for task_id in TASK_REGISTRY:
        env = DataCleanEnv()
        obs = env.reset(task_id, seed=seed)
        total_reward = 0.0

        # Heuristic agent: follow corruption_flags in order
        for _ in range(30):
            if env._state.done:
                break

            action = _heuristic_action(obs)
            result = env.step(action)
            total_reward += result.reward
            obs = result.observation

            if result.done:
                break

        final_score = env.grade()
        results[task_id] = {
            "total_reward":     round(total_reward, 4),
            "final_score":      round(final_score, 4),
            "steps_used":       env._state.step,
            "quality_scores":   env._compute_quality_scores(),
            "provenance_ok":    env.verify_provenance(),
        }

    return {
        "agent":   "heuristic (no LLM)",
        "seed":    seed,
        "results": results,
    }


def _heuristic_action(obs: Observation) -> DataCleanAction:
    """
    Deterministic heuristic: scan columns for issues and fix them in priority order.
    Priority: duplicates → nulls → type_chaos → outliers → done
    """
    # 1. Remove duplicates first
    if obs.duplicate_rate > 0.005:
        return DataCleanAction(
            action_type="remove_duplicates",
            confidence=0.95,
        )

    # 2. Fix type_chaos columns (fix_dtype before fill so numeric ops work)
    for col in obs.columns:
        if "type_chaos" in col.corruption_flags:
            return DataCleanAction(
                action_type="fix_dtype",
                column=col.name,
                params={"target_dtype": "float64"},
                confidence=0.85,
            )

    # 3. Fill nulls (heaviest first)
    nulls_cols = sorted(
        [c for c in obs.columns if c.null_rate > 0.01],
        key=lambda c: c.null_rate,
        reverse=True,
    )
    for col in nulls_cols:
        strategy = "median" if col.mean is not None else "mode"
        return DataCleanAction(
            action_type="fill_nulls",
            column=col.name,
            params={"strategy": strategy},
            confidence=0.80,
        )

    # 4. Clip outliers
    for col in obs.columns:
        if "heavy_outliers" in col.corruption_flags:
            return DataCleanAction(
                action_type="clip_outliers",
                column=col.name,
                params={"method": "iqr"},
                confidence=0.80,
            )

    # 5. Done
    return DataCleanAction(action_type="done", confidence=0.70)


# ── Gradio web UI (optional, enabled via env var) ─────────────────────────────

def _build_gradio_app():
    """Build the Gradio interactive demo UI."""
    try:
        import gradio as gr
    except ImportError:
        return None

    _ui_env = DataCleanEnv()
    _ui_state: dict[str, Any] = {"episode_id": None, "obs": None}

    def ui_reset(task_id: str, seed: int):
        obs = _ui_env.reset(task_id, seed=int(seed))
        _ui_state["obs"] = obs
        return (
            obs.to_prompt(),
            f"Episode started. Quality: {obs.quality_scores}",
            json.dumps(obs.quality_scores, indent=2),
        )

    def ui_step(action_type: str, column: str, strategy: str,
                target_dtype: str, clip_method: str, confidence: float):
        if _ui_env._state is None or _ui_env._state.done:
            return "No active episode. Click Reset first.", "", ""

        params: dict[str, Any] = {}
        col = column.strip() or None

        if action_type == "fill_nulls" and strategy:
            params["strategy"] = strategy
        elif action_type == "fix_dtype" and target_dtype:
            params["target_dtype"] = target_dtype
        elif action_type == "clip_outliers" and clip_method:
            params["method"] = clip_method

        try:
            action = DataCleanAction(
                action_type=action_type,
                column=col,
                params=params,
                confidence=confidence,
            )
            result = _ui_env.step(action)
            obs = result.observation
            _ui_state["obs"] = obs

            breakdown = json.dumps(result.reward_breakdown, indent=2)
            status = (
                f"Step {obs.step} | Reward: {result.reward:+.4f} | "
                f"Done: {result.done}\n{obs.last_action_result}"
            )
            return obs.to_prompt(), status, breakdown
        except Exception as exc:
            return str(exc), f"Error: {exc}", ""

    def ui_grade():
        if _ui_env._df is None:
            return "No active episode."
        score = _ui_env.grade()
        prov  = _ui_env.verify_provenance()
        qs    = _ui_env._compute_quality_scores()
        return json.dumps({"score": score, "provenance": prov, "breakdown": qs}, indent=2)

    with gr.Blocks(title="DataClean-Env", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧹 DataClean-Env\nInteractive RL environment for data cleaning agents.")

        with gr.Row():
            task_dd   = gr.Dropdown(["task_1", "task_2", "task_3"], value="task_1", label="Task")
            seed_num  = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("▶ Reset", variant="primary")

        obs_box    = gr.Textbox(label="Observation", lines=20, max_lines=30)
        status_box = gr.Textbox(label="Last action result")

        gr.Markdown("### Take an action")
        with gr.Row():
            action_dd = gr.Dropdown(
                ["fill_nulls", "remove_duplicates", "fix_dtype",
                 "clip_outliers", "rename_column", "drop_column", "done"],
                value="remove_duplicates", label="Action type"
            )
            col_txt    = gr.Textbox(label="Column (leave blank if N/A)")
            conf_sl    = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Confidence")

        with gr.Row():
            strategy_dd  = gr.Dropdown(["mean", "median", "mode", "constant", "ffill"],
                                        value="median", label="fill_nulls strategy")
            dtype_dd     = gr.Dropdown(["float64", "int64", "str", "datetime64", "bool"],
                                        value="float64", label="fix_dtype target")
            clip_dd      = gr.Dropdown(["iqr", "zscore", "percentile"],
                                        value="iqr", label="clip_outliers method")

        step_btn   = gr.Button("⚡ Step", variant="primary")
        reward_box = gr.Textbox(label="Reward breakdown (JSON)")
        grade_btn  = gr.Button("📊 Grade current state")
        grade_box  = gr.Textbox(label="Grade result")

        reset_btn.click(ui_reset, [task_dd, seed_num], [obs_box, status_box, reward_box])
        step_btn.click(ui_step,
                       [action_dd, col_txt, strategy_dd, dtype_dd, clip_dd, conf_sl],
                       [obs_box, status_box, reward_box])
        grade_btn.click(ui_grade, [], [grade_box])

    return demo


# Mount Gradio if enabled
if os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() == "true":
    try:
        import gradio as gr
        from gradio.routes import mount_gradio_app
        demo = _build_gradio_app()
        if demo:
            app = mount_gradio_app(app, demo, path="/web")
            print("Gradio UI mounted at /web")
    except ImportError:
        print("Gradio not installed — skipping web UI. Run: pip install gradio")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        workers=1,
        reload=False,
    )
