"""
DataClean-Env — Pydantic models
Observation, Action, StepResult, EpisodeState, OpsLog
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ── Action types ────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    FILL_NULLS       = "fill_nulls"
    REMOVE_DUPLICATES = "remove_duplicates"
    FIX_DTYPE        = "fix_dtype"
    CLIP_OUTLIERS    = "clip_outliers"
    RENAME_COLUMN    = "rename_column"
    DROP_COLUMN      = "drop_column"
    DONE             = "done"          # Agent signals episode complete


# ── Column-level stats inside Observation ───────────────────────────────────

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_rate: float = Field(ge=0.0, le=1.0)
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    sample_values: list[Any] = Field(default_factory=list)
    corruption_flags: list[str] = Field(
        default_factory=list,
        description="Detected issues: heavy_nulls, heavy_outliers, type_chaos, duplicates"
    )


# ── Observation returned by reset() and step() ──────────────────────────────

class Observation(BaseModel):
    task_id: str
    episode_id: str
    step: int
    budget_remaining: int
    n_rows: int
    n_cols: int
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    columns: list[ColumnProfile]
    last_action_result: str | None = None   # Human-readable feedback on last op
    ops_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Immutable log of all operations applied so far"
    )
    # Grader preview — partial scores so agent can self-monitor
    quality_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension quality 0..1 (null_score, type_score, outlier_score, dup_score)"
    )

    def to_prompt(self) -> str:
        """Render observation as an LLM-friendly prompt string."""
        col_lines = []
        for c in self.columns:
            flags = ", ".join(c.corruption_flags) if c.corruption_flags else "clean"
            stats = ""
            if c.mean is not None:
                stats = f" | mean={c.mean:.2f}, std={c.std:.2f}, min={c.min:.2f}, max={c.max:.2f}"
            col_lines.append(
                f"  - {c.name} [{c.dtype}] nulls={c.null_rate:.1%} unique={c.n_unique}{stats} flags=[{flags}]"
            )
        cols_text = "\n".join(col_lines)
        scores_text = " | ".join(f"{k}={v:.3f}" for k, v in self.quality_scores.items())
        ops_text = f"{len(self.ops_log)} operations applied" if self.ops_log else "no operations yet"

        return f"""=== DataClean-Env | Task: {self.task_id} | Episode: {self.episode_id} ===
Step: {self.step} | Budget remaining: {self.budget_remaining}
Rows: {self.n_rows} | Cols: {self.n_cols} | Duplicate rate: {self.duplicate_rate:.1%}

Column profiles:
{cols_text}

Quality scores: {scores_text}
Ops log: {ops_text}
Last action result: {self.last_action_result or "N/A"}

Respond ONLY with a JSON object matching the DataCleanAction schema:
{{
  "action_type": "<one of: fill_nulls | remove_duplicates | fix_dtype | clip_outliers | rename_column | drop_column | done>",
  "column": "<target column name, or null if not applicable>",
  "params": {{}},
  "confidence": <float 0.0-1.0>
}}
"""


# ── Action sent by the agent ─────────────────────────────────────────────────

class DataCleanAction(BaseModel):
    action_type: ActionType
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this action (used for calibration reward)"
    )

    @field_validator("column")
    @classmethod
    def strip_column(cls, v: str | None) -> str | None:
        return v.strip() if v else v

    class Config:
        use_enum_values = True


# ── Single step result ───────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

    # Reward breakdown for transparency / debugging
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Itemised reward components"
    )


# ── Full episode state (internal) ────────────────────────────────────────────

class EpisodeState(BaseModel):
    task_id: str
    episode_id: str
    seed: int
    step: int = 0
    max_steps: int = 20
    total_reward: float = 0.0
    done: bool = False
    ops_log: list[dict[str, Any]] = Field(default_factory=list)
    final_score: float | None = None

    class Config:
        arbitrary_types_allowed = True


# ── Ops log entry (immutable record) ────────────────────────────────────────

class OpsLogEntry(BaseModel):
    step: int
    action_type: str
    column: str | None
    params: dict[str, Any]
    confidence: float
    reward_delta: float
    timestamp_ns: int           # monotonic nanoseconds for ordering
    df_shape_before: tuple[int, int]
    df_shape_after: tuple[int, int]
    success: bool
    error_message: str | None = None
