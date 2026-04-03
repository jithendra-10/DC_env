"""
DataClean-Env — Core Environment
Handles episode lifecycle, action execution, reward computation,
provenance/ops-log tracking, and confidence-calibration scoring.
"""
from __future__ import annotations

import time
import uuid
import json
import hashlib
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from dataclean.models import (
    ActionType,
    ColumnProfile,
    DataCleanAction,
    EpisodeState,
    Observation,
    OpsLogEntry,
    StepResult,
)
from dataclean.tasks import TASK_REGISTRY, Task


# ── Reward constants ─────────────────────────────────────────────────────────

R_NULL_FIXED        =  0.10   # per-column null rate drops below threshold
R_DTYPE_FIXED       =  0.10   # column dtype corrected
R_OUTLIER_CLIPPED   =  0.08   # outliers clipped correctly
R_DUPLICATE_REMOVED =  0.12   # duplicate rows removed
R_RENAME_CORRECT    =  0.05   # column renamed to canonical name
R_DROP_CORRECT      =  0.04   # irrelevant column dropped
R_DONE_BONUS        =  0.15   # agent calls DONE and quality meets threshold
R_STEP_PENALTY      = -0.01   # per step (efficiency pressure)
R_WRONG_COLUMN      = -0.05   # action on wrong/clean column
R_INVALID_OP        = -0.03   # malformed params / unsupported op
R_PROVENANCE_BONUS  =  0.05   # ops log is reproducible (checked at episode end)

# Quality threshold to unlock DONE bonus
QUALITY_THRESHOLD   =  0.80

# Confidence calibration
CONFIDENCE_BONUS    =  0.04   # high-confidence + correct action
CONFIDENCE_PENALTY  = -0.06   # high-confidence + wrong action
CONFIDENCE_HIGH     =  0.75   # threshold for "high confidence"


# ── Main environment class ────────────────────────────────────────────────────

class DataCleanEnv:
    """
    OpenEnv-compatible data cleaning environment.

    Lifecycle
    ---------
    env = DataCleanEnv()
    obs = env.reset(task_id="task_1", seed=42)
    result = env.step(DataCleanAction(action_type="fill_nulls", column="age", confidence=0.9))
    ...
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._raw_df: pd.DataFrame | None = None   # pristine copy for provenance check
        self._task: Task | None = None
        self._state: EpisodeState | None = None
        self._ops_log: list[OpsLogEntry] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str, seed: int = 42, custom_df: pd.DataFrame | None = None) -> Observation:
        """Start a new episode. Returns the first observation."""
        rng = np.random.default_rng(seed)

        if custom_df is not None:
            # --- Dynamic Sandbox Task ---
            class CustomTask(Task):
                def __init__(self):
                    self.description = "Dynamic user uploaded sandbox dataset."
                    self.max_steps = 30
                    self.canonical_column_names = {}
                    self.irrelevant_columns = []
                def generate(self, r): return custom_df.copy(deep=True)
                def grade(self, df):
                    # We grade custom CSVs dynamically based entirely on data-type math!
                    null_score = float(1 - df.isna().mean().mean())
                    dup_score  = float(1 - df.duplicated().mean())
                    overall = np.mean([null_score, dup_score])
                    return float(round(overall, 4))
            self._task = CustomTask()
            self._df = self._task.generate(rng)
        else:
            if task_id not in TASK_REGISTRY:
                raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY)}")
            self._task = TASK_REGISTRY[task_id]
            self._df = self._task.generate(rng)

        self._raw_df = self._df.copy(deep=True)   # immutable snapshot
        self._ops_log = []

        self._state = EpisodeState(
            task_id="custom_upload" if custom_df is not None else task_id,
            episode_id=str(uuid.uuid4()),
            seed=seed,
            step=0,
            max_steps=self._task.max_steps,
        )

        return self._build_observation(last_action_result="Episode started. Inspect columns and begin cleaning.")

    def step(self, action: DataCleanAction) -> StepResult:
        """Apply one action. Returns (observation, reward, done, info)."""
        if self._state is None or self._df is None or self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        t0 = time.monotonic_ns()
        shape_before = self._df.shape
        reward = R_STEP_PENALTY   # base penalty every step
        reward_breakdown: dict[str, float] = {"step_penalty": R_STEP_PENALTY}
        success = True
        error_msg: str | None = None
        feedback: str = ""

        # ── Dispatch action ──────────────────────────────────────────────────
        try:
            action_reward, feedback = self._dispatch(action)
            reward += action_reward
            reward_breakdown["action"] = action_reward
        except Exception as exc:
            reward += R_INVALID_OP
            reward_breakdown["invalid_op"] = R_INVALID_OP
            success = False
            error_msg = str(exc)
            feedback = f"Error: {exc}"

        # ── Confidence calibration ───────────────────────────────────────────
        conf_reward = self._confidence_reward(action.confidence, action_reward if success else R_INVALID_OP)
        reward += conf_reward
        reward_breakdown["confidence"] = conf_reward

        # ── Ops log entry ────────────────────────────────────────────────────
        entry = OpsLogEntry(
            step=self._state.step,
            action_type=action.action_type,
            column=action.column,
            params=action.params,
            confidence=action.confidence,
            reward_delta=reward,
            timestamp_ns=t0,
            df_shape_before=shape_before,
            df_shape_after=self._df.shape,
            success=success,
            error_message=error_msg,
        )
        self._ops_log.append(entry)
        self._state.ops_log.append(entry.model_dump())

        # ── Update state ─────────────────────────────────────────────────────
        self._state.step += 1
        self._state.total_reward += reward

        # ── Terminal conditions ───────────────────────────────────────────────
        done = False
        if action.action_type == ActionType.DONE:
            done = True
            done_reward, provenance_reward = self._terminal_rewards()
            reward += done_reward + provenance_reward
            reward_breakdown["done_bonus"] = done_reward
            reward_breakdown["provenance"] = provenance_reward
            self._state.total_reward += done_reward + provenance_reward

        if self._state.step >= self._state.max_steps:
            done = True
            feedback += " | Budget exhausted."

        self._state.done = done
        obs = self._build_observation(last_action_result=feedback)

        return StepResult(
            observation=obs,
            reward=round(reward, 6),
            done=done,
            info={
                "step": self._state.step,
                "total_reward": round(self._state.total_reward, 6),
                "success": success,
                "error": error_msg,
            },
            reward_breakdown=reward_breakdown,
        )

    def grade(self) -> float:
        """Return final quality score 0..1 using the task's grader."""
        if self._df is None or self._task is None:
            raise RuntimeError("No active episode.")
        return self._task.grade(self._df)

    def verify_provenance(self) -> bool:
        """
        Replay all ops from the raw dataframe and check if the result
        matches the current dataframe. True = ops log is reproducible.
        """
        if self._raw_df is None or not self._ops_log:
            return False
        try:
            replay_df = self._raw_df.copy(deep=True)
            for entry in self._ops_log:
                if not entry.success:
                    continue
                action = DataCleanAction(
                    action_type=entry.action_type,
                    column=entry.column,
                    params=entry.params,
                    confidence=entry.confidence,
                )
                replay_df = self._apply_action(replay_df, action)
            # Compare by hash of CSV representation (order-insensitive)
            return self._df_hash(replay_df) == self._df_hash(self._df)
        except Exception:
            return False

    def get_ops_log(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self._ops_log]

    # ── Private: action dispatch ──────────────────────────────────────────────

    def _dispatch(self, action: DataCleanAction) -> tuple[float, str]:
        """Route action to handler. Returns (reward, feedback)."""
        at = action.action_type

        if at == ActionType.DONE:
            return 0.0, "Agent signalled DONE."

        if at == ActionType.REMOVE_DUPLICATES:
            return self._act_remove_duplicates(action)

        # All remaining actions need a column
        if not action.column:
            raise ValueError(f"Action '{at}' requires a column name.")
        if action.column not in self._df.columns:
            raise ValueError(f"Column '{action.column}' not found in dataframe.")

        if at == ActionType.FILL_NULLS:
            return self._act_fill_nulls(action)
        if at == ActionType.FIX_DTYPE:
            return self._act_fix_dtype(action)
        if at == ActionType.CLIP_OUTLIERS:
            return self._act_clip_outliers(action)
        if at == ActionType.RENAME_COLUMN:
            return self._act_rename_column(action)
        if at == ActionType.DROP_COLUMN:
            return self._act_drop_column(action)

        raise ValueError(f"Unknown action_type: {at}")

    def _apply_action(self, df: pd.DataFrame, action: DataCleanAction) -> pd.DataFrame:
        """Pure function — apply action to df and return result (for provenance replay)."""
        env = DataCleanEnv.__new__(DataCleanEnv)
        env._df = df
        env._task = self._task
        env._dispatch(action)
        return env._df

    # ── Action handlers ───────────────────────────────────────────────────────

    def _act_fill_nulls(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        strategy = action.params.get("strategy", "mean")
        null_before = self._df[col].isna().mean()

        if null_before < 0.005:
            return R_WRONG_COLUMN, f"Column '{col}' is already clean (null_rate={null_before:.1%})."

        if strategy == "mean":
            val = self._df[col].mean()
        elif strategy == "median":
            val = self._df[col].median()
        elif strategy == "mode":
            val = self._df[col].mode().iloc[0] if not self._df[col].mode().empty else None
        elif strategy == "constant":
            val = action.params.get("value", 0)
        elif strategy == "ffill":
            self._df[col] = self._df[col].fillna(method="ffill")
            null_after = self._df[col].isna().mean()
            return R_NULL_FIXED * (1 - null_after), f"Filled nulls in '{col}' with ffill. null_rate: {null_before:.1%} → {null_after:.1%}"
        else:
            raise ValueError(f"Unknown fill strategy '{strategy}'. Use: mean|median|mode|constant|ffill")

        if val is None:
            raise ValueError(f"Could not compute fill value for column '{col}'.")

        self._df[col] = self._df[col].fillna(val)
        null_after = self._df[col].isna().mean()
        reward = R_NULL_FIXED * (1 - null_after)
        return reward, f"Filled nulls in '{col}' (strategy={strategy}, value={val:.4g}). null_rate: {null_before:.1%} → {null_after:.1%}"

    def _act_remove_duplicates(self, action: DataCleanAction) -> tuple[float, str]:
        subset = action.params.get("subset", None)
        before = len(self._df)
        dup_rate_before = self._df.duplicated(subset=subset).mean()

        if dup_rate_before < 0.005:
            return R_WRONG_COLUMN, f"No significant duplicates found (dup_rate={dup_rate_before:.1%})."

        self._df = self._df.drop_duplicates(subset=subset).reset_index(drop=True)
        after = len(self._df)
        removed = before - after
        return R_DUPLICATE_REMOVED, f"Removed {removed} duplicate rows. Shape: ({before},{self._df.shape[1]}) → ({after},{self._df.shape[1]})"

    def _act_fix_dtype(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        target_dtype = action.params.get("target_dtype")
        if not target_dtype:
            raise ValueError("fix_dtype requires params.target_dtype (e.g. 'int64', 'float64', 'str').")

        before_dtype = str(self._df[col].dtype)
        if before_dtype == target_dtype:
            return R_WRONG_COLUMN, f"Column '{col}' is already dtype '{target_dtype}'."

        try:
            if target_dtype in ("int64", "int32", "int"):
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
            elif target_dtype in ("float64", "float32", "float"):
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("float64")
            elif target_dtype in ("str", "string", "object"):
                self._df[col] = self._df[col].astype(str)
            elif target_dtype in ("datetime64", "datetime"):
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce")
            elif target_dtype == "bool":
                self._df[col] = self._df[col].astype(bool)
            else:
                self._df[col] = self._df[col].astype(target_dtype)
        except Exception as exc:
            raise ValueError(f"Could not cast '{col}' to '{target_dtype}': {exc}")

        return R_DTYPE_FIXED, f"Fixed dtype of '{col}': {before_dtype} → {str(self._df[col].dtype)}"

    def _act_clip_outliers(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        method = action.params.get("method", "iqr")

        if not pd.api.types.is_numeric_dtype(self._df[col]):
            raise ValueError(f"Column '{col}' is not numeric; cannot clip outliers.")

        before_std = self._df[col].std()

        if method == "iqr":
            q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif method == "zscore":
            mu, sigma = self._df[col].mean(), self._df[col].std()
            threshold = action.params.get("threshold", 3.0)
            lo, hi = mu - threshold * sigma, mu + threshold * sigma
        elif method == "percentile":
            lo_pct = action.params.get("lower_pct", 1)
            hi_pct = action.params.get("upper_pct", 99)
            lo, hi = self._df[col].quantile(lo_pct / 100), self._df[col].quantile(hi_pct / 100)
        else:
            raise ValueError(f"Unknown clip method '{method}'. Use: iqr|zscore|percentile")

        outliers_before = ((self._df[col] < lo) | (self._df[col] > hi)).sum()
        if outliers_before == 0:
            return R_WRONG_COLUMN, f"No outliers detected in '{col}' using method={method}."

        self._df[col] = self._df[col].clip(lower=lo, upper=hi)
        after_std = self._df[col].std()
        reduction = 1 - (after_std / before_std) if before_std > 0 else 0
        reward = R_OUTLIER_CLIPPED * (1 + reduction)
        return reward, f"Clipped {outliers_before} outliers in '{col}' (method={method}, lo={lo:.3g}, hi={hi:.3g}). std: {before_std:.3g} → {after_std:.3g}"

    def _act_rename_column(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        new_name = action.params.get("new_name")
        if not new_name:
            raise ValueError("rename_column requires params.new_name.")
        if new_name in self._df.columns:
            raise ValueError(f"Column '{new_name}' already exists.")

        canonical = self._task.canonical_column_names if self._task else {}
        self._df.rename(columns={col: new_name}, inplace=True)

        if new_name in canonical.values():
            return R_RENAME_CORRECT, f"Renamed '{col}' → '{new_name}' (matches canonical schema)."
        return R_RENAME_CORRECT * 0.5, f"Renamed '{col}' → '{new_name}' (non-canonical, partial credit)."

    def _act_drop_column(self, action: DataCleanAction) -> tuple[float, str]:
        col = action.column
        irrelevant = self._task.irrelevant_columns if self._task else []

        self._df.drop(columns=[col], inplace=True)

        if col in irrelevant:
            return R_DROP_CORRECT, f"Dropped irrelevant column '{col}' (correct)."
        return R_WRONG_COLUMN * 0.5, f"Dropped column '{col}' — not marked irrelevant (partial penalty)."

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _confidence_reward(self, confidence: float, action_reward: float) -> float:
        """Reward calibration: high-confidence correct = bonus; high-confidence wrong = penalty."""
        if confidence >= CONFIDENCE_HIGH:
            if action_reward > 0:
                return CONFIDENCE_BONUS
            else:
                return CONFIDENCE_PENALTY
        return 0.0

    def _terminal_rewards(self) -> tuple[float, float]:
        """Compute DONE bonus + provenance bonus."""
        quality = self.grade()
        done_reward = R_DONE_BONUS if quality >= QUALITY_THRESHOLD else R_DONE_BONUS * quality
        provenance_reward = R_PROVENANCE_BONUS if self.verify_provenance() else 0.0
        return done_reward, provenance_reward

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(self, last_action_result: str | None = None) -> Observation:
        assert self._df is not None
        assert self._state is not None

        columns = []
        for col in self._df.columns:
            series = self._df[col]
            null_rate = float(series.isna().mean())
            flags: list[str] = []

            if null_rate > 0.20:
                flags.append("heavy_nulls")
            if null_rate > 0.01:
                flags.append("has_nulls")

            mean_val = std_val = min_val = max_val = None
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 0:
                    mean_val = float(clean.mean())
                    std_val  = float(clean.std()) if len(clean) > 1 else 0.0
                    min_val  = float(clean.min())
                    max_val  = float(clean.max())
                    # IQR outlier check
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    n_outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                    if n_outliers / max(len(clean), 1) > 0.05:
                        flags.append("heavy_outliers")

            # Type chaos: numeric column stored as object
            if series.dtype == object:
                numeric_parseable = pd.to_numeric(series.dropna(), errors="coerce").notna().mean()
                if numeric_parseable > 0.7:
                    flags.append("type_chaos")

            sample = series.dropna().head(3).tolist()

            columns.append(ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_rate=null_rate,
                n_unique=int(series.nunique()),
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                sample_values=sample,
                corruption_flags=flags,
            ))

        dup_rate = float(self._df.duplicated().mean())
        quality_scores = self._compute_quality_scores()

        return Observation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id,
            step=self._state.step,
            budget_remaining=self._state.max_steps - self._state.step,
            n_rows=len(self._df),
            n_cols=len(self._df.columns),
            duplicate_rate=dup_rate,
            columns=columns,
            last_action_result=last_action_result,
            ops_log=self._state.ops_log,
            quality_scores=quality_scores,
        )

    def _compute_quality_scores(self) -> dict[str, float]:
        """Quick quality preview: 0..1 per dimension."""
        df = self._df
        null_score = float(1 - df.isna().mean().mean())
        dup_score  = float(1 - df.duplicated().mean())

        # Type score: fraction of numeric-intended columns with correct dtype
        type_issues = 0
        numeric_cols = 0
        for col in df.columns:
            if df[col].dtype == object:
                numeric_parseable = pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean()
                if numeric_parseable > 0.7:
                    numeric_cols += 1
                    type_issues += 1
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols += 1

        type_score = 1.0 - (type_issues / max(numeric_cols, 1))

        # Outlier score
        outlier_fracs = []
        for col in df.select_dtypes(include="number").columns:
            clean = df[col].dropna()
            if len(clean) < 4:
                continue
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            frac = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean()
            outlier_fracs.append(frac)
        outlier_score = float(1 - np.mean(outlier_fracs)) if outlier_fracs else 1.0

        return {
            "null_score":    round(null_score, 4),
            "type_score":    round(type_score, 4),
            "outlier_score": round(outlier_score, 4),
            "dup_score":     round(dup_score, 4),
            "overall":       round(np.mean([null_score, type_score, outlier_score, dup_score]), 4),
        }

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _df_hash(df: pd.DataFrame) -> str:
        """Stable hash of a dataframe (sort first to be order-insensitive)."""
        sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        return hashlib.md5(sorted_df.to_csv(index=False).encode()).hexdigest()
