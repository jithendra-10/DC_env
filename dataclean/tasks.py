"""
DataClean-Env — Task Registry
3 tasks with escalating difficulty, per-column corruption profiles,
and fully deterministic pandas graders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ── Task dataclass ────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    description: str
    max_steps: int
    generate: Callable[[np.random.Generator], pd.DataFrame]
    grade: Callable[[pd.DataFrame], float]
    canonical_column_names: dict[str, str] = field(default_factory=dict)
    irrelevant_columns: list[str] = field(default_factory=list)


# ── Shared grader helpers ─────────────────────────────────────────────────────

def _null_score(df: pd.DataFrame, cols: list[str], threshold: float = 0.01) -> float:
    scores = [1.0 if df[c].isna().mean() <= threshold else max(0, 1 - df[c].isna().mean() * 5) for c in cols if c in df.columns]
    return float(np.mean(scores)) if scores else 0.0

def _dtype_score(df: pd.DataFrame, expected: dict[str, str]) -> float:
    scores = []
    for col, expected_kind in expected.items():
        if col not in df.columns:
            scores.append(0.0); continue
        actual = df[col].dtype
        if expected_kind == "numeric":
            scores.append(1.0 if pd.api.types.is_numeric_dtype(actual) else 0.0)
        elif expected_kind == "string":
            scores.append(1.0 if actual == object else 0.5)
        elif expected_kind == "datetime":
            scores.append(1.0 if pd.api.types.is_datetime64_any_dtype(actual) else 0.0)
        else:
            scores.append(1.0 if str(actual) == expected_kind else 0.0)
    return float(np.mean(scores)) if scores else 1.0

def _outlier_score(df: pd.DataFrame, cols: list[str], max_frac: float = 0.05) -> float:
    scores = []
    for c in cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        clean = df[c].dropna()
        if len(clean) < 4:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        frac = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean()
        scores.append(1.0 if frac <= max_frac else max(0, 1 - frac * 10))
    return float(np.mean(scores)) if scores else 1.0

def _dup_score(df: pd.DataFrame) -> float:
    dup_rate = df.duplicated().mean()
    return 1.0 if dup_rate < 0.005 else max(0.0, 1 - dup_rate * 10)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Simple employee dataset
# Single-column imputation + basic dtype fix
# Difficulty: EASY  |  max_steps: 15
# ══════════════════════════════════════════════════════════════════════════════

def _generate_task1(rng: np.random.Generator) -> pd.DataFrame:
    n = 300
    ages      = rng.integers(22, 65, size=n).astype(float)
    salaries  = rng.normal(55_000, 12_000, n)
    depts     = rng.choice(["Engineering", "Marketing", "Sales", "HR"], size=n)
    tenure    = rng.integers(0, 20, size=n).astype(float)

    # Corruption profile
    # age: heavy_nulls
    null_idx = rng.choice(n, size=int(n * 0.22), replace=False)
    ages[null_idx] = np.nan

    # salary: outliers
    outlier_idx = rng.choice(n, size=12, replace=False)
    salaries[outlier_idx] = rng.choice([-99_999, 500_000, 999_000], size=12)

    # tenure: stored as string (type_chaos)
    tenure_str = [str(int(v)) if not np.isnan(v) else "N/A" for v in tenure]

    # 30 duplicate rows
    base_df = pd.DataFrame({"age": ages, "salary": salaries, "department": depts, "years_at_company": tenure_str})
    dupe_rows = base_df.sample(n=30, random_state=int(rng.integers(0, 9999)))
    df = pd.concat([base_df, dupe_rows], ignore_index=True)

    return df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)


def _grade_task1(df: pd.DataFrame) -> float:
    ns = _null_score(df, ["age", "salary", "years_at_company"])
    ts = _dtype_score(df, {"age": "numeric", "salary": "numeric", "years_at_company": "numeric"})
    os_ = _outlier_score(df, ["salary"])
    ds = _dup_score(df)
    return round(0.30 * ns + 0.25 * ts + 0.25 * os_ + 0.20 * ds, 4)


TASK_1 = Task(
    task_id="task_1",
    description=(
        "Employee dataset (300 rows). "
        "Challenges: 22% null ages, salary outliers, tenure stored as string, 30 duplicate rows. "
        "Grader checks null_rate ≤ 1%, numeric dtypes, IQR-clean salary, no duplicates."
    ),
    max_steps=15,
    generate=_generate_task1,
    grade=_grade_task1,
    canonical_column_names={"years_at_company": "years_at_company"},
    irrelevant_columns=[],
)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — E-commerce orders dataset
# Multi-column mixed corruption + irrelevant column to drop
# Difficulty: MEDIUM  |  max_steps: 18
# ══════════════════════════════════════════════════════════════════════════════

def _generate_task2(rng: np.random.Generator) -> pd.DataFrame:
    n = 500
    order_ids   = [f"ORD-{i:05d}" for i in range(n)]
    amounts     = rng.lognormal(mean=4.5, sigma=0.8, size=n)
    quantities  = rng.integers(1, 20, size=n).astype(float)
    ratings     = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
    categories  = rng.choice(["Electronics", "Clothing", "Food", "Books", "Sports"], size=n)
    timestamps  = pd.date_range("2023-01-01", periods=n, freq="2h").astype(str).tolist()

    # Irrelevant column: internal tracking hash
    tracking = [f"HASH-{rng.integers(0, 99999):05d}" for _ in range(n)]

    # Corruption profile per column
    # amounts: heavy_outliers
    outlier_idx = rng.choice(n, size=25, replace=False)
    amounts[outlier_idx] = rng.choice([-500, 50_000, 999_999], size=25)

    # quantities: heavy_nulls (28%)
    null_idx = rng.choice(n, size=int(n * 0.28), replace=False)
    quantities[null_idx] = np.nan

    # ratings: nulls (15%)
    null_r = rng.choice(n, size=int(n * 0.15), replace=False)
    ratings = ratings.astype(float)
    ratings[null_r] = np.nan

    # timestamps: type_chaos (stored as string already, intentional)
    # order_amount misspelled column name
    df = pd.DataFrame({
        "order_id":       order_ids,
        "order_amount":   amounts,
        "qty":            quantities,      # should be "quantity"
        "customer_rating": ratings,
        "category":       categories,
        "created_at":     timestamps,
        "internal_hash":  tracking,        # irrelevant — should be dropped
    })

    # 40 duplicates
    dupe_rows = df.sample(n=40, random_state=int(rng.integers(0, 9999)))
    df = pd.concat([df, dupe_rows], ignore_index=True)

    return df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)


def _grade_task2(df: pd.DataFrame) -> float:
    ns = _null_score(df, ["order_amount", "qty", "customer_rating"])
    ts = _dtype_score(df, {"order_amount": "numeric", "qty": "numeric", "customer_rating": "numeric"})
    os_ = _outlier_score(df, ["order_amount"])
    ds = _dup_score(df)
    # Bonus: irrelevant column dropped
    drop_bonus = 0.0 if "internal_hash" in df.columns else 0.05
    raw = 0.25 * ns + 0.25 * ts + 0.25 * os_ + 0.20 * ds + drop_bonus
    return round(min(raw, 1.0), 4)


TASK_2 = Task(
    task_id="task_2",
    description=(
        "E-commerce orders (500 rows). "
        "Challenges: 28% null quantities, 15% null ratings, amount outliers, "
        "40 duplicates, irrelevant 'internal_hash' column to drop, "
        "timestamps stored as strings. "
        "Grader checks all nulls, dtypes, outliers, duplicates, and whether 'internal_hash' was dropped."
    ),
    max_steps=18,
    generate=_generate_task2,
    grade=_grade_task2,
    canonical_column_names={"qty": "quantity"},
    irrelevant_columns=["internal_hash"],
)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Healthcare patient dataset
# Mixed corruption profiles per column — agent must diagnose each independently
# Difficulty: HARD  |  max_steps=20
# ══════════════════════════════════════════════════════════════════════════════

def _generate_task3(rng: np.random.Generator) -> pd.DataFrame:
    n = 800

    # Demographics
    ages       = rng.integers(18, 90, size=n).astype(float)
    genders    = rng.choice(["M", "F", "Other"], size=n, p=[0.48, 0.48, 0.04])
    weights_kg = rng.normal(75, 15, n)
    heights_cm = rng.normal(170, 10, n)

    # Clinical
    systolic   = rng.normal(120, 18, n)   # blood pressure
    glucose    = rng.normal(100, 25, n)   # mg/dL
    cholesterol = rng.normal(190, 35, n)
    diagnosis  = rng.choice(["Hypertension", "Diabetes", "Healthy", "Obesity", "Hyperlipidemia"], size=n)
    admission_date = pd.date_range("2020-01-01", periods=n, freq="6h").astype(str).tolist()

    # Admin (irrelevant)
    admin_notes = [f"note_{rng.integers(0, 999)}" for _ in range(n)]

    # === Per-column corruption profiles (mixed — agent must handle each differently) ===

    # age: heavy_nulls (25%)
    ages[rng.choice(n, size=int(n * 0.25), replace=False)] = np.nan

    # weight: heavy_outliers (physiologically impossible values)
    wt_outlier = rng.choice(n, size=30, replace=False)
    weights_kg[wt_outlier] = rng.choice([-10, 5, 500, 999], size=30)

    # glucose: type_chaos — mix of numeric and string codes
    glucose = glucose.tolist()
    chaos_idx = rng.choice(n, size=int(n * 0.18), replace=False)
    for i in chaos_idx:
        glucose[i] = rng.choice(["N/A", "pending", "---", "HIGH", "??"])
    glucose = pd.array(glucose, dtype=object)

    # cholesterol: nulls (12%) + moderate outliers
    cholesterol = cholesterol.astype(float)
    cholesterol[rng.choice(n, size=int(n * 0.12), replace=False)] = np.nan
    chol_out = rng.choice(n, size=15, replace=False)
    cholesterol[chol_out] = rng.choice([-50, 1500, 2000], size=15)

    # systolic: nulls (8%)
    systolic = systolic.astype(float)
    systolic[rng.choice(n, size=int(n * 0.08), replace=False)] = np.nan

    df = pd.DataFrame({
        "patient_age":    ages,
        "gender":         genders,
        "weight_kg":      weights_kg,
        "height_cm":      heights_cm,
        "systolic_bp":    systolic,
        "glucose_mgdl":   glucose,
        "cholesterol":    cholesterol,
        "diagnosis":      diagnosis,
        "admission_date": admission_date,
        "admin_notes":    admin_notes,   # irrelevant
    })

    # 60 duplicates
    dupe_rows = df.sample(n=60, random_state=int(rng.integers(0, 9999)))
    df = pd.concat([df, dupe_rows], ignore_index=True)

    return df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)


def _grade_task3(df: pd.DataFrame) -> float:
    null_cols    = ["patient_age", "systolic_bp", "cholesterol"]
    outlier_cols = ["weight_kg", "cholesterol"]
    dtype_map    = {
        "patient_age": "numeric",
        "weight_kg": "numeric",
        "glucose_mgdl": "numeric",
        "cholesterol": "numeric",
        "systolic_bp": "numeric",
    }

    ns  = _null_score(df, null_cols)
    ts  = _dtype_score(df, dtype_map)
    os_ = _outlier_score(df, outlier_cols)
    ds  = _dup_score(df)

    # Glucose type chaos specifically rewarded
    glucose_clean = 0.0
    if "glucose_mgdl" in df.columns:
        numeric_frac = pd.to_numeric(df["glucose_mgdl"], errors="coerce").notna().mean()
        glucose_clean = float(numeric_frac)

    drop_bonus = 0.0 if "admin_notes" in df.columns else 0.05

    raw = (0.25 * ns + 0.20 * ts + 0.20 * os_ + 0.15 * ds
           + 0.15 * glucose_clean + drop_bonus)
    return round(min(raw, 1.0), 4)


TASK_3 = Task(
    task_id="task_3",
    description=(
        "Healthcare patient records (800 rows). "
        "Mixed corruption profiles: 25% null ages, physiologically impossible weight outliers, "
        "glucose stored as mixed strings/numbers (type_chaos), 12% null cholesterol with outliers, "
        "8% null systolic BP, 60 duplicates, irrelevant 'admin_notes' column. "
        "Agent must diagnose each column independently and apply different strategies. "
        "This task is designed to challenge GPT-4-class models."
    ),
    max_steps=20,
    generate=_generate_task3,
    grade=_grade_task3,
    canonical_column_names={},
    irrelevant_columns=["admin_notes"],
)


# ── Registry ──────────────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, Task] = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}
