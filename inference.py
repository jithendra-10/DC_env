"""
inference.py — DataClean-Env Baseline Inference Script
=======================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM
                   (default: https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier
                   (default: meta-llama/Meta-Llama-3-70B-Instruct)
    HF_TOKEN       Your HuggingFace API key

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
    export HF_TOKEN=hf_...
    python inference.py

Runs all 3 tasks. Completes in < 20 minutes on vcpu=2, 8GB RAM.
Scores saved to baseline_scores.json.
"""

import os
import sys
import json
import time

from openai import OpenAI

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Required env variables (exactly as spec mandates) ────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.")
    print("Get your token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

# ── OpenAI client → HuggingFace router ───────────────────────────────────────
# Uses OpenAI client as required. API_BASE_URL points to HF router so
# Llama-3-70B-Instruct (Meta's model) is called through HF infrastructure.

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Config ────────────────────────────────────────────────────────────────────

SEED         = 42
MAX_STEPS    = 15        # conservative — keeps runtime well under 20 min
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
STEP_TIMEOUT = 45        # seconds per LLM call

SYSTEM_PROMPT = """You are an expert data cleaning agent inside DataClean-Env.

Your goal: clean a pandas DataFrame through structured operations.

STRATEGY (follow this order):
1. Remove duplicates first — always
2. Fix type_chaos columns (numeric data stored as strings)
3. Fill nulls — heaviest columns first (highest null_rate)
4. Clip outliers — columns with heavy_outliers flag
5. Drop irrelevant columns if present
6. Call "done" when overall quality >= 0.90

CONFIDENCE RULES:
- 0.85–0.95 when certain the action will help
- 0.50–0.70 when unsure
- High confidence (>=0.75) on WRONG action = -0.06 penalty
- High confidence on CORRECT action = +0.04 bonus

Respond ONLY with valid JSON. No markdown, no explanation, no code fences.
{
  "action_type": "fill_nulls | remove_duplicates | fix_dtype | clip_outliers | rename_column | drop_column | done",
  "column": "<column name or null>",
  "params": {},
  "confidence": <float 0.0-1.0>
}"""

# ── Import environment directly (self-contained, no server needed) ────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclean.env import DataCleanEnv
from dataclean.models import DataCleanAction
from dataclean.tasks import TASK_REGISTRY


# ── Observation → prompt ──────────────────────────────────────────────────────

def obs_to_prompt(obs) -> str:
    col_lines = []
    for c in obs.columns:
        flags = ", ".join(c.corruption_flags) if c.corruption_flags else "clean"
        stats = ""
        if c.mean is not None:
            stats = (f" mean={c.mean:.2f} std={c.std:.2f}"
                     f" min={c.min:.2f} max={c.max:.2f}")
        col_lines.append(
            f"  {c.name:25s} dtype={c.dtype:8s}"
            f" nulls={c.null_rate:.1%} unique={c.n_unique:4d}"
            f"{stats} flags=[{flags}]"
        )

    scores  = obs.quality_scores
    overall = scores.get("overall", 0)

    return (
        f"Step {obs.step} | Budget remaining: {obs.budget_remaining} | "
        f"Rows: {obs.n_rows} | Dup rate: {obs.duplicate_rate:.1%}\n"
        f"Quality: overall={overall:.3f} | "
        + " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if k != "overall")
        + "\n\nColumn profiles:\n" + "\n".join(col_lines)
        + f"\n\nLast action: {obs.last_action_result or 'N/A'}"
        + f"\nOps applied: {len(obs.ops_log)}"
        + "\n\nChoose your next action (JSON only):"
    )


# ── Safe JSON parser ──────────────────────────────────────────────────────────

def parse_action(raw: str) -> dict:
    text = raw.strip()

    # Strip markdown fences if model adds them
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # Extract first { ... } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, seed: int = SEED) -> dict:
    env = DataCleanEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    print(f"\n[DEBUG] [{task_id}] {obs.n_rows} rows × {obs.n_cols} cols")
    print(f"[DEBUG] Initial quality: {obs.quality_scores}")

    total_reward = 0.0
    history      = []
    rewards_list = []
    
    log_start(task=task_id, env="dataclean-env", model=MODEL_NAME)

    for step in range(MAX_STEPS):

        # Build prompt and conversation history
        obs_text = obs_to_prompt(obs)
        history.append({"role": "user", "content": obs_text})
        if len(history) > 10:
            history = history[-10:]

        # ── OpenAI client call (required by hackathon rules) ──────────────────
        raw = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=STEP_TIMEOUT,
            )
            raw = response.choices[0].message.content or ""

        except Exception as exc:
            print(f"[DEBUG] [step {step+1}] LLM error: {exc} — using done")
            raw = '{"action_type":"done","column":null,"params":{},"confidence":0.5}'

        # ── Parse action ──────────────────────────────────────────────────────
        try:
            action_dict = parse_action(raw)
        except (json.JSONDecodeError, ValueError):
            print(f"[DEBUG] [step {step+1}] Parse error: {raw[:80]!r} — using done")
            action_dict = {
                "action_type": "done", "column": None,
                "params": {},         "confidence": 0.5,
            }

        # Clamp confidence to valid range
        action_dict["confidence"] = max(
            0.0, min(1.0, float(action_dict.get("confidence", 0.5)))
        )

        print(f"[DEBUG] Step {step+1:02d} | {action_dict.get('action_type','?'):20s} "
              f"col={str(action_dict.get('column')):18s} "
              f"conf={action_dict['confidence']:.2f}")

        # ── Step environment ──────────────────────────────────────────────────
        step_error = None
        done = False
        step_reward = 0.0
        
        try:
            result = env.step(DataCleanAction(
                action_type=action_dict.get("action_type", "done"),
                column=action_dict.get("column"),
                params=action_dict.get("params", {}),
                confidence=action_dict["confidence"],
            ))
            step_reward = result.reward
            obs = result.observation
            done = result.done
        except Exception as exc:
            step_error = str(exc)
            done = True
            print(f"[DEBUG] Env error: {exc}")

        total_reward += step_reward
        rewards_list.append(step_reward)
        
        action_str = json.dumps(action_dict, separators=(',', ':'))
        log_step(step=step+1, action=action_str, reward=step_reward, done=done, error=step_error)

        print(f"[DEBUG]        reward={step_reward:+.4f} | "
              f"quality={obs.quality_scores.get('overall',0):.3f} | "
              f"{obs.last_action_result}")

        history.append({"role": "assistant", "content": raw})

        if done:
            break

        # Early exit when fully clean
        if obs.quality_scores.get("overall", 0) >= 0.99:
            try:
                result_final = env.step(DataCleanAction(action_type="done", confidence=0.9))
                rewards_list.append(result_final.reward)
                total_reward += result_final.reward
                log_step(step=step+2, action='{"action_type":"done","confidence":0.9}', reward=result_final.reward, done=True, error=None)
            except Exception:
                pass
            print("[DEBUG] Early exit — quality 1.0 reached")
            break

    # ── Grade ─────────────────────────────────────────────────────────────────
    final_score   = env.grade()
    provenance_ok = env.verify_provenance()
    quality       = env._compute_quality_scores()

    success = final_score >= 0.90
    log_end(success=success, steps=env._state.step, score=final_score, rewards=rewards_list)

    print(f"\n[DEBUG] Score: {final_score:.4f} | "
          f"Reward: {total_reward:.4f} | "
          f"Steps: {env._state.step} | "
          f"Provenance: {provenance_ok}")

    return {
        "task_id":        task_id,
        "seed":           seed,
        "model":          MODEL_NAME,
        "final_score":    round(final_score, 4),
        "total_reward":   round(total_reward, 4),
        "quality_scores": quality,
        "provenance_ok":  provenance_ok,
        "steps_used":     env._state.step,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[DEBUG] " + "=" * 60)
    print("[DEBUG] DataClean-Env  —  Baseline Inference")
    print(f"[DEBUG] Model    : {MODEL_NAME}")
    print(f"[DEBUG] Endpoint : {API_BASE_URL}")
    print(f"[DEBUG] Seed     : {SEED}")
    print("[DEBUG] " + "=" * 60)

    t_start  = time.time()
    results  = []

    for task_id in TASK_REGISTRY:
        print(f"\n[DEBUG] Running {task_id}...")
        result  = run_episode(task_id=task_id, seed=SEED)
        results.append(result)

        elapsed = time.time() - t_start
        print(f"[DEBUG] Elapsed: {elapsed:.0f}s")

        # Hard safety guard — stop if approaching 20-min wall
        if elapsed > 1050:
            print("\n[DEBUG] WARNING: Approaching 20-min runtime limit. Stopping early.")
            break

    # ── Summary table ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n[DEBUG] {'='*60}")
    print(f"[DEBUG] FINAL RESULTS  (model={MODEL_NAME.split('/')[-1]}, seed={SEED})")
    print(f"[DEBUG] {'─'*60}")
    print(f"[DEBUG] {'Task':<10} {'Score':>8} {'Reward':>10} {'Steps':>7} {'Provenance':>12}")
    print(f"[DEBUG] {'─'*60}")
    for r in results:
        prov = "✓" if r["provenance_ok"] else "✗"
        print(f"[DEBUG] {r['task_id']:<10} {r['final_score']:>8.4f} "
              f"{r['total_reward']:>10.4f} "
              f"{r['steps_used']:>7d} {prov:>12}")
    if results:
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"[DEBUG] {'─'*60}")
        print(f"[DEBUG] {'Average':<10} {avg:>8.4f}")
    print(f"[DEBUG] {'='*60}")
    print(f"[DEBUG] Total runtime : {total_elapsed:.1f}s")

    # ── Save scores ───────────────────────────────────────────────────────────
    output = {
        "model":             MODEL_NAME,
        "api_base_url":      API_BASE_URL,
        "seed":              SEED,
        "total_runtime_s":   round(total_elapsed, 1),
        "results":           results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nScores saved → baseline_scores.json")


if __name__ == "__main__":
    main()
