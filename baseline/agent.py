"""
DataClean-Env — GPT-4o-mini Baseline Agent
Connects to the running server and runs a full cleaning episode per task.

Usage
-----
# Against local server:
python baseline/agent.py

# Against HuggingFace Space:
python baseline/agent.py --url https://huggingface.co/spaces/USERNAME/dataclean-env

# Single task:
python baseline/agent.py --task task_3 --seed 42

Environment variables
---------------------
OPENAI_API_KEY   required
DATACLEAN_URL    optional (overrides --url)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import Any

import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_URL  = "http://localhost:7860"
MODEL        = "gpt-4o-mini"
MAX_STEPS    = 20
SEED         = 42
TEMPERATURE  = 0.2   # low temp = more consistent, reproducible scores

SYSTEM_PROMPT = """You are an expert data cleaning agent operating inside DataClean-Env.

Your goal is to clean a pandas DataFrame by choosing the best sequence of operations.
You will receive an observation describing the current state of the dataframe.

RULES:
1. Analyse the column profiles carefully — null_rate, dtype, corruption_flags, and sample_values.
2. Fix the most impactful issues first: duplicates → type_chaos → heavy_nulls → outliers → minor nulls.
3. DO NOT apply an operation to a column that is already clean (null_rate < 0.01, correct dtype, no outliers).
   Applying operations to clean columns gives a negative reward.
4. Set confidence honestly:
   - Use 0.85–0.95 when you are certain the operation will help.
   - Use 0.50–0.70 when you are unsure (e.g. ambiguous column name, borderline outliers).
   - High confidence on a WRONG action gives a −0.06 penalty. Calibration matters.
5. Call "done" when quality_scores.overall ≥ 0.90 or when you have fixed all flagged columns.
6. You have a step budget — do not waste steps on already-clean columns.

CONFIDENCE CALIBRATION (important for your score):
- confidence ≥ 0.75 on correct action → +0.04 bonus
- confidence ≥ 0.75 on wrong action  → −0.06 penalty
- Be honest about uncertainty.

Respond ONLY with valid JSON — no markdown, no explanation, no code fences.
Schema:
{
  "action_type": "fill_nulls | remove_duplicates | fix_dtype | clip_outliers | rename_column | drop_column | done",
  "column": "<column name or null>",
  "params": {},
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence explaining why>"
}"""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def api(client: httpx.Client, method: str, path: str, **kwargs) -> dict[str, Any]:
    url = client.base_url.copy_with(path=path)
    resp = client.request(method, str(url), timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_episode(
    http: httpx.Client,
    openai_client: OpenAI,
    task_id: str,
    seed: int = SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one full episode. Returns episode summary dict."""

    # Reset
    reset_data = api(http, "POST", "/reset", json={"task_id": task_id, "seed": seed})
    episode_id = reset_data["episode_id"]
    obs        = reset_data["observation"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} | Episode: {episode_id[:8]}... | Seed: {seed}")
        print(f"Shape: {obs['n_rows']} rows × {obs['n_cols']} cols")
        print(f"Initial quality: {obs['quality_scores']}")
        print(f"{'='*60}")

    total_reward = 0.0
    history: list[dict] = []   # conversation history for context

    for step in range(MAX_STEPS):
        # Build prompt from current observation
        obs_text = _obs_to_prompt(obs)

        # Add to conversation history (keep last 6 turns to stay within context)
        history.append({"role": "user", "content": obs_text})
        if len(history) > 12:
            history = history[-12:]

        # Call GPT-4o-mini
        try:
            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=256,
            )
            raw = response.choices[0].message.content
            action_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  [step {step}] JSON parse error: {exc} — raw: {raw[:100]}")
            action_dict = {"action_type": "done", "confidence": 0.5, "reasoning": "parse error"}

        # Strip reasoning before sending to env (not part of action schema)
        reasoning = action_dict.pop("reasoning", "")
        # Clamp confidence
        action_dict["confidence"] = max(0.0, min(1.0, float(action_dict.get("confidence", 0.5))))

        if verbose:
            print(f"\n  Step {step+1:02d} | {action_dict.get('action_type'):20s} "
                  f"col={str(action_dict.get('column')):15s} "
                  f"conf={action_dict.get('confidence', 0):.2f}")
            print(f"           reasoning: {reasoning}")

        # Send action to server
        step_data = api(http, "POST", "/step", json={
            "episode_id": episode_id,
            "action": action_dict,
        })

        reward   = step_data["reward"]
        done     = step_data["done"]
        obs      = step_data["observation"]
        breakdown = step_data["reward_breakdown"]
        total_reward += reward

        if verbose:
            print(f"           reward={reward:+.4f} {breakdown}")
            print(f"           feedback: {obs['last_action_result']}")
            print(f"           quality: {obs['quality_scores']}")

        # Add assistant turn to history
        history.append({"role": "assistant", "content": raw})

        if done:
            break

        # Early exit if fully clean
        if obs["quality_scores"].get("overall", 0) >= 0.99:
            # One final done call
            api(http, "POST", "/step", json={
                "episode_id": episode_id,
                "action": {"action_type": "done", "confidence": 0.9},
            })
            if verbose:
                print(f"\n  Early exit — quality 1.0 reached at step {step+1}")
            break

    # Final grade
    grade_data = api(http, "POST", "/grader", json={"episode_id": episode_id})
    state_data = api(http, "GET",  f"/state?episode_id={episode_id}")

    summary = {
        "task_id":       task_id,
        "episode_id":    episode_id,
        "seed":          seed,
        "total_reward":  round(total_reward, 4),
        "final_score":   grade_data["score"],
        "quality_scores": grade_data["quality_scores"],
        "provenance_ok": grade_data["provenance_reproducible"],
        "steps_used":    state_data["step"],
        "model":         MODEL,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"EPISODE COMPLETE")
        print(f"  Final score:   {summary['final_score']:.4f}")
        print(f"  Total reward:  {summary['total_reward']:.4f}")
        print(f"  Steps used:    {summary['steps_used']}")
        print(f"  Provenance:    {summary['provenance_ok']}")
        print(f"  Quality:       {summary['quality_scores']}")
        print(f"{'─'*60}")

    return summary


def _obs_to_prompt(obs: dict) -> str:
    """Render observation dict as a concise LLM prompt."""
    col_lines = []
    for c in obs["columns"]:
        flags = ", ".join(c["corruption_flags"]) if c["corruption_flags"] else "clean"
        stats = ""
        if c["mean"] is not None:
            stats = f" mean={c['mean']:.2f} std={c['std']:.2f} min={c['min']:.2f} max={c['max']:.2f}"
        col_lines.append(
            f"  {c['name']:25s} dtype={c['dtype']:8s} nulls={c['null_rate']:.1%}"
            f" unique={c['n_unique']:4d}{stats} flags=[{flags}]"
        )

    scores = obs.get("quality_scores", {})
    overall = scores.get("overall", 0)
    budget  = obs.get("budget_remaining", "?")

    return (
        f"Step {obs['step']} | Budget remaining: {budget} | "
        f"Rows: {obs['n_rows']} | Dup rate: {obs['duplicate_rate']:.1%}\n"
        f"Quality: overall={overall:.3f} | "
        + " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if k != "overall")
        + f"\n\nColumn profiles:\n" + "\n".join(col_lines)
        + f"\n\nLast action: {obs.get('last_action_result', 'N/A')}"
        + f"\nOps applied: {len(obs.get('ops_log', []))}"
        + "\n\nChoose your next action:"
    )


# ── Nemotron-compatible wrapper ───────────────────────────────────────────────

class NemotronAgentWrapper:
    """
    Wraps the GPT-4o-mini agent in the interface Nemotron's evaluator expects.
    The OpenEnv Phase 2 judge uses this wrapper to standardise evaluation.

    Interface contract:
      agent.reset(task_id, seed) -> observation_dict
      agent.step(observation)    -> action_dict
      agent.score()              -> float
    """

    def __init__(self, server_url: str = DEFAULT_URL):
        self.server_url  = server_url.rstrip("/")
        self._http       = httpx.Client(base_url=self.server_url)
        self._openai     = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self._episode_id: str | None = None
        self._history: list[dict] = []
        self._total_reward = 0.0

    def reset(self, task_id: str = "task_1", seed: int = 42) -> dict:
        data = api(self._http, "POST", "/reset", json={"task_id": task_id, "seed": seed})
        self._episode_id  = data["episode_id"]
        self._history     = []
        self._total_reward = 0.0
        return data["observation"]

    def step(self, observation: dict) -> dict:
        obs_text = _obs_to_prompt(observation)
        self._history.append({"role": "user", "content": obs_text})
        if len(self._history) > 12:
            self._history = self._history[-12:]

        response = self._openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self._history],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        raw = response.choices[0].message.content
        action_dict = json.loads(raw)
        action_dict.pop("reasoning", None)
        action_dict["confidence"] = max(0.0, min(1.0, float(action_dict.get("confidence", 0.5))))

        step_data = api(self._http, "POST", "/step", json={
            "episode_id": self._episode_id,
            "action": action_dict,
        })
        self._total_reward += step_data["reward"]
        self._history.append({"role": "assistant", "content": raw})
        return action_dict

    def score(self) -> float:
        if not self._episode_id:
            return 0.0
        data = api(self._http, "POST", "/grader", json={"episode_id": self._episode_id})
        return data["score"]

    def close(self):
        self._http.close()


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DataClean-Env GPT-4o-mini baseline agent")
    parser.add_argument("--url",   default=os.environ.get("DATACLEAN_URL", DEFAULT_URL))
    parser.add_argument("--task",  default=None, help="Run a single task (task_1/2/3). Default: all.")
    parser.add_argument("--seed",  type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    http_client    = httpx.Client(base_url=args.url)
    openai_client  = OpenAI(api_key=api_key)

    # Check server health
    try:
        health = api(http_client, "GET", "/health")
        print(f"Server: {health['environment']} v{health['version']} @ {args.url}")
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    task_ids = [args.task] if args.task else ["task_1", "task_2", "task_3"]
    all_results = []

    for task_id in task_ids:
        result = run_episode(
            http_client,
            openai_client,
            task_id=task_id,
            seed=args.seed,
            verbose=not args.quiet,
        )
        all_results.append(result)
        time.sleep(0.5)   # brief pause between tasks

    # Summary table
    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY  (model={MODEL}, seed={args.seed})")
    print(f"{'─'*60}")
    print(f"{'Task':<10} {'Score':>8} {'Reward':>10} {'Steps':>7} {'Provenance':>12}")
    print(f"{'─'*60}")
    for r in all_results:
        prov = "✓" if r["provenance_ok"] else "✗"
        print(f"{r['task_id']:<10} {r['final_score']:>8.4f} {r['total_reward']:>10.4f} "
              f"{r['steps_used']:>7d} {prov:>12}")
    avg = sum(r["final_score"] for r in all_results) / len(all_results)
    print(f"{'─'*60}")
    print(f"{'Average':<10} {avg:>8.4f}")
    print(f"{'='*60}")

    # Write results to JSON for README
    out_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "seed": args.seed, "results": all_results}, f, indent=2)
    print(f"\nScores written to {out_path}")


if __name__ == "__main__":
    main()
