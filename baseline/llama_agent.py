"""
DataClean-Env — Open-Source Baseline Agent
Uses Meta's Llama-3-70B-Instruct via HuggingFace Inference API.

Why Llama-3-70B?
  - Meta's flagship open-source model (hackathon co-organiser)
  - Runs on HuggingFace Inference API (hackathon co-organiser)
  - Proves DataClean-Env is fully model-agnostic

Usage
-----
# Against local server:
python baseline/llama_agent.py

# Against HuggingFace Space:
python baseline/llama_agent.py --url https://huggingface.co/spaces/USERNAME/dataclean-env

Environment variables
---------------------
HF_TOKEN        required  — HuggingFace token (read access is enough)
DATACLEAN_URL   optional  — overrides --url
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_URL   = "http://localhost:7860"
HF_MODEL      = "meta-llama/Meta-Llama-3-70B-Instruct"
HF_API_BASE   = "https://api-inference.huggingface.co/models"
MAX_STEPS     = 20
SEED          = 42
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """You are an expert data cleaning agent operating inside DataClean-Env.

Your goal: clean a pandas DataFrame through structured operations.

STRATEGY (follow this order):
1. Remove duplicates first — always
2. Fix type_chaos columns (numeric data stored as strings) before filling nulls
3. Fill nulls — heaviest columns first (highest null_rate)
4. Clip outliers — columns with heavy_outliers flag
5. Drop irrelevant columns if you see them in the observation
6. Call "done" when overall quality >= 0.90 or all flags are resolved

CONFIDENCE RULES (this affects your score):
- Set 0.85-0.95 when you are certain the action will help
- Set 0.50-0.70 when unsure
- High confidence (>=0.75) on a WRONG action = -0.06 penalty
- High confidence on a CORRECT action = +0.04 bonus
- Be honest about uncertainty

OUTPUT: Respond ONLY with a JSON object. No markdown, no explanation.
{
  "action_type": "fill_nulls | remove_duplicates | fix_dtype | clip_outliers | rename_column | drop_column | done",
  "column": "<column name or null>",
  "params": {},
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}"""


# ── HuggingFace Inference API client ─────────────────────────────────────────

class LlamaClient:
    """
    Thin wrapper around HuggingFace Inference API for Llama-3-70B-Instruct.
    Uses the /v1/chat/completions compatible endpoint.
    """

    def __init__(self, hf_token: str, model: str = HF_MODEL):
        self.model   = model
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type":  "application/json",
        }
        # HF serverless inference — chat completions endpoint
        self.url = f"https://api-inference.huggingface.co/v1/chat/completions"

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """Call Llama-3-70B and return the assistant message string."""
        payload = {
            "model":      self.model,
            "messages":   messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": temperature,
            "stream":     False,
        }
        resp = httpx.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        if resp.status_code == 503:
            # Model is loading — wait and retry once
            print("  [Llama] Model loading, waiting 20s...")
            time.sleep(20)
            resp = httpx.post(self.url, headers=self.headers, json=payload, timeout=90)

        resp.raise_for_status()
        data = resp.json()

        # Extract content from OpenAI-compatible response
        return data["choices"][0]["message"]["content"]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def api(client: httpx.Client, method: str, path: str, **kwargs) -> dict[str, Any]:
    url  = client.base_url.copy_with(path=path)
    resp = client.request(method, str(url), timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _obs_to_prompt(obs: dict) -> str:
    """Render observation dict as a concise prompt string."""
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

    scores  = obs.get("quality_scores", {})
    overall = scores.get("overall", 0)
    budget  = obs.get("budget_remaining", "?")

    return (
        f"Step {obs['step']} | Budget: {budget} | "
        f"Rows: {obs['n_rows']} | Dup rate: {obs['duplicate_rate']:.1%}\n"
        f"Quality: overall={overall:.3f} | "
        + " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if k != "overall")
        + "\n\nColumn profiles:\n" + "\n".join(col_lines)
        + f"\n\nLast action: {obs.get('last_action_result', 'N/A')}"
        + f"\nOps applied: {len(obs.get('ops_log', []))}"
        + "\n\nChoose your next action (JSON only):"
    )


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_episode(
    http: httpx.Client,
    llama: LlamaClient,
    task_id: str,
    seed: int = SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one full episode with Llama-3-70B. Returns summary dict."""

    # Reset
    reset_data = api(http, "POST", "/reset", json={"task_id": task_id, "seed": seed})
    episode_id = reset_data["episode_id"]
    obs        = reset_data["observation"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Llama-3-70B] Task: {task_id} | Episode: {episode_id[:8]}...")
        print(f"Shape: {obs['n_rows']} rows × {obs['n_cols']} cols")
        print(f"Initial quality: {obs['quality_scores']}")
        print(f"{'='*60}")

    total_reward = 0.0
    history: list[dict] = []

    for step in range(MAX_STEPS):
        obs_text = _obs_to_prompt(obs)
        history.append({"role": "user", "content": obs_text})
        if len(history) > 12:
            history = history[-12:]

        # Call Llama-3-70B
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
            ]
            raw = llama.chat(messages, temperature=0.1)

            # Strip any markdown fences Llama might add
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("```")[1]
                if raw_clean.startswith("json"):
                    raw_clean = raw_clean[4:]
                raw_clean = raw_clean.strip()

            action_dict = json.loads(raw_clean)

        except json.JSONDecodeError as exc:
            if verbose:
                print(f"  [step {step}] JSON parse error — raw: {raw[:120]}")
            action_dict = {"action_type": "done", "confidence": 0.5, "reasoning": "parse error"}

        except Exception as exc:
            if verbose:
                print(f"  [step {step}] API error: {exc}")
            action_dict = {"action_type": "done", "confidence": 0.5, "reasoning": str(exc)}

        # Strip reasoning before sending to env
        reasoning = action_dict.pop("reasoning", "")
        action_dict["confidence"] = max(0.0, min(1.0, float(action_dict.get("confidence", 0.5))))

        if verbose:
            print(f"\n  Step {step+1:02d} | {action_dict.get('action_type'):20s} "
                  f"col={str(action_dict.get('column')):15s} "
                  f"conf={action_dict.get('confidence', 0):.2f}")
            print(f"           reasoning: {reasoning}")

        # Send to server
        try:
            step_data = api(http, "POST", "/step", json={
                "episode_id": episode_id,
                "action": action_dict,
            })
        except Exception as exc:
            if verbose:
                print(f"  [step {step}] Server error: {exc}")
            break

        reward    = step_data["reward"]
        done      = step_data["done"]
        obs       = step_data["observation"]
        breakdown = step_data["reward_breakdown"]
        total_reward += reward

        if verbose:
            print(f"           reward={reward:+.4f} | {breakdown}")
            print(f"           quality: {obs['quality_scores']}")

        history.append({"role": "assistant", "content": raw if 'raw' in dir() else json.dumps(action_dict)})

        if done:
            break

        # Early exit if fully clean
        if obs["quality_scores"].get("overall", 0) >= 0.99:
            api(http, "POST", "/step", json={
                "episode_id": episode_id,
                "action": {"action_type": "done", "confidence": 0.92},
            })
            if verbose:
                print(f"\n  Early exit — quality 1.0 at step {step+1}")
            break

    # Final grade
    grade_data = api(http, "POST", "/grader", json={"episode_id": episode_id})
    state_data = api(http, "GET",  f"/state?episode_id={episode_id}")

    summary = {
        "task_id":        task_id,
        "episode_id":     episode_id,
        "seed":           seed,
        "total_reward":   round(total_reward, 4),
        "final_score":    grade_data["score"],
        "quality_scores": grade_data["quality_scores"],
        "provenance_ok":  grade_data["provenance_reproducible"],
        "steps_used":     state_data["step"],
        "model":          HF_MODEL,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"EPISODE COMPLETE  [{HF_MODEL.split('/')[-1]}]")
        print(f"  Final score:  {summary['final_score']:.4f}")
        print(f"  Total reward: {summary['total_reward']:.4f}")
        print(f"  Steps used:   {summary['steps_used']}")
        print(f"  Provenance:   {summary['provenance_ok']}")
        print(f"{'─'*60}")

    return summary


# ── Nemotron-compatible wrapper (Llama variant) ───────────────────────────────

class LlamaNemotronWrapper:
    """
    Nemotron-compatible wrapper using Llama-3-70B-Instruct.
    Same 3-method interface as NemotronAgentWrapper in agent.py —
    drop-in replacement for Phase 2 judge evaluation.

    Usage:
        agent = LlamaNemotronWrapper(server_url="...", hf_token="hf_...")
        obs   = agent.reset("task_1", seed=42)
        while True:
            action = agent.step(obs)
            break
        score = agent.score()
        agent.close()
    """

    def __init__(self, server_url: str = DEFAULT_URL, hf_token: str | None = None):
        self.server_url   = server_url.rstrip("/")
        self._http        = httpx.Client(base_url=self.server_url)
        self._llama       = LlamaClient(hf_token or os.environ["HF_TOKEN"])
        self._episode_id: str | None = None
        self._history:    list[dict] = []
        self._total_reward = 0.0

    def reset(self, task_id: str = "task_1", seed: int = 42) -> dict:
        data = api(self._http, "POST", "/reset", json={"task_id": task_id, "seed": seed})
        self._episode_id   = data["episode_id"]
        self._history      = []
        self._total_reward = 0.0
        return data["observation"]

    def step(self, observation: dict) -> dict:
        obs_text = _obs_to_prompt(observation)
        self._history.append({"role": "user", "content": obs_text})
        if len(self._history) > 12:
            self._history = self._history[-12:]

        raw = self._llama.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, *self._history],
            temperature=0.1,
        )

        # Clean up any markdown fences
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("```")[1]
            if raw_clean.startswith("json"):
                raw_clean = raw_clean[4:]
            raw_clean = raw_clean.strip()

        action_dict = json.loads(raw_clean)
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
    parser = argparse.ArgumentParser(
        description=f"DataClean-Env — Llama-3-70B-Instruct baseline agent"
    )
    parser.add_argument("--url",   default=os.environ.get("DATACLEAN_URL", DEFAULT_URL))
    parser.add_argument("--task",  default=None, help="Single task (task_1/2/3). Default: all.")
    parser.add_argument("--seed",  type=int, default=SEED)
    parser.add_argument("--model", default=HF_MODEL, help="HF model ID to use")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("Get your token at: https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    http_client = httpx.Client(base_url=args.url)
    llama       = LlamaClient(hf_token=hf_token, model=args.model)

    # Server health check
    try:
        health = api(http_client, "GET", "/health")
        print(f"Server : {health['environment']} v{health['version']} @ {args.url}")
        print(f"Model  : {args.model}")
        print(f"Seed   : {args.seed}")
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    task_ids    = [args.task] if args.task else ["task_1", "task_2", "task_3"]
    all_results = []

    for task_id in task_ids:
        result = run_episode(
            http_client, llama,
            task_id=task_id,
            seed=args.seed,
            verbose=not args.quiet,
        )
        all_results.append(result)
        time.sleep(1.0)

    # Summary table
    print(f"\n{'='*60}")
    print(f"LLAMA-3-70B BASELINE  (seed={args.seed})")
    print(f"Model: {args.model}")
    print(f"{'─'*60}")
    print(f"{'Task':<10} {'Score':>8} {'Reward':>10} {'Steps':>7} {'Provenance':>12}")
    print(f"{'─'*60}")
    for r in all_results:
        prov = "✓" if r["provenance_ok"] else "✗"
        print(f"{r['task_id']:<10} {r['final_score']:>8.4f} "
              f"{r['total_reward']:>10.4f} {r['steps_used']:>7d} {prov:>12}")

    avg = sum(r["final_score"] for r in all_results) / len(all_results)
    print(f"{'─'*60}")
    print(f"{'Average':<10} {avg:>8.4f}")
    print(f"{'='*60}")

    # Write to JSON
    out_path = os.path.join(os.path.dirname(__file__), "llama_baseline_scores.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":   args.model,
            "seed":    args.seed,
            "results": all_results,
        }, f, indent=2)
    print(f"\nScores written to {out_path}")
    print("\nTo add to README:")
    print(f"  Model: {args.model.split('/')[-1]}")
    for r in all_results:
        print(f"  {r['task_id']}: score={r['final_score']:.4f}  reward={r['total_reward']:.4f}")


if __name__ == "__main__":
    main()
