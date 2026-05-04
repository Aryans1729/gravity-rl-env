"""
Minimal Harbor-style task runner for the Gravity RL environment.

Reads harbor_task.yaml, spins up an agentic loop with the Anthropic API,
and reports pass/fail for each run.

Usage:
    python harbor_runner.py --runs 3 --model claude-opus-4-5
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent
INSTRUCTION = (HERE / "task" / "instruction.md").read_text()
VERIFIER    = HERE / "verifier" / "verifier.py"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Python coding agent competing in a coding RL benchmark.

    Your task is described in the user's first message.

    Use the `write_file` tool to write simulator.py.  After you submit,
    the verifier will run automatically and you will receive the full
    output.  If you score below 70/100, fix every failing test and
    resubmit.  You have up to 8 attempts.

    Rules:
    - Only use Python's standard library (math is fine; no scipy/astropy).
    - The class must be named GravitySimulator.
    - Semi-implicit Euler: update v first, then x using the NEW v.
    - Read verifier failure messages carefully before rewriting.
""")

TOOLS = [
    {
        "name": "write_file",
        "description": (
            "Write (or overwrite) simulator.py in the workspace. "
            "Call this to submit your implementation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Must be 'simulator.py'",
                },
                "content": {
                    "type": "string",
                    "description": "Full Python source of the file.",
                },
            },
            "required": ["filename", "content"],
        },
    }
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_verifier(sim_path: Path) -> tuple[int, bool, str]:
    """Return (score, passed, full_output)."""
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(sim_path)],
        capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    m = re.search(r"TOTAL SCORE\s*:\s*(\d+)/\d+", output)
    score = int(m.group(1)) if m else 0
    # Honour the threshold baked into the verifier rather than hardcoding one here
    passed = bool(re.search(r"VERDICT\s*:\s*PASSED", output))
    return score, passed, output


def extract_tool_uses(content):
    return [b for b in content if b.type == "tool_use"]


def extract_text(content):
    parts = [b.text for b in content if hasattr(b, "text")]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(model: str, run_index: int, verbose: bool = True,
             save_best: Path | None = None) -> dict:
    client   = anthropic.Anthropic()
    workdir  = Path(tempfile.mkdtemp(prefix=f"harbor_run{run_index}_"))
    sim_path = workdir / "simulator.py"

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'─'*60}")
    log(f"  RUN {run_index}  |  model: {model}")
    log(f"{'─'*60}")

    messages = [{"role": "user", "content": INSTRUCTION}]
    score    = 0
    passed   = False
    attempts = 0

    for attempt in range(1, 3):
        attempts = attempt
        log(f"\n  [Attempt {attempt}/2] Calling {model}...")

        t0 = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            log(f"  API ERROR: {e}")
            break

        elapsed = time.time() - t0
        log(f"  Response in {elapsed:.1f}s  |  stop_reason={response.stop_reason}")

        # ---- collect file writes ----------------------------------------
        tool_uses   = extract_tool_uses(response.content)
        file_written = False

        for tu in tool_uses:
            if tu.name == "write_file":
                filename = tu.input.get("filename", "")
                content  = tu.input.get("content", "")
                if "simulator" in filename:
                    sim_path.write_text(content)
                    file_written = True
                    log(f"  Wrote {filename}  ({len(content)} chars)")

        # Fallback: fish a python code block out of the text
        if not file_written:
            text = extract_text(response.content)
            m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
            if m:
                sim_path.write_text(m.group(1))
                file_written = True
                log(f"  Extracted code block from text ({len(m.group(1))} chars)")

        if not file_written:
            log("  No simulator.py produced this attempt.")
            # Add assistant turn so we can continue conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "You did not write simulator.py. Please use the write_file tool now.",
            })
            continue

        # ---- run verifier -----------------------------------------------
        score, passed, verifier_out = run_verifier(sim_path)
        verdict = "PASS" if passed else "FAIL"
        log(f"  Verifier: {score}/100  [{verdict}]")

        # Print condensed verifier summary
        for line in verifier_out.splitlines():
            if any(tag in line for tag in ("[PASS]", "[FAIL]", "[PART]", "TOTAL", "VERDICT", "Energy drift")):
                log(f"    {line.strip()}")

        if passed:
            log(f"\n  ✓ Passed on attempt {attempt}!")
            if save_best and not save_best.exists():
                save_best.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(sim_path, save_best)
                log(f"  Saved best solution → {save_best}")
            break

        if attempt == 2:
            log(f"\n  ✗ Exhausted all 2 attempts. Final score: {score}/100")
            break

        # ---- build reflection turn ---------------------------------------
        messages.append({"role": "assistant", "content": response.content})

        # Build a redacted verifier summary: show only summary rows, not
        # per-sub-test details.  This prevents the model from seeing the
        # exact wrong values it produced and forces it to reason from
        # first principles about what to fix.
        summary_lines = [
            line for line in verifier_out.splitlines()
            if any(tag in line for tag in
                   ("[PASS]", "[FAIL]", "[PART]", "TOTAL SCORE", "VERDICT", "====="))
        ]
        redacted_out = "\n".join(summary_lines)

        # tool_result blocks for every tool_use in this turn
        tool_results = []
        for tu in tool_uses:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": (
                    f"simulator.py written successfully.\n\n"
                    f"VERIFIER SUMMARY:\n{redacted_out}"
                ),
            })

        # If there were no tool uses, send verifier output as plain text
        if not tool_results:
            tool_results = [{
                "type": "text",
                "text": f"VERIFIER SUMMARY:\n{redacted_out}\n\nPlease fix the issues and resubmit with write_file.",
            }]

        messages.append({"role": "user", "content": tool_results})

    shutil.rmtree(workdir)

    return {
        "run":      run_index,
        "model":    model,
        "passed":   passed,
        "score":    score,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harbor-style task runner")
    parser.add_argument("--runs",      type=int,  default=3,                    help="Number of independent runs")
    parser.add_argument("--model",     type=str,  default="claude-opus-4-5",    help="Anthropic model ID")
    parser.add_argument("--quiet",     action="store_true",                      help="Suppress per-attempt logs")
    parser.add_argument("--save-best", type=str,  default=None,                 help="Copy first passing simulator.py to this path")
    args = parser.parse_args()

    save_best = Path(args.save_best) if args.save_best else None

    results = []
    for i in range(1, args.runs + 1):
        r = run_once(args.model, i, verbose=not args.quiet, save_best=save_best)
        results.append(r)

    # ---- final report -------------------------------------------------------
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n{'═'*60}")
    print(f"  HARBOR TASK RUNNER — FINAL REPORT")
    print(f"  Model  : {args.model}")
    print(f"  Task   : gravity-rl-simulator")
    print(f"{'═'*60}")
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  Run {r['run']}:  {status}  |  score={r['score']}/100  |  attempts={r['attempts']}")
    print(f"{'─'*60}")
    print(f"  Pass rate: {passed_count}/{args.runs}  ({100*passed_count//args.runs}%)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
