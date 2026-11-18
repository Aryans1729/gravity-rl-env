"""Scores the reference, an alternative correct solution and three reward hacks
against the OLD (v1) and NEW (v2) verifiers, over several random seeds.

Run from the repo root:  python3 dev/check_verifier.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, "tasks", "gravity-simulator", "tests", "verifier.py")
OLD = os.path.join(ROOT, "dev", "old_verifier.py")
SEEDS = range(1, 11)

SOLVE = os.path.join(ROOT, "tasks", "gravity-simulator", "solution", "solve.sh")
REFERENCE = os.path.join(tempfile.mkdtemp(), "reference_simulator.py")

SUBMISSIONS = [
    ("reference (should PASS)", REFERENCE, True),
    ("alt correct (should PASS)", "dev/alt_correct_simulator.py", True),
    ("hack: explicit Euler", "dev/hacks/explicit_euler.py", False),
    ("hack: cached total_energy", "dev/hacks/cached_energy.py", False),
    ("hack: verifier hijack", "dev/hacks/verifier_hijack.py", False),
]


def old_verdict(path):
    out = subprocess.run([sys.executable, OLD, path], capture_output=True, text=True).stdout
    m = re.search(r"TOTAL SCORE\s*:\s*(\d+)/(\d+)", out)
    return ("PASSED" in out), (m.group(0).split(":")[1].strip() if m else "n/a")


def new_verdict(path, seed):
    env = dict(os.environ, VERIFIER_SEED=str(seed))
    out = subprocess.run([sys.executable, "-I", NEW, path], capture_output=True, text=True, env=env).stdout
    m = re.search(r"SCORE\s*:\s*(\d+/\d+)", out)
    return ("VERDICT : PASSED" in out), (m.group(1) if m else "n/a")


def main():
    subprocess.run(["bash", SOLVE, REFERENCE], check=True, capture_output=True)
    ok_all = True
    print(f"{'submission':30} {'old verifier':22} {'new verifier (10 seeds)':26}")
    print("-" * 80)
    for label, rel_path, should_pass in SUBMISSIONS:
        path = os.path.join(ROOT, rel_path)
        o_pass, o_score = old_verdict(path)
        runs = [new_verdict(path, s) for s in SEEDS]
        n_pass = sum(p for p, _ in runs)
        scores = sorted({s for _, s in runs})
        new_txt = f"{n_pass}/{len(runs)} pass, score {scores[0] if len(scores) == 1 else scores[0] + '..' + scores[-1]}"
        print(f"{label:30} {('PASS ' if o_pass else 'FAIL ') + o_score:22} {new_txt}")
        expected = len(runs) if should_pass else 0
        ok_all &= (n_pass == expected)
    print("-" * 80)
    print("NEW VERIFIER BEHAVES AS EXPECTED" if ok_all else "MISMATCH - investigate")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
