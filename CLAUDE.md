# Gravity RL Environment — Developer Guide

## What this project is

A coding task for AI agents, packaged as a Harbor task. The agent implements a 2D
N-body gravity simulator in `/app/simulator.py`. A hardened verifier scores it on
randomized hidden scenarios and writes a binary reward (1 = pass, 0 = fail).

The verifier is the core of the project. Treat any change that makes it easier to pass
without correct physics as a bug. See `HARDENING.md` for the reward hacks it defends
against.

---

## Directory layout

```
gravity-rl-env/
├── tasks/gravity-simulator/       Harbor task (everything Harbor needs)
│   ├── instruction.md             Spec shown to the agent
│   ├── task.toml                  Timeouts (agent 900s, verifier 600s) and metadata
│   ├── environment/Dockerfile     Agent image: python:3.12-slim, WORKDIR /app
│   ├── solution/solve.sh          Reference solution; `bash solve.sh <path>` writes it anywhere
│   └── tests/
│       ├── test.sh                Harbor entrypoint; writes /logs/verifier/reward.txt
│       ├── verifier.py            Scenario generation, reference physics, scoring
│       └── worker.py              Runs agent code; reports raw observations only
├── dev/
│   ├── check_verifier.py          Scores reference, alt-correct and 3 hacks on old and new verifier
│   ├── hacks/                     explicit_euler.py, cached_energy.py, verifier_hijack.py
│   ├── alt_correct_simulator.py   Independent correct solution
│   └── old_verifier.py            v1 verifier (insecure; kept only for comparison)
├── gym/gravity_env.py             Gym-style wrapper (demo only, imports root simulator.py)
├── simulator.py                   Simulator for the gym demo; not visible to the agent
├── simulator_visual.html          Browser visualiser
├── HARDENING.md                   Hacks found, fixes, evidence
└── README.md                      GitHub-facing overview
```

---

## Commands

```bash
# Must end with "NEW VERIFIER BEHAVES AS EXPECTED". Run after ANY verifier change.
python3 dev/check_verifier.py

# Full report for one submission; VERIFIER_SEED makes the run reproducible
VERIFIER_SEED=7 python3 -I tasks/gravity-simulator/tests/verifier.py simulator.py

# End-to-end in Docker (Docker Desktop must be running)
harbor run -p tasks/gravity-simulator -a oracle
harbor view jobs
```

---

## How the verifier works

1. `verifier.py` draws random scenarios for T1–T12 from a seed.
2. For each test it starts `python3 -I worker.py /app/simulator.py` with a 20s timeout
   and sends the scenario on stdin.
3. `worker.py` silences the agent's stdout/stderr, runs the scenario, and returns raw
   observations (states, return values, exception names) on a private file descriptor,
   tagged with a per-run nonce.
4. `verifier.py` scores those observations against its own reference integrator.
5. Pass = score ≥ 122/125 **and** full marks on gating tests T6, T8, T10.

### Rules to keep when editing

- **Never import or exec agent code in `verifier.py`.** Agent code runs only in `worker.py`.
- **Keep `worker.py` free of reference physics and scoring.** Anything there is
  available to agent code at runtime.
- **Compute expected values in the parent from scenario parameters**, never from the
  agent's own methods. T10 computes energy from the returned state for this reason.
- **Randomize new scenarios** with the `rng` passed to `make_scenarios`. No fixed
  expected constants.
- **Check integrator behaviour on displacements at large dt.** At small dt the
  semi-implicit vs explicit Euler difference falls below any useful tolerance.
- **After any change**, run `dev/check_verifier.py`, and also check the reference and
  `dev/alt_correct_simulator.py` over many seeds to catch false failures:
  ```bash
  for s in $(seq 1 40); do VERIFIER_SEED=$s python3 -I tasks/gravity-simulator/tests/verifier.py dev/alt_correct_simulator.py >/dev/null || echo "seed $s failed"; done
  ```
- **Adding a reward hack:** put it in `dev/hacks/`, add it to `SUBMISSIONS` in
  `dev/check_verifier.py`, and document it in `HARDENING.md`.

---

## Physics notes

**Force law:** `F_ij = G * m_i * m_j / r_ij²`, directed from i toward j. No softening;
coincident bodies raise `ZeroDivisionError`.

**Semi-implicit Euler, per step:**
```
1. Compute all accelerations from current positions
2. v_new = v_old + a * dt
3. x_new = x_old + v_new * dt   ← uses the NEW velocity
```
This order keeps energy bounded over long runs. At 300 steps per orbit, semi-implicit
Euler drifts < 0.3% over ~3 orbits; explicit Euler drifts > 20%. T10 relies on this gap.

**Orbital period (T11):** Kepler period of the relative two-body orbit:
`a = -μ / (2ε)`, `ε = v_rel²/2 − μ/r`, `μ = G(m1 + m2)`, `T = 2π·sqrt(a³/μ)`.

---

## Extending the gym demo

- **New scenarios:** add a branch in `GravityEnv._setup_scenario`.
- **Thrust actions:** override `GravityEnv.apply_action` to change body velocities
  before `_sim.step(1)`.
- **Reward shaping:** replace the energy-drift reward in `GravityEnv.step`.

The gym demo is separate from the Harbor task. Changing it does not affect scoring.

---

## Dependencies

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.11 (the agent's image uses 3.12) |
| Packages | none; the task and verifier use the standard library only |
| Docker + Harbor | only for `harbor run` |