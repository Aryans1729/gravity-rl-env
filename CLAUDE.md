# Gravity RL Environment — Project Guide

## What this project is

A self-contained **coding RL task** designed for the Harbor framework with the
Terminus 2 agent.  The agent is given a physics problem (implement a 2D N-body
gravity simulator), writes `simulator.py`, and is scored 0–100 by an automated
verifier.  The episode succeeds when the score reaches ≥ 70.

---

## Directory layout

```
gravity-rl-env/
├── gym/
│   └── gravity_env.py          OpenAI-gym-style wrapper around GravitySimulator
├── task/
│   └── instruction.md          The prompt shown to the agent
├── verifier/
│   └── verifier.py             Automated scorer (10 tests, 100 pts total)
├── agent/
│   └── terminus2_config.yaml   Terminus 2 / Harbor agent settings
├── harbor_task.yaml            Top-level Harbor task definition
├── simulator.py                Reference solution (scores 100/100)
├── simulator_visual.html       Interactive browser-based N-body visualiser
├── CLAUDE.md                   This file
└── README.md                   GitHub-facing overview
```

---

## Key files explained

### `task/instruction.md`
The complete specification given to the agent.  It covers:
- The exact Python interface the agent must implement
- The physics (Newton's law of gravitation in 2D)
- The semi-implicit Euler integration order (forces → velocities → positions)
- All required error conditions
- The scoring breakdown

### `verifier/verifier.py`
Runs 10 tests and prints a detailed report.  Usage:

```bash
python verifier/verifier.py simulator.py
```

Tests (abridged):

| ID  | Description                         | Points |
|-----|-------------------------------------|--------|
| T1  | API surface present                 | 5      |
| T2  | Insertion order preserved           | 10     |
| T3  | ValueError on 6th body              | 5      |
| T4  | ValueError with < 2 bodies          | 5      |
| T5  | ZeroDivisionError on collision      | 5      |
| T6  | Single-step Sun–Earth accuracy      | 25     |
| T7  | 3-body symmetric force              | 15     |
| T8  | total_energy() at t = 0             | 10     |
| T9  | reset() clears state                | 5      |
| T10 | 1 000-step energy stability         | 10     |

### `simulator.py`
Reference implementation that scores 100/100.  When the agent rewrites this
file, the verifier is run against the agent's version.  The reference copy is
committed to the repo so the environment can always be validated.

### `gym/gravity_env.py`
A gym-compatible wrapper exposing `reset()`, `step()`, and `render()`.  Reward
is the negative fractional energy drift per step (0 = perfect conservation).
Built-in scenarios: `earth_sun`, `earth_sun_jupiter`, `figure8`, `random_3body`.

### `agent/terminus2_config.yaml`
Configures the Terminus 2 agent: model (claude-opus-4-7), temperature,
verification loop, allowed tools, and reflection strategy for retries.

### `harbor_task.yaml`
Top-level Harbor task file.  Defines the environment, agent pointer, scoring
rubric, and execution policy (Docker sandbox, 300 s wall time, 8 attempts max).

---

## Running locally

```bash
# Score the reference solution
python verifier/verifier.py simulator.py

# Quick smoke-test of the gym wrapper
python - <<'EOF'
from gym.gravity_env import GravityEnv
env = GravityEnv(scenario="earth_sun", dt=10.0, max_steps=10)
obs, info = env.reset()
print("Initial energy:", info["energy"])
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  step={info['step']}  drift={info['energy_drift']:.2e}")
print(env.render())
EOF

# Open the visual simulator
open simulator_visual.html
```

---

## Physics notes

**Force law**

```
F_ij = G * m_i * m_j / r_ij²
```

Direction: unit vector from i toward j.  Newton's third law applied via Newton's
third law (body j gets equal and opposite force).

**Semi-implicit Euler**

```
1. Compute all accelerations from current positions
2. v_new = v_old + a * dt
3. x_new = x_old + v_new * dt   ← uses NEW velocity
```

Step 3 is the distinguishing feature.  It provides better energy behaviour than
explicit Euler and is required to pass T10.

---

## Extending the environment

- **New scenarios**: add a branch in `GravityEnv._setup_scenario`.
- **Thrust actions**: override `GravityEnv.apply_action` to modify `_sim` body
  velocities before `_sim.step(1)` is called.
- **More bodies**: raise the `add_body` limit in `simulator.py` and the
  observation-space dimensionality in `_get_obs`.
- **Reward shaping**: replace the energy-drift reward in `GravityEnv.step` with
  e.g. orbit-stability or target-proximity rewards.

---

## Dependencies

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.11  |
| numpy       | optional (not used in reference solution) |
| scipy       | not used / not allowed |
| gym / gymnasium | not required (gym wrapper is self-contained) |
