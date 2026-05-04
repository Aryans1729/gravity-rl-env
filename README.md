# Gravity RL Environment

A **coding reinforcement-learning benchmark** where an AI agent must implement a
physically accurate 2D N-body gravitational simulator in Python and iteratively
improve it until it passes an automated verifier.

Built for the [Harbor](https://github.com/harbor-ml/harbor) framework with the
[Terminus 2](https://github.com/harbor-ml/terminus) agent.

---

## Overview

```
Agent receives instruction.md
          ↓
Agent writes simulator.py
          ↓
Verifier scores it (0–100)
          ↓
Score ≥ 70?  →  Episode success
Score < 70?  →  Agent reflects on errors and retries (up to 8×)
```

The verifier runs 10 tests covering API correctness, numerical accuracy of the
semi-implicit Euler integrator, and long-run energy stability.

---

## Quick start

```bash
# Clone
git clone <repo-url> gravity-rl-env
cd gravity-rl-env

# Score the reference solution
python verifier/verifier.py simulator.py

# Run the gym environment smoke-test (stdlib only, no pip needed)
python - <<'EOF'
from gym.gravity_env import GravityEnv
env = GravityEnv(scenario="earth_sun", dt=10.0, max_steps=100)
obs, info = env.reset()
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break
print(env.render())
print(f"Final energy drift: {info['energy_drift']*100:.4f}%")
EOF

# Open the interactive visual simulator
open simulator_visual.html     # macOS
# xdg-open simulator_visual.html  # Linux
```

No pip installs required — the entire project uses the Python standard library.

---

## Project structure

```
gravity-rl-env/
├── gym/
│   └── gravity_env.py          Gym-style RL wrapper
├── task/
│   └── instruction.md          Agent prompt / task specification
├── verifier/
│   └── verifier.py             Automated scorer
├── agent/
│   └── terminus2_config.yaml   Harbor + Terminus 2 agent settings
├── harbor_task.yaml            Harbor task definition
├── simulator.py                Reference solution (100/100)
├── simulator_visual.html       Interactive browser visualiser
├── CLAUDE.md                   Developer guide
└── README.md                   This file
```

---

## The task

The agent must implement `GravitySimulator` with this interface:

```python
class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0): ...
    def add_body(self, name, mass, x, y, vx=0.0, vy=0.0) -> None: ...
    def step(self, n: int = 1) -> None: ...          # semi-implicit Euler
    def get_state(self) -> list[dict]: ...
    def total_energy(self) -> float: ...
    def reset(self) -> None: ...
```

Key physics constraint: **semi-implicit Euler** — velocities are updated from
forces, then positions are updated from the *new* velocities.  This small detail
separates agents that understand numerical integration from those that don't.

---

## Scoring

| Test | Points | What it checks |
|------|--------|----------------|
| T1 | 5 | Class instantiation and method presence |
| T2 | 10 | `get_state()` insertion order |
| T3 | 5 | `ValueError` on 6th body |
| T4 | 5 | `ValueError` on `step()` with < 2 bodies |
| T5 | 5 | `ZeroDivisionError` on coincident bodies |
| T6 | 25 | Single-step Sun–Earth Euler accuracy |
| T7 | 15 | 3-body equilateral triangle force symmetry |
| T8 | 10 | `total_energy()` at t = 0 |
| T9 | 5 | `reset()` clears all state |
| T10 | 10 | 1 000-step energy drift < 1 % |
| **Total** | **100** | |

Pass threshold: **70 / 100**

---

## Visual simulator

`simulator_visual.html` runs entirely in the browser with no dependencies.

Features:
- Up to 5 bodies with randomised masses and colours
- Configurable gravitational constant G, number of bodies, and time step dt
- Orbital trail rendering
- Real-time energy readout
- Pause / reset controls

Open it with any modern browser.

---

## Running with Harbor

```bash
harbor run harbor_task.yaml
```

The task definition in `harbor_task.yaml` configures:
- Docker sandbox (Python 3.12-slim, no network)
- Terminus 2 agent with 8 retry attempts
- Reflection on verifier output between attempts
- 300-second wall-time limit

---

## Gym environment

`gym/gravity_env.py` wraps `GravitySimulator` in a standard `reset` / `step` /
`render` interface.

Built-in scenarios:

| Name | Bodies | Description |
|------|--------|-------------|
| `earth_sun` | 2 | Earth orbiting the Sun |
| `earth_sun_jupiter` | 3 | Add Jupiter's gravitational influence |
| `figure8` | 3 | Chenciner-Montgomery figure-8 choreography |
| `random_3body` | 3 | Equilateral triangle with tangential velocities |

Reward: negative fractional energy drift per step (0 = perfect conservation).

---

## Dependencies

- Python ≥ 3.11
- No external packages required

---

## License

MIT
