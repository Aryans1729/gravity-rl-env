# Gravity RL Environment

A **coding task for AI agents**: implement a physically accurate 2D N-body gravity
simulator in Python. It runs as a [Harbor](https://harborframework.com) task, with a
hardened verifier that checks real physics on randomized hidden scenarios.

The interesting part is the verifier. Version 1 accepted submissions that were wrong,
including one with no physics at all. [HARDENING.md](HARDENING.md) documents each
reward hack, how it was closed, and the evidence.

---

## Results at a glance

Each submission scored by the old (v1) and new (v2) verifier, over 10 random seeds:

| Submission | v1 verifier | v2 verifier |
|---|---|---|
| Reference solution | PASS 125/125 | 10/10 pass |
| Different but correct solution | PASS 125/125 | 10/10 pass |
| Explicit Euler (wrong update order) | 121/125 | 0/10 pass |
| Explicit Euler + cached `total_energy()` | 121/125 | 0/10 pass |
| Empty class that hijacks the verifier | **PASS 125/125** | 0/10 pass, 0/125 |

Reproduce it:

```bash
python3 dev/check_verifier.py
```

---

## Quick start

Requirements: Python 3.11+, no packages. Harbor runs also need Docker.

```bash
git clone <repo-url> gravity-rl-env
cd gravity-rl-env

# 1. Old vs new verifier against the reference, a second correct solution and 3 hacks
python3 dev/check_verifier.py

# 2. Full verifier report for one submission (set VERIFIER_SEED to reproduce a run)
python3 -I tasks/gravity-simulator/tests/verifier.py simulator.py

# 3. Run the task in Docker with Harbor's oracle (reference) solution
harbor run -p tasks/gravity-simulator -a oracle
harbor view jobs

# 4. See the physics in the browser
open simulator_visual.html
```

To run a real agent, pick it with `-a` and a model with `-m`, for example
`harbor run -p tasks/gravity-simulator -a claude-code -m anthropic/claude-haiku-4-5`.
See the [Harbor docs](https://docs.harborframework.com/tutorials/create-a-task) for
agents and credentials.

---

## Project structure

```
gravity-rl-env/
├── tasks/gravity-simulator/       The Harbor task
│   ├── instruction.md             Spec shown to the agent
│   ├── task.toml                  Timeouts and metadata
│   ├── environment/Dockerfile     Agent's container (python:3.12-slim, /app)
│   ├── solution/solve.sh          Reference solution (oracle)
│   └── tests/
│       ├── test.sh                Writes 1/0 to /logs/verifier/reward.txt
│       ├── verifier.py            Scenarios, reference physics, scoring
│       └── worker.py              Runs agent code in isolation, reports raw observations
├── dev/
│   ├── check_verifier.py          Old vs new verifier comparison
│   ├── hacks/                     Three reward-hacking submissions
│   ├── alt_correct_simulator.py   Second correct solution (guards against false fails)
│   └── old_verifier.py            v1 verifier, kept for the comparison
├── gym/gravity_env.py             Gym-style wrapper for demos
├── simulator.py                   Simulator used by the gym demo
├── simulator_visual.html          Interactive browser visualiser
├── HARDENING.md                   What was broken, how it was fixed
└── CLAUDE.md                      Developer guide
```

The agent's container is built only from `environment/`, so the agent never sees the
tests or the reference solution.

---

## The task

The agent writes `/app/simulator.py` with this interface:

```python
class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0): ...
    def add_body(self, name, mass, x, y, vx=0.0, vy=0.0) -> None: ...
    def step(self, n: int = 1) -> None: ...          # semi-implicit Euler
    def get_state(self) -> list[dict]: ...
    def total_energy(self) -> float: ...
    def reset(self) -> None: ...
    def center_of_mass(self) -> tuple[float, float]: ...
    def orbital_period(self, name) -> float: ...
```

The key physics constraint is **semi-implicit Euler**: update velocities from the forces
first, then update positions with the *new* velocities. Getting this order wrong is the
most common mistake, and the verifier is built to catch it.
Full spec: [`tasks/gravity-simulator/instruction.md`](tasks/gravity-simulator/instruction.md).

---

## How it's scored

Every test runs the agent's code in a separate process on scenarios randomized per run.
All scoring happens outside that process, against an independent reference integrator.

| Test | Points | What it checks |
|------|--------|----------------|
| T1 | 5 | All methods present and callable |
| T2 | 10 | `get_state()` order and values |
| T3 | 5 | `ValueError` on a 6th body |
| T4 | 5 | `ValueError` on `step()` with fewer than 2 bodies |
| T5 | 5 | `ZeroDivisionError` on coincident bodies |
| **T6** | 25 | Integrator accuracy at large time steps (detects wrong update order) |
| T7 | 15 | 3-body symmetric forces |
| **T8** | 10 | `total_energy()` correctness |
| T9 | 5 | `reset()` gives a clean simulator |
| **T10** | 10 | ~3 orbits: trajectory, independently computed energy drift < 1%, honest `total_energy()` |
| T11 | 15 | `orbital_period(name)` (Kepler, relative orbit) |
| T12 | 15 | `center_of_mass()` |
| **Total** | **125** | |

**Pass:** at least 122/125 **and** full marks on T6, T8 and T10. Harbor reward is 1 for
a pass, 0 otherwise. Per-test details go to `/logs/verifier/details.json`.

---

## Gym environment

`gym/gravity_env.py` wraps the simulator in a `reset` / `step` / `render` interface.
The reward is the negative fractional energy drift per step (0 = perfect conservation).

| Scenario | Bodies | Description |
|------|--------|-------------|
| `earth_sun` | 2 | Earth orbiting the Sun |
| `earth_sun_jupiter` | 3 | Adds Jupiter's influence |
| `figure8` | 3 | Chenciner–Montgomery figure-8 choreography |
| `random_3body` | 3 | Equilateral triangle with tangential velocities |

```bash
python3 - <<'EOF'
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
```

---

## Visual simulator

`simulator_visual.html` runs in any browser with no dependencies: up to 5 bodies,
configurable G and time step, orbital trails, a live energy readout, and pause/reset.

---

## License

MIT