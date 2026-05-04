"""
Gravity RL Environment — starter scaffold.

The agent must implement GravitySimulator in simulator.py (placed next to this
file or on PYTHONPATH).  This module wraps it in a minimal gym-style interface
so training harnesses can call reset() / step() / render() in the standard way.
"""

import math
import importlib.util
import os
from pathlib import Path


def _load_simulator(path: str | None = None):
    """Load GravitySimulator from a file path or fall back to the package."""
    if path is None:
        path = os.environ.get("SIMULATOR_PATH", str(Path(__file__).parent.parent / "simulator.py"))
    spec = importlib.util.spec_from_file_location("simulator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GravitySimulator


# ---------------------------------------------------------------------------
# Observation / action spaces (lightweight stand-ins — no gym dependency)
# ---------------------------------------------------------------------------

class Box:
    """Minimal continuous space descriptor."""
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def __repr__(self):
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


class Discrete:
    """Minimal discrete space descriptor."""
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return f"Discrete(n={self.n})"


# ---------------------------------------------------------------------------
# GravityEnv
# ---------------------------------------------------------------------------

class GravityEnv:
    """
    2-D N-body gravity environment with a configurable reward signal.

    Observation
    -----------
    Flat vector of length 6*N containing [x, y, vx, vy, mass, 0.0] per body,
    normalised by AU (1.496e11 m) for positions and 30 km/s for velocities.

    Action space
    ------------
    Currently a no-op placeholder (Discrete(1)).  Override `apply_action` to
    add velocity-kick actions, thrust, etc.

    Reward
    ------
    Negative fractional energy drift per step.  Perfect energy conservation →
    reward = 0.  Large drift → large negative reward.

    Episode termination
    -------------------
    - `max_steps` exceeded
    - A body escapes beyond `escape_radius` AU
    - Two bodies collide (ZeroDivisionError caught internally)
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        scenario: str = "earth_sun",
        G: float = 6.674e-11,
        dt: float = 10.0,
        max_steps: int = 1000,
        escape_radius: float = 5.0,          # in AU
        simulator_path: str | None = None,
    ):
        self.G = G
        self.dt = dt
        self.max_steps = max_steps
        self.escape_radius = escape_radius * 1.496e11  # convert AU → m
        self.scenario = scenario
        self._GravitySimulator = _load_simulator(simulator_path)

        self._sim: object = None
        self._step_count: int = 0
        self._initial_energy: float = 0.0
        self._n_bodies: int = 0

        self.observation_space: Box | None = None
        self.action_space = Discrete(1)

    # ------------------------------------------------------------------
    # Public gym API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[list[float], dict]:
        """Reset the environment and return (observation, info)."""
        self._sim = self._GravitySimulator(G=self.G, dt=self.dt)
        self._step_count = 0
        self._setup_scenario(self.scenario)
        self._n_bodies = len(self._sim.get_state())
        self.observation_space = Box(-1e12, 1e12, shape=(6 * self._n_bodies,))
        self._initial_energy = self._sim.total_energy()
        obs = self._get_obs()
        return obs, {"energy": self._initial_energy, "n_bodies": self._n_bodies}

    def step(self, action) -> tuple[list[float], float, bool, bool, dict]:
        """
        Apply action, advance one dt, return (obs, reward, terminated, truncated, info).
        """
        self.apply_action(action)
        terminated = False
        try:
            self._sim.step(1)
        except ZeroDivisionError:
            terminated = True

        self._step_count += 1
        obs = self._get_obs()
        energy = self._sim.total_energy()
        drift = abs(energy - self._initial_energy) / (abs(self._initial_energy) + 1e-30)
        reward = -drift

        if not terminated:
            terminated = self._check_escape()

        truncated = self._step_count >= self.max_steps
        info = {
            "step": self._step_count,
            "energy": energy,
            "energy_drift": drift,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "ansi") -> str:
        """Return a simple ASCII snapshot of the current state."""
        lines = [f"Step {self._step_count}"]
        for b in self._sim.get_state():
            au = 1.496e11
            lines.append(
                f"  {b['name']:10s}  x={b['x']/au:+.4f} AU  y={b['y']/au:+.4f} AU"
                f"  vx={b['vx']/1e3:+.2f} km/s  vy={b['vy']/1e3:+.2f} km/s"
            )
        return "\n".join(lines)

    def close(self):
        self._sim = None

    # ------------------------------------------------------------------
    # Extensible hooks
    # ------------------------------------------------------------------

    def apply_action(self, action):
        """Override to add thrust or other body modifications."""
        pass

    def _setup_scenario(self, name: str):
        """Populate the simulator with a named scenario."""
        AU = 1.496e11
        M_SUN = 1.989e30
        M_EARTH = 5.972e24
        M_JUPITER = 1.898e27

        if name == "earth_sun":
            self._sim.add_body("Sun",   M_SUN,     0.0,         0.0,    vy=0.0)
            self._sim.add_body("Earth", M_EARTH,   AU,          0.0,    vy=29_780.0)

        elif name == "earth_sun_jupiter":
            self._sim.add_body("Sun",     M_SUN,     0.0,             0.0, vy=0.0)
            self._sim.add_body("Earth",   M_EARTH,   AU,              0.0, vy=29_780.0)
            self._sim.add_body("Jupiter", M_JUPITER, 5.2 * AU,        0.0, vy=13_070.0)

        elif name == "figure8":
            # Chenciner-Montgomery figure-8 choreography (scaled to AU-ish units)
            m = 1e30
            scale = 1e11
            v_scale = 1e3
            self._sim.add_body("A", m, -0.97000436 * scale,  0.24308753 * scale,
                                vx=0.93240737 / 2 * v_scale, vy=0.86473146 / 2 * v_scale)
            self._sim.add_body("B", m,  0.97000436 * scale, -0.24308753 * scale,
                                vx=0.93240737 / 2 * v_scale, vy=0.86473146 / 2 * v_scale)
            self._sim.add_body("C", m,  0.0,                 0.0,
                                vx=-0.93240737 * v_scale,   vy=-0.86473146 * v_scale)

        elif name == "random_3body":
            import random
            rng = random.Random()
            m = 1e26
            for i in range(3):
                angle = i * 2 * math.pi / 3
                r = 1e9
                self._sim.add_body(
                    f"P{i}", m,
                    r * math.cos(angle), r * math.sin(angle),
                    vx=-1e3 * math.sin(angle), vy=1e3 * math.cos(angle),
                )

        else:
            raise ValueError(f"Unknown scenario: {name!r}")

    def _get_obs(self) -> list[float]:
        AU = 1.496e11
        V_REF = 3e4
        obs = []
        for b in self._sim.get_state():
            obs.extend([
                b["x"] / AU,
                b["y"] / AU,
                b["vx"] / V_REF,
                b["vy"] / V_REF,
                math.log10(b["mass"]),
                0.0,
            ])
        return obs

    def _check_escape(self) -> bool:
        for b in self._sim.get_state():
            r = math.sqrt(b["x"] ** 2 + b["y"] ** 2)
            if r > self.escape_radius:
                return True
        return False
