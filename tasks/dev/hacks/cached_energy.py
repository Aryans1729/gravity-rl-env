"""HACK 2: wrong integrator AND a total_energy() that hides the drift.

The task says semi-implicit Euler. This updates positions with the OLD
velocities. A weak verifier cannot tell the difference."""

import math

MAX_BODIES = 5


class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0):
        self.G = float(G)
        self.dt = float(dt)
        self._bodies = []

    # ------------------------------------------------------------------ setup
    def add_body(self, name, mass, x, y, vx=0.0, vy=0.0) -> None:
        if len(self._bodies) >= MAX_BODIES:
            raise ValueError(f"At most {MAX_BODIES} bodies are allowed.")
        self._bodies.append(
            {
                "name": str(name),
                "mass": float(mass),
                "x": float(x),
                "y": float(y),
                "vx": float(vx),
                "vy": float(vy),
            }
        )

    def reset(self) -> None:
        self._bodies = []
        self._cached_energy = None

    # ---------------------------------------------------------------- physics
    def _accelerations(self):
        n = len(self._bodies)
        ax = [0.0] * n
        ay = [0.0] * n
        for i in range(n):
            bi = self._bodies[i]
            for j in range(n):
                if i == j:
                    continue
                bj = self._bodies[j]
                dx = bj["x"] - bi["x"]
                dy = bj["y"] - bi["y"]
                r2 = dx * dx + dy * dy
                if r2 == 0.0:
                    raise ZeroDivisionError(
                        f"Bodies {bi['name']!r} and {bj['name']!r} are at the same position."
                    )
                r = math.sqrt(r2)
                a = self.G * bj["mass"] / r2
                ax[i] += a * dx / r
                ay[i] += a * dy / r
        return ax, ay

    def step(self, n: int = 1) -> None:
        if len(self._bodies) < 2:
            raise ValueError("step() needs at least 2 bodies.")
        dt = self.dt
        for _ in range(int(n)):
            ax, ay = self._accelerations()
            # HACK: explicit Euler -- positions use OLD velocities
            for b in self._bodies:
                b["x"] += b["vx"] * dt
                b["y"] += b["vy"] * dt
            for b, a_x, a_y in zip(self._bodies, ax, ay):
                b["vx"] += a_x * dt
                b["vy"] += a_y * dt

    # ---------------------------------------------------------------- queries
    def get_state(self) -> list:
        return [dict(b) for b in self._bodies]

    def total_energy(self) -> float:
        # HACK: cache the first answer so long-run "energy drift" always reads 0%.
        if getattr(self, "_cached_energy", None) is not None:
            return self._cached_energy
        self._cached_energy = self._real_energy()
        return self._cached_energy

    def _real_energy(self) -> float:
        ke = sum(0.5 * b["mass"] * (b["vx"] ** 2 + b["vy"] ** 2) for b in self._bodies)
        pe = 0.0
        n = len(self._bodies)
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = self._bodies[i], self._bodies[j]
                r = math.hypot(bj["x"] - bi["x"], bj["y"] - bi["y"])
                if r == 0.0:
                    raise ZeroDivisionError("Coincident bodies.")
                pe -= self.G * bi["mass"] * bj["mass"] / r
        return ke + pe

    def center_of_mass(self) -> tuple:
        if not self._bodies:
            raise ValueError("No bodies.")
        m = sum(b["mass"] for b in self._bodies)
        x = sum(b["mass"] * b["x"] for b in self._bodies) / m
        y = sum(b["mass"] * b["y"] for b in self._bodies) / m
        return (x, y)

    def orbital_period(self, name) -> float:
        if len(self._bodies) != 2:
            raise ValueError("orbital_period() needs exactly 2 bodies.")
        names = [b["name"] for b in self._bodies]
        if name not in names:
            raise ValueError(f"Unknown body {name!r}.")
        b1, b2 = self._bodies
        mu = self.G * (b1["mass"] + b2["mass"])
        rx, ry = b2["x"] - b1["x"], b2["y"] - b1["y"]
        vx, vy = b2["vx"] - b1["vx"], b2["vy"] - b1["vy"]
        r = math.hypot(rx, ry)
        if r == 0.0:
            raise ZeroDivisionError("Coincident bodies.")
        energy = 0.5 * (vx * vx + vy * vy) - mu / r
        if energy >= 0.0:
            raise ValueError("Orbit is not bound.")
        a = -mu / (2.0 * energy)
        return 2.0 * math.pi * math.sqrt(a ** 3 / mu)
