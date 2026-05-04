import math


class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0):
        self.G = G
        self.dt = dt
        self.bodies = []
        self.time = 0.0

    def add_body(self, name: str, mass: float, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        if len(self.bodies) >= 5:
            raise ValueError("Cannot add more than 5 bodies")
        self.bodies.append({"name": name, "mass": mass, "x": x, "y": y, "vx": vx, "vy": vy})

    def step(self, n: int = 1) -> None:
        if len(self.bodies) < 2:
            raise ValueError("Need at least 2 bodies to step")
        for _ in range(n):
            forces = [[0.0, 0.0] for _ in self.bodies]
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    bi = self.bodies[i]
                    bj = self.bodies[j]
                    dx = bj["x"] - bi["x"]
                    dy = bj["y"] - bi["y"]
                    r2 = dx * dx + dy * dy
                    if r2 == 0.0:
                        raise ZeroDivisionError("Two bodies at identical position")
                    r = math.sqrt(r2)
                    F = self.G * bi["mass"] * bj["mass"] / r2
                    Fx = F * dx / r
                    Fy = F * dy / r
                    forces[i][0] += Fx
                    forces[i][1] += Fy
                    forces[j][0] -= Fx
                    forces[j][1] -= Fy
            for i, body in enumerate(self.bodies):
                ax = forces[i][0] / body["mass"]
                ay = forces[i][1] / body["mass"]
                body["vx"] += ax * self.dt
                body["vy"] += ay * self.dt
                body["x"] += body["vx"] * self.dt
                body["y"] += body["vy"] * self.dt
            self.time += self.dt

    def get_state(self) -> list[dict]:
        return [dict(b) for b in self.bodies]

    def total_energy(self) -> float:
        ke = sum(0.5 * b["mass"] * (b["vx"] ** 2 + b["vy"] ** 2) for b in self.bodies)
        pe = 0.0
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                bi = self.bodies[i]
                bj = self.bodies[j]
                dx = bj["x"] - bi["x"]
                dy = bj["y"] - bi["y"]
                r = math.sqrt(dx * dx + dy * dy)
                pe += -self.G * bi["mass"] * bj["mass"] / r
        return ke + pe

    def reset(self) -> None:
        self.bodies = []
        self.time = 0.0
