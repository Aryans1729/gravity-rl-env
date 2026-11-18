# 2D N-Body Gravity Simulator

Create the file `/app/simulator.py` containing a class `GravitySimulator` that simulates
Newtonian gravity between point masses in two dimensions.

Use only the Python standard library. Put everything in that single file.

## Interface

```python
class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0): ...
    def add_body(self, name, mass, x, y, vx=0.0, vy=0.0) -> None: ...
    def step(self, n: int = 1) -> None: ...
    def get_state(self) -> list[dict]: ...
    def total_energy(self) -> float: ...
    def reset(self) -> None: ...
    def center_of_mass(self) -> tuple[float, float]: ...
    def orbital_period(self, name) -> float: ...
```

Units are SI throughout (kg, m, m/s, s).

## Behaviour

**`add_body`** adds a point mass. The simulator holds at most 5 bodies; adding a sixth
raises `ValueError` and leaves the existing bodies unchanged.

**`get_state`** returns one dict per body, in the order the bodies were added, with the
keys `name`, `mass`, `x`, `y`, `vx`, `vy`. Values must be the simulator's current values.

**`step(n)`** advances the simulation by `n` time steps of size `dt`.

- Every body attracts every other body with force `G * m_i * m_j / r**2`, directed
  along the line between them. There is no softening.
- Integrate with **semi-implicit (symplectic) Euler**. In each step:
  1. compute the acceleration of every body from the current positions;
  2. update every velocity: `v += a * dt`;
  3. update every position using the **new** velocity: `x += v * dt`.
- Raise `ValueError` if fewer than 2 bodies are present.
- Raise `ZeroDivisionError` if two bodies occupy exactly the same position.

**`total_energy`** returns the total mechanical energy of the current state: the sum of
kinetic energies `0.5 * m * v**2` plus the gravitational potential energy
`-G * m_i * m_j / r` of every pair, each pair counted once.

**`reset`** removes all bodies. After a reset the simulator must behave exactly like a
newly created one with the same `G` and `dt`.

**`center_of_mass`** returns `(x, y)`, the mass-weighted mean position of all bodies.
Raise `ValueError` if there are no bodies.

**`orbital_period(name)`** returns the period, in seconds, of the Keplerian orbit
described by the current positions and velocities of a two-body system. `name` must be
one of the two bodies; both bodies share the same period. Treat it as the relative orbit
of the pair (use the separation and relative velocity of the two bodies, and their total
mass). Raise `ValueError` if the simulator does not hold exactly 2 bodies, if `name` is
unknown, or if the orbit is not bound.

## Notes

Your simulator will be checked on scenarios you have not seen, including long runs,
large time steps and unequal masses. Correct physics matters more than matching any
particular example.
