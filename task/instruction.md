# Task: Implement a 2D N-Body Gravitational Simulator

## Your Goal

Create a file called **`simulator.py`** in the current working directory containing a class called `GravitySimulator` that correctly implements a 2D N-body gravitational simulator using semi-implicit Euler integration.

Your solution will be automatically scored by running:

```
python verifier/verifier.py simulator.py
```

You must score **122 / 125 or higher** to pass.

---

## Required Interface

```python
class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0):
        # G = gravitational constant (m³ kg⁻¹ s⁻²)
        # dt = time step in seconds

    def add_body(self, name: str, mass: float, x: float, y: float,
                 vx: float = 0.0, vy: float = 0.0) -> None:
        # Add a body. Raise ValueError if this would exceed 5 bodies.

    def step(self, n: int = 1) -> None:
        # Advance the simulation by n steps.
        # Raise ValueError if fewer than 2 bodies are present.
        # Raise ZeroDivisionError if any two bodies share the same position.

    def get_state(self) -> list[dict]:
        # Return list of body dicts with keys: name, mass, x, y, vx, vy
        # Bodies must appear in insertion order.

    def total_energy(self) -> float:
        # KE = Σ ½ m (vx² + vy²)
        # PE = Σ_{i<j} −G m_i m_j / r_ij
        # Return KE + PE

    def reset(self) -> None:
        # Clear all bodies and reset time to 0.

    def orbital_period(self, name: str) -> float:
        # Return an estimate of the orbital period (in seconds) of the named body
        # around the system's centre of mass, based on the current positions
        # and masses of all bodies.
        # Raise ValueError if no body with that name exists.

    def center_of_mass(self) -> tuple[float, float]:
        # Return the mass-weighted centre of all bodies as an (x, y) tuple.
```

---

## Physics

### Gravitational force between body i and body j

```
r  = sqrt((x_j - x_i)² + (y_j - y_i)²)
F  = G * m_i * m_j / r²
Fx = F * (x_j - x_i) / r      # force on i in x direction
Fy = F * (y_j - y_i) / r      # force on i in y direction
```

Body j receives equal and opposite forces: `(-Fx, -Fy)`.

### Integration order (semi-implicit Euler)

For each time step:

1. **Compute ALL pairwise forces** and accumulate accelerations.  
   Do this for every body before changing any velocity or position.
2. **Update velocities** using the accelerations:
   ```
   vx += ax * dt
   vy += ay * dt
   ```
3. **Update positions** using the **new** velocities (this is what makes it semi-implicit):
   ```
   x += vx * dt
   y += vy * dt
   ```

> This order is critical. Using the old velocity for the position update (explicit Euler) will give wrong answers and fail the tests.

---

## Error Conditions

| Condition | Exception |
|-----------|-----------|
| Adding a 6th body | `ValueError` |
| Calling `step()` with < 2 bodies | `ValueError` |
| Two bodies at identical (x, y) when `step()` is called | `ZeroDivisionError` |

---

## Constraints

- Use only the Python standard library (`math`) and optionally `numpy`.
- Do **not** use `scipy`, `astropy`, or any other physics/astronomy library.
- `get_state()` must return body dictionaries in the order they were added.

---

## Scoring Breakdown (125 points total)

| Test | Points | Description |
|------|--------|-------------|
| T1 | 5 | Class instantiation and required method presence |
| T2 | 10 | `add_body` / `get_state` insertion order |
| T3 | 5 | `ValueError` on 6th body |
| T4 | 5 | `ValueError` on `step()` with < 2 bodies |
| T5 | 5 | `ZeroDivisionError` on coincident bodies |
| T6 | 25 | Single-step Euler accuracy (Sun–Earth system) |
| T7 | 15 | 3-body symmetric force correctness |
| T8 | 10 | `total_energy()` at t=0 |
| T9 | 5 | `reset()` clears all state |
| T10 | 10 | 1000-step energy stability (< 1 % drift for full marks) |
| T11 | 15 | `orbital_period(name)` — estimate the orbital period of the named body around the system's centre of mass |
| T12 | 15 | `center_of_mass()` — mass-weighted (x, y) for 1-, 2-, and 3-body cases |

---

## Example Usage

```python
from simulator import GravitySimulator

sim = GravitySimulator(G=6.674e-11, dt=1.0)

M_SUN   = 1.989e30
M_EARTH = 5.972e24
AU      = 1.496e11

sim.add_body("Sun",   M_SUN,   0.0, 0.0, vx=0.0, vy=0.0)
sim.add_body("Earth", M_EARTH, AU,  0.0, vx=0.0, vy=29780.0)

print("Initial energy:", sim.total_energy())

sim.step(1000)

state = sim.get_state()
for body in state:
    print(f"{body['name']}: x={body['x']:.3e} m, y={body['y']:.3e} m")

sim.reset()
```

---

## Tips

- Start by getting the force calculation right for a 2-body system.
- The verifier uses exact floating-point comparisons with a tight tolerance (~1e-6 relative error), so avoid accumulating unnecessary rounding.
- For energy stability (T10), double-check that positions use the **updated** velocity, not the original.
- `math.sqrt` is sufficient — no need for `numpy`.
