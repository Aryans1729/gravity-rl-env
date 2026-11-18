# Hardening the Gravity RL Environment

The first version of this environment rewarded submissions that were wrong. This file
records what was broken, how it was fixed, and the evidence. Reproduce everything with:

```bash
python3 dev/check_verifier.py
```

## Results

Each submission was scored by the v1 verifier and by the v2 verifier over 10 random seeds.

| Submission | v1 verifier | v2 verifier |
|---|---|---|
| Reference solution | PASS 125/125 | 10/10 pass, 125/125 |
| Different but correct solution | PASS 125/125 | 10/10 pass, 125/125 |
| Explicit Euler (wrong update order) | 121/125 | 0/10 pass, 98/125 |
| Explicit Euler + cached `total_energy()` | 121/125 | 0/10 pass, 96/125 |
| No physics, hijacks the verifier | **PASS 125/125** | 0/10 pass, 0/125 |

The old `harbor_task.yaml` also compared the earned score (121) against a pass mark of 70
out of 100, so the two explicit-Euler hacks would have counted as successes there too.

## What was broken and how it was fixed

**1. Agent code ran inside the verifier.** v1 imported `simulator.py` into its own
process, so module-level code could replace the verifier's `test()` function and award
itself full marks with an empty class.
*Fix:* every test runs the agent's code in a separate `python3 -I worker.py` process. The
worker holds no reference physics and no scoring; it only reports raw observations. All
scoring happens in the parent against an independent reference integrator.

**2. The integrator check could not see the update order.** v1 checked absolute
positions after one 1-second step of a Sun–Earth system. The difference between
semi-implicit and explicit Euler there is about 6 mm on 1.5e11 m, far inside the 1e-6
tolerance.
*Fix:* T6 uses large time steps (100–600 s) and compares *displacements* and *velocity
changes*, where the two methods differ by tens of percent. A diagnosis line names the
explicit-Euler mistake when it happens.

**3. Energy stability trusted the agent's own `total_energy()`.** A cached value made the
drift read 0%, and the v1 run (1000 steps of 10 s, a few hours of a one-year orbit) was too
short for drift to show anyway.
*Fix:* T10 runs ~3.3 orbits at 300 steps per orbit, computes energy itself from the
returned state, checks the trajectory against the reference, and checks that
`total_energy()` agrees with the state at the start and the end. Semi-implicit Euler stays
under 0.3% drift here; explicit Euler drifts over 30%.

**4. Fixed scenarios.** Every expected value in v1 was a constant, and the agent could
read the verifier.
*Fix:* all scenarios are randomized per run (`VERIFIER_SEED` reproduces a run), and tests
live in `tests/`, which is not copied into the agent's image.

**5. Partial credit hid core failures.** Missing the physics could still clear the bar.
*Fix:* T6, T8 and T10 must be full marks, on top of the 122/125 threshold.

**6. A hanging submission could stall the verifier.** *Fix:* 20-second worker timeout;
after one timeout the remaining tests are skipped.

## Known limits

- Agent code still runs during verification, so a submission written to anticipate this
  exact verifier could try to read files in `/tests` at runtime. Because tests are hidden
  during the agent phase and scenarios are random, it would have to guess blindly.
- Tolerances were checked against two independent correct implementations over 40 seeds
  each with no false failures. A very different but valid formulation (for example,
  extended precision) should still be tried before trusting the bar.
