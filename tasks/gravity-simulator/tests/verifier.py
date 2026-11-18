"""Hardened verifier for the 2D N-body gravity simulator task.

Design (each point closes a reward-hacking path found in v1):
  * The agent's code never runs in the scoring process. Every test runs in a
    fresh `python3 -I worker.py` subprocess with a timeout. worker.py holds no
    reference physics, so agent code cannot borrow or patch it. The worker only
    reports raw observations (states, return values, exception types); all
    scoring is done here, in the parent, against an independent reference
    integrator.
    -> agent code cannot patch the verifier, forge the score line, or hang it.
  * Agent stdout/stderr is discarded; the worker's result travels on a private
    file descriptor tagged with a per-run nonce.
  * Scenarios are randomized per run (set VERIFIER_SEED to reproduce).
    -> fixed expected values cannot be hardcoded.
  * The integrator is checked on displacements over large dt, so explicit
    Euler (wrong update order) is actually detectable.
  * Energy stability is computed from the returned state, not from the agent's
    total_energy(), and runs ~3 orbits at 300 steps/orbit, where semi-implicit
    Euler stays < 1% and explicit Euler drifts > 20%.
  * Core physics tests (T6, T8, T10) gate the pass: full marks required.

Usage:
  python3 -I verifier.py /app/simulator.py [--reward-file PATH] [--details-file PATH]
Exit code 0 = pass, 1 = fail, 2 = usage error.
"""

import json
import math
import os
import random
import secrets
import subprocess
import sys

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
G = 6.674e-11
PASS_THRESHOLD = 122
GATING_TESTS = ("T6", "T8", "T10")
WORKER_TIMEOUT_SEC = 20
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

# ==========================================================================
# PARENT SIDE  (reference physics, scenario generation, scoring)
# ==========================================================================


def ref_step(bodies, g, dt, n, semi_implicit=True):
    b = [dict(x) for x in bodies]
    for _ in range(n):
        acc = []
        for i, bi in enumerate(b):
            ax = ay = 0.0
            for j, bj in enumerate(b):
                if i == j:
                    continue
                dx, dy = bj["x"] - bi["x"], bj["y"] - bi["y"]
                r2 = dx * dx + dy * dy
                if r2 == 0.0:
                    raise ZeroDivisionError
                r = math.sqrt(r2)
                a = g * bj["m"] / r2
                ax += a * dx / r
                ay += a * dy / r
            acc.append((ax, ay))
        if semi_implicit:
            for bi, (ax, ay) in zip(b, acc):
                bi["vx"] += ax * dt
                bi["vy"] += ay * dt
            for bi in b:
                bi["x"] += bi["vx"] * dt
                bi["y"] += bi["vy"] * dt
        else:  # explicit Euler, used only to diagnose wrong update order
            for bi, (ax, ay) in zip(b, acc):
                bi["x"] += bi["vx"] * dt
                bi["y"] += bi["vy"] * dt
                bi["vx"] += ax * dt
                bi["vy"] += ay * dt
    return b


def ref_energy(bodies, g):
    ke = sum(0.5 * b["m"] * (b["vx"] ** 2 + b["vy"] ** 2) for b in bodies)
    pe = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            r = math.hypot(bodies[j]["x"] - bodies[i]["x"], bodies[j]["y"] - bodies[i]["y"])
            pe -= g * bodies[i]["m"] * bodies[j]["m"] / r
    return ke + pe


def kepler_period(b1, b2, g):
    mu = g * (b1["m"] + b2["m"])
    r = math.hypot(b2["x"] - b1["x"], b2["y"] - b1["y"])
    v2 = (b2["vx"] - b1["vx"]) ** 2 + (b2["vy"] - b1["vy"]) ** 2
    a = -mu / (2.0 * (0.5 * v2 - mu / r))
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def rel(got, exp):
    return abs(got - exp) / abs(exp) if exp != 0 else abs(got)


def vec_rel(got, exp):
    """Relative error of a 2-vector, scaled by the reference vector's length."""
    norm = math.hypot(*exp)
    err = math.hypot(got[0] - exp[0], got[1] - exp[1])
    return err / norm if norm else err


def with_masses(agent_state, ref_bodies):
    """Attach masses known to the verifier (by position in insertion order)."""
    return [dict(s, m=r["m"]) for s, r in zip(agent_state, ref_bodies)]


def body(name, m, x, y, vx=0.0, vy=0.0):
    return {"name": name, "m": m, "x": x, "y": y, "vx": vx, "vy": vy}


def polar(r, ang):
    return r * math.cos(ang), r * math.sin(ang)


# ------------------------------------------------------------ scenario makers
def make_scenarios(rng):
    sc = {}
    tau = 2 * math.pi

    sc["T1"] = {}

    names = rng.sample(["Io", "Rhea", "Titan", "Vesta", "Ceres", "Pallas", "Juno", "Hebe"], 3)
    sc["T2"] = {"G": G, "dt": 1.0, "bodies": [
        body(n, rng.uniform(1e22, 1e26), rng.uniform(-1e10, 1e10), rng.uniform(-1e10, 1e10),
             rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3)) for n in names]}

    sc["T3"] = {}
    sc["T4"] = {}

    px, py = rng.uniform(-1e10, 1e10), rng.uniform(-1e10, 1e10)
    sc["T5"] = {"G": G, "dt": 1.0, "bodies": [body("X", 1e24, px, py), body("Y", 3e24, px, py)]}

    # T6: two bodies, large dt so a*dt^2 is comparable to v*dt (order is visible).
    r = rng.uniform(5e9, 2e10)
    ang = rng.uniform(0, tau)
    x2, y2 = polar(r, ang)
    v0 = rng.uniform(200, 2000)
    vang = rng.uniform(0, tau)
    vx2, vy2 = polar(v0, vang)
    sc["T6"] = {"G": G, "dt": rng.uniform(100, 600), "chunks": [1, 3],
                "bodies": [body("Star", rng.uniform(5e29, 3e30), 0.0, 0.0,
                                rng.uniform(-50, 50), rng.uniform(-50, 50)),
                           body("Planet", rng.uniform(1e23, 1e26), x2, y2, vx2, vy2)]}

    # T7: equilateral triangle at rest, random size / mass / rotation.
    R = rng.uniform(5e8, 5e9)
    rot = rng.uniform(0, tau)
    m = rng.uniform(1e25, 1e27)
    tri = [body(f"P{i}", m, *polar(R, rot + i * tau / 3)) for i in range(3)]
    sc["T7"] = {"G": G, "dt": rng.uniform(10, 200), "chunks": [1], "bodies": tri}

    # T8: two random configurations with velocities.
    def rand_cfg(k):
        return {"G": G, "dt": 1.0, "bodies": [
            body(f"B{i}", rng.uniform(1e23, 1e30), *polar(rng.uniform(1e9, 1e11), rng.uniform(0, tau)),
                 rng.uniform(-3e4, 3e4), rng.uniform(-3e4, 3e4)) for i in range(k)]}
    sc["T8"] = {"configs": [rand_cfg(3), rand_cfg(2)]}

    sc["T9"] = {"first": rand_cfg(2), "second": rand_cfg(2)}
    sc["T9"]["second"]["G"] = G
    sc["T9"]["second"]["dt"] = 1.0

    # T10: near-circular orbit, 300 steps per orbit, 1000 steps (~3.3 orbits).
    m1 = rng.uniform(1e29, 2e30)
    m2 = rng.uniform(1e22, 1e25)
    r = rng.uniform(5e10, 2e11)
    mu = G * (m1 + m2)
    period = tau * math.sqrt(r ** 3 / mu)
    v = math.sqrt(mu / r) * rng.uniform(0.97, 1.03)
    ang = rng.uniform(0, tau)
    x2, y2 = polar(r, ang)
    vx2, vy2 = polar(v, ang + math.pi / 2)
    sc["T10"] = {"G": G, "dt": period / 300.0, "chunks": [100] * 10, "energy_at_end": True,
                 "bodies": [body("Sun", m1, 0.0, 0.0), body("World", m2, x2, y2, vx2, vy2)]}

    # T11: one eccentric bound orbit + one equal-mass circular binary.
    m1 = rng.uniform(1e29, 3e30)
    m2 = rng.uniform(1e23, 1e27)
    r = rng.uniform(5e10, 3e11)
    mu = G * (m1 + m2)
    v = math.sqrt(mu / r) * rng.uniform(0.8, 1.2)
    ang = rng.uniform(0, tau)
    ecc = {"G": G, "dt": 1.0, "query": rng.choice(["Primary", "Companion"]),
           "bodies": [body("Primary", m1, 0.0, 0.0),
                      body("Companion", m2, *polar(r, ang), *polar(v, ang + math.pi / 2))]}
    mb = rng.uniform(1e25, 1e28)
    sep = rng.uniform(5e9, 5e10)
    vc = math.sqrt(G * mb / (2.0 * sep))
    ang = rng.uniform(0, tau)
    hx, hy = polar(sep / 2, ang)
    ux, uy = polar(vc, ang + math.pi / 2)
    binary = {"G": G, "dt": 1.0, "query": rng.choice(["Alpha", "Beta"]),
              "bodies": [body("Alpha", mb, -hx, -hy, -ux, -uy), body("Beta", mb, hx, hy, ux, uy)]}
    sc["T11"] = {"cases": [ecc, binary]}

    # T12: centre of mass for 2, 3 and 1 bodies.
    def com_case(k):
        return {"G": G, "dt": 1.0, "bodies": [
            body(f"C{i}", rng.uniform(1e22, 1e27), rng.uniform(-1e10, 1e10), rng.uniform(-1e10, 1e10))
            for i in range(k)]}
    sc["T12"] = {"cases": [com_case(2), com_case(3), com_case(1)]}
    return sc


# ------------------------------------------------------------------- runner
def run_worker(sim_path, test_id, params):
    nonce = secrets.token_hex(16)
    job = json.dumps({"nonce": nonce, "test": test_id, "params": params})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", WORKER_PATH, sim_path],
            input=job, capture_output=True, text=True, timeout=WORKER_TIMEOUT_SEC,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Timed out after {WORKER_TIMEOUT_SEC}s"}
    for line in proc.stdout.splitlines():
        if line.startswith(nonce):
            try:
                return json.loads(line[len(nonce):])
            except json.JSONDecodeError:
                break
    return {"error": f"Worker produced no result (exit code {proc.returncode})"}


# ------------------------------------------------------------------- scoring
def score_t1(o, p):
    missing = [k for k, ok in o.items() if not ok]
    return (0 if missing else 5), ("All methods present." if not missing else f"Missing/uncallable: {missing}")


def score_t2(o, p):
    st, ref = o["state"], p["bodies"]
    if len(st) != len(ref):
        return 0, f"Expected {len(ref)} bodies, got {len(st)}."
    order_ok = [s["name"] for s in st] == [b["name"] for b in ref]
    vals_ok = all(rel(s[k], b[k]) < 1e-12 for s, b in zip(st, ref) for k in ("x", "y", "vx", "vy"))
    return 6 * order_ok + 4 * vals_ok, f"order={'OK' if order_ok else 'FAIL'}, values={'OK' if vals_ok else 'FAIL'}"


def score_t3(o, p):
    ok = o["raised"] == "ValueError" and o["count"] == 5
    return (5 if ok else 0), f"raised={o['raised']}, bodies kept={o['count']}"


def score_t4(o, p):
    e, s = o["empty"] == "ValueError", o["one"] == "ValueError"
    return 2 * e + 3 * s, f"0 bodies: {o['empty']}, 1 body: {o['one']}"


def score_t5(o, p):
    ok = o["raised"] == "ZeroDivisionError"
    return (5 if ok else 0), f"raised={o['raised']}"


def _delta_checks(start, end, ref_end, tol):
    """Compare displacement and velocity change of every body against reference."""
    notes, passed, total = [], 0, 0
    for s0, s1, r1 in zip(start, end, ref_end):
        for kind, kx, ky in (("dpos", "x", "y"), ("dvel", "vx", "vy")):
            got = (s1[kx] - s0[kx], s1[ky] - s0[ky])
            exp = (r1[kx] - s0[kx], r1[ky] - s0[ky])
            err = vec_rel(got, exp)
            ok = err <= tol
            total += 1
            passed += ok
            notes.append(f"{s0['name']}.{kind}: relerr={err:.2e} {'OK' if ok else 'FAIL'}")
    return passed, total, notes


def score_t6(o, p):
    states = o["states"]
    if len(states) != 3 or any(len(s) != 2 for s in states):
        return 0, "Unexpected state shape."
    b0 = p["bodies"]
    after1 = ref_step(b0, p["G"], p["dt"], 1)
    after4 = ref_step(after1, p["G"], p["dt"], 3)
    p1, t1, n1 = _delta_checks(states[0], states[1], after1, 1e-6)
    p2, t2, n2 = _delta_checks(states[1], states[2], after4, 1e-6)
    earned = round(25 * (p1 + p2) / (t1 + t2))
    notes = ["step(1):"] + n1 + ["step(3) after that:"] + n2
    if earned < 25:
        exp1 = ref_step(b0, p["G"], p["dt"], 1, semi_implicit=False)
        _, _, en = _delta_checks(states[0], states[1], exp1, 1e-6)
        if all(n.endswith("OK") for n in en):
            notes.append("DIAGNOSIS: matches EXPLICIT Euler (positions updated with old velocities).")
    return earned, "\n".join(notes)


def score_t7(o, p):
    start, end = o["states"]
    ref = ref_step(p["bodies"], p["G"], p["dt"], 1)
    score, notes = 0, []
    for s0, s1, r1 in zip(start, end, ref):
        got = (s1["vx"] - s0["vx"], s1["vy"] - s0["vy"])
        exp = (r1["vx"] - s0["vx"], r1["vy"] - s0["vy"])
        ok = vec_rel(got, exp) < 1e-6
        score += 3 * ok
        notes.append(f"{s0['name']}: dv relerr={vec_rel(got, exp):.2e} {'OK' if ok else 'FAIL'}")
    speeds = [math.hypot(s["vx"], s["vy"]) for s in end]
    sym = max(speeds) > 0 and (max(speeds) - min(speeds)) / max(speeds) < 1e-9
    inward = all(s["vx"] * -b["x"] + s["vy"] * -b["y"] > 0 for s, b in zip(end, p["bodies"]))
    score += 3 * sym + 3 * inward
    notes.append(f"equal speeds={'OK' if sym else 'FAIL'}, pointing to centre={'OK' if inward else 'FAIL'}")
    return score, "\n".join(notes)


def score_t8(o, p):
    pts, notes = (6, 4), []
    earned = 0
    for got, cfg, w in zip(o["energies"], p["configs"], pts):
        exp = ref_energy(cfg["bodies"], cfg["G"])
        ok = rel(got, exp) < 1e-9
        earned += w * ok
        notes.append(f"{len(cfg['bodies'])}-body: relerr={rel(got, exp):.2e} {'OK' if ok else 'FAIL'}")
    return earned, "\n".join(notes)


def score_t9(o, p):
    ref = ref_step(p["second"]["bodies"], G, 1.0, 1)
    fresh_ok = len(o["state"]) == 2 and all(
        vec_rel((s["x"], s["y"]), (r["x"], r["y"])) < 1e-9 and vec_rel((s["vx"], s["vy"]), (r["vx"], r["vy"])) < 1e-6
        for s, r in zip(o["state"], ref))
    return 2 * o["empty"] + 3 * fresh_ok, f"empty after reset={o['empty']}, clean restart={'OK' if fresh_ok else 'FAIL'}"


def score_t10(o, p):
    states = o["states"]
    masses = p["bodies"]
    g = p["G"]
    ref_final = ref_step(masses, g, p["dt"], sum(p["chunks"]))
    final = with_masses(states[-1], masses)
    sep = math.hypot(masses[1]["x"] - masses[0]["x"], masses[1]["y"] - masses[0]["y"])
    traj_err = max(math.hypot(f["x"] - r["x"], f["y"] - r["y"]) / sep for f, r in zip(final, ref_final))
    traj_ok = traj_err < 1e-6
    e0 = ref_energy(with_masses(states[0], masses), g)
    drift = max(rel(ref_energy(with_masses(s, masses), g), e0) for s in states[1:])
    drift_ok = drift < 0.01
    # total_energy() must agree with the state at BOTH ends (catches cached values).
    honest = (rel(o["energy_start"], e0) < 1e-9
              and rel(o["energy_end"], ref_energy(final, g)) < 1e-9)
    earned = 4 * traj_ok + 4 * drift_ok + 2 * honest
    return earned, (f"trajectory vs reference: relerr={traj_err:.2e} {'OK' if traj_ok else 'FAIL'}\n"
                    f"independent energy drift: {drift * 100:.4f}% {'OK' if drift_ok else 'FAIL'}\n"
                    f"total_energy() matches state at start and end: {'OK' if honest else 'FAIL'}")


def score_t11(o, p):
    earned, notes = 0, []
    for got, case, (full, part) in zip(o["periods"], p["cases"], ((5, 2), (10, 5))):
        exp = kepler_period(case["bodies"][0], case["bodies"][1], case["G"])
        err = rel(got, exp)
        pts = full if err < 0.05 else part if err < 0.20 else 0
        earned += pts
        notes.append(f"{case['bodies'][1]['name']} system: err={err:.2%} -> {pts}/{full}")
    return earned, "\n".join(notes)


def score_t12(o, p):
    earned, notes = 0, []
    for got, case, w in zip(o["coms"], p["cases"], (5, 6, 4)):
        bs = case["bodies"]
        m = sum(b["m"] for b in bs)
        exp = (sum(b["m"] * b["x"] for b in bs) / m, sum(b["m"] * b["y"] for b in bs) / m)
        ok = vec_rel(got, exp) < 1e-9
        earned += w * ok
        notes.append(f"{len(bs)}-body CoM: {'OK' if ok else 'FAIL'}")
    return earned, "\n".join(notes)


TESTS = [
    ("T1", "Instantiation & API surface", 5, score_t1),
    ("T2", "add_body / get_state ordering and values", 10, score_t2),
    ("T3", "ValueError on 6th body", 5, score_t3),
    ("T4", "ValueError on step with <2 bodies", 5, score_t4),
    ("T5", "ZeroDivisionError on coincident bodies", 5, score_t5),
    ("T6", "Semi-implicit Euler accuracy (randomized, large dt)", 25, score_t6),
    ("T7", "3-body symmetric forces (randomized)", 15, score_t7),
    ("T8", "total_energy() correctness", 10, score_t8),
    ("T9", "reset() gives a clean simulator", 5, score_t9),
    ("T10", "Long-run stability, energy checked independently", 10, score_t10),
    ("T11", "orbital_period(name)", 15, score_t11),
    ("T12", "center_of_mass()", 15, score_t12),
]


def main(argv):
    if len(argv) < 2:
        print("Usage: python3 -I verifier.py <path_to_simulator.py> [--reward-file P] [--details-file P]")
        return 2
    sim_path = os.path.abspath(argv[1])
    opts = dict(zip(argv[2::2], argv[3::2]))
    if not os.path.isfile(sim_path):
        print(f"ERROR: file not found: {sim_path}")
        return 2

    seed = int(os.environ.get("VERIFIER_SEED", secrets.randbits(32)))
    scenarios = make_scenarios(random.Random(seed))

    results = []
    timed_out = False
    for tid, name, pts, scorer in TESTS:
        if timed_out:
            results.append({"id": tid, "name": name, "points": pts, "earned": 0,
                            "detail": "Skipped: an earlier test timed out."})
            continue
        out = run_worker(sim_path, tid, scenarios[tid])
        timed_out = out.get("error", "").startswith("Timed out")
        if "obs" in out:
            try:
                earned, detail = scorer(out["obs"], scenarios[tid])
            except Exception as e:  # malformed observations from agent code
                earned, detail = 0, f"Could not score observations: {type(e).__name__}: {e}"
        else:
            earned = 0
            detail = out.get("load_error") or out.get("error", "unknown error")
        results.append({"id": tid, "name": name, "points": pts, "earned": int(min(earned, pts)), "detail": detail})

    total = sum(r["earned"] for r in results)
    max_total = sum(r["points"] for r in results)
    gates_failed = [r["id"] for r in results if r["id"] in GATING_TESTS and r["earned"] < r["points"]]
    passed = total >= PASS_THRESHOLD and not gates_failed

    print("=" * 64)
    print(f"  GRAVITY SIMULATOR VERIFIER v2   (seed={seed})")
    print("=" * 64)
    for r in results:
        status = "PASS" if r["earned"] == r["points"] else ("PART" if r["earned"] else "FAIL")
        print(f"\n[{status}] {r['id']}: {r['name']} [{r['earned']}/{r['points']}]")
        for line in str(r["detail"]).split("\n"):
            print(f"       {line}")
    print("\n" + "=" * 64)
    print(f"  SCORE   : {total}/{max_total}  (threshold {PASS_THRESHOLD})")
    print(f"  GATES   : {', '.join(GATING_TESTS)} must be full marks -> "
          f"{'OK' if not gates_failed else 'FAILED ' + ', '.join(gates_failed)}")
    print(f"  VERDICT : {'PASSED' if passed else 'FAILED'}")
    print("=" * 64)

    if "--reward-file" in opts:
        with open(opts["--reward-file"], "w") as f:
            f.write("1" if passed else "0")
    if "--details-file" in opts:
        with open(opts["--details-file"], "w") as f:
            json.dump({"seed": seed, "score": total, "max": max_total, "passed": passed,
                       "gates_failed": gates_failed, "tests": results}, f, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
