"""Worker for the gravity verifier.

Runs the agent's simulator in an isolated process and reports raw
observations only. It contains NO reference physics and NO scoring,
so agent code running here has nothing to borrow or patch.

Invoked by verifier.py as:  python3 -I worker.py <simulator.py>   (job JSON on stdin)
"""

import json
import os
import sys

G = 6.674e-11

# ==========================================================================
# WORKER SIDE  (runs agent code; reports raw observations only)
# ==========================================================================


def _worker_main(sim_path):
    import importlib.util
    import traceback

    # Read the job and set up the private result channel BEFORE agent code runs.
    job = json.loads(sys.stdin.read())
    nonce = job["nonce"]
    result_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    def emit(payload):
        os.write(result_fd, (nonce + json.dumps(payload) + "\n").encode())

    try:
        spec = importlib.util.spec_from_file_location("agent_simulator", sim_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Cls = getattr(mod, "GravitySimulator")
    except Exception as e:  # noqa: BLE001
        emit({"load_error": f"{type(e).__name__}: {e}"})
        return

    try:
        obs = WORKER_TESTS[job["test"]](Cls, job["params"])
        emit({"obs": obs})
    except Exception as e:  # noqa: BLE001
        emit({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)})


def _state(sim):
    return [
        {k: (float(b[k]) if k != "name" else str(b[k])) for k in ("name", "x", "y", "vx", "vy")}
        for b in sim.get_state()
    ]


def _expect_raise(fn):
    try:
        fn()
        return "none"
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def _new(Cls, p):
    sim = Cls(G=p["G"], dt=p["dt"])
    for b in p["bodies"]:
        sim.add_body(b["name"], b["m"], b["x"], b["y"], vx=b["vx"], vy=b["vy"])
    return sim


def w_t1(Cls, p):
    sim = Cls(G=G, dt=1.0)
    names = ["add_body", "step", "get_state", "total_energy", "reset", "orbital_period", "center_of_mass"]
    return {n: callable(getattr(sim, n, None)) for n in names}


def w_t2(Cls, p):
    return {"state": _state(_new(Cls, p))}


def w_t3(Cls, p):
    sim = Cls(G=G, dt=1.0)
    for i in range(5):
        sim.add_body(f"P{i}", 1e20, float(i + 1) * 1e9, 0.0)
    return {"raised": _expect_raise(lambda: sim.add_body("Extra", 1e20, 99e9, 0.0)),
            "count": len(sim.get_state())}


def w_t4(Cls, p):
    empty = Cls(G=G, dt=1.0)
    lone = Cls(G=G, dt=1.0)
    lone.add_body("Lone", 1e24, 0.0, 0.0)
    return {"empty": _expect_raise(lambda: empty.step(1)), "one": _expect_raise(lambda: lone.step(1))}


def w_t5(Cls, p):
    return {"raised": _expect_raise(lambda: _new(Cls, p).step(1))}


def w_step_states(Cls, p):
    """Run step(n) for each n in p['chunks']; record state after each chunk."""
    sim = _new(Cls, p)
    states = [_state(sim)]
    energy_start = float(sim.total_energy()) if p.get("energy_at_end") else None
    for n in p["chunks"]:
        sim.step(n)
        states.append(_state(sim))
    out = {"states": states}
    if p.get("energy_at_end"):
        out["energy_start"] = energy_start
        out["energy_end"] = float(sim.total_energy())
    return out


def w_t8(Cls, p):
    return {"energies": [float(_new(Cls, cfg).total_energy()) for cfg in p["configs"]]}


def w_t9(Cls, p):
    sim = _new(Cls, p["first"])
    sim.step(10)
    sim.reset()
    empty_after_reset = len(sim.get_state()) == 0
    for b in p["second"]["bodies"]:
        sim.add_body(b["name"], b["m"], b["x"], b["y"], vx=b["vx"], vy=b["vy"])
    sim.step(1)
    return {"empty": empty_after_reset, "state": _state(sim)}


def w_t11(Cls, p):
    out = []
    for case in p["cases"]:
        out.append(float(_new(Cls, case).orbital_period(case["query"])))
    return {"periods": out}


def w_t12(Cls, p):
    out = []
    for case in p["cases"]:
        x, y = _new(Cls, case).center_of_mass()
        out.append([float(x), float(y)])
    return {"coms": out}


WORKER_TESTS = {
    "T1": w_t1, "T2": w_t2, "T3": w_t3, "T4": w_t4, "T5": w_t5,
    "T6": w_step_states, "T7": w_step_states, "T8": w_t8, "T9": w_t9,
    "T10": w_step_states, "T11": w_t11, "T12": w_t12,
}


if __name__ == "__main__":
    _worker_main(sys.argv[1])
