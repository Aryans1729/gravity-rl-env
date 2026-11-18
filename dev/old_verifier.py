import sys
import math
import importlib.util
import traceback
from pathlib import Path

def load_agent_module(path):
    spec = importlib.util.spec_from_file_location("agent_simulator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rel_error(got, expected):
    if expected == 0:
        return abs(got)
    return abs(got - expected) / abs(expected)

def approx(got, expected, rtol=1e-5):
    return rel_error(got, expected) <= rtol

G = 6.674e-11
M_SUN = 1.989e30
M_EARTH = 5.972e24
X_EARTH = 1.496e11

_r = X_EARTH
_F = G * M_SUN * M_EARTH / _r**2
_dvx_earth = (-_F / M_EARTH) * 1.0
_vx_earth_1 = _dvx_earth
_vy_earth_1 = 29780.0
_x_earth_1 = X_EARTH + _vx_earth_1 * 1.0
_y_earth_1 = 0.0 + _vy_earth_1 * 1.0
_dvx_sun = (_F / M_SUN) * 1.0
_vx_sun_1 = _dvx_sun
_x_sun_1 = 0.0 + _vx_sun_1 * 1.0
_KE0 = 0.5 * M_EARTH * 29780.0**2
_PE0 = -G * M_SUN * M_EARTH / X_EARTH
_E0 = _KE0 + _PE0

results = []

class VerifierResult:
    def __init__(self, name, points, earned, detail):
        self.name = name
        self.points = points
        self.earned = earned
        self.detail = detail

def test(name, points, fn):
    try:
        earned, detail = fn()
    except Exception as e:
        earned = 0
        detail = f"EXCEPTION: {traceback.format_exc(limit=3)}"
    results.append(VerifierResult(name, points, earned, detail))

def run_all_tests(Cls):
    def t1():
        sim = Cls(G=6.674e-11, dt=1.0)
        assert hasattr(sim, "add_body")
        assert hasattr(sim, "step")
        assert hasattr(sim, "get_state")
        assert hasattr(sim, "total_energy")
        assert hasattr(sim, "reset")
        return 5, "All required methods present."
    test("T1: Instantiation & API surface", 5, t1)

    def t2():
        sim = Cls(G=6.674e-11, dt=1.0)
        sim.add_body("A", 1e24, 0.0, 0.0)
        sim.add_body("B", 2e24, 1e9, 0.0)
        sim.add_body("C", 3e24, 0.0, 1e9)
        state = sim.get_state()
        assert len(state) == 3
        assert state[0]["name"] == "A"
        assert state[1]["name"] == "B"
        assert state[2]["name"] == "C"
        return 10, "Insertion order preserved."
    test("T2: add_body / get_state ordering", 10, t2)

    def t3():
        sim = Cls()
        for i in range(5):
            sim.add_body(f"P{i}", 1e20, float(i)*1e9, 0.0)
        try:
            sim.add_body("Extra", 1e20, 99e9, 0.0)
            return 0, "FAIL: No ValueError raised for 6th body."
        except ValueError:
            return 5, "Correctly raised ValueError on 6th body."
    test("T3: ValueError on 6th body", 5, t3)

    def t4():
        sim = Cls()
        sim.add_body("Lone", 1e24, 0.0, 0.0)
        try:
            sim.step(1)
            return 0, "FAIL: No ValueError raised."
        except ValueError:
            return 5, "Correctly raised ValueError."
    test("T4: ValueError on step with <2 bodies", 5, t4)

    def t5():
        sim = Cls()
        sim.add_body("X", 1e24, 5e9, 5e9)
        sim.add_body("Y", 1e24, 5e9, 5e9)
        try:
            sim.step(1)
            return 0, "FAIL: No ZeroDivisionError."
        except ZeroDivisionError:
            return 5, "Correctly raised ZeroDivisionError."
    test("T5: ZeroDivisionError on coincident bodies", 5, t5)

    def t6():
        sim = Cls(G=6.674e-11, dt=1.0)
        sim.add_body("Sun", M_SUN, 0.0, 0.0)
        sim.add_body("Earth", M_EARTH, X_EARTH, 0.0, vy=29780.0)
        sim.step(1)
        st = sim.get_state()
        sun, earth = st[0], st[1]
        checks = [
            ("earth.x", earth["x"], _x_earth_1, 1e-6),
            ("earth.y", earth["y"], _y_earth_1, 1e-6),
            ("earth.vx", earth["vx"], _vx_earth_1, 1e-6),
            ("earth.vy", earth["vy"], _vy_earth_1, 1e-8),
            ("sun.x", sun["x"], _x_sun_1, 1e-4),
            ("sun.vx", sun["vx"], _vx_sun_1, 1e-4),
        ]
        passed = 0
        notes = []
        for label, got, exp, tol in checks:
            err = rel_error(got, exp) if exp != 0 else abs(got - exp)
            ok = err <= tol
            notes.append(f"  {label}: got={got:.6e} exp={exp:.6e} relerr={err:.2e} {'OK' if ok else 'FAIL'}")
            if ok:
                passed += 1
        earned = round(25 * passed / len(checks))
        return earned, f"{passed}/{len(checks)} checks passed.\n" + "\n".join(notes)
    test("T6: Single-step Euler accuracy", 25, t6)

    def t7():
        sim = Cls(G=6.674e-11, dt=1.0)
        m = 1e26
        R = 1e9
        positions = [(R, 0.0), (-R/2, R*math.sqrt(3)/2), (-R/2, -R*math.sqrt(3)/2)]
        for i, (x, y) in enumerate(positions):
            sim.add_body(f"P{i}", m, x, y)
        sim.step(1)
        state = sim.get_state()
        score = 0
        notes = []
        for i, b in enumerate(state):
            ox, oy = positions[i]
            spd = math.sqrt(b["vx"]**2 + b["vy"]**2)
            dot = b["vx"] * (-ox) + b["vy"] * (-oy)
            ok = spd > 0 and dot > 0
            notes.append(f"  P{i}: speed={spd:.3e}, dot={dot:.3e} {'OK' if ok else 'FAIL'}")
            if ok:
                score += 1
        side = R * math.sqrt(3)
        F_pair = G * m * m / side**2
        F_net = F_pair * math.sqrt(3)
        v_exp = (F_net / m) * 1.0
        spd0 = math.sqrt(state[0]["vx"]**2 + state[0]["vy"]**2)
        mag_ok = rel_error(spd0, v_exp) < 1e-4
        notes.append(f"  Magnitude: got={spd0:.6e} exp={v_exp:.6e} {'OK' if mag_ok else 'FAIL'}")
        if mag_ok:
            score += 2
        earned = round(15 * score / 5)
        return earned, "\n".join(notes)
    test("T7: 3-body symmetric force check", 15, t7)

    def t8():
        sim = Cls(G=6.674e-11, dt=1.0)
        sim.add_body("Sun", M_SUN, 0.0, 0.0)
        sim.add_body("Earth", M_EARTH, X_EARTH, 0.0, vy=29780.0)
        E = sim.total_energy()
        err = rel_error(E, _E0)
        ok = err < 1e-6
        return (10 if ok else 0), f"energy={E:.6e}, expected={_E0:.6e}, relerr={err:.2e}"
    test("T8: total_energy at t=0", 10, t8)

    def t9():
        sim = Cls()
        sim.add_body("A", 1e24, 0.0, 0.0)
        sim.add_body("B", 1e24, 1e9, 0.0)
        sim.step(10)
        sim.reset()
        state = sim.get_state()
        assert len(state) == 0
        sim.add_body("New", 1e24, 0.0, 0.0)
        assert len(sim.get_state()) == 1
        return 5, "Reset works correctly."
    test("T9: reset()", 5, t9)

    def t10():
        sim = Cls(G=6.674e-11, dt=10.0)
        sim.add_body("Sun", M_SUN, 0.0, 0.0)
        sim.add_body("Earth", M_EARTH, X_EARTH, 0.0, vy=29780.0)
        E0 = sim.total_energy()
        sim.step(1000)
        E1 = sim.total_energy()
        drift = rel_error(E1, E0)
        ok = drift < 0.01
        return (10 if ok else 5 if drift < 0.1 else 0), f"Energy drift: {drift*100:.4f}%"
    test("T10: 1000-step energy stability", 10, t10)

    def t11():
        import math as _m
        _G = 6.674e-11
        earned = 0
        notes = []
        try:
            # Sub-test 1: Sun-Earth orbital period (5 pts)
            # Correct answer: ~3.156e7 s (1 year), well-known result
            sim = Cls(G=_G, dt=1.0)
            sim.add_body("Sun",   1.989e30, 0.0,      0.0, vx=0.0, vy=0.0)
            sim.add_body("Earth", 5.972e24, 1.496e11, 0.0, vx=0.0, vy=29780.0)
            T_earth = sim.orbital_period("Earth")
            T_1yr = 3.156e7
            err1 = abs(T_earth - T_1yr) / T_1yr
            if err1 < 0.05:
                earned += 5
                notes.append(f"  Earth period: {T_earth:.4e}s err={err1:.2%}: OK")
            elif err1 < 0.20:
                earned += 2
                notes.append(f"  Earth period: {T_earth:.4e}s err={err1:.2%}: PARTIAL")
            else:
                notes.append(f"  Earth period: {T_earth:.4e}s err={err1:.2%}: FAIL")

            # Sub-test 2: equal-mass circular orbit (10 pts)
            # Alpha(-1e10,0,vy=+v) and Beta(+1e10,0,vy=-v), each m=1e26 kg
            # Full separation R = 2e10 m.
            # Correct Kepler: T = 2π * sqrt(R³ / (G * M_total)) ≈ 1.538e8 s
            # Common wrong approach (using r_CoM = R/2 instead of R):
            #   gives T ≈ 5.44e7 s, off by factor 1/(2√2) ≈ 0.354
            _m_body = 1e26
            _R = 2e10
            _v = _m.sqrt(_G * _m_body / (2.0 * _R))   # circular orbit speed ≈ 408.47 m/s
            _T_exp = 2 * _m.pi * _m.sqrt(_R**3 / (_G * 2 * _m_body))  # ≈ 1.538e8 s
            sim2 = Cls(G=_G, dt=1.0)
            sim2.add_body("Alpha", _m_body, -_R / 2, 0.0, vx=0.0, vy=+_v)
            sim2.add_body("Beta",  _m_body, +_R / 2, 0.0, vx=0.0, vy=-_v)
            T_alpha = sim2.orbital_period("Alpha")
            err2 = abs(T_alpha - _T_exp) / _T_exp
            if err2 < 0.05:
                earned += 10
                notes.append(f"  Equal-mass orbit: {T_alpha:.4e}s err={err2:.2%}: OK")
            elif err2 < 0.20:
                earned += 5
                notes.append(f"  Equal-mass orbit: {T_alpha:.4e}s err={err2:.2%}: PARTIAL")
            else:
                # Deliberately omit expected value to keep the fix non-obvious
                notes.append(f"  Equal-mass orbit: {T_alpha:.4e}s: FAIL")
        except AttributeError:
            return 0, "  FAIL: orbital_period method not found on GravitySimulator"
        except ValueError as e:
            notes.append(f"  ValueError: {e}")
        except Exception as e:
            notes.append(f"  EXCEPTION: {traceback.format_exc(limit=2)}")
        return earned, "\n".join(notes)
    test("T11: orbital_period(name)", 15, t11)

    def t12():
        earned = 0
        notes = []
        try:
            # Sub-test 1: 2-body CoM with unequal masses
            sim = Cls()
            sim.add_body("A", 1e24, 0.0, 0.0)
            sim.add_body("B", 3e24, 4e9, 0.0)
            # CoM_x = (0*1e24 + 4e9*3e24) / 4e24 = 3e9,  CoM_y = 0
            x, y = sim.center_of_mass()
            if rel_error(x, 3e9) < 1e-9 and abs(y) < 1.0:
                earned += 5
                notes.append(f"  2-body CoM: OK (x={x:.3e}, y={y:.3e})")
            else:
                notes.append(
                    f"  2-body CoM: FAIL x={x:.3e} (exp 3.000e+09), y={y:.3e} (exp ~0)"
                )
            # Sub-test 2: 3-body CoM with non-trivial y component
            sim2 = Cls()
            sim2.add_body("X", 2e24, 0.0, 0.0)
            sim2.add_body("Y", 2e24, 6e9, 0.0)
            sim2.add_body("Z", 4e24, 3e9, 6e9)
            # total_mass = 8e24
            # CoM_x = (0 + 12e33 + 12e33) / 8e24 = 3e9
            # CoM_y = (0 +     0 + 24e33) / 8e24 = 3e9
            x2, y2 = sim2.center_of_mass()
            if rel_error(x2, 3e9) < 1e-9 and rel_error(y2, 3e9) < 1e-9:
                earned += 6
                notes.append(f"  3-body CoM: OK (x={x2:.3e}, y={y2:.3e})")
            else:
                notes.append(
                    f"  3-body CoM: FAIL x={x2:.3e} (exp 3e9), y={y2:.3e} (exp 3e9)"
                )
            # Sub-test 3: single-body CoM (trivial case)
            sim3 = Cls()
            sim3.add_body("Solo", 5e24, 7e9, 3e9)
            x3, y3 = sim3.center_of_mass()
            if rel_error(x3, 7e9) < 1e-9 and rel_error(y3, 3e9) < 1e-9:
                earned += 4
                notes.append(f"  1-body CoM: OK (x={x3:.3e}, y={y3:.3e})")
            else:
                notes.append(
                    f"  1-body CoM: FAIL x={x3:.3e} (exp 7e9), y={y3:.3e} (exp 3e9)"
                )
        except AttributeError:
            return 0, "  FAIL: center_of_mass method not found on GravitySimulator"
        except Exception as e:
            notes.append(f"  EXCEPTION: {traceback.format_exc(limit=2)}")
        return earned, "\n".join(notes)
    test("T12: center_of_mass()", 15, t12)

def print_report():
    total_pts = sum(r.points for r in results)
    total_earn = sum(r.earned for r in results)
    pct = 100 * total_earn / total_pts if total_pts else 0
    print("=" * 60)
    print("  GRAVITY SIMULATOR — VERIFIER REPORT")
    print("=" * 60)
    for r in results:
        status = "PASS" if r.earned == r.points else ("PART" if r.earned > 0 else "FAIL")
        print(f"\n[{status}] {r.name} [{r.earned}/{r.points} pts]")
        for line in r.detail.split("\n"):
            print(f"       {line}")
    print("\n" + "=" * 60)
    print(f"  TOTAL SCORE : {total_earn}/{total_pts} ({pct:.1f}%)")
    verdict = "PASSED" if total_earn >= 122 else "FAILED"
    print(f"  VERDICT     : {verdict} (threshold: 122/{total_pts})")
    print("=" * 60)
    return total_earn >= 122

def main():
    if len(sys.argv) < 2:
        print("Usage: python verifier.py <path_to_simulator.py>")
        sys.exit(2)
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(2)
    try:
        mod = load_agent_module(path)
        Cls = getattr(mod, "GravitySimulator")
    except AttributeError:
        print("ERROR: 'GravitySimulator' class not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading module: {e}")
        sys.exit(1)
    run_all_tests(Cls)
    passed = print_report()
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
