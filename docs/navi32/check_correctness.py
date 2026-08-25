#!/usr/bin/env python3
"""Numerical correctness check for a built catalog.

WHY THIS IS NOT A grep. The first version of this check tested `grep -qiE "PASS|pass"` against
hipblaslt-bench -v output. That word never appears: -v emits numeric `norm_error,atol,rtol`
columns instead. So the gate reported 0 pass / 26 fail on a catalog whose every run actually
succeeded (norm_error ~5e-05 against rtol 1e-02). A check that greps for a token the tool does
not emit fails 100% of the time and looks exactly like a real defect.

So: parse the CSV row the tool actually prints, and compare norm_error against rtol.
"""
import argparse, json, os, re, subprocess, sys

BENCH = "/home/vmijovic/navi32/src/projects/hipblaslt/build/release/clients/hipblaslt-bench"
LOCK = "/home/vmijovic/navi32/.gpu.lock"


def run(lib, m, n, k, timeout=120):
    cmd = ["flock", "-w", "600", LOCK, BENCH, "--api_method", "c",
           "-m", str(m), "-n", str(n), "-k", str(k), "--transA", "T", "--transB", "N",
           "--lda", str(k), "--ldb", str(k), "--ldc", str(m), "--ldd", str(m),
           "--a_type", "f16_r", "--b_type", "f16_r", "--c_type", "f16_r", "--d_type", "f16_r",
           "--compute_type", "f32_r", "--algo_method", "heuristic", "--requested_solution", "1",
           "--initialization", "trig_float", "--cold_iters", "1", "--iters", "1", "-v"]
    env = dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=lib)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "hipblaslt-bench"], capture_output=True)
        return None, "timeout"
    hdr = None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith("transA,transB") or ",norm_error," in s:
            hdr = [c.strip() for c in s.split(",")]
        elif hdr and s and (s[0].isalpha() or s[0].isdigit()) and s.count(",") == len(hdr) - 1:
            row = [c.strip() for c in s.split(",")]
            d = dict(zip(hdr, row))
            if "norm_error" in d:
                try:
                    return float(d["norm_error"]), float(d.get("rtol", 0.01))
                except ValueError:
                    pass
    return None, "no norm_error row"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--shapes", required=True, help="json {'shapes':[...]} or a bare list")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    d = json.load(open(a.shapes))
    shapes = d["shapes"] if isinstance(d, dict) else d
    seen, sel = set(), []
    for s in shapes:                      # spread over strata, not just the head of the list
        st = s.get("stratum", "?")
        if st in seen and len(sel) > a.n // 2:
            continue
        seen.add(st); sel.append(s)
        if len(sel) >= a.n:
            break
    npass = nfail = 0
    for s in sel:
        err, tol = run(a.lib, s["M"], s["N"], s["K"])
        if err is None:
            nfail += 1
            print(f"    FAIL m={s['M']} n={s['N']} k={s['K']}: {tol}")
        elif err <= tol:
            npass += 1
        else:
            nfail += 1
            print(f"    FAIL m={s['M']} n={s['N']} k={s['K']}: norm_error={err:g} > rtol={tol:g}")
    print(f"  correctness: {npass} pass / {nfail} fail  (of {len(sel)} run)")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
