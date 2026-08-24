#!/usr/bin/env python3
"""Phase 0 instrument: measure the full shape x kernel matrix with --algo_method all.

One bench call per shape returns every solution's GFlop/s on that shape (298 rows for
the navi31 HHS pool). That lets ANY candidate catalog be evaluated offline -- no need to
build and benchmark each policy separately.

Instrument noise measured on this machine at 60 CU: iters=20 gives median 0.50% / p90
3.03% spread over 3 repeats (iters=5 was p90 11.7%, iters=60 bought ~0.7pt for 3x cost).

Resumable: one JSON line per shape, skips shapes already present.
"""
import argparse, json, os, random, re, subprocess, sys, time

BENCH = "/home/vmijovic/navi32/src/projects/hipblaslt/build/release/clients/hipblaslt-bench"
LOCK = "/home/vmijovic/navi32/.gpu.lock"
LIB = "/home/vmijovic/navi32/libs/wgm8/library/gfx1100"
ROW = re.compile(r"^\[(\d+)\]:.*\n\s*T,N,([^\n]*)", re.M)


def sweep_one(m, n, k, b, cus, iters, timeout):
    cmd = [BENCH, "--api_method", "c", "-m", str(m), "-n", str(n), "-k", str(k),
           "--transA", "T", "--transB", "N", "--lda", str(k), "--ldb", str(k),
           "--ldc", str(m), "--ldd", str(m),
           "--a_type", "f16_r", "--b_type", "f16_r", "--c_type", "f16_r",
           "--d_type", "f16_r", "--compute_type", "f32_r",
           "--algo_method", "all", "--initialization", "trig_float",
           "--cold_iters", str(max(1, iters // 3)), "--iters", str(iters),
           "--sm_count_target", str(cus)]
    if b and b > 1:
        cmd += ["--batch_count", str(b)]
    env = dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=LIB,
               HIPBLASLT_BENCH_CU_MASK=str(cus))
    try:
        p = subprocess.run(["flock", "-w", "600", LOCK] + cmd, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "hipblaslt-bench"], capture_output=True)
        return None
    out = {}
    for mm in ROW.finditer(p.stdout):
        si = int(mm.group(1))
        try: gf = float(mm.group(2).split(",")[-3])
        except (ValueError, IndexError): continue
        # index appears twice for the winner replay; keep the FIRST (the enumerated run)
        if si not in out and gf > 0:
            out[si] = gf
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True, help="json list of {M,N,B,K,stratum}")
    ap.add_argument("--out", default="/home/vmijovic/navi32/results/P0_matrix.jsonl")
    ap.add_argument("--cus", type=int, default=60)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    shapes = json.load(open(a.shapes))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line); done.add((d["M"], d["N"], d["B"], d["K"]))
            except Exception: pass
    todo = [s for s in shapes if (s["M"], s["N"], s["B"], s["K"]) not in done]
    print(f"{len(shapes)} shapes, {len(done)} done, {len(todo)} to go", flush=True)

    fh = open(a.out, "a")
    t0 = time.time(); ok = bad = 0
    for i, s in enumerate(todo, 1):
        # very large problems: fewer iters, they are already far above the noise floor
        it = a.iters if s["M"] * s["N"] * s["K"] * max(s["B"], 1) < 2e10 else max(6, a.iters // 3)
        r = sweep_one(s["M"], s["N"], s["K"], s["B"], a.cus, it, a.timeout)
        if r:
            ok += 1
            fh.write(json.dumps({**{k: s[k] for k in ("M", "N", "B", "K", "stratum")},
                                 "iters": it, "gf": r}) + "\n")
        else:
            bad += 1
        fh.flush()
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {el/60:.1f}m ok={ok} err={bad} "
                  f"eta {el/i*(len(todo)-i)/60:.0f}m", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
