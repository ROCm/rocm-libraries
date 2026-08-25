#!/usr/bin/env python3
"""Does `--algo_method all`'s top-K contain the TRUE (single-dispatch) winner?

This decides whether a practical re-tune exists. `--solution_index` is the valid ranking
instrument (r=+0.989 vs the catalog benchmark) but costs one process per (kernel, shape), so a
full 9680 x 298 sweep is infeasible. The obvious rescue is two-stage: shortlist cheaply with
`--algo_method all`, then rank only the shortlist single-dispatch.

That rescue has a specific failure mode: `--algo_method all` overstates throughput
KERNEL-DEPENDENTLY (1.33x for one kernel, 2.11x for another, bias ratio 1.16x), so its ordering
is not merely noisy -- it is systematically wrong. If the true winner is not in its top-K, the
two-stage scheme cannot work at any K worth using.

Method: on a handful of shapes, measure ALL 298 kernels ONE PROCESS AT A TIME, and ask where the
single-dispatch winner sits in the enumeration's ranking.
"""
import json, os, re, subprocess, sys, statistics, collections

BENCH = "/home/vmijovic/navi32/src/projects/hipblaslt/build/release/clients/hipblaslt-bench"
LIB = "/home/vmijovic/navi32/libs/wgm8/library/gfx1100"
LOCK = "/home/vmijovic/navi32/.gpu.lock"
ROW = re.compile(r"^\s*T,N,(.*)$", re.M)


def single(m, n, k, si, it):
    cmd = ["flock", "-w", "600", LOCK, BENCH, "--api_method", "c",
           "-m", str(m), "-n", str(n), "-k", str(k), "--transA", "T", "--transB", "N",
           "--lda", str(k), "--ldb", str(k), "--ldc", str(m), "--ldd", str(m),
           "--a_type", "f16_r", "--b_type", "f16_r", "--c_type", "f16_r", "--d_type", "f16_r",
           "--compute_type", "f32_r", "--algo_method", "index", "--solution_index", str(si),
           "--initialization", "trig_float",
           "--cold_iters", str(max(1, it // 3)), "--iters", str(it),
           "--sm_count_target", "60", "--flush", "--rotating", "512"]
    env = dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=LIB, HIPBLASLT_BENCH_CU_MASK="60")
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "hipblaslt-bench"], capture_output=True)
        return None
    best = None
    for mm in ROW.finditer(p.stdout):
        try:
            best = float(mm.group(1).split(",")[-3])
        except (ValueError, IndexError):
            pass
    return best if best and best > 0 else None


def main():
    shapes = json.load(open(sys.argv[1]))
    enum = {}
    for line in open("/home/vmijovic/navi32/results/P1_cold_matrix.jsonl"):
        d = json.loads(line)
        enum[(d["M"], d["N"], d["B"], d["K"])] = {int(a): v for a, v in d["gf"].items() if v > 0}
    out = open("/home/vmijovic/navi32/results/P14_poc_rows.jsonl", "a")
    for s in shapes:
        key = (s["M"], s["N"], s["B"], s["K"])
        e = enum.get(key)
        if not e:
            continue
        est = s.get("est_us") or 25.0
        it = int(min(400, max(6, 10000 / est)))
        sd = {}
        for si in sorted(e):
            v = single(s["M"], s["N"], s["K"], si, it)
            if v:
                sd[si] = v
        if len(sd) < 50:
            continue
        out.write(json.dumps({"M": s["M"], "N": s["N"], "B": s["B"], "K": s["K"],
                              "stratum": s.get("stratum", "?"), "sd": sd}) + "\n")
        out.flush()
        # where does the single-dispatch winner sit in the enumeration's ranking?
        win = max(sd, key=sd.get)
        order = sorted(sd, key=lambda i: -e[i])
        rank = order.index(win) + 1
        print(f"  {s['M']}x{s['N']}x{s['K']} ({s.get('stratum')}): "
              f"true winner is rank {rank}/{len(sd)} in the enumeration; "
              f"enum's own pick delivers {100*sd[order[0]]/sd[win]:.1f}% of achievable",
              flush=True)
    out.close()


if __name__ == "__main__":
    main()
