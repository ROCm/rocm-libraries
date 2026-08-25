#!/usr/bin/env python3
"""Is the `--algo_method all` ranking bias caused by the 60-CU EMULATION, or intrinsic?

This is the question that decides whether the campaign can simply be redone on real navi32.

Everything measured so far used `--sm_count_target 60` plus `HIPBLASLT_BENCH_CU_MASK=60` on a
gfx1100 (96 CU) part. If the enumeration's disagreement with single-dispatch is an artefact of
that emulation -- e.g. the masked stream behaving differently when 298 kernels are dispatched
back-to-back -- then `--algo_method all` may be perfectly sound on native hardware and the whole
approach is salvageable. If the disagreement persists UNMASKED and UNTARGETED, the instrument is
broken independently of the emulation.

For each shape, both instruments are run in both regimes, and the two kernels compared are the
same pair the re-map disagreed about (incumbent vs the enumeration's pick).
"""
import json, os, re, subprocess, statistics, sys
import yaml
try:    from yaml import CSafeLoader as L
except ImportError: from yaml import SafeLoader as L

BENCH = "/home/vmijovic/navi32/src/projects/hipblaslt/build/release/clients/hipblaslt-bench"
LIB = "/home/vmijovic/navi32/libs/wgm8/library/gfx1100"
LOCK = "/home/vmijovic/navi32/.gpu.lock"
ENUM = re.compile(r"^\[(\d+)\]:.*\n\s*T,N,([^\n]*)", re.M)
ROW = re.compile(r"^\s*T,N,(.*)$", re.M)


def base(m, n, k, it):
    return [BENCH, "--api_method", "c", "-m", str(m), "-n", str(n), "-k", str(k),
            "--transA", "T", "--transB", "N", "--lda", str(k), "--ldb", str(k),
            "--ldc", str(m), "--ldd", str(m),
            "--a_type", "f16_r", "--b_type", "f16_r", "--c_type", "f16_r", "--d_type", "f16_r",
            "--compute_type", "f32_r", "--initialization", "trig_float",
            "--cold_iters", str(max(1, it // 3)), "--iters", str(it),
            "--flush", "--rotating", "512"]


def run(cmd, masked):
    env = dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=LIB)
    if masked:
        env["HIPBLASLT_BENCH_CU_MASK"] = "60"
    try:
        return subprocess.run(["flock", "-w", "600", LOCK] + cmd, env=env,
                              capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "hipblaslt-bench"], capture_output=True)
        return ""


def enum_all(m, n, k, it, masked):
    cmd = base(m, n, k, it) + ["--algo_method", "all"]
    if masked:
        cmd += ["--sm_count_target", "60"]
    out, g = run(cmd, masked), {}
    for mm in ENUM.finditer(out):
        si = int(mm.group(1))
        try: v = float(mm.group(2).split(",")[-3])
        except (ValueError, IndexError): continue
        if si not in g and v > 0:
            g[si] = v
    return g


def one(m, n, k, si, it, masked):
    cmd = base(m, n, k, it) + ["--algo_method", "index", "--solution_index", str(si)]
    if masked:
        cmd += ["--sm_count_target", "60"]
    out, best = run(cmd, masked), None
    for mm in ROW.finditer(out):
        try: best = float(mm.group(1).split(",")[-3])
        except (ValueError, IndexError): pass
    return best if best and best > 0 else None


def corr(x, y):
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/n/(sx*sy) if sx and sy else float("nan")


def main():
    ship = yaml.load(open("/home/vmijovic/navi32/arms/hhs_remap_gated/x.yaml"), Loader=L)
    ext = yaml.load(open("/home/vmijovic/navi32/arms/hhs_remap_extship/x.yaml"), Loader=L)
    src = yaml.load(open("/home/vmijovic/navi32/src/projects/hipblaslt/library/src/amd_detail/"
                         "rocblaslt/src/Tensile/Logic/asm_full/navi31/GridBased/"
                         "navi31_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml"), Loader=L)
    ln = {s["SolutionIndex"]: s.get("KernelNameMin") for s in ship[5]}
    sn = {s.get("KernelNameMin"): s["SolutionIndex"] for s in src[5]}
    l2s = {i: sn[n] for i, n in ln.items() if n in sn}
    q = json.load(open("/home/vmijovic/navi32/state/eval_fullcov.json"))["shapes"]
    cand = [s for s in q if ship[7][s["row"]][1][0] != ext[7][s["row"]][1][0]][:int(sys.argv[1])]
    res = {True: ([], []), False: ([], [])}
    for s in cand:
        a = l2s.get(ship[7][s["row"]][1][0]); b = l2s.get(ext[7][s["row"]][1][0])
        it = int(min(400, max(6, 10000/(s.get("est_us") or 25.0))))
        for masked in (True, False):
            g = enum_all(s["M"], s["N"], s["K"], it, masked)
            if a not in g or b not in g: continue
            sa = one(s["M"], s["N"], s["K"], a, it, masked)
            sb = one(s["M"], s["N"], s["K"], b, it, masked)
            if not sa or not sb: continue
            res[masked][0].append(g[b]/g[a])
            res[masked][1].append(sb/sa)
    print(f"{'regime':<34}{'n':>4}{'enum median':>13}{'single med':>12}{'corr':>8}{'sign':>7}")
    print("-"*78)
    for masked, lab in ((True, "MASKED  (sm_count_target+CU mask)"), (False, "UNMASKED (native 96 CU)")):
        e, sd = res[masked]
        if len(e) < 4:
            print(f"{lab:<34}{len(e):>4}   too few"); continue
        agree = 100*sum(1 for x, y in zip(e, sd) if (x > 1) == (y > 1))/len(e)
        print(f"{lab:<34}{len(e):>4}{statistics.median(e):>13.3f}{statistics.median(sd):>12.3f}"
              f"{corr(e, sd):>8.3f}{agree:>6.0f}%")
    print("\nIf UNMASKED agrees but MASKED does not, the emulation is the culprit and the")
    print("approach is salvageable on native navi32. If both disagree, the instrument is broken.")


if __name__ == "__main__":
    main()
