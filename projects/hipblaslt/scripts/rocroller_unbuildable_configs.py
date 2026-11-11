#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Record the rocRoller GEMM configurations that rocRoller fails to build.

A build without rocRoller (HIPBLASLT_ENABLE_ROCROLLER=OFF) walks rocRoller's
ranked tiles and serves the checked-in kernel for each tile rocRoller would
run. rocRoller passes over a tile whose kernel fails to build, and the OFF
build cannot build anything, so it reads those configurations from
tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def.

    build  builds every configuration of a list with the hipblaslt-bench of a
           HIPBLASLT_ENABLE_ROCROLLER=ON build (rocRoller JIT, one solution
           index per configuration) and appends one record per configuration.
    write  writes the table from records. A configuration is recorded when it
           failed to build; one with records that both built and failed is an
           error. With --records-out it also writes the merged records as CSV,
           and with --check it compares instead of writing.

A configuration is one JSON object: layout ("TN", "NN", "NT" or "TT"), typeA,
typeB, typeC, typeD (FP4, FP6, BF6, FP8, BF8, Half, BFloat16, Float), tile
[m, n, k], wgm and streamK. A record adds built (true or false) and pass (the
rocRoller pass that failed, or null). The scales are always 32-element E8M0
blocks, as rocRoller's MX GEMMs take them. Stream-K turns workgroup mapping
off, so a Stream-K configuration is recorded without it. Every configuration
is built with temporal loads; its non-temporal variants are taken to build
alike.

A file ending in .csv holds the same records one per line (layout, typeA,
typeB, typeC, typeD, tile MxNxK, wgm, streamK, and built, failed or the
failing pass), under "# Source:" and "# rocRoller tree:" lines that write
copies into the table. scripts/rocroller_build_records.csv is the checked-in
set. A CSV list also serves build as its configurations.

build needs a GPU and an ON build, so CI does not run it. Rerun both steps
whenever rocRoller changes which configurations build:

    rocroller_unbuildable_configs.py build --build <ON build dir> \
        --configs scripts/rocroller_build_records.csv --gpus 0,1 \
        --source "develop <sha>, <how the configurations were chosen>" --out records.csv
    rocroller_unbuildable_configs.py write records.csv \
        --out tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def \
        --records-out scripts/rocroller_build_records.csv
"""

import argparse
import collections
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading

HIP_TYPES = {
    "FP4": ("f4_r", "HIP_R_4F_E2M1"),
    "FP6": ("f6_r", "HIP_R_6F_E2M3"),
    "BF6": ("bf6_r", "HIP_R_6F_E3M2"),
    "FP8": ("f8_r", "HIP_R_8F_E4M3"),
    "BF8": ("bf8_r", "HIP_R_8F_E5M2"),
    "Half": ("f16_r", "HIP_R_16F"),
    "BFloat16": ("bf16_r", "HIP_R_16BF"),
    "Float": ("f32_r", "HIP_R_32F"),
}
LAYOUTS = ("TN", "NN", "NT", "TT")
TYPE_ORDER = list(HIP_TYPES)
KNOWN_PASSES = ("WaitcntObserver", "SwizzleScale", "AddPrefetch", "AddTransposeLoadCT",
                "InProcessAssembler", "RegisterAllocator", "Scheduler", "Assembler")
BUILD_ERROR = "Error building the following kernel:"
CSV_FIELDS = ("layout", "typeA", "typeB", "typeC", "typeD", "tile", "wgm", "streamK", "result")
CSV_HEADER = re.compile(r"# (Source|rocRoller tree): (.*)")
TREE = re.compile(r"[0-9a-f]{40}")


def key(c):
    """The configuration as the table records it; Stream-K drops workgroup mapping."""
    if c["layout"] not in LAYOUTS:
        raise ValueError(f"unknown layout {c['layout']!r}")
    for t in ("typeA", "typeB", "typeC", "typeD"):
        if c[t] not in HIP_TYPES:
            raise ValueError(f"unknown {t} {c[t]!r}")
    m, n, k = (int(x) for x in c["tile"])
    sk = bool(c["streamK"])
    return (c["layout"], c["typeA"], c["typeB"], c["typeC"], c["typeD"], m, n, k,
            bool(c["wgm"]) and not sk, sk)


def sort_key(k):
    return (LAYOUTS.index(k[0]),) + tuple(TYPE_ORDER.index(t) for t in k[1:5]) + k[5:]


def read_records(path):
    """(records, header): JSONL records, or CSV records and their Source and rocRoller tree lines."""
    if not path.endswith(".csv"):
        return [json.loads(line) for line in open(path) if line.strip()], {}
    header, lines = {}, []
    for line in open(path, newline=""):
        m = CSV_HEADER.fullmatch(line.rstrip("\n"))
        if m:
            header[m.group(1)] = m.group(2)
        elif line.strip() and not line.startswith("#"):
            lines.append(line)
    records = []
    for r in csv.DictReader(lines):
        if tuple(r) != CSV_FIELDS:
            raise SystemExit(f"{path}: columns {tuple(r)}, expected {CSV_FIELDS}")
        result = r["result"]
        records.append(dict(layout=r["layout"], typeA=r["typeA"], typeB=r["typeB"], typeC=r["typeC"],
                            typeD=r["typeD"], tile=[int(x) for x in r["tile"].split("x")],
                            wgm=r["wgm"] == "1", streamK=r["streamK"] == "1", built=result == "built",
                            **{"pass": None if result in ("built", "failed") else result}))
    return records, header


def csv_header(source, tree):
    return "\n".join([
        "# Copyright Advanced Micro Devices, Inc., or its affiliates.",
        "# SPDX-License-Identifier: MIT",
        "#",
        "# Generated by projects/hipblaslt/scripts/rocroller_unbuildable_configs.py. Do not edit.",
        "# The rocRoller GEMM configurations rocRoller tried to build, for 32-element E8M0 block",
        "# scales, and whether each built (built) or failed (failed, or the pass that failed).",
        "# write renders unbuildable_configs.def from them; build --configs rebuilds them.",
        f"# Source: {source}",
        f"# rocRoller tree: {tree}",
        ",".join(CSV_FIELDS),
    ]) + "\n"


def csv_line(k, built, p):
    layout, a, b, c, d, m, n, kk, wgm, sk = k
    result = "built" if built else (p or "failed")
    return f"{layout},{a},{b},{c},{d},{m}x{n}x{kk},{int(wgm)},{int(sk)},{result}\n"


def rocroller_tree(build):
    """The git tree id of shared/rocroller at HEAD of the checkout an ON build was configured from."""
    home = None
    for line in open(os.path.join(build, "CMakeCache.txt")):
        if line.startswith("CMAKE_HOME_DIRECTORY:"):
            home = line.split("=", 1)[1].strip()
    if home is None:
        raise SystemExit(f"{build}/CMakeCache.txt has no CMAKE_HOME_DIRECTORY; pass --rocroller-tree")
    top = subprocess.run(["git", "-C", home, "rev-parse", "--show-toplevel"], check=True, capture_output=True,
                         text=True).stdout.strip()
    return subprocess.run(["git", "-C", top, "rev-parse", "HEAD:shared/rocroller"], check=True,
                          capture_output=True, text=True).stdout.strip()


def solution_index(m, n, k, wgm, sk):
    """solution_selection.cpp parametersToIndex, with tail loops and temporal loads."""
    r = k // 32
    r |= (n // 16) << 7
    r |= (m // 16) << 15
    r |= int(wgm) << 23
    r |= int(sk) << 24
    r |= 1 << 25
    r |= 1 << 31
    return r - (1 << 32)


def failed_pass(log):
    last = log.rsplit(BUILD_ERROR, 1)[-1]
    m = re.search(r"(\w+)\.(?:cpp|hpp):\d+: FatalError", last)
    if m:
        return m.group(1)
    if "with a different waitcnt state" in last:
        return "WaitcntObserver"
    return next((p for p in KNOWN_PASSES if p in last), None)


def build_one(c, build, gpu, scratch):
    layout, a, b, cc, d, m, n, k, wgm, sk = key(c)
    hb = os.path.join(build, "projects", "hipblaslt")
    env = {v: x for v, x in os.environ.items() if not v.startswith(("HIPBLASLT_", "ROCROLLER_", "TENSILE_"))}
    env.update(
        LD_LIBRARY_PATH=":".join([os.path.join(build, "shared", "rocroller"), os.path.join(hb, "library"),
                                  os.path.join(hb, "tensilelite"), os.path.join(build, "shared", "origami"),
                                  env.get("LD_LIBRARY_PATH", "")]),
        HIPBLASLT_TENSILE_LIBPATH=os.path.join(hb, "Tensile", "library", "gfx950"),
        HIP_VISIBLE_DEVICES=str(gpu), TMPDIR=scratch, HIPBLASLT_USE_ROCROLLER="1", HIPBLASLT_LOG_LEVEL="4",
        HIPBLASLT_LOG_FILE=os.path.join(scratch, "hipblaslt.log"))
    if os.path.exists(env["HIPBLASLT_LOG_FILE"]):
        os.remove(env["HIPBLASLT_LOG_FILE"])
    cmd = [os.path.join(hb, "clients", "hipblaslt-bench"), "-m", str(2 * m), "-n", str(2 * n), "-k", str(4 * k),
           "--transA", layout[0], "--transB", layout[1], "--a_type", HIP_TYPES[a][0], "--b_type", HIP_TYPES[b][0],
           "--c_type", HIP_TYPES[cc][0], "--d_type", HIP_TYPES[d][0], "--compute_type", "f32_r",
           "--scaleA", "3", "--scaleB", "3", "--algo_method", "index",
           "--solution_index", str(solution_index(m, n, k, wgm, sk)), "--iters", "1", "--cold_iters", "0"]
    p = subprocess.run(cmd, env=env, cwd=scratch, capture_output=True, text=True, timeout=1800)
    log = p.stdout + p.stderr
    if os.path.exists(env["HIPBLASLT_LOG_FILE"]):
        log += open(env["HIPBLASLT_LOG_FILE"], errors="replace").read()
    ran = p.returncode == 0 and "Gflops" in p.stdout
    rec = dict(layout=layout, typeA=a, typeB=b, typeC=cc, typeD=d, tile=[m, n, k], wgm=wgm, streamK=sk)
    if ran:
        return dict(rec, built=True, pass_=None)
    if BUILD_ERROR in log:
        return dict(rec, built=False, pass_=failed_pass(log))
    tail = " | ".join(log.strip().splitlines()[-3:])[:300]
    raise RuntimeError(f"{rec}: neither ran nor failed to build (rc {p.returncode}): {tail}")


def cmd_build(a):
    configs, _ = read_records(a.configs)
    gpus = [int(g) for g in a.gpus.split(",")]
    lock = threading.Lock()
    slots = [g for g in gpus for _ in range(a.per_gpu)]
    free = list(enumerate(slots))
    errors = 0
    as_csv = a.out.endswith(".csv")
    if as_csv and not (os.path.exists(a.out) and os.path.getsize(a.out)):
        if not a.source:
            raise SystemExit("a new CSV --out needs --source")
        with open(a.out, "w") as out:
            out.write(csv_header(a.source, a.rocroller_tree or rocroller_tree(a.build)))

    def work(c):
        with lock:
            slot, gpu = free.pop()
        try:
            with tempfile.TemporaryDirectory(prefix=f"rr_unbuildable_{slot}_") as scratch:
                return build_one(c, a.build, gpu, scratch)
        finally:
            with lock:
                free.append((slot, gpu))

    with open(a.out, "a") as out, concurrent.futures.ThreadPoolExecutor(len(slots)) as pool:
        for c, fut in [(c, pool.submit(work, c)) for c in configs]:
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - report and keep building the rest
                errors += 1
                print(f"error: {e}", file=sys.stderr)
                continue
            rec["pass"] = rec.pop("pass_")
            out.write(csv_line(key(rec), rec["built"], rec["pass"]) if as_csv else json.dumps(rec) + "\n")
            out.flush()
    return 1 if errors else 0


def render(rows, source, tree):
    """rows: {key: failing pass or None}, the unbuildable configurations."""
    passes = collections.Counter(p or "unknown" for p in rows.values())
    lines = [
        "// Copyright Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier: MIT",
        "//",
        "// Generated by projects/hipblaslt/scripts/rocroller_unbuildable_configs.py. Do not edit.",
        "// The rocRoller GEMM configurations rocRoller fails to build, for 32-element E8M0 block",
        "// scales. A build without rocRoller passes over these ranked tiles, as rocRoller does.",
        "// Each was built with temporal loads; its non-temporal variants are taken to build alike.",
        f"// Source: {source}",
        f"// rocRoller tree: {tree}",
        f"// {len(rows)} configurations; failing pass: "
        + ", ".join(f"{p} {n}" for p, n in sorted(passes.items(), key=lambda x: (-x[1], x[0]))) + ".",
        "//",
        "// X(transA, transB, typeA, typeB, typeC, typeD, tileM, tileN, tileK, workgroupMapping, streamK)",
    ]
    b = lambda v: "true" if v else "false"  # noqa: E731
    for k in sorted(rows, key=sort_key):
        layout, a, bt, c, d, m, n, kk, wgm, sk = k
        lines.append(f"X({b(layout[0] == 'T')}, {b(layout[1] == 'T')}, {HIP_TYPES[a][1]}, {HIP_TYPES[bt][1]}, "
                     f"{HIP_TYPES[c][1]}, {HIP_TYPES[d][1]}, {m}, {n}, {kk}, {b(wgm)}, {b(sk)})")
    return "\n".join(lines) + "\n"


def parse_def(text):
    """The configurations of a table render() wrote, as keys."""
    hip = {v[1]: k for k, v in HIP_TYPES.items()}
    keys = []
    for line in text.splitlines():
        m = re.fullmatch(r"X\((\w+), (\w+), (\w+), (\w+), (\w+), (\w+), (\d+), (\d+), (\d+), (\w+), (\w+)\)", line)
        if m:
            ta, tb, a, b, c, d, mm, n, k, wgm, sk = m.groups()
            layout = ("T" if ta == "true" else "N") + ("T" if tb == "true" else "N")
            keys.append((layout, hip[a], hip[b], hip[c], hip[d], int(mm), int(n), int(k), wgm == "true",
                         sk == "true"))
    return keys


def merge(paths):
    """{key: (built, failing pass)} over record files, and the Source and rocRoller tree their CSV headers name."""
    state, headers = {}, collections.defaultdict(set)
    for path in paths:
        records, header = read_records(path)
        for name, value in header.items():
            headers[name].add(value)
        for r in records:
            k = key(r)
            built, p = bool(r["built"]), r.get("pass")
            if k in state and state[k][0] != built:
                raise SystemExit(f"{k}: built in one record and failed in another")
            if k not in state or (state[k][1] is None and p):
                state[k] = (built, p)
    return state, headers


def from_headers(given, headers, name, flag):
    if given:
        return given
    values = headers.get(name, set())
    if len(values) != 1:
        raise SystemExit(f"pass {flag}: the records name {len(values)} different '{name}' lines")
    return next(iter(values))


def cmd_write(a):
    state, headers = merge(a.records)
    source = from_headers(a.source, headers, "Source", "--source")
    tree = from_headers(a.rocroller_tree, headers, "rocRoller tree", "--rocroller-tree")
    if not TREE.fullmatch(tree):
        raise SystemExit(f"rocRoller tree {tree!r} is not a 40-digit git tree id")
    rows = {k: p for k, (built, p) in state.items() if not built}
    outputs = [(a.out, render(rows, source, tree))]
    if a.records_out:
        outputs.append((a.records_out, csv_header(source, tree)
                        + "".join(csv_line(k, *state[k]) for k in sorted(state, key=sort_key))))
    if a.check:
        stale = [path for path, text in outputs
                 if not os.path.exists(path) or open(path, newline="").read() != text]
        for path in stale:
            print(f"{path} differs from what the records render", file=sys.stderr)
        print(f"{len(state)} configurations, {len(rows)} unbuildable: "
              + ("differs" if stale else "up to date"))
        return 1 if stale else 0
    for path, text in outputs:
        with open(path, "w", newline="") as f:
            f.write(text)
    print(f"{len(state)} configurations, {len(rows)} unbuildable -> {', '.join(p for p, _ in outputs)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--build", required=True, help="HIPBLASLT_ENABLE_ROCROLLER=ON build directory")
    b.add_argument("--configs", required=True, help="configurations, JSONL or CSV records")
    b.add_argument("--gpus", required=True, help="comma-separated HIP_VISIBLE_DEVICES ids")
    b.add_argument("--per-gpu", type=int, default=4)
    b.add_argument("--out", required=True, help="records, JSONL or (ending in .csv) CSV, appended to")
    b.add_argument("--source", help="what the records are built with, for a new CSV --out")
    b.add_argument("--rocroller-tree",
                   help="git tree id of the build's shared/rocroller, for a new CSV --out; "
                        "default: HEAD:shared/rocroller of the checkout --build was configured from")
    b.set_defaults(fn=cmd_build)
    w = sub.add_parser("write")
    w.add_argument("records", nargs="+", help="records, JSONL or CSV")
    w.add_argument("--source", help="what the records were built with; default: the CSV records' Source line")
    w.add_argument("--rocroller-tree",
                   help="git tree id of the shared/rocroller the records were built with; "
                        "default: the CSV records' rocRoller tree line")
    w.add_argument("--out", required=True)
    w.add_argument("--records-out", help="also write the merged records here as CSV")
    w.add_argument("--check", action="store_true",
                   help="compare with --out and --records-out instead of writing; exit 1 if they differ")
    w.set_defaults(fn=cmd_write)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
