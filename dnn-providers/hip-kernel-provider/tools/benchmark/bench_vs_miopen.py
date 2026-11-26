#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the hip-kernel-provider HIP_MLOPS_ENGINE against MIOpenDriver.

Covers BatchNorm forward training, LayerNorm forward and LayerNorm backward.

  run      generate a bundle tree for the shapes in bench_matrix.json, time the
           provider through `hipdnn_integration_tests --benchmark`, time MIOpen
           with `MIOpenDriver -t 1`, and write a CSV plus a markdown summary.
  compare  diff the provider timings of two CSVs (before/after a change).

Example:
  bench_vs_miopen.py run --build-dir /tmp/sb --label baseline
  bench_vs_miopen.py compare results/baseline.csv results/after.csv
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MATRIX = HERE / "bench_matrix.json"
DEFAULT_TEST_CONFIG = HERE.parent.parent / "config" / "HIP_MLOPS_ENGINE.toml"
ENGINE = "HIP_MLOPS_ENGINE"
TIER = "standard"

# Template topologies in the integration bundle tree that this tool clones.
OPS = {
    "bn_fwd": ("Batchnorm", "Default"),
    "ln_fwd": ("Layernorm", "Default"),
    "ln_bwd": ("LayernormBackward", "Variant2"),
}
SUITE = {op: f"{TIER}_{name}_{topo}" for op, (name, topo) in OPS.items()}

# short name -> (bundle data_type, MIOpenDriver command suffix)
DTYPES = {
    "fp32": ("float", ""),
    "fp16": ("half", "fp16"),
    "bf16": ("bfloat16", "bfp16"),
}

CSV_FIELDS = [
    "op",
    "case",
    "dtype",
    "layout",
    "dims",
    "normalized_dim_count",
    "tag",
    "status",
    "provider_min_ms",
    "provider_robust_ms",
    "miopen_min_ms",
    "miopen_avg_ms",
    "ratio_vs_miopen",
]


def strides_for(dims, layout):
    """Contiguous strides for logical NC[D]HW dims in the given memory layout."""
    n = len(dims)
    if layout.startswith("NC"):
        order = list(range(n - 1, -1, -1))  # W fastest ... N slowest
    else:
        order = [1] + list(range(n - 1, 1, -1)) + [0]  # C fastest, then W ... H, N
    strides = [0] * n
    acc = 1
    for axis in order:
        strides[axis] = acc
        acc *= dims[axis]
    return strides


def tensor(uid, dims, layout, dtype):
    return {
        "uid": uid,
        "dims": list(dims),
        "strides": strides_for(dims, layout),
        "data_type": dtype,
    }


def scalar(uid, dtype):
    return {"uid": uid, "dims": [1], "strides": [1], "data_type": dtype}


def bn_case(dims, layout, dtype_name):
    io = DTYPES[dtype_name][0]
    per_channel = [1, dims[1]] + [1] * (len(dims) - 2)
    t = lambda uid, d, dt: tensor(uid, d, layout, dt)
    tensors = [
        t(0, per_channel, "float"),
        t(1, per_channel, "float"),
        t(2, per_channel, "float"),
        t(3, per_channel, "float"),
        t(4, dims, io),
        t(5, dims, io),
        t(6, per_channel, "float"),
        scalar(7, "double"),
        scalar(8, "double"),
        t(9, per_channel, "float"),
        t(10, per_channel, "float"),
        t(11, per_channel, "float"),
    ]
    return {"io_data_type": io, "tensors": tensors}


def ln_dims(dims, k):
    stat = list(dims[: len(dims) - k]) + [1] * k
    affine = [1] * (len(dims) - k) + list(dims[len(dims) - k :])
    return stat, affine


def ln_fwd_case(dims, layout, dtype_name, k):
    io = DTYPES[dtype_name][0]
    stat, affine = ln_dims(dims, k)
    t = lambda uid, d, dt: tensor(uid, d, layout, dt)
    tensors = [
        t(0, stat, "float"),
        t(1, stat, "float"),
        t(2, dims, io),
        t(3, affine, "float"),
        scalar(4, "float"),
        t(5, affine, "float"),
        t(6, dims, io),
    ]
    return {"tensors": tensors, "attributes": {"normalized_dim_count": k}}


def ln_bwd_case(dims, layout, dtype_name, k):
    io = DTYPES[dtype_name][0]
    stat, affine = ln_dims(dims, k)
    t = lambda uid, d, dt: tensor(uid, d, layout, dt)
    tensors = [
        t(0, affine, "float"),
        t(1, affine, "float"),
        t(2, dims, io),
        t(3, dims, io),
        t(4, dims, io),
        scalar(5, "float"),
        t(6, stat, "float"),
        t(7, stat, "float"),
        t(8, affine, "float"),
    ]
    return {"tensors": tensors, "attributes": {"normalized_dim_count": k}}


def expand_matrix(matrix):
    """Return one dict per (op, shape, dtype, layout) to benchmark."""
    entries = []
    bn = matrix["batchnorm_fwd_train"]
    for dims in bn["shapes"]:
        for layout in bn["layouts"]:
            for dtype in bn["dtypes"]:
                entries.append(
                    dict(
                        op="bn_fwd",
                        dims=dims,
                        layout=layout,
                        dtype=dtype,
                        k=0,
                        tag="resnet",
                    )
                )
    for c in matrix["layernorm"]["cases"]:
        for layout in c["layouts"]:
            for dtype in c["dtypes"]:
                for op in ("ln_fwd", "ln_bwd"):
                    entries.append(
                        dict(
                            op=op,
                            dims=c["dims"],
                            layout=layout,
                            dtype=dtype,
                            k=c["normalized_dim_count"],
                            tag=c["tag"],
                        )
                    )
    for e in entries:
        parts = [str(d) for d in e["dims"]] + [e["dtype"], e["layout"].lower()]
        if e["op"] != "bn_fwd":
            parts.append(f"k{e['k']}")
        e["case"] = "_".join(parts)
    return entries


def build_bundles(entries, template_root, out_root):
    """Write a bundle tree containing only the benchmark cases."""
    if out_root.exists():
        shutil.rmtree(out_root)
    builders = {"bn_fwd": bn_case, "ln_fwd": ln_fwd_case, "ln_bwd": ln_bwd_case}
    for op, (name, topo) in OPS.items():
        src = template_root / TIER / name / topo
        dst = out_root / TIER / name / topo
        dst.mkdir(parents=True)
        shutil.copy(src / "graph.template.json", dst / "graph.template.json")
        sweep = json.loads((src / "sweep.json").read_text())
        metadata = sweep["cases"][0]["metadata"]
        cases = []
        for e in entries:
            if e["op"] != op:
                continue
            if op == "bn_fwd":
                values = builders[op](e["dims"], e["layout"], e["dtype"])
            else:
                values = builders[op](e["dims"], e["layout"], e["dtype"], e["k"])
            md = copy.deepcopy(metadata)
            md["notes"] = "Generated by tools/benchmark/bench_vs_miopen.py"
            cases.append({"id": e["case"], "values": values, "metadata": md})
        (dst / "sweep.json").write_text(
            json.dumps({"version": 1, "cases": cases}, indent=1)
        )


def run_provider(args, bundle_root, out_dir):
    if not args.test_config.is_file():
        sys.exit(f"test config not found: {args.test_config}")
    exe = args.build_dir / "bin" / "hipdnn_integration_tests"
    plugin = (
        args.build_dir
        / "lib"
        / "hipdnn_plugins"
        / "engines"
        / "libhip_kernel_provider.so"
    )
    env = os.environ.copy()
    lib_dirs = [str(args.rocm / "lib"), str(args.build_dir / "lib")]
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
    env["HIP_VISIBLE_DEVICES"] = args.device
    results = {}
    log = out_dir / "provider.log"
    with open(log, "w") as logf:
        for op in args.ops:
            cmd = [
                str(exe),
                "--test-article",
                str(plugin),
                "--test-engine",
                ENGINE,
                "--golden-data-dir",
                str(bundle_root),
                "--verification-mode",
                args.verification_mode,
                "--test-config",
                str(args.test_config),
                "--enforce-support-claims=false",
                "--benchmark",
                "--benchmark-warmup",
                str(args.warmup),
                "--benchmark-iterations",
                str(args.iterations),
                f"--gtest_filter={SUITE[op]}.*",
            ]
            print(f"provider: {op}", flush=True)
            logf.write("$ " + " ".join(cmd) + "\n")
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logf.write(proc.stdout + "\n")
            for line in proc.stdout.splitlines():
                m = re.match(
                    r"\[ BENCHMARK \] \S+?\.(\S+): .*? min=([\d.]+) ms robust=([\d.]+) ms",
                    line,
                )
                if m:
                    results[(op, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return results


def miopen_args(e):
    suffix = DTYPES[e["dtype"]][1]
    dims = e["dims"]
    if e["op"] == "bn_fwd":
        n, c, h, w = dims
        return [
            f"bnorm{suffix}",
            "-n",
            str(n),
            "-c",
            str(c),
            "-H",
            str(h),
            "-W",
            str(w),
        ] + ["-m", "1", "-F", "1", "-r", "1", "-s", "1", "-L", e["layout"]]
    depth = ["-D", str(dims[2])] if len(dims) == 5 else []
    return (
        [f"layernorm{suffix}", "-N", str(dims[0]), "-C", str(dims[1])]
        + depth
        + ["-H", str(dims[-2]), "-W", str(dims[-1])]
        + ["-L", e["layout"], "-o", str(len(dims) - e["k"]), "-m", "0"]
    )


def run_miopen(args, entries):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.rocm / 'lib'}:" + env.get("LD_LIBRARY_PATH", "")
    env["HIP_VISIBLE_DEVICES"] = args.device
    cache = {}
    results = {}
    log = args.out_dir / "miopen.log"
    with open(log, "w") as logf:
        for e in entries:
            if e["op"] not in args.ops:
                continue
            base = miopen_args(e)
            # One LayerNorm driver run produces both the forward and backward time.
            key = tuple(base)
            if key not in cache:
                cmd = (
                    [str(args.miopen_driver)]
                    + base
                    + ["-t", "1", "-V", "0", "-i", str(args.miopen_iterations)]
                )
                print("miopen:", " ".join(cmd[1:]), flush=True)
                logf.write("$ " + " ".join(cmd) + "\n")
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                logf.write(proc.stdout + "\n")
                cache[key] = proc.stdout
            out = cache[key]
            if e["op"] == "bn_fwd":
                mn = re.search(r"GPU Kernel Min Time .*?Elapsed: ([\d.]+) ms", out)
                av = re.search(r"GPU Kernel Avg Time .*?Elapsed: ([\d.]+) ms", out)
                if av:
                    results[(e["op"], e["case"])] = (
                        float(mn.group(1)) if mn else None,
                        float(av.group(1)),
                    )
            else:
                which = "Forward" if e["op"] == "ln_fwd" else "Backward"
                av = re.search(
                    rf"GPU Kernel Time {which} LayerNorm Elapsed: ([\d.]+) ms", out
                )
                if av:
                    results[(e["op"], e["case"])] = (None, float(av.group(1)))
    return results


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def write_outputs(entries, prov, mio, out_dir, label):
    rows = []
    for e in entries:
        key = (e["op"], e["case"])
        if key not in prov and key not in mio:
            continue
        pmin, prob = prov.get(key, (None, None))
        mmin, mavg = mio.get(key, (None, None))
        if prob is None:
            status = "provider-no-result"  # declined, skipped or failed verification
        elif mavg is None:
            status = "miopen-no-result"
        else:
            status = "ok"
        rows.append(
            {
                "op": e["op"],
                "case": e["case"],
                "dtype": e["dtype"],
                "layout": e["layout"],
                "dims": "x".join(str(d) for d in e["dims"]),
                "normalized_dim_count": e["k"],
                "tag": e["tag"],
                "status": status,
                "provider_min_ms": pmin,
                "provider_robust_ms": prob,
                "miopen_min_ms": mmin,
                "miopen_avg_ms": mavg,
                "ratio_vs_miopen": (prob / mavg) if prob and mavg else None,
            }
        )
    csv_path = out_dir / f"{label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"# {label}: hip-kernel-provider vs MIOpenDriver",
        "",
        "ratio = provider robust time / MIOpen average kernel time (above 1.0: provider slower).",
        "",
    ]
    for op in OPS:
        op_rows = [r for r in rows if r["op"] == op]
        if not op_rows:
            continue
        lines += [
            f"## {op}",
            "",
            "| dims | k | layout | dtype | tag | provider (ms) | MIOpen (ms) | ratio | status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        fmt = lambda v: f"{v:.4f}" if v is not None else "-"
        for r in op_rows:
            ratio = f"{r['ratio_vs_miopen']:.2f}" if r["ratio_vs_miopen"] else "-"
            lines.append(
                f"| {r['dims']} | {r['normalized_dim_count']} | {r['layout']} | {r['dtype']} "
                f"| {r['tag']} | {fmt(r['provider_robust_ms'])} | {fmt(r['miopen_avg_ms'])} "
                f"| {ratio} | {r['status']} |"
            )
        g = geomean([r["ratio_vs_miopen"] for r in op_rows])
        if g:
            lines += ["", f"Geometric mean ratio: {g:.2f}"]
        lines.append("")
    md_path = out_dir / f"{label}.md"
    md_path.write_text("\n".join(lines))
    return csv_path, md_path, rows


def cmd_run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(args.matrix.read_text())
    entries = expand_matrix(matrix)
    if args.filter:
        entries = [e for e in entries if re.search(args.filter, e["case"])]
    if not entries:
        sys.exit("no benchmark cases selected")
    template_root = args.build_dir / "lib" / "integration-test-bundles"
    bundle_root = args.out_dir / "bundles"
    build_bundles(entries, template_root, bundle_root)
    prov = {} if args.skip_provider else run_provider(args, bundle_root, args.out_dir)
    mio = {} if args.skip_miopen else run_miopen(args, entries)
    csv_path, md_path, rows = write_outputs(
        entries, prov, mio, args.out_dir, args.label
    )
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\nwrote {csv_path}\nwrote {md_path}")
    print(f"{len(rows) - len(bad)} compared, {len(bad)} without a full result")
    for op in OPS:
        g = geomean([r["ratio_vs_miopen"] for r in rows if r["op"] == op])
        if g:
            print(f"  {op}: geometric mean provider/MIOpen = {g:.2f}")


def load_csv(path):
    with open(path) as f:
        return {(r["op"], r["case"]): r for r in csv.DictReader(f)}


def cmd_compare(args):
    before, after = load_csv(args.before), load_csv(args.after)
    print(
        "| op | case | before (ms) | after (ms) | speedup | MIOpen (ms) | after/MIOpen |"
    )
    print("|---|---|---|---|---|---|---|")
    speedups = {}
    for key in sorted(before.keys() & after.keys()):
        b, a = before[key], after[key]
        try:
            pb, pa = float(b["provider_robust_ms"]), float(a["provider_robust_ms"])
        except ValueError:
            continue
        mio = a["miopen_avg_ms"] or b["miopen_avg_ms"]
        mio_f = float(mio) if mio else None
        speedups.setdefault(key[0], []).append(pb / pa)
        print(
            f"| {key[0]} | {key[1]} | {pb:.4f} | {pa:.4f} | {pb / pa:.2f}x "
            f"| {mio_f:.4f} | {pa / mio_f:.2f} |"
            if mio_f
            else f"| {key[0]} | {key[1]} | {pb:.4f} | {pa:.4f} | {pb / pa:.2f}x | - | - |"
        )
    print()
    for op, vals in speedups.items():
        print(
            f"{op}: geometric mean speedup {geomean(vals):.3f}x over {len(vals)} cases"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark the provider against MIOpenDriver")
    r.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="superbuild dir with bin/hipdnn_integration_tests and the provider plugin",
    )
    r.add_argument("--label", default="run", help="name of the output files")
    r.add_argument("--out-dir", type=Path, default=Path("bench_results"))
    r.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    r.add_argument(
        "--verification-mode",
        choices=["auto", "gpu", "cpu"],
        default="gpu",
        help="reference used to verify provider output before timing; some cases "
        "(LayerNorm backward k=3) only pass against the CPU reference",
    )
    r.add_argument(
        "--test-config",
        type=Path,
        default=DEFAULT_TEST_CONFIG,
        help="engine test config (validator overrides)",
    )
    r.add_argument(
        "--rocm", type=Path, default=Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    )
    r.add_argument(
        "--miopen-driver",
        type=Path,
        default=None,
        help="default: <rocm>/bin/MIOpenDriver",
    )
    r.add_argument("--ops", nargs="+", choices=list(OPS), default=list(OPS))
    r.add_argument("--filter", help="regex on the case id to run a subset")
    r.add_argument("--device", default="0", help="HIP_VISIBLE_DEVICES for both tools")
    r.add_argument("--warmup", type=int, default=50, help="provider warmup executions")
    r.add_argument(
        "--iterations", type=int, default=100, help="provider timed executions"
    )
    r.add_argument("--miopen-iterations", type=int, default=100)
    r.add_argument("--skip-provider", action="store_true")
    r.add_argument("--skip-miopen", action="store_true")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="compare provider timings of two runs")
    c.add_argument("before", type=Path)
    c.add_argument("after", type=Path)
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args()
    if args.cmd == "run" and args.miopen_driver is None:
        args.miopen_driver = args.rocm / "bin" / "MIOpenDriver"
    args.fn(args)


if __name__ == "__main__":
    main()
