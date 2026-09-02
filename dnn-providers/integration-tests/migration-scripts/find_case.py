#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Query bundle cases by any parameter — shape, dtype, layout, op, input range.

Reads sweep.json + graph.template.json directly (single source of truth).

Examples::

    # List all batchnorm cases
    python3 find_case.py --op Batchnorm

    # Find fp16 nhwc cases
    python3 find_case.py --dtype fp16 --layout nhwc

    # Find cases that have an epsilon input (any range)
    python3 find_case.py --input epsilon

    # Find cases where epsilon is in [-1,1]
    python3 find_case.py --input epsilon:-1,1

    # Find by shape
    python3 find_case.py --shape 1x16x3x3

    # Combine filters
    python3 find_case.py --op Batchnorm --dtype bfp16 --input scale:-2,2

    # Show full detail for a specific case (by id or substring)
    python3 find_case.py --id f446b9
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_BUNDLE_DIR = SCRIPT_DIR.parent / "integration-test-bundles"


def _uid_to_role(template):
    roles = {}
    for node in template.get("nodes", []):
        for port, uid_val in node.get("inputs", {}).items():
            name = port.removesuffix("_tensor_uid").removesuffix("_uid")
            if isinstance(uid_val, int):
                roles[uid_val] = name
        for port, uid_val in node.get("outputs", {}).items():
            name = port.removesuffix("_tensor_uid").removesuffix("_uid")
            if isinstance(uid_val, int):
                roles[uid_val] = name
    return roles


def _infer_layout(dims, strides):
    if not dims or not strides or len(dims) != len(strides):
        return None
    if len(dims) < 3:
        return None
    pairs = list(zip(dims, strides))
    spatial = pairs[2:]
    channel = pairs[1]
    if not spatial:
        return None
    if channel[1] <= spatial[-1][1]:
        names = {3: "nlc", 4: "nhwc", 5: "ndhwc"}
    else:
        names = {3: "ncl", 4: "nchw", 5: "ncdhw"}
    return names.get(len(dims))


def _num(v):
    if isinstance(v, float):
        s = f"{v:g}"
        return s.replace(".", "p") if "." in s else s
    return str(v)


def _fmt_range(spec, role="?"):
    if not isinstance(spec, dict):
        return f"{role}={spec}"
    if "lo" in spec and "hi" in spec:
        rng = f"[{_num(spec['lo'])},{_num(spec['hi'])}]"
    elif "value" in spec:
        rng = (
            f"={_num(spec['value'])}"
            if isinstance(spec["value"], (int, float))
            else f"={spec['value']}"
        )
    else:
        rng = spec.get("kind", "?")
    seed = f" seed={spec['seed']}" if "seed" in spec else ""
    return f"{role}{rng}{seed}"


def _load_all_cases(bundle_dir):
    cases = []
    for root, dirs, files in os.walk(bundle_dir):
        if "sweep.json" not in files:
            continue
        sweep_path = os.path.join(root, "sweep.json")
        template_path = os.path.join(root, "graph.template.json")
        rel = os.path.relpath(root, bundle_dir)
        parts = rel.split(os.sep)
        tier = parts[0] if len(parts) > 0 else ""
        op = parts[1] if len(parts) > 1 else ""
        topo = parts[2] if len(parts) > 2 else ""

        try:
            with open(sweep_path) as f:
                sweep = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        template = {}
        if os.path.exists(template_path):
            try:
                with open(template_path) as f:
                    template = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        roles = _uid_to_role(template)

        for case in sweep.get("cases", []):
            values = case.get("values", {})
            tensors = values.get("tensors", [])
            rep = _rep_tensor(tensors)
            dims = rep.get("dims", [])
            strides = rep.get("strides", [])
            raw_dt = (values.get("io_data_type") or rep.get("data_type", "")).lower()
            dt = _DTYPE_DISPLAY.get(raw_dt, raw_dt)
            layout = _infer_layout(dims, strides) or ""
            meta = case.get("metadata", {})
            inputs = meta.get("inputs", {})

            input_roles = {}
            for uid, spec in inputs.items():
                try:
                    role = roles.get(int(uid), f"t{uid}")
                except (ValueError, TypeError):
                    role = f"t{uid}"
                input_roles[role] = spec

            cases.append(
                {
                    "id": case.get("id", ""),
                    "tier": tier,
                    "op": op,
                    "topology": topo,
                    "shape": dims,
                    "dtype": dt,
                    "layout": layout,
                    "inputs": input_roles,
                    "seed": meta.get("seed"),
                    "origin": meta.get("reference_source", ""),
                    "suite": f"{tier}_{op}_{topo}",
                    "gtest": f"{tier}_{op}_{topo}/{case.get('id', '')}",
                    "_raw_inputs": inputs,
                    "_roles": roles,
                }
            )
    return cases


def _rep_tensor(tensors):
    if not tensors:
        return {}
    best = tensors[0]
    best_rank = len(best.get("dims", []))
    best_nonunit = sum(1 for d in best.get("dims", []) if d > 1)
    for t in tensors[1:]:
        rank = len(t.get("dims", []))
        nonunit = sum(1 for d in t.get("dims", []) if d > 1)
        if rank > best_rank or (rank == best_rank and nonunit > best_nonunit):
            best = t
            best_rank = rank
            best_nonunit = nonunit
    return best


def _parse_input_filter(s):
    m = re.match(r"^(\w+):\[?([^,\[\]]+),([^,\[\]]+)\]?$", s)
    if m:
        return (
            m.group(1),
            float(m.group(2).replace("p", ".")),
            float(m.group(3).replace("p", ".")),
        )
    if re.match(r"^\w+$", s):
        return (s, None, None)
    print(
        f"  Invalid input filter: {s!r}  (expected: role  or  role:lo,hi)",
        file=sys.stderr,
    )
    sys.exit(1)


_DTYPE_DISPLAY = {
    "bfloat16": "bfp16",
    "half": "fp16",
    "float": "fp32",
    "double": "fp64",
}

_DTYPE_ALIASES = {v: v for v in _DTYPE_DISPLAY.values()}
_DTYPE_ALIASES.update({k: v for k, v in _DTYPE_DISPLAY.items()})
_DTYPE_ALIASES["bf16"] = "bfp16"


def _matches(case, args):
    if args.op and args.op.lower() not in case["op"].lower():
        return False
    if args.dtype:
        needle = _DTYPE_ALIASES.get(args.dtype.lower(), args.dtype.lower())
        if needle != case["dtype"]:
            return False
    if args.layout and args.layout.lower() != case["layout"]:
        return False
    if args.shape:
        target = [int(x) for x in args.shape.split("x")]
        if case["shape"] != target:
            return False
    if args.tier and args.tier.lower() != case["tier"].lower():
        return False
    if args.id and args.id not in case["id"]:
        return False
    if args.origin and args.origin.lower() not in case["origin"].lower():
        return False
    for rf in args.input or []:
        role, lo, hi = _parse_input_filter(rf)
        spec = case["inputs"].get(role)
        if not spec:
            return False
        if lo is not None and isinstance(spec, dict):
            if abs(spec.get("lo", 0) - lo) > 1e-9 or abs(spec.get("hi", 0) - hi) > 1e-9:
                return False
    return True


def _print_table(cases):
    if not cases:
        print("  No matching cases found.")
        return
    headers = ["suite", "id", "shape", "dtype", "layout", "inputs"]
    rows = []
    for c in cases:
        shape = "x".join(str(d) for d in c["shape"]) if c["shape"] else "-"
        if c["inputs"]:
            inp = " | ".join(
                _fmt_range(spec, role) for role, spec in sorted(c["inputs"].items())
            )
        else:
            inp = "-"
        rows.append(
            [c["suite"], c["id"], shape, c["dtype"] or "-", c["layout"] or "-", inp]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _print_detail(case):
    print(f"\n  Case: {case['id']}")
    print(f"  Suite: {case['suite']}")
    print(f"  GTest: --gtest_filter='{case['gtest']}'")
    shape = "x".join(str(d) for d in case["shape"]) if case["shape"] else "-"
    print(f"  Shape: {shape}")
    print(f"  Dtype: {case['dtype'] or '-'}")
    print(f"  Layout: {case['layout'] or '-'}")
    print(f"  Seed: {case['seed']}")
    if case["origin"]:
        print(f"  Origin: {case['origin']}")
    if case["inputs"]:
        print(f"  Inputs:")
        for role, spec in sorted(case["inputs"].items()):
            print(f"    {_fmt_range(spec, role)}")
    else:
        print(f"  Inputs: (no overrides — uses per-op defaults)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--op", help="filter by operation name (substring)")
    ap.add_argument("--dtype", help="filter by data type (e.g. fp16, bfp16, fp32)")
    ap.add_argument("--layout", help="filter by layout (e.g. nhwc, nchw)")
    ap.add_argument("--shape", help="filter by shape (e.g. 1x16x3x3)")
    ap.add_argument("--tier", help="filter by tier (quick, full, standard)")
    ap.add_argument(
        "--input",
        action="append",
        help="filter by input role, e.g. --input epsilon  or  --input epsilon:-1,1",
    )
    ap.add_argument("--id", help="filter by case id (substring match)")
    ap.add_argument(
        "--origin", help="filter by reference_source C++ suite name (substring)"
    )
    ap.add_argument(
        "--detail", action="store_true", help="show full detail for each match"
    )
    args = ap.parse_args()

    all_cases = _load_all_cases(args.bundle_dir)

    if not any(
        [
            args.op,
            args.dtype,
            args.layout,
            args.shape,
            args.tier,
            args.input,
            args.id,
            args.origin,
        ]
    ):
        print(f"  {len(all_cases)} total cases in {args.bundle_dir}")
        print(
            f"  Use --op, --dtype, --layout, --shape, --input, --id, --origin to filter."
        )
        return 0

    matches = [c for c in all_cases if _matches(c, args)]
    print(f"  {len(matches)}/{len(all_cases)} cases match\n")

    if args.detail:
        for c in matches:
            _print_detail(c)
    else:
        _print_table(matches)

    return 0


if __name__ == "__main__":
    sys.exit(main())
