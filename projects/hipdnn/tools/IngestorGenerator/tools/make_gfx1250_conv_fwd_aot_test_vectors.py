#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Golden launch plans of hipkernel:Gfx1250ConvFwdAot, from rocKE's dispatcher.

The engine's C++ port of admission and launch geometry is proven against the
dispatcher it ports, through the packaging adapter's
``gfx1250_conv_fwd_aot_serves`` / ``gfx1250_conv_fwd_aot_plan``
(``builders/gfx1250/conv_fwd_aot.py``): for every catalog instance (the
``--instances`` lists, ``configs/gfx1250_conv_fwd_aot_<dtype>.instances.json``) and a
problem set, whether the instance admits the problem and, if it does, the launch plan
it runs: per step the binary (its ``instance_id``), grid, block and every kernarg, and
the workspaces. Pointers and buffer sizes are symbolic (``{"buf": name, "offset": n}``,
``{"bytes": name}``), so the record is what the engine must compute, not an address.

``--header`` writes the C++ golden header the engine's unit test reads: the KMD fields
of every instance (as the catalog carries them), one shared problem set (``--shapes``
plus ``--synthetic`` problems around each instance's baked fields), and per instance
the number of problems it admits and a 64-bit FNV-1a digest folded over every admitted
problem's canonical plan (:func:`canonical_plan`, which the test restates). The header
records its problem set, so ``--problems-from-header <header>`` regenerates it byte for
byte without the original shapes file; a new problem set needs ``--shapes``. ``--out``
writes every record as JSON lines instead, for a wider offline comparison.

Usage:
  python make_gfx1250_conv_fwd_aot_test_vectors.py --rocke-root <rocke> \\
      --instances configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json \\
      --header TestVectors.hpp --problems-from-header TestVectors.hpp
  python make_gfx1250_conv_fwd_aot_test_vectors.py --rocke-root <rocke> \\
      --instances ... --out vectors.jsonl [--shapes shapes.txt] [--per-kernel 24]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_PTR_SHIFT = 40
_SIZE_SHIFT = 52


def _ptrs_sizes(names):
    ptrs = {n: (i + 1) << _PTR_SHIFT for i, n in enumerate(names)}
    sizes = {n: (i + 1) << _SIZE_SHIFT for i, n in enumerate(names)}
    return ptrs, sizes


def _symbolic(value: int, names, ptrs, sizes):
    # Kernargs are i32: a size or pointer marker never collides with a value.
    if value >= 1 << _SIZE_SHIFT:
        return {"bytes": names[(value >> _SIZE_SHIFT) - 1]}
    if value >= 1 << _PTR_SHIFT:
        i = (value >> _PTR_SHIFT) - 1
        return {"buf": names[i], "offset": value - ptrs[names[i]]}
    return value


def _problem_dict(p: dict) -> dict:
    return {k: int(v) if not isinstance(v, str) else v for k, v in p.items()}


def _plan_record(plan, names):
    ptrs, sizes = _ptrs_sizes(names)
    steps = []
    for step in plan.steps:
        values = plan.values(step, ptrs, sizes)
        steps.append(
            {
                "kernel": step.identity.stable_hash(),
                "grid": list(step.grid),
                "block": list(step.block),
                "args": [
                    [
                        a["name"],
                        a["type"],
                        _symbolic(int(values[a["name"]]), names, ptrs, sizes),
                    ]
                    for a in step.signature
                ],
            }
        )
    return {"steps": steps, "workspaces": dict(sorted(dict(plan.workspaces).items()))}


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
#: The problem fields of the header, in order.
_FIELDS = ("N", "C", "K", "H", "W", "Y", "X", "G", "sh", "sw", "ph", "pw", "dh", "dw")


def fnv1a(digest: int, text: str) -> int:
    for byte in text.encode():
        digest = ((digest ^ byte) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def canonical_plan(plan: dict) -> str:
    """A plan record as one string: per step ``kernel|gx,gy,gz|bx|`` and its args
    (``name=i:v`` / ``name=p:buf+off`` / ``name=b:buf``), then ``ws:`` and the
    workspaces in name order."""
    out = []
    for step in plan["steps"]:
        args = []
        for name, _type, value in step["args"]:
            if isinstance(value, int):
                args.append(f"{name}=i:{value}")
            elif "buf" in value:
                args.append(f"{name}=p:{value['buf']}+{value['offset']}")
            else:
                args.append(f"{name}=b:{value['bytes']}")
        grid = ",".join(map(str, step["grid"]))
        out.append(f"{step['kernel']}|{grid}|{step['block'][0]}|{','.join(args)};")
    ws = ",".join(f"{k}={v}" for k, v in sorted(plan["workspaces"].items()))
    return "".join(out) + f"ws:{ws}"


def _synthetic(identity, rng: random.Random, count: int):
    """Problems around ``identity``'s baked fields."""
    out = []
    direct = identity.is_direct
    for _ in range(count):
        if direct and identity.filter_h:
            Y, X = identity.filter_h, identity.filter_w
            ph, pw = identity.pad_h, identity.pad_w
            sh = sw = identity.stride_h
            dh, dw = identity.dilation_h, identity.dilation_w
        else:
            Y = X = rng.choice((1, 1, 3, 3, 3, 5, 7))
            ph = pw = rng.choice((0, Y // 2, Y // 2))
            sh = sw = rng.choice((1, 1, 2))
            dh = dw = rng.choice((1, 1, 1, 2))
        cpg = identity.cpg or rng.choice(
            (1, 3, 4, 8, 16, 32, 64, 96, 128, 256, 320, 512)
        )
        kpg = identity.kpg or rng.choice((1, 3, 4, 8, 16, 32, 64, 96, 128, 256, 512))
        if direct and identity.cpg == 1 and identity.kpg == 1:
            G = rng.choice((8, 32, 64, 96, 144, 256, 384, 960))
        else:
            G = rng.choice((1, 1, 1, 2, 4, 8, 32))
        N = rng.choice((1, 1, 2, 3, 4, 8, 16, 32, 64))
        H = rng.choice((1, 3, 7, 8, 14, 17, 28, 31, 56, 64, 112, 128, 224, 300))
        W = rng.choice((1, 3, 7, 8, 14, 17, 28, 31, 56, 64, 112, 128, 224, 300))
        if H + 2 * ph < dh * (Y - 1) + 1 or W + 2 * pw < dw * (X - 1) + 1:
            continue
        out.append(
            dict(N=N, C=cpg * G, K=kpg * G, H=H, W=W, Y=Y, X=X, G=G, sh=sh, sw=sw,
                 ph=ph, pw=pw, dh=dh, dw=dw, dtype=identity.dtype_a)  # fmt: skip
        )
    return out


def _parse_shapes(path: Path):
    """MIOpenDriver-style lines (``convfp16 -n .. -c ..``) -> problems."""
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("conv"):
            continue
        a = dict(re.findall(r"-(\w) (\S+)", line))
        i = lambda k: int(a[k])  # noqa: E731
        out.append(
            dict(N=i("n"), C=i("c"), K=i("k"), H=i("H"), W=i("W"), Y=i("y"), X=i("x"),
                 G=i("g"), sh=i("u"), sw=i("v"), ph=i("p"), pw=i("q"), dh=i("l"),
                 dw=i("j"), dtype="bf16" if line.startswith("convbfp16") else "fp16")  # fmt: skip
        )
    return out


def problems_from_header(path: Path) -> list[dict]:
    """The problem set an earlier ``--header`` run recorded, in its order."""
    text = path.read_text(encoding="utf-8")
    start = text.index("PROBLEMS = {{")
    end = text.index("}};", start)
    out = []
    for row in re.findall(r"\{([-0-9, ]+)\}", text[start:end]):
        values = [int(v) for v in row.split(",")]
        problem = dict(zip(_FIELDS, values))
        problem["dtype"] = "bf16" if values[len(_FIELDS)] else "fp16"
        out.append(problem)
    return out


def _metadata(spec) -> str:
    """``spec``'s KMD fields as ``name=value;``, those at 0 or empty left out."""
    out = []
    for name in type(spec).__dataclass_fields__:
        value = getattr(spec, name)
        if value not in (0, ""):
            out.append(f"{name}={value};")
    return "".join(out)


def _header(args, A, specs, rng) -> None:
    """The unit test's golden header (see the module docstring)."""
    if args.problems_from_header:
        unique = problems_from_header(args.problems_from_header)
    else:
        problems = _parse_shapes(args.shapes) if args.shapes else []
        for spec in specs:
            problems += _synthetic(spec.to_identity(), rng, args.synthetic)
        unique = list(
            {tuple(p[f] for f in _FIELDS) + (p["dtype"],): p for p in problems}.values()
        )
    int_fields = [f for f in A.SPEC_FIELDS if f not in A.STRING_FIELDS]
    string_fields = [f for f in A.SPEC_FIELDS if f in A.STRING_FIELDS]
    kernels = []
    for spec in specs:
        digest, admitted = _FNV_OFFSET, 0
        for index, p in enumerate(unique):
            if (
                p["dtype"] != spec.dtype
                or not A.gfx1250_conv_fwd_aot_serves(spec, p)[0]
            ):
                continue
            plan = A.gfx1250_conv_fwd_aot_plan(spec, p)
            record = _plan_record(plan, ["x", "w", "y", *sorted(plan.workspaces)])
            digest = fnv1a(digest, f"{index}:{canonical_plan(record)};")
            admitted += 1
        kernels.append((spec, admitted, digest))
    lines = [
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier:  MIT",
        "",
        "// GENERATED by projects/hipdnn/tools/IngestorGenerator/tools/"
        "make_gfx1250_conv_fwd_aot_test_vectors.py; do not edit.",
        "// Re-check against the in-tree dispatcher: --instances "
        "configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json",
        "// --header <this file> --problems-from-header <this file>.",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace gfx1250_conv_fwd_aot_vectors",
        "{",
        "",
        "// clang-format off",
        "",
        "/// N, C, K, H, W, Y, X, groups, stride h/w, pad h/w, dilation h/w, bf16.",
        f"inline constexpr std::array<std::array<int32_t, {len(_FIELDS) + 1}>, {len(unique)}> PROBLEMS = {{{{",
    ]
    for p in unique:
        values = [p[f] for f in _FIELDS] + [int(p["dtype"] == "bf16")]
        lines.append("    {" + ", ".join(map(str, values)) + "},")
    lines += [
        "}};",
        "",
        "/// The KMD fields every instance carries; a row states those it does not leave",
        "/// at 0 (an int) or empty (a string).",
        f"inline constexpr std::array<const char*, {len(int_fields)}> INT_FIELDS = {{{{",
        *[f'    "{name}",' for name in int_fields],
        "}};",
        f"inline constexpr std::array<const char*, {len(string_fields)}> STRING_FIELDS = {{{{",
        *[f'    "{name}",' for name in string_fields],
        "}};",
        "",
        "struct KernelVector",
        "{",
        "    const char* instanceId;",
        "    const char* metadata; ///< `name=value;` per stated KMD field.",
        "    int64_t admitted; ///< Problems it admits.",
        "    uint64_t digest; ///< FNV-1a over every admitted problem's canonical plan.",
        "};",
        "",
        f"inline constexpr std::array<KernelVector, {len(kernels)}> KERNELS = {{{{",
    ]
    lines += [
        f'    {{"{spec.instance_id}", "{_metadata(spec)}", {admitted}, 0x{digest:016X}ULL}},'
        for spec, admitted, digest in kernels
    ]
    lines += [
        "}};",
        "",
        "// clang-format on",
        "",
        "} // namespace gfx1250_conv_fwd_aot_vectors",
        "",
    ]
    args.header.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"{args.header}: {len(unique)} problems, {len(kernels)} kernels")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rocke-root", required=True, type=Path)
    ap.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        required=True,
        help="the catalog's instance lists (configs/gfx1250_conv_fwd_aot_<dtype>.instances.json)",
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--header", type=Path, help="write the unit test's golden header")
    ap.add_argument("--synthetic", type=int, help="problems per kernel (default 4)")
    ap.add_argument("--shapes", type=Path, help="extra problems, MIOpenDriver lines")
    ap.add_argument(
        "--problems-from-header",
        type=Path,
        help="with --header: reuse the problem set this earlier header records",
    )
    ap.add_argument(
        "--per-kernel", type=int, help="with --out: problems per kernel (default 24)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.problems_from_header and not args.header:
        ap.error("--problems-from-header needs --header")
    if args.problems_from_header and (args.shapes or args.synthetic is not None):
        ap.error(
            "--problems-from-header reuses a recorded set: no --shapes or --synthetic"
        )
    if bool(args.header) == bool(args.out):
        ap.error("give exactly one of --header or --out")
    if args.out and args.synthetic is not None:
        ap.error("--synthetic belongs to --header; --out takes --per-kernel")
    if args.header and args.per_kernel is not None:
        ap.error("--per-kernel belongs to --out; --header takes --synthetic")
    if args.synthetic is None:
        args.synthetic = 4
    if args.per_kernel is None:
        args.per_kernel = 24
    sys.path[:0] = [
        str(args.rocke_root / "library"),
        str(args.rocke_root / "platform" / "python"),
        str(Path(__file__).resolve().parent),
    ]
    from builders.gfx1250 import conv_fwd_aot as A

    import make_gfx1250_conv_fwd_aot_config as config_tool

    specs, rejected = config_tool.read_catalog(args.instances)
    if rejected:
        ap.error(f"{len(rejected)} unsupported instance(s), e.g. {rejected[0][1]}")
    rng = random.Random(args.seed)
    if args.header:
        _header(args, A, specs, rng)
        return
    shared = _parse_shapes(args.shapes) if args.shapes else []
    n_rec = n_adm = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for spec in specs:
            problems = _synthetic(spec.to_identity(), rng, args.per_kernel) + [
                p for p in shared if p["dtype"] == spec.dtype
            ]
            for p in problems:
                ok, why = A.gfx1250_conv_fwd_aot_serves(spec, p)
                rec = {
                    "kernel": spec.instance_id,
                    "problem": _problem_dict(p),
                    "admit": ok,
                }
                if ok:
                    plan = A.gfx1250_conv_fwd_aot_plan(spec, p)
                    names = ["x", "w", "y", *sorted(plan.workspaces)]
                    rec["plan"] = _plan_record(plan, names)
                    n_adm += 1
                else:
                    rec["why"] = why
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n_rec += 1
    print(f"{args.out}: {n_rec} records, {n_adm} admitted")


if __name__ == "__main__":
    main()
