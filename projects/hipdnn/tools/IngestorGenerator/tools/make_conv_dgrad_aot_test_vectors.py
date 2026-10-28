#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the golden vectors the native hipkernel:Gfx950ConvDgradAot tests compare with.

The native pack (``packs/Gfx950ConvDgradGeometry.hpp``) ports the launch plan of the
rocKE adapter (``builders/common/convolution_backward_data.py``,
``gfx950_conv_dgrad_aot_plan``). This tool runs the adapter over the shipped catalog
(``configs/gfx950_conv_dgrad_aot_<dtype>.instances.json``) and a seeded set of
synthetic problems, and writes, for the C++ tests:

* every catalog instance's KMD metadata, as the descriptor states it;
* the problems, and per problem the instances of its dtype that serve it with one
  digest folding the digest (``Gfx950ConvDgradPlan.digest``) of each one's plan;
* a few plans in full (``Gfx950ConvDgradPlan.serialize``), one of each kind, so a
  mismatch can be read rather than only detected.

Regenerate after a catalog or adapter change::

    python tools/make_conv_dgrad_aot_test_vectors.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_dgrad_aot_{fp16,bf16}.instances.json \\
        --output "$PROVIDER"/src/tests/engines/kernel_ingestor_engine/packs/Gfx950ConvDgradTestVectors.hpp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import make_conv_dgrad_aot_config as _cfg  # noqa: E402

_SEED = 950
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
)


def _problem(**kw):
    from builders.common.convolution_backward_data import Gfx950ConvDgradProblem

    return Gfx950ConvDgradProblem(**kw)


def _shape(
    n, c, k, h, w, y, x, s=1, p=0, d=1, g=1, dtype="fp16", sw=None, pw=None, dw=None
):
    return _problem(
        N=n,
        Hi=h,
        Wi=w,
        C=c,
        K=k,
        Y=y,
        X=x,
        sH=s,
        sW=s if sw is None else sw,
        pH=p,
        pW=p if pw is None else pw,
        dH=d,
        dW=d if dw is None else dw,
        groups=g,
        dtype=dtype,
    )


def synthetic_problems(specs) -> list:
    """A seeded problem set reaching every family and launch mode of the catalog.

    Hand-picked rows for the launch modes (tilde strides, fold_width, patch embeddings,
    the zero-filled dX, halo widths on either side of a binary's bound, batch slices,
    dilation, asymmetric strides), then per direct instance problems of its own
    capability (served and not), then random dense and grouped shapes.
    """
    rng = random.Random(_SEED)
    rows = [
        _shape(2, 64, 64, 14, 14, 3, 3, 1, 1),
        _shape(2, 32, 48, 15, 15, 3, 3, 2, 1),
        _shape(4, 64, 128, 28, 28, 1, 1, 2, 0),
        _shape(4, 64, 128, 28, 28, 1, 1, 1, 0),
        _shape(2, 128, 256, 32, 32, 2, 2, 2, 0),
        _shape(2, 3, 64, 64, 64, 4, 4, 4, 0),
        _shape(2, 3, 128, 224, 224, 16, 16, 16, 0),
        _shape(2, 3, 64, 30, 30, 4, 4, 5, 0),
        _shape(1, 3, 32, 64, 64, 7, 7, 2, 3),
        _shape(1, 16, 32, 31, 29, 7, 7, 4, 3),
        _shape(1, 24, 40, 20, 20, 3, 3, 1, 2, 2),
        _shape(1, 16, 24, 17, 13, 3, 3, 3, 2, sw=2, pw=1, dw=1),
        _shape(2, 64, 64, 32, 32, 3, 3, 1, 1),
        _shape(2, 64, 64, 32, 96, 3, 3, 1, 1),
        _shape(2, 64, 64, 32, 97, 3, 3, 1, 1),
        _shape(2, 128, 128, 16, 64, 5, 5, 1, 2),
        _shape(1, 128, 64, 4, 25600, 1, 11, 1, 0, pw=5),
        _shape(2, 192, 128, 17, 17, 1, 7, 1, 0, pw=3),
        _shape(2, 192, 128, 17, 17, 7, 1, 1, 3, pw=0),
        _shape(16, 256, 256, 768, 768, 3, 3, 1, 1, dtype="bf16"),
        _shape(3, 128, 128, 1024, 1024, 3, 3, 1, 1),
        _shape(1, 8, 320, 32, 32, 3, 3, 1, 1),
        _shape(2, 16, 1280, 9, 9, 3, 3, 1, 1),
        _shape(1, 4, 320, 64, 64, 3, 3, 1, 1),
        _shape(2, 4, 320, 48, 40, 3, 3, 1, 1, dtype="bf16"),
        _shape(2, 8, 1280, 16, 16, 3, 3, 1, 1, g=4),
        _shape(1, 12, 320, 20, 24, 3, 3, 1, 1, dtype="bf16"),
        _shape(1, 16, 320, 40, 40, 3, 3, 1, 1),
        _shape(1, 20, 320, 40, 40, 3, 3, 1, 1),
        _shape(1, 16, 288, 40, 40, 3, 3, 1, 1),
        _shape(2, 8, 8, 2, 2, 3, 3, 2, 1, g=8),
    ]
    directs = [s for s in specs if s.is_direct and not s.is_helper]
    for spec in directs:
        for _ in range(2):
            cpg = spec.cpg or rng.choice((4, 8, 12, 16, 20, 24, 32))
            kpg = spec.kpg or rng.choice((32, 64, 96, 160, 320))
            groups = rng.choice((1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 128, 256))
            if spec.is_depthwise:
                cpg = kpg = 1
                groups = rng.choice(
                    (8, 16, 24, 32, 64, 96, 128, 144, 192, 256, 384, 512)
                )
            hw = rng.choice((4, 7, 8, 14, 16, 28, 32, 56, 64, 112))
            rows.append(
                _shape(
                    rng.choice((1, 2, 4, 8, 16)),
                    cpg * groups,
                    kpg * groups,
                    hw,
                    rng.choice((hw, hw + 3)),
                    spec.filter_h,
                    spec.filter_w,
                    spec.stride_h,
                    spec.pad_h,
                    g=groups,
                    dtype=spec.dtype,
                )
            )
    for _ in range(120):
        y = rng.choice((1, 1, 2, 3, 3, 3, 5, 7))
        s = rng.choice((1, 1, 1, 2, 2, 4)) if y > 1 else rng.choice((1, 2))
        p = rng.choice((0, (y - 1) // 2, y // 2))
        groups = rng.choice((1, 1, 1, 2, 4, 8, 32))
        cpg = rng.choice((1, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256))
        kpg = rng.choice((8, 16, 32, 64, 128, 256))
        hw = rng.choice((7, 8, 13, 14, 16, 28, 32, 56, 64))
        rows.append(
            _shape(
                rng.choice((1, 2, 4, 8, 32)),
                cpg * groups,
                kpg * groups,
                hw,
                rng.choice((hw, hw + 1, 2 * hw)),
                y,
                y,
                s,
                p,
                g=groups,
                dtype=rng.choice(("fp16", "bf16")),
            )
        )
    seen, out = set(), []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


def _cpp_int(value: int) -> str:
    return f"{value}LL" if not -(1 << 31) <= value < (1 << 31) else str(value)


def _defaults() -> dict:
    """The KMD fields with a default_value, and the default."""
    return {
        field["name"]: field["default_value"]
        for field in _cfg.kmd_fields()
        if "default_value" in field
    }


def _instance_fields(spec) -> str:
    """Every KMD field as name=value;..., except those at their default_value."""
    defaults = _defaults()
    parts = []
    for field in _cfg.kmd_fields():
        name = field["name"]
        value = getattr(spec, name)
        if name in defaults and value == defaults[name]:
            continue
        parts.append(f"{name}={value}")
    return ";".join(parts)


def _problem_values(problem) -> list:
    return [
        problem.N,
        problem.C,
        problem.K,
        problem.Hi,
        problem.Wi,
        problem.Y,
        problem.X,
        problem.sH,
        problem.sW,
        problem.pH,
        problem.pW,
        problem.dH,
        problem.dW,
        problem.groups,
    ]


def _fold(digests) -> int:
    """FNV-1a 64 over ``digests``, each as 8 little-endian bytes (as planDigest hashes)."""
    h = 0xCBF29CE484222325
    for digest in digests:
        for byte in digest.to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def render(specs, problems) -> str:
    from builders.common.convolution_backward_data import gfx950_conv_dgrad_aot_plan

    by_dtype = {
        dtype: [s for s in specs if s.dtype == dtype] for dtype in ("fp16", "bf16")
    }
    rows = []
    full = {}
    for index, problem in enumerate(problems):
        served = []
        digests = []
        for spec in by_dtype[problem.dtype]:
            plan, _ = gfx950_conv_dgrad_aot_plan(spec, problem)
            served.append("1" if plan is not None else "0")
            if plan is None:
                continue
            digests.append(plan.digest())
            kind = (
                spec.direct_variant,
                len(plan.steps) > 1 and not spec.is_direct,
                plan.zero_dx,
                bool(spec.patch),
                bool(spec.dy_halo),
                plan.launched != problem,
            )
            if kind not in full:
                full[kind] = (
                    index,
                    by_dtype[problem.dtype].index(spec),
                    plan.serialize(),
                )
        rows.append((index, "".join(served), digests))

    out = [
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier:  MIT",
        "",
        "#pragma once",
        "",
        "// clang-format off",
        "// GENERATED test data, do not edit: projects/hipdnn/tools/IngestorGenerator/tools/",
        "// make_conv_dgrad_aot_test_vectors.py. The shipped catalog's KMD metadata, and the",
        "// launch plans of the rocKE adapter rocke/library/builders/common/",
        "// convolution_backward_data.py (gfx950_conv_dgrad_aot_plan) on synthetic problems.",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <vector>",
        "",
        "namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_vectors",
        "{",
        "",
        "/// N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups.",
        "using ProblemValues = std::array<int64_t, 14>;",
        "",
        "/// One catalog instance: its dtype and its KMD fields as name=value;..., those at",
        "/// their KMD default_value (DEFAULT_FIELDS) left out.",
        "struct CatalogInstance",
        "{",
        "    const char* dtype;",
        "    const char* fields;",
        "};",
        "",
        "/// The KMD fields that hold strings; every other one is an int.",
        "inline constexpr std::array<const char*, "
        + str(len(_STRING_FIELDS))
        + "> STRING_FIELDS{"
        + ", ".join(f'"{f}"' for f in _STRING_FIELDS)
        + "};",
        "",
        "/// The KMD fields with a default_value, as name=value;...",
        'inline constexpr const char* DEFAULT_FIELDS = "'
        + ";".join(f"{k}={v}" for k, v in _defaults().items())
        + '";',
        "",
        "inline const std::vector<CatalogInstance>& catalogInstances()",
        "{",
        "    static const std::vector<CatalogInstance> s_instances{",
    ]
    for spec in specs:
        out.append(f'        {{"{spec.dtype}", "{_instance_fields(spec)}"}},')
    out += [
        "    };",
        "    return s_instances;",
        "}",
        "",
        "/// One problem; served has one character per catalogInstances() entry of its dtype,",
        "/// in catalog order ('1' served), and digest folds the plan digest of each served",
        "/// one, in that order, with FNV-1a 64 (each as 8 little-endian bytes).",
        "struct ServedRow",
        "{",
        "    ProblemValues problem;",
        "    const char* dtype;",
        "    const char* served;",
        "    uint64_t digest;",
        "};",
        "",
        "inline const std::vector<ServedRow>& servedMatrix()",
        "{",
        "    static const std::vector<ServedRow> s_rows{",
    ]
    for index, served, digests in rows:
        problem = problems[index]
        values = ", ".join(str(v) for v in _problem_values(problem))
        out.append(
            f'        {{{{{values}}}, "{problem.dtype}", "{served}", {_fold(digests)}ULL}},'
        )
    out += [
        "    };",
        "    return s_rows;",
        "}",
        "",
        "/// A plan in full (Gfx950ConvDgradPlan.serialize): servedMatrix()[row] with the",
        "/// instance-th catalog instance of the row's dtype.",
        "struct FullPlan",
        "{",
        "    size_t row;",
        "    size_t instance;",
        "    std::vector<int64_t> serialized;",
        "};",
        "",
        "inline const std::vector<FullPlan>& fullPlans()",
        "{",
        "    static const std::vector<FullPlan> s_plans{",
    ]
    for kind in sorted(full, key=repr):
        row, instance, serialized = full[kind]
        values = ", ".join(_cpp_int(v) for v in serialized)
        out.append(f"        {{{row}, {instance}, {{{values}}}}},")
    out += [
        "    };",
        "    return s_plans;",
        "}",
        "",
        "} // namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_test_vectors",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rocke-root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    _cfg._bind_rocke(args.rocke_root)
    identities = []
    for path in args.instances:
        for entry in json.loads(path.read_text()):
            identities.append(entry.get("identity", entry))
    specs = []
    seen = set()
    for spec, reason in _cfg.instance_specs(identities, args.jobs):
        if spec is None:
            print(f"unsupported instance: {reason}", file=sys.stderr)
            return 1
        if spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    specs.sort(key=lambda s: (s.dtype, s.instance_id))
    problems = synthetic_problems(specs)
    args.output.write_text(render(specs, problems))
    print(f"{len(specs)} instances, {len(problems)} problems written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
