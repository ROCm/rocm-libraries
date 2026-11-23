#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the golden vectors the native forward AOT engine tests compare with.

One tool for every forward engine of the rocKE adapter's ``TARGETS``; the catalog's
architecture picks the engine (``hipkernel:Gfx942ConvFwdAot`` ships its vectors). The
native pack (``packs/Gfx950ConvFwdGeometry.hpp``, ``packs/Gfx950ConvFwdNative.cpp``)
ports what the rocKE adapter (``builders/common/convolution_forward.py``) decides for
an instance on a problem. This tool runs the adapter over an engine's shipped catalog
(``configs/<arch>_conv_fwd_aot_<dtype>.instances.json``) and a seeded set of synthetic
problems, among them one each instance serves, and writes, for the C++ tests:

* every catalog instance's KMD metadata, as the descriptor states it;
* the problems, and per problem the instances of its dtype that serve it
  (``<arch>_conv_fwd_aot_serves``), those the engine then runs
  (``<arch>_conv_fwd_aot_runtime_candidates``), and one digest folding the digest of each
  serving instance's launch plan (:func:`plan_values`: the launches' grids, blocks and
  kernel arguments, a batch split, the workspace);
* a few plans in full, one of each kind, so a mismatch can be read rather than only
  detected.

Regenerate after a catalog or adapter change, under the catalog's LLVM flavor::

    PACKS="$PROVIDER"/src/tests/engines/kernel_ingestor_engine/packs
    ROCKE_LLVM_FLAVOR=llvm22 python tools/make_conv_aot_test_vectors.py \\
        --rocke-root "$ROCKE" \\
        --instances configs/gfx942_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$PACKS"/Gfx942ConvFwdTestVectors.hpp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import make_conv_aot_config as _cfg  # noqa: E402

_SEED = 942
#: The kernel arguments every launch passes before the AOT problem block: the three
#: buffers and their byte counts. The plans hold the block only.
_LEADING_ARGUMENTS = ("A", "B", "D", "A_bytes", "B_bytes", "D_bytes")
_U32 = 0xFFFFFFFF


def namespace(arch: str) -> str:
    """The C++ namespace of ``arch``'s vectors; a ValueError unless ``arch`` is one of
    the adapter's ``TARGETS``."""
    if arch not in _cfg.arches():
        raise ValueError(f"no packaged forward engine for arch {arch!r}")
    return f"{arch}_fwd_test_vectors"


def kmd_fields() -> list:
    """The KMD fields of every forward engine, as ``build_config`` writes them: name,
    type, and the default_value of those that have one."""
    fields = [
        {"name": name, "type": "int" if name in _cfg._INT_FIELDS else "string"}
        for name in _cfg.SPEC_ORDER
    ]
    for name, default in {**_cfg.DIRECT_FIELDS, **_cfg.DIRECT_KNOB_FIELDS}.items():
        fields.append(
            {
                "name": name,
                "type": "int" if name in _cfg._INT_FIELDS else "string",
                "default_value": default,
            }
        )
    return fields


def _shape(
    n,
    c,
    k,
    h,
    w,
    y,
    x,
    s=1,
    p=0,
    d=1,
    g=1,
    dtype="fp16",
    sw=None,
    pw=None,
    dw=None,
):
    from builders.common.convolution_forward import Gfx950ConvFwdSpec

    return Gfx950ConvFwdSpec(
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
        dtype=dtype,
        groups=g,
    )


def synthetic_problems(specs) -> list:
    """A seeded problem set reaching every family and launch mode of the catalog.

    Hand-picked rows first: pointwise, the common filters at strides 1 and 2 with and
    without padding, dilation, asymmetric strides and padding (which no direct kernel
    takes), depthwise graphs for the direct kernels and the group-merged instances
    (group counts each merge degree does and does not divide), grouped graphs of
    several channels per group, small and odd channel counts, and graphs large enough
    to split the batch or to pass the 32-bit launch ABI. Then per compiled direct
    capability problems of its own filter, stride, padding and dilation (served and
    not), then random dense, grouped and depthwise shapes. Last, for each instance
    none of these serves, one it does (:func:`_serving_problems`).

    The problems are built as the adapter's problem spec directly, not through
    ``<arch>_conv_fwd_aot_problem_for_request``: the C++ tests check the request to
    problem translation, by binding each problem's graph with the engine's graph
    matcher.
    """
    rng = random.Random(_SEED)
    rows = [
        # Pointwise, dense and strided, small and wide.
        _shape(2, 64, 64, 14, 14, 1, 1),
        _shape(8, 256, 512, 28, 28, 1, 1, dtype="bf16"),
        _shape(4, 64, 128, 28, 28, 1, 1, 2, 0),
        _shape(32, 8, 8, 192, 192, 1, 1),
        _shape(1, 1024, 2048, 7, 7, 1, 1, dtype="bf16"),
        # 3x3, 5x5, 7x7 and 11x11 at strides 1 and 2, padded and not.
        _shape(2, 64, 64, 14, 14, 3, 3, 1, 1),
        _shape(2, 64, 64, 14, 14, 3, 3, 1, 1, dtype="bf16"),
        _shape(2, 32, 48, 15, 15, 3, 3, 2, 1),
        _shape(4, 128, 128, 56, 56, 3, 3, 1, 0, dtype="bf16"),
        _shape(2, 128, 256, 32, 32, 3, 3, 2, 0),
        _shape(2, 64, 96, 27, 27, 5, 5, 1, 2),
        _shape(2, 48, 64, 31, 31, 5, 5, 2, 2, dtype="bf16"),
        _shape(2, 3, 64, 224, 224, 7, 7, 2, 3),
        _shape(1, 16, 32, 31, 29, 7, 7, 1, 3, dtype="bf16"),
        _shape(2, 3, 96, 227, 227, 11, 11, 4, 0),
        _shape(1, 16, 64, 40, 40, 11, 11, 1, 5, dtype="bf16"),
        # Dilation, and asymmetric strides, padding and filters.
        _shape(1, 24, 40, 20, 20, 3, 3, 1, 2, 2),
        _shape(2, 32, 64, 16, 16, 3, 3, 2, 2, d=2, dtype="bf16"),
        _shape(1, 16, 32, 19, 23, 3, 3, 1, 1, d=3, dw=2),
        _shape(1, 16, 24, 17, 13, 3, 3, 3, 2, d=2, sw=2, pw=1, dw=1),
        _shape(2, 192, 128, 17, 17, 1, 7, 1, 0, pw=3),
        _shape(2, 192, 128, 17, 17, 7, 1, 1, 3, pw=0, dtype="bf16"),
        # Depthwise: the std, spatial, tiled and col kernels, and the group-merged
        # instances, over group counts gm 4, 16 and 32 divide and do not.
        _shape(2, 32, 32, 14, 14, 3, 3, 1, 1, g=32),
        _shape(2, 48, 48, 28, 28, 3, 3, 1, 1, g=48),
        _shape(4, 64, 64, 56, 56, 3, 3, 1, 1, g=64, dtype="bf16"),
        _shape(2, 96, 96, 56, 56, 3, 3, 2, 1, g=96),
        _shape(2, 128, 128, 28, 28, 3, 3, 1, 1, g=128),
        _shape(2, 24, 24, 14, 14, 3, 3, 1, 1, g=24, dtype="bf16"),
        _shape(1, 20, 20, 9, 9, 3, 3, 1, 1, g=20),
        _shape(2, 36, 36, 17, 17, 5, 5, 1, 2, g=36, dtype="bf16"),
        _shape(2, 256, 256, 14, 14, 5, 5, 2, 2, g=256),
        _shape(1, 512, 512, 7, 7, 7, 7, 1, 3, g=512, dtype="bf16"),
        _shape(4, 144, 144, 112, 112, 3, 3, 1, 1, g=144),
        _shape(2, 97, 97, 31, 31, 3, 3, 1, 1, g=97),
        _shape(2, 64, 64, 15, 15, 3, 3, 1, 0, g=64),
        # Grouped: channels per group 1x2 (channel multiplier), 3x6, 4, 8, 16, 32, 64.
        _shape(2, 32, 64, 14, 14, 3, 3, 1, 1, g=32),
        _shape(2, 12, 24, 14, 14, 3, 3, 1, 1, g=4, dtype="bf16"),
        _shape(2, 64, 64, 28, 28, 3, 3, 1, 1, g=16),
        _shape(2, 64, 64, 9, 19, 3, 3, 1, 1, g=8),
        _shape(2, 128, 128, 28, 28, 3, 3, 1, 1, g=8, dtype="bf16"),
        _shape(2, 256, 256, 14, 14, 3, 3, 2, 1, g=8),
        _shape(2, 512, 512, 14, 14, 3, 3, 1, 1, g=8, dtype="bf16"),
        _shape(2, 24, 24, 14, 14, 1, 1, 1, 0, g=3),
        _shape(2, 256, 256, 64, 64, 3, 3, 1, 1, g=2),
        # Small and odd channel counts.
        _shape(2, 3, 32, 32, 32, 3, 3, 1, 1),
        _shape(1, 3, 5, 9, 9, 3, 3, 1, 1, dtype="bf16"),
        _shape(2, 6, 16, 14, 14, 3, 3, 1, 1),
        _shape(2, 4, 12, 14, 14, 3, 3, 1, 1, dtype="bf16"),
        _shape(2, 32, 5, 14, 14, 3, 3, 1, 1),
        _shape(1, 9, 48, 10, 10, 5, 5, 1, 2, g=3),
        _shape(1, 1, 1, 8, 8, 3, 3, 1, 1),
        # Large: a batch the non-grouped kernels split, and a tensor past the 32-bit
        # launch ABI that only a batch-split launch serves.
        _shape(36, 8, 8, 500, 500, 1, 1),
        _shape(128, 128, 128, 256, 256, 3, 3, 1, 1),
        _shape(64, 64, 64, 256, 256, 3, 3, 1, 1, dtype="bf16"),
        _shape(1, 64, 64, 1024, 1024, 3, 3, 1, 1),
        _shape(1, 4, 8, 4, 25600, 1, 11, 1, 0, pw=5),
    ]
    # Per compiled direct capability (variant, filter, stride, padding, dilation,
    # channels per group) and dtype, three problems of its own geometry: the catalog
    # holds many tunings of each, which all see the same problems.
    capabilities = {}
    for spec in specs:
        if spec.is_direct:
            key = (
                spec.dtype,
                spec.direct_variant,
                spec.filter_h,
                spec.filter_w,
                spec.stride_h,
                spec.stride_w,
                spec.pad_h,
                spec.pad_w,
                spec.dilation_h,
                spec.dilation_w,
                spec.cpg,
                spec.kpg,
            )
            capabilities.setdefault(key, spec)
    for spec in capabilities.values():
        for _ in range(3):
            cpg = spec.cpg or rng.choice((8, 16, 32, 48, 64))
            kpg = spec.kpg or rng.choice((16, 32, 64, 96, 128))
            groups = rng.choice((1, 2, 4, 8, 16, 32))
            if spec.direct_variant in ("std", "spatial", "tiled", "col"):
                cpg = kpg = 1
                groups = rng.choice((8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384))
            elif spec.direct_variant == "nongrouped":
                groups = 1
            hw = rng.choice((4, 7, 8, 14, 16, 28, 32, 56, 64, 112))
            rows.append(
                _shape(
                    rng.choice((1, 2, 4, 8)),
                    cpg * groups,
                    kpg * groups,
                    hw,
                    rng.choice((hw, hw + 3)),
                    spec.filter_h,
                    spec.filter_w,
                    spec.stride_h,
                    spec.pad_h,
                    d=spec.dilation_h,
                    g=groups,
                    dtype=spec.dtype,
                    sw=spec.stride_w,
                    pw=spec.pad_w,
                    dw=spec.dilation_w,
                )
            )
    for _ in range(100):
        y = rng.choice((1, 1, 3, 3, 3, 5, 7))
        s = rng.choice((1, 1, 1, 2, 2)) if y > 1 else rng.choice((1, 2))
        p = rng.choice((0, (y - 1) // 2, y // 2))
        groups = rng.choice((1, 1, 1, 2, 4, 8, 32))
        cpg = rng.choice((1, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256))
        kpg = rng.choice((1, 8, 16, 32, 64, 128, 256))
        if rng.random() < 0.15:
            groups = rng.choice((16, 32, 64, 96, 128, 256))
            cpg = kpg = 1
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
    return out + _serving_problems(specs, out)


def _adapter(arch: str, name: str):
    """The adapter's ``<arch>_conv_fwd_aot_<name>``: the engine's own entry point, so a
    rule only one architecture has reaches its vectors."""
    from builders.common import convolution_forward

    return getattr(convolution_forward, f"{arch}_conv_fwd_aot_{name}")


def _candidate_problems(spec):
    """Problems of ``spec``'s own geometry, in a fixed order, for
    :func:`_serving_problems` to try: its filter, stride, padding and dilation when it
    is direct; channels per group its channel tile and its output-channel tile divide
    (at least split_c channel tiles for a split-C kernel), or the compiled ones; output widths and heights multiples of its tile; a batch of
    one or two."""
    direct = spec.is_direct
    y, x = (spec.filter_h, spec.filter_w) if direct else (3, 3)
    s_h, s_w = (spec.stride_h, spec.stride_w) if direct else (1, 1)
    p_h, p_w = (spec.pad_h, spec.pad_w) if direct else (1, 1)
    d_h, d_w = (spec.dilation_h, spec.dilation_w) if direct else (1, 1)
    variant = spec.direct_variant if direct else None
    if variant == "nongrouped" or not direct:
        group_counts = (1,)
    elif variant == "grouped_halo":
        group_counts = (2, 4, 8)
    else:
        group_counts = (8, 32, 64)
    if direct and spec.cpg:
        channels = [(spec.cpg, spec.kpg)]
    else:
        c_tile = getattr(spec, "ck", None) or 16
        k_tile = getattr(spec, "tile_k_out", None) or 16
        # A split-C kernel needs at least split_c chunks of its channel tile.
        split_c = getattr(spec, "split_c", 0) or 0
        mults = (1, 2, 4) + ((split_c,) if split_c > 4 else ())
        channels = [(c_tile * m, k_tile * m) for m in mults]
    w_tile = getattr(spec, "tile_w", None) or 16
    h_tile = getattr(spec, "tile_h", None) or 8
    for groups in group_counts:
        for cpg, kpg in channels:
            for wo in (w_tile, 2 * w_tile, 3 * w_tile, 4 * w_tile):
                for ho in (h_tile, 2 * h_tile, wo):
                    hi = (ho - 1) * s_h + d_h * (y - 1) + 1 - 2 * p_h
                    wi = (wo - 1) * s_w + d_w * (x - 1) + 1 - 2 * p_w
                    if hi < 1 or wi < 1:
                        continue
                    for n in (1, 2):
                        yield _shape(
                            n,
                            cpg * groups,
                            kpg * groups,
                            hi,
                            wi,
                            y,
                            x,
                            s_h,
                            p_h,
                            d=d_h,
                            g=groups,
                            dtype=spec.dtype,
                            sw=s_w,
                            pw=p_w,
                            dw=d_w,
                        )


def _serving_problems(specs, problems) -> list:
    """For each instance, in catalog order, that serves none of ``problems`` (nor of
    the rows this adds), the first of its :func:`_candidate_problems` it serves, so
    the vectors check every instance's admission and launch plan. A ValueError when an
    instance serves none of them."""
    (arch,) = {spec.ARCH for spec in specs}
    serves = _adapter(arch, "serves")
    added = []
    for spec in specs:
        rows = [p for p in problems + added if p.dtype == spec.dtype]
        if any(serves(spec, p)[0] for p in rows):
            continue
        found = next(
            (p for p in _candidate_problems(spec) if serves(spec, p)[0]),
            None,
        )
        if found is None:
            raise ValueError(f"no synthetic problem serves {spec.instance_id}")
        added.append(found)
    return added


def _is_batch_split(spec, problem) -> bool:
    from builders.common.convolution_forward import conv_fwd_aot_batch_chunks

    return (
        spec.is_direct
        and spec.direct_variant in ("nongrouped", "grouped_halo")
        and conv_fwd_aot_batch_chunks(problem) is not None
    )


def _launch(step) -> list:
    """One launch: grid x, y, z, block x, then the AOT problem block of its kernel
    arguments, each as the u32 the kernel reads."""
    if tuple(step.block[1:]) != (1, 1):
        raise ValueError(f"a packaged launch has a one-dimensional block: {step.block}")
    names = [entry["name"] for entry in step.signature]
    if tuple(names[: len(_LEADING_ARGUMENTS)]) != _LEADING_ARGUMENTS:
        raise ValueError(f"unexpected leading kernel arguments: {names}")
    values = step.args.to_launch_values(0, 0, 0, 0, 0, 0)
    block = [values[name] & _U32 for name in names[len(_LEADING_ARGUMENTS) :]]
    return [*step.grid, step.block[0], len(block), *block]


def plan_values(spec, problem) -> list:
    """The adapter's launch plan of ``spec`` on a ``problem`` it serves, as integers:

    the batch-split launch count (0: one launch of the whole problem); per launch its
    images (N), grid x, y, z, block x, the argument count and the AOT problem block
    (``ConvArgs.to_launch_values`` after the buffers and their byte counts); then the
    workspace bytes (``split_workspace_bytes``) and whether a fold follows
    (``split_reduce_step``), with the fold's grid and block when it does.
    """
    from builders.common.convolution_forward import conv_fwd_aot_batch_chunks

    if _is_batch_split(spec, problem):
        chunks = conv_fwd_aot_batch_chunks(problem)
        values = [len(chunks)]
        for images in chunks:
            values += [images] + _launch(spec.launch_step(replace(problem, N=images)))
    else:
        values = [0, problem.N] + _launch(spec.launch_step(problem))
    values.append(spec.split_workspace_bytes(problem))
    if spec.is_split_c:
        reduce = spec.split_reduce_step(problem)
        values += [1, *reduce.grid, reduce.block[0]]
    else:
        values.append(0)
    return values


def _fnv(values) -> int:
    """FNV-1a 64 over ``values``, each as 8 little-endian two's-complement bytes."""
    h = 0xCBF29CE484222325
    for value in values:
        for byte in (value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def plan_digest(values) -> int:
    """A plan's digest: :func:`_fnv` of its :func:`plan_values`."""
    return _fnv(values)


def fold(digests) -> int:
    """A row's digest: :func:`_fnv` of its serving plans' digests, in catalog order."""
    return _fnv(digests)


def hex_bits(bits) -> str:
    """``bits`` (one bool per instance) four to a lowercase hex digit, the first
    instance the most significant bit of the first digit, padded with 0 bits."""
    bits = list(bits) + [False] * (-len(bits) % 4)
    return "".join(
        "%x" % (bits[i] << 3 | bits[i + 1] << 2 | bits[i + 2] << 1 | bits[i + 3])
        for i in range(0, len(bits), 4)
    )


def _field_value(spec, field) -> str:
    """``spec``'s value of ``field`` as the instance row states it: empty at the
    field's KMD default_value."""
    value = getattr(spec, field["name"])
    if "default_value" in field and value == field["default_value"]:
        return ""
    return str(value)


def _kind(spec, problem) -> tuple:
    """What tells one plan apart from another in shape: the family and variant, the
    nongrouped atom, a group merge, a batch split."""
    return (
        spec.direct_variant,
        spec.atom,
        spec.is_merged,
        spec.is_split_c,
        _is_batch_split(spec, problem),
    )


def _cpp_int(value: int) -> str:
    return f"{value}LL" if not -(1 << 31) <= value < (1 << 31) else str(value)


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


def _kernarg_names(specs) -> dict:
    """The AOT problem block's argument names of each family, by whether it is direct."""
    names = {}
    for spec in specs:
        signature = [entry["name"] for entry in spec.launch_signature()]
        names.setdefault(spec.is_direct, signature[len(_LEADING_ARGUMENTS) :])
    return names


def render(specs, problems) -> str:
    (arch,) = {spec.ARCH for spec in specs}
    conv_fwd_aot_serves = _adapter(arch, "serves")
    conv_fwd_aot_runtime_candidates = _adapter(arch, "runtime_candidates")
    cpp_namespace = namespace(arch)
    fields = kmd_fields()
    by_dtype = {
        dtype: [s for s in specs if s.dtype == dtype] for dtype in ("fp16", "bf16")
    }
    rows = []
    full = {}
    for index, problem in enumerate(problems):
        dtype_specs = by_dtype[problem.dtype]
        served = [conv_fwd_aot_serves(spec, problem)[0] for spec in dtype_specs]
        serving = [spec for spec, ok in zip(dtype_specs, served) if ok]
        selected_ids = {
            id(s) for s in conv_fwd_aot_runtime_candidates(problem, serving)
        }
        selected = [id(spec) in selected_ids for spec in dtype_specs]
        digests = []
        for position, spec in enumerate(dtype_specs):
            if not served[position]:
                continue
            values = plan_values(spec, problem)
            digests.append(plan_digest(values))
            full.setdefault(_kind(spec, problem), (index, position, values))
        rows.append((problem, hex_bits(served), hex_bits(selected), fold(digests)))
    kernargs = _kernarg_names(specs)

    def names_array(name, values):
        return (
            f"inline constexpr std::array<const char*, {len(values)}> {name}{{"
            + ", ".join(f'"{v}"' for v in values)
            + "};"
        )

    out = [
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier:  MIT",
        "",
        "#pragma once",
        "",
        "// clang-format off",
        "// GENERATED test data, do not edit: projects/hipdnn/tools/IngestorGenerator/tools/",
        "// make_conv_aot_test_vectors.py. The shipped catalog's KMD metadata, and what the",
        "// rocKE adapter rocke/library/builders/common/convolution_forward.py decides on",
        "// synthetic problems: conv_fwd_aot_serves, conv_fwd_aot_runtime_candidates and",
        "// each serving instance's launch plan.",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <vector>",
        "",
        f"namespace hip_kernel_provider::kernel_ingestor_engine::{cpp_namespace}",
        "{",
        "",
        "/// N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups.",
        "using ProblemValues = std::array<int64_t, 14>;",
        "",
        "/// The KMD fields, in KMD order.",
        names_array("FIELDS", [f["name"] for f in fields]),
        "",
        "/// The KMD fields that hold strings; every other one is an int.",
        names_array(
            "STRING_FIELDS", [f["name"] for f in fields if f["type"] == "string"]
        ),
        "",
        "/// The KMD fields with a default_value, as name=value;...",
        'inline constexpr const char* DEFAULT_FIELDS = "'
        + ";".join(
            f"{f['name']}={f['default_value']}" for f in fields if "default_value" in f
        )
        + '";',
        "",
        "/// The AOT problem block of each family's kernel arguments (after A, B, D and their",
        "/// byte counts), as the adapter's launch signature names it.",
        names_array("IMPLICIT_GEMM_KERNARGS", kernargs.get(False, [])),
        names_array("DIRECT_KERNARGS", kernargs.get(True, [])),
        "",
        "/// One catalog instance: its dtype and its KMD field values, comma-separated in",
        "/// FIELDS order, empty where the value is the field's default_value.",
        "struct CatalogInstance",
        "{",
        "    const char* dtype;",
        "    const char* values;",
        "};",
        "",
        "inline const std::vector<CatalogInstance>& catalogInstances()",
        "{",
        "    static const std::vector<CatalogInstance> s_instances{",
    ]
    for spec in specs:
        values = ",".join(_field_value(spec, f) for f in fields)
        out.append(f'        {{"{spec.dtype}", "{values}"}},')
    out += [
        "    };",
        "    return s_instances;",
        "}",
        "",
        "/// One problem. served and selected hold one bit per catalogInstances() entry of the",
        "/// problem's dtype, in catalog order, four to a hex digit, the first instance the",
        "/// digit's most significant bit: served those conv_fwd_aot_serves admits, selected",
        "/// those conv_fwd_aot_runtime_candidates keeps. digest folds the plan digest of each",
        "/// serving instance, in that order, with FNV-1a 64 (each as 8 little-endian bytes);",
        "/// a plan digest is FNV-1a 64 over the plan's values (see FullPlan), likewise.",
        "struct ServedRow",
        "{",
        "    ProblemValues problem;",
        "    const char* dtype;",
        "    const char* served;",
        "    const char* selected;",
        "    uint64_t digest;",
        "};",
        "",
        "inline const std::vector<ServedRow>& servedMatrix()",
        "{",
        "    static const std::vector<ServedRow> s_rows{",
    ]
    for problem, served, selected, digest in rows:
        values = ", ".join(str(v) for v in _problem_values(problem))
        out.append(
            f'        {{{{{values}}}, "{problem.dtype}", "{served}", "{selected}", '
            f"{digest}ULL}},"
        )
    out += [
        "    };",
        "    return s_rows;",
        "}",
        "",
        "/// A plan in full: servedMatrix()[row] with the instance-th catalog instance of the",
        "/// row's dtype. Values: the batch-split launch count (0: one launch of the whole",
        "/// problem); per launch its images, grid x, y, z, block x, the argument count and",
        "/// the AOT problem block (as u32); the workspace bytes; whether a split-C fold",
        "/// follows, then the fold's grid x, y, z and block x.",
        "struct FullPlan",
        "{",
        "    size_t row;",
        "    size_t instance;",
        "    std::vector<int64_t> values;",
        "};",
        "",
        "inline const std::vector<FullPlan>& fullPlans()",
        "{",
        "    static const std::vector<FullPlan> s_plans{",
    ]
    for kind in sorted(full, key=repr):
        row, instance, values = full[kind]
        rendered = ", ".join(_cpp_int(v) for v in values)
        out.append(f"        {{{row}, {instance}, {{{rendered}}}}},")
    out += [
        "    };",
        "    return s_plans;",
        "}",
        "",
        f"}} // namespace hip_kernel_provider::kernel_ingestor_engine::{cpp_namespace}",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root",
        type=Path,
        default=_cfg.DEFAULT_ROCKE_ROOT,
        help="rocKE tree (platform/, library/); default: the in-repo tree.",
    )
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
        if spec.instance_id in seen:
            # The header must list the shipped catalog exactly, not a deduplication.
            print(f"duplicate instance: {spec.instance_id}", file=sys.stderr)
            return 1
        seen.add(spec.instance_id)
        specs.append(spec)
    if len({spec.ARCH for spec in specs}) != 1:
        print("the instances are of more than one engine", file=sys.stderr)
        return 1
    specs.sort(key=lambda s: (s.dtype, s.instance_id))
    problems = synthetic_problems(specs)
    args.output.write_text(render(specs, problems), encoding="utf-8")
    print(f"{len(specs)} instances, {len(problems)} problems written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
