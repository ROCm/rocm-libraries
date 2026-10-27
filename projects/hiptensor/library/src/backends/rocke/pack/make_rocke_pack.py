#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################
"""Builds the rocKE kernel pack of hipTensor's rocKE backend.

Runs at hipTensor build time only (HIPTENSOR_ENABLE_ROCKE_BACKEND=ON). It imports rocKE from a
source tree, builds the shape-generic kernels the backend launches, compiles them with the COMGR
library rocKE loads, reads every kernel's argument layout from its code object metadata and
writes one pack per GPU target:

    <out>/<arch>/rocke.htpk          binary pack: header, kernel table, argument and attribute
                                     tables, route table, string table, code objects
    <out>/<arch>/rocke.index.json    the same index as JSON (tooling only, never read at run time)

hipTensor's C++ runtime reads the binary pack (rocke_pack.cpp); it never needs Python.

Kernel families (attribute "family"):

    ew       strided elementwise kernel (permute, elementwise binary / trinary, contraction
             operand packing and epilogue); attribute math=1: the rows that also implement the
             trigonometric and hyperbolic operators, their inverses, ceil and floor
    reduce   strided reduction kernels (row, col) and the split finalize kernel (math as for ew)
    gemm     universal GEMM bodies with an f32 output (routes h16: f16 / bf16 operands,
             x6: f32 operands split into three bf16 parts, cf32: complex f32 operands)
    gett64   f64 / complex f64 strided batched GEMM with runtime sizes
    gett     the launches of rocKE's tuned GETT route table (f16 / bf16 contractions of fixed
             shapes): pack kernels and the GETT kernel, with a launch record per problem

With --jit-recipes it also writes the recipes of the backend's JIT (HIPTENSOR_ENABLE_ROCKE_JIT):

    <out>/rocke.recipes              one parametric rocKE recipe (CBOR, rocke.recipe/v1) per
                                     problem-specialized elementwise kernel class
                                     (kernels.common.hiptensor_aot.jit_ew), rolled over every
                                     geometry constant; every pack's route table records the
                                     file's SHA-256 (jit.recipes.sha256)

The route table carries the tuning constants of rocKE's body rules and, per tuned GETT problem,
``gett.tuned.<hash>`` (the launch record) and ``gett.tuned.<hash>.key`` (the canonical problem,
see :func:`gett_problem_key`). With --select-models, it also carries the kernel selection models
of the target, ``select.model.<cell>`` (``<dir>/<arch>/<cell>.model``, written by the selection
training flow; see rocke_select.hpp for the format).

Usage:
    make_rocke_pack.py --rocke-src DIR --arch gfx950 [--arch gfx942] --out DIR
                        [--jobs N] [--cache-dir DIR] [--compile-timeout S]
                        [--select-models DIR] [--jit-recipes]
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import struct
import sys
import time

# Must match hiptensor::rockebackend::kKernelAbi and the reader in rocke_pack.cpp.
KERNEL_ABI = 1
PACK_FORMAT_VERSION = 1
PACK_MIN_READER = 1
PACK_MAGIC = b"HTPKIDX\0"

HEADER_SIZE = 160
KERNEL_RECORD_SIZE = 96
ARG_RECORD_SIZE = 16
ATTR_RECORD_SIZE = 16
ROUTE_RECORD_SIZE = 8

ARG_KIND_POINTER = 0
ARG_KIND_VALUE = 1

# Must match the reader in rocke_jit.cpp.
RECIPES_MAGIC = b"HTJITRC\0"
RECIPES_FORMAT_VERSION = 1
RECIPES_MIN_READER = 1
RECIPES_HEADER_SIZE = 40
ATTR_KIND_INT = 0
ATTR_KIND_STRING = 1

# Emitter knobs some rocKE modules read at import time: never let the build environment change
# the kernels.
_SCRUB_PREFIXES = ("ROCKE_EXP_", "CK_", "ROCKE_FP8_", "ROCKE_GEMM_", "TENSOR_GETT_")


def _scrub_env():
    for k in list(os.environ):
        if k.startswith(_SCRUB_PREFIXES):
            del os.environ[k]
    os.environ["ROCKE_BACKEND"] = "python"


def _setup_paths(src):
    for p in (os.path.join(src, "library"), os.path.join(src, "platform", "python")):
        if not os.path.isdir(p):
            raise SystemExit(f"rocKE source tree {src}: missing {p}")
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------
# Kernel descriptors: (family, params) tuples, picklable, rebuilt in the compile workers.
# ------------------------------------------------------------------------------------------------

EW_ROWS_1 = [  # permute / pack: (A, D)
    ("f16", "f16"),
    ("f16", "f32"),
    ("bf16", "bf16"),
    ("bf16", "f32"),
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("f64", "f64"),
    # contraction operands converted to f64 (the F64 route)
    ("f16", "f64"),
    ("bf16", "f64"),
    ("f32", "f64"),
]
EW_ROWS_2 = [  # binary: (A, C, D); epilogues read an f32 / f64 A
    ("f16", "f16", "f16"),
    ("f16", "f16", "f32"),
    ("bf16", "bf16", "bf16"),
    ("bf16", "bf16", "f32"),
    ("f32", "f32", "f32"),
    ("f32", "f16", "f16"),
    ("f32", "bf16", "bf16"),
    ("f64", "f64", "f64"),
    # epilogues of the F64 route
    ("f64", "f16", "f16"),
    ("f64", "bf16", "bf16"),
    ("f64", "f32", "f32"),
]
EW_ROWS_3 = [  # trinary: (A, B, C, D)
    ("f16", "f16", "f16", "f16"),
    ("f16", "f16", "f16", "f32"),
    ("bf16", "bf16", "bf16", "bf16"),
    ("bf16", "bf16", "bf16", "f32"),
    ("f32", "f32", "f32", "f32"),
    ("f64", "f64", "f64", "f64"),
]
# complex (component types): pack (A, D) and epilogue (A, C, D) rows
EW_ROWS_C1 = [("f32", "f32"), ("f64", "f64"), ("f32", "f64")]
EW_ROWS_C2 = [("f32", "f32", "f32"), ("f64", "f64", "f64"), ("f64", "f32", "f32")]

# Contractions of mixed data types (A, B, C, D of different types): the operand conversions of
# the pack step (to f16 / bf16 for the 16-bit compute types: one rounding to nearest even from
# any type) and the epilogue rows of every C / D type pair, after an f32 (GEMM routes) or an
# f64 (F64 route) result. Kernels of the pack only: the JIT keeps their launches on the pack
# kernel (no recipe classes).
_REAL = ("f16", "bf16", "f32", "f64")
EW_ROWS_1_MIXED = [
    (a, d) for d in ("f16", "bf16") for a in _REAL if (a, d) not in EW_ROWS_1
]
EW_ROWS_2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in _REAL
    for d in _REAL
    if (t, c, d) not in EW_ROWS_2
]
EW_ROWS_C2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in ("f32", "f64")
    for d in ("f32", "f64")
    if (t, c, d) not in EW_ROWS_C2
]
REDUCE_ROWS = [("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"), ("f64", "f64")]

GEMM_BODIES = {
    # route: bodies
    "h16": ("RIW", "RIW_SK", "C3", "Q3_SK"),
    "x6": ("RIW", "RIW_SK", "C3"),
    "cf32": ("RIW", "RIW_SK", "Q3_SK"),
}
GEMM_ARCHES = ("gfx950",)
# The body of the 64-bit-address GEMM kernels (rocke.dispatch.gemm.cf32 picks it for operands
# past the DMA buffer range: 64-bit tile origins need the non-persistent grid).
GEMM_WIDE_BODY = "RIW"
GETT64_CLASS_ARCHES = ("gfx90a", "gfx942", "gfx950")
# The thresholds of rocKE's emulated-f32 body rule (rocke.dispatch.gemm.cf32), shipped in the
# route table so the C++ runtime picks the body the rocKE rule picks.
GEMM_ROUTE_KEYS = (
    "CF32_SK_ALWAYS_WAVES",
    "CF32_SK_MAX_WAVES",
    "CF32_SK_MAX_TAIL",
    "CF32_SMALL_MAX_WAVES",
)


def rocke_features():
    """The optional kernel families the rocKE tree provides: ``{family: reason missing or ""}``.

    Older rocKE trees lack some of them (for example the emulated-f32 GEMM routes); the pack is
    then built without them, every one is reported, and the C++ runtime declines the problems
    that would need them."""
    out = {}
    try:
        from rocke.dispatch.gemm import cf32  # noqa: F401
        from rocke.instances.common.gemm_prologue import Prologue  # noqa: F401

        out["gemm"] = ""
    except ImportError as e:
        out["gemm"] = f"no emulated-f32 GEMM routes ({e})"
    try:
        from rocke.dispatch.families import tensor_gett as tg

        for name in (
            "TENSOR_GETT_TUNED_FILE",
            "tuned_entry_problem",
            "gett_operand_sources",
        ):
            getattr(tg, name)
        out["gett"] = ""
    except (ImportError, AttributeError) as e:
        out["gett"] = f"no tuned GETT route table ({e})"
    try:
        from rocke.instances.common import gett_f64 as g

        for name in (
            "runtime_template",
            "TUNED_CONFIGS",
            "K_BLOCK_MIN_K",
            "_with_k_block",
        ):
            getattr(g, name)
        out["gett64"] = ""
    except (ImportError, AttributeError) as e:
        out["gett64"] = f"no runtime-size f64 GETT ({e})"
    try:
        from kernels.common.hiptensor_aot import jit_ew

        for name in ("EwJitSpec", "jit_classes", "roll_jit_ew", "EW_JIT_ABI"):
            getattr(jit_ew, name)
        out["jit"] = ""
    except (ImportError, AttributeError) as e:
        out["jit"] = f"no JIT recipes of the elementwise kernel ({e})"
    return out


def ew_params(idx64, mixed=True):
    """The rows of the generic elementwise kernel (EwSpec arguments); ``mixed`` adds the rows of
    mixed-type contractions."""
    rows1 = EW_ROWS_1 + (EW_ROWS_1_MIXED if mixed else [])
    rows2 = EW_ROWS_2 + (EW_ROWS_2_MIXED if mixed else [])
    rowsc2 = EW_ROWS_C2 + (EW_ROWS_C2_MIXED if mixed else [])
    out = []
    for a, d in rows1:
        out.append(dict(arity=1, a=a, d=d, idx64=idx64))
    for a, c, d in rows2:
        out.append(dict(arity=2, a=a, c=c, d=d, idx64=idx64))
    for a, bb, c, d in EW_ROWS_3:
        out.append(dict(arity=3, a=a, b=bb, c=c, d=d, idx64=idx64))
    for a, d in EW_ROWS_C1:
        out.append(dict(arity=1, a=a, d=d, cplx=True, idx64=idx64))
    for a, c, d in rowsc2:
        out.append(dict(arity=2, a=a, c=c, d=d, cplx=True, idx64=idx64))
    # the same real rows with the math operators (trigonometric, hyperbolic, their inverses,
    # floor, ceil): separate kernels, so the rows without them keep their registers
    out += [dict(p, math=True) for p in out if not p.get("cplx")]
    return out


def kernel_descriptors(arch, num_cus):
    out = []
    features = rocke_features()
    for idx64 in (False, True):
        out.extend(("ew", p) for p in ew_params(idx64))
        for math in (False, True):
            for a, d in REDUCE_ROWS:
                for mode in ("row", "col", "final"):
                    p = dict(mode=mode, a=a, d=d, idx64=idx64)
                    if math:
                        p["math"] = True
                    out.append(("reduce", p))
    if arch in GEMM_ARCHES and not features["gemm"]:
        for route, bodies in GEMM_BODIES.items():
            dts = ("fp16", "bf16") if route == "h16" else ("bf16",)
            for dt in dts:
                for body in bodies:
                    for batched in (False, True):
                        out.append(
                            (
                                "gemm",
                                dict(route=route, dtype=dt, body=body, batched=batched),
                            )
                        )
                # the 64-bit-address body for operands past the DMA buffer range or a result
                # past 2^31 elements (rocKE's wide_addr adjustment; one batch per launch)
                out.append(
                    (
                        "gemm",
                        dict(
                            route=route,
                            dtype=dt,
                            body=GEMM_WIDE_BODY,
                            batched=False,
                            wide=True,
                        ),
                    )
                )
    for key in tuned_gett_keys(arch) if not features["gett"] else []:
        out.append(("gett_plan", dict(key=key)))
    if arch in GETT64_CLASS_ARCHES and not features["gett64"]:
        from rocke.instances.common.gett_f64 import TUNED_CONFIGS

        for cplx in (False, True):
            kind = "complex" if cplx else "real"
            classes = sorted(
                c for (a, k, c) in TUNED_CONFIGS if a == arch and k == kind
            )
            for cls in classes + ["default"]:
                for dclass in ("m", "n", "x"):
                    for bilinear in (False, True):
                        for kblock in (False, True):
                            out.append(
                                (
                                    "gett64",
                                    dict(
                                        cplx=cplx,
                                        cls=cls,
                                        dclass=dclass,
                                        bilinear=bilinear,
                                        kblock=kblock,
                                    ),
                                )
                            )
    return out


def _gemm_spec(params, arch):
    """The universal GEMM spec of a gemm descriptor, with its prologue attached."""
    from dataclasses import replace

    from rocke.dispatch.gemm.cf32 import cf32_body
    from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        UniversalGemmSpec,
        is_valid_spec,
    )

    route, dt, body = params["route"], params["dtype"], params["body"]
    wide = bool(params.get("wide", False))
    tile, trait, bs = cf32_body(body, arch)
    spec = UniversalGemmSpec(
        name=f"hiptensor_{route}_{body.lower()}{'_w64' if wide else ''}",
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
        block_size=bs,
        batched=bool(params["batched"]),
    )
    if wide:
        # rocKE's own launch adjustment for 64-bit tile origins (launched on extents padded to
        # tile multiples: no padding adjustments apply)
        from rocke.dispatch.gemm.cf32 import _adjust_k

        if params["batched"]:
            return None, "the 64-bit-address body has no batched form"
        spec = _adjust_k(spec, tile.tile_m, tile.tile_n, True)
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        return None, why
    if route == "h16":
        pro = Prologue(src="same", recipe="cvt", out_f32=True)
    else:
        pro = Prologue(
            src="f32",
            recipe="x6",
            out_f32=True,
            split_at="use",
            cplx_b=(route == "cf32"),
            psub=True,
        )
    attach_prologue(spec, pro)
    _ = replace
    return spec, "ok"


def _gett64_spec(params, arch, num_cus):
    from dataclasses import replace

    from rocke.instances.common.gett_f64 import (
        TUNED_CONFIGS,
        _with_k_block,
        arch_spec,
        gemm_problem,
        is_valid_spec,
        runtime_template,
    )

    cplx = params["cplx"]
    M, N, K, L = 64, 64, 64, 2
    # A [L][M][K] and B [L][N][K] packed with K contiguous (K padded to even by the pack step);
    # D (and C) [L][M][N] with M contiguous ("m"), N contiguous ("n") or neither ("x").
    d_strides = {"m": (1, M), "n": (N, 1), "x": (2, 2 * M)}[params["dclass"]]
    prob = gemm_problem(
        M,
        N,
        K,
        a_strides=(K, 1),
        b_strides=(K, 1),
        d_strides=d_strides,
        batch=L,
        batch_strides=(M * K, N * K, 4 * M * N),
        complex_=cplx,
    )
    kind = "complex" if cplx else "real"
    cfg = (
        {}
        if params["cls"] == "default"
        else dict(TUNED_CONFIGS[(arch, kind, params["cls"])])
    )
    cfg.pop("split_k", None)
    spec = arch_spec(
        prob, arch, bilinear=bool(params["bilinear"]), runtime_sizes=True, **cfg
    )
    spec = replace(spec, split_k=1, problem=runtime_template(prob))
    if params["kblock"]:
        kb = _with_k_block(spec, arch)
        if kb.k_block == spec.k_block:
            return None, "no k_block variant"
        spec = kb
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        return None, why
    return spec, "ok"


def tuned_gett_keys(arch):
    """Keys of rocKE's shipped tuned GETT route table for ``arch``."""
    from pathlib import Path

    from rocke.dispatch.families import tensor_gett as tg

    path = Path(tg.__file__).with_name(tg.TENSOR_GETT_TUNED_FILE)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return sorted(raw.get("entries", {}).get(arch, {}))


def fnv1a64(text):
    h = 0xCBF29CE484222325
    for byte in text.encode():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def gett_problem_key(dtype, bilinear, tensors):
    """Canonical key of a contraction for the tuned GETT plans; hipTensor's rocke_plan.cpp
    builds the same string. ``tensors``: D, A, B as lists of ``(mode, extent, stride)``.
    Extent-1 modes are dropped, each tensor's modes are ordered by stride, and modes are named
    by first appearance in that order (D, then A, then B)."""
    names = {}
    parts = [f"{dtype}|{'bilinear' if bilinear else 'scale'}"]
    for label, modes in zip("DAB", tensors):
        ms = sorted((m for m in modes if m[1] != 1), key=lambda m: m[2])
        for m, _e, _s in ms:
            names.setdefault(m, f"x{len(names)}")
        parts.append(label + ":" + ",".join(f"{names[m]}:{e}:{st}" for m, e, st in ms))
    return "|".join(parts)


def _row_major(modes, ext):
    st, s = {}, 1
    for m in reversed(list(modes)):
        st[m] = s
        s *= int(ext[m])
    return st


def build_gett_plan(params, arch):
    """The kernels and the launch record of one tuned GETT problem.

    Record: ``v1;ws=<workspace bytes>;steps=<n>`` then per launch
    ``|k=<kernel>;g=<x>,<y>,<z>;b=<threads>;a=<name>=<value>,...`` where a value is
    ``<buffer>+<byte offset>`` (buffers A, B, C, O (the output), W (the workspace)), ``#<int>``
    or ``@a`` / ``@b`` (the host scalars alpha / beta, f32)."""
    from rocke.dispatch.families import tensor_gett as tg
    from rocke.instances.common.tensor import GettKnobs, TensorGettSpec

    key = params["key"]
    prob = tg.tuned_entry_problem(arch, key)
    if prob is None:
        return None, "no tuned entry"
    ext = {k: int(v) for k, v in prob["extents"].items()}
    dt = "fp16" if prob["dtype"] in ("f16", "fp16") else prob["dtype"]
    bilinear = bool(prob["bilinear"])
    spec = TensorGettSpec.make(
        prob["modes_a"],
        prob["modes_b"],
        prob["modes_d"],
        ext,
        dtype=dt,
        knobs=GettKnobs(bilinear=bilinear),
    )
    req = tg.TensorGettRequest(problem=spec, arch=arch)
    if tg.tensor_gett_route_key(req) != key:
        return None, "the tuned entry's key does not reproduce"
    gp = tg.plan_tensor_gett(req)
    kernels = []
    steps = []
    ws_off = {}
    off = 0
    for st in gp.packs:
        k = tg.build_pack_step(st, arch)
        names = [p.name for p in k.params]
        if sorted(names) != ["X", "Y"]:
            return None, f"pack kernel params {names}"
        ws_off[st.operand] = off
        steps.append(
            (
                k,
                tuple(st.grid),
                tuple(st.block),
                {"X": f"{st.operand}+0", "Y": f"W+{off}"},
            )
        )
        off += -(-int(st.workspace_bytes) // 256) * 256
        kernels.append(k)
    kd = gp.gett.candidate.build(gp.gett.spec, arch)
    pl = gp.gett.spec.plan()
    binds = {
        "C": "O+0",
        "M": f"#{pl.M}",
        "N": f"#{pl.N}",
        "K": f"#{pl.K}",
        "alpha": "@a",
    }
    if bilinear:
        binds.update(Cin="C+0", beta="@b")
    for pname, (kind, v) in zip(("A", "B"), tg.gett_operand_sources(gp)):
        if kind == "pack":
            binds[pname] = f"W+{ws_off[gp.packs[v].operand]}"
        else:
            binds[pname] = f"{v}+0"
    names = [p.name for p in kd.params]
    if set(names) != set(binds):
        return (
            None,
            f"GETT kernel params {names} are not the bound ones {sorted(binds)}",
        )
    steps.append((kd, tuple(gp.gett.grid), tuple(gp.gett.block), binds))
    kernels.append(kd)
    # The canonical key with the packed row-major strides the tuned problem was measured with
    # (in hipTensor: the same memory layout, mode lists reversed and column-major strides).
    tensors = []
    for modes in (prob["modes_d"], prob["modes_a"], prob["modes_b"]):
        st = _row_major(modes, ext)
        tensors.append([(m, ext[m], st[m]) for m in modes])
    pkey = gett_problem_key("f16" if dt == "fp16" else dt, bilinear, tensors)
    rec = [f"v1;ws={off};steps={len(steps)}"]
    for k, grid, block, b in steps:
        rec.append(
            f"k={k.name};g={grid[0]},{grid[1]},{grid[2]};b={block[0]};a="
            + ",".join(f"{n}={v}" for n, v in sorted(b.items()))
        )
    return dict(kernels=kernels, key=pkey, record="|".join(rec)), "ok"


def build_kernel(family, params, arch, num_cus):
    """(KernelDef, attributes) of a descriptor, or (None, reason) when it does not apply."""
    if family == "ew":
        from kernels.common.hiptensor_aot.ew import EW_ABI, EwSpec, build_ew

        spec = EwSpec(**params)
        attrs = dict(
            family="ew",
            abi=EW_ABI,
            arity=spec.arity,
            cplx=int(spec.cplx),
            idx64=int(spec.idx64),
            math=int(spec.math),
            a=spec.a,
            b=spec.b or "",
            c=spec.c or "",
            d=spec.d,
            block=spec.block,
            # capability: rank bucket R8, i32 coordinates
            max_rank=8,
            limit_index=(1 << 31) - 1,
        )
        return build_ew(spec), attrs
    if family == "reduce":
        from kernels.common.hiptensor_aot.reduce import (
            REDUCE_ABI,
            ReduceSpec,
            build_reduce,
        )

        spec = ReduceSpec(**params)
        attrs = dict(
            family="reduce",
            abi=REDUCE_ABI,
            mode=spec.mode,
            a=spec.a,
            d=spec.d,
            idx64=int(spec.idx64),
            math=int(spec.math),
            block=spec.block,
            max_rank=8,
            limit_index=(1 << 31) - 1,
        )
        return build_reduce(spec), attrs
    if family == "gemm":
        from rocke.instances.common.gemm_universal import (
            DTL_BUFFER_NUM_RECORDS,
            build_universal_gemm,
            streamk_workspace_bytes,
        )

        spec, why = _gemm_spec(params, arch)
        if spec is None:
            return None, why
        t, tr = spec.tile, spec.trait
        fb, pb = streamk_workspace_bytes(spec)
        attrs = dict(
            family="gemm",
            abi="hiptensor-aot-gemm/v1",
            route=params["route"],
            dtype=params["dtype"],
            body=params["body"],
            batched=int(params["batched"]),
            tile_m=t.tile_m,
            tile_n=t.tile_n,
            tile_k=t.tile_k,
            block=spec.block_size,
            persistent_ctas=int(tr.persistent_ctas or 0),
            streamk=int(bool(tr.streamk)),
            sk_flags_bytes=fb,
            sk_part_bytes=pb,
            wide=int(bool(params.get("wide", False))),
            # capability: launched on extents padded to tile multiples; the operands within
            # the DMA buffer range, C within the i32 element offsets, rows of tiles on grid y.
            # The 64-bit-address body: only one tile's operand rows within the DMA buffer
            # range and one tile's C rows within the i32 element offsets.
            limit_operand_bytes=DTL_BUFFER_NUM_RECORDS,
            limit_c_elements=(1 << 31) - 1,
            limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
            limit_c_tile_elements=1 << 31,
            limit_grid_yz=65535,
        )
        return build_universal_gemm(spec, arch), attrs
    if family == "gett64":
        from rocke.instances.common import gett_f64 as g64
        from rocke.instances.common.gett_f64 import build_gett_f64

        spec, why = _gett64_spec(params, arch, num_cus)
        if spec is None:
            return None, why
        attrs = dict(
            family="gett64",
            abi="rocke-gett-f64/v2",
            cplx=int(params["cplx"]),
            cls=params["cls"],
            dclass=params["dclass"],
            bilinear=int(params["bilinear"]),
            kblock=int(spec.k_block or 0),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            block=int(spec.threads),
            # capability: bytes one buffer resource addresses, batch on grid y
            limit_span_bytes=g64._MAX_SPAN,
            limit_batch=g64.MAX_GRID_Y,
        )
        return build_gett_f64(spec, arch=arch), attrs
    raise ValueError(family)


# ------------------------------------------------------------------------------------------------
# Code object metadata (NT_AMDGPU_METADATA, msgpack)
# ------------------------------------------------------------------------------------------------


class _MsgPack:
    def __init__(self, data):
        self.d = data
        self.i = 0

    def _take(self, n):
        v = self.d[self.i : self.i + n]
        if len(v) != n:
            raise ValueError("truncated msgpack")
        self.i += n
        return v

    def _u(self, n):
        return int.from_bytes(self._take(n), "big")

    def value(self):
        t = self._u(1)
        if t <= 0x7F:
            return t
        if t >= 0xE0:
            return t - 0x100
        if 0x80 <= t <= 0x8F:
            return self._map(t & 0x0F)
        if 0x90 <= t <= 0x9F:
            return [self.value() for _ in range(t & 0x0F)]
        if 0xA0 <= t <= 0xBF:
            return self._take(t & 0x1F).decode()
        simple = {0xC0: None, 0xC2: False, 0xC3: True}
        if t in simple:
            return simple[t]
        if t in (0xC4, 0xC5, 0xC6):
            return bytes(self._take(self._u({0xC4: 1, 0xC5: 2, 0xC6: 4}[t])))
        if t == 0xCA:
            return struct.unpack(">f", self._take(4))[0]
        if t == 0xCB:
            return struct.unpack(">d", self._take(8))[0]
        if 0xCC <= t <= 0xCF:
            return self._u(1 << (t - 0xCC))
        if 0xD0 <= t <= 0xD3:
            n = 1 << (t - 0xD0)
            return int.from_bytes(self._take(n), "big", signed=True)
        if t in (0xD9, 0xDA, 0xDB):
            return self._take(self._u({0xD9: 1, 0xDA: 2, 0xDB: 4}[t])).decode()
        if t in (0xDC, 0xDD):
            return [self.value() for _ in range(self._u(2 if t == 0xDC else 4))]
        if t in (0xDE, 0xDF):
            return self._map(self._u(2 if t == 0xDE else 4))
        raise ValueError(f"unsupported msgpack type 0x{t:02x}")

    def _map(self, n):
        out = {}
        for _ in range(n):
            k = self.value()
            out[k] = self.value()
        return out


def code_object_metadata(co):
    """The decoded NT_AMDGPU_METADATA note of an ELF64 code object."""
    if co[:4] != b"\x7fELF" or co[4] != 2:
        raise ValueError("not an ELF64 code object")
    shoff = struct.unpack_from("<Q", co, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", co, 0x3A)
    for i in range(shnum):
        base = shoff + i * shentsize
        sh_type = struct.unpack_from("<I", co, base + 4)[0]
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", co, base + 0x18)
        p = off
        while p + 12 <= off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", co, p)
            name = co[p + 12 : p + 12 + namesz].rstrip(b"\0")
            dp = p + 12 + ((namesz + 3) & ~3)
            if name == b"AMDGPU" and ntype == 32:
                return _MsgPack(co[dp : dp + descsz]).value()
            p = dp + ((descsz + 3) & ~3)
    raise ValueError("code object has no NT_AMDGPU_METADATA note")


def kernel_metadata(co, name):
    md = code_object_metadata(co)
    for k in md.get("amdhsa.kernels", []):
        if k.get(".name") == name:
            return k
    raise ValueError(f"kernel {name} not in the code object metadata")


# ------------------------------------------------------------------------------------------------
# Compile workers
# ------------------------------------------------------------------------------------------------


def _worker_init(src):
    _scrub_env()
    _setup_paths(src)


def _compile_one(job):
    """Builds, lowers and compiles one descriptor. Returns a result dict (never raises)."""
    family, params, arch, num_cus, cache_dir, compiler_id = job
    t0 = time.time()
    try:
        if family == "gett_plan":
            plan, why = build_gett_plan(params, arch)
            if plan is None:
                return dict(status="skip", reason=why, family=family, params=params)
            outs = [
                _compile_kernel(
                    k,
                    dict(family="gett", abi="rocke-tensor-gett/v1"),
                    arch,
                    cache_dir,
                    compiler_id,
                )
                for k in plan["kernels"]
            ]
            route = fnv1a64(plan["key"])
            return dict(
                status="ok",
                family=family,
                params=params,
                many=outs,
                routes={
                    f"gett.tuned.{route}": plan["record"],
                    f"gett.tuned.{route}.key": plan["key"],
                },
                seconds=time.time() - t0,
            )
        kernel, attrs = build_kernel(family, params, arch, num_cus)
        if kernel is None:
            return dict(status="skip", reason=attrs, family=family, params=params)
        r = _compile_kernel(kernel, attrs, arch, cache_dir, compiler_id)
        r.update(status="ok", family=family, params=params, seconds=time.time() - t0)
        return r
    except Exception as e:  # noqa: BLE001 - reported as a manifest error by the caller
        import traceback

        return dict(
            status="error",
            reason=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            family=family,
            params=params,
        )


def _compile_kernel(kernel, attrs, arch, cache_dir, compiler_id):
    """Lowers and compiles one kernel through the content-addressed cache; raises on failure."""
    from rocke.helpers.compile import lower_kernel_for_comgr
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    if True:
        ci = lower_kernel_for_comgr(kernel, arch=arch, backend="python")
        if ci.fresh_process:
            raise RuntimeError(
                f"{kernel.name} needs a fresh compile process (history-dependent codegen)"
            )
        key = hashlib.sha256(
            "\n".join(
                [compiler_id, ci.isa, " ".join(ci.options), ci.llvm_text]
            ).encode()
        ).hexdigest()
        co = None
        cached = os.path.join(cache_dir, key + ".co") if cache_dir else None
        if cached and os.path.isfile(cached):
            with open(cached, "rb") as f:
                co = f.read()
        if co is None:
            co, _ = build_hsaco_from_llvm_ir(
                ci.llvm_text, isa=ci.isa, options=list(ci.options), fresh_process=False
            )
            if cached:
                tmp = cached + f".{os.getpid()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(co)
                os.replace(tmp, cached)
        md = kernel_metadata(co, kernel.name)
        return dict(
            name=kernel.name,
            attrs=attrs,
            co=co,
            md=_md_subset(md),
            param_names=[p.name for p in kernel.params],
        )


def _md_subset(md):
    args = []
    for a in md.get(".args", []):
        kind = a.get(".value_kind", "")
        if kind.startswith("hidden_"):
            continue
        args.append(
            dict(
                name=a.get(".name", ""),
                offset=int(a[".offset"]),
                size=int(a[".size"]),
                kind=ARG_KIND_POINTER if kind == "global_buffer" else ARG_KIND_VALUE,
            )
        )
    return dict(
        args=args,
        kernarg_size=int(md.get(".kernarg_segment_size", 0)),
        group_segment=int(md.get(".group_segment_fixed_size", 0)),
        private_segment=int(md.get(".private_segment_fixed_size", 0)),
        max_flat_workgroup_size=int(md.get(".max_flat_workgroup_size", 0)),
        wavefront_size=int(md.get(".wavefront_size", 64)),
        sgpr=int(md.get(".sgpr_count", 0)),
        vgpr=int(md.get(".vgpr_count", 0)),
        agpr=int(md.get(".agpr_count", 0)),
        symbol=md.get(".symbol", ""),
    )


# ------------------------------------------------------------------------------------------------
# JIT recipes
# ------------------------------------------------------------------------------------------------


def jit_class_params():
    """The JIT kernel classes of every generic elementwise row: picklable (base, vec, unit)."""
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    # the math rows (one vec-1 class each) only with 32-bit offsets: their recipes carry the
    # f64 programs of every math operator per input, and a launch of the 64-bit-offset math
    # rows keeps its kernel-pack kernel
    bases = [
        EwSpec(**p)
        for idx64 in (False, True)
        for p in ew_params(idx64, mixed=False)
        if not (idx64 and p.get("math"))
    ]
    return [(dict(c.base.__dict__), c.vec, tuple(c.unit)) for c in jit_classes(bases)]


def _recipe_source_hash(src):
    """What a rolled recipe depends on: the kernel builders, the IR builder and the roller."""
    h = hashlib.sha256()
    for sub in (
        "library/kernels/common/hiptensor_aot",
        "platform/python/rocke/portable_ir/src",
        "platform/python/rocke/portable_ir/utils",
    ):
        d = os.path.join(src, sub)
        for root, _dirs, files in sorted(os.walk(d)):
            for fn in sorted(files):
                if fn.endswith(".py"):
                    with open(os.path.join(root, fn), "rb") as f:
                        h.update(fn.encode() + f.read().replace(b"\r\n", b"\n"))
    for name in ("core/ir.py", "helpers/elementary.py"):
        path = os.path.join(src, "platform/python/rocke", name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _roll_one(job):
    """Rolls one JIT class into a recipe (CBOR). Returns a result dict (never raises)."""
    base, vec, unit, cache_dir, source_hash = job
    t0 = time.time()
    try:
        from kernels.common.hiptensor_aot.ew import EwSpec
        from kernels.common.hiptensor_aot.jit_ew import (
            EW_JIT_ABI,
            EwJitSpec,
            roll_jit_ew,
        )
        from rocke.portable_ir.src import abi as recipe_abi
        from rocke.portable_ir.src.recipe_bundle import cbor_decode, cbor_encode

        spec = EwJitSpec(EwSpec(**base), vec, tuple(unit))
        key = spec.key()
        axes = [name for name, _ in spec.axes()]
        cached = None
        if cache_dir:
            digest = hashlib.sha256((source_hash + "|" + key).encode()).hexdigest()
            cached = os.path.join(cache_dir, "jit-" + digest + ".cbor")
        blob = None
        if cached and os.path.isfile(cached):
            with open(cached, "rb") as f:
                blob = f.read()
        if blob is None:
            recipe = recipe_abi.stamp(roll_jit_ew(spec))
            blob = cbor_encode(recipe)
            if cached:
                tmp = cached + f".{os.getpid()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(blob)
                os.replace(tmp, cached)
        need = recipe_abi.recipe_min_reader(cbor_decode(blob))
        return dict(
            status="ok",
            key=key,
            family=EW_JIT_ABI,
            axes=axes,
            cbor=blob,
            min_reader=need,
            seconds=time.time() - t0,
        )
    except Exception as e:  # noqa: BLE001 - reported by the caller
        import traceback

        return dict(
            status="error",
            reason=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            key=str(base),
        )


def write_recipes(path, entries):
    """Writes rocke.recipes: header, table (key, family, axes, offset, size, SHA-256), CBOR
    recipes. Returns the file's SHA-256 and size."""

    def s(text):
        b = text.encode()
        return struct.pack("<I", len(b)) + b

    table = bytearray()
    data = bytearray()
    for e in entries:
        table += s(e["key"]) + s(e["family"]) + struct.pack("<I", len(e["axes"]))
        for a in e["axes"]:
            table += s(a)
        table += struct.pack("<QQ", len(data), len(e["cbor"]))
        table += hashlib.sha256(e["cbor"]).digest()
        data += e["cbor"]
    need = max([1] + [e["min_reader"] for e in entries])
    data_off = RECIPES_HEADER_SIZE + len(table)
    out = bytearray(RECIPES_MAGIC)
    out += struct.pack(
        "<IIII", RECIPES_FORMAT_VERSION, RECIPES_MIN_READER, len(entries), need
    )
    out += struct.pack("<QQ", RECIPES_HEADER_SIZE, data_off)
    assert len(out) == RECIPES_HEADER_SIZE
    out += table + data
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    return hashlib.sha256(out).hexdigest(), len(out)


def build_recipes(src, out_dir, jobs, cache_dir):
    """Rolls every JIT class in a process pool and writes <out_dir>/rocke.recipes. Returns
    (sha256, count); exits on a class that does not roll."""
    t0 = time.time()
    source_hash = _recipe_source_hash(src)
    work = [(b, v, u, cache_dir, source_hash) for b, v, u in jit_class_params()]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs, mp_context=ctx, initializer=_worker_init, initargs=(src,)
    ) as pool:
        results = list(pool.map(_roll_one, work))
    errors = [r for r in results if r["status"] != "ok"]
    for r in errors:
        print(f"rocKE pack ERROR recipe {r['key']}: {r['reason']}", file=sys.stderr)
    if errors:
        raise SystemExit(2)
    entries = sorted(results, key=lambda r: r["key"])
    sha, size = write_recipes(os.path.join(out_dir, "rocke.recipes"), entries)
    secs = sum(r["seconds"] for r in results)
    print(
        f"rocKE pack: {len(entries)} JIT recipes, {size} bytes, sha256 {sha[:16]}, "
        f"{secs:.0f} CPU-s, wall {time.time() - t0:.0f} s",
        flush=True,
    )
    return sha, len(entries)


# ------------------------------------------------------------------------------------------------
# Pack writer
# ------------------------------------------------------------------------------------------------


class _Strings:
    def __init__(self):
        self.buf = bytearray(b"\0")
        self.index = {"": 0}

    def add(self, s):
        s = str(s)
        if s not in self.index:
            self.index[s] = len(self.buf)
            self.buf += s.encode() + b"\0"
        return self.index[s]


def _spec_key(arch, attrs, name):
    return f"hiptensor-rocke/v{KERNEL_ABI}|{attrs['family']}|{arch}|{name}"


def write_pack(path_dir, arch, kernels, routes, compiler_id, rocke_version):
    """Writes rocke.htpk and rocke.index.json. ``kernels`` are ok results sorted by name."""
    os.makedirs(path_dir, exist_ok=True)
    st = _Strings()
    krec = bytearray()
    args = bytearray()
    attrs = bytearray()
    blob = bytearray()
    index_json = []
    n_args = n_attrs = 0
    for k in kernels:
        co = k["co"]
        sha = hashlib.sha256(co).digest()
        md = k["md"]
        arg_first, attr_first = n_args, n_attrs
        for a in md["args"]:
            args += struct.pack(
                "<IIII", st.add(a["name"]), a["offset"], a["size"], a["kind"]
            )
            n_args += 1
        for key in sorted(k["attrs"]):
            v = k["attrs"][key]
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, int):
                attrs += struct.pack("<IIq", st.add(key), ATTR_KIND_INT, v)
            else:
                attrs += struct.pack("<IIq", st.add(key), ATTR_KIND_STRING, st.add(v))
            n_attrs += 1
        # 16-byte aligned code objects
        while len(blob) % 16:
            blob.append(0)
        off = len(blob)
        blob += co
        spec_key = _spec_key(arch, k["attrs"], k["name"])
        krec += struct.pack(
            "<IIIIIIIIIIII",
            st.add(k["name"]),
            st.add(k["attrs"]["family"]),
            st.add(spec_key),
            arg_first,
            len(md["args"]),
            attr_first,
            len(k["attrs"]),
            md["kernarg_size"],
            md["group_segment"],
            md["private_segment"],
            md["max_flat_workgroup_size"],
            md["wavefront_size"],
        )
        krec += struct.pack("<QQ", off, len(co))
        krec += sha
        assert len(krec) % KERNEL_RECORD_SIZE == 0
        index_json.append(
            dict(
                name=k["name"],
                spec_key=spec_key,
                attrs=k["attrs"],
                args=md["args"],
                kernarg_size=md["kernarg_size"],
                group_segment=md["group_segment"],
                private_segment=md["private_segment"],
                max_flat_workgroup_size=md["max_flat_workgroup_size"],
                wavefront_size=md["wavefront_size"],
                registers=dict(sgpr=md["sgpr"], vgpr=md["vgpr"], agpr=md["agpr"]),
                code_size=len(co),
                code_sha256=sha.hex(),
            )
        )
    rrec = bytearray()
    for key in sorted(routes):
        rrec += struct.pack("<II", st.add(key), st.add(routes[key]))
    pack_id = hashlib.sha256(bytes(krec) + bytes(blob)).hexdigest()
    arch_off = st.add(arch)
    cid_off = st.add(compiler_id)
    rv_off = st.add(rocke_version)
    pid_off = st.add(pack_id)

    offset = HEADER_SIZE
    sections = []
    for data in (krec, args, attrs, rrec, st.buf):
        while offset % 16:
            offset += 1
        sections.append((offset, data))
        offset += len(data)
    while offset % 16:
        offset += 1
    blob_off = offset
    hdr = bytearray()
    hdr += PACK_MAGIC
    hdr += struct.pack(
        "<IIIIIIII",
        PACK_FORMAT_VERSION,
        PACK_MIN_READER,
        KERNEL_ABI,
        len(kernels),
        n_args,
        n_attrs,
        len(routes),
        0,
    )
    hdr += struct.pack(
        "<QQQQQQ",
        sections[0][0],
        sections[1][0],
        sections[2][0],
        sections[3][0],
        sections[4][0],
        len(st.buf),
    )
    hdr += struct.pack("<QQ", blob_off, len(blob))
    hdr += struct.pack("<IIII", arch_off, cid_off, rv_off, pid_off)
    assert len(hdr) <= HEADER_SIZE, len(hdr)
    hdr += b"\0" * (HEADER_SIZE - len(hdr))
    out = bytearray(hdr)
    for soff, data in sections:
        out += b"\0" * (soff - len(out))
        out += data
    out += b"\0" * (blob_off - len(out))
    out += blob
    tmp = os.path.join(path_dir, "rocke.htpk.tmp")
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, os.path.join(path_dir, "rocke.htpk"))
    with open(os.path.join(path_dir, "rocke.index.json"), "w", newline="\n") as f:
        json.dump(
            dict(
                format_version=PACK_FORMAT_VERSION,
                min_reader=PACK_MIN_READER,
                kernel_abi=KERNEL_ABI,
                arch=arch,
                compiler_id=compiler_id,
                rocke_version=rocke_version,
                pack_id=pack_id,
                routes=routes,
                kernels=index_json,
            ),
            f,
            indent=1,
            sort_keys=True,
        )
    return pack_id, len(out)


def _compiler_id():
    from rocke.runtime.comgr import loaded_compiler_info

    info = loaded_compiler_info()
    if info is None or info.llvm_version is None:
        raise SystemExit("rocKE could not load a COMGR library to compile the pack")
    v = ".".join(str(x) for x in info.llvm_version)
    return f"llvm-{v};comgr={os.path.basename(info.comgr_path or info.requested_comgr or '?')}"


def _rocke_version(src):
    try:
        import rocke

        v = getattr(rocke, "__version__", "")
    except ImportError:
        v = ""
    h = hashlib.sha256()
    for sub in ("library/kernels/common/hiptensor_aot",):
        d = os.path.join(src, sub)
        for root, _dirs, files in sorted(os.walk(d)):
            for fn in sorted(files):
                if fn.endswith(".py"):
                    with open(os.path.join(root, fn), "rb") as f:
                        h.update(fn.encode() + f.read().replace(b"\r\n", b"\n"))
    return f"rocke {v or 'source'}; hiptensor_aot {h.hexdigest()[:16]}"


def routes_for(arch, num_cus):
    features = rocke_features()
    r = {"num_cus": str(num_cus)}
    for family, missing in sorted(features.items()):
        if missing:
            r[f"missing.{family}"] = missing
    if arch in GEMM_ARCHES and not features["gemm"]:
        from rocke.dispatch.gemm import cf32 as rcf32

        for k in GEMM_ROUTE_KEYS:
            r[f"gemm.{k.lower()}"] = repr(float(getattr(rcf32, k)))
    if arch in GETT64_CLASS_ARCHES and not features["gett64"]:
        from rocke.instances.common import gett_f64 as g

        r["gett64.k_block_min_k"] = str(g.K_BLOCK_MIN_K)
    return r


def select_models(model_dir, arch):
    """Route entries of the selection models of ``arch``: ``{select.model.<cell>: text}``."""
    out = {}
    d = os.path.join(model_dir, arch) if model_dir else ""
    if not d or not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".model"):
            continue
        with open(os.path.join(d, fn), encoding="ascii") as f:
            text = f.read().strip()
        if not text.startswith("v1;schema=") or "\n" in text or "\0" in text:
            raise SystemExit(
                f"rocKE pack: {os.path.join(d, fn)} is not a selection model"
            )
        out["select.model." + fn[: -len(".model")]] = text
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rocke-src", required=True)
    ap.add_argument("--arch", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--compile-timeout", type=float, default=600.0)
    ap.add_argument(
        "--num-cus", type=int, default=0, help="CU count for persistent bodies"
    )
    ap.add_argument(
        "--only", default="", help="development: regex over descriptors to build"
    )
    ap.add_argument(
        "--select-models",
        default="",
        help="directory of kernel selection models (<dir>/<arch>/<cell>.model)",
    )
    ap.add_argument(
        "--jit-recipes",
        action="store_true",
        help="also write the JIT recipes (<out>/rocke.recipes)",
    )
    a = ap.parse_args(argv)

    src = os.path.abspath(a.rocke_src)
    _scrub_env()
    _setup_paths(src)
    compiler_id = _compiler_id()
    rocke_version = _rocke_version(src)
    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)
    # the CPUs this process may run on (a batch job's cgroup can be far below os.cpu_count())
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    jobs = a.jobs or max(1, (cpus or 2) - 1)
    print(f"rocKE pack: src={src} compiler={compiler_id} jobs={jobs}", flush=True)
    for family, missing in sorted(rocke_features().items()):
        if missing:
            print(
                f"rocKE pack WARNING: family {family} not built: {missing}", flush=True
            )

    jit_routes = {}
    if a.jit_recipes:
        missing = rocke_features()["jit"]
        if missing:
            # an older rocKE tree: an empty recipe file, so the JIT declines every launch
            print(f"rocKE pack WARNING: no JIT recipes: {missing}", flush=True)
            sha, _ = write_recipes(os.path.join(a.out, "rocke.recipes"), [])
            count = 0
        else:
            sha, count = build_recipes(src, a.out, jobs, a.cache_dir)
        jit_routes = {"jit.recipes.sha256": sha, "jit.recipes.count": str(count)}

    status = 0
    for arch in a.arch:
        from rocke.helpers.split_k import arch_num_cus

        num_cus = a.num_cus or arch_num_cus(arch)
        descs = kernel_descriptors(arch, num_cus)
        if a.only:
            import re

            descs = [
                d for d in descs if re.search(a.only, f"{d[0]} {sorted(d[1].items())}")
            ]
        work = [(f, p, arch, num_cus, a.cache_dir, compiler_id) for f, p in descs]
        t0 = time.time()
        results = []
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx, initializer=_worker_init, initargs=(src,)
        ) as pool:
            futs = [pool.submit(_compile_one, w) for w in work]
            deadline = time.time() + a.compile_timeout * max(1, -(-len(work) // jobs))
            for f in futs:
                try:
                    results.append(f.result(timeout=max(1.0, deadline - time.time())))
                except concurrent.futures.TimeoutError:
                    print("rocKE pack: compile timeout", file=sys.stderr)
                    for p in pool._processes.values():  # noqa: SLF001
                        p.kill()
                    return 3
        ok = [r for r in results if r["status"] == "ok"]
        skipped = [r for r in results if r["status"] == "skip"]
        errors = [r for r in results if r["status"] == "error"]
        for r in errors:
            print(
                f"rocKE pack ERROR {arch} {r['family']} {r['params']}: {r['reason']}",
                file=sys.stderr,
            )
        if errors:
            status = 2
            continue
        extra_routes = {}
        flat = []
        for r in ok:
            if "many" in r:
                flat.extend(r["many"])
                extra_routes.update(r["routes"])
            else:
                flat.append(r)
        names = {}
        for r in flat:
            if r["name"] in names:
                prev = names[r["name"]]
                if prev["co"] != r["co"]:
                    print(
                        f"rocKE pack ERROR {arch}: two kernels named {r['name']}",
                        file=sys.stderr,
                    )
                    status = 2
                continue
            names[r["name"]] = r
        kernels = sorted(names.values(), key=lambda r: r["name"])
        routes = routes_for(arch, num_cus)
        routes.update(extra_routes)
        routes.update(jit_routes)
        models = select_models(a.select_models, arch)
        routes.update(models)
        pack_id, size = write_pack(
            os.path.join(a.out, arch), arch, kernels, routes, compiler_id, rocke_version
        )
        secs = sum(r.get("seconds", 0.0) for r in ok)
        print(
            f"rocKE pack {arch}: {len(kernels)} kernels ({len(skipped)} descriptors do not "
            f"apply), {size} bytes, pack id {pack_id[:16]}, compile {secs:.0f} CPU-s, "
            f"wall {time.time() - t0:.0f} s, selection models: "
            f"{', '.join(sorted(k[len('select.model.'):] for k in models)) or 'none'}",
            flush=True,
        )
        for r in skipped:
            print(f"  not applicable: {r['family']} {r['params']}: {r['reason']}")
    return status


if __name__ == "__main__":
    sys.exit(main())
