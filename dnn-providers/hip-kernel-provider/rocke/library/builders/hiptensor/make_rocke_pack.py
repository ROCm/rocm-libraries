#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Builds the rocKE kernel pack of hipTensor's rocKE backend with rocKE's Python engine.

The Python reference of hipTensor's C++ pack builder (hipTensor's
library/src/backends/rocke/pack/rocke_pack_builder.cpp, which hipTensor's build runs; the build
needs no Python). Both write the same packs for the same rocKE sources: only the two header
strings that name the toolchain (compiler id, rocKE version) differ unless the C++ builder is
given this script's strings (--compiler-id, --rocke-version).
library/tests/test_hiptensor_pack_cross_check.py asserts that equality; hipTensor's build never
runs this script.

It imports rocKE from a source tree (by default the one this file belongs to), builds the
shape-generic kernels the backend launches, compiles them with the COMGR library rocKE loads,
reads every kernel's argument layout from its code object metadata and writes one pack per GPU
target:

    <out>/<arch>/rocke.htpk          binary pack: header, kernel table, argument and attribute
                                     tables, route table, string table, code objects
    <out>/<arch>/rocke.index.json    the same index as JSON (tooling only, never read at run time)

hipTensor's C++ runtime reads the binary pack (rocke_runtime.cpp, rocke_pack_format.cpp); it
never needs Python.

Kernel families (attribute "family"):

    ew       strided elementwise kernel (permute, elementwise binary / trinary, contraction
             operand packing and epilogue); attribute math=1: the rows that also implement the
             trigonometric and hyperbolic operators, their inverses, ceil and floor
    reduce   strided reduction kernels (row, col) and the split finalize kernel (math as for ew)
    rreduce  the tuned reduction classes (kernels.common.hiptensor_aot.rreduce: rocKE's native
             strided reduction bodies with the geometry as kernel arguments) of the target's
             palette and tuned problems (rocke/dispatch/families/hiptensor_rreduce.json)
    rreduce_final
             the split finalize of the tuned reduction classes
    gemm     universal GEMM bodies with an f32 output (routes h16: f16 / bf16 operands;
             x3 / x4 / x6 / x9: f32 operands split into bf16 parts, the recipes of
             rocke.instances.common.gemm_prologue; cf32 (x6) and cf32x3 / cf32x4 / cf32x9:
             complex f32 operands); the bodies of each target (GEMM_BODIES: gfx950 the
             DMA ring bodies, gfx942 the register-staged bodies of a set per route), each
             non-batched and batched, plus the 64-bit-address form of one body per route
    gett64   f64 / complex f64 strided batched GEMM with runtime sizes
    gett32   the same strided GEMM with f32 / complex f32 words (f32 products, f32
             accumulation): the native route of f32 contractions
    gett     the launches of rocKE's tuned and AOT GETT route tables (f16 / bf16 contractions
             of fixed shapes; the AOT table freezes the product route of each problem): pack
             kernels and the GETT kernel, with a launch record per problem
    rgett    rocKE's ring GETT body (f16 / bf16 contractions on gfx950) with its index maps as
             kernel arguments: the operands read and D written in place, one class per lane
             width / D vector width (kernels.common.hiptensor_aot.rgett)
    dgett    the direct GETT: any contraction (every data type, operator, layout) read and
             written in place with no workspace, register-staged MFMA tiles
             (kernels.common.hiptensor_aot.dgett)
    ewv      the vector classes of the elementwise kernel (kernels.common.hiptensor_aot.jit_ew with
             its geometry as kernel arguments): each thread handles a 16-byte vector of D's fastest
             mode, the inputs that are contiguous along it read with vector loads
    ewt      the tiled elementwise kernel of permutations and elementwise operations whose inputs
             are fastest along another mode than the output (an LDS tile per transposed input,
             16-byte vectors on both sides; kernels.common.hiptensor_aot.ewt)
    ewf      the flat elementwise kernel of operations whose tensors are all contiguous with the
             same layout (no index arithmetic, two 16-byte vectors per thread, streaming
             accesses; kernels.common.hiptensor_aot.ewf), for the all-16-bit rows on gfx942 and
             gfx950
    ewp      the tuned tiled elementwise classes of permutations and elementwise operations whose
             inputs are not all laid out like the output (16-byte accesses on both sides of an
             LDS tile, element transposition or whole shared-mode atoms;
             kernels.common.hiptensor_aot.ewp), the palette and tuned problems of
             kernels.common.hiptensor_aot.ewp_host, on gfx942 and gfx950; attribute rank: the
             class's place in the palette (the untuned choice's order)

The backend's JIT (HIPTENSOR_ENABLE_ROCKE_JIT) needs nothing from this script: hipTensor builds
the problem-specialized kernels (kernels.common.hiptensor_aot.jit_ew) at run time with their C++
twins, which it links.

The route table carries the tuning constants of rocKE's body rules, per problem of the tuned
and AOT GETT route tables ``gett.tuned.<hash>`` (the launch record) and ``gett.tuned.<hash>.key``
(the canonical problem, see :func:`gett_problem_key`), per tuned reduction problem
``reduce.tuned.<hash>`` (the knobs) and ``reduce.tuned.<hash>.key`` (the canonical problem,
``rreduce_host.problem_key``), and per tuned elementwise problem ``ewp.tuned.<hash>`` (the kernel
name of its class) and ``ewp.tuned.<hash>.key`` (``ewp_host.problem_key``). With --select-models, it also carries the kernel selection models
of the target, ``select.model.<cell>`` (``<dir>/<arch>/<cell>.model``, written by the selection
training flow; see rocke_select.hpp for the format).

With --gemm-models DIR (rocKE's ``dispatch/gemm/selector_models``), it writes next to the pack of
every target rocKE's tensor GETT planner serves (``TENSOR_GETT_ARCHS``) ``<out>/<arch>/gemm_models``:
the GEMM selection model files the planner's model loader reads for that target
(:func:`gemm_model_files`, the files of ``rocke::gemm_dispatch::Models::load``) and ``index.txt``
(``<sha256> <file>`` per file, by name), which hipTensor's contraction JIT loads and checks at run
time. They are data next to the pack, not part of it; hipTensor's C++ builder writes the same
directory (library/tests/test_hiptensor_pack_cross_check.py compares them). A selection model of
format v3 that names the digest of the planner's models it was trained with (``gemm=<sha256 of
index.txt>``) must name this index's digest.

With --list, it writes each target's manifest ``<out>/<arch>.manifest`` instead of its pack,
without lowering or compiling anything (:func:`pack_manifest`): the kernels with their attributes
and argument names, the routes and the kernel classes of hipTensor's JIT. The manifests of
hipTensor's targets are kept in ``manifests/`` next to this script (regenerate them with
``make_rocke_pack.py --list --arch gfx90a --arch gfx942 --arch gfx950 --out
library/builders/hiptensor/manifests`` when a kernel, attribute, route or JIT class changes);
library/tests/test_hiptensor_pack_cross_check.py checks that they are current, and hipTensor's
test rocke_pack_manifest_test checks that its C++ builder writes the same.

Usage:
    make_rocke_pack.py --arch gfx950 [--arch gfx942] --out DIR [--rocke-src DIR]
                        [--jobs N] [--cache-dir DIR] [--compile-timeout S] [--num-cus N]
                        [--select-models DIR] [--only REGEX] [--gemm-models DIR]
    make_rocke_pack.py --list --arch gfx950 [--arch gfx942] --out DIR [--num-cus N]
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import struct
import sys
import time

# Must match hiptensor::rockebackend::kKernelAbi and hipTensor's pack reader and writer
# (rocke_runtime.cpp, pack/rocke_pack_format.cpp).
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


# rocke/library/builders/hiptensor/make_rocke_pack.py -> rocke
_DEFAULT_ROCKE_SRC = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


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
# kernel (no JIT classes).
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

# The GEMM routes (pack order) and, per target, the bodies of each route
# (rocke.dispatch.gemm.cf32.cf32_body names).
GEMM_ROUTES = ("h16", "x6", "cf32", "x3", "x4", "x9", "cf32x3", "cf32x4", "cf32x9")
# gfx942: the register-staged bodies of rocke.dispatch.gemm.cf32.CF32_ROUTE_BODIES (x3 / x4 / x9
# on x6's, cf32x3 / cf32x4 / cf32x9 on cf32's)
_G942_H16 = (
    "G64x64x128L3",
    "G128x128x128L3",
    "G64x32x128U",
    "G192x128x64L3",
    "G256x256x64L3",
    "G32x32x128U",
    "G256x224x64L3",
)
_G942_X6 = (
    "G64x64x128L3",
    "G32x32x128L3",
    "G128x128x128L3",
    "G192x128x64L1S3",
    "G256x224x64L1",
    "G128x64x64",
    "G64x32x128U",
)
_G942_CF32 = (
    "G64x64x128L3",
    "G32x32x128L3",
    "G128x128x128L3",
    "G192x128x64US3",
    "G256x224x64L1",
    "G128x64x64U",
    "G64x32x128U",
)
GEMM_BODIES = {
    "gfx950": {
        # route: bodies
        "h16": ("RIW", "RIW_SK", "C3", "Q3_SK"),
        "x6": ("RIW", "RIW_SK", "C3"),
        "cf32": ("RIW", "RIW_SK", "Q3_SK"),
        # the other f32 recipes (HIPTENSOR_ROCKE_F32_RECIPE), on the bodies of x6 / cf32
        "x3": ("RIW", "RIW_SK", "C3"),
        "x4": ("RIW", "RIW_SK", "C3"),
        "x9": ("RIW", "RIW_SK", "C3"),
        "cf32x3": ("RIW", "RIW_SK", "Q3_SK"),
        "cf32x4": ("RIW", "RIW_SK", "Q3_SK"),
        "cf32x9": ("RIW", "RIW_SK", "Q3_SK"),
    },
    "gfx942": {
        r: _G942_H16 if r == "h16" else _G942_CF32 if r.startswith("cf32") else _G942_X6
        for r in GEMM_ROUTES
    },
}
GEMM_ARCHES = tuple(GEMM_BODIES)
# The body of the 64-bit-address GEMM kernels of each target, for the h16 route and the f32 routes
# (rocke.dispatch.gemm.cf32.CF32_WIDE_BODY: picked for operands past the 32-bit offsets of the
# bodies; gfx950: 64-bit tile origins need the non-persistent grid; gfx942: wide_addr on a
# register-staged body, 64-bit batch and tile origins).
GEMM_WIDE_BODY = {
    "gfx950": {"h16": "RIW", "f32": "RIW"},
    "gfx942": {"h16": "G256x224x64L3", "f32": "G256x224x64L1"},
}
GETT64_CLASS_ARCHES = ("gfx90a", "gfx942", "gfx950")
# The native f32 strided GEMM (gett32): the pack targets whose MMA catalog has the f32 atom, with
# the shape classes of the f64 GEMM's tuned configurations.
GETT32_CLASS_ARCHES = ("gfx942", "gfx950")
# The direct GETT (every target with MFMA) and the ring GETT classes (gfx950: ds_read_b64_tr_b16).
DGETT_ARCHES = ("gfx90a", "gfx942", "gfx950")
RGETT_ARCHES = ("gfx950",)
# The constants of rocKE's emulated-f32 body rule of each target (rocke.dispatch.gemm.cf32
# CF32_BODY_RULE), shipped in the route table so the C++ runtime picks the body the rocKE rule
# picks: "waves" (gfx950) the thresholds below; "cost" (gfx942) the entries of
# gemm_cost_routes().
GEMM_ROUTE_KEYS = (
    "CF32_SK_ALWAYS_WAVES",
    "CF32_SK_MAX_WAVES",
    "CF32_SK_MAX_TAIL",
    "CF32_SMALL_MAX_WAVES",
)


def _wide_body(arch, route):
    return GEMM_WIDE_BODY[arch]["h16" if route == "h16" else "f32"]


def gemm_split_mode(arch, route):
    """The operand split of an f32 route's prologue on ``arch`` (rocke.dispatch.gemm.cf32
    split_mode; "rne" on trees without it)."""
    from rocke.dispatch.gemm import cf32 as rcf32

    f = getattr(rcf32, "split_mode", None)
    return "rne" if f is None or route == "h16" else f(arch, route)


def gemm_cost_routes(arch):
    """The route entries of the "cost" body rule of ``arch`` (rocke.dispatch.gemm.cf32):
    ``gemm.rule`` = ``cost``; per route and body of the route ``gemm.cost.<route>.<body>.a`` /
    ``.b`` / ``.occ`` (gemm_body_coef, body names in lowercase); the x6 recipe's native f32 route
    (CF32_NATIVE_FIRST): ``gemm.x6_admits_gett32`` = 1 and its thresholds
    ``gemm.x6_gett32_first.<route>`` (real multiply-adds; routes x6 and cf32)."""
    from rocke.dispatch.gemm import cf32 as rcf32

    r = {"gemm.rule": "cost"}
    for route in GEMM_ROUTES:
        for b in GEMM_BODIES[arch].get(route, ()):
            coef = rcf32.gemm_body_coef(b, route)
            if coef is not None:
                for k, v in zip(("a", "b", "occ"), coef):
                    r[f"gemm.cost.{route}.{b.lower()}.{k}"] = repr(float(v))
    native = rcf32.CF32_NATIVE_FIRST.get(arch, {})
    if native:
        r["gemm.x6_admits_gett32"] = "1"
        for route, lim in sorted(native.items()):
            r[f"gemm.x6_gett32_first.{route}"] = repr(float(lim))
    return r


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
        from rocke.dispatch.families import tensor_gett as tg

        for name in ("TENSOR_GETT_AOT_FILE", "aot_entry_problem", "aot_route_plan"):
            getattr(tg, name)
        out["gett_aot"] = out["gett"]
    except (ImportError, AttributeError) as e:
        out["gett_aot"] = f"no GETT AOT route table ({e})"
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
        from rocke.instances.common import gett_f64 as g

        for name in (
            "GETT_DTYPES",
            "f32_atom",
            "runtime_template",
            "TUNED_CONFIGS_F32",
            "tuned_configs",
        ):
            getattr(g, name)
        out["gett32"] = ""
    except (ImportError, AttributeError) as e:
        out["gett32"] = f"no f32 GETT ({e})"
    try:
        from kernels.common.hiptensor_aot import dgett

        for name in ("DgettSpec", "build_dgett", "dgett_config", "DGETT_ABI"):
            getattr(dgett, name)
        out["dgett"] = ""
    except (ImportError, AttributeError) as e:
        out["dgett"] = f"no direct GETT ({e})"
    try:
        from kernels.common.hiptensor_aot import rgett

        for name in ("RgettSpec", "build_rgett", "rgett_classes", "RGETT_ABI"):
            getattr(rgett, name)
        out["rgett"] = ""
    except (ImportError, AttributeError) as e:
        out["rgett"] = f"no ring GETT classes ({e})"
    try:
        from kernels.common.hiptensor_aot import ewt

        for name in ("EwtSpec", "build_ewt", "EWT_ABI", "SLOTS_B"):
            getattr(ewt, name)
        out["ewt"] = ""
    except (ImportError, AttributeError) as e:
        out["ewt"] = f"no tiled elementwise kernel ({e})"
    try:
        from kernels.common.hiptensor_aot import jit_ew

        for name in ("EwJitSpec", "jit_classes", "EW_JIT_ABI", "EWV_ABI"):
            getattr(jit_ew, name)
        out["jit"] = ""
    except (ImportError, AttributeError) as e:
        out["jit"] = f"no vector classes of the elementwise kernel ({e})"
    try:
        from kernels.common.hiptensor_aot import ewf

        for name in ("EwfSpec", "build_ewf", "ewf_classes", "EWF_ABI"):
            getattr(ewf, name)
        out["ewf"] = ""
    except (ImportError, AttributeError) as e:
        out["ewf"] = f"no flat elementwise kernel ({e})"
    try:
        from kernels.common.hiptensor_aot import ewp, ewp_host

        for name in ("EwpSpec", "build_ewp", "EWP_ABI", "SLOTS_B"):
            getattr(ewp, name)
        for name in ("pack_classes", "routes"):
            getattr(ewp_host, name)
        out["ewp"] = ""
    except (ImportError, AttributeError) as e:
        out["ewp"] = f"no tuned tiled elementwise classes ({e})"
    try:
        from kernels.common.hiptensor_aot import rreduce, rreduce_host

        for name in (
            "RreduceSpec",
            "build_rreduce",
            "build_rreduce_final",
            "RREDUCE_ABI",
        ):
            getattr(rreduce, name)
        for name in ("pack_classes", "final_classes", "routes"):
            getattr(rreduce_host, name)
        rreduce_host.load_table()
        out["rreduce"] = ""
    except (ImportError, AttributeError, OSError, ValueError) as e:
        out["rreduce"] = f"no tuned reduction classes ({e})"
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
    if not features["rreduce"]:
        # the tuned reduction classes of the target (its palette, then the classes of its tuned
        # problems) and the finalize classes
        from dataclasses import asdict

        from kernels.common.hiptensor_aot import rreduce_host

        out.extend(("rreduce", asdict(s)) for s in rreduce_host.pack_classes(arch))
        out.extend(
            ("rreduce_final", asdict(f)) for f in rreduce_host.final_classes(arch)
        )
    if not features["ewt"]:
        # the real rows of the API's elementwise operations (32-bit offsets, no math operators)
        for p in ew_params(False, mixed=False):
            if p.get("cplx") or p.get("math"):
                continue
            # and their folded classes (several small planes per tile)
            for fold in (False, True):
                q = {k: v for k, v in p.items() if k in ("arity", "a", "b", "c", "d")}
                if fold:
                    q["fold"] = True
                out.append(("ewt", q))
    if not features["jit"]:
        # the vector classes of the same rows (every combination of vector / gathered inputs)
        from kernels.common.hiptensor_aot.ew import EwSpec
        from kernels.common.hiptensor_aot.jit_ew import jit_classes

        for p in ew_params(False, mixed=False):
            if p.get("cplx") or p.get("math"):
                continue
            for c in jit_classes([EwSpec(**p)]):
                if c.vec > 1:
                    out.append(
                        ("ewv", dict(base=dict(p), vec=c.vec, unit=tuple(c.unit)))
                    )
    if not features["ewf"]:
        # the flat classes of the same rows (streaming targets)
        from dataclasses import asdict

        from kernels.common.hiptensor_aot.ewf import ewf_classes

        out.extend(
            ("ewf", asdict(c)) for c in ewf_classes(ew_params(False, mixed=False), arch)
        )
    if not features["ewp"]:
        # the tuned tiled elementwise classes of the target (palette, then tuned problems)
        from dataclasses import asdict

        from kernels.common.hiptensor_aot import ewp_host

        out.extend(("ewp", asdict(s)) for s in ewp_host.pack_classes(arch))
    if arch in GEMM_ARCHES and not features["gemm"]:
        for route in GEMM_ROUTES:
            bodies = GEMM_BODIES[arch].get(route, ())
            if not bodies:
                continue
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
                            body=_wide_body(arch, route),
                            batched=False,
                            wide=True,
                        ),
                    )
                )
    for key in tuned_gett_keys(arch) if not features["gett"] else []:
        out.append(("gett_plan", dict(key=key)))
    for key in aot_gett_keys(arch) if not features["gett_aot"] else []:
        out.append(("gett_plan", dict(key=key, table="aot")))
    if arch in DGETT_ARCHES and not features["dgett"]:
        for cplx in (False, True):
            for comp in ("f32", "f64") if cplx else ("f16", "bf16", "f32", "f64"):
                for a_fast in ("k", "r"):
                    for b_fast in ("k", "r"):
                        for math in (False,) if cplx else (False, True):
                            for idx64 in (False, True):
                                out.append(
                                    (
                                        "dgett",
                                        dict(
                                            comp=comp,
                                            cplx=cplx,
                                            a_fast=a_fast,
                                            b_fast=b_fast,
                                            math=math,
                                            idx64=idx64,
                                        ),
                                    )
                                )
    if arch in RGETT_ARCHES and not features["rgett"]:
        from dataclasses import asdict

        from kernels.common.hiptensor_aot.rgett import rgett_classes

        out.extend(("rgett", asdict(c)) for c in rgett_classes())
    for family, arches in (
        ("gett64", GETT64_CLASS_ARCHES),
        ("gett32", GETT32_CLASS_ARCHES),
    ):
        if arch not in arches or features[family]:
            continue
        from rocke.instances.common.gett_f64 import tuned_configs

        for cplx in (False, True):
            kind = "complex" if cplx else "real"
            dtype = "f32" if family == "gett32" else "f64"
            classes = sorted(tuned_configs(arch, kind, dtype))
            for cls in classes + ["default"]:
                for dclass in ("m", "n", "x"):
                    for bilinear in (False, True):
                        for kblock in (False, True):
                            out.append(
                                (
                                    family,
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


def _rcf32():
    from rocke.dispatch.gemm import cf32

    return cf32


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
        # real routes name their recipe; complex routes are cf32 (x6) or cf32<recipe>
        cplx = route.startswith("cf32")
        recipe = (route[len("cf32") :] or "x6") if cplx else route
        pro = dict(
            src="f32",
            recipe=recipe,
            out_f32=True,
            split_at="use",
            cplx_b=cplx,
            psub=True,
        )
        # the target's operand split of the route (a Prologue field of newer trees)
        sm = gemm_split_mode(arch, route)
        if sm != "rne":
            pro["split_mode"] = sm
        pro = Prologue(**pro)
    attach_prologue(spec, pro)
    _ = replace
    return spec, "ok"


def _gett64_spec(params, arch, num_cus, dtype="f64"):
    """The spec of a gett64 (``dtype`` "f64") or gett32 (``dtype`` "f32") descriptor."""
    from dataclasses import replace

    from rocke.instances.common.gett_f64 import (
        _with_k_block,
        arch_spec,
        gemm_problem,
        is_valid_spec,
        runtime_template,
        tuned_configs,
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
        dtype=dtype,
    )
    kind = "complex" if cplx else "real"
    cfg = (
        {}
        if params["cls"] == "default"
        else dict(tuned_configs(arch, kind, dtype)[params["cls"]])
    )
    cfg.pop("split_k", None)
    if dtype == "f32":
        cfg["name"] = "rocke_gett"
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


def _table_keys(fname, arch):
    from pathlib import Path

    from rocke.dispatch.families import tensor_gett as tg

    path = Path(tg.__file__).with_name(fname)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return sorted(raw.get("entries", {}).get(arch, {}))


def tuned_gett_keys(arch):
    """Keys of rocKE's shipped tuned GETT route table for ``arch``."""
    from rocke.dispatch.families import tensor_gett as tg

    return _table_keys(tg.TENSOR_GETT_TUNED_FILE, arch)


def aot_gett_keys(arch):
    """Keys of rocKE's GETT AOT route table for ``arch`` (the product routes of fixed problems)."""
    from rocke.dispatch.families import tensor_gett as tg

    return _table_keys(tg.TENSOR_GETT_AOT_FILE, arch)


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
    """The kernels and the launch record of one problem of the tuned GETT route table, or of the
    AOT route table with ``params["table"] == "aot"``.

    Record: ``v1;ws=<workspace bytes>;steps=<n>`` then per launch
    ``|k=<kernel>;g=<x>,<y>,<z>;b=<threads>;a=<name>=<value>,...`` where a value is
    ``<buffer>+<byte offset>`` (buffers A, B, C, O (the output), W (the workspace)), ``#<int>``
    or ``@a`` / ``@b`` (the host scalars alpha / beta, f32)."""
    from rocke.dispatch.families import tensor_gett as tg
    from rocke.instances.common.tensor import GettKnobs, TensorGettSpec

    key = params["key"]
    aot = params.get("table") == "aot"
    prob = (tg.aot_entry_problem if aot else tg.tuned_entry_problem)(arch, key)
    if prob is None:
        return None, "no route table entry"
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
        return None, "the route table entry's key does not reproduce"
    # the AOT entry's frozen route (equal to plan_tensor_gett, which rocKE's tests check)
    gp = tg.aot_route_plan(req) if aot else tg.plan_tensor_gett(req)
    if gp is None:
        raise ValueError(f"the AOT route table entry {key} gives no plan")
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
    # The canonical key with the packed row-major strides the entry's problem has
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
    if family == "rreduce":
        from kernels.common.hiptensor_aot.rreduce import (
            RREDUCE_ABI,
            RreduceSpec,
            build_rreduce,
        )

        spec = RreduceSpec(**params)
        attrs = dict(
            family="rreduce",
            abi=RREDUCE_ABI,
            mode=spec.mode,
            a=spec.a,
            d=spec.d,
            op=spec.op,
            nr=spec.nr,
            block=spec.block,
            vec=spec.vec,
            lanes=spec.lanes,
            unroll=spec.unroll,
            rows=spec.rows,
            nt=int(spec.nt),
            una=int(spec.una),
            idx64=int(spec.idx64),
            max_rank=8,
        )
        return build_rreduce(spec), attrs
    if family == "rreduce_final":
        from kernels.common.hiptensor_aot.rreduce import (
            RREDUCE_ABI,
            RreduceFinalSpec,
            build_rreduce_final,
        )

        spec = RreduceFinalSpec(**params)
        attrs = dict(
            family="rreduce_final",
            abi=RREDUCE_ABI,
            a=spec.a,
            d=spec.d,
            op=spec.op,
            lanes=spec.lanes,
            idx64=int(spec.idx64),
            block=spec.block,
            max_rank=8,
        )
        return build_rreduce_final(spec), attrs
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
        try:
            from rocke.dispatch.gemm.cf32 import gemm_launch_limits

            limits = gemm_launch_limits(spec)
        except ImportError:  # an older rocKE tree: the ring bodies' limits
            limits = dict(
                limit_operand_bytes=DTL_BUFFER_NUM_RECORDS,
                limit_c_elements=(1 << 31) - 1,
                limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
                limit_c_tile_elements=1 << 31,
                limit_grid_yz=65535,
            )
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
            # capability (rocke.dispatch.gemm.cf32.gemm_launch_limits): launched on extents
            # padded to tile multiples; the operands (all batches) within the 32-bit offsets of
            # the body (the DMA buffer range of the ring bodies), C within the i32 element
            # offsets, rows of tiles on grid y. The 64-bit-address body: only one tile's operand
            # rows within the DMA buffer range and one tile's C rows within the i32 element
            # offsets.
            **limits,
        )
        if arch in getattr(_rcf32(), "CF32_SPLIT_MODE", {}):
            # the target's operand split of the route ("none": h16, no split)
            route = params["route"]
            attrs["split_mode"] = (
                "none" if route == "h16" else gemm_split_mode(arch, route)
            )
        return build_universal_gemm(spec, arch), attrs
    if family == "ewv":
        from kernels.common.hiptensor_aot.ew import EwSpec
        from kernels.common.hiptensor_aot.jit_ew import EWV_ABI, EwJitSpec, build_ew_jit

        base = EwSpec(**params["base"])
        spec = EwJitSpec(base, params["vec"], tuple(params["unit"]))
        attrs = dict(
            family="ewv",
            abi=EWV_ABI,
            arity=base.arity,
            cplx=0,
            idx64=0,
            math=0,
            a=base.a,
            b=base.b or "",
            c=base.c or "",
            d=base.d,
            block=base.block,
            vec=spec.vec,
            unit="".join("u" if x else "g" for x in spec.unit),
            max_rank=8,
            limit_index=(1 << 31) - 1,
        )
        return build_ew_jit(spec, None), attrs
    if family == "ewf":
        from kernels.common.hiptensor_aot.ewf import (
            EWF_ABI,
            EWF_UNROLL,
            EwfSpec,
            build_ewf,
        )

        spec = EwfSpec(**params)
        attrs = dict(
            family="ewf",
            abi=EWF_ABI,
            arity=spec.arity,
            a=spec.a,
            b=spec.b or "",
            c=spec.c or "",
            d=spec.d,
            block=spec.block,
            vec=spec.vec,
            unroll=EWF_UNROLL,
            nt=int(spec.nt),
            limit_index=(1 << 31) - 1,
        )
        return build_ewf(spec), attrs
    if family == "ewp":
        from kernels.common.hiptensor_aot import ewp as ep
        from kernels.common.hiptensor_aot import ewp_host

        spec = ep.EwpSpec(**params)
        attrs = dict(
            family="ewp",
            abi=ep.EWP_ABI,
            arity=spec.arity,
            a=spec.a,
            b=spec.b or "",
            c=spec.c or "",
            d=spec.d,
            kinds=spec.kinds,
            mode=spec.mode,
            to=spec.to,
            tt=spec.tt,
            tw=spec.tw,
            block=spec.block,
            nt=spec.nt,
            swz=spec.swz,
            swsh=spec.swsh,
            walk=spec.walk,
            tpc=spec.tpc,
            ilp=spec.ilp,
            vec=spec.vec,
            vb=spec.vb,
            rag=spec.rag,
            vt=spec.vect,
            rank=ewp_host.pack_classes(arch).index(spec),
            # capability: batch digit slots, i32 element offsets
            slots_b=ep.SLOTS_B,
            limit_index=(1 << 31) - 1,
        )
        return ep.build_ewp(spec), attrs
    if family == "ewt":
        from kernels.common.hiptensor_aot import ewt as et

        spec = et.EwtSpec(**params)
        attrs = dict(
            family="ewt",
            abi=et.EWT_ABI,
            arity=spec.arity,
            a=spec.a,
            b=spec.b or "",
            c=spec.c or "",
            d=spec.d,
            block=et.THREADS,
            tile_o=et.TILE_O,
            tile_t=spec.tile_t(),
            fold=int(spec.fold),
            # capability: batch digit slots, i32 element offsets; folded: planes per tile (the
            # LDS table of batch offsets)
            slots_b=et.SLOTS_B,
            max_planes=et.THREADS,
            limit_index=(1 << 31) - 1,
        )
        return et.build_ewt(spec), attrs
    if family == "dgett":
        from kernels.common.hiptensor_aot import dgett as dg

        spec = dg.DgettSpec(**params)
        try:
            spec.check(arch)
        except ValueError as e:
            return None, str(e)
        cfg = dg.dgett_config(spec)
        attrs = dict(
            family="dgett",
            abi=dg.DGETT_ABI,
            comp=spec.comp,
            cplx=int(spec.cplx),
            a_fast=spec.a_fast,
            b_fast=spec.b_fast,
            math=int(spec.math),
            idx64=int(spec.idx64),
            tile_m=cfg.tm,
            tile_n=cfg.tn,
            tile_k=cfg.tk,
            block=dg.THREADS,
            # capability: digit slots of the M / N spaces, the K tiles, the in-tile K digits and
            # the batch space; i32 coordinates; 32-bit offsets unless idx64
            slots_mn=dg.SLOTS_MN,
            slots_kt=dg.SLOTS_KT,
            slots_kl=dg.SLOTS_KL,
            slots_l=dg.SLOTS_L,
            limit_index=(1 << 31) - 1,
            limit_grid_yz=65535,
        )
        return dg.build_dgett(spec, arch=arch), attrs
    if family == "rgett":
        from kernels.common.hiptensor_aot import rgett as rg

        spec = rg.RgettSpec(**params)
        attrs = dict(
            family="rgett",
            abi=rg.RGETT_ABI,
            dtype=spec.dtype,
            bilinear=int(spec.bilinear),
            vec_a=spec.vec_a,
            vec_b=spec.vec_b,
            vec_d=spec.vec_d,
            pair=int(spec.epi_pair),
            wide=int(spec.wide_d),
            # digits per digit space the class decomposes (0: every slot), C prefetch distance
            slots=spec.slots,
            cp=spec.c_prefetch,
            tile_m=256,
            tile_n=256,
            tile_k=spec.tile_k,
            block=256,
            slots_row=rg.SLOTS_ROW,
            slots_kt=rg.SLOTS_KT,
            slots_kl=rg.SLOTS_KL,
            # capability: operands within the DMA buffer range, D and C within the epilogue's
            # 32-bit offsets (the wide classes: masked summed offsets), rows of tiles on grid y
            limit_operand_bytes=0x7FFF0000,
            limit_d_bytes=rg.MAX_D_BYTES if spec.wide_d else 0x7FFFFFF0,
            limit_grid_yz=65535,
        )
        return rg.build_rgett(spec, arch=arch), attrs
    if family in ("gett64", "gett32"):
        from rocke.instances.common import gett_f64 as g64
        from rocke.instances.common.gett_f64 import build_gett_f64

        f32 = family == "gett32"
        spec, why = _gett64_spec(params, arch, num_cus, "f32" if f32 else "f64")
        if spec is None:
            return None, why
        attrs = dict(
            family=family,
            abi="rocke-gett-f32/v1" if f32 else "rocke-gett-f64/v2",
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

        if getattr(rcf32, "CF32_BODY_RULE", {}).get(arch, "waves") == "cost":
            r.update(gemm_cost_routes(arch))
        else:
            for k in GEMM_ROUTE_KEYS:
                r[f"gemm.{k.lower()}"] = repr(float(getattr(rcf32, k)))
    if arch in GETT64_CLASS_ARCHES and not features["gett64"]:
        from rocke.instances.common import gett_f64 as g

        r["gett64.k_block_min_k"] = str(g.K_BLOCK_MIN_K)
    if not features["rreduce"]:
        # the tuned reduction problems of the target: reduce.tuned.<hash> (knobs) and .key
        from kernels.common.hiptensor_aot import rreduce_host

        r.update(rreduce_host.routes(arch))
    if not features["ewp"]:
        # the tuned elementwise problems of the target: ewp.tuned.<hash> (class) and .key
        from kernels.common.hiptensor_aot import ewp_host

        r.update(ewp_host.routes(arch))
    return r


def hiptensor_jit_classes():
    """The kernel names of the classes hipTensor's JIT specializes (its rule:
    ``jitClassAdmitted`` of hipTensor's rocke_ew_rows.hpp): every class of :func:`jit_classes`
    of every row of the pack's generic elementwise kernel (32- and 64-bit offsets, the
    mixed-type rows of contractions, the math rows), so every launch of that kernel has one.
    """
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    names = set()
    for idx64 in (False, True):
        for p in ew_params(idx64):
            names.update(c.kernel_name() for c in jit_classes([EwSpec(**p)]))
    return sorted(names)


def _manifest_field(s):
    return str(s).replace("\\", "\\\\").replace("\n", "\\n")


def pack_manifest(arch, num_cus):
    """The manifest of the pack of ``arch`` (``--list``; hipTensor's C++ builder writes the same
    text): a comment line, then one line per kernel in pack order (``kernel <name>
    <attribute>=<value>... params=<count>:<digest>``, strings quoted; the digest: the first 16
    hex digits of the SHA-256 of the comma-joined argument names), one per route (``route
    <key>=<value>``, sorted, no selection models) and one per kernel class of hipTensor's JIT
    (``jit <kernel name>``, sorted). Nothing is lowered or compiled."""
    kernels = {}
    routes = routes_for(arch, num_cus)
    for family, params in kernel_descriptors(arch, num_cus):
        if family == "gett_plan":
            plan, _why = build_gett_plan(params, arch)
            if plan is None:
                continue
            built = [
                (k, dict(family="gett", abi="rocke-tensor-gett/v1"))
                for k in plan["kernels"]
            ]
            route = fnv1a64(plan["key"])
            routes[f"gett.tuned.{route}"] = plan["record"]
            routes[f"gett.tuned.{route}.key"] = plan["key"]
        else:
            kernel, attrs = build_kernel(family, params, arch, num_cus)
            if kernel is None:
                continue
            built = [(kernel, attrs)]
        for kernel, attrs in built:
            fields = [f"kernel {kernel.name}"]
            for k, v in sorted(attrs.items()):
                if isinstance(v, (bool, int)):
                    fields.append(f"{k}={int(v)}")
                else:
                    fields.append(f'{k}="{_manifest_field(v)}"')
            names = ",".join(p.name for p in kernel.params)
            digest = hashlib.sha256(names.encode()).hexdigest()[:16]
            fields.append(f"params={len(kernel.params)}:{digest}")
            # a kernel several descriptors build keeps the first one's record, as in the pack
            kernels.setdefault(kernel.name, " ".join(fields))
    out = [f"# rocKE kernel pack manifest {arch}"]
    out += [kernels[n] for n in sorted(kernels)]
    out += [
        f"route {_manifest_field(k)}={_manifest_field(v)}"
        for k, v in sorted(routes.items())
    ]
    out += [f"jit {n}" for n in hiptensor_jit_classes()]
    return "\n".join(out) + "\n"


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
        if (
            not text.startswith(("v1;schema=", "v2;schema=", "v3;schema="))
            or "\n" in text
            or "\0" in text
        ):
            raise SystemExit(
                f"rocKE pack: {os.path.join(d, fn)} is not a selection model"
            )
        out["select.model." + fn[: -len(".model")]] = text
    return out


def gemm_model_files(model_dir, arch):
    """``{file name: bytes}``: the files rocKE's tensor GETT planner's GEMM model loader reads
    for ``arch`` from ``model_dir`` (the C++ loader ``rocke::gemm_dispatch::Models::load`` and
    the Python selector alike: per selector kind (rcr, native) ``gemm_<kind>_<arch>.model.txt``,
    then its ``.meta.json`` sidecar, and for a sidecar of the kind's feature version and this
    target its guard bags; the skinny sidecar ``skinny_cfg_<arch>.meta.json`` and, for its
    feature version and target, the model and bags it names). A file the loader needs and does
    not find is an error."""
    import json

    files = {}

    def read(name):
        path = os.path.join(model_dir, name)
        if not name or "/" in name or "\\" in name or not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            files[name] = f.read()
        return files[name]

    for kind, stem, versions in (
        ("rcr", "gemm_rcr_", ("g4-v1", "r2c-v2")),
        ("native", "gemm_native_", ("r2n-v1",)),
    ):
        if read(stem + arch + ".model.txt") is None:
            continue
        meta = read(stem + arch + ".meta.json")
        if meta is None:
            continue
        m = json.loads(meta.decode("utf-8"))
        if m.get("feature_version") not in versions or m.get("arch") != arch:
            continue
        guard = m.get("guard")
        bags = guard.get("bags") if isinstance(guard, dict) else None
        for b in bags if isinstance(bags, list) else []:
            if isinstance(b, str) and read(b) is None:
                raise SystemExit(
                    f"gemm models of {arch}: {b}: guard bag named by the sidecar is missing"
                )
    meta = read("skinny_cfg_" + arch + ".meta.json")
    if meta is not None:
        m = json.loads(meta.decode("utf-8"))
        if m.get("feature_version") == "r3sk-v1" and m.get("arch") == arch:
            bags = m.get("bags")
            names = [m.get("model") if isinstance(m.get("model"), str) else ""]
            names += [
                b
                for b in (bags if isinstance(bags, list) else [])
                if isinstance(b, str)
            ]
            for n in names:
                if read(n) is None:
                    raise SystemExit(
                        f"gemm models of {arch}: {n}: model file named by the sidecar is missing"
                    )
    return files


def write_gemm_models(model_dir, pack_dir, arch):
    """``<pack_dir>/gemm_models`` for a target the tensor GETT planner serves (removed for the
    others); the SHA-256 (hex) of its index, or "" when none is written."""
    from rocke.instances.common.tensor.gett import TENSOR_GETT_ARCHS

    dst = os.path.join(pack_dir, "gemm_models")
    shutil.rmtree(dst, ignore_errors=True)
    if arch not in TENSOR_GETT_ARCHS:
        return ""
    files = gemm_model_files(model_dir, arch)
    if not files:
        raise SystemExit(f"gemm models of {arch}: no model file in {model_dir}")
    os.makedirs(dst)
    index = ""
    for name in sorted(files):
        with open(os.path.join(dst, name), "wb") as f:
            f.write(files[name])
        index += hashlib.sha256(files[name]).hexdigest() + " " + name + "\n"
    with open(os.path.join(dst, "index.txt"), "w", newline="\n") as f:
        f.write(index)
    return hashlib.sha256(index.encode("ascii")).hexdigest()


def select_model_gemm_digest(text):
    """The ``gemm=`` digest a v3 selection model names ("" for "-" and other formats)."""
    if not text.startswith("v3;"):
        return ""
    for field in text.split("|", 1)[0].split(";"):
        if field.startswith("gemm="):
            return "" if field == "gemm=-" else field[len("gemm=") :]
    return ""


def gemm_digest_errors(models, digest):
    """The selection models (``{select.model.<cell>: text}``) that name another digest of the
    planner's GEMM selection models than ``digest`` (the index this pack ships): the build refuses
    them (their planner features would not be the ones the library computes)."""
    out = []
    for key, text in sorted(models.items()):
        want = select_model_gemm_digest(text)
        if want and digest and want != digest:
            out.append(
                f"{key} was trained with the GEMM selection models {want}, the planner's are "
                f"{digest}"
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--rocke-src",
        default=_DEFAULT_ROCKE_SRC,
        help="rocKE source tree (default: the tree of this script)",
    )
    ap.add_argument("--arch", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--list",
        action="store_true",
        help="write the manifest <out>/<arch>.manifest of each pack instead of the pack "
        "(nothing is compiled)",
    )
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
        "--gemm-models",
        default="",
        help="rocKE's GEMM selection models (dispatch/gemm/selector_models): written to "
        "<out>/<arch>/gemm_models for the targets rocKE's tensor GETT planner serves",
    )
    a = ap.parse_args(argv)
    if not a.arch:
        ap.error("--arch is required")

    src = os.path.abspath(a.rocke_src)
    _scrub_env()
    _setup_paths(src)
    if a.list:
        from rocke.helpers.split_k import arch_num_cus

        os.makedirs(a.out, exist_ok=True)
        for arch in a.arch:
            text = pack_manifest(arch, a.num_cus or arch_num_cus(arch))
            with open(os.path.join(a.out, f"{arch}.manifest"), "w", newline="\n") as f:
                f.write(text)
            print(f"rocKE pack {arch}: manifest {os.path.join(a.out, arch)}.manifest")
        return 0
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
        models = select_models(a.select_models, arch)
        routes.update(models)
        pack_id, size = write_pack(
            os.path.join(a.out, arch), arch, kernels, routes, compiler_id, rocke_version
        )
        gemm_files = ""
        if a.gemm_models:
            digest = write_gemm_models(a.gemm_models, os.path.join(a.out, arch), arch)
            gemm_files = digest[:16] or "none"
            for err in gemm_digest_errors(models, digest):
                print(f"rocKE pack ERROR {arch}: {err}", file=sys.stderr)
                status = 2
        secs = sum(r.get("seconds", 0.0) for r in ok)
        print(
            f"rocKE pack {arch}: {len(kernels)} kernels ({len(skipped)} descriptors do not "
            f"apply), {size} bytes, pack id {pack_id[:16]}, compile {secs:.0f} CPU-s, "
            f"wall {time.time() - t0:.0f} s, selection models: "
            f"{', '.join(sorted(k[len('select.model.'):] for k in models)) or 'none'}"
            + (f", GETT planner models: {gemm_files}" if a.gemm_models else ""),
            flush=True,
        )
        for r in skipped:
            print(f"  not applicable: {r['family']} {r['params']}: {r['reason']}")
    return status


if __name__ == "__main__":
    sys.exit(main())
