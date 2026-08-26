#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared Tile-Engine perf-flag construction for the block-scale quant bridges.

Single source of truth for the ``-mllvm`` flag set the bridge ctypes ``.so`` is
compiled with, so the flags cannot drift across the per-op utils files.  Before
this module, ``_te_perf_flags`` / ``_coerce_flag_ok`` were copy-pasted verbatim
into ``gemm_{aquant,rowcolquant,tensor_quant}_utils.py`` (5-flag set), and
``gemm_abquant_utils.py`` carried its own near-identical copy with two extra
flags -- a live example of the duplication drifting.

The base set is the 5 authoritative develop Tile-Engine flags
(``composablekernel/CMakeLists.txt`` L521/L528/L535/L546/L547).  abquant's gfx950
EightWaves fast path needs two additional flags; they are passed via ``extra=``
so the single base definition stays canonical.
"""

import functools
import hashlib
import os
import subprocess

# Each Tile-Engine perf flag, spelled exactly once.  Bridges differ in the ORDER
# they emit these (bquant's TU mirrors Old-TE's gemm_quant literally), and flag
# order can change codegen, so the order is a per-op choice made from this one
# table rather than a per-op copy of the flag strings.
_FLAG = {
    "fno_offload_uniform_block": ("-fno-offload-uniform-block",),
    "lsr_drop_solution": ("-mllvm", "--lsr-drop-solution=1"),
    "enable_post_misched": ("-mllvm", "-enable-post-misched=0"),
    "early_inline_all": ("-mllvm", "-amdgpu-early-inline-all=true"),
    "function_calls": ("-mllvm", "-amdgpu-function-calls=false"),
    "coerce_illegal_types": ("-mllvm", "-amdgpu-coerce-illegal-types=1"),
}

# The authoritative develop Tile-Engine perf flag set (5 flags), in the order
# composablekernel/CMakeLists.txt injects them.
TE_ORDER_DEFAULT = (
    "fno_offload_uniform_block",
    "lsr_drop_solution",
    "enable_post_misched",
    "early_inline_all",
    "function_calls",
)

# gemm_bquant emits the same five in the order Old-TE's gemm_quant TU uses, and
# puts the probe-gated coerce flag FIRST rather than last.
TE_ORDER_BQUANT = (
    "early_inline_all",
    "function_calls",
    "lsr_drop_solution",
    "enable_post_misched",
    "fno_offload_uniform_block",
)

_TE_BASE_FLAGS = [f for key in TE_ORDER_DEFAULT for f in _FLAG[key]]


@functools.lru_cache(maxsize=None)
def coerce_flag_supported(hipcc):
    """True iff the local clang accepts ``-mllvm -amdgpu-coerce-illegal-types=1``.

    ROCm 7.2 clang>=22 removed it and aborts the compile, so gate on it.  The
    kernels are bit-accurate without it (it only tightens register allocation on
    older toolchains).
    """
    try:
        r = subprocess.run(
            [hipcc, "-x", "hip", "-c", "-mllvm",
             "-amdgpu-coerce-illegal-types=1", "-", "-o", "/dev/null"],
            input="int main(){return 0;}", text=True,
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def te_perf_flags(hipcc, extra=None, order=TE_ORDER_DEFAULT, coerce_first=False):
    """The Tile-Engine ``-mllvm`` perf flags for a bridge ctypes ``.so`` compile.

    Without these, ``hipcc -O3`` register allocation on the block-scale hot loops
    spills to scratch and collapses occupancy, so the bridge kernel runs slower
    than the byte-identical Old-TE kernel.  Kept in lockstep with the develop TE
    build for fair parity.  Disabled entirely when ``CK_BRIDGE_NO_TE_FLAGS=1``.

    ``extra`` appends op-specific flags (e.g. abquant EightWaves:
    ``-enable-noalias-to-md-conversion``, ``-greedy-reverse-local-assignment``;
    bquant: ``--offload-compress``) after the base set.

    ``order`` and ``coerce_first`` reproduce each bridge's exact emitted flag
    sequence -- flag order can change codegen, so the sequences are preserved
    rather than normalized; only the flag *strings* are shared.
    """
    if os.environ.get("CK_BRIDGE_NO_TE_FLAGS") == "1":
        return []
    flags = [f for key in order for f in _FLAG[key]]
    if extra:
        flags += list(extra)
    if coerce_flag_supported(hipcc):
        coerce = list(_FLAG["coerce_illegal_types"])
        flags = coerce + flags if coerce_first else flags + coerce
    return flags


def flags_cache_tag(hipcc, extra=None):
    """Short digest of the flag set a ``.so`` would actually be compiled with.

    The compiled ``.so`` is cached on disk by kernel name and arch.  The flag set
    is not part of that key, so flipping ``CK_BRIDGE_NO_TE_FLAGS`` -- or moving to
    a toolchain where the coerce probe answers differently -- silently reused a
    ``.so`` built with the *other* flags, and the resulting parity number
    described a build nobody asked for.  Fold this tag into the filename.
    """
    flags = te_perf_flags(hipcc, extra=extra)
    payload = "\x00".join(flags) if flags else "no-te-flags"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
