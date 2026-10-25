#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/barrier_fence_emit.py -- Python reference emitter for the
# barrier ``fence`` attribute (tile.sync / tile.sync_lds_only /
# tile.s_barrier_bare with fence="compiler" or "workgroup"). Selects one config
# by argv[1] and prints, by argv[2] (default "ll"), the native Python engine's
# LLVM IR, the ck.dsl.ir/v1 serialization or the verifier diagnostics:
#
#   0..3  the six-barrier probe of tests/core/test_barrier_fence.{py,cpp}
#         (unfenced, "compiler", "workgroup" on each barrier kind) on gfx942,
#         gfx950, gfx1151 and gfx1201;
#   4     the probe with the builder default fence (``fence_barriers``) on
#         gfx950, an explicit "none" opting one barrier out;
#   5..9  attention backward kernels whose every barrier carries the compiler
#         fence: the general main kernel (gfx942 d64, gfx950 d64 with the
#         LDS-only loop barriers, gfx1151 d32), the split dQ kernel (gfx950)
#         and the THD work-list prep kernel (gfx1201).
#
# The portable-IR harnesses (rocke.portable_ir record_coverage / parity_matrix)
# take the first buildable config, the probe. Config 9 holds an scf.if_else,
# which their recorder does not record yet; the other configs record.
#
# There is no C counterpart (barrier_fence_emit.c): the attention backward
# family is Python-only (no C++ builder of these kernels exists; see the
# C++-engine waiver in dsl_docs/instances/attention_bwd.md). The C++ engine's
# own fence emission is unit-tested by tests/core/test_barrier_fence.cpp (the
# probe), and library/tests/test_barrier_fence_parity_emit.py lowers this
# emitter's "ir" output with rocke_engine.lower_serialized_ir when the engine is
# built, comparing it byte for byte with the "ll" output (the cross-engine
# check of every config, backward kernels included).
from dataclasses import dataclass

from rocke.core import ir as _ir
from _emit_common import run_emit

# fence of the six probe barriers (None: unfenced), in build order; the same
# list as MODES in tests/core/test_barrier_fence.{py,cpp}
PROBE_MODES = (None, "compiler", None, "workgroup", None, "workgroup")


@dataclass(frozen=True)
class FenceProbeSpec:
    """The probe kernel: ``default`` is the builder's ``fence_barriers``."""

    name: str = "fence_probe"
    default: str = ""


@dataclass(frozen=True)
class BwdKernelSpec:
    """One attention backward kernel: ``stage`` and its spec fields."""

    stage: str
    fields: tuple


def _probe(spec: FenceProbeSpec):
    # resolved at call time, so a recorder that rebinds rocke.core.ir.IRBuilder
    # (portable_ir record_kernel) sees the probe
    b = _ir.IRBuilder(spec.name)
    if spec.default:
        b.fence_barriers = spec.default
        b.sync()
        b.sync_lds_only()
        b.s_barrier_bare()
        b.sync(fence="none")
        b.sync(fence="workgroup")
        return b.kernel
    b.sync()
    b.sync(fence="compiler")
    b.sync_lds_only()
    b.sync_lds_only(fence="workgroup")
    b.s_barrier_bare(fence="none")
    b.s_barrier_bare(fence="workgroup")
    return b.kernel


def _bwd(spec: BwdKernelSpec, arch: str):
    fields = dict(spec.fields)
    if spec.stage == "main":
        from kernels.common.attention_bwd import AttnBwdSpec, build_attn_bwd_main

        return build_attn_bwd_main(AttnBwdSpec(**fields), arch=arch)
    if spec.stage == "dq":
        from kernels.common.attention_bwd_dq import AttnBwdDqSpec, build_attn_bwd_dq

        return build_attn_bwd_dq(AttnBwdDqSpec(**fields), arch=arch)
    if spec.stage == "prep":
        from kernels.common.attention_bwd_aux import (
            AttnBwdAuxSpec,
            build_attn_bwd_prep,
        )

        return build_attn_bwd_prep(AttnBwdAuxSpec(**fields), arch=arch)
    raise SystemExit(f"unknown backward stage {spec.stage!r}")


_CONFIGS = {
    0: (FenceProbeSpec(), "gfx942"),
    1: (FenceProbeSpec(), "gfx950"),
    2: (FenceProbeSpec(), "gfx1151"),
    3: (FenceProbeSpec(), "gfx1201"),
    4: (FenceProbeSpec(name="fence_default", default="compiler"), "gfx950"),
    5: (BwdKernelSpec("main", (("head_size", 64),)), "gfx942"),
    6: (
        BwdKernelSpec("main", (("head_size", 64), ("loop_sync", "lds_only"))),
        "gfx950",
    ),
    7: (
        BwdKernelSpec("main", (("head_size", 32), ("mask_class", "none"))),
        "gfx1151",
    ),
    8: (BwdKernelSpec("dq", (("head_size", 64), ("edge_tiles", True))), "gfx950"),
    9: (
        BwdKernelSpec(
            "prep", (("head_size", 64), ("seq_mode", "thd"), ("worklist", True))
        ),
        "gfx1201",
    ),
}


def _spec(idx: int):
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    return _CONFIGS[idx]


def build_barrier_fence_kernel(spec, *, arch: str):
    """The kernel of one config (a probe or a backward kernel)."""
    if isinstance(spec, FenceProbeSpec):
        return _probe(spec)
    return _bwd(spec, arch)


def _kernel(idx: int):
    spec, arch = _spec(idx)
    return build_barrier_fence_kernel(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        build_barrier_fence_kernel,
        usage="usage: barrier_fence_emit.py <config_index 0..9> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
