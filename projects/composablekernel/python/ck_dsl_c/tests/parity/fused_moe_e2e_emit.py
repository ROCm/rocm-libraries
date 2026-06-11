#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fused_moe_e2e_emit.py -- Python reference emitter for the
# fused_moe_e2e (end-to-end fused-MoE forward orchestrator) parity harness.
#
# Selects one of the sampled FusedMoeForwardSpec configs by argv[1] (the config
# index), constructs FusedMoeForward(spec) so the __init__ shape-aware tile-swap
# policy + arch resolution runs exactly as the live orchestrator would, then
# lowers each lowerable pipeline stage to AMDGPU LLVM IR via lower_kernel_to_llvm
# and prints them concatenated to stdout, each prefixed with a stage banner, so
# the output can be byte-compared with the C emitter fused_moe_e2e_emit.c.
#
# The orchestrator emits NO single monolithic kernel; it composes pre-existing
# sub-kernel builders selected by the (tile-policy-adjusted) spec. The C port's
# ckc_fused_moe_forward_lower_to_llvm currently wires three lowerable stages --
# ROUTER (topk_softmax), GATE_UP_GEMM and DOWN_GEMM (both the batched-GEMM
# builder shape) -- so this emitter lowers exactly those three, in the same
# order, to keep the byte comparison apples-to-apples. (The sort / gather /
# silu_mul / topk_reduce stages return NOTIMPL on the C side and are therefore
# excluded from the diff here.)
import sys

from ck_dsl import lower_kernel_to_llvm
from ck_dsl.instances.common.fused_moe_e2e import (
    FusedMoeForwardSpec,
    FusedMoeForward,
)
from ck_dsl.instances.common.topk_softmax import build_topk_softmax
from ck_dsl.instances.common.batched_gemm import build_batched_gemm


# The six sampled configs (must match the C emitter's make_spec()).
_CONFIGS = (
    dict(tokens=1, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="f16"),
    dict(tokens=8, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="f16"),
    dict(tokens=32, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="f16"),
    dict(tokens=128, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="f16"),
    dict(tokens=1, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="bf16"),
    dict(tokens=128, experts=32, topk=5, hidden=8192, intermediate=8192, dtype="f16"),
)


def _spec(idx: int) -> FusedMoeForwardSpec:
    if idx < 0 or idx >= len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    cfg = _CONFIGS[idx]
    return FusedMoeForwardSpec(arch="gfx950", **cfg)


# Lowerable stages, in the C emitter's order. Each entry is (banner, builder)
# where the builder takes the (tile-policy-adjusted) FusedMoeForward instance.
def _emit_router(fwd: FusedMoeForward) -> str:
    spec = fwd.spec.to_topk_softmax_spec()
    kernel = build_topk_softmax(spec)
    return lower_kernel_to_llvm(kernel, arch="gfx950")


def _emit_batched_gemm(fwd: FusedMoeForward) -> str:
    # GATE_UP_GEMM and DOWN_GEMM both lower via the batched-GEMM builder shape
    # (the orchestrator parameterises it per-call; the representative .ll for
    # golden diffing is the batched_gemm builder), matching the C port.
    spec = fwd.spec.to_batched_gemm_spec()
    kernel = build_batched_gemm(spec)
    return lower_kernel_to_llvm(kernel, arch="gfx950")


_STAGES = (
    ("ROUTER", _emit_router),
    ("GATE_UP_GEMM", _emit_batched_gemm),
    ("DOWN_GEMM", _emit_batched_gemm),
)


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: fused_moe_e2e_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    # Construct the orchestrator so __init__ runs the arch resolve + shape-aware
    # tile-swap policy; the stages below read the post-swap fwd.spec.
    fwd = FusedMoeForward(spec)
    out = []
    for banner, emit in _STAGES:
        out.append(f"; === fused_moe_e2e stage: {banner} ===\n")
        out.append(emit(fwd))
        if not out[-1].endswith("\n"):
            out.append("\n")
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
