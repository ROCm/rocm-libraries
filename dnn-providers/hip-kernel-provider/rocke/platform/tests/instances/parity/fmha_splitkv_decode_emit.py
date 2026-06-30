#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_splitkv_decode_emit.py -- Python reference emitter for the
# split-KV decode FMHA parity harness. Selects one of 6 sampled
# FmhaFwdSplitKvDecodeSpec configs by argv[1] (0..5) and a phase by argv[2]
# ("seg" or "reduce"), builds via build_fmha_fwd_splitkv_decode_segment /
# build_fmha_fwd_splitkv_decode_reduce and prints
# _native_lower(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter fmha_splitkv_decode_emit.c.
import sys

from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_splitkv_decode import (
    FmhaFwdSplitKvDecodeSpec,
    build_fmha_fwd_splitkv_decode_segment,
    build_fmha_fwd_splitkv_decode_reduce,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int) -> FmhaFwdSplitKvDecodeSpec:
    if idx == 0:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
                dtype="f16",
                mask_mode="none",
            ),
            batch=1,
            num_segments=4,
        )
    if idx == 1:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=8),
                dtype="f16",
                mask_mode="causal",
            ),
            batch=2,
            num_segments=8,
        )
    if idx == 2:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=16, num_kv_heads=2),
                dtype="bf16",
                mask_mode="none",
            ),
            batch=4,
            num_segments=16,
            use_mfma_body=False,
        )
    if idx == 3:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=32, num_kv_heads=4),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=2048,
            ),
            batch=1,
            num_segments=32,
        )
    if idx == 4:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=12, num_kv_heads=3),
                dtype="bf16",
                mask_mode="none",
            ),
            batch=8,
            num_segments=64,
        )
    if idx == 5:
        return FmhaFwdSplitKvDecodeSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=16, num_kv_heads=8),
                dtype="f16",
                mask_mode="causal",
            ),
            batch=2,
            num_segments=128,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: fmha_splitkv_decode_emit.py <config_index 0..5> [<seg|reduce>|<ll|ir|verify>]\n"
        )
        return 2
    idx = int(sys.argv[1])

    # argv[2] may be a phase ("seg"/"reduce") or a mode ("ll"/"ir"/"verify").
    # If it looks like a mode, treat it as mode with default phase "seg".
    # If it looks like a phase, treat it as phase with default mode "ll".
    # If absent, default phase="seg", mode="ll".
    phase = "seg"
    mode = "ll"
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if arg2 in ("ir", "verify", "ll"):
            mode = arg2
        elif arg2 in ("seg", "reduce"):
            phase = arg2
        else:
            sys.stderr.write(f"unknown phase {arg2!r} (want seg|reduce)\n")
            return 2

    spec = _spec(idx)
    if phase == "seg":
        kernel = build_fmha_fwd_splitkv_decode_segment(spec, arch="gfx950")
    else:
        kernel = build_fmha_fwd_splitkv_decode_reduce(spec, arch="gfx950")

    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
