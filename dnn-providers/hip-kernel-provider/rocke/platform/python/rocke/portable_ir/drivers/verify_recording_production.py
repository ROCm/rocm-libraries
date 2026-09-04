#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# verify_recording_production.py -- wire RecordingIRBuilder into UNMODIFIED
# production builders and check it records the right thing.
#
# For each production kernel we:
#   1. run the real build_* via record_kernel() (rebinds IRBuilder -> recorder)
#      to obtain (KernelDef, live-recorded recipe);
#   2. derive an INDEPENDENT recipe by post-hoc walking the same KernelDef
#      (kernel_to_recipe); the live recording must match it exactly -- this is
#      the correctness check (region routing / op ordering / nothing dropped);
#   3. cross-check against the already-byte-identical-verified kerneldef_to_recipe
#      walk; these agree except for N-result ops, where the recorder is richer
#      ('outs' vs the older single 'out'). Such ops are reported, not failed.
#
# No device / comgr needed -- building production kernels is pure Python.
#
#   python3 -m rocke.portable_ir.drivers.verify_recording_production

import sys

from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe
from rocke.portable_ir.src.recording_builder import kernel_to_recipe, record_kernel


def _count(prog):
    n = 0
    for i in prog:
        if i.get("op") in ("emit", "scf_for", "scf_if", "ret"):
            n += 1
        for key in ("body", "then", "else"):
            if key in i:
                n += _count(i[key])
    return n


def _multi_result_opcodes(prog):
    found = []
    for i in prog:
        if i.get("op") == "emit" and "outs" in i:
            found.append(i["opcode"])
        for key in ("body", "then", "else"):
            if key in i:
                found += _multi_result_opcodes(i[key])
    return found


def _cases():
    """(label, builder-module, zero-arg build callable) for real kernels."""
    from rocke.instances.common import (
        elementwise,
        reduce as reduce_mod,
        transpose,
    )

    # attention_unified lives in the rocke LIBRARY tree (kernels/), not platform.
    from kernels.common import attention_unified
    from rocke.portable_ir.examples import export_mha

    yield (
        "attn2d fp16 D128",
        attention_unified,
        lambda: export_mha.build("fp16", 128, 2048, 1, 32, 1),
    )
    yield (
        "attn2d bf16 D64",
        attention_unified,
        lambda: export_mha.build("bf16", 64, 4096, 1, 32, 1),
    )
    yield (
        "attn2d fp16 D128 gqa8",
        attention_unified,
        lambda: export_mha.build("fp16", 128, 2048, 1, 32, 8),
    )
    yield (
        "elementwise silu bf16",
        elementwise,
        lambda: elementwise.build_elementwise(
            elementwise.ElementwiseSpec(op="silu", dtype="bf16", block_size=64, vec=8)
        ),
    )
    yield (
        "reduce sum f16",
        reduce_mod,
        lambda: reduce_mod.build_reduce2d(
            reduce_mod.Reduce2DSpec(
                n_per_block=4096,
                op="sum",
                block_size=256,
                vec=4,
                dtype="f16",
                wave_size=64,
            )
        ),
    )
    yield (
        "transpose f16 64x64",
        transpose,
        lambda: transpose.build_transpose2d(
            transpose.Transpose2DSpec(
                tile_m=64, tile_n=64, vec=8, dtype="f16", lds_pad=8, grid_order="row"
            )
        ),
    )


def main() -> int:
    print(f"{'kernel':<26} {'ops':>6}  {'rec==walk':>9}  {'==legacy':>9}  multi-result")
    print("-" * 78)
    failures = 0
    for label, module, build in _cases():
        kernel, recorded = record_kernel(build, module)

        oracle = kernel_to_recipe(kernel)  # independent post-hoc walk
        match_walk = recorded == oracle

        legacy = kerneldef_to_recipe(kernel)  # byte-identity-proven path
        match_legacy = recorded == legacy
        multi = sorted(set(_multi_result_opcodes(recorded["program"])))
        # legacy mismatch is EXPECTED iff there are N-result ops (richer 'outs').
        legacy_ok = match_legacy or bool(multi)

        ok = match_walk and legacy_ok
        failures += not ok
        nops = _count([i for i in recorded["program"] if i.get("op") != "param"])
        print(
            f"{label:<26} {nops:>6}  {'PASS' if match_walk else 'FAIL':>9}  "
            f"{('PASS' if match_legacy else ('N/A*' if multi else 'FAIL')):>9}  "
            f"{','.join(multi) if multi else '-'}"
        )

    print("-" * 78)
    print(
        "* N/A: recorder emits richer 'outs' for multi-result ops; legacy walk keeps single 'out'."
    )
    if failures:
        print(f"\nFAILED: {failures} kernel(s) recorded incorrectly")
        return 1
    print("\nOK: every production kernel records identically to its KernelDef")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
