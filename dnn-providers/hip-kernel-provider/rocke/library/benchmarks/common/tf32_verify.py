# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The TF32 ``--verify`` rules the conv sweep drivers share.

``benchmark_implicit_gemm_conv`` and ``benchmark_direct_conv`` (with its AOT
cache, ``direct_kernel_sweep``) bound, rank and gate a TF32 sweep the same way;
the rules live here once so the two drivers cannot drift.
"""

from __future__ import annotations

# TF32 rows of the --verify bound on the peak-normalised error
# max|out-ref| / max(max|ref|, 1) against an IEEE fp32 reference. Two rows
# because the two arches run TF32 differently: gfx942 encodes XF32 natively
# (operands truncated to a 10-bit mantissa), gfx950 emulates it with three bf16
# MFMAs over the full fp32 payload, about as accurate as fp32. 1e-4 on the
# emulated path still fails a dropped cross term or a single bf16 pass (~2e-3).
# Truncating both operands loses up to 2^-9 of every product, and same-sign
# products keep that loss in the sum (four products of 0x3f001fff: 1.0 against
# 1.00195), so the native bound is twice that, 2^-8; uniform(-1, 1) data lands
# near 8e-4.
TF32_VERIFY_TOL = {
    "tf32_emulated": 1e-4,
    "tf32_native": 2**-8,
}


def rank_verified(results: list, tflops, passed, what: str = "") -> tuple[list, list]:
    """``(ranked, failed)``: ``results`` by TFLOPS, verify failures split off.

    A kernel that failed ``--verify`` is reported, never ranked: it must not
    become the sweep's ``Best:`` (a fast kernel that silently computes in a
    lower precision would otherwise win a TF32 sweep). ``passed is None``
    (verify not run) ranks as before. ``tflops`` / ``passed`` read a result;
    ``what`` names the kernels in the message (e.g. ``"wgrad "``).
    """
    ordered = sorted(results, key=tflops, reverse=True)
    ranked = [r for r in ordered if passed(r) is not False]
    failed = [r for r in ordered if passed(r) is False]
    if failed:
        print(
            f"\n{len(failed)} {what}kernel(s) failed verify and are not ranked.",
            flush=True,
        )
    return ranked, failed


def tf32_implies_verify(args, has_tf32: bool, message: str) -> None:
    """Turn on ``args.verify`` (printing ``message``) when the run has TF32 work.

    A TF32 kernel that silently computes at lower precision still runs fast, so
    a TF32 ranking only means something over verified kernels.
    """
    if has_tf32 and not args.verify:
        print(message)
        args.verify = True
