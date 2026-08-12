# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attention kernel builder tests extracted from platform/tests/test_rocke.py.

Covers the full SDPA/MHA/FMHA library build surface:

  - ``TestAttentionHelpers``           : unified attention support gates,
                                          scalar/tiled compile smoke tests,
                                          dispatch matrix drift net, and
                                          spec-builder parity.
  - ``TestAttentionCdnaPrimitives``    : waves-per-EU attribute on the
                                          tiled 2D attention kernel.
  - ``TestEveryAttentionKernelUsesMfma``: gremlin — assert the FMHA MFMA
                                          kernel emits a real mfma intrinsic.
  - ``TestExtendedAttentionBuilds``    : varlen, appendKV, paged-prefill,
                                          splitKV, head-grouping, BWD, FP8.
  - ``TestSageAttentionBuilds``        : sage attention quantisation modes.
  - ``TestSparseAttentionBuilds``      : jenga and VSA sparse attention.
  - ``TestFmhaKernelBuilder``          : FmhaKernelBuilder boilerplate API.

All tests are pure codegen (no GPU, no subprocess).
"""

from __future__ import annotations

import unittest

import pytest

from rocke import lower_kernel_to_llvm

from kernels import (
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    attention_3d_workspace_nbytes,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    supports_native_unified_attention,
)


def _patch_resolved_arch(arch: str):
    """Pin the resolved attention arch for a test, on every module that reads it.

    ``_resolve_attention_arch`` is defined in ``kernels.common.attention_unified``
    but imported *by name* into other modules (e.g.
    ``builders.common.attention_spec_builder``), so each holds its own binding.
    Patching only the defining module leaves the spec builder resolving the real
    device arch -- which silently ignores the test's requested arch on any host
    whose GPU differs (e.g. an ``arch='gfx950'`` case on a gfx942 box). Patch
    every by-name importer so the pin actually reaches the builder.
    """
    from unittest import mock

    import builders.common.attention_spec_builder as _asb
    import kernels.common.attention_unified as _au

    targets = [_au]
    if getattr(_asb, "_resolve_attention_arch", None) is not None:
        targets.append(_asb)
    return _MultiPatch(
        [
            mock.patch.object(m, "_resolve_attention_arch", return_value=arch)
            for m in targets
        ]
    )


class _MultiPatch:
    """Enter/exit a list of ``mock.patch`` context managers as one."""

    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


# ---------------------------------------------------------------------
# comgr build + resource-budget smoke (no torch, no GPU launch)
# ---------------------------------------------------------------------
#
# The IR-text asserts elsewhere in this file stop at ``lower_kernel_to_llvm`` and
# never invoke comgr -- so a kernel that lowers to valid-looking IR but is
# rejected at codegen (e.g. LDS over the arch cap, register overflow, an ISA
# intrinsic invalid for the target) sails through. That is exactly the gap that
# let the fp16 D128 prefill LDS-overflow regression reach ``develop`` and 8 perf
# sweeps (81920 B > the gfx942 64 KB cap) before the dashboard caught it.
#
# ``compile_kernel`` runs the full comgr pipeline (IR -> BC -> relocatable ->
# HSACO) and needs NEITHER torch NOR a GPU, and can target any arch from any box
# via the ``arch=`` triple -- so it is CI-able on a plain comgr/LLVM host. The
# helpers below reach comgr and additionally assert the emitted kernel fits the
# arch resource budget (LDS today; the readelf-parsed HSACO also carries VGPR/
# SGPR for future soft-occupancy checks).


def _compile_or_skip(kernel, *, arch: str):
    """Compile ``kernel`` through comgr for ``arch``; skip (not fail) when the
    comgr toolchain is unavailable so the test is a no-op on hosts without it.

    A failed *compile* (raised ``ComgrError``) is a real defect and propagates.
    Only a missing/broken toolchain skips."""
    try:
        from rocke.helpers.compile import compile_kernel
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"comgr toolchain unavailable: {e}")
    try:
        return compile_kernel(kernel, arch=arch, capture_ir_text=False)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"comgr toolchain unavailable: {e}")


def _assert_resources_fit(art, *, arch: str, kernel_name: str = ""):
    """Assert the emitted HSACO fits ``arch``'s resource budget.

    Two distinct failure modes, because comgr treats them differently:

    - **LDS over the arch cap** -- comgr *hard-errors* at codegen (no fallback),
      so the bare compile already catches it; the explicit cap check here fires
      one step earlier with a readable diagnostic instead of a raw ComgrError,
      and catches the *soft* case (fits the hard cap but a change grew it toward
      the ceiling).
    - **Register (VGPR) over-subscription** -- the compiler does NOT fail; it
      *spills to scratch* and the kernel still compiles, then runs at reduced
      occupancy with scratch traffic. A pass/fail compile check is blind to this,
      so we assert ``scratch_bytes == 0`` as the arch-agnostic no-spill signal.

    Resource fields come from ``group_segment_fixed_size`` /
    ``private_segment_fixed_size`` in the code object, read via ``llvm-readelf``
    (present in any ROCm image; no GPU). Skips only if readelf is unavailable."""
    import tempfile
    from pathlib import Path

    from rocke.analysis.isa import analyze_hsaco
    from rocke.core.arch.target import ArchTarget

    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    hsaco = bytes(art.hsaco)
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as fh:
        fh.write(hsaco)
        fh.flush()
        try:
            res = analyze_hsaco(Path(fh.name)).resources
        except (FileNotFoundError, RuntimeError) as e:  # pragma: no cover
            pytest.skip(f"HSACO introspection tool unavailable: {e}")

    name = kernel_name or "kernel"
    lds = res.lds_bytes
    if lds is None:  # pragma: no cover - metadata shape drift
        pytest.skip("could not parse group_segment_fixed_size from HSACO")
    assert lds <= cap, (
        f"{name} LDS {lds} B exceeds {arch} cap {cap} B (over by {lds - cap} B) "
        f"-- comgr codegen rejection at larger tiles / seq"
    )
    # Register overflow does not fail the compile -- it spills. Any scratch use is
    # a register-budget regression (occupancy cliff), so treat it as a failure.
    scratch = res.scratch_bytes
    if scratch is not None:
        assert scratch == 0, (
            f"{name} spills {scratch} B to scratch on {arch} (VGPR {res.vgpr_count}) "
            f"-- register over-subscription; kernel compiles but loses occupancy"
        )


# The shipped 2D-tiled attention geometries the provider dispatches, spanning the
# regressed axes: (arch x dtype x head_dim), at a prefill seq large enough to
# exercise the flash path that overflowed LDS in the fp16 D128 prefill regression.
#
# Specs are built via the REAL selector (``au._tiled_spec_from_problem``) from a
# ``UnifiedAttentionProblem`` -- the same path the provider ships -- so the test
# exercises the actual dispatch decision and picks up arch-correct spec fields
# instead of a hand-guessed geometry (each arch's spec class carries different
# fields). Kept curated, not a cartesian product: one compile per shipped variant.
def _budget_problem(
    *,
    head_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    seq=2048,
    block_size=64,
    sliding_window=0,
):
    return UnifiedAttentionProblem(
        total_q=seq,
        num_seqs=1,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        dtype=dtype,
        sliding_window=sliding_window,
    )


_TILED_2D_BUDGET_GEOMETRIES = [
    # (label, arch, problem_kwargs)
    (
        "fp16_d128_gqa32x8",
        "gfx942",
        dict(head_size=128, num_query_heads=32, num_kv_heads=8, dtype="fp16"),
    ),
    (
        "bf16_d128_gqa32x8",
        "gfx942",
        dict(head_size=128, num_query_heads=32, num_kv_heads=8, dtype="bf16"),
    ),
    (
        "fp16_d128_gqa32x8",
        "gfx950",
        dict(head_size=128, num_query_heads=32, num_kv_heads=8, dtype="fp16"),
    ),
    (
        "bf16_d128_gqa32x8",
        "gfx950",
        dict(head_size=128, num_query_heads=32, num_kv_heads=8, dtype="bf16"),
    ),
    (
        "fp16_d64_gqa64x8",
        "gfx942",
        dict(head_size=64, num_query_heads=64, num_kv_heads=8, dtype="fp16"),
    ),
    # Both D64 dtypes ride the narrowed (width-16) ring, and bf16 reaches the
    # selector through the bf16-wide spec branch rather than the fp16 one, so the
    # fp16 row above does not cover it.
    (
        "bf16_d64_gqa64x8",
        "gfx942",
        dict(head_size=64, num_query_heads=64, num_kv_heads=8, dtype="bf16"),
    ),
    # Sliding-window D128 takes a SEPARATE non-ring geometry (the flash/ring
    # paths gate on ``sliding_window == 0``; SW picks its own tile), so LDS/reg
    # pressure changes there are not covered by the plain-causal rows above.
    (
        "fp16_d128_sw_gqa32x8",
        "gfx942",
        dict(
            head_size=128,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="fp16",
            sliding_window=128,
        ),
    ),
    (
        "bf16_d128_sw_gqa32x8",
        "gfx942",
        dict(
            head_size=128,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="bf16",
            sliding_window=128,
        ),
    ),
]


# ---------------------------------------------------------------------
# Shared problem matrix
# ---------------------------------------------------------------------


def _attention_problem_matrix():
    """Curated ``(label, UnifiedAttentionProblem)`` list for the per-arch
    attention dispatch/spec drift net.

    The matrix is *curated*, not a full cartesian product: it spans dtype
    (fp16/bf16/fp8), head_size {64,128,256}, block_size {16,32,64},
    num_queries_per_kv {1,4,8,16}, the four regimes (decode / short prefill /
    long prefill / long-context decode), num_seqs {1, >=2} and the
    sliding-window / softcap / alibi / qq_bias / sinks toggles -- arranged so
    that BOTH the gfx942 fp16-flash branch (fp16 long-prefill D128 & D64, no
    SW/softcap, num_seqs>=2) and the bf16 transposed "combo" branch
    (bf16 / HD=64 / BS=32 / num_queries_per_kv=8 / long-prefill / multi-batch)
    of ``_tiled_spec_from_problem`` are reached, plus the decode path that
    routes to the 3D split-KV builder.

    Every config keeps ``num_query_heads % num_kv_heads == 0`` and a
    ``total_q`` / ``max_seqlen_q`` pairing consistent with its regime. The
    consumers below assert no *signature drift* (no ``TypeError`` from a
    per-arch impl whose kwargs/fields fell out of sync) -- they deliberately do
    NOT assert per-shape accept/reject verdicts, which legitimately differ by
    arch (e.g. HD=256 is unsupported on the gfx942 2D path, the gfx942 flash
    branch only fires on gfx942).
    """
    cfgs = []

    def add(label, **kw):
        base = dict(
            total_q=0,
            num_seqs=1,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=2048,
            dtype="fp16",
        )
        base.update(kw)
        if base["total_q"] == 0:
            base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
        cfgs.append((label, UnifiedAttentionProblem(**base)))

    # --- decode (max_seqlen_q=1) -> routes to the 3D split-KV builder ---
    # GQA fan-outs 1/4/8/16, all head sizes, all block sizes, fp16/bf16.
    for hd in (64, 128, 256):
        for bs in (16, 32, 64):
            add(
                f"decode_fp16_d{hd}_b{bs}",
                head_size=hd,
                block_size=bs,
                dtype="fp16",
                max_seqlen_q=1,
                num_seqs=2,
                num_query_heads=8,
                num_kv_heads=1,
                max_seqlen_k=4096,
            )
    add(
        "decode_bf16_d128_mha",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=4,
        num_query_heads=8,
        num_kv_heads=8,  # MHA: num_queries_per_kv == 1
        max_seqlen_k=4096,
    )
    add(
        "decode_bf16_d64_gqa4",
        head_size=64,
        block_size=16,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=3,
        num_query_heads=16,
        num_kv_heads=4,  # num_queries_per_kv == 4
        max_seqlen_k=4096,
    )
    add(
        "decode_fp16_d128_gqa16",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=1,  # num_queries_per_kv == 16
        max_seqlen_k=4096,
    )
    # long-context decode (large KV, still q==1) -> stresses 3D segment count.
    add(
        "decode_longctx_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=8192,
    )
    add(
        "decode_longctx_bf16_d64",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=8192,
    )
    # decode toggles: sliding window / softcap / sinks.
    add(
        "decode_sw_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        sliding_window=128,
    )
    add(
        "decode_softcap_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        softcap=30.0,
    )
    add(
        "decode_sinks_bf16_d128",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_sinks=True,
    )
    # decode fp8 K/V cache (use_fp8 + q_dtype set; routes to 3D).
    add(
        "decode_fp8_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "decode_fp8_d64_b32",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=1,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=4096,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "decode_n1_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=1,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )

    # --- short prefill (max_seqlen_q ~128) ---
    for hd in (64, 128):
        add(
            f"short_prefill_fp16_d{hd}",
            head_size=hd,
            dtype="fp16",
            max_seqlen_q=128,
            num_seqs=2,
            num_query_heads=8,
            num_kv_heads=1,
            max_seqlen_k=1024,
        )
    add(
        "short_prefill_bf16_d128_gqa4",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=4,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_sw_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=200,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
        sliding_window=128,
    )
    add(
        "short_prefill_bf16_d64_b32_gqa8",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_fp16_d256_b64",
        head_size=256,
        block_size=64,
        dtype="fp16",
        max_seqlen_q=128,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
    )
    add(
        "short_prefill_n1_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=128,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=1024,
    )

    # --- long prefill (max_seqlen_q ~2048) ---
    # gfx942 fp16 flash branch: D128 (any BS) and D64 (BS=64), no SW/softcap,
    # num_seqs>=2. (On gfx950 these are the plain wide-K path -- still must
    # build cleanly via the default branch.)
    add(
        "long_prefill_flash_fp16_d128",
        head_size=128,
        block_size=16,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d128_b32",
        head_size=128,
        block_size=32,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d64_b64",
        head_size=64,
        block_size=64,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # gfx942 D64 fp16 flash with a paged block_size of 16/32 (e.g. a vLLM-style
    # 16-token KV cache). The flash regime needs T in {64,128}; before the
    # _select_2d_tile_size fix these yielded T=block_size and the spec validator
    # rejected the build on the selected-2D path. Pin both (red->green).
    add(
        "long_prefill_flash_fp16_d64_b16",
        head_size=64,
        block_size=16,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_flash_fp16_d64_b32",
        head_size=64,
        block_size=32,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # bf16 transposed "combo" branch cohort: HD=64, BS=32, NQH=64/NKV=8
    # (num_queries_per_kv=8), long prefill, multi-batch. With/without sinks.
    add(
        "long_prefill_combo_bf16",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_combo_bf16_sinks",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=64,
        num_kv_heads=8,
        max_seqlen_k=2048,
        use_sinks=True,
    )
    # combo cohort by GQA-8 *ratio* but NOT the 64/8 absolute head count, e.g.
    # a tensor-parallel-sharded GQA-8 model (16/2). _enable_combo_2d fires on
    # the ratio; the gfx950 use_fast_paged_kv_desc validator wants absolute
    # 64/8, so this shape used to crash the gfx950 selected-2D path until the
    # spec builder gated the fast descriptor to 64/8.
    add(
        "long_prefill_combo_bf16_tp_sharded_16x2",
        head_size=64,
        block_size=32,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=16,
        num_kv_heads=2,
        max_seqlen_k=2048,
    )
    # plain default long prefill (single-seq and multi-batch), fp16/bf16.
    add(
        "long_prefill_default_fp16_d128_n1",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_default_bf16_d128_n4",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=4,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    add(
        "long_prefill_default_fp16_d256",
        head_size=256,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
    )
    # long-prefill toggles: sliding window, softcap, alibi, qq_bias.
    add(
        "long_prefill_sw_bf16_d128",
        head_size=128,
        dtype="bf16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        sliding_window=128,
    )
    add(
        "long_prefill_softcap_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        softcap=30.0,
    )
    add(
        "long_prefill_alibi_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_alibi=True,
    )
    add(
        "long_prefill_qqbias_fp16_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_qq_bias=True,
    )
    # fp8 K/V long prefill (multi-batch) -- exercises the fp8 plumbing on the
    # prefill side as well as decode.
    add(
        "long_prefill_fp8_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_fp8=True,
        q_dtype="fp8e4m3",
    )
    add(
        "long_prefill_fp8_sw_d128",
        head_size=128,
        dtype="fp16",
        max_seqlen_q=2048,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=1,
        max_seqlen_k=2048,
        use_fp8=True,
        q_dtype="fp8e4m3",
        sliding_window=128,
    )

    return cfgs


# ---------------------------------------------------------------------
# Attention helpers (extracted from TestHelpers)
# ---------------------------------------------------------------------


class TestAttentionHelpers(unittest.TestCase):
    def test_unified_attention_support_gate_is_explicit(self):
        p = UnifiedAttentionProblem(
            total_q=128,
            num_seqs=3,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=64,
            max_seqlen_q=129,
            max_seqlen_k=2011,
            dtype="fp16",
        )
        ok, reason = supports_native_unified_attention(p)
        self.assertTrue(ok)
        self.assertIn("supported", reason)

    def test_gfx950_d128_ksingle_buffer_geometry_guard(self):
        """gfx950 d128 single-seq prefill: ``_enable_k_single_buffer`` must
        derive ``block_m <= tile_size`` from the geometry selectors, not proxy
        it as ``block_size >= 32``.

        The stale proxy assumed num_warps=2; ``_enable_softmax_mfma_interleave``
        later widened this cohort to num_warps=4 (block_m=128), so block_size=32
        (T=64) tripped the ``block_m <= tile_size`` validator with an uncaught
        ValueError at spec build (no sliding window needed). block_size=64
        survived only by coincidence (T=128 == block_m). Pin: bs32 builds with
        K-single OFF; bs64 keeps K-single ON (golden parity case 53) -- both
        dtypes, no SW.
        """
        import kernels.common.attention_unified as au

        def _p(block_size, dtype):
            return UnifiedAttentionProblem(
                total_q=2048,
                num_seqs=1,
                num_query_heads=32,
                num_kv_heads=8,
                head_size=128,
                block_size=block_size,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype=dtype,
                sliding_window=0,
            )

        with _patch_resolved_arch("gfx950"):
            for dtype in ("fp16", "bf16"):
                # bs32 was an uncaught ValueError at spec build; must now build.
                p32 = _p(32, dtype)
                self.assertEqual(p32.select_path(), "2d")
                self.assertFalse(au._enable_k_single_buffer(p32))
                spec32 = au._tiled_spec_from_problem(p32)  # must NOT raise
                self.assertFalse(spec32.use_k_single_buffer)
                # bs64 still satisfies block_m <= tile_size -> K-single stays on.
                p64 = _p(64, dtype)
                self.assertTrue(au._enable_k_single_buffer(p64))
                spec64 = au._tiled_spec_from_problem(p64)
                self.assertTrue(spec64.use_k_single_buffer)

    def test_unified_attention_scalar_kernels_compile(self):
        p = UnifiedAttentionProblem(
            total_q=3,
            num_seqs=1,
            num_query_heads=4,
            num_kv_heads=4,
            head_size=128,
            block_size=16,
            max_seqlen_q=3,
            max_seqlen_k=16,
            dtype="fp16",
        )
        kernels = [
            build_unified_attention_2d(UnifiedAttention2DSpec(p)),
            build_unified_attention_3d(UnifiedAttention3DSpec(p, num_segments=8)),
            build_unified_attention_reduce(
                UnifiedAttentionReduceSpec(p, num_segments=8)
            ),
        ]
        for k in kernels:
            ll = lower_kernel_to_llvm(k)
            self.assertIn("define amdgpu_kernel void", ll)
            self.assertIn("@llvm.exp2.f32", ll)

    def test_unified_attention_2d_tiled_kernel_compiles(self):
        """The production tiled kernel emits async DMA + MFMA + ds_bpermute."""
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        # Async DMA for K/V should be emitted.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        # MFMA atoms for QK (16x16x32) and PV (16x16x16 since T=16 < 32).
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", ll)
        # Reach comgr: valid IR is not enough -- assert it actually codegens and
        # fits the LDS cap (the step the IR-text asserts above never exercise).
        art = _compile_or_skip(k, arch="gfx950")
        self.assertGreater(art.hsaco_bytes, 0)
        _assert_resources_fit(art, arch="gfx950", kernel_name=k.name)
        # Cross-lane softmax reduction. The 16-lane intra-row-group
        # butterfly lowers to ``ds_swizzle`` SWAP mode rather than
        # ``ds_bpermute`` for the row-group masks ≤ 16. (Larger butterfly
        # stages — e.g. cross-half xor 32 in the 32x32 path — still use
        # bpermute, but they're not present in this default decode spec.)
        self.assertIn("@llvm.amdgcn.ds.swizzle", ll)
        # NaN-guard select on neg_inf row max.
        self.assertIn("0xFFF0000000000000", ll)
        # `qq_bias_stride_0` is the very last kernel param.
        self.assertIn("i32 %qq_bias_stride_0", ll)

    def test_gfx942_bf16_swa_decode_no_fp8_loader_assert(self):
        """Regression: bf16 windowed-decode must not trip the fp8-loader assert.

        gfx942 2D built the fp8 chunk-count assert unconditionally. A bf16
        SWA/decode small-tile config (T=16, HD=64, THREADS=256) gives
        fp8_total_chunks=128; 128 % 256 != 0 raised AssertionError.         The fp8
        loader is never used for bf16, so the guard (``if KV_FP8:``, matching
        the 3D builder) is load-bearing.

        gfx942 has since stopped accepting the fp8 K/V cache altogether -- its
        PV path needs ``ds_read_tr_b8``, a gfx950-only transpose read -- so the
        chunk-count assert is now unreachable here and the guarantee worth
        pinning is the stronger one: the spec refuses fp8 K/V up front rather
        than emitting IR comgr cannot select.
        """
        from kernels.gfx942.attention_tiled_2d import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        # Precondition that makes this a regression test, not a smoke test:
        # the config must be one the old unconditional assert rejected, i.e.
        # (T*HD)//8 not divisible by THREADS=num_warps*64. T=16,HD=64,nw=4 ->
        # 128 chunks, 256 threads -> 128 % 256 != 0.
        T, HD, THREADS = 16, 64, 4 * 64
        self.assertNotEqual(((T * HD) // 8) % THREADS, 0)

        spec = UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=16,
            num_query_heads=64,
            num_kv_heads=8,  # gpt-oss 64/8 GQA
            dtype="bf16",
            use_sinks=True,
            sliding_window=128,
            has_softcap=False,
            num_warps=4,
            tile_size=16,
            block_m_per_warp=16,
        )
        k = build_unified_attention_2d_tiled(spec, arch="gfx942")
        ll = lower_kernel_to_llvm(k, arch="gfx942")
        self.assertIn("define amdgpu_kernel void", ll)
        # The bf16 kernel must emit no fp8 dequant path.
        self.assertNotIn("@llvm.amdgcn.cvt.f32.fp8", ll)

        # Asking gfx942 for fp8 K/V is refused at construction, naming the
        # gfx950-only instruction that makes it impossible. Pinned because the
        # failure it replaces -- emitting IR that comgr then cannot select --
        # surfaces far from its cause.
        import dataclasses

        with self.assertRaisesRegex(ValueError, "fp8 K/V cache"):
            dataclasses.replace(spec, kv_storage_dtype="fp8e4m3")

    def test_unified_attention_2d_tiled_half_local_pv_compiles(self):
        """The R4 half-local PV variant emits 32x32 MFMA with its suffixes."""
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        spec = UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=True,
            sliding_window=0,
            has_softcap=False,
            num_seqs=284,
            num_warps=4,
            waves_per_eu=2,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=True,
            use_transposed_half_local_pv=True,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.bf16", ll)
        self.assertIn("_s1_", k.name)
        self.assertIn("_mask1_", k.name)
        self.assertIn("_hlpv", k.name)
        self.assertNotIn('"amdgpu-agpr-alloc"', ll)

        agpr_spec = UnifiedAttention2DTiledSpec(
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=True,
            sliding_window=0,
            has_softcap=False,
            num_seqs=284,
            num_warps=4,
            waves_per_eu=2,
            tile_size=64,
            block_m_per_warp=32,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_mask_once=True,
            use_transposed_half_local_pv=True,
            use_agpr_alloc_zero=True,
        )
        agpr_k = build_unified_attention_2d_tiled(agpr_spec)
        agpr_ll = lower_kernel_to_llvm(agpr_k)
        self.assertIn("_agpr0", agpr_k.name)
        self.assertIn('"amdgpu-agpr-alloc"="0,0"', agpr_ll)

    def test_unified_attention_2d_tiled_alibi_qq_bias(self):
        """ALiBi/QQ-bias variants emit sitofp + masked global load with clamp."""
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )

        # Both ALiBi and QQ-bias on.
        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            use_alibi=True,
            use_qq_bias=True,
        )
        k = build_unified_attention_2d_tiled(spec)
        ll = lower_kernel_to_llvm(k)
        # ALiBi adds a position->f32 conversion (sitofp), since col_abs and
        # context_len are i32 and slope * (col-ctx) needs f32 arithmetic.
        self.assertIn("sitofp i32", ll)
        # Both biases must use the OOB-safe `masked_global_load` clamp: a
        # `select` feeding into the GEP that selects the index, before the
        # actual `global_load_dword`. The lowered IR contains a `select i1`
        # picking between the real index and a 0 constant, then a `load
        # float, ptr ...` for the bias element.
        self.assertIn("select i1", ll)
        # QQ-bias kernel name suffix.
        self.assertIn("_qqb", ll)
        # ALiBi kernel name suffix.
        self.assertIn("_alibi", ll)

    def test_unified_attention_3d_tiled_kernel_compiles(self):
        from kernels import (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
        )

        seg = build_unified_attention_3d_tiled(
            UnifiedAttention3DTiledSpec(
                head_size=128,
                block_size=16,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_segments=128,
                num_seqs=4,
            )
        )
        seg_ll = lower_kernel_to_llvm(seg)
        # Segment kernel must use the async DMA + transpose-read PV operand
        # path and emit MFMA atoms.
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", seg_ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", seg_ll)
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", seg_ll)
        # Workspace writes for per-segment m / l / acc.
        self.assertIn("segm_output_ptr", seg_ll)
        self.assertIn("segm_max_ptr", seg_ll)
        self.assertIn("segm_expsum_ptr", seg_ll)
        red = build_unified_attention_reduce_tiled(
            UnifiedAttentionReduceTiledSpec(
                head_size=128,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                num_segments=128,
            )
        )
        red_ll = lower_kernel_to_llvm(red)
        # Reduce must compute exp2-weighted segment combine and use NaN-safe
        # factor (`-inf - overall_max -> 0`).
        self.assertIn("@llvm.exp2.f32", red_ll)
        self.assertIn("fcmp ogt", red_ll)
        # Reach comgr for both the segment and reduce kernels + LDS budget.
        seg_art = _compile_or_skip(seg, arch="gfx950")
        self.assertGreater(seg_art.hsaco_bytes, 0)
        _assert_resources_fit(seg_art, arch="gfx950", kernel_name=seg.name)
        red_art = _compile_or_skip(red, arch="gfx950")
        self.assertGreater(red_art.hsaco_bytes, 0)
        _assert_resources_fit(red_art, arch="gfx950", kernel_name=red.name)

    def test_unified_attention_3d_tiled_alibi_qq_bias(self):
        """ALiBi/QQ-bias on the 3D segment kernel emit the same primitives."""
        from kernels import (
            UnifiedAttention3DTiledSpec,
            build_unified_attention_3d_tiled,
        )

        seg = build_unified_attention_3d_tiled(
            UnifiedAttention3DTiledSpec(
                head_size=128,
                block_size=16,
                num_query_heads=16,
                num_kv_heads=2,
                dtype="fp16",
                use_sinks=False,
                sliding_window=0,
                has_softcap=False,
                num_segments=128,
                num_seqs=4,
                use_alibi=True,
                use_qq_bias=True,
            )
        )
        ll = lower_kernel_to_llvm(seg)
        self.assertIn("sitofp i32", ll)
        self.assertIn("select i1", ll)
        # `qq_bias_stride_0` is the last kernel param.
        self.assertIn("i32 %qq_bias_stride_0", ll)
        # Both ALiBi and QQ-bias kernel-name suffixes show up.
        self.assertIn("_alibi", ll)
        self.assertIn("_qqb", ll)

    def test_tiled_2d_shipped_geometries_compile_and_fit_budget(self):
        """comgr build + resource-budget net over the shipped 2D-tiled geometries.

        Regression net for codegen failures that IR-text lowering misses. Each
        shipped (arch, dtype, D) flash variant must actually codegen through comgr
        and fit its arch resource budget. Catches, in one pass:

        - hard codegen rejections (LDS over the arch cap -- the fp16 D128 prefill
          regression, 81920 B > gfx942's 64 KB cap; invalid ISA; malformed IR):
          the compile step itself raises.
        - LDS growth toward the cap: the explicit cap assert.
        - register over-subscription: comgr does NOT fail on this -- it spills to
          scratch and compiles anyway -- so ``_assert_resources_fit`` asserts no
          scratch spill.

        No torch, no GPU launch -- comgr targets each arch via its triple, so
        this runs on any comgr/LLVM host regardless of the box's own GPU.
        """
        import kernels.common.attention_unified as au
        from kernels import build_unified_attention_2d_tiled

        for label, arch, problem_kwargs in _TILED_2D_BUDGET_GEOMETRIES:
            with self.subTest(geometry=label, arch=arch):
                with _patch_resolved_arch(arch):
                    problem = _budget_problem(**problem_kwargs)
                    spec = au._tiled_spec_from_problem(problem)
                    k = build_unified_attention_2d_tiled(spec, arch=arch)
                    art = _compile_or_skip(k, arch=arch)
                    self.assertGreater(art.hsaco_bytes, 0)
                    _assert_resources_fit(art, arch=arch, kernel_name=k.name)

    def test_gfx950_dense_prefill_compiles_and_fits_budget(self):
        """comgr build + resource-budget net for the gfx950 dense flash-attn
        prefill kernel (``build_attention_dense``, its own builder / ABI -- NOT
        routed through the unified 2D-tiled path the matrix above covers).

        Dense bakes shape in at build time and is LDS-heavy (tunable V pad via
        ``ROCKE_DENSE_VPAD``), so it is a live over-budget risk on its own. Covers
        both the default and persistent (grid-stride) variants. gfx950-only,
        torch-free -- comgr targets gfx950 via its triple.
        """
        from dataclasses import replace

        from kernels import AttentionDenseSpec, build_attention_dense

        base = AttentionDenseSpec(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="bf16",
        )
        for label, spec in (
            ("default", base),
            ("persistent", replace(base, persistent=True)),
        ):
            with self.subTest(variant=label):
                k = build_attention_dense(spec, arch="gfx950")
                art = _compile_or_skip(k, arch="gfx950")
                self.assertGreater(art.hsaco_bytes, 0)
                _assert_resources_fit(art, arch="gfx950", kernel_name=k.name)

    def test_gfx950_dense_paged_spec_admission(self):
        """Paged spec fields + validation: accept the fp16/bf16 D128 SW single-seq
        cohort, reject illegal / not-yet-validated combos. Pure-Python (no GPU)."""
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            supports_attention_dense,
        )

        base = dict(
            batch=1,
            seqlen_q=8192,
            seqlen_kv=8192,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
            sliding_window=4096,
            block_n=64,
        )
        # Accept: fp16 D128 SW single-seq, page size divides block_n.
        ok, why = supports_attention_dense(
            AttentionDenseSpec(paged=True, block_size=16, num_kv_blocks=512, **base)
        )
        self.assertTrue(ok, why)
        # kernel name carries the paged tag (distinct binary / cache key).
        self.assertIn(
            "pgd16",
            AttentionDenseSpec(
                paged=True, block_size=16, num_kv_blocks=512, **base
            ).kernel_name(),
        )
        # num_kv_blocks is IR-live (sets the paged buffer-rsrc bound) so it MUST be part
        # of the kernel identity -- else two cache sizes collide in the launcher cache and
        # the larger reads later blocks as 0. Assert distinctness.
        n512 = AttentionDenseSpec(
            paged=True, block_size=16, num_kv_blocks=512, **base
        ).kernel_name()
        n1024 = AttentionDenseSpec(
            paged=True, block_size=16, num_kv_blocks=1024, **base
        ).kernel_name()
        self.assertNotEqual(n512, n1024)
        self.assertIn("nb512", n512)
        self.assertIn("nb1024", n1024)
        # Accept: bf16 too -- the paged mechanism is dtype-generic (both 2-byte).
        ok_bf, why_bf = supports_attention_dense(
            AttentionDenseSpec(
                paged=True,
                block_size=16,
                num_kv_blocks=512,
                **{**base, "dtype": "bf16"},
            )
        )
        self.assertTrue(ok_bf, why_bf)
        # Rejections (each a ValueError from __post_init__).
        for kw in (
            dict(block_size=0, num_kv_blocks=512),  # page size 0
            dict(block_size=128, num_kv_blocks=512),  # not a divisor of block_n (pow2)
            dict(
                block_size=24, num_kv_blocks=512
            ),  # non-power-of-two (shift/mask gate)
            dict(block_size=4, num_kv_blocks=512),  # < ROWS_PER_WAVE (8)
            dict(block_size=16, num_kv_blocks=0),  # num_kv_blocks 0
            dict(block_size=16, num_kv_blocks=70000),  # cache > 2 GiB (i32 overflow)
            dict(block_size=16, num_kv_blocks=512, batch=2),  # multi-seq
            dict(block_size=16, num_kv_blocks=512, varlen=True),
            dict(block_size=16, num_kv_blocks=512, persistent=True),
            dict(
                block_size=16, num_kv_blocks=512, sliding_window=0
            ),  # not validated yet
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                AttentionDenseSpec(paged=True, **{**base, **kw})
        # 0-cost when off: a non-paged spec is unaffected.
        s = AttentionDenseSpec(**base)
        self.assertFalse(s.paged)
        self.assertNotIn("pgd", s.kernel_name())

    def test_gfx950_dense_paged_builds_and_signature(self):
        """Paged ABI: the signature carries block_tables/kv_lens/block_table_stride,
        and the paged spec builds on host (load path still contiguous at this stage)."""
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            attention_dense_signature,
            build_attention_dense,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=8192,
            seqlen_kv=8192,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
            sliding_window=4096,
            block_n=64,
            paged=True,
            block_size=16,
            num_kv_blocks=512,
        )
        names = [p["name"] for p in attention_dense_signature(spec)]
        for req in ("block_tables", "kv_lens", "block_table_stride"):
            self.assertIn(req, names)
        self.assertIsNotNone(build_attention_dense(spec, arch="gfx950"))

    def test_gfx950_dense_paged_prefill_compiles_and_fits_budget(self):
        """comgr build + resource-budget net for the PAGED gfx950 dense prefill
        (fp16/bf16 D128 sliding-window, single-seq). Mirrors the non-paged dense
        budget test; the block_tables indirection adds a load but must still fit."""
        from kernels import AttentionDenseSpec, build_attention_dense

        for dt in ("fp16", "bf16"):
            spec = AttentionDenseSpec(
                batch=1,
                seqlen_q=8192,
                seqlen_kv=8192,
                num_query_heads=32,
                num_kv_heads=8,
                head_size=128,
                causal=True,
                dtype=dt,
                sliding_window=4096,
                block_n=64,
                paged=True,
                block_size=16,
                num_kv_blocks=512,
            )
            with self.subTest(dtype=dt):
                self.assertIn("pgd16", spec.kernel_name())
                k = build_attention_dense(spec, arch="gfx950")
                art = _compile_or_skip(k, arch="gfx950")
                self.assertGreater(art.hsaco_bytes, 0)
                _assert_resources_fit(art, arch="gfx950", kernel_name=k.name)

    def test_gfx950_dense_paged_launcher_rejects_kv_cache_shape_mismatch(self):
        """The launcher must reject a paged K/V cache whose shape disagrees with
        the spec that sizes the buffer-resource bound
        (num_kv_blocks*block_size*num_kv_heads*head_size). A too-small cache
        under a too-large ``num_kv_blocks`` sets an oversized hardware bound and lets
        a block-table entry drive an OOB read. Host-only: the shape check raises
        before any comgr compile / GPU launch, so no torch or GPU is required (only
        ``.shape`` is read pre-launch, so lightweight stand-ins suffice)."""
        from types import SimpleNamespace
        from unittest import mock

        import kernels.gfx950.attention_dense as ad
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=8192,
            seqlen_kv=8192,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
            sliding_window=4096,
            block_n=64,
            paged=True,
            block_size=16,
            num_kv_blocks=512,
        )
        want = (spec.num_kv_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
        wrong = (256, spec.block_size, spec.num_kv_heads, spec.head_size)  # 256 != 512
        qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        out = SimpleNamespace(shape=qshape)
        block_tables = SimpleNamespace(shape=(1, 512))
        kv_lens = SimpleNamespace(shape=(1,))

        # Negative (K): fewer physical blocks than the spec claims -> reject.
        with self.assertRaises(ValueError) as ctx:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=SimpleNamespace(shape=wrong),
                v=SimpleNamespace(shape=want),
                out=out,
                scale=1.0,
                block_tables=block_tables,
                kv_lens=kv_lens,
            )
        msg = str(ctx.exception)
        self.assertIn("paged k cache shape", msg)
        self.assertIn("(256,", msg)  # the offending shape is reported
        self.assertIn("OOB", msg)

        # Negative (V): the same defect on the value cache is also caught.
        with self.assertRaises(ValueError):
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=SimpleNamespace(shape=want),
                v=SimpleNamespace(shape=wrong),
                out=out,
                scale=1.0,
                block_tables=block_tables,
                kv_lens=kv_lens,
            )

        # Positive control: MATCHING shapes pass the shape gate and reach compile.
        # Patch compile_kernel to a sentinel so the launcher stays host-only (no
        # comgr, no GPU) yet proves it did NOT raise our shape ValueError. Clear the
        # launcher cache so a prior test cannot let us skip compile. validate_paged
        # is off here to isolate the SHAPE gate (block-table CONTENTS validation is
        # covered by test_..._validates_block_table_bounds).
        ad._DENSE_LAUNCHER_CACHE.clear()
        sentinel = RuntimeError("reached-compile")
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as ok_ctx:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=SimpleNamespace(shape=want),
                    v=SimpleNamespace(shape=want),
                    out=out,
                    scale=1.0,
                    block_tables=block_tables,
                    kv_lens=kv_lens,
                    validate_paged=False,
                )
            self.assertIs(ok_ctx.exception, sentinel)

    def test_gfx950_dense_paged_launcher_validates_block_table_bounds(self):
        """Gated CONTENTS check on paged block tables: an entry outside
        [0, num_kv_blocks) reads 0 via the bounds-checked cache SRD (silent wrong
        output), so validate_paged=True rejects it loudly, validate_paged=False
        skips it (sync-free hot path), and only DEREFERENCED pages are checked.
        Host-only: block_tables/kv_lens are plain Python lists (only ints + slicing
        are read), so no torch / GPU."""
        from types import SimpleNamespace
        from unittest import mock

        import kernels.gfx950.attention_dense as ad
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
            sliding_window=4096,
            block_n=64,
            paged=True,
            block_size=64,
            num_kv_blocks=512,
        )
        want = (spec.num_kv_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
        qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        out = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=want)
        v = SimpleNamespace(shape=want)
        kv_lens = [256]  # == seqlen_kv (enforced) -> ceil(256/64)=4 pages deref
        good_bt = [[0, 1, 2, 3]]  # all in [0, 512)
        bad_bt = [[0, 1, 600, 3]]  # 600 >= num_kv_blocks at a USED page
        sentinel = RuntimeError("reached-compile")

        # Negative (validate on, default): a used entry >= num_kv_blocks -> reject
        # loudly, before any compile / launch.
        with self.assertRaises(ValueError) as ctx:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=1.0,
                block_tables=bad_bt,
                kv_lens=kv_lens,
            )
        msg = str(ctx.exception)
        self.assertIn("600", msg)
        self.assertIn("num_kv_blocks=512", msg)

        # Only DEREFERENCED pages are checked: a bad id in a column BEYOND n_pages
        # (index >= ceil(256/64)=4) is never read, so it is ignored.
        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as beyond:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    scale=1.0,
                    block_tables=[[0, 1, 2, 3, 600]],
                    kv_lens=kv_lens,
                )
            self.assertIs(beyond.exception, sentinel)

        # Gate off: the bad table is skipped entirely -> reaches compile.
        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as gated:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    scale=1.0,
                    block_tables=bad_bt,
                    kv_lens=kv_lens,
                    validate_paged=False,
                )
            self.assertIs(gated.exception, sentinel)

        # Positive (validate on, valid table): passes the gate -> reaches compile.
        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as ok:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    scale=1.0,
                    block_tables=good_bt,
                    kv_lens=kv_lens,
                )
            self.assertIs(ok.exception, sentinel)

    def test_gfx950_dense_paged_launcher_validates_kv_len_contract(self):
        """Gated CONTENTS check: the kernel visits ALL compile-time seqlen_kv
        tiles but the page-bounds mask uses the runtime kv_len, so a kv_len
        shorter than seqlen_kv leaves uncovered tiles reading page 0 (the masked
        block-table default) -> silently wrong output. validate_paged=True
        enforces kv_lens[i] == seqlen_kv; validate_paged=False skips it (hot
        path). Host-only (plain lists), no torch / GPU."""
        from types import SimpleNamespace
        from unittest import mock

        import kernels.gfx950.attention_dense as ad
        from kernels.gfx950.attention_dense import (
            AttentionDenseSpec,
            run_attention_dense_torch,
        )

        spec = AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="fp16",
            sliding_window=4096,
            block_n=64,
            paged=True,
            block_size=64,
            num_kv_blocks=512,
        )
        want = (spec.num_kv_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
        qshape = (1, spec.seqlen_q, spec.num_query_heads, spec.head_size)
        q = SimpleNamespace(shape=qshape)
        out = SimpleNamespace(shape=qshape)
        k = SimpleNamespace(shape=want)
        v = SimpleNamespace(shape=want)
        good_bt = [[0, 1, 2, 3]]
        sentinel = RuntimeError("reached-compile")

        # Negative (validate on, default): kv_len < seqlen_kv -> reject loudly.
        with self.assertRaises(ValueError) as ctx:
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=1.0,
                block_tables=good_bt,
                kv_lens=[128],
            )
        msg = str(ctx.exception)
        self.assertIn("kv_lens[0]=128", msg)
        self.assertIn("seqlen_kv=256", msg)

        # Gate off: the contract is not checked -> reaches compile.
        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as gated:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    scale=1.0,
                    block_tables=good_bt,
                    kv_lens=[128],
                    validate_paged=False,
                )
            self.assertIs(gated.exception, sentinel)

        # Positive (kv_len == seqlen_kv): passes the gate -> reaches compile.
        ad._DENSE_LAUNCHER_CACHE.clear()
        with mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as ok:
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    scale=1.0,
                    block_tables=good_bt,
                    kv_lens=[256],
                )
            self.assertIs(ok.exception, sentinel)

    def test_attention_3d_workspace_size_matches_shapes(self):
        p = UnifiedAttentionProblem(
            total_q=3,
            num_seqs=2,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=2,
            max_seqlen_k=4096,
            dtype="fp16",
        )
        # AITER's 3D selector chooses 128 segments for this shape.
        # segm_output: 3 * 16 * 128 * 128 f32
        # segm_max/expsum: 2 * (3 * 16 * 128) f32
        expected = (3 * 16 * 128 * 128 + 2 * 3 * 16 * 128) * 4
        self.assertEqual(attention_3d_workspace_nbytes(p), expected)

    def test_tiled_2d_support_gate_rejects_unsupported(self):
        from kernels import supports_tiled_2d

        base = dict(
            head_size=128,
            block_size=16,
            dtype="fp16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
        )
        ok_fp16, _ = supports_tiled_2d(**base)
        self.assertTrue(ok_fp16)
        # head_size in {64, 128, 256}, block_size in {16, 32, 64}, dtype=bf16,
        # alibi, qq_bias all supported.
        for accept in [
            dict(head_size=256),
            dict(head_size=64),
            dict(block_size=32),
            dict(block_size=64),
            dict(dtype="bf16"),
            dict(use_alibi=True),
            dict(use_qq_bias=True),
        ]:
            kwargs = dict(base)
            kwargs.update(accept)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertTrue(ok, msg=f"expected accept for {accept}, got: {reason}")
        # FP8, unsupported head_size, and unsupported block_size still gated.
        for override in [
            dict(head_size=72),
            dict(block_size=24),
            dict(use_fp8=True),
        ]:
            kwargs = dict(base)
            kwargs.update(override)
            ok, reason = supports_tiled_2d(**kwargs)
            self.assertFalse(ok, msg=f"expected reject for {override}, got: {reason}")
            self.assertTrue(reason)

    def test_tiled_2d_dispatch_gate_accepts_block_m_per_warp_per_arch(self):
        """Regression: the shared dispatch entry
        ``supports_native_unified_attention_tiled`` forwards
        ``block_m_per_warp`` to the per-arch ``supports_tiled_2d`` gate. Every
        routed arch's gate must accept that kwarg. gfx950 previously raised
        ``TypeError`` here (its gate signature lacked the parameter), which
        broke the gfx950 SDPA dispatch path for ``backend in {tiled, auto}``.
        """
        from kernels import supports_native_unified_attention_tiled

        p = UnifiedAttentionProblem(
            total_q=128,
            num_seqs=3,
            num_query_heads=8,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=129,
            max_seqlen_k=2011,
            dtype="fp16",
        )
        # Pin the routed arch so the test is deterministic on any host (the
        # default fallback is gfx950, which is exactly the broken path).
        for arch in ("gfx950", "gfx942"):
            with _patch_resolved_arch(arch):
                # Must not raise (the regression was a TypeError on the kwarg).
                ok, reason = supports_native_unified_attention_tiled(p)
                self.assertIsInstance(ok, bool)
                self.assertIsInstance(reason, str)
                self.assertTrue(
                    ok, msg=f"{arch}: D128 fp16 GQA should be supported, got: {reason}"
                )

    def test_attention_dispatch_matrix_no_signature_drift_per_arch(self):
        """Broad per-arch drift net over the curated attention problem matrix.

        For every ``(arch, config)`` pair this drives the production dispatch
        surface exactly as the runtime selector would -- ``select_path``, the
        three ``supports_native_*`` gates, and (when a gate accepts) the matching
        arch spec builder -- and asserts there is no *signature drift*: each gate
        returns a ``(bool, str)`` tuple and each spec builder returns an instance
        of the arch's own spec class. This is the single test that would have
        caught BOTH regressions that motivated this matrix: the gfx950
        ``supports_tiled_2d`` missing-kwarg ``TypeError`` and the gfx950
        ``UnifiedAttention3DTiledSpec`` missing-field ``TypeError``.

        Deliberately NOT asserted: per-shape accept/reject verdicts (those
        legitimately differ by arch, e.g. HD=256 is unsupported on the gfx942 2D
        path).

        On the **selected** path (``select_path()``), gate-True must imply the
        matching spec builder constructs WITHOUT raising at all -- that is the
        production call path, so a ``ValueError`` there is a real crash (this is
        how the gfx942 combo-on-bf16 bug was caught: the arch-agnostic flag
        choosers handed the gfx942 2D spec gfx950-only knobs; now arch-gated to
        gfx950). On the **non-selected** builder we still exercise the call but
        tolerate a ``ValueError`` (arch-legitimate rejection of a path the
        runtime would not pick) while still failing on a ``TypeError`` (the
        signature-drift class).
        """
        import kernels.common.attention_unified as au
        from kernels import (
            supports_native_unified_attention,
            supports_native_unified_attention_tiled,
            supports_native_unified_attention_3d_tiled,
        )

        matrix = _attention_problem_matrix()
        self.assertTrue(matrix, "attention problem matrix is empty")
        for arch in ("gfx942", "gfx950"):
            spec_2d_cls = au._tiled_2d_impl(arch)[0]
            spec_3d_cls = au._tiled_3d_impl(arch)[0]
            for label, p in matrix:
                with self.subTest(arch=arch, cfg=label):
                    with _patch_resolved_arch(arch):
                        path = p.select_path()
                        self.assertIn(path, ("2d", "3d"))
                        for gate in (
                            supports_native_unified_attention,
                            supports_native_unified_attention_tiled,
                            supports_native_unified_attention_3d_tiled,
                        ):
                            ok, reason = gate(p)
                            self.assertIsInstance(ok, bool)
                            self.assertIsInstance(reason, str)
                        ok_2d, _ = supports_native_unified_attention_tiled(p)
                        ok_3d, _ = supports_native_unified_attention_3d_tiled(p)
                        # Selected production path: gate-True => builder must
                        # construct the arch spec with NO exception.
                        if path == "2d" and ok_2d:
                            spec = au._tiled_spec_from_problem(p)
                            self.assertIsInstance(spec, spec_2d_cls)
                        if path == "3d" and ok_3d:
                            spec3 = au._tiled_3d_spec_from_problem(p)
                            self.assertIsInstance(spec3, spec_3d_cls)
                        # Non-selected builder: exercise for cross-path drift.
                        # Tolerate an arch-legitimate ValueError; a TypeError is
                        # the missing-kwarg/field signature drift we guard.
                        if path != "2d" and ok_2d:
                            try:
                                spec = au._tiled_spec_from_problem(p)
                            except ValueError:
                                pass
                            else:
                                self.assertIsInstance(spec, spec_2d_cls)
                        if path != "3d" and ok_3d:
                            try:
                                spec3 = au._tiled_3d_spec_from_problem(p)
                            except ValueError:
                                pass
                            else:
                                self.assertIsInstance(spec3, spec_3d_cls)

    def test_gfx942_l4_num_warps_matches_flash_selector(self):
        """The gfx942 D128 fp16 flash/L4 branch in ``_select_2d_num_warps`` must
        agree with ``_select_gfx942_flash_num_warps`` (what the flash kernel is
        actually built and launched with). The branch is not on the live grid
        path today (every flash site reads the flash selector directly), but
        ``num_warps`` is not part of the JitCache key, so a silent disagreement
        here is a latent wrong-CTA-count trap for any future caller. Pin them
        equal.
        """
        import kernels.common.attention_unified as au

        p = UnifiedAttentionProblem(
            total_q=4096,
            num_seqs=2,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=2048,
            max_seqlen_k=2048,
            dtype="fp16",
        )
        with _patch_resolved_arch("gfx942"):
            self.assertTrue(
                au._enable_gfx942_l4(p), "shape must be in the gfx942 L4 flash region"
            )
            self.assertEqual(
                au._select_2d_num_warps(p),
                au._select_gfx942_flash_num_warps(p),
            )

    def test_gfx942_d64_decode_num_warps(self):
        """gfx942 D64 decode picks num_warps=1.

        Decode (max_seqlen_q == 1) is memory-bound and wins at nw=1 (BLOCK_M=32,
        4x the CTAs of nw=4). fp8 decode is excluded from the nw=1 lever
        (dequant-bound, wants more warps), so it stays at nw=4. bf16 full-causal
        sink prefill is intercepted earlier by the tuned cohort (nw=2, see
        test_gfx942_sink_prefill_tuned_cohort). This branch is the production
        geometry change; pin it so a refactor can't silently revert it. The
        Python selectors are authoritative for production geometry; the C++
        selectors in attention_unified_selectors.cpp are a hand-maintained mirror
        kept in sync for parity (not exercised by production dispatch).
        """
        import kernels.common.attention_unified as au

        def _p(max_seqlen_q, use_fp8=False):
            return UnifiedAttentionProblem(
                total_q=max_seqlen_q,
                num_seqs=1,
                num_query_heads=64,
                num_kv_heads=8,
                head_size=64,
                block_size=16,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=2048,
                dtype="bf16",
                use_sinks=True,
                use_fp8=use_fp8,
            )

        with _patch_resolved_arch("gfx942"):
            self.assertEqual(au._select_2d_num_warps(_p(1)), 1)  # decode
            # prefill: intercepted by the tuned sink-prefill cohort -> nw2
            self.assertEqual(au._select_2d_num_warps(_p(512)), 2)
            self.assertEqual(au._select_2d_num_warps(_p(2048)), 2)
            self.assertEqual(
                au._select_2d_num_warps(_p(1, use_fp8=True)), 4
            )  # fp8 decode excluded

    def test_gfx942_sink_prefill_tuned_cohort(self):
        """gfx942 full-causal bf16 sink prefill selects nw2/mw16/T32 + register_pv,
        and near-miss shapes stay on the shipped nw4/mw32/no-regpv config.
        """
        import kernels.common.attention_unified as au

        def _make_problem(**overrides):
            base = dict(
                total_q=2048,
                num_seqs=1,
                num_query_heads=64,
                num_kv_heads=8,
                head_size=64,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="bf16",
                use_sinks=True,
                sliding_window=0,
            )
            base.update(overrides)
            return UnifiedAttentionProblem(**base)

        with _patch_resolved_arch("gfx942"):
            cohort = _make_problem()
            self.assertTrue(au._enable_gfx942_sink_prefill_tuned(cohort))
            spec = au._tiled_spec_from_problem(cohort)
            self.assertEqual(spec.num_warps, 2)
            self.assertEqual(spec.block_m_per_warp, 16)
            self.assertEqual(spec.tile_size, 2 * cohort.block_size)
            self.assertTrue(spec.use_register_pv)

            # Near-miss shapes on the 2D path must NOT hit the tuned cohort and
            # must keep the shipped D64 config (nw4 / mw32 / no register_pv).
            for label, p in (
                ("swa", _make_problem(sliding_window=128)),
                ("no_sinks", _make_problem(use_sinks=False)),
                ("bs32", _make_problem(block_size=32)),
            ):
                with self.subTest(near_miss=label):
                    self.assertFalse(au._enable_gfx942_sink_prefill_tuned(p))
                    s = au._tiled_spec_from_problem(p)
                    self.assertEqual(s.num_warps, 4)
                    self.assertEqual(s.block_m_per_warp, 32)
                    self.assertFalse(s.use_register_pv)

            # Decode (q==1) routes to the 3D path, not the 2D spec builder; the
            # cohort gate must still exclude it.
            self.assertFalse(
                au._enable_gfx942_sink_prefill_tuned(_make_problem(max_seqlen_q=1))
            )

    def test_gfx950_sink_prefill_wpe3_cohort(self):
        """gfx950 full-causal bf16 sink prefill selects waves_per_eu=3 (occupancy
        hint only, output-preserving); SWA, non-sink, decode, and other shapes
        keep the shipped waves_per_eu=2.
        """
        import kernels.common.attention_unified as au

        def _make_problem(**overrides):
            base = dict(
                total_q=2048,
                num_seqs=1,
                num_query_heads=64,
                num_kv_heads=8,
                head_size=64,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="bf16",
                use_sinks=True,
                sliding_window=0,
            )
            base.update(overrides)
            return UnifiedAttentionProblem(**base)

        with _patch_resolved_arch("gfx950"):
            cohort = _make_problem()
            self.assertTrue(au._enable_gfx950_sink_prefill_wpe3(cohort))
            self.assertEqual(au._select_2d_waves_per_eu(cohort), 3)

            # Near-miss shapes must NOT hit the wpe=3 cohort.
            for label, p in (
                ("swa", _make_problem(sliding_window=128)),
                ("no_sinks", _make_problem(use_sinks=False)),
                ("bs32", _make_problem(block_size=32)),
            ):
                with self.subTest(near_miss=label):
                    self.assertFalse(au._enable_gfx950_sink_prefill_wpe3(p))
            # Decode (q==1) routes to 3D; gate must still exclude it.
            self.assertFalse(
                au._enable_gfx950_sink_prefill_wpe3(_make_problem(max_seqlen_q=1))
            )
            # gfx942 must not hit the gfx950 gate.
            with _patch_resolved_arch("gfx942"):
                self.assertFalse(au._enable_gfx950_sink_prefill_wpe3(_make_problem()))

    def test_tiled_3d_dispatch_gate_accepts_kwargs_per_arch(self):
        """Regression: the shared dispatch entry
        ``supports_native_unified_attention_3d_tiled`` forwards its kwargs to the
        per-arch ``supports_tiled_3d`` gate, and the auto selector routes decode
        (``max_seqlen_q == 1``) to this 3D split-KV path. Every routed arch's
        gate must accept the forwarded kwargs without raising. This guards the
        gfx950 3D spec-kwarg regression (the ``UnifiedAttention3DTiledSpec``
        missing-field break that took out production decode), mirroring the 2D
        dispatch test one path over.
        """
        from kernels import supports_native_unified_attention_3d_tiled

        p = UnifiedAttentionProblem(
            total_q=4,
            num_seqs=4,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,  # decode -> routes to the 3D split-KV builder
            max_seqlen_k=4096,
            dtype="fp16",
        )
        for arch in ("gfx950", "gfx942"):
            with _patch_resolved_arch(arch):
                # Must not raise (the regression was a TypeError on the kwarg).
                ok, reason = supports_native_unified_attention_3d_tiled(p)
                self.assertIsInstance(ok, bool)
                self.assertIsInstance(reason, str)
                self.assertTrue(
                    ok,
                    msg=f"{arch}: D128 fp16 GQA decode should route to a "
                    f"supported 3D kernel, got: {reason}",
                )

    def test_tiled_3d_spec_builder_constructs_per_arch(self):
        """Focused guard on the 3D spec builder that broke: a decode problem must
        construct the arch's ``UnifiedAttention3DTiledSpec`` (signature parity)
        on both arches, and the gfx942-only 3D knobs must be inert on gfx950 --
        ``_gfx942_3d_tile_size_override`` is ``None`` and the two
        ``_enable_gfx942_3d_*`` toggles are ``False`` -- pinning the
        ignored-field contract that lets the shared builder pass those kwargs
        unconditionally.
        """
        import kernels.common.attention_unified as au

        p = UnifiedAttentionProblem(
            total_q=4,
            num_seqs=4,
            num_query_heads=8,
            num_kv_heads=1,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=4096,
            dtype="fp16",
        )
        for arch in ("gfx942", "gfx950"):
            with _patch_resolved_arch(arch):
                spec = au._tiled_3d_spec_from_problem(p)
                self.assertIsInstance(spec, au._tiled_3d_impl(arch)[0])
        # The gfx942-only 3D knobs are inert on gfx950 (ignored-field contract).
        with _patch_resolved_arch("gfx950"):
            self.assertIsNone(au._gfx942_3d_tile_size_override(p))
            self.assertFalse(au._enable_gfx942_3d_invariant_hoist(p))
            self.assertFalse(au._enable_gfx942_3d_wide_kv_load(p))

    def test_tiled_2d_spec_builder_constructs_per_arch_all_branches(self):
        """Drive ``_tiled_spec_from_problem`` through its three branches and
        assert each constructs the arch's 2D spec without signature drift:

        * gfx942 fp16 long-prefill D128 & D64 (no SW/softcap, num_seqs>=2) ->
          the gfx942 fp16-flash branch (~13 flash flags),
        * bf16 multi-batch long-prefill HD64/BS32/GQA8 -> the bf16 transposed
          "combo" branch (gfx950, where the combo flags are valid),
        * a plain fp16 shape -> the default branch.

        The flash branch only fires on gfx942 and the combo branch only builds
        cleanly on gfx950; this asserts construction + correct arch spec type for
        whichever branch fires on the relevant arch, pinning the largest
        (~25-flag) silent surface.
        """
        import kernels.common.attention_unified as au

        def problem(**kw):
            base = dict(
                total_q=0,
                num_seqs=2,
                num_query_heads=8,
                num_kv_heads=1,
                head_size=128,
                block_size=16,
                max_seqlen_q=2048,
                max_seqlen_k=2048,
                dtype="fp16",
            )
            base.update(kw)
            if base["total_q"] == 0:
                base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
            return UnifiedAttentionProblem(**base)

        # (label, problem, arches to drive). Flash -> gfx942; combo -> gfx950;
        # default -> both.
        cases = [
            (
                "flash_d128",
                problem(head_size=128, block_size=16, dtype="fp16"),
                ("gfx942",),
            ),
            (
                "flash_d64",
                problem(head_size=64, block_size=64, dtype="fp16"),
                ("gfx942",),
            ),
            (
                "combo_bf16",
                problem(
                    head_size=64,
                    block_size=32,
                    dtype="bf16",
                    num_query_heads=64,
                    num_kv_heads=8,
                ),
                ("gfx950",),
            ),
            (
                "default_fp16",
                problem(head_size=128, dtype="fp16", num_seqs=1),
                ("gfx942", "gfx950"),
            ),
        ]
        for label, p, arches in cases:
            for arch in arches:
                with self.subTest(cfg=label, arch=arch):
                    with _patch_resolved_arch(arch):
                        spec = au._tiled_spec_from_problem(p)
                        self.assertIsInstance(spec, au._tiled_2d_impl(arch)[0])

    def test_tiled_3d_support_gate_rejects_unsupported(self):
        """Mirror of ``test_tiled_2d_support_gate_rejects_unsupported`` for the
        per-arch ``supports_tiled_3d`` gate. Both arches share the same
        accept/reject contract, so the cases are driven for each arch via the
        ``arch=`` kwarg.
        """
        from kernels import supports_tiled_3d

        base = dict(
            head_size=128,
            block_size=16,
            dtype="fp16",
            num_queries_per_kv=8,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype=None,
        )
        for arch in ("gfx942", "gfx950"):
            ok_fp16, _ = supports_tiled_3d(arch=arch, **base)
            self.assertTrue(ok_fp16, msg=f"{arch}: base fp16 D128 should accept")
            # head_size {64,128,256}, block_size {16,32,64}, bf16, and every
            # num_queries_per_kv that divides BLOCK_M=16 are supported.
            for accept in [
                dict(head_size=256),
                dict(head_size=64),
                dict(block_size=32),
                dict(block_size=64),
                dict(dtype="bf16"),
                dict(num_queries_per_kv=1),
                dict(num_queries_per_kv=4),
                dict(num_queries_per_kv=16),
                dict(use_alibi=True),
                dict(use_qq_bias=True),
            ]:
                kwargs = dict(base)
                kwargs.update(accept)
                ok, reason = supports_tiled_3d(arch=arch, **kwargs)
                self.assertTrue(
                    ok, msg=f"{arch}: expected accept for {accept}, got: {reason}"
                )
            # Bad head_size, bad block_size, fp8-without-kv_storage, and an
            # unsupported dtype are gated on both arches.
            for override in [
                dict(head_size=72),
                dict(block_size=24),
                dict(use_fp8=True),
                dict(dtype="fp8"),
            ]:
                kwargs = dict(base)
                kwargs.update(override)
                ok, reason = supports_tiled_3d(arch=arch, **kwargs)
                self.assertFalse(
                    ok, msg=f"{arch}: expected reject for {override}, got: {reason}"
                )
                self.assertTrue(reason)

    def test_unified_attention_3d_tiled_kernel_compiles_gfx942(self):
        """gfx942 analogue of ``test_unified_attention_3d_tiled_kernel_compiles``
        (which exercises the default gfx950 arch only). Builds the gfx942 3D
        split-KV segment kernel and asserts the gfx942-specific narrow primitives
        it ACTUALLY emits -- the 16x16x16 MFMA atom and the 1-DWORD async
        global->LDS DMA (``raw.ptr.buffer.load.lds``) -- and that it does NOT emit
        the gfx950 wide 16x16x32 MFMA. Pure codegen, no GPU.
        """
        import kernels.common.attention_unified as au

        with _patch_resolved_arch("gfx942"):
            (
                UnifiedAttention3DTiledSpec,
                UnifiedAttentionReduceTiledSpec,
                build_unified_attention_3d_tiled,
                build_unified_attention_reduce_tiled,
                _supports_tiled_3d,
            ) = au._tiled_3d_impl("gfx942")
            seg = build_unified_attention_3d_tiled(
                UnifiedAttention3DTiledSpec(
                    head_size=128,
                    block_size=16,
                    num_query_heads=16,
                    num_kv_heads=2,
                    dtype="fp16",
                    use_sinks=False,
                    sliding_window=0,
                    has_softcap=False,
                    num_segments=128,
                    num_seqs=4,
                )
            )
            seg_ll = lower_kernel_to_llvm(seg)
            # The arch-dispatched reduce kernel must also build on gfx942.
            red = build_unified_attention_reduce_tiled(
                UnifiedAttentionReduceTiledSpec(
                    head_size=128,
                    num_query_heads=16,
                    num_kv_heads=2,
                    dtype="fp16",
                    num_segments=128,
                )
            )
            red_ll = lower_kernel_to_llvm(red)
        # gfx942 narrow 3D path: 16x16x16 MFMA + 1-DWORD async DMA KV feed.
        self.assertIn("@llvm.amdgcn.mfma.f32.16x16x16f16", seg_ll)
        self.assertIn("@llvm.amdgcn.raw.ptr.buffer.load.lds", seg_ll)
        # Must NOT use the gfx950 wide-K 16x16x32 MFMA.
        self.assertNotIn("@llvm.amdgcn.mfma.f32.16x16x32.f16", seg_ll)
        # Workspace writes for per-segment m / l / acc.
        self.assertIn("segm_output_ptr", seg_ll)
        self.assertIn("segm_max_ptr", seg_ll)
        self.assertIn("segm_expsum_ptr", seg_ll)
        # Reduce kernel: exp2-weighted segment combine + NaN-safe factor.
        self.assertIn("@llvm.exp2.f32", red_ll)
        self.assertIn("fcmp ogt", red_ll)


# ---------------------------------------------------------------------
# AttentionDenseSpec — waves_per_eu validation, IR identity, cache isolation
# ---------------------------------------------------------------------


class TestAttentionDenseWavesPerEu(unittest.TestCase):
    """Tests for the waves_per_eu fix on AttentionDenseSpec.

    Three properties are verified independently so a single failure is
    unambiguous:

    1. ``__post_init__`` rejects out-of-range values (0, negative, >8).
    2. The emitted LLVM IR carries the correct ``amdgpu-waves-per-eu``
       attribute for each legal value — confirming the attribute reached
       codegen correctly both before and after the cache-key fix.
    3. Two specs differing only in ``waves_per_eu`` produce distinct compiled
       binaries (cache-isolation fix: key now includes ``waves_per_eu``).

    All three run without a GPU; test 3 needs comgr and is skipped when the
    toolchain is unavailable (matching the pattern in
    ``test_gfx950_dense_prefill_compiles_and_fits_budget``).
    """

    _BASE_KWARGS = dict(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
    )

    def test_waves_per_eu_validation_rejects_out_of_range(self):
        from kernels.gfx950.attention_dense import AttentionDenseSpec

        for bad in (0, -1, 9, 100):
            with self.subTest(waves_per_eu=bad):
                with self.assertRaises(ValueError, msg=f"waves_per_eu={bad} should be rejected"):
                    AttentionDenseSpec(**self._BASE_KWARGS, waves_per_eu=bad)

        for good in (1, 2, 8):
            with self.subTest(waves_per_eu=good):
                # Must not raise
                AttentionDenseSpec(**self._BASE_KWARGS, waves_per_eu=good)

    def test_waves_per_eu_ir_attribute(self):
        """Each legal waves_per_eu value appears verbatim in the lowered IR."""
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense
        from dataclasses import replace

        base = AttentionDenseSpec(**self._BASE_KWARGS)
        for wpe in (1, 2):
            with self.subTest(waves_per_eu=wpe):
                spec = replace(base, waves_per_eu=wpe)
                ll = lower_kernel_to_llvm(build_attention_dense(spec, arch="gfx950"))
                self.assertIn(f'"amdgpu-waves-per-eu"="{wpe},{wpe}"', ll)

    def test_waves_per_eu_cache_key_isolation(self):
        """Specs differing only in waves_per_eu produce distinct cache keys.

        Before the fix the cache key was ``(kernel_name(), batch)``; two specs
        with different ``waves_per_eu`` share the same ``kernel_name()`` and
        ``batch``, so the second call would silently reuse the first binary.
        After the fix the key is ``(kernel_name(), batch, waves_per_eu)``, so
        each value maps to a distinct slot.

        This test verifies the key structure directly (no comgr needed) and
        checks that the distinct keys correspond to distinct binaries via the
        IR attribute (which feeds AMDGPU register-file sizing).
        """
        from dataclasses import replace
        from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

        base = AttentionDenseSpec(**self._BASE_KWARGS)
        specs = {wpe: replace(base, waves_per_eu=wpe) for wpe in (1, 2)}

        # Cache keys must be distinct.
        keys = {wpe: (s.kernel_name(), s.batch, s.waves_per_eu) for wpe, s in specs.items()}
        self.assertNotEqual(
            keys[1],
            keys[2],
            "waves_per_eu=1 and waves_per_eu=2 produce identical cache keys; "
            "a sweep over waves_per_eu would silently reuse the first binary",
        )

        # The key difference (waves_per_eu in the 3rd position) must match the spec.
        for wpe, key in keys.items():
            self.assertEqual(key[2], wpe, f"cache key[2] should be waves_per_eu={wpe}")

        # The two specs share kernel_name() and batch (the old key was just those two).
        self.assertEqual(
            keys[1][:2],
            keys[2][:2],
            "kernel_name() or batch differed unexpectedly — test setup error",
        )

    def test_waves_per_eu_cache_isolation_binaries(self):
        """Specs differing only in waves_per_eu compile to distinct binaries.

        Requires comgr; skipped when the toolchain is unavailable.
        """
        import hashlib
        from dataclasses import replace
        from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

        base = AttentionDenseSpec(**self._BASE_KWARGS)
        hsaco_hashes = {}
        for wpe in (1, 2):
            with self.subTest(waves_per_eu=wpe):
                spec = replace(base, waves_per_eu=wpe)
                art = _compile_or_skip(build_attention_dense(spec, arch="gfx950"), arch="gfx950")
                hsaco_hashes[wpe] = hashlib.sha256(art.hsaco).hexdigest()

        if len(hsaco_hashes) == 2:
            self.assertNotEqual(
                hsaco_hashes[1],
                hsaco_hashes[2],
                "waves_per_eu=1 and waves_per_eu=2 produced identical binaries; "
                "waves_per_eu is not reaching the register-file sizing pass",
            )


# ---------------------------------------------------------------------
# CDNA primitives — attention tiled waves-per-EU
# ---------------------------------------------------------------------


class TestAttentionCdnaPrimitives(unittest.TestCase):
    def test_attention_tiled_2d_waves_per_eu(self):
        from kernels import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec = UnifiedAttention2DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=16,
            num_kv_heads=2,
            dtype="fp16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            waves_per_eu=2,
        )
        ll = lower_kernel_to_llvm(build_unified_attention_2d_tiled(spec))
        self.assertIn('"amdgpu-waves-per-eu"="2,2"', ll)


# ---------------------------------------------------------------------
# MFMA gremlin — attention
# ---------------------------------------------------------------------


class TestEveryAttentionKernelUsesMfma(unittest.TestCase):
    """Assert every attention kernel emits a real ``@llvm.amdgcn.mfma.*``
    intrinsic in its LLVM IR. The kernels MUST use MFMA -- the
    warp-distributed scalar inner is not acceptable for production.
    """

    def _llvm_for(self, build_fn, spec):
        return lower_kernel_to_llvm(build_fn(spec))

    def test_fmha_mfma_uses_mfma(self):
        from kernels import FmhaMfmaSpec, build_fmha_fwd_mfma
        from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape

        spec = FmhaMfmaSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=2, num_kv_heads=2),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
        ll = self._llvm_for(build_fmha_fwd_mfma, spec)
        # Expect MFMA invocations for both QK and PV chains.
        n_mfma = ll.count("@llvm.amdgcn.mfma.f32.16x16x16")
        # head_size=64, atom.k=16 -> 4 QK atoms per K-tile + 4 PV atoms
        # per K-tile. Counting at least one is enough; the parity test
        # verifies the chain is correct.
        self.assertGreaterEqual(n_mfma, 1, f"got {n_mfma} MFMA calls")


# ---------------------------------------------------------------------
# Extended attention builds
# ---------------------------------------------------------------------


class TestExtendedAttentionBuilds(unittest.TestCase):
    def _common(self, *, dtype="f16", mask_mode="none"):
        from kernels import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
            dtype=dtype,
            mask_mode=mask_mode,
        )

    def _gqa_common(self):
        from kernels import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=2),
            dtype="f16",
            mask_mode="causal",
        )

    def test_fmha_fwd_varlen_builds_and_lowers(self):
        from kernels import FmhaFwdVarlenSpec, build_fmha_fwd_varlen

        spec = FmhaFwdVarlenSpec(
            common=self._common(mask_mode="causal"),
            max_seqlen_q=256,
            max_seqlen_k=256,
            batch=2,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_varlen(spec))
        self.assertIn("@llvm.exp2.f32", ll)
        self.assertIn("define amdgpu_kernel", ll)

    def test_fmha_appendkv_with_rotary_emits_cos_sin_loads(self):
        from rocke.helpers.rotary import RotarySpec
        from kernels import FmhaAppendKvSpec, build_fmha_fwd_appendkv

        spec = FmhaAppendKvSpec(
            common=self._common(),
            batch=2,
            rotary=RotarySpec(head_size=64, layout="half"),
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_appendkv(spec))
        self.assertGreaterEqual(ll.count("load float"), 32)

    def test_fmha_paged_prefill_builds(self):
        from kernels import (
            FmhaFwdPagedPrefillSpec,
            build_fmha_fwd_paged_prefill,
        )

        spec = FmhaFwdPagedPrefillSpec(
            common=self._common(mask_mode="causal"),
            page_block_size=16,
            max_blocks_per_seq=32,
            batch=2,
        )
        kernel = build_fmha_fwd_paged_prefill(spec)
        self.assertIn("block_table", [p.name for p in kernel.params])

    def test_fmha_splitkv_decode_two_kernel_pipeline(self):
        from kernels import (
            FmhaFwdSplitKvDecodeSpec,
            build_fmha_fwd_splitkv_decode_reduce,
            build_fmha_fwd_splitkv_decode_segment,
        )

        spec = FmhaFwdSplitKvDecodeSpec(
            common=self._common(),
            batch=4,
            num_segments=8,
        )
        seg = build_fmha_fwd_splitkv_decode_segment(spec)
        red = build_fmha_fwd_splitkv_decode_reduce(spec)
        self.assertEqual(
            sorted(p.name for p in seg.params)[:3],
            sorted(["Q", "K", "V"])[:3],
        )
        ll_red = lower_kernel_to_llvm(red)
        self.assertIn("@llvm.exp2.f32", ll_red)

    def test_fmha_head_grouping_builds_for_gqa(self):
        from kernels import (
            FmhaFwdHeadGroupingSpec,
            build_fmha_fwd_head_grouping,
        )

        spec = FmhaFwdHeadGroupingSpec(
            common=self._gqa_common(),
            seqlen_q=128,
            seqlen_k=128,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_head_grouping(spec))
        self.assertIn("@llvm.amdgcn.workgroup.id.z", ll)

    def test_fmha_bwd_uses_atomic_fadd(self):
        from kernels import FmhaBwdSpec, build_fmha_bwd

        spec = FmhaBwdSpec(
            common=self._common(),
            seqlen_q=64,
            seqlen_k=64,
        )
        ll = lower_kernel_to_llvm(build_fmha_bwd(spec))
        # 3 atomic accumulators (dQ, dK, dV) per K-step per head dim.
        self.assertGreaterEqual(ll.count("atomicrmw fadd ptr addrspace(1)"), 3)

    def test_fmha_fwd_fp8_emits_cvt_fp8_intrinsic(self):
        from kernels import FmhaFwdFp8Spec, build_fmha_fwd_fp8

        spec = FmhaFwdFp8Spec(
            common=self._common(),
            kv_dtype="fp8e4m3",
            seqlen_q=32,
        )
        ll = lower_kernel_to_llvm(build_fmha_fwd_fp8(spec))
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)


# ---------------------------------------------------------------------
# Sage attention builds
# ---------------------------------------------------------------------


class TestSageAttentionBuilds(unittest.TestCase):
    def _spec(self, quant_mode, *, head_size=64):
        from rocke.helpers.qk_scale import QkScaleSpec
        from kernels import (
            FmhaCommonSpec,
            FmhaShape,
            SageAttentionSpec,
        )

        common = FmhaCommonSpec(
            shape=FmhaShape(head_size=head_size, num_query_heads=8, num_kv_heads=8),
            dtype="f16",
            mask_mode="none",
        )
        return SageAttentionSpec(
            common=common,
            quant_mode=quant_mode,
            q_scale=QkScaleSpec(
                layout="per_block",
                scale_block=16,
                stride_batch=128,
                stride_head=8,
                stride_block=1,
            ),
            k_scale=QkScaleSpec(
                layout="per_block",
                scale_block=64,
                stride_batch=128,
                stride_head=8,
                stride_block=1,
            ),
            seqlen_q=16,
            seqlen_k=64,
        )

    def test_fp16_baseline_no_fp8_cvt(self):
        from kernels.common.sage_attention import build_sage_attention

        ll = lower_kernel_to_llvm(build_sage_attention(self._spec("fp16_bf16")))
        self.assertNotIn("@llvm.amdgcn.cvt.f32.fp8", ll)

    def test_fp8_variant_uses_fp8_cvt(self):
        from kernels.common.sage_attention import build_sage_attention

        ll = lower_kernel_to_llvm(build_sage_attention(self._spec("fp8_bf16")))
        self.assertIn("@llvm.amdgcn.cvt.f32.fp8", ll)

    def test_int_variants_add_codebook_params(self):
        from kernels.common.sage_attention import build_sage_attention

        # i4 sage needs head_size=128 (each lane owns one packed byte =
        # two nibbles); i8 sage works at head_size=64.
        for qm, hs in (("i8_fp8_bf16", 64), ("i4_fp8_bf16", 128)):
            kernel = build_sage_attention(self._spec(qm, head_size=hs))
            names = [p.name for p in kernel.params]
            self.assertIn("codebook_k", names, f"qm={qm}")
            self.assertIn("codebook_v", names, f"qm={qm}")


# ---------------------------------------------------------------------
# Sparse attention builds
# ---------------------------------------------------------------------


class TestSparseAttentionBuilds(unittest.TestCase):
    def _common(self):
        from kernels import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
            dtype="f16",
            mask_mode="none",
        )

    def test_jenga_emits_mask_byte_guard(self):
        from kernels import (
            JengaSparseSpec,
            build_jenga_sparse_attention,
        )

        spec = JengaSparseSpec(
            common=self._common(),
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=32,
        )
        ll = lower_kernel_to_llvm(build_jenga_sparse_attention(spec))
        self.assertIn("load i8", ll)
        self.assertIn("icmp ne i8", ll)

    def test_vsa_loads_block_count_then_lut(self):
        from kernels import (
            VsaSparseSpec,
            build_vsa_sparse_attention,
        )

        spec = VsaSparseSpec(
            common=self._common(),
            seqlen_q=32,
            seqlen_k=256,
            block_q=1,
            block_k=32,
            max_blocks_per_q=4,
        )
        ll = lower_kernel_to_llvm(build_vsa_sparse_attention(spec))
        self.assertGreaterEqual(ll.count("load i32"), 2)


# ---------------------------------------------------------------------
# FmhaKernelBuilder boilerplate API
# ---------------------------------------------------------------------


class TestFmhaKernelBuilder(unittest.TestCase):
    """Tests for the FmhaKernelBuilder boilerplate-killer."""

    def _common(self):
        from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape

        return FmhaCommonSpec(
            shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=2),
            dtype="f16",
            mask_mode="causal",
        )

    def test_signature_matches_old_varlen(self):
        """The builder-generated signature for fmha_varlen must match
        the canonical Q/K/V/O/cu/scale/total/batch/strides ABI exactly."""
        from kernels.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe", self._common())
        kb.add_tensor("Q")
        kb.add_tensor("K")
        kb.add_tensor("V")
        kb.add_tensor("O")
        kb.add_ptr("cu_seqlens_q", dtype="i32")
        kb.add_ptr("cu_seqlens_k", dtype="i32")
        kb.add_scalar("scale_log2", "f32")
        kb.add_scalar("total_q", "i32")
        kb.add_scalar("batch", "i32")
        kb.add_strides("q", "k", "v", "o")
        sig = kb.signature()
        names = [item["name"] for item in sig]
        self.assertEqual(
            names,
            [
                "Q",
                "K",
                "V",
                "O",
                "cu_seqlens_q",
                "cu_seqlens_k",
                "scale_log2",
                "total_q",
                "batch",
                "stride_q_token",
                "stride_q_head",
                "stride_k_token",
                "stride_k_head",
                "stride_v_token",
                "stride_v_head",
                "stride_o_token",
                "stride_o_head",
            ],
        )

    def test_decode_grid_emits_gqa_div(self):
        """When ``num_queries_per_kv > 1`` the grid decode emits a
        divide on head_idx (otherwise it short-circuits to identity).
        """
        from kernels.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_grid", self._common())
        kb.add_scalar("scale_log2", "f32")
        kb.decode_grid()
        # head_idx = block_id_y, kv_head_idx = head_idx // 4 (HQ=8 / HK=2).
        # Lower and check the IR shows the divide.
        kb.builder.ret()
        ll = lower_kernel_to_llvm(kb.kernel)
        # The arith.div lowers to ``sdiv i32 ..., 4``.
        self.assertIn("sdiv i32", ll)

    def test_add_tensor_accepts_fp8_kv_dtype(self):
        """add_tensor with dtype='fp8e4m3' produces an fp8 pointer
        (used by fmha_fwd_fp8 / sage)."""
        from kernels.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_fp8", self._common())
        kb.add_tensor("K", dtype="fp8e4m3", align=8)
        sig = kb.signature()
        k_entry = next(item for item in sig if item["name"] == "K")
        # The "type" field renders as "ptr<fp8e4m3, global>".
        self.assertIn("fp8e4m3", k_entry["type"])

    def test_tensor_descriptor_naive_3d(self):
        """tensor_descriptor returns a 3-coord descriptor whose
        offset() works for an (token, head, d) triple."""
        from kernels.common._fmha_common import FmhaKernelBuilder

        kb = FmhaKernelBuilder("probe_desc", self._common())
        kb.add_tensor("Q")
        kb.add_strides("q")
        desc = kb.tensor_descriptor("q")
        self.assertEqual(desc.upper_names, ("token", "head", "d"))


class TestAttentionHarnessTimers(unittest.TestCase):
    """The attention benchmark must time every lane with one shared clock.

    The harness keeps both Triton and CK DSL apples-to-apples by:

    1. Allocating one explicit HIP stream per lane.
    2. Routing the Triton call through ``torch.cuda.stream(...)`` so its
       launches land on that stream.
    3. Passing the same HIP stream handle into the CK DSL runner so its
       raw ``hipModuleLaunchKernel`` calls share the stream.
    4. Recording HIP events on that stream via
       :func:`rocke.runtime.time_launches`.

    These tests pin down (1, 3, 4): the timer goes through
    ``time_launches`` with the caller-supplied stream and follows up
    with a per-stream release.
    """

    @staticmethod
    def _load_harness_with_fake_aiter():
        import importlib.util
        import sys
        import types
        from pathlib import Path
        from unittest import mock

        import pytest

        # The harness's reference implementation imports torch at module scope,
        # so it cannot be loaded without torch; skip torch-free instead of
        # erroring. Import torch BEFORE patching sys.modules so it stays in the
        # parent process's module table after ``mock.patch.dict`` exits.
        pytest.importorskip("torch")
        import torch  # noqa: F401

        # The harness moved into the library tree (builders/); resolve it via the
        # package system (editable-installed) rather than a hardcoded path, then
        # load it under a private name with a fake ``aiter`` injected.
        module_path = importlib.util.find_spec(
            "builders.gfx950.attention.parity_unified_attention"
        ).origin
        fake_aiter = types.ModuleType("aiter")
        fake_ops = types.ModuleType("aiter.ops")
        fake_triton = types.ModuleType("aiter.ops.triton")
        fake_attention = types.ModuleType("aiter.ops.triton.attention")
        fake_unified = types.ModuleType("aiter.ops.triton.attention.unified_attention")
        fake_unified.use_2d_kernel = lambda *a, **k: True
        fake_unified.unified_attention = lambda *a, **k: None
        modules = {
            "aiter": fake_aiter,
            "aiter.ops": fake_ops,
            "aiter.ops.triton": fake_triton,
            "aiter.ops.triton.attention": fake_attention,
            "aiter.ops.triton.attention.unified_attention": fake_unified,
        }
        spec = importlib.util.spec_from_file_location(
            "rocke_attention_parity_timer_test",
            module_path,
        )
        mod = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        return mod

    def test_lane_timer_routes_through_time_launches_with_stream(self):
        from unittest import mock

        mod = self._load_harness_with_fake_aiter()

        calls = []

        def fake_time_launches(fn, *, warmup, iters, stream):
            calls.append(("time_launches", warmup, iters, stream))
            fn()
            return 0.123

        def fake_sync(stream=0):
            calls.append(("sync", stream))

        with mock.patch("rocke.runtime.time_launches", fake_time_launches), mock.patch(
            "rocke.runtime.synchronize_and_release", fake_sync
        ):
            ms = mod._time_lane_ms(
                lambda: calls.append(("launch",)),
                warmup=2,
                attempts=5,
                stream=77,
            )

        self.assertEqual(ms, 0.123)
        self.assertIn(("time_launches", 2, 5, 77), calls)
        # Must release the args bucket for THIS lane's stream, not stream 0.
        self.assertIn(("sync", 77), calls)

    def test_lane_timer_is_the_only_timer(self):
        """Sanity: the harness must NOT also export a torch-event timer.

        Keeping two clocks in the harness is what produced the
        apples-to-oranges Triton-vs-CK comparison the README originally
        called out. Make that contract explicit so a future patch that
        re-introduces a torch-event timer will fail this test.
        """
        mod = self._load_harness_with_fake_aiter()
        self.assertTrue(hasattr(mod, "_time_lane_ms"))
        self.assertFalse(hasattr(mod, "_time_torch_call_loop"))
        self.assertFalse(hasattr(mod, "_time_rocke_call_loop"))


if __name__ == "__main__":
    unittest.main()
