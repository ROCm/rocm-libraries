#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
from pathlib import Path

from rocke.core.ir_golden import (
    GOLDEN_FLAVORS,  # noqa: F401 -- re-exported for rocke_installed_golden_test
    golden_document,
    golden_main,
    lower_golden_case,
    run_golden_cases,
    write_golden_shards,
)
from rocke.core.ir_golden import check_golden as _check_golden


def safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_") + ".ll"


def current_flavor() -> str:
    """Return ROCKE_LLVM_FLAVOR when explicitly set, otherwise use loaded
    COMGR compiler evidence. Only the plain dump run (no ``--check`` /
    ``--write``) uses this selection; the gate checks every flavor.
    """
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


# The (record, llvm) lowering every corpus shares, exposed as the corpus
# protocol's lower_case() that platform/tools/check_ir_validity.py and
# platform/tools/_corpus.py read (not a compatibility alias); see
# rocke.core.ir_golden.lower_golden_case for why the lowerer and flavor are
# pinned.
lower_case = lower_golden_case


def gemm_spec(
    name,
    arch,
    tile_m,
    tile_n,
    tile_k,
    warp_m,
    warp_n,
    wtm,
    wtn,
    wtk,
    pipeline,
    epilogue,
    wave_size,
    cshuffle_no_alias=False,
    dtype="fp16",
    **trait,
):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name=name,
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
        ),
        trait=TraitSpec(
            pipeline=pipeline,
            scheduler="intrawave",
            epilogue=epilogue,
            cshuffle_no_alias=cshuffle_no_alias,
            **{"pad_m": True, "pad_n": True, "pad_k": True, **trait},
        ),
        data=(
            DataSpec("fp64", "fp64", "fp64", "fp64")
            if dtype == "fp64"
            else DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16")
        ),
        wave_size=wave_size,
    )


def build_gemm(name, arch, *args, **kwargs):
    def _build():
        from rocke.instances.common.gemm_universal import build_universal_gemm

        return build_universal_gemm(gemm_spec(name, arch, *args, **kwargs), arch=arch)

    return _build


def build_moe_sort(phase, tokens, topk, experts, block, arch):
    def _build():
        from rocke.instances.common.moe_sorting import (
            MoeSortingSpec,
            build_moe_sort_histogram,
            build_moe_sort_scan,
            build_moe_sort_scatter,
        )

        spec = MoeSortingSpec(
            tokens=tokens,
            topk=topk,
            experts=experts,
            block_size=block,
            name=f"irhash_moe_sort_{phase}",
        )
        return {
            "histogram": build_moe_sort_histogram,
            "scan": build_moe_sort_scan,
            "scatter": build_moe_sort_scatter,
        }[phase](spec, arch=arch)

    return _build


def build_fused_moe(phase, tokens, experts, topk, hidden, intermediate, dtype="f16"):
    def _build():
        from rocke.instances.common.fused_moe import (
            FusedMoeSpec,
            build_moe_gather,
            build_moe_silu_mul,
            build_moe_topk_weighted_reduce,
        )

        spec = FusedMoeSpec(
            tokens=tokens,
            experts=experts,
            topk=topk,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
            block_size=128,
            vec=4,
            name=f"irhash_fused_moe_{phase}",
        )
        return {
            "gather": build_moe_gather,
            "silu": build_moe_silu_mul,
            "reduce": build_moe_topk_weighted_reduce,
        }[phase](spec)

    return _build


def build_attention_2d(
    name,
    arch,
    *,
    head_size,
    block_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    sliding_window=0,
    has_softcap=False,
    use_alibi=False,
    use_sinks=False,
    **kw,
):
    def _build():
        from kernels.common.attention_unified import _tiled_2d_impl

        SpecCls, _build_fn, _ = _tiled_2d_impl(arch)
        spec = SpecCls(
            head_size=head_size,
            block_size=block_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            use_sinks=use_sinks,
            sliding_window=sliding_window,
            has_softcap=has_softcap,
            use_alibi=use_alibi,
            **kw,
        )
        return _build_fn(spec, arch=arch)

    return _build


def build_attention_3d(
    name,
    arch,
    *,
    head_size,
    block_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    num_segments,
    sliding_window=0,
    has_softcap=False,
    **kw,
):
    def _build():
        from kernels.common.attention_unified import _tiled_3d_impl

        SpecCls, _, build_fn, _, _ = _tiled_3d_impl(arch)
        spec = SpecCls(
            head_size=head_size,
            block_size=block_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            use_sinks=False,
            sliding_window=sliding_window,
            has_softcap=has_softcap,
            num_segments=num_segments,
            **kw,
        )
        return build_fn(spec, arch=arch)

    return _build


def build_attention_reduce(
    name, arch, *, head_size, num_query_heads, num_kv_heads, dtype, num_segments
):
    def _build():
        from kernels.common.attention_unified import _tiled_3d_impl

        _, SpecCls, _, build_fn, _ = _tiled_3d_impl(arch)
        spec = SpecCls(
            head_size=head_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            num_segments=num_segments,
        )
        return build_fn(spec, arch=arch)

    return _build


def build_attention_reduce_own_module(
    name,
    arch,
    *,
    head_size,
    num_query_heads,
    num_kv_heads,
    dtype,
    num_segments,
    emit_lse=False,
):
    """Split-KV reduce built from ``arch``'s own module (``build_attention_reduce``
    above always builds the gfx950 module), with or without the LSE output."""

    def _build():
        import importlib

        mod = importlib.import_module(f"kernels.{arch}.attention_tiled_3d")
        spec = mod.UnifiedAttentionReduceTiledSpec(
            head_size=head_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            num_segments=num_segments,
            emit_lse=emit_lse,
        )
        return mod.build_unified_attention_reduce_tiled(spec, arch=arch)

    return _build


def build_attention_scalar(name, arch, *, three_d=False, reduce=False, **problem):
    """Scalar (correctness-oracle) unified attention: 2D, split-3D segment kernel,
    or (``reduce``) the cross-segment reduce that finishes the split-3D path.

    ``problem`` patches a small ``UnifiedAttentionProblem``; the band fields
    (``causal_top_left``, ``right_bound``, ``sliding_window``) select the mask.
    """

    def _build():
        from kernels import (
            UnifiedAttention2DSpec,
            UnifiedAttention3DSpec,
            UnifiedAttentionProblem,
            UnifiedAttentionReduceSpec,
            build_unified_attention_2d,
            build_unified_attention_3d,
            build_unified_attention_reduce,
        )

        fields = dict(
            total_q=64,
            num_seqs=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=64,
            max_seqlen_k=64,
            dtype="fp16",
        )
        fields.update(problem)
        p = UnifiedAttentionProblem(**fields)
        if reduce:
            return build_unified_attention_reduce(
                UnifiedAttentionReduceSpec(p, num_segments=8), arch=arch
            )
        if three_d:
            return build_unified_attention_3d(
                UnifiedAttention3DSpec(p, num_segments=8), arch=arch
            )
        return build_unified_attention_2d(UnifiedAttention2DSpec(p), arch=arch)

    return _build


def build_attention_dense(arch, *, geometry=None, **over):
    """Dense flash-attn prefill spec (library ``kernels/gfx950/attention_dense``).

    ``geometry`` selects a shared tile geometry and ``over`` patches the base spec;
    a small Sq keeps the IR compact while still exercising the full pipeline (both
    the default one-CTA-per-q-block grid and the persistent grid-stride grid).
    """

    def _build():
        from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
        from kernels.gfx950.attention_dense import (
            Gfx950AttentionDenseSpec,
            build_attention_dense as _build_dense,
        )

        spec = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="bf16",
        )
        if geometry is not None:
            spec.update(DENSE_TILE_GEOMETRIES[geometry])
        spec.update(over)
        return _build_dense(Gfx950AttentionDenseSpec(**spec))

    return _build


def build_attention_dense_gfx942(**over):
    """Dense flash-attn prefill spec (library ``kernels/gfx942/attention_dense``)."""

    def _build():
        from kernels.gfx942.attention_dense import (
            Gfx942AttentionDenseSpec,
            build_attention_dense as _build_dense,
        )

        spec = dict(
            batch=1,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="bf16",
        )
        spec.update(over)
        return _build_dense(Gfx942AttentionDenseSpec(**spec), arch="gfx942")

    return _build


def build_kda_chunkwise_gfx950(kind, arch, **over):
    """gfx950 chunkwise KDA kernel from a representative spec.

    ``kind`` selects one of the three emitted kernels. Spec overrides keep the
    case table compact while covering state flags, fused preprocessing, and a
    non-default scan partition.
    """

    def _build():
        from kernels.gfx950.kda_chunkwise import (
            KdaChunkFusedSpec,
            KdaChunkPrepSpec,
            KdaChunkScanSpec,
            KdaTileSpec,
            build_kda_chunk_fused,
            build_kda_chunk_prep,
            build_kda_chunk_scan,
        )

        if kind == "prep":
            return build_kda_chunk_prep(KdaChunkPrepSpec(**over), arch=arch)
        if kind == "scan":
            scan_over = dict(over)
            tile_over = scan_over.pop("tile", {})
            tile = KdaTileSpec(**tile_over) if tile_over else KdaTileSpec()
            return build_kda_chunk_scan(
                KdaChunkScanSpec(tile=tile, **scan_over), arch=arch
            )
        if kind == "fused":
            return build_kda_chunk_fused(KdaChunkFusedSpec(**over), arch=arch)
        raise ValueError(f"unknown KDA kernel kind {kind!r}")

    return _build


def build_kda_chunkwise_gfx942(kind, arch, **over):
    """Build one representative gfx942 chunkwise KDA kernel."""

    def _build():
        from kernels.gfx942.kda_chunkwise import (
            KdaChunkFusedSpec,
            KdaChunkPrepSpec,
            KdaChunkScanSpec,
            build_kda_chunk_fused,
            build_kda_chunk_prep,
            build_kda_chunk_scan,
        )

        specs = {
            "prep": (KdaChunkPrepSpec, build_kda_chunk_prep),
            "scan": (KdaChunkScanSpec, build_kda_chunk_scan),
            "fused": (KdaChunkFusedSpec, build_kda_chunk_fused),
        }
        try:
            spec_type, builder = specs[kind]
        except KeyError as exc:
            raise ValueError(f"unknown gfx942 KDA kernel kind {kind!r}") from exc
        return builder(spec_type(**over), arch=arch)

    return _build


def build_lightning_indexer_case(arch, **over):
    """Build one representative lightning-indexer kernel for ``arch``.

    The kernel is arch-neutral (gfx942/gfx950); ``arch`` only selects the lowering
    target. Overrides are plain ``IndexerSpec`` fields; ``block_size`` is lifted
    into the tile spec so the case table stays flat.
    """

    def _build():
        from kernels.common.lightning_indexer import (
            IndexerSpec,
            IndexerTileSpec,
            build_lightning_indexer,
        )

        opts = dict(over)
        block_size = opts.pop("block_size", None)
        tile = (
            IndexerTileSpec(block_size=block_size)
            if block_size is not None
            else IndexerTileSpec()
        )
        return build_lightning_indexer(IndexerSpec(tile=tile, **opts), arch=arch)

    return _build


def _d256_problem(head_size=256):
    """Validated D256 cohort point (GQA 16/2, hd256, bs16, sq4096 bf16).

    ``head_size=192`` gives the D192 point of the same cohort (same lean body)."""
    from kernels.common.attention_unified import UnifiedAttentionProblem

    return UnifiedAttentionProblem(
        total_q=4096,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=head_size,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        num_cus=120,
    )


def _d128_swa_fold_problem():
    """Validated gfx942 GQA head-fold cohort point (GQA 32/8 = 4:1, hd128, bs16,
    sq8192 bf16, sliding window).

    ``block_size=16`` on purpose: BPT = BN // BS is the number of paged KV blocks
    consumed per 32-key tile, so bs16 walks TWO block-table entries per tile where
    bs32 walks one. Pinning the bs16 variant records the busier paged gather.
    """
    from kernels.common.attention_unified import UnifiedAttentionProblem

    return UnifiedAttentionProblem(
        total_q=8192,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=8192,
        max_seqlen_k=8192,
        dtype="bf16",
        sliding_window=4096,
        num_cus=120,
    )


@contextlib.contextmanager
def _pinned_attention_arch(arch):
    """Pin the *memoized runtime device arch* the D256 fast routes gate on.

    Those routes consult ``_resolve_attention_arch`` -- NOT any per-case argument
    -- so a non-gfx950 host would otherwise select the FALLBACK spec and the
    recorded sha would become host-dependent. Two things are pinned, and the
    second is the load-bearing one:

    * the function is rebound wholesale on its defining module (per its
      docstring: "Tests that monkeypatch this function replace it wholesale");
    * ``_RESOLVED_ATTENTION_ARCH``, the process-wide memo the function returns
      before doing anything else, is set too. The memo is what makes the pin
      independent of *how* a consumer reached the resolver: the spec builder
      goes through its ``attention_unified`` module handle today, but a bound
      import anywhere would freeze the function reference and see only the memo.
      This harness is installed standalone (``platform/CMakeLists.txt``) and run
      by ``rocke_installed_golden_test.py`` without the library test tree, so it
      cannot delegate that invariant to a guard test.
    """
    import kernels.common.attention_unified as au

    pin = lambda: arch  # noqa: E731
    o_fn, o_memo = au._resolve_attention_arch, au._RESOLVED_ATTENTION_ARCH
    au._resolve_attention_arch = pin
    au._RESOLVED_ATTENTION_ARCH = arch
    try:
        yield
    finally:
        au._resolve_attention_arch = o_fn
        au._RESOLVED_ATTENTION_ARCH = o_memo


def build_attention_d256_gfx950(arch):
    """D256 bf16 prefill *fast* spec: 32x32-transposed + ``use_kq_lds_pad``
    (slab-granularity K_lds pad) + ``use_softmax_mfma_interleave``.

    The slab pad's async-DMA write and its padded read remap must agree
    byte-for-byte or numerics silently corrupt, and an addressing drift would pass
    every other CPU test and fail only on GPU (#9233 review). Raises if the fast
    spec / pad+interleave is not selected, so this can never pin the fallback.
    """

    def _build():
        from kernels import build_unified_attention_2d_tiled
        from kernels.common.attention_unified import (
            _d256_gfx950_fast,
            _tiled_spec_from_problem,
        )

        problem = _d256_problem()
        with _pinned_attention_arch(arch):
            if not _d256_gfx950_fast(problem, arch):
                raise RuntimeError(
                    f"D256 fast spec not selected under pinned arch {arch!r}; "
                    "would pin the fallback (no pad/interleave)"
                )
            spec = _tiled_spec_from_problem(problem, arch)
            if not (spec.use_kq_lds_pad and spec.use_softmax_mfma_interleave):
                raise RuntimeError(
                    "expected pad+interleave in the D256 fast spec; got "
                    f"pad={spec.use_kq_lds_pad} "
                    f"interleave={spec.use_softmax_mfma_interleave}"
                )
            return build_unified_attention_2d_tiled(spec, arch=arch)

    return _build


def build_attention_d256_gfx942(arch, head_size=256):
    """D256 gfx942 4-warp GQA fast path.

    Mirrors ``build_attention_d256_gfx950`` but pins the *gfx942* cohort: raises
    if ``_d256_gfx942_fast`` is not selected (so this can never pin the fallback),
    then lowers the dedicated ``build_gfx942_4warp_gqa`` builder -- the
    byte-sensitive V wide-load -> ``V_lds`` transpose kernel worth pinning.
    ``head_size=192`` pins the same lean body at D192, including its
    head-size-derived V-fill stride and the K/V key and Q row bounds.
    """

    def _build():
        from kernels.common.attention_unified import (
            _d256_gfx942_fast,
            _tiled_spec_from_problem,
        )
        from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

        problem = _d256_problem(head_size)
        with _pinned_attention_arch(arch):
            if not _d256_gfx942_fast(problem, arch):
                raise RuntimeError(
                    f"D256 gfx942 fast route not selected under pinned arch "
                    f"{arch!r}; would pin the fallback (slow scalar path)"
                )
            spec = _tiled_spec_from_problem(problem, arch)
            return build_gfx942_4warp_gqa(spec, arch=arch)

    return _build


def build_attention_d128_swa_fold_gfx942(arch):
    """gfx942 GQA head-fold (D128 sliding-window bf16), the DEFAULT for its cohort.

    Mirrors ``build_attention_d256_gfx942`` but pins the *fold* cohort. The D256
    case above cannot stand in for this one: ``build_gfx942_4warp_gqa`` returns
    ``_build_gfx942_4warp_gqa_lean`` at ``HD in _4WGQA_LEAN_HEAD_SIZES`` (192, 256)
    early-return, which is ~20 lines BEFORE the D128 fold body -- so the D256
    golden never lowers a single line of the folded kernel.

    The fold rewrites device-side (token, head) index math, the folded head index
    ``kv_head*4 + m%4`` and the launch grid. Its other guards both have escape
    hatches: the CPU test only greps the kernel NAME (a mangled kernel keeps its
    name), and the on-GPU oracle is skipped on every non-gfx942 host. This golden
    is the only one of the three that runs at every llvm flavor on any host, with
    no GPU.

    Raises if the cohort is not fold-eligible, so a future predicate narrowing can
    never silently re-bless the UNFOLDED kernel under this case id.
    """

    def _build():
        from kernels.common.attention_unified import (
            _tiled_spec_from_problem,
            gfx942_gqa_fold_eligible,
        )
        from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

        problem = _d128_swa_fold_problem()
        with _pinned_attention_arch(arch):
            if not gfx942_gqa_fold_eligible(
                problem.head_size,
                problem.num_queries_per_kv,
                problem.sliding_window,
                problem.dtype,
                problem.block_size,
            ):
                raise RuntimeError(
                    f"GQA head-fold not selected under pinned arch {arch!r}; "
                    "would pin the unfolded kernel"
                )
            spec = _tiled_spec_from_problem(problem, arch)
            kernel = build_gfx942_4warp_gqa(spec, arch=arch)
            if not kernel.name.endswith("_4wgqa_fold"):
                raise RuntimeError(
                    f"fold-eligible cohort lowered {kernel.name!r}, which is not "
                    "the folded kernel; the predicate and the builder disagree"
                )
            return kernel

    return _build


def build_grouped_gemm_case(name, arch, m, n, k, e):
    def _build():
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
        )

        spec = GroupedGemmSpec(M=m, N=n, K=k, E=e, name=name)
        kernel, _bs, _tm, _tn = build_grouped_gemm(spec)
        return kernel

    return _build


def build_mxfp8_gemm_case(dtype, matrix_path):
    def _build():
        from rocke.instances.gfx1250.block_scaled_gemm import (
            BlockScaledGemmSpec,
            build_block_scaled_gemm,
        )

        return build_block_scaled_gemm(
            BlockScaledGemmSpec(
                name=f"irhash_mxfp8_{dtype}_{matrix_path}",
                M=32,
                N=48,
                K=256,
                dtype_a=dtype,
                dtype_b=dtype,
                dtype_c="bf16",
                scale_dtype="e8m0",
                matrix_path=matrix_path,
                block_k=16 if matrix_path == "wmma_scale16" else 32,
            )
        )

    return _build


def build_nvfp4_gemm_case(dtype, k):
    def _build():
        from rocke.instances.gfx1250.nvfp4_gemm import (
            NvFp4GemmSpec,
            build_nvfp4_gemm,
        )

        return build_nvfp4_gemm(
            NvFp4GemmSpec(f"irhash_nvfp4_{dtype}", 32, 48, k, dtype),
            arch="gfx1250",
        )

    return _build


def build_streaming_copy(arch):
    """bf16x8 copy whose load and store both carry TemporalHint.STREAMING, so
    the golden pins the full nontemporal emission (`!nontemporal !5` on both
    ops plus the one module-level `!5 = !{i32 1}` node), not just substrings."""

    def _build():
        from rocke.core.ir import BF16, IRBuilder, PtrType, TemporalHint

        b = IRBuilder(f"irhash_nontemporal_copy_{arch}")
        src = b.param(
            "S", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
        )
        dst = b.param("D", PtrType(BF16, "global"), noalias=True, align=16)
        off = b.mul(b.thread_id_x(), b.const_i32(8))
        v = b.global_load_vN(src, off, BF16, 8, temporal_hint=TemporalHint.STREAMING)
        b.global_store_vN(dst, off, v, 8, temporal_hint=TemporalHint.STREAMING)
        b.ret()
        return b.kernel

    return _build


def build_f8f6f4_16x16x128(kind, fmt=None):
    """One-wave K=128 f8f6f4 MFMA on gfx950. ``hero`` is the unscaled fp8 atom
    (both scales literal 0); ``mx`` is the MX op with unit E8M0 scales;
    ``unscaled`` is the unscaled op for every format. ``fmt`` sets the data type
    of A and B (default fp8e4m3), so the golden pins the shared declaration and
    each format's selectors at every flavor."""

    def _build():
        from rocke.core.ir import F32, I32, IRBuilder, PtrType

        suffix = f"_{fmt}" if fmt else ""
        b = IRBuilder(f"irhash_f8f6f4_16x16x128_{kind}{suffix}")
        b.kernel.attrs["max_workgroup_size"] = 64
        pa = b.param("A", PtrType(I32, "global"), align=32)
        pb = b.param("B", PtrType(I32, "global"), align=32)
        pc = b.param("C", PtrType(F32, "global"), align=16)
        tid = b.thread_id_x()
        off8 = b.mul(tid, b.const_i32(8))
        a = b.global_load_vN(pa, off8, I32, 8, align=32)
        bb = b.global_load_vN(pb, off8, I32, 8, align=32)
        acc = b.vector_splat(b.const_f32(0.0), 4)
        kw = {"a_format": fmt, "b_format": fmt} if fmt else {}
        if kind == "hero":
            r = b.mma("mfma_f32_16x16x128_fp8", a, bb, acc)
        elif kind == "unscaled":
            r = b.mfma_f32_16x16x128_fp4fp6fp8(a, bb, acc, **kw)
        else:
            r = b.mfma_scale_f32_16x16x128_f8f6f4(
                a, bb, acc, b.const_i32(127), b.const_i32(127), **kw
            )
        b.global_store_vN(pc, b.mul(tid, b.const_i32(4)), r, 4, align=16)
        b.ret()
        return b.kernel

    return _build


def build_warp_shuffle_xor_dpp_reduce(arch, masks):
    """16-lane i32 sum ``v = v + warp_shuffle_xor_dpp(v, m)`` over ``masks``:
    pins the per-target ``tile.dpp_xor`` encodings inside the reduction they
    exist for, so a change that alters the op's IR in both engines at once
    still trips the golden. gfx9 uses masks 1, 2, 7, 15 (quad_perm x2,
    row_half_mirror, row_mirror); a row_xmask target uses 1, 2, 4, 8."""

    def _build():
        from rocke.core.ir import I32, IRBuilder, PtrType

        b = IRBuilder(f"irhash_warp_shuffle_xor_dpp_reduce_{arch}")
        src = b.param("S", PtrType(I32, "global"), noalias=True, readonly=True)
        dst = b.param("D", PtrType(I32, "global"), noalias=True)
        tid = b.thread_id_x()
        acc = b.global_load(src, tid, I32, align=4)
        for m in masks:
            acc = b.add(acc, b.warp_shuffle_xor_dpp(acc, m))
        b.global_store(dst, tid, acc, align=4)
        b.ret()
        return b.kernel

    return _build


def build_warp_shuffle_xor_dpp_masks_f32(arch):
    """f32 shuffle by every register-only gfx9 mask (1, 2, 3, 7, 8, 15),
    summed: pins each table entry and the f32 <-> i32 bitcast path."""

    def _build():
        from rocke.core.ir import F32, IRBuilder, PtrType

        b = IRBuilder(f"irhash_warp_shuffle_xor_dpp_masks_f32_{arch}")
        src = b.param("S", PtrType(F32, "global"), noalias=True, readonly=True)
        dst = b.param("D", PtrType(F32, "global"), noalias=True)
        tid = b.thread_id_x()
        x = b.global_load(src, tid, F32, align=4)
        acc = x
        for m in (1, 2, 3, 7, 8, 15):
            acc = b.fadd(acc, b.warp_shuffle_xor_dpp(x, m))
        b.global_store(dst, tid, acc, align=4)
        b.ret()
        return b.kernel

    return _build


def cases():
    out = []

    def add(family, case_id, arch, build):
        out.append({"family": family, "case_id": case_id, "arch": arch, "build": build})

    # GEMM: tile/atom/pipeline variants across CDNA and RDNA arches.
    add(
        "gemm",
        "gemm/gfx942/t64x64x16",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_a",
            "gfx942",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx942/t128x128x16",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_b",
            "gfx942",
            128,
            128,
            16,
            2,
            2,
            32,
            32,
            8,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx950/t128x128x32",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_a",
            "gfx950",
            128,
            128,
            32,
            2,
            2,
            32,
            32,
            16,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx950/t128x128x32/cshuffle_no_alias",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_a_noalc",
            "gfx950",
            128,
            128,
            32,
            2,
            2,
            32,
            32,
            16,
            "compv4",
            "cshuffle",
            64,
            cshuffle_no_alias=True,
        ),
    )
    add(
        "gemm",
        "gemm/gfx950/t64x128x32",
        "gfx950",
        build_gemm(
            "irhash_gemm_950_b",
            "gfx950",
            64,
            128,
            32,
            2,
            2,
            16,
            16,
            32,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1151/t32x32x16",
        "gfx1151",
        build_gemm(
            "irhash_gemm_1151_a",
            "gfx1151",
            32,
            32,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1151/t64x32x16",
        "gfx1151",
        build_gemm(
            "irhash_gemm_1151_b",
            "gfx1151",
            64,
            32,
            16,
            2,
            1,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1201/t32x32x16",
        "gfx1201",
        build_gemm(
            "irhash_gemm_1201_a",
            "gfx1201",
            32,
            32,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1201/t64x32x16",
        "gfx1201",
        build_gemm(
            "irhash_gemm_1201_b",
            "gfx1201",
            64,
            32,
            16,
            2,
            1,
            16,
            16,
            16,
            "mem",
            "default",
            32,
        ),
    )
    # Every other member of gfx11-generic / gfx12-generic has a row of its own
    # that points at its generic's facts; one case each, the gfx1151 / gfx1201
    # t32x32x16 configuration, shows it lowers under its own name.
    for member in (
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1152",
        "gfx1153",
        "gfx1154",
        "gfx1200",
    ):
        add(
            "gemm",
            f"gemm/{member}/t32x32x16",
            member,
            build_gemm(
                f"irhash_gemm_{member[3:]}_a",
                member,
                32,
                32,
                16,
                2,
                2,
                16,
                16,
                16,
                "mem",
                "default",
                32,
            ),
        )
    # gfx90a is wave64/MFMA like gfx942; mirror its 16x16x16 atom variants.
    add(
        "gemm",
        "gemm/gfx90a/t64x64x16",
        "gfx90a",
        build_gemm(
            "irhash_gemm_90a_a",
            "gfx90a",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            64,
        ),
    )
    add(
        "gemm",
        "gemm/gfx90a/t128x128x16",
        "gfx90a",
        build_gemm(
            "irhash_gemm_90a_b",
            "gfx90a",
            128,
            128,
            16,
            2,
            2,
            32,
            32,
            8,
            "compv4",
            "cshuffle",
            64,
        ),
    )
    # gfx908 is wave64/MFMA with the same f16 16x16x16 atom as gfx90a.
    add(
        "gemm",
        "gemm/gfx908/t64x64x16",
        "gfx908",
        build_gemm(
            "irhash_gemm_908_a",
            "gfx908",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            16,
            "mem",
            "default",
            64,
        ),
    )
    # gfx1250 is wave32/WMMA like gfx1201, but its fp16 atom is K=32 (16x16x32),
    # so warp_tile_k and tile_k are 32 rather than 16. NOTE: these lower through
    # the Python engine only -- the C++ engine has no gfx1250 ISA backend yet
    # (rocke_ll_backend_for rejects gfx1250; see cpp/core/lower_llvm/mma.cpp), so
    # they are not part of the C-vs-Python byte-identity gate until that lands.
    add(
        "gemm",
        "gemm/gfx1250/t32x32x32",
        "gfx1250",
        build_gemm(
            "irhash_gemm_1250_a",
            "gfx1250",
            32,
            32,
            32,
            2,
            2,
            16,
            16,
            32,
            "mem",
            "default",
            32,
        ),
    )
    add(
        "gemm",
        "gemm/gfx1250/t64x32x32",
        "gfx1250",
        build_gemm(
            "irhash_gemm_1250_b",
            "gfx1250",
            64,
            32,
            32,
            2,
            1,
            16,
            16,
            32,
            "mem",
            "default",
            32,
        ),
    )

    # fp64 GEMM on gfx942: 16x16x4 atom, plus the DTL + prefetch + swizzle path.
    add(
        "gemm",
        "gemm/gfx942/f64_t64x64x16",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_f64",
            "gfx942",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            4,
            "compv4",
            "default",
            64,
            dtype="fp64",
        ),
    )
    add(
        "gemm",
        "gemm/gfx942/f64_t64x64x16/dtl_prefetch_swizzle",
        "gfx942",
        build_gemm(
            "irhash_gemm_942_f64_dtl",
            "gfx942",
            64,
            64,
            16,
            2,
            2,
            16,
            16,
            4,
            "compv4",
            "default",
            64,
            dtype="fp64",
            direct_to_lds=True,
            dtl_prefetch=True,
            lds_swizzle=True,
            pad_k=False,
        ),
    )
    # Homogeneous FP8/BF8 with E8M0 scales, shared by source and installed gates.
    for dtype in ("fp8e4m3", "bf8e5m2"):
        for matrix_path in ("wmma_scale", "wmma_scale16"):
            add(
                "gemm",
                f"gemm/gfx1250/mxfp8/{matrix_path}/{dtype}",
                "gfx1250",
                build_mxfp8_gemm_case(dtype, matrix_path),
            )

    # Tensor-scaled FP4/E4M3, including both output epilogues and K unroll counts.
    for dtype in ("bf16", "fp16"):
        for k in (128, 256):
            add(
                "gemm",
                f"gemm/gfx1250/nvfp4/{dtype}/k{k}",
                "gfx1250",
                build_nvfp4_gemm_case(dtype, k),
            )

    # MoE: sorting phases and fused-MoE streaming phases.
    for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
        add(
            "moe",
            f"moe_sort/{arch}/hist_t32_k2_e8",
            arch,
            build_moe_sort("histogram", 32, 2, 8, 128, arch),
        )
    add(
        "moe",
        "moe_sort/gfx950/scan_t64_k2_e16",
        "gfx950",
        build_moe_sort("scan", 64, 2, 16, 128, "gfx950"),
    )
    add(
        "moe",
        "moe_sort/gfx950/scatter_t64_k2_e16",
        "gfx950",
        build_moe_sort("scatter", 64, 2, 16, 128, "gfx950"),
    )
    add(
        "moe",
        "fused_moe/gfx950/gather_h128",
        "gfx950",
        build_fused_moe("gather", 32, 8, 2, 128, 256),
    )
    add(
        "moe",
        "fused_moe/gfx950/silu_i256",
        "gfx950",
        build_fused_moe("silu", 32, 8, 2, 128, 256),
    )
    add(
        "moe",
        "fused_moe/gfx950/reduce_h128",
        "gfx950",
        build_fused_moe("reduce", 32, 8, 2, 128, 256),
    )

    # Attention: 2D tiled (gfx950 wide-K), 3D tiled + reduce (gfx950),
    # and 2D tiled narrow (gfx942 MFMA-16x16x16 path).
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_bf16_d128_b64",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_bf16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64_softcap",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16_sc",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            has_softcap=True,
        ),
    )
    add(
        "attention",
        "attention/gfx950/2d_fp16_d128_b64_alibi",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16_alibi",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            use_alibi=True,
        ),
    )
    # fp16 + sinks COMBO: the transposed-32x32 combo spec `_enable_combo_2d`
    # admits for fp16+sinks. The combo knobs are passed explicitly so the golden
    # hashes the real combo kernel (matching emit parity idx54), not a plain 2D
    # spec. fp16 cannot set use_fast_paged_kv_desc (bf16-only).
    _sink_combo_kw = dict(
        num_seqs=2,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    )
    add(
        "attention",
        "attention/gfx950/2d_fp16_d64_b32_gqa8_sinks_combo",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_fp16_d64_sink_combo",
            "gfx950",
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="fp16",
            use_sinks=True,
            **_sink_combo_kw,
        ),
    )
    # bf16 + sinks COMBO twin (adds use_fast_paged_kv_desc, bf16-only).
    add(
        "attention",
        "attention/gfx950/2d_bf16_d64_b32_gqa8_sinks_combo",
        "gfx950",
        build_attention_2d(
            "irhash_attn_950_2d_bf16_d64_sink_combo",
            "gfx950",
            head_size=64,
            block_size=32,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=True,
            use_fast_paged_kv_desc=True,
            **_sink_combo_kw,
        ),
    )
    add(
        "attention",
        "attention/gfx950/3d_fp16_d128_b64",
        "gfx950",
        build_attention_3d(
            "irhash_attn_950_3d_fp16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/3d_bf16_d128_b64",
        "gfx950",
        build_attention_3d(
            "irhash_attn_950_3d_bf16",
            "gfx950",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/reduce_fp16_d128",
        "gfx950",
        build_attention_reduce(
            "irhash_attn_950_reduce_fp16",
            "gfx950",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/reduce_bf16_d128",
        "gfx950",
        build_attention_reduce(
            "irhash_attn_950_reduce_bf16",
            "gfx950",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx950/reduce_lse_bf16_d128",
        "gfx950",
        build_attention_reduce_own_module(
            "irhash_attn_950_reduce_lse_bf16",
            "gfx950",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
            emit_lse=True,
        ),
    )
    # Tiled 2D with LSE output, one case per epilogue branch (each branch has
    # its own LSE store site).
    for _branch, _branch_kw in (
        (
            "transposed32",
            dict(
                use_mfma_32x32=True,
                block_m_per_warp=32,
                use_transposed_qk_32x32=True,
                use_transposed_scalar_state=True,
            ),
        ),
        ("mfma32", dict(use_mfma_32x32=True, block_m_per_warp=32)),
        ("striped16", {}),
    ):
        add(
            "attention",
            f"attention/gfx950/2d_lse_{_branch}_bf16_d128",
            "gfx950",
            build_attention_2d(
                f"irhash_attn_950_2d_lse_{_branch}_bf16",
                "gfx950",
                head_size=128,
                block_size=64,
                num_query_heads=4,
                num_kv_heads=2,
                dtype="bf16",
                emit_lse=True,
                **_branch_kw,
            ),
        )
    add(
        "attention",
        "attention/gfx942/2d_fp16_d128_b64",
        "gfx942",
        build_attention_2d(
            "irhash_attn_942_2d_fp16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
        ),
    )
    add(
        "attention",
        "attention/gfx942/2d_bf16_d128_b64",
        "gfx942",
        build_attention_2d(
            "irhash_attn_942_2d_bf16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
        ),
    )
    # bf16 D128 decode on a paged cache past the i32 buffer range: i64 K/V
    # addressing with four pages per tile and per-page buffer resources.
    add(
        "attention",
        "attention/gfx942/2d_bf16_d128_b16_decode_i64",
        "gfx942",
        build_attention_2d(
            "irhash_attn_942_2d_bf16_decode_i64",
            "gfx942",
            head_size=128,
            block_size=16,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="bf16",
            num_seqs=256,
            waves_per_eu=2,
            use_register_pv=True,
            tile_size=64,
            use_i64_kv_addr=True,
        ),
    )
    # LSE-on tiled 2D, one case per epilogue branch (transposed 32x32x8 direct
    # store, non-transposed 32x32x8 via Acc_lds, narrow 16x16 striped). The
    # LSE-off cases above stay byte-identical.
    for _branch, _branch_kw in (
        (
            "transposed32",
            dict(
                use_mfma_32x32x8=True,
                use_transposed_qk_32x32=True,
                block_m_per_warp=32,
            ),
        ),
        ("mfma32", dict(use_mfma_32x32x8=True, block_m_per_warp=32)),
        ("striped16", {}),
    ):
        add(
            "attention",
            f"attention/gfx942/2d_lse_{_branch}_bf16_d128",
            "gfx942",
            build_attention_2d(
                f"irhash_attn_942_2d_lse_{_branch}_bf16",
                "gfx942",
                head_size=128,
                block_size=64,
                num_query_heads=4,
                num_kv_heads=2,
                dtype="bf16",
                emit_lse=True,
                **_branch_kw,
            ),
        )
    add(
        "attention",
        "attention/gfx942/3d_fp16_d128_b64",
        "gfx942",
        build_attention_3d(
            "irhash_attn_942_3d_fp16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/3d_bf16_d128_b64",
        "gfx942",
        build_attention_3d(
            "irhash_attn_942_3d_bf16",
            "gfx942",
            head_size=128,
            block_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_fp16_d128",
        "gfx942",
        build_attention_reduce(
            "irhash_attn_942_reduce_fp16",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_bf16_d128",
        "gfx942",
        build_attention_reduce(
            "irhash_attn_942_reduce_bf16",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_lse_bf16_d128",
        "gfx942",
        build_attention_reduce_own_module(
            "irhash_attn_942_reduce_lse_bf16",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
            emit_lse=True,
        ),
    )
    add(
        "attention",
        "attention/gfx942/reduce_bf16_d128_own_module",
        "gfx942",
        build_attention_reduce_own_module(
            "irhash_attn_942_reduce_bf16_own_module",
            "gfx942",
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="bf16",
            num_segments=4,
        ),
    )

    # Dense flash-attn prefill (gfx950-only builder). Registered as a table rather
    # than 18 add() blocks because every case is the same base spec with one or two
    # overrides; the variant name is the only thing that differs case to case.
    for _variant, _over in (
        # --- default (one CTA per q-block/head) grid ---
        ("default_causal_sq512", {}),
        ("swa_w128_sq512", {"sliding_window": 128}),
        ("varlen_sq512", {"varlen": True}),
        ("lazy_off_sq512", {"lazy_rescale": False}),
        ("fp16_h64_sq512", {"dtype": "fp16", "head_size": 64}),
        ("bn128_sq512", {"block_n": 128}),
        ("noncausal_sq512", {"causal": False}),
        ("hqswz_sq512", {"nonpersist_decode": "hq_minor_swz"}),
        # --- bottom-right diagonal. Four cases pin the aligned and arbitrary
        # shifted-diagonal routes, the sink composition, and the BM128 geometry.
        (
            "bottom_right_sq512",
            {"seqlen_kv": 1024, "causal_bottom_right": True},
        ),
        (
            "ragged_bottom_right_sq500",
            {
                "seqlen_q": 500,
                "seqlen_kv": 1234,
                "ragged": True,
                "causal_bottom_right": True,
            },
        ),
        (
            "bottom_right_sinks_sq512",
            {
                "seqlen_kv": 1024,
                "causal_bottom_right": True,
                "use_sinks": True,
            },
        ),
        (
            "bottom_right_bm128_sq512",
            {
                "seqlen_kv": 1024,
                "causal_bottom_right": True,
                "geometry": "bm128",
            },
        ),
        # KV-segment split of the default grid (segment epilogue + range).
        (
            "bottom_right_kvs4_bm128_sq512",
            {
                "seqlen_kv": 2048,
                "causal_bottom_right": True,
                "geometry": "bm128",
                "kv_segments": 4,
            },
        ),
        # --- persistent (grid-stride) grid + decode variants ---
        ("persistent_causal_sq512", {"persistent": True, "num_persistent": 256}),
        (
            "persist_qbmaj_sq512",
            {"persistent": True, "num_persistent": 256, "persist_decode": "qb_major"},
        ),
        (
            "persist_hkvmaj_sq512",
            {"persistent": True, "num_persistent": 256, "persist_decode": "hkv_major"},
        ),
        (
            "persist_hqswz_sq512",
            {
                "persistent": True,
                "num_persistent": 256,
                "persist_decode": "hq_minor_swz",
            },
        ),
        (
            "persist_intl_sq512",
            {"persistent": True, "num_persistent": 256, "interleave": True},
        ),
        (
            "persist_lazy_off_sq512",
            {"persistent": True, "num_persistent": 256, "lazy_rescale": False},
        ),
        (
            "persist_swa_w128_sq512",
            {"persistent": True, "num_persistent": 256, "sliding_window": 128},
        ),
        (
            "persist_wdma_gqapair_fp16_sq512",
            {
                "dtype": "fp16",
                "num_query_heads": 32,
                "num_kv_heads": 8,
                "persistent": True,
                "num_persistent": 16,
                "persist_decode": "gqa_pair",
                "wide_lds_dma": True,
            },
        ),
        # D=64 packed-row DMA loader (2 rows/instr, unpadded LDS) on the persistent
        # builder -- locks the head_size=64 fix (fp16_h64 above only exercises the
        # default builder).
        (
            "persist_h64_sq512",
            {"persistent": True, "num_persistent": 256, "head_size": 64},
        ),
        # ragged (non-256 seqlen) in-kernel path: on-chip boundary padding. causal
        # (no key mask), non-causal (ktok<seqlen_kv key mask), D=64, and the
        # persistent variant -- lock all four ragged codegen shapes.
        ("ragged_causal_sq500", {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True}),
        (
            "ragged_full_sq500",
            {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True, "causal": False},
        ),
        (
            "ragged_h64_sq500",
            {"seqlen_q": 500, "seqlen_kv": 500, "ragged": True, "head_size": 64},
        ),
        (
            "persist_ragged_sq500",
            {
                "seqlen_q": 500,
                "seqlen_kv": 500,
                "ragged": True,
                "persistent": True,
                "num_persistent": 256,
            },
        ),
        # paged-KV load path (block_tables indirection). Gates the PAGED DSL through
        # BOTH the golden (Python lowering byte-stability) and the cpp/python
        # byte-identity gate. fp16 D128 sliding-window single-seq (the validated
        # cohort); num_kv_blocks = seqlen_kv / block_size (32 = 512 / 16).
        (
            "paged_swa_fp16_sq512",
            {
                "dtype": "fp16",
                "paged": True,
                "block_size": 16,
                "num_kv_blocks": 32,
                "sliding_window": 256,
            },
        ),
        # --- Attention band: the cuDNN (left = sliding_window, right = right_bound)
        # band on the top-left diagonal. causal=False is the unbounded right side.
        # Non-causal (left-only) window: default + persistent grids.
        ("noncausal_window_sq512", {"causal": False, "sliding_window": 128}),
        (
            "persist_noncausal_window_sq512",
            {
                "causal": False,
                "sliding_window": 128,
                "persistent": True,
                "num_persistent": 256,
            },
        ),
        # Two-sided window and plain lookahead (right_bound > 0 needs causal=True).
        (
            "two_sided_window_sq512",
            {"sliding_window": 128, "right_bound": 64},
        ),
        ("lookahead_sq512", {"right_bound": 64}),
        (
            "persist_lookahead_sq512",
            {"right_bound": 64, "persistent": True, "num_persistent": 256},
        ),
        # Cross-attention between different lengths: full attention, ragged both ways,
        # and top-left causal with seqlen_q > seqlen_kv (key-pad mask on a causal body).
        (
            "full_ragged_cross_197x400",
            {"seqlen_q": 197, "seqlen_kv": 400, "ragged": True, "causal": False},
        ),
        (
            "top_left_ragged_cross_400x197",
            {"seqlen_q": 400, "seqlen_kv": 197, "ragged": True},
        ),
        # Rows with no visible key (window past the keys): the zero-output guard and
        # the start-tile clamp, on both grids.
        (
            "window_empty_rows_1024x512",
            {"seqlen_q": 1024, "seqlen_kv": 512, "sliding_window": 128},
        ),
        (
            "persist_window_empty_rows_1024x512",
            {
                "seqlen_q": 1024,
                "seqlen_kv": 512,
                "sliding_window": 128,
                "persistent": True,
                "num_persistent": 256,
            },
        ),
        # --- Sinks ---
        # Full-sink (use_sinks=True, no sliding window): D64 and D128, default + persistent
        ("sinks_d64_sq512", {"use_sinks": True, "head_size": 64}),
        ("sinks_d128_sq512", {"use_sinks": True}),
        (
            "persist_sinks_d64_sq512",
            {
                "persistent": True,
                "num_persistent": 256,
                "use_sinks": True,
                "head_size": 64,
            },
        ),
        (
            "persist_sinks_d128_sq512",
            {"persistent": True, "num_persistent": 256, "use_sinks": True},
        ),
        # SWA-sink (use_sinks=True + sliding_window>0): D64 and D128, default + persistent
        (
            "swa_sinks_d64_sq512",
            {"use_sinks": True, "sliding_window": 128, "head_size": 64},
        ),
        ("swa_sinks_d128_sq512", {"use_sinks": True, "sliding_window": 128}),
        (
            "persist_swa_sinks_d64_sq512",
            {
                "persistent": True,
                "num_persistent": 256,
                "use_sinks": True,
                "sliding_window": 128,
                "head_size": 64,
            },
        ),
        (
            "persist_swa_sinks_d128_sq512",
            {
                "persistent": True,
                "num_persistent": 256,
                "use_sinks": True,
                "sliding_window": 128,
            },
        ),
        # --- optional LSE output: the ordinary epilogue, the packed (varlen)
        # store guard, and the persistent epilogue with keyless rows + sink LSE.
        ("lse_default_sq512", {"emit_lse": True}),
        ("lse_varlen_sq512", {"emit_lse": True, "varlen": True}),
        (
            "lse_persist_empty_window_sinks_sq256",
            {
                "emit_lse": True,
                "persistent": True,
                "num_persistent": 256,
                "seqlen_q": 256,
                "seqlen_kv": 64,
                "sliding_window": 64,
                "use_sinks": True,
            },
        ),
    ):
        add(
            "attention_dense",
            f"attention_dense/gfx950/{_variant}",
            "gfx950",
            build_attention_dense("gfx950", **_over),
        )

    # Dense flash-attn prefill on gfx942: the block orders of both grids (the
    # library's gfx942 dense golden pins more variants, but outside this gate).
    _persist942 = {"persistent": True, "num_persistent": 304}
    for _variant, _over in (
        ("causal_sq512", {}),
        ("hqswz_sq512", {"nonpersist_decode": "hq_minor_swz"}),
        ("persist_causal_sq512", _persist942),
        ("persist_hqswz_sq512", {**_persist942, "persist_decode": "hq_minor_swz"}),
        ("persist_hkvmaj_sq512", {**_persist942, "persist_decode": "hkv_major"}),
        # Bottom-right causal: runtime diagonal offset on both grids, and the
        # diagonal split with the KV-segment split.
        ("bottom_right_sq512", {"seqlen_kv": 1024, "causal_bottom_right": True}),
        (
            "persist_bottom_right_sq512",
            {**_persist942, "seqlen_kv": 1024, "causal_bottom_right": True},
        ),
        (
            "bottom_right_dsplit_kvs4_sq512",
            {
                "seqlen_kv": 2048,
                "causal_bottom_right": True,
                "causal_diag_split": True,
                "kv_segments": 4,
                "block_m": 128,
            },
        ),
    ):
        add(
            "attention_dense",
            f"attention_dense/gfx942/{_variant}",
            "gfx942",
            build_attention_dense_gfx942(**_over),
        )

    # Attention band on the unified kernels: top-left alignment, full attention
    # (right_bound=-1), lookahead and non-causal / two-sided windows. These bodies have
    # no C++ mirror yet, so this golden is the only pin of their lowered IR.
    for _case, _kw in (
        ("top_left", dict(causal_top_left=True)),
        ("no_mask", dict(right_bound=-1)),
        ("rb16", dict(right_bound=16)),
        (
            "top_left_sw64",
            dict(causal_top_left=True, sliding_window=64),
        ),
        (
            "left_only_sw64",
            dict(causal_top_left=True, right_bound=-1, sliding_window=64),
        ),
        (
            "two_sided_sw64_rb16",
            dict(causal_top_left=True, right_bound=16, sliding_window=64),
        ),
    ):
        add(
            "attention",
            f"attention/gfx950/2d_fp16_d128_b64_{_case}",
            "gfx950",
            build_attention_2d(
                f"irhash_attn_950_2d_fp16_{_case}",
                "gfx950",
                head_size=128,
                block_size=64,
                num_query_heads=4,
                num_kv_heads=2,
                dtype="fp16",
                **_kw,
            ),
        )
    # The transposed-32x32 combo path (mask_once / mask_limit / peel sites).
    for _case, _kw in (
        ("top_left", dict(causal_top_left=True)),
        ("no_mask", dict(right_bound=-1)),
        ("rb16", dict(right_bound=16)),
    ):
        add(
            "attention",
            f"attention/gfx950/2d_fp16_d64_b32_gqa8_combo_{_case}",
            "gfx950",
            build_attention_2d(
                f"irhash_attn_950_2d_fp16_d64_combo_{_case}",
                "gfx950",
                head_size=64,
                block_size=32,
                num_query_heads=64,
                num_kv_heads=8,
                dtype="fp16",
                **_sink_combo_kw,
                **_kw,
            ),
        )
    for _case, _kw in (
        ("top_left", dict(causal_top_left=True)),
        ("no_mask", dict(right_bound=-1)),
        ("rb16", dict(right_bound=16)),
    ):
        add(
            "attention",
            f"attention/gfx950/3d_bf16_d128_b64_{_case}",
            "gfx950",
            build_attention_3d(
                f"irhash_attn_950_3d_bf16_{_case}",
                "gfx950",
                head_size=128,
                block_size=64,
                num_query_heads=4,
                num_kv_heads=2,
                dtype="bf16",
                num_segments=4,
                **_kw,
            ),
        )
    # Scalar oracle kernels (arch-neutral, lowered for gfx950).
    for _case, _kw in (
        ("top_left", dict(causal_top_left=True)),
        ("no_mask", dict(right_bound=-1)),
        ("sw16_rb8", dict(sliding_window=16, right_bound=8)),
    ):
        add(
            "attention",
            f"attention/gfx950/scalar2d_fp16_d128_{_case}",
            "gfx950",
            build_attention_scalar(
                f"irhash_attn_950_scalar2d_{_case}", "gfx950", **_kw
            ),
        )
    for _case, _kw in (
        ("top_left", dict(causal_top_left=True)),
        ("sw16_rb8", dict(sliding_window=16, right_bound=8)),
    ):
        add(
            "attention",
            f"attention/gfx950/scalar3d_fp16_d128_{_case}",
            "gfx950",
            build_attention_scalar(
                f"irhash_attn_950_scalar3d_{_case}",
                "gfx950",
                three_d=True,
                **_kw,
            ),
        )
    # The scalar split-3D reduce: empty-segment and empty-row guards.
    for _dtype in ("fp16", "bf16"):
        add(
            "attention",
            f"attention/gfx950/scalar_reduce_{_dtype}_d128_seg8",
            "gfx950",
            build_attention_scalar(
                f"irhash_attn_950_scalar_reduce_{_dtype}",
                "gfx950",
                reduce=True,
                dtype=_dtype,
            ),
        )
    # 40 query heads over 8 KV heads: the head -> KV-head magic division is by 5, so the
    # magic sum carries a real multiply-high term (a power-of-two ratio has multiplier 1).
    for _dim, _three_d in (("2d", False), ("3d", True)):
        add(
            "attention",
            f"attention/gfx950/scalar{_dim}_fp16_d128_q40_kv8",
            "gfx950",
            build_attention_scalar(
                f"irhash_attn_950_scalar{_dim}_q40_kv8",
                "gfx950",
                three_d=_three_d,
                num_query_heads=40,
                num_kv_heads=8,
            ),
        )

    # D256 bf16 prefill fast specs: the byte-sensitive slab-pad (gfx950) and
    # 4-warp GQA V-transpose (gfx942) routes.
    add(
        "attention_d256",
        "attention_d256/gfx950/pad_interleave",
        "gfx950",
        build_attention_d256_gfx950("gfx950"),
    )
    add(
        "attention_d256",
        "attention_d256/gfx942/4warp_gqa",
        "gfx942",
        build_attention_d256_gfx942("gfx942"),
    )
    add(
        "attention_d256",
        "attention_d256/gfx942/4warp_gqa_d192",
        "gfx942",
        build_attention_d256_gfx942("gfx942", head_size=192),
    )

    # gfx942 GQA head-fold (D128 sliding-window bf16). Registered SEPARATELY from
    # the D256 case above because that one early-returns into the lean D256 kernel
    # and never reaches the D128 fold body -- and the fold is default-ON for its
    # cohort with no other host-runnable guard.
    add(
        "attention_d128_swa",
        "attention_d128_swa/gfx942/4warp_gqa_fold",
        "gfx942",
        build_attention_d128_swa_fold_gfx942("gfx942"),
    )
    # Grouped GEMM: hand-authored dense grouped bf16 GEMM (gfx950 / CDNA4),
    # production lever defaults. Python-lowered only (no C++ engine mirror), so
    # this pins the Python emitter's IR; re-bless when its codegen changes.
    add(
        "grouped_gemm",
        "grouped_gemm/gfx950/m8192n1024k512e64",
        "gfx950",
        build_grouped_gemm_case(
            "irhash_grouped_gemm_950", "gfx950", 8192, 1024, 512, 64
        ),
    )

    # Chunkwise KDA: each emitted kernel plus the ABI/resource-sensitive
    # variants that change preprocessing, state pointers, or scan geometry.
    for _case_id, _kind, _over in (
        ("prep_default", "prep", {}),
        (
            "prep_raw",
            "prep",
            {
                "raw_inputs": True,
                "fuse_qk_l2norm": True,
                "fuse_gate": True,
                "fuse_beta_sigmoid": True,
                "has_dt_bias": True,
            },
        ),
        ("scan_default", "scan", {}),
        (
            "scan_h0_noht",
            "scan",
            {"has_initial_state": True, "store_final_state": False},
        ),
        ("scan_vs4", "scan", {"tile": {"block_size": 64}, "value_splits": 4}),
        ("fused_default", "fused", {}),
        (
            "fused_h0_noht",
            "fused",
            {"has_initial_state": True, "store_final_state": False},
        ),
    ):
        add(
            "kda_chunkwise",
            f"kda_chunkwise/gfx950/{_case_id}",
            "gfx950",
            build_kda_chunkwise_gfx950(_kind, "gfx950", **_over),
        )

    # gfx942 KDA: all three emitted kernels, state-sensitive ABI variants, and
    # DK64 fused coverage for the guarded partial Kt pass.
    for _case_id, _kind, _over in (
        ("prep_default", "prep", {}),
        ("scan_default", "scan", {}),
        (
            "scan_h0_noht",
            "scan",
            {"has_initial_state": True, "store_final_state": False},
        ),
        ("fused_default", "fused", {}),
        ("fused_dk64", "fused", {"head_k": 64}),
        (
            "fused_h0_noht",
            "fused",
            {"has_initial_state": True, "store_final_state": False},
        ),
    ):
        add(
            "kda_chunkwise",
            f"kda_chunkwise/gfx942/{_case_id}",
            "gfx942",
            build_kda_chunkwise_gfx942(_kind, "gfx942", **_over),
        )

    # TemporalHint.STREAMING on global_load_vN / global_store_vN; lowering
    # accepts it only on gfx942 / gfx950.
    for _arch in ("gfx942", "gfx950"):
        add(
            "nontemporal",
            f"nontemporal/{_arch}/copy_bf16x8",
            _arch,
            build_streaming_copy(_arch),
        )

    # K=128 f8f6f4 MFMA (gfx950): the unscaled fp8 hero atom and the MX op share
    # one declaration; the validity gate links both.
    for _kind in ("hero", "mx"):
        add(
            "mma_atom",
            f"mma_atom/gfx950/f8f6f4_16x16x128_{_kind}",
            "gfx950",
            build_f8f6f4_16x16x128(_kind),
        )
    for _fmt in ("bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"):
        add(
            "mma_atom",
            f"mma_atom/gfx950/f8f6f4_16x16x128_mx_{_fmt}",
            "gfx950",
            build_f8f6f4_16x16x128("mx", _fmt),
        )
    for _fmt in ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1"):
        add(
            "mma_atom",
            f"mma_atom/gfx950/f8f6f4_16x16x128_unscaled_{_fmt}",
            "gfx950",
            build_f8f6f4_16x16x128("unscaled", _fmt),
        )

    # warp_shuffle_xor_dpp (tile.dpp_xor): gfx9 CDNA picks the base-DPP
    # encoding per mask; a row_xmask target (gfx1201) keeps 0x160 | mask.
    for _arch in ("gfx942", "gfx950"):
        add(
            "crosslane",
            f"crosslane/{_arch}/warp_shuffle_xor_dpp_reduce_i32",
            _arch,
            build_warp_shuffle_xor_dpp_reduce(_arch, (1, 2, 7, 15)),
        )
        add(
            "crosslane",
            f"crosslane/{_arch}/warp_shuffle_xor_dpp_masks_f32",
            _arch,
            build_warp_shuffle_xor_dpp_masks_f32(_arch),
        )
    add(
        "crosslane",
        "crosslane/gfx1201/warp_shuffle_xor_dpp_reduce_i32",
        "gfx1201",
        build_warp_shuffle_xor_dpp_reduce("gfx1201", (1, 2, 4, 8)),
    )

    # Lightning indexer (DSA scoring, bf16) on gfx942 and gfx950 -- the kernel is
    # arch-neutral (no arch-specific atom, intrinsic, or LDS), so both lower the
    # same specs (scalar-v1 model-shaped cases: DeepSeek H_I=64, GLM H_I=32, a
    # small case; plus the MFMA body on 16-aligned shapes). The emitted IR is
    # currently identical across arches, so the two per-arch goldens hold the same
    # SHAs; the split is kept to catch future arch-specific divergence (fp8, wider
    # atoms) the moment it appears.
    for _arch in ("gfx942", "gfx950"):
        for _case_id, _over in (
            (
                "deepseek_hi64",
                {
                    "n_index_heads": 64,
                    "index_head_dim": 128,
                    "seqlen_q": 8,
                    "seqlen_k": 64,
                },
            ),
            (
                "glm_hi32",
                {
                    "n_index_heads": 32,
                    "index_head_dim": 128,
                    "seqlen_q": 8,
                    "seqlen_k": 64,
                },
            ),
            (
                "small",
                {
                    "n_index_heads": 4,
                    "index_head_dim": 16,
                    "seqlen_q": 8,
                    "seqlen_k": 32,
                    "block_size": 64,
                },
            ),
            (
                "mfma_deepseek_hi64",
                {
                    "n_index_heads": 64,
                    "index_head_dim": 128,
                    "seqlen_q": 16,
                    "seqlen_k": 64,
                    "body": "mfma",
                    "block_size": 64,
                },
            ),
            (
                "mfma_glm_hi32",
                {
                    "n_index_heads": 32,
                    "index_head_dim": 128,
                    "seqlen_q": 16,
                    "seqlen_k": 64,
                    "body": "mfma",
                    "block_size": 64,
                },
            ),
        ):
            add(
                "lightning_indexer",
                f"lightning_indexer/{_arch}/{_case_id}",
                _arch,
                build_lightning_indexer_case(_arch, **_over),
            )

    return out


# The golden stores one sub-document per llvm flavor, and the gate
# (check_golden) verifies all of them from any host: the flavor is an
# argument to lowering, so nothing about the running ROCm vintage limits which
# sub-documents can be checked. The same committed golden is therefore valid,
# and verified, on every ROCm vintage. GOLDEN_FLAVORS is LLVM_FLAVORS, so a new
# flavor fails the gate until the golden is re-blessed. The committed golden is
# a directory of per-family shards (see rocke.core.ir_golden.load_golden), so
# changes to different families touch different files and merge cleanly. The
# writer and command line are rocke.core.ir_golden's, shared with the library
# corpora; the functions below bind them to cases().


def run(
    ir_dir: Path | None = None,
    *,
    flavor: str,
    shards: set[str] | None = None,
):
    """Lower every case (or only those in ``shards``) at ``flavor``."""
    return run_golden_cases(
        cases(),
        flavor,
        lower=lower_case,
        shards=shards,
        ir_dir=ir_dir,
        ir_name=safe,
    )


def build_golden(shards: set[str] | None = None) -> dict:
    """Run every case (or only those in ``shards``) under each golden flavor
    and return the flavor-keyed doc."""
    return golden_document(lambda fl: run(flavor=fl, shards=shards))


def write_golden(golden_dir: Path, shards: set[str] | None = None) -> dict:
    """Bless ``golden_dir`` as one ``<shard>.json`` per family.

    With ``shards``, only those files are rewritten (a shard left with no cases
    is deleted); the rest are untouched. Without it every shard is rewritten
    and shard files for families that no longer exist are deleted. Returns the
    flavor-keyed doc that was written.
    """
    doc = build_golden(shards)
    write_golden_shards(golden_dir, doc, shards)
    return doc


def check_golden(golden_path: Path, flavor: str | None = None) -> list[str]:
    """Compare a fresh run against the golden sub-doc(s). Empty list == OK.

    With no ``flavor``, every flavor in :data:`GOLDEN_FLAVORS` is checked, not
    just the one this host autodetects; see :func:`rocke.core.ir_golden.check_golden`.
    """
    return _check_golden(golden_path, lambda fl: run(flavor=fl), flavor)


def main():
    golden_main(
        cases=cases,
        write_golden=write_golden,
        check_golden=check_golden,
        dump=lambda ir_dir, flavor: run(ir_dir, flavor=flavor or current_flavor()),
    )


if __name__ == "__main__":
    main()
