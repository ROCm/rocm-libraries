# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/FlyDSL
#   commit:   89ad52fbbb9e252396829a8a44b12e18b7acd970 (v0.3.4.1+19)
#   path:     kernels/attention/flash_attn_generic.py
#   compiler: flydsl 0.3.4 (pip wheel)
#
# CDNA (gfx942, gfx950) attention for the hipkernel:flydsl_sdpa_cdna engine.
# Modifications are listed in README.md beside this file.

"""Generic f16/bf16 flash-attention kernel builder for FlyDSL.

The kernel keeps Q, softmax probabilities, and O accumulators in MFMA32 register
layout, streams K/V through LDS, and performs the online softmax over KV blocks.
Q/K/V/O are flattened from BSHD, with GQA/MQA represented by a smaller KV-head
count.

Launches one block per Q tile and head:
``grid_x = batch * ceil(seq_len / BLOCK_M) * num_heads``. Tail Q/KV tiles are
handled by buffer-resource bounds and softmax masking, so dense seq_len only
needs to be positive rather than a multiple of 128.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len >= 1.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.attention_cdna.flash_attn_utils import (
    GenericFlashAttnContext,
    GenericGemmHelper,
    GenericKvGmemToLdsLoader,
    GenericKvLdsToVgprLoader,
    GenericPageIdLoader,
    GenericQLoader,
    GenericSoftmaxHelper,
    GenericStoreHelper,
    _make_flash_attn_generic_traits,
    _waitcnt_vm_n,
    daz_denormal_attr,
)
from kernels.common.kernels_common import dtype_to_elem_type


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    num_kv_heads=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cross_seqlen=False,
    varlen=False,
    paged=False,
    kv_cache_layout="linear",
    skip_kv_pad_mask=None,
    return_lse=False,
    extra_llvm_options=None,
    k_pad=None,
    vt_pad=None,
):
    """Build a generic f16/bf16 flash-attention launcher.

    ``num_heads`` is the Q/O head count. For GQA/MQA, pass a smaller
    ``num_kv_heads``; each group of ``num_heads // num_kv_heads`` Q heads shares
    one K/V head. Dense launches accept any positive ``seq_len``.
    """
    gpu_arch = get_hip_arch()

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    if dtype_str == "fp8":
        raise ValueError(
            "generic flash_attn_func supports f16/bf16 only; fp8 is routed by flash_attn_interface"
        )

    if block_m is None and num_heads >= 32:
        _launcher_m128 = build_flash_attn_func_module_primary(
            num_heads,
            head_dim,
            causal,
            dtype_str,
            sm_scale,
            waves_per_eu,
            flat_work_group_size=256,
            block_m=128,
            unsafe_fp_math=unsafe_fp_math,
            fast_fp_math=fast_fp_math,
            daz=daz,
            path_tag=path_tag,
            num_kv_heads=num_kv_heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cross_seqlen=cross_seqlen,
            varlen=varlen,
            paged=paged,
            kv_cache_layout=kv_cache_layout,
            skip_kv_pad_mask=skip_kv_pad_mask,
            return_lse=return_lse,
            extra_llvm_options=extra_llvm_options,
            k_pad=k_pad,
            vt_pad=vt_pad,
        )
        _launcher_m256 = build_flash_attn_func_module_primary(
            num_heads,
            head_dim,
            causal,
            dtype_str,
            sm_scale,
            waves_per_eu,
            flat_work_group_size=512,
            block_m=256,
            unsafe_fp_math=unsafe_fp_math,
            fast_fp_math=fast_fp_math,
            daz=daz,
            path_tag=path_tag,
            num_kv_heads=num_kv_heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cross_seqlen=cross_seqlen,
            varlen=varlen,
            paged=paged,
            kv_cache_layout=kv_cache_layout,
            skip_kv_pad_mask=skip_kv_pad_mask,
            return_lse=return_lse,
            extra_llvm_options=extra_llvm_options,
            k_pad=k_pad,
            vt_pad=vt_pad,
        )
        _bs_threshold = (
            2048 * num_heads if gpu_arch.startswith("gfx942") else 4096 * num_heads
        )

        def _extract_batch(args, kwargs):
            B = args[4] if len(args) > 4 else kwargs.get("batch_size", None)
            return B if isinstance(B, int) else 1

        def _auto_launch(*args, **kwargs):
            B = _extract_batch(args, kwargs)
            S = args[5] if len(args) > 5 else kwargs.get("seq_len", 128)
            bs = (B if isinstance(B, int) else 1) * (S if isinstance(S, int) else 128)
            if bs * num_heads >= _bs_threshold:
                return _launcher_m256(*args, **kwargs)
            return _launcher_m128(*args, **kwargs)

        if hasattr(_launcher_m128, "compile"):
            _auto_launch.compile = _launcher_m256.compile
        return _auto_launch

    _validate_block_m = 128 if block_m is None else block_m
    _validate_fwg = flat_work_group_size
    if _validate_fwg is None:
        _validate_fwg = 256 if _validate_block_m <= 128 else 512
    _validate_num_waves = _validate_fwg // 64
    _validate_rows_per_wave = _validate_block_m // _validate_num_waves
    if path_tag.upper() in ("N32", "N128"):
        _validate_path = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        _validate_path = "N128"
    else:
        _validate_path = "N32"
    _validate_block_n_out = 128 if _validate_path == "N128" else 64
    _validate_has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    _validate_enable_gfx942_dma = gpu_arch.startswith("gfx942") and (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_DMA", "0") == "1"
    )
    _validate_enable_dma = _validate_enable_gfx942_dma or (
        _validate_has_lds_load_b128
        and (
            _validate_path == "N128"
            or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1")
        )
    )

    assert (
        _validate_rows_per_wave == 32
    ), f"BLOCK_M/NUM_WAVES must be 32, got {_validate_rows_per_wave}"
    assert _validate_block_m % _validate_num_waves == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert _validate_fwg in (
        64,
        128,
        256,
        512,
    ), f"flat_work_group_size must be 64, 128, 256, or 512, got {_validate_fwg}"
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert _validate_block_n_out % 64 == 0

    paged = bool(paged)
    if paged and kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"generic paged kernel supports linear/vectorized kv_cache_layout, got {kv_cache_layout!r}"
        )
    if paged and _validate_enable_dma:
        raise NotImplementedError(
            "generic paged kernel requires the non-DMA path (build with path_tag='N32')"
        )

    traits = _make_flash_attn_generic_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        gpu_arch,
        causal=causal,
        dtype_str=dtype_str,
        flat_work_group_size=flat_work_group_size,
        block_m=block_m,
        path_tag=path_tag,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        paged=paged,
        kv_cache_layout=kv_cache_layout,
        waves_per_eu=waves_per_eu,
        daz=daz,
        fast_fp_math=fast_fp_math,
        unsafe_fp_math=unsafe_fp_math,
        sm_scale=sm_scale,
        skip_kv_pad_mask=skip_kv_pad_mask,
        return_lse=return_lse,
        k_pad=k_pad,
        vt_pad=vt_pad,
    )
    _flash_attn_generic_cache_tag = traits.cache_tag

    def _extract_seq_len(args, kwargs):
        """Return the launch-time seq_len as int, or None if not statically known."""
        S = args[5] if len(args) > 5 else kwargs.get("seq_len", None)
        try:
            return int(S)
        except (TypeError, ValueError):
            return None

    def _guard_seqlen(_dispatched):
        """Enforce the only correctness floor (seq_len >= 1). A symbolic/non-int
        seq_len is let through; dense routing is a perf policy, not a bound."""

        def _guarded(*args, **kwargs):
            S_int = _extract_seq_len(args, kwargs)
            if S_int is not None and S_int < 1:
                raise ValueError(f"flash_attn_func: seq_len must be >= 1, got {S_int}.")
            return _dispatched(*args, **kwargs)

        if hasattr(_dispatched, "compile"):
            _guarded.compile = _dispatched.compile
        return _guarded

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)
    # The launchers' default for the runtime scale argument (modification 2).
    _default_sm_scale = sm_scale

    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        sm_scale: fx.Float32,
        num_heads_q: fx.Int32,
        num_heads_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        lse_on: fx.Int32,
        diag_offset: fx.Int32,
        stride_q_b: fx.Int32,
        stride_q_h: fx.Int32,
        stride_o_b: fx.Int32,
        stride_o_n: fx.Int32,
        stride_o_h: fx.Int32,
        stride_kv_b: fx.Int32,
        stride_kv_h: fx.Int32,
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _flash_attn_generic_cache_tag
        ctx = GenericFlashAttnContext(
            traits,
            K,
            V,
            seq_len,
            seq_len_kv,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            stride_token_q=stride_q_n,
            stride_token_kv=stride_kv_n,
            lse_on=lse_on,
            diag_offset=diag_offset,
            view_strides={
                "q_b": stride_q_b,
                "q_h": stride_q_h,
                "o_b": stride_o_b,
                "o_n": stride_o_n,
                "o_h": stride_o_h,
                "kv_b": stride_kv_b,
                "kv_h": stride_kv_h,
            },
        )
        ctx.init_types_and_pointers()
        gemm_helper = GenericGemmHelper(ctx)
        softmax_helper = GenericSoftmaxHelper(ctx)
        kv_gmem_to_lds = GenericKvGmemToLdsLoader(ctx)
        kv_lds_to_vgpr = GenericKvLdsToVgprLoader(ctx)
        q_loader = GenericQLoader(ctx)
        store_helper = GenericStoreHelper(ctx)

        ctx.init_sequence_indices()
        ctx.init_lds_view(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_block_mapping()
        ctx.init_sequence_lengths(CuSeqQ, CuSeqKv)
        ctx.init_load_mapping()

        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        ctx.init_kv_batch_pointers()
        if const_expr(traits.PAGED):
            kv_gmem_to_lds.page_ids = GenericPageIdLoader(
                ctx, BlockTable, block_table_stride
            )
        if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
            kv_gmem_to_lds.init_dma_nomajor()

        # Per-batch Q/O (and LSE when RETURN_LSE) descriptors, then K/V DMA
        # resources when DMA is enabled.
        ctx.init_descriptors(Q, O, LSE)
        if const_expr(traits.ENABLE_DMA):
            kv_gmem_to_lds.init_dma()

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        ctx.init_q_row()
        q_b_packs = q_loader.load_all(ctx.q_rsrc, ctx.q_row)

        ctx.init_constants(sm_scale)
        ctx.init_kv_bounds()
        kv_upper = ctx.kv_upper

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF
        _pipe_k = (
            traits.ENABLE_GFX942_KV_GPFETCH
            and not _use_dma_dbuf
            and not traits.ENABLE_PREFETCH_3BUF
            and traits.N_SUBTILES == 1
        )
        init_args = [ctx.c_neg_inf, ctx.c_zero_f]
        for _ in range_constexpr(traits.D_CHUNKS):
            init_args.append(ctx.c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(fx.Index(0))
            kv_gmem_to_lds.coop_dma_k(fx.Index(0), buf_id=0)
            rocdl.s_waitcnt(0)
        if const_expr(_pipe_k):
            _k0_vecs = kv_gmem_to_lds.coop_load_k_global(fx.Index(0))
            for _kb in range_constexpr(traits.NUM_BATCHES_KV):
                init_args.append(_k0_vecs[_kb])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, traits.BLOCK_N_OUT, init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            _cur_buf_id = (
                inner_iter_args[2 + traits.D_CHUNKS] if _use_dma_dbuf else None
            )
            _carried_k_vecs = (
                [
                    inner_iter_args[2 + traits.D_CHUNKS + _kb]
                    for _kb in range_constexpr(traits.NUM_BATCHES_KV)
                ]
                if _pipe_k
                else None
            )
            preload_k_count = (
                traits.NUM_PREFETCH_K
                if traits.NUM_PREFETCH_K < traits.N_SUBTILES
                else traits.N_SUBTILES
            )

            if const_expr(traits.ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = (
                        traits.CK_LDS_SEQ[pre_k % len(traits.CK_LDS_SEQ)]
                        % traits.NUM_PREFETCH_K
                    )
                    pre_k_start = kv_block_start + pre_k * traits.BLOCK_N
                    if const_expr(traits.ENABLE_DMA):
                        kv_gmem_to_lds.coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        kv_gmem_to_lds.coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(traits.ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(traits.N_SUBTILES):
                kv_start = kv_block_start + kv_sub * traits.BLOCK_N

                if const_expr(traits.ENABLE_PREFETCH_3BUF):
                    k_slot = (
                        traits.CK_LDS_SEQ[kv_sub % len(traits.CK_LDS_SEQ)]
                        % traits.NUM_PREFETCH_K
                    )
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = fx.Index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = fx.Index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < traits.N_SUBTILES):
                        kv_gmem_to_lds.coop_dma_k(
                            kv_block_start + (kv_sub + 1) * traits.BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + fx.Index(traits.BLOCK_N_OUT)
                        _has_next = _next_kv < kv_upper
                        _prefetch_next_k = kv_gmem_to_lds.coop_dma_k
                        if _has_next:
                            _prefetch_next_k(_next_kv, _next_k_buf_id)
                    rocdl.sched_barrier(0)
                    k_base = kv_gmem_to_lds.k_buf_base(_k_buf_id)
                else:
                    k_slot = 0
                    if const_expr(_pipe_k):
                        _waitcnt_vm_n(0)
                        kv_gmem_to_lds.coop_store_k_lds(_carried_k_vecs, k_slot)
                        rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                        _next_kv_start = kv_block_start + fx.Index(traits.BLOCK_N_OUT)
                        _next_k_vecs = kv_gmem_to_lds.coop_load_k_global(_next_kv_start)
                    elif const_expr(traits.ENABLE_GFX942_KV_GPFETCH):
                        _kv_k_vecs = kv_gmem_to_lds.coop_load_k_global(kv_start)
                        kv_gmem_to_lds.coop_store_k_lds(_kv_k_vecs, k_slot)
                        rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    elif const_expr(traits.ENABLE_GFX942_DMA):
                        kv_gmem_to_lds.coop_dma_k(kv_start, k_slot)
                        rocdl.s_waitcnt(0)
                    else:
                        kv_gmem_to_lds.coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = kv_gmem_to_lds.k_buf_base(k_slot)

                if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    kv_gmem_to_lds.coop_dma_v_nomajor(kv_start, 0)
                elif const_expr(
                    not traits.ENABLE_GFX942_KV_GPFETCH
                    and (
                        not traits.USE_HW_TR
                        or (not traits.ENABLE_DMA and not traits.ENABLE_PREFETCH_3BUF)
                    )
                ):
                    _v_vecs_prefetch = kv_gmem_to_lds.coop_load_v_global(kv_start)

                # ==== GEMM1: S = K @ Q^T. Bulk-read K packs, DMA-prefetch V, pipeline MFMAs ====
                _k_lo, _k_hi = kv_lds_to_vgpr.load_k_packs(k_base)
                if const_expr(traits.ENABLE_GFX942_KV_GPFETCH):
                    s_acc_lo = ctx.c_zero_v16f32
                    s_acc_hi = ctx.c_zero_v16f32
                    for ks in range_constexpr(traits.K_STEPS_QK):
                        if const_expr(ks == 0):
                            if const_expr(traits.V_PERM_TR):
                                _v_vecs_prefetch = (
                                    kv_gmem_to_lds.coop_load_v_global_perm(kv_start)
                                )
                            else:
                                _v_vecs_prefetch = kv_gmem_to_lds.coop_load_v_global(
                                    kv_start
                                )
                        s_acc_lo = gemm_helper.mfma_acc(
                            _k_lo[ks], q_b_packs[ks], s_acc_lo
                        )
                        s_acc_hi = gemm_helper.mfma_acc(
                            _k_hi[ks], q_b_packs[ks], s_acc_hi
                        )
                        if const_expr(
                            ks + traits.QK_PREFETCH_DEPTH < traits.K_STEPS_QK
                        ):
                            (
                                _k_lo[ks + traits.QK_PREFETCH_DEPTH],
                                _k_hi[ks + traits.QK_PREFETCH_DEPTH],
                            ) = kv_lds_to_vgpr.load_k_pack_at(
                                ks + traits.QK_PREFETCH_DEPTH
                            )
                else:
                    if const_expr(
                        traits.ENABLE_DMA
                        and not traits.ENABLE_PREFETCH_3BUF
                        and not traits.USE_HW_TR
                    ):
                        kv_gmem_to_lds.coop_dma_v(kv_start, 0)
                        rocdl.sched_barrier(0)
                    s_acc_lo, s_acc_hi = gemm_helper.gemm1_accumulate(
                        kv_lds_to_vgpr,
                        _k_lo,
                        _k_hi,
                        q_b_packs,
                        kv_gmem_to_lds,
                        kv_start,
                    )

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo, s_raw_hi = softmax_helper.split_scores(s_acc_lo, s_acc_hi)
                s_raw_lo, s_raw_hi = softmax_helper.apply_kv_mask(
                    s_raw_lo, s_raw_hi, kv_start
                )
                if const_expr(
                    traits.ENABLE_GFX942_KV_GPFETCH
                    and traits.DTYPE_STR == "bf16"
                    and not traits.USE_K16
                ):
                    m_new_raw, corr, neg_scaled_max = (
                        softmax_helper.online_softmax_stats(
                            m_running, s_raw_lo, s_raw_hi
                        )
                    )
                    o_accs, corr_vec = softmax_helper.rescale_o_accs(o_accs, corr)
                else:
                    m_new_raw, l_new, corr, p_vals_lo, p_vals_hi = (
                        softmax_helper.online_softmax(
                            m_running, l_running, s_raw_lo, s_raw_hi
                        )
                    )
                    o_accs, corr_vec = softmax_helper.rescale_o_accs(o_accs, corr)

                if const_expr(
                    traits.ENABLE_PREFETCH_3BUF
                    and (kv_sub + preload_k_count) < traits.N_SUBTILES
                ):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * traits.BLOCK_N
                    next_k_slot = (
                        traits.CK_LDS_SEQ[next_k_sub % len(traits.CK_LDS_SEQ)]
                        % traits.NUM_PREFETCH_K
                    )
                    if const_expr(traits.ENABLE_DMA):
                        kv_gmem_to_lds.coop_dma_k(next_k_start, next_k_slot)
                    else:
                        kv_gmem_to_lds.coop_load_k(next_k_start, next_k_slot)

                if const_expr(traits.ENABLE_PREFETCH_3BUF):
                    v_slot = (
                        traits.CK_LDS_SEQ[kv_sub % len(traits.CK_LDS_SEQ)]
                        % traits.NUM_PREFETCH_V
                    )
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    kv_gmem_to_lds.coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(traits.ENABLE_DMA):
                    v_base = kv_gmem_to_lds.v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                elif const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
                    v_slot = 0
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = kv_gmem_to_lds.v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    if const_expr(traits.V_PERM_TR):
                        kv_gmem_to_lds.coop_store_v_lds_perm(_v_vecs_prefetch, v_slot)
                    else:
                        kv_gmem_to_lds.coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()

                # ==== Build P packs, then GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi ====
                if const_expr(
                    traits.ENABLE_GFX942_KV_GPFETCH
                    and traits.DTYPE_STR == "bf16"
                    and not traits.USE_K16
                ):
                    o_accs, l_new = softmax_helper.gemm2_gpfetch_fused(
                        gemm_helper,
                        kv_lds_to_vgpr,
                        o_accs,
                        corr_vec,
                        corr,
                        l_running,
                        s_raw_lo,
                        s_raw_hi,
                        neg_scaled_max,
                        v_base,
                    )
                else:
                    p_packs_lo = softmax_helper.build_p_packs(p_vals_lo)
                    p_packs_hi = softmax_helper.build_p_packs(p_vals_hi)
                    o_accs = gemm_helper.gemm2_pv(
                        kv_lds_to_vgpr, o_accs, p_packs_lo, p_packs_hi, v_base, corr_vec
                    )

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(traits.N_SUBTILES % 2 == 1):
                    _yield_args.append(fx.Index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            if const_expr(_pipe_k):
                for _kb in range_constexpr(traits.NUM_BATCHES_KV):
                    _yield_args.append(_next_k_vecs[_kb])
            loop_results = yield _yield_args

        # ---- Normalize and store O (128-bit buffer_store_dwordx4) ----
        store_helper.finalize_o(loop_results)

    @flyc.jit
    def launch_flash_attn_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        sm_scale: fx.Float32,
        num_heads_q: fx.Int32,
        num_heads_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        lse_on: fx.Int32,
        diag_offset: fx.Int32,
        stride_q_b: fx.Int32,
        stride_q_h: fx.Int32,
        stride_o_b: fx.Int32,
        stride_o_n: fx.Int32,
        stride_o_h: fx.Int32,
        stride_kv_b: fx.Int32,
        stride_kv_h: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        grid_x = bs_idx * num_q_tiles * traits.NUM_HEADS_Q

        passthrough_entries = (
            [
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_generic_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            seq_len,
            seq_len_kv,
            CuSeqQ,
            CuSeqKv,
            BlockTable,
            block_table_stride,
            sm_scale,
            num_heads_q,
            num_heads_kv,
            stride_q_n,
            stride_kv_n,
            lse_on,
            diag_offset,
            stride_q_b,
            stride_q_h,
            stride_o_b,
            stride_o_n,
            stride_o_h,
            stride_kv_b,
            stride_kv_h,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
                "llvm.denormal_fpenv": (
                    daz_denormal_attr() if const_expr(daz) else None
                ),
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers were measured with ROCm/llvm-project `felix/tune_fmha`;
    # other LLVM revisions usually leave a few percent of peak throughput on the table.
    _llvm_opts = {
        "enable-post-misched": os.getenv("FLYDSL_LLVM_ENABLE_POST_MISCHED", "0") == "1",
        "lsr-drop-solution": True,
        # A caller's additions (modification 7); none by default.
        **(extra_llvm_options or {}),
    }
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": _llvm_opts,
    }

    def _launch(
        Q,
        K,
        V,
        Out,
        batch_size,
        seq_len,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        block_table=None,
        block_table_stride=0,
        seq_len_kv=None,
        lse=None,
        sm_scale=None,
        stream=None,
    ):
        # Dense/non-paged pass the output tensor as a placeholder for the unused
        # cu_seqlens / block_table slots; the kernel only reads them under
        # const_expr(VARLEN) / const_expr(PAGED).
        cq = cu_seqlens_q if cu_seqlens_q is not None else Out
        ck = cu_seqlens_kv if cu_seqlens_kv is not None else Out
        bt = block_table if block_table is not None else Out
        # RETURN_LSE stores fp32 LSE, so Out is not a safe placeholder for a missing lse.
        if return_lse and lse is None:
            raise ValueError(
                "flash_attn_generic was built with return_lse=True but no `lse` tensor was "
                "provided; pass an fp32 [batch, num_heads, seq_len] LSE tensor."
            )
        ls = lse if lse is not None else Out
        skv = seq_len if seq_len_kv is None else seq_len_kv
        scale = _default_sm_scale if sm_scale is None else sm_scale
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_generic(
                Q,
                K,
                V,
                Out,
                ls,
                cq,
                ck,
                bt,
                block_table_stride,
                batch_size,
                seq_len,
                skv,
                scale,
                traits.NUM_HEADS_Q,
                traits.NUM_HEADS_KV,
                traits.STRIDE_TOKEN_Q,
                traits.STRIDE_TOKEN_KV,
                1,
                skv - seq_len,
                seq_len * traits.STRIDE_TOKEN_Q,
                traits.HEAD_DIM,
                seq_len * traits.STRIDE_TOKEN_Q,
                traits.STRIDE_TOKEN_Q,
                traits.HEAD_DIM,
                skv * traits.STRIDE_TOKEN_KV,
                traits.HEAD_DIM,
                fx.Stream(stream),
            )

    def _compile(
        Q,
        K,
        V,
        Out,
        batch_size,
        seq_len,
        seq_len_kv=None,
        lse=None,
        sm_scale=None,
        stream=None,
    ):
        if return_lse and lse is None:
            raise ValueError(
                "flash_attn_generic was built with return_lse=True but no `lse` tensor was "
                "provided; pass an fp32 [batch, num_heads, seq_len] LSE tensor."
            )
        ls = lse if lse is not None else Out
        skv = seq_len if seq_len_kv is None else seq_len_kv
        scale = _default_sm_scale if sm_scale is None else sm_scale
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_generic,
                Q,
                K,
                V,
                Out,
                ls,
                Out,
                Out,
                Out,
                0,
                batch_size,
                seq_len,
                skv,
                scale,
                traits.NUM_HEADS_Q,
                traits.NUM_HEADS_KV,
                traits.STRIDE_TOKEN_Q,
                traits.STRIDE_TOKEN_KV,
                1,
                skv - seq_len,
                seq_len * traits.STRIDE_TOKEN_Q,
                traits.HEAD_DIM,
                seq_len * traits.STRIDE_TOKEN_Q,
                traits.STRIDE_TOKEN_Q,
                traits.HEAD_DIM,
                skv * traits.STRIDE_TOKEN_KV,
                traits.HEAD_DIM,
                fx.Stream(stream),
            )

    _launch.compile = _compile

    def _wrap_pad_mask_dispatch(_inner):
        if skip_kv_pad_mask is not None or varlen or paged or causal:
            return _inner

        _launch_skip = build_flash_attn_func_module_primary(
            num_heads,
            head_dim,
            causal,
            dtype_str,
            sm_scale,
            waves_per_eu,
            flat_work_group_size=flat_work_group_size,
            block_m=block_m,
            unsafe_fp_math=unsafe_fp_math,
            fast_fp_math=fast_fp_math,
            daz=daz,
            path_tag=path_tag,
            num_kv_heads=num_kv_heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cross_seqlen=cross_seqlen,
            varlen=varlen,
            paged=paged,
            kv_cache_layout=kv_cache_layout,
            skip_kv_pad_mask=True,
            return_lse=return_lse,
            extra_llvm_options=extra_llvm_options,
            k_pad=k_pad,
            vt_pad=vt_pad,
        )
        _launch_mask = build_flash_attn_func_module_primary(
            num_heads,
            head_dim,
            causal,
            dtype_str,
            sm_scale,
            waves_per_eu,
            flat_work_group_size=flat_work_group_size,
            block_m=block_m,
            unsafe_fp_math=unsafe_fp_math,
            fast_fp_math=fast_fp_math,
            daz=daz,
            path_tag=path_tag,
            num_kv_heads=num_kv_heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cross_seqlen=cross_seqlen,
            varlen=varlen,
            paged=paged,
            kv_cache_layout=kv_cache_layout,
            skip_kv_pad_mask=False,
            return_lse=return_lse,
            extra_llvm_options=extra_llvm_options,
            k_pad=k_pad,
            vt_pad=vt_pad,
        )

        def _pad_dispatch(*args, **kwargs):
            S_int = _extract_seq_len(args, kwargs)
            if S_int is not None and S_int % 64 == 0:
                return _launch_skip(*args, **kwargs)
            return _launch_mask(*args, **kwargs)

        if hasattr(_inner, "compile"):
            _pad_dispatch.compile = _inner.compile
        return _pad_dispatch

    _launch = _wrap_pad_mask_dispatch(_launch)

    if block_m is None:
        return _guard_seqlen(_launch)
    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
