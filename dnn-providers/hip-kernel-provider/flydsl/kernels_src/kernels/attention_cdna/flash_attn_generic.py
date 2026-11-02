# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Vendored from: https://github.com/ROCm/FlyDSL
#   commit:   89ad52fbbb9e252396829a8a44b12e18b7acd970 (v0.3.4.1+19)
#   path:     kernels/attention/flash_attn_generic.py
#   compiler: flydsl 0.3.4 (pip wheel)
#
# CDNA (gfx90a, gfx942, gfx950) attention for the hipkernel:flydsl_sdpa_cdna engine.
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
    SINK_F32,
    SINK_OTHER_16BIT,
    SINK_QUERY_DTYPE,
    _is_gfx942_class,
    _load_seq_lens,
    _make_flash_attn_generic_traits,
    _no_keys,
    _waitcnt_vm_n,
    daz_denormal_attr,
    load_sink_logit,
)
from kernels.common.kernels_common import dtype_to_elem_type
from kernels.common import buffer_ops
from flydsl.expr.utils.arith import _to_raw as as_mlir_value


# The sliding-window argument's "no window" (modification 11): below any key's row
# offset, so no key is left of it.
_NO_WINDOW = -(1 << 30)
# Threads per merge workgroup (modification 14): one (batch, head, query) row each.
_MERGE_THREADS = 64


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
    has_bias=False,
    head_dim_v=None,
    qk_mask=False,
    split_kv=False,
    pack_gqa=False,
    dv_split=1,
    k_chunk=None,
    has_sink=False,
    bias_kstrided=False,
    has_dropout=False,
    v_mask=False,
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
            has_bias=has_bias,
            head_dim_v=head_dim_v,
            qk_mask=qk_mask,
            split_kv=split_kv,
            pack_gqa=pack_gqa,
            dv_split=dv_split,
            k_chunk=k_chunk,
            has_sink=has_sink,
            bias_kstrided=bias_kstrided,
            has_dropout=has_dropout,
            v_mask=v_mask,
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
            has_bias=has_bias,
            head_dim_v=head_dim_v,
            qk_mask=qk_mask,
            split_kv=split_kv,
            pack_gqa=pack_gqa,
            dv_split=dv_split,
            k_chunk=k_chunk,
            has_sink=has_sink,
            bias_kstrided=bias_kstrided,
            has_dropout=has_dropout,
            v_mask=v_mask,
        )
        _bs_threshold = (
            2048 * num_heads if _is_gfx942_class(gpu_arch) else 4096 * num_heads
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
    elif (
        dtype_str in ("f16", "bf16")
        and causal
        and head_dim == 128
        # The masked and narrow-V loads are on the N32 path (modification 13).
        and not qk_mask
        and head_dim_v in (None, head_dim)
        and dv_split == 1
    ):
        _validate_path = "N128"
    else:
        _validate_path = "N32"
    _validate_block_n_out = 128 if _validate_path == "N128" else 64
    _validate_has_lds_load_b128 = not _is_gfx942_class(gpu_arch)
    _validate_enable_gfx942_dma = _is_gfx942_class(gpu_arch) and (
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
    assert dtype_str in (
        "f16",
        "bf16",
        "f32",
    ), "flash_attn_func only supports f16, bf16 and f32"
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
        head_dim_v=head_dim_v,
        qk_mask=qk_mask,
        split_kv=split_kv,
        pack_gqa=pack_gqa,
        dv_split=dv_split,
        k_chunk=k_chunk,
        v_mask=v_mask,
    )
    # Chunked K stages on the plain load path (modification 18).
    _k_chunk_cols = traits.HEAD_DIM // traits.K_CHUNKS
    assert traits.K_CHUNKS == 1 or not (
        traits.ENABLE_GFX942_KV_GPFETCH
        or traits.ENABLE_DMA
        or traits.ENABLE_PREFETCH_3BUF
    ), "K chunks take the plain load path"
    # A packed build addresses Q, O and the LSE per row; the bias per head is not
    # (modification 15).
    assert not (pack_gqa and has_bias), "a packed GQA build takes no bias"
    # The bias read one key at a time, at a runtime key stride and element type
    # (modification 31), is a kind of bias build.
    assert has_bias or not bias_kstrided, "a strided bias build is a bias build"
    # A bias build differs from the plain one in its body only (modification 12), as a
    # sink build does (modification 19).
    _flash_attn_generic_cache_tag = (
        traits.cache_tag,
        bool(has_bias),
        bool(has_sink),
        bool(bias_kstrided),
        bool(has_dropout),
        # V's own runtime width (modification 40).
        bool(v_mask),
    )
    # Dropout (modification 33) draws on each query row's own position, so a build
    # that packs a GQA group into rows or splits the keys takes none.
    assert not (
        has_dropout and (pack_gqa or split_kv)
    ), "a dropout build is a prefill build"

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
    # O is [B, S, H, head_dim_v] in the launchers' BSHD layout (modification 13).
    _stride_token_o = traits.NUM_HEADS_Q * traits.HEAD_DIM_V
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
        # Upstream's cu_seqlens slots, which only a VARLEN build reads as such; a dense
        # build reads them as the per-batch query and key lengths of a padded launch,
        # INT32 [B], and the page-table stride slot, which only a PAGED build reads,
        # as its `varlen_flags` (modification 32).
        SeqLenQ: fx.Tensor,
        SeqLenKv: fx.Tensor,
        BlockTable: fx.Tensor,
        varlen_flags: fx.Int32,
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
        window_lo: fx.Int32,
        Bias: fx.Tensor,
        bias_stride0: fx.Int32,
        stride_bias_b: fx.Int32,
        stride_bias_h: fx.Int32,
        stride_v_n: fx.Int32,
        stride_v_b: fx.Int32,
        stride_v_h: fx.Int32,
        head_dim_qk: fx.Int32,
        WsO: fx.Tensor,
        WsLse: fx.Tensor,
        num_splits: fx.Int32,
        Sink: fx.Tensor,
        sink_stride: fx.Int32,
        sink_on: fx.Int32,
        stride_lse_b: fx.Int32,
        stride_lse_h: fx.Int32,
        stride_lse_n: fx.Int32,
        stride_bias_k: fx.Int32,
        bias_f32: fx.Int32,
        # Appended once for the changes that follow (modification 32): packed
        # sequences' row offsets, INT32 [B + 1], read by the prefill builds under
        # `VARLEN_PACKED` (modification 34); the Philox seed and offset tensors (8 bytes
        # each), the offset's by-value part as two words, the keep threshold and the
        # kept probabilities' scale (modification 33).
        RowOffQ: fx.Tensor,
        RowOffKv: fx.Tensor,
        DropoutSeed: fx.Tensor,
        DropoutOffset: fx.Tensor,
        dropout_offset2_lo: fx.Int32,
        dropout_offset2_hi: fx.Int32,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        # V's own group size, Hq / Hv, appended once (modification 42): read where V's base
        # takes its own head stride.
        v_group: fx.Int32,
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
                "v_b": stride_v_b,
                "v_h": stride_v_h,
            },
            window_lo=window_lo,
            stride_token_v=stride_v_n,
            head_dim_qk=head_dim_qk,
            num_splits=num_splits,
            ws_o=WsO,
            ws_lse=WsLse,
            # Read only by a sink build; every object takes the arguments.
            sink=Sink if const_expr(has_sink) else None,
            sink_stride=sink_stride,
            sink_on=sink_on,
            lse_strides=(stride_lse_b, stride_lse_h, stride_lse_n),
            # Per-batch lengths (modification 32): a dense build's, not a VARLEN or
            # PAGED build's, whose slots these are.
            varlen_flags=(
                None if const_expr(traits.VARLEN or traits.PAGED) else varlen_flags
            ),
            # V's own heads (modification 42) on a build that reads V through its own
            # strides at runtime head counts; the others take V's head as K's.
            v_group=v_group,
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
        # Packed sequences (modification 34) on the prefill builds; a split or packed-GQA
        # (decode) build reads no row offsets.
        ctx.init_sequence_lengths(
            SeqLenQ,
            SeqLenKv,
            row_offs=(
                None
                if const_expr(traits.SPLIT_KV or traits.PACK_GQA)
                else (RowOffQ, RowOffKv)
            ),
        )
        ctx.init_load_mapping()

        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        ctx.init_kv_batch_pointers()
        if const_expr(traits.PAGED):
            # A PAGED build's page-table stride (modification 32's slot).
            kv_gmem_to_lds.page_ids = GenericPageIdLoader(ctx, BlockTable, varlen_flags)
        if const_expr(traits.KV_VECTORIZED and traits.V_NOMAJOR_DMA):
            kv_gmem_to_lds.init_dma_nomajor()

        # Per-batch Q/O (and LSE when RETURN_LSE) descriptors, then K/V DMA
        # resources when DMA is enabled.
        ctx.init_descriptors(Q, O, LSE)
        if const_expr(has_bias):
            ctx.init_bias(
                Bias,
                bias_stride0,
                stride_bias_b,
                stride_bias_h,
                # Read only by a strided bias build (modification 31).
                kstrided=(
                    (stride_bias_k, bias_f32) if const_expr(bias_kstrided) else None
                ),
            )
        if const_expr(traits.ENABLE_DMA):
            kv_gmem_to_lds.init_dma()

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        ctx.init_q_row()
        q_b_packs = q_loader.load_all(ctx.q_rsrc, ctx.q_row)

        ctx.init_constants(sm_scale)
        if const_expr(has_dropout):
            # The Philox key, the row's first counter, the threshold and the scale
            # (modification 33); every other object takes the arguments unread.
            ctx.init_dropout(
                DropoutSeed,
                DropoutOffset,
                dropout_offset2_lo,
                dropout_offset2_hi,
                idropout_p,
                dropout_scale,
            )
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
            kv_gmem_to_lds.coop_dma_k(ctx.kv_lower, buf_id=0)
            rocdl.s_waitcnt(0)
        if const_expr(_pipe_k):
            _k0_vecs = kv_gmem_to_lds.coop_load_k_global(ctx.kv_lower)
            for _kb in range_constexpr(traits.NUM_BATCHES_KV):
                init_args.append(_k0_vecs[_kb])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            ctx.kv_lower, kv_upper, traits.BLOCK_N_OUT, init=init_args
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
                    # The layout the V store below expects: paired rows when it is the
                    # permuting one, as on the prefetch path (modification 25).
                    if const_expr(traits.V_PERM_TR):
                        _v_vecs_prefetch = kv_gmem_to_lds.coop_load_v_global_perm(
                            kv_start
                        )
                    else:
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
                        _q_pack = q_b_packs[ks]
                        s_acc_lo = gemm_helper.mfma_acc(_k_lo[ks], _q_pack, s_acc_lo)
                        s_acc_hi = gemm_helper.mfma_acc(_k_hi[ks], _q_pack, s_acc_hi)
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
                    # The K tile's further column chunks (modification 18): each waits for
                    # every wave to have read the last, overwrites it, and continues the
                    # scores.
                    for _kc in range_constexpr(1, traits.K_CHUNKS):
                        gpu.barrier()
                        kv_gmem_to_lds.coop_load_k(
                            kv_start, k_slot, col_off=_kc * _k_chunk_cols
                        )
                        gpu.barrier()
                        _k_lo, _k_hi = kv_lds_to_vgpr.load_k_packs(k_base)
                        s_acc_lo, s_acc_hi = gemm_helper.gemm1_accumulate(
                            kv_lds_to_vgpr,
                            _k_lo,
                            _k_hi,
                            q_b_packs,
                            kv_gmem_to_lds,
                            kv_start,
                            s_init=(s_acc_lo, s_acc_hi),
                            ks_off=_kc * traits.K_STEPS_QK,
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
                    if const_expr(has_dropout):
                        # After the row sum, so l and the LSE stay undropped
                        # (modification 33).
                        p_vals_lo, p_vals_hi = softmax_helper.apply_dropout(
                            p_vals_lo, p_vals_hi, kv_start
                        )

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
                        m_new_raw,
                        kv_start=kv_start,
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

    @flyc.kernel(known_block_size=[_MERGE_THREADS, 1, 1])
    def flash_attn_generic_merge_kernel(
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        WsO: fx.Tensor,
        WsLse: fx.Tensor,
        seq_len: fx.Int32,
        num_heads_q: fx.Int32,
        lse_on: fx.Int32,
        num_splits: fx.Int32,
        stride_o_b: fx.Int32,
        stride_o_n: fx.Int32,
        stride_o_h: fx.Int32,
        Sink: fx.Tensor,
        sink_stride: fx.Int32,
        sink_on: fx.Int32,
        stride_lse_b: fx.Int32,
        stride_lse_h: fx.Int32,
        stride_lse_n: fx.Int32,
        # Appended once (modification 32): the per-batch query lengths of a padded
        # launch and its `varlen_flags`, and packed sequences' row offsets, read by no
        # object yet.
        SeqLenQ: fx.Tensor,
        RowOffQ: fx.Tensor,
        varlen_flags: fx.Int32,
    ):
        """Combine the KV splits' partials of one (batch, head, query) row by their LSEs
        (modification 14): O = sum_i w_i O_i / sum_i w_i with w_i = exp(lse_i - max), and
        LSE = max + ln(sum_i w_i); a row no split reached gets O = 0 and LSE = -inf, and
        a row a split left NaN gets NaN (modification 29). A padded row -- past its
        batch's query length -- gets O = 0 and LSE = -inf (modification 32)."""
        elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        row = fx.Index(gpu.block_idx.x)  # (b * Hq + h) * Sq + s
        tid = fx.Index(gpu.thread_idx.x)
        sq = fx.Index(seq_len)
        hq = fx.Index(num_heads_q)
        ns = fx.Index(num_splits)
        s_idx = row % sq
        bh = row // sq
        h_idx = bh % hq
        b_idx = bh // hq
        # Whether the row is a real query position of its batch (modification 32): the
        # batch's own length, read when the launch is padded (the key length's slot is
        # the query's: only the query length matters here).
        len_q, _, _ = _load_seq_lens(
            varlen_flags, (SeqLenQ, SeqLenQ), b_idx, seq_len, seq_len
        )
        row_live = fx.Int32(s_idx) < len_q
        wsl = buffer_ops.create_buffer_resource(WsLse, max_size=True)
        wso = buffer_ops.create_buffer_resource(WsO, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)
        c_ninf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        # -inf (a split no key reached) is tested by its bits: the kernels build with
        # no-infs fast-math, under which `x == -inf` folds to false and exp(-inf - -inf)
        # turned a keyless row's 0 * 0 into NaN (modification 27).
        c_ninf_bits = fx.Int32(-8388608)  # 0xFF800000

        def is_ninf(x):
            return fx.Float32(x).bitcast(fx.Int32) == c_ninf_bits

        def lse_at(i):
            return fx.Float32(
                buffer_ops.buffer_load(
                    wsl,
                    as_mlir_value(fx.Int32(row * ns + i)),
                    vec_width=1,
                    dtype=fx.Float32,
                )
            )

        m = c_ninf
        for i, carry in range(fx.Index(0), ns, fx.Index(1), init=[m]):
            m = yield [fx.Float32(carry[0]).maximumf(lse_at(i))]
        m_all = fx.Float32(m)
        m_safe = is_ninf(m_all).select(c_zero, m_all)
        # Every column, the last pass partial where V is not a multiple of 64 wide
        # (modification 26).
        cols = -(-traits.HEAD_DIM_V // _MERGE_THREADS)
        partial = traits.HEAD_DIM_V % _MERGE_THREADS != 0
        init = [c_zero] + [c_zero for _ in range(cols)]
        res = init
        for i, carry in range(fx.Index(0), ns, fx.Index(1), init=init):
            li = lse_at(i)
            w = is_ninf(li).select(c_zero, fx.math.exp(li - m_safe))
            acc = []
            for j in range_constexpr(cols):
                d = tid + fx.Index(j * _MERGE_THREADS)
                if const_expr(partial and j == cols - 1):
                    d = (d < fx.Index(traits.HEAD_DIM_V)).select(d, fx.Index(0))
                ov = fx.Float32(
                    buffer_ops.buffer_load(
                        wso,
                        as_mlir_value(
                            fx.Int32((row * ns + i) * fx.Index(traits.HEAD_DIM_V) + d)
                        ),
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )
                # A split that reached no key of this row wrote no partial O; its slot
                # holds whatever the workspace held, and 0 * NaN would poison the row
                # (modification 27).
                acc.append(
                    fx.Float32(carry[1 + j]) + is_ninf(li).select(c_zero, w * ov)
                )
            res = yield [fx.Float32(carry[0]) + w] + acc
        l_all = fx.Float32(res[0])
        # "No keys" is l == +0, tested on the bits so that fast math cannot turn it
        # into a compare that is false for NaN: a split whose sum was NaN stored a NaN
        # LSE, which maximumf carries into m_all and every weight, so the row's O and
        # LSE come out NaN (modification 29).
        no_keys = _no_keys(l_all)
        inv = no_keys.select(c_zero, fx.Float32(1.0) / l_all)
        m_lse, l_lse = m_all, l_all
        # The attention sink's logit joins the denominator once, here rather than in
        # any split's partial (modification 19); a row no key reached gets LSE = sink.
        # It joins against the max over the splits' LSEs and the sink, m_top, which
        # scales O by keep = exp(m_all - m_top): exp(sink - m_all) alone overflows to
        # +inf once the sink leads by more than ~88.7 (modification 29).
        sink_val = c_ninf
        if const_expr(has_sink):
            # In f32 or a 16-bit dtype (modifications 36, 41).
            sink_val = load_sink_logit(
                Sink, sink_stride, sink_on, h_idx, elem_dtype, c_ninf
            )
            m_top = fx.Float32(m_all.maximumf(sink_val))
            keep = fx.Float32(fx.math.exp(m_all - m_top))
            l_top = l_all * keep + fx.math.exp(sink_val - m_top)
            inv = no_keys.select(c_zero, keep / l_top)
            m_lse, l_lse = m_top, l_top
        o_base = (
            b_idx * fx.Index(stride_o_b)
            + h_idx * fx.Index(stride_o_h)
            + s_idx * fx.Index(stride_o_n)
        )

        def store_o(value_of):
            for j in range_constexpr(cols):
                d = tid + fx.Index(j * _MERGE_THREADS)
                # The last pass's columns past V's width store nowhere.
                in_width = (
                    d < fx.Index(traits.HEAD_DIM_V)
                    if const_expr(partial and j == cols - 1)
                    else None
                )
                buffer_ops.buffer_store(
                    as_mlir_value(value_of(j)),
                    o_rsrc,
                    as_mlir_value(fx.Int32(o_base + d)),
                    mask=in_width,
                )

        # A padded row (modification 32) gets O = 0 and LSE = -inf, never the sink, by a
        # branch on the row (one per workgroup) rather than selects: fast math would merge
        # a select with the no-keys test and, assuming no NaN, lose a NaN row's NaN.
        if row_live:
            store_o(lambda j: (fx.Float32(res[1 + j]) * inv).to(elem_dtype))
        else:
            store_o(lambda j: c_zero.to(elem_dtype))
        if tid == fx.Index(0):
            if fx.Int32(lse_on) != fx.Int32(0):
                lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
                # Through the LSE's own strides (modification 20).
                lse_off = as_mlir_value(
                    fx.Int32(
                        b_idx * fx.Index(stride_lse_b)
                        + h_idx * fx.Index(stride_lse_h)
                        + s_idx * fx.Index(stride_lse_n)
                    )
                )
                if row_live:
                    lse = no_keys.select(sink_val, m_lse + fx.math.log(l_lse))
                    buffer_ops.buffer_store(
                        as_mlir_value(fx.Float32(lse)), lse_rsrc, lse_off
                    )
                else:
                    buffer_ops.buffer_store(
                        as_mlir_value(fx.Int32(-8388608)),  # 0xFF800000, -inf
                        lse_rsrc,
                        lse_off,
                    )

    @flyc.jit
    def launch_flash_attn_generic(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        SeqLenQ: fx.Tensor,
        SeqLenKv: fx.Tensor,
        BlockTable: fx.Tensor,
        varlen_flags: fx.Int32,
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
        window_lo: fx.Int32,
        Bias: fx.Tensor,
        bias_stride0: fx.Int32,
        stride_bias_b: fx.Int32,
        stride_bias_h: fx.Int32,
        stride_v_n: fx.Int32,
        stride_v_b: fx.Int32,
        stride_v_h: fx.Int32,
        head_dim_qk: fx.Int32,
        WsO: fx.Tensor,
        WsLse: fx.Tensor,
        num_splits: fx.Int32,
        Sink: fx.Tensor,
        sink_stride: fx.Int32,
        sink_on: fx.Int32,
        stride_lse_b: fx.Int32,
        stride_lse_h: fx.Int32,
        stride_lse_n: fx.Int32,
        stride_bias_k: fx.Int32,
        bias_f32: fx.Int32,
        RowOffQ: fx.Tensor,
        RowOffKv: fx.Tensor,
        DropoutSeed: fx.Tensor,
        DropoutOffset: fx.Tensor,
        dropout_offset2_lo: fx.Int32,
        dropout_offset2_hi: fx.Int32,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        v_group: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        grid_x = bs_idx * num_q_tiles * traits.NUM_HEADS_Q
        if const_expr(traits.PACK_GQA):
            # (batch, row tile, KV head): the group's rows packed (modification 15).
            group = fx.Index(num_heads_q) // fx.Index(num_heads_kv)
            row_tiles = (group * sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
            grid_x = bs_idx * row_tiles * fx.Index(num_heads_kv)
        grid_y = fx.Index(num_splits) if const_expr(traits.SPLIT_KV) else fx.Index(1)

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
            SeqLenQ,
            SeqLenKv,
            BlockTable,
            varlen_flags,
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
            window_lo,
            Bias,
            bias_stride0,
            stride_bias_b,
            stride_bias_h,
            stride_v_n,
            stride_v_b,
            stride_v_h,
            head_dim_qk,
            WsO,
            WsLse,
            num_splits,
            Sink,
            sink_stride,
            sink_on,
            stride_lse_b,
            stride_lse_h,
            stride_lse_n,
            stride_bias_k,
            bias_f32,
            RowOffQ,
            RowOffKv,
            DropoutSeed,
            DropoutOffset,
            dropout_offset2_lo,
            dropout_offset2_hi,
            idropout_p,
            dropout_scale,
            v_group,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    f"{int(flat_work_group_size)},{int(flat_work_group_size)}"
                    if const_expr(flat_work_group_size is not None)
                    else None
                ),
                "passthrough": passthrough_entries,
                # f32 (modification 35): exact fp32 keeps f32 denormal inputs (DESIGN F-d).
                "llvm.denormal_fpenv": (
                    daz_denormal_attr()
                    if const_expr(daz and traits.DTYPE_STR != "f32")
                    else None
                ),
            },
        ).launch(
            grid=(grid_x, grid_y, traits.DV_SPLIT),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )
        if const_expr(traits.SPLIT_KV):
            # The merge kernel, compiled into the same object (modification 14); the
            # dispatcher launches it only when num_splits > 1.
            flash_attn_generic_merge_kernel(
                O,
                LSE,
                WsO,
                WsLse,
                seq_len,
                num_heads_q,
                lse_on,
                num_splits,
                stride_o_b,
                stride_o_n,
                stride_o_h,
                Sink,
                sink_stride,
                sink_on,
                stride_lse_b,
                stride_lse_h,
                stride_lse_n,
                SeqLenQ,
                RowOffQ,
                varlen_flags,
            ).launch(
                grid=(bs_idx * sl_idx * fx.Index(num_heads_q), 1, 1),
                block=(_MERGE_THREADS, 1, 1),
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
        bias=None,
        bias_stride0=0,
        ws_o=None,
        ws_lse=None,
        num_splits=1,
        sink=None,
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
        # The bias (modification 12): read for its pointer; strides are arguments. A
        # build without one never reads the slot, so O stands in.
        bs = bias
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
                seq_len * _stride_token_o,
                _stride_token_o,
                traits.HEAD_DIM_V,
                skv * traits.STRIDE_TOKEN_KV,
                traits.HEAD_DIM,
                _NO_WINDOW,
                bs if bias is not None else Out,
                bias_stride0,
                0,
                0,
                traits.STRIDE_TOKEN_V,
                skv * traits.STRIDE_TOKEN_V,
                traits.HEAD_DIM_V,
                traits.HEAD_DIM,
                ws_o if ws_o is not None else Out,
                ws_lse if ws_lse is not None else Out,
                num_splits,
                sink if sink is not None else Out,
                1,
                (
                    (
                        SINK_F32
                        if sink.element_size() == 4
                        else (
                            SINK_QUERY_DTYPE
                            if sink.dtype == Q.dtype
                            else SINK_OTHER_16BIT
                        )
                    )
                    if sink is not None
                    else 0
                ),
                # Packed [B, H, Sq] LSE (modification 20).
                traits.NUM_HEADS_Q * seq_len,
                seq_len,
                1,
                # The bias's key stride and element type (modification 31): contiguous,
                # in the query's dtype.
                1,
                0,
                # Packed sequences' row offsets and the dropout arguments
                # (modification 32), read by no object yet.
                Out,
                Out,
                Out,
                Out,
                0,
                0,
                0,
                1.0,
                # V's group (modification 42): V's head is K's.
                traits.NUM_HEADS_Q // traits.NUM_HEADS_KV,
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
        bias=None,
        bias_stride0=0,
        ws_o=None,
        ws_lse=None,
        num_splits=1,
        sink=None,
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
        # The bias (modification 12): read for its pointer; strides are arguments. A
        # build without one never reads the slot, so O stands in.
        bs = bias
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
                seq_len * _stride_token_o,
                _stride_token_o,
                traits.HEAD_DIM_V,
                skv * traits.STRIDE_TOKEN_KV,
                traits.HEAD_DIM,
                _NO_WINDOW,
                bs if bias is not None else Out,
                bias_stride0,
                0,
                0,
                traits.STRIDE_TOKEN_V,
                skv * traits.STRIDE_TOKEN_V,
                traits.HEAD_DIM_V,
                traits.HEAD_DIM,
                ws_o if ws_o is not None else Out,
                ws_lse if ws_lse is not None else Out,
                num_splits,
                sink if sink is not None else Out,
                1,
                (
                    (
                        SINK_F32
                        if sink.element_size() == 4
                        else (
                            SINK_QUERY_DTYPE
                            if sink.dtype == Q.dtype
                            else SINK_OTHER_16BIT
                        )
                    )
                    if sink is not None
                    else 0
                ),
                # Packed [B, H, Sq] LSE (modification 20).
                traits.NUM_HEADS_Q * seq_len,
                seq_len,
                1,
                # The bias's key stride and element type (modification 31): contiguous,
                # in the query's dtype.
                1,
                0,
                # Packed sequences' row offsets and the dropout arguments
                # (modification 32), read by no object yet.
                Out,
                Out,
                Out,
                Out,
                0,
                0,
                0,
                1.0,
                # V's group (modification 42): V's head is K's.
                traits.NUM_HEADS_Q // traits.NUM_HEADS_KV,
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
            has_bias=has_bias,
            head_dim_v=head_dim_v,
            qk_mask=qk_mask,
            split_kv=split_kv,
            pack_gqa=pack_gqa,
            dv_split=dv_split,
            k_chunk=k_chunk,
            has_sink=has_sink,
            bias_kstrided=bias_kstrided,
            has_dropout=has_dropout,
            v_mask=v_mask,
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
            has_bias=has_bias,
            head_dim_v=head_dim_v,
            qk_mask=qk_mask,
            split_kv=split_kv,
            pack_gqa=pack_gqa,
            dv_split=dv_split,
            k_chunk=k_chunk,
            has_sink=has_sink,
            bias_kstrided=bias_kstrided,
            has_dropout=has_dropout,
            v_mask=v_mask,
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
