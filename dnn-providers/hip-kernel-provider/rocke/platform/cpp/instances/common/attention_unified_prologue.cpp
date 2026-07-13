// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_attention_unified_attention_unified_prologue.c -- SHARED PROLOGUE +
 * ABI bucket for the C99 port of the scalar unified-attention kernel builders
 * (rocke/instances/common/attention_unified.py).
 *
 * Implements the phase functions every build_unified_attention_* body threads
 * BEFORE its kind-specific online-softmax loop + epilogue:
 *
 *   rocke_attn_unified_ctx_init            -- zero/populate ctx, derive geometry
 *   rocke_attn_unified_declare_scalar_params (Python _declare_scalar_attn_params)
 *   rocke_attn_unified_emit_find_seq_idx   (Python _emit_find_seq_idx_scan +
 *                                         the 2D inline linear cu_q scan)
 *   rocke_attn_unified_emit_prologue       -- grid ids + per-seq geometry +
 *                                         SSA constants + (3D) segment span
 *
 * The shared descriptor builders (_q_descriptor, _paged_kv_descriptor,
 * _segm_descriptors) and the magic-div geometry (_magic_div / _magic_div_mod)
 * are ALREADY ported in the selector helper TU; this prologue delegates to them
 * via helper_helper_rocke.instances.common.attention_unified_selectors.h so the
 * builder-call sequence stays byte-identical.
 *
 * The builder-call sequence in each function reproduces the corresponding Python
 * span rocke_b_* op-for-op, in order.
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually.
 */
#include <math.h>
#include <string.h>

#include "rocke/instance_attention_unified_internal.h"

/* ===================================================================== *
 *  rocke_attn_unified_ctx_init
 *
 *  Python: the head of each build_unified_attention_*:
 *      p = spec.problem
 *      if p.dtype not in ("fp16", "bf16"): raise ValueError(...)   (2D only)
 *      dtype = spec.dtype_ir
 *  plus deriving num_queries_per_kv (= p.num_query_heads // num_kv_heads, which
 *  raises on a non-divisible ratio) and max_blocks
 *  (= ceil(max_seqlen_k / block_size)).
 *
 *  Zero-inits the ctx, copies the problem/spec slice into it, mirrors `p` into
 *  the selector-helper problem shape (ctx->sel_p) so the ported emit/descriptor
 *  helpers can be called, then validates. On a non-divisible head ratio or an
 *  unsupported dtype, sets b's sticky error and returns false.
 * ===================================================================== */
bool rocke_attn_unified_ctx_init(rocke_attn_unified_build_ctx_t* ctx,
                                 rocke_ir_builder_t* b,
                                 rocke_attn_unified_kind_t kind,
                                 const rocke_unified_attention_problem_t* p,
                                 const rocke_type_t* dtype,
                                 int num_segments)
{
    if(ctx == NULL || b == NULL || p == NULL)
    {
        return false;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->b = b;
    ctx->kind = kind;
    ctx->p = p;
    ctx->dtype = dtype;
    ctx->num_segments = num_segments;
    ctx->kernel = rocke_ir_builder_kernel(b);

    /* Mirror `p` into the selector-helper problem shape so the ported emit /
     * physical-block-and-token / descriptor helpers (which take a
     * rocke_unified_attn_problem_t*) can be called without re-deriving it. Only
     * the fields those helpers read are semantically required, but we carry the
     * full overlap for parity. */
    {
        rocke_unified_attn_problem_t* sp = &ctx->sel_p;
        sp->total_q = p->total_q;
        sp->num_seqs = p->num_seqs;
        sp->num_query_heads = p->num_query_heads;
        sp->num_kv_heads = p->num_kv_heads;
        sp->head_size = p->head_size;
        sp->block_size = p->block_size;
        sp->max_seqlen_q = p->max_seqlen_q;
        sp->max_seqlen_k = p->max_seqlen_k;
        sp->dtype = p->dtype;
        sp->q_dtype = p->q_dtype;
        sp->sliding_window = p->sliding_window;
        sp->softcap = p->softcap;
        sp->use_sinks = p->use_sinks;
        sp->use_alibi = p->use_alibi;
        sp->use_qq_bias = p->use_qq_bias;
        sp->use_fp8 = p->use_fp8;
        sp->num_sms = p->num_sms;
        sp->num_kv_blocks = p->num_kv_blocks;
    }

    /* Python (2D only): if p.dtype not in ("fp16", "bf16"): raise ValueError.
     * The 3D/reduce builders take dtype = spec.dtype_ir directly (F16/BF16) and
     * do not re-validate the string. Mirror that: gate only the 2D kind. */
    if(kind == ROCKE_ATTN_UNIFIED_2D)
    {
        const char* dt = p->dtype;
        bool ok = dt != NULL && (strcmp(dt, "fp16") == 0 || strcmp(dt, "bf16") == 0);
        if(!ok)
        {
            /* Python: raise ValueError("scalar 2D kernel currently supports
             * fp16/bf16"). The dtype is resolved by the glue driver before any
             * IR is emitted; on an unsupported dtype the build aborts without
             * emitting ops (matching the sibling 2D builder's NULL-on-bad-dtype
             * gate). Return false; the driver returns NULL to the caller. */
            return false;
        }
    }

    /* num_queries_per_kv = num_query_heads // num_kv_heads (Python @property
     * raises ValueError on a non-divisible ratio). The selector helper sets the
     * builder's sticky error and returns 0 in that case. */
    ctx->num_queries_per_kv = rocke_unified_attn_num_queries_per_kv(b, &ctx->sel_p);
    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }

    /* max_blocks = ceil(max_seqlen_k / block_size). Python folds this inline as
     * (p.max_seqlen_k + p.block_size - 1) // p.block_size when computing the
     * block_tables row stride. */
    if(p->block_size > 0)
    {
        ctx->max_blocks = (p->max_seqlen_k + p->block_size - 1) / p->block_size;
    }
    else
    {
        ctx->max_blocks = 0;
    }

    return true;
}

/* ===================================================================== *
 *  rocke_attn_unified_declare_scalar_params
 *
 *  Python: _declare_scalar_attn_params(b, dtype_ir) -> dict.
 *
 *  Declares the shared scalar-attention ABI prefix in EXACTLY the Python order
 *  (param declaration order is load-bearing -- it fixes the kernel arg ABI):
 *      query / key / value, sink, block_tables, seq_lens, alibi, qq_bias, cu_q,
 *      then the scale / k_scale / v_scale f32 scalars.
 *  Fills ctx->abi_*. (The reduce kernel uses its own narrower declaration --
 *  rocke_attn_unified_declare_reduce_params -- so this is NOT called for reduce.)
 * ===================================================================== */
void rocke_attn_unified_declare_scalar_params(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_type_t* dt = ctx->dtype;
    const rocke_type_t* dtp = rocke_ptr_type(b, dt, "global");
    const rocke_type_t* i32p = rocke_ptr_type(b, rocke_i32(), "global");
    const rocke_type_t* f32p = rocke_ptr_type(b, rocke_f32(), "global");

    rocke_param_opts_t o;

    /* query = b.param("query_ptr", PtrType(dtype,"global"),
     *                 noalias=True, readonly=True, align=16) */
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->abi_query = rocke_b_param(b, "query_ptr", dtp, &o);

    /* key = b.param("key_cache_ptr", ..., noalias=True, readonly=True, align=16) */
    ctx->abi_key = rocke_b_param(b, "key_cache_ptr", dtp, &o);

    /* value = b.param("value_cache_ptr", ..., noalias=True, readonly=True, align=16) */
    ctx->abi_value = rocke_b_param(b, "value_cache_ptr", dtp, &o);

    /* sink = b.param("sink_ptr", PtrType(dtype,"global"), readonly=True, align=16) */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 16;
    o.align_set = true;
    ctx->abi_sink = rocke_b_param(b, "sink_ptr", dtp, &o);

    /* block_tables = b.param("block_tables_ptr", PtrType(I32,"global"),
     *                        readonly=True, align=4) */
    memset(&o, 0, sizeof(o));
    o.readonly = true;
    o.readonly_set = true;
    o.align = 4;
    o.align_set = true;
    ctx->abi_block_tables = rocke_b_param(b, "block_tables_ptr", i32p, &o);

    /* seq_lens = b.param("seq_lens_ptr", PtrType(I32,"global"),
     *                    readonly=True, align=4) */
    ctx->abi_seq_lens = rocke_b_param(b, "seq_lens_ptr", i32p, &o);

    /* alibi = b.param("alibi_slopes_ptr", PtrType(F32,"global"),
     *                 readonly=True, align=4) */
    ctx->abi_alibi = rocke_b_param(b, "alibi_slopes_ptr", f32p, &o);

    /* qq_bias = b.param("qq_bias_ptr", PtrType(F32,"global"),
     *                   readonly=True, align=4) */
    ctx->abi_qq_bias = rocke_b_param(b, "qq_bias_ptr", f32p, &o);

    /* cu_q = b.param("query_start_len_ptr", PtrType(I32,"global"),
     *               readonly=True, align=4) */
    ctx->abi_cu_q = rocke_b_param(b, "query_start_len_ptr", i32p, &o);

    /* scale = b.param("scale", F32); k_scale = ...; v_scale = ... */
    ctx->abi_scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
    ctx->abi_k_scale = rocke_b_param(b, "k_scale", rocke_f32(), NULL);
    ctx->abi_v_scale = rocke_b_param(b, "v_scale", rocke_f32(), NULL);
}

/* ===================================================================== *
 *  rocke_attn_unified_emit_find_seq_idx
 *
 *  Python (2D, inline): scan cu_q for the largest i with cu_q[i] <= q_tok:
 *      seq_init = b.const_i32(0)
 *      scan = b.scf_for_iter(0, num_seqs, 1, [("seq_idx", seq_init)], iv="si")
 *      with scan as (si, (seq_idx,)):
 *          start_i  = b.global_load_i32(cu_q, si)
 *          le       = b.cmp_le(start_i, q_tok)
 *          next_seq = b.select(le, si, seq_idx)
 *          b.scf_yield(next_seq)
 *      seq_idx = scan.results[0]
 *
 *  Python (3D): _emit_find_seq_idx_scan(b, cu_q, q_tok, num_seqs) delegates to
 *  binary_search_seq_idx (per_token, 32 iterations). That helper is not yet
 *  exported to the C layer; per the helper-header note the 3D scan falls back to
 *  the SAME inline linear cu_q scan (numerically identical seq_idx) until
 *  rocke_binary_search_seq_idx lands. So both kinds emit the inline linear scan.
 *
 *  Fills ctx->seq_idx.
 * ===================================================================== */
void rocke_attn_unified_emit_find_seq_idx(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* seq_init = rocke_b_const_i32(b, 0);
    rocke_iter_arg_t iter[1];
    rocke_for_t scan;

    iter[0].name = "seq_idx";
    iter[0].init = seq_init;

    /* scan = b.scf_for_iter(0, num_seqs, 1, [("seq_idx", seq_init)], iv="si")
     * NOTE: the Python loop upper bound is the `num_seqs` PARAM Value (stored in
     * ctx->num_seqs by the driver), not the int spec field. */
    {
        rocke_value_t* scan_lb = rocke_b_const_i32(b, 0);
        rocke_value_t* scan_step = rocke_b_const_i32(b, 1);
        scan = rocke_b_scf_for_iter(
            b, scan_lb, ctx->num_seqs, scan_step, iter, 1, "si", false, true);
    }
    rocke_b_region_enter(b, scan.body);
    {
        rocke_value_t* si = scan.iv;
        rocke_value_t* seq_idx = scan.iter_vars[0];
        rocke_value_t* start_i = rocke_b_global_load_i32(b, ctx->abi_cu_q, si, 4);
        rocke_value_t* le = rocke_b_cmp_le(b, start_i, ctx->q_tok);
        rocke_value_t* next_seq = rocke_b_select(b, le, si, seq_idx);
        rocke_value_t* yields[1];
        yields[0] = next_seq;
        rocke_b_scf_yield(b, yields, 1);
    }
    rocke_b_region_leave(b);

    /* seq_idx = scan.results[0] */
    ctx->seq_idx = rocke_op_result(b, scan.op);
}

/* ===================================================================== *
 *  rocke_attn_unified_emit_prologue
 *
 *  The common prologue after the ABI prefix declaration: grid ids, the cu_q
 *  bounds / q_len / query_pos / kv_len / context_len / kv_head geometry, the
 *  SSA constants, and (for 3D) the segm_idx/dim split + segment span.
 *
 *  Mirrors the prologue spans of build_unified_attention_2d (2620-2665) and
 *  build_unified_attention_3d (2833-2865). The 2D/reduce grid uses
 *  block_id_z() directly as `dim`; the 3D grid reads zd = block_id_z() and
 *  splits (segm_idx, dim) = magic_div_mod(zd, head_size).
 *
 *  PRECONDITION: ctx->num_seqs (the param Value), ctx->abi_* and (for 2D)
 *  ctx->abi_sink must already be set by the declare phase, and the grid-id
 *  ordering matters: in Python the seq-idx scan reads q_tok, which is computed
 *  here first.
 * ===================================================================== */
void rocke_attn_unified_emit_prologue(rocke_attn_unified_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;

    /* ---- grid ids + thread predicate ---- */
    /* q_tok = b.block_id_x(); q_head = b.block_id_y() */
    ctx->q_tok = rocke_b_block_id_x(b);
    ctx->q_head = rocke_b_block_id_y(b);

    if(ctx->kind == ROCKE_ATTN_UNIFIED_3D)
    {
        /* zd = b.block_id_z(); (segm_idx, dim) = _magic_div_mod(b, zd, head_size) */
        ctx->zd = rocke_b_block_id_z(b);
        if(!rocke_unified_attn_magic_div_mod(
               b, ctx->zd, ctx->p->head_size, &ctx->segm_idx, &ctx->dim))
        {
            return;
        }
    }
    else
    {
        /* 2D / reduce: dim = b.block_id_z() */
        ctx->dim = rocke_b_block_id_z(b);
    }

    /* tid = b.thread_id_x(); active = b.cmp_eq(tid, const_i32(0)) */
    ctx->tid = rocke_b_thread_id_x(b);
    ctx->active = rocke_b_cmp_eq(b, ctx->tid, rocke_b_const_i32(b, 0));

    /* ---- seq-idx scan ---- */
    rocke_attn_unified_emit_find_seq_idx(ctx);

    /* ---- per-sequence geometry ---- */
    /* cu_start = b.global_load_i32(cu_q, seq_idx) */
    ctx->cu_start = rocke_b_global_load_i32(b, ctx->abi_cu_q, ctx->seq_idx, 4);
    /* cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, 1)) */
    ctx->cu_stop = rocke_b_global_load_i32(
        b, ctx->abi_cu_q, rocke_b_add(b, ctx->seq_idx, rocke_b_const_i32(b, 1)), 4);
    /* q_len = b.sub(cu_stop, cu_start) */
    ctx->q_len = rocke_b_sub(b, ctx->cu_stop, ctx->cu_start);
    /* query_pos = b.sub(q_tok, cu_start) */
    ctx->query_pos = rocke_b_sub(b, ctx->q_tok, ctx->cu_start);
    /* kv_len = b.global_load_i32(seq_lens, seq_idx) */
    ctx->kv_len = rocke_b_global_load_i32(b, ctx->abi_seq_lens, ctx->seq_idx, 4);
    /* context_len = b.sub(kv_len, q_len) */
    ctx->context_len = rocke_b_sub(b, ctx->kv_len, ctx->q_len);
    /* kv_head = _magic_div(b, q_head, num_queries_per_kv) */
    ctx->kv_head = rocke_unified_attn_magic_div(b, ctx->q_head, ctx->num_queries_per_kv);

    /* ---- 3D segment span (Python 2848-2858) ---- */
    if(ctx->kind == ROCKE_ATTN_UNIFIED_3D)
    {
        int seg_block = ctx->num_segments * ctx->p->block_size;
        rocke_value_t* seg_stop_raw;
        rocke_value_t* lt;

        /* tiles_per_segment = _magic_div(b,
         *     b.add(kv_len, const_i32(num_segments*block_size - 1)),
         *     num_segments*block_size) */
        ctx->tiles_per_segment = rocke_unified_attn_magic_div(
            b, rocke_b_add(b, ctx->kv_len, rocke_b_const_i32(b, seg_block - 1)), seg_block);

        /* seg_start = b.mul(segm_idx, b.mul(tiles_per_segment, const_i32(block_size)))
         * (Python re-emits b.mul(tiles_per_segment, const_i32(block_size)) twice;
         * no CSE -- reproduce both muls inline so the op stream matches.) */
        ctx->seg_start = rocke_b_mul(
            b,
            ctx->segm_idx,
            rocke_b_mul(b, ctx->tiles_per_segment, rocke_b_const_i32(b, ctx->p->block_size)));

        /* seg_stop_i = b.mul(b.add(segm_idx, 1),
         *                    b.mul(tiles_per_segment, const_i32(block_size))) */
        seg_stop_raw = rocke_b_mul(
            b,
            rocke_b_add(b, ctx->segm_idx, rocke_b_const_i32(b, 1)),
            rocke_b_mul(b, ctx->tiles_per_segment, rocke_b_const_i32(b, ctx->p->block_size)));

        /* seg_stop_i = b.select(b.cmp_lt(seg_stop_i, kv_len), seg_stop_i, kv_len) */
        lt = rocke_b_cmp_lt(b, seg_stop_raw, ctx->kv_len);
        ctx->seg_stop_i = rocke_b_select(b, lt, seg_stop_raw, ctx->kv_len);
    }

    /* ---- SSA constants ---- */
    /* neg_inf = const_f32(-inf); zero_f = const_f32(0.0); one_f = const_f32(1.0);
     * rcp_ln2 = const_f32(1.4426950408889634) */
    ctx->neg_inf = rocke_b_const_f32(b, -INFINITY);
    ctx->zero_f = rocke_b_const_f32(b, 0.0);
    if(ctx->kind == ROCKE_ATTN_UNIFIED_2D)
    {
        /* one_f is only materialised in the 2D prologue (3D inits l to zero). */
        ctx->one_f = rocke_b_const_f32(b, 1.0);
    }
    ctx->rcp_ln2 = rocke_b_const_f32(b, ROCKE_UNIFIED_ATTN_RCP_LN2);

    /* ---- online-softmax loop init (kind-dependent) ----
     * 2D: if use_sinks: sink_h = global_load(sink, q_head, dtype, align=2);
     *         init_m = fmul(cast_to_f32(sink_h), rcp_ln2); init_l = one_f
     *     else: init_m = neg_inf; init_l = one_f
     *     init_acc = zero_f
     * 3D: init_m = neg_inf; init_l = zero_f; init_acc = zero_f */
    if(ctx->kind == ROCKE_ATTN_UNIFIED_2D)
    {
        if(ctx->p->use_sinks)
        {
            ctx->sink_h = rocke_b_global_load(b, ctx->abi_sink, ctx->q_head, ctx->dtype, 2);
            ctx->init_m = rocke_b_fmul(b, rocke_b_cast_to_f32(b, ctx->sink_h), ctx->rcp_ln2);
            ctx->init_l = ctx->one_f;
        }
        else
        {
            ctx->init_m = ctx->neg_inf;
            ctx->init_l = ctx->one_f;
        }
        ctx->init_acc = ctx->zero_f;
    }
    else if(ctx->kind == ROCKE_ATTN_UNIFIED_3D)
    {
        ctx->init_m = ctx->neg_inf;
        ctx->init_l = ctx->zero_f;
        ctx->init_acc = ctx->zero_f;
    }

    /* ---- shared descriptors threaded by all three kernels ----
     * q_desc      = _q_descriptor(p)              (2D/reduce)
     * kv_desc     = _paged_kv_descriptor(p)       (2D/3D)
     * (ml_desc, out_desc) = _segm_descriptors(p, num_segments) (3D/reduce)
     * The 2D body builds q_desc + kv_desc here; the 3D/reduce bodies build their
     * segm descriptors in the epilogue/loop phases. We populate every descriptor
     * the kind needs so phase functions only read ctx fields. */
    ctx->q_desc = rocke_unified_attn_q_descriptor(b, &ctx->sel_p);
    ctx->kv_desc = rocke_unified_attn_paged_kv_descriptor(&ctx->sel_p);
    if(ctx->kind == ROCKE_ATTN_UNIFIED_3D)
    {
        if(!rocke_unified_attn_segm_descriptors(
               b, &ctx->sel_p, ctx->num_segments, &ctx->ml_desc, &ctx->out_desc))
        {
            return;
        }
    }
}
