/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_fmha_bwd.c -- C99 port of ck_dsl/instances/common/fmha_bwd.py.
 *
 * Byte-identical builder-call sequence vs the Python build_fmha_bwd: same ops,
 * same order, same operands. The warp-distributed scalar body (one wave64 warp
 * per CTA) recomputes P from the saved (M, L), runs two K-loops, and
 * accumulates dQ / dK / dV through f32 global atomics. No MFMA atoms are
 * emitted, so the IR is arch-portable; `arch` only drives is_valid_spec.
 *
 * Reused C ports:
 *   - ckc/helper_ck_dsl.helpers.attention.h  (apply_attention_mask,
 *                                             warp_xor_reduce_sum)
 *   - ckc/helper_ck_dsl.helpers.io.h         (load_lane_slice_f32)
 *   - ckc/helper_ck_dsl.helpers.spec.h       (kernel_name_join)
 *   - ckc/helper_ck_dsl.core.arch.h          (ArchTarget)
 *   - ckc/helper_ck_dsl.instances.common._fmha_common.h (FmhaKernelBuilder,
 *                                             validate_common_spec)
 *   - ckc/instance_gemm_universal.h          (lower-to-.ll convenience pattern)
 */
#include "ckc/instance_fmha_bwd.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/arch_target.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/helper_ck_dsl.helpers.attention.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_warp_body.h" /* WARP_SIZE */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  spec defaults + kernel_name
 * ===================================================================== */

ckc_fmha_bwd_spec_t ckc_fmha_bwd_spec_default(ckc_fmha_common_spec_t common,
                                              int seqlen_q,
                                              int seqlen_k)
{
    ckc_fmha_bwd_spec_t s;
    s.common = common;
    s.seqlen_q = seqlen_q;
    s.seqlen_k = seqlen_k;
    s.name = "ck_dsl_fmha_bwd";
    s.use_mfma_body = false;
    s.output_grad_dtype = "f32";
    return s;
}

/* FmhaBwdSpec.kernel_name(): kernel_name_join(name, H{..}, HQ{..}, HK{..},
 * dtype, Q{..}, K{..}, mask_mode). */
ckc_status_t ckc_fmha_bwd_kernel_name(const ckc_fmha_bwd_spec_t* spec,
                                      char* out,
                                      size_t out_cap)
{
    char h[32], hq[32], hk[32], q[32], k[32];
    const char* parts[7];
    const char* mask_name;

    if (spec == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }

    snprintf(h, sizeof(h), "H%d", spec->common.shape.head_size);
    snprintf(hq, sizeof(hq), "HQ%d", spec->common.shape.num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", spec->common.shape.num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->seqlen_k);

    mask_name = ckc_fmha_mask_mode_name(spec->common.mask_mode);
    if (mask_name == NULL)
    {
        mask_name = "";
    }

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = spec->common.dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = mask_name;

    return ckc_kernel_name_join(spec->name, parts, 7, NULL, NULL, 0, out, out_cap,
                                NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

bool ckc_fmha_bwd_is_valid_spec(const ckc_fmha_bwd_spec_t* spec,
                                const char* arch,
                                char* reason,
                                size_t reason_cap)
{
    const ckc_arch_target_t* target;
    const char* why = NULL;
    ckc_arena_t arena;
    bool ok;

    if (spec == NULL)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "null spec");
        }
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch); KeyError -> (False, str(e)). */
    target = ckc_arch_target_from_gfx(arch);
    if (target == NULL)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "%s", arch);
        }
        return false;
    }

    /* ok, why = validate_common_spec(spec.common) */
    if (ckc_arena_init(&arena, 0) != 0)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "oom");
        }
        return false;
    }
    ok = ckc_fmha_validate_common_spec(&arena, &spec->common, &why);
    if (!ok)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "%s", why ? why : "invalid common spec");
        }
        ckc_arena_destroy(&arena);
        return false;
    }
    ckc_arena_destroy(&arena);

    if (spec->seqlen_q <= 0 || spec->seqlen_k <= 0)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap,
                     "seqlen_q / seqlen_k must be > 0 (got %d, %d)",
                     spec->seqlen_q, spec->seqlen_k);
        }
        return false;
    }
    if (spec->common.shape.head_size % target->wave_size != 0)
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap,
                     "fmha_bwd warp body needs head_size %% %d == 0 (got %d) on %s",
                     target->wave_size, spec->common.shape.head_size, arch);
        }
        return false;
    }
    if (target->wave_size > ckc_arch_max_threads_per_block(target))
    {
        if (reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap,
                     "wave_size %d > max_threads_per_block %d on %s",
                     target->wave_size, ckc_arch_max_threads_per_block(target),
                     arch);
        }
        return false;
    }

    if (reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "ok");
    }
    return true;
}

/* ===================================================================== *
 *  _declare_params -- the bwd kernel ABI (shared between build + sig).
 * ===================================================================== */

static void fmha_bwd_declare_params(ckc_fmha_kernel_builder_t* kb)
{
    /* kb.add_tensor("Q"/"K"/"V"/"dO", readonly=True) */
    ckc_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    ckc_fmha_kernel_builder_add_tensor(kb, "dO", NULL, true, false, 16);
    /* kb.add_ptr("M_saved"/"L_saved", dtype="f32", readonly=True) */
    ckc_fmha_kernel_builder_add_ptr(kb, "M_saved", "f32", true, 4);
    ckc_fmha_kernel_builder_add_ptr(kb, "L_saved", "f32", true, 4);
    /* kb.add_ptr("dQ"/"dK"/"dV", dtype="f32", readonly=False) */
    ckc_fmha_kernel_builder_add_ptr(kb, "dQ", "f32", false, 4);
    ckc_fmha_kernel_builder_add_ptr(kb, "dK", "f32", false, 4);
    ckc_fmha_kernel_builder_add_ptr(kb, "dV", "f32", false, 4);
    /* kb.add_scalar(...) */
    ckc_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "scale_inv", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    /* kb.add_strides("q", "k", "v") */
    {
        const char* qkv[3] = {"q", "k", "v"};
        ckc_fmha_kernel_builder_add_strides(kb, qkv, 3);
    }
    /* kb.add_strides("do") */
    {
        const char* doarr[1] = {"do"};
        ckc_fmha_kernel_builder_add_strides(kb, doarr, 1);
    }
    /* kb.add_scalar("stride_dq_token"/"stride_dk_token"/"stride_dv_token", "i32") */
    ckc_fmha_kernel_builder_add_scalar(kb, "stride_dq_token", "i32");
    ckc_fmha_kernel_builder_add_scalar(kb, "stride_dk_token", "i32");
    ckc_fmha_kernel_builder_add_scalar(kb, "stride_dv_token", "i32");
}

/* Map a FmhaCommonSpec mask mode to the attention-helper mask enum. The bwd
 * body only ever passes mask_mode in {none, causal, sliding_window} (the only
 * modes is_valid_spec / validate_common_spec admit here); alibi / custom are
 * not reachable for the warp body but are mapped to NONE defensively. */
static ckc_attn_mask_mode_t fmha_bwd_attn_mask(ckc_fmha_mask_mode_t m)
{
    switch (m)
    {
        case CKC_FMHA_MASK_CAUSAL:
            return CKC_ATTN_MASK_CAUSAL;
        case CKC_FMHA_MASK_SLIDING_WINDOW:
            return CKC_ATTN_MASK_SLIDING_WINDOW;
        case CKC_FMHA_MASK_NONE:
        case CKC_FMHA_MASK_ALIBI:
        case CKC_FMHA_MASK_CUSTOM:
        default:
            return CKC_ATTN_MASK_NONE;
    }
}

/* ===================================================================== *
 *  build_fmha_bwd
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_fmha_bwd(ckc_fmha_kernel_builder_t* kb,
                                     const ckc_fmha_bwd_spec_t* spec,
                                     const char* arch)
{
    return ckc::guard_builder(ckc_fmha_kernel_builder_builder(kb), [&]() -> ckc_kernel_def_t* {
        char reason[512];
        char kname[1024];
        const ckc_fmha_common_spec_t* s;
        int H, ept;
        ckc_attn_mask_mode_t attn_mask;
        ckc_ir_builder_t* b;

        /* registered Values */
        ckc_value_t *Q, *K, *V, *dO;
        ckc_value_t *M_saved, *L_saved, *dQ, *dK, *dV;
        ckc_value_t *scale_log2, *scale_inv, *seqlen_k;
        ckc_value_t *q_token, *head_idx, *kv_head_idx;
        ckc_value_t *tid, *lane_d_base, *q_row, *do_row;
        ckc_value_t *ml_row, *m_qt, *l_qt, *inv_l, *zero_f;
        ckc_value_t **q_lane, **do_lane;
        ckc_value_t *rowsum_dp;
        ckc_for_t k_loop_1, k_loop_2;
        int kk;

        if (kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if (arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); raise ValueError on reject. */
        if (!ckc_fmha_bwd_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            /* Initialise a builder so the caller can read the sticky error. */
            if (ckc_fmha_bwd_kernel_name(spec, kname, sizeof(kname)) != CKC_OK)
            {
                snprintf(kname, sizeof(kname), "ck_dsl_fmha_bwd");
            }
            if (ckc_fmha_kernel_builder_init(kb, kname, &spec->common) == CKC_OK)
            {
                ckc_i_set_err(&kb->b, CKC_ERR_VALUE, "invalid fmha_bwd spec: %s",
                              reason);
            }
            return NULL;
        }

        s = &spec->common;
        H = s->shape.head_size;
        if (H % CKC_FMHA_WARP_SIZE != 0)
        {
            if (ckc_fmha_bwd_kernel_name(spec, kname, sizeof(kname)) != CKC_OK)
            {
                snprintf(kname, sizeof(kname), "ck_dsl_fmha_bwd");
            }
            if (ckc_fmha_kernel_builder_init(kb, kname, &spec->common) == CKC_OK)
            {
                ckc_i_set_err(&kb->b, CKC_ERR_VALUE,
                              "fmha_bwd warp body needs head_size %% %d == 0; got %d",
                              CKC_FMHA_WARP_SIZE, H);
            }
            return NULL;
        }
        ept = H / CKC_FMHA_WARP_SIZE;
        attn_mask = fmha_bwd_attn_mask(s->mask_mode);

        /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
        if (ckc_fmha_bwd_kernel_name(spec, kname, sizeof(kname)) != CKC_OK)
        {
            return NULL;
        }
        if (ckc_fmha_kernel_builder_init(kb, kname, s) != CKC_OK)
        {
            return NULL;
        }

        /* kb.block_size(WARP_SIZE) */
        ckc_fmha_kernel_builder_block_size(kb, CKC_FMHA_WARP_SIZE);
        /* _declare_params(kb) */
        fmha_bwd_declare_params(kb);
        /* kb.decode_grid() */
        ckc_fmha_kernel_builder_decode_grid(kb, -1, false, &q_token, &head_idx,
                                            &kv_head_idx);
        /* b = kb.builder */
        b = ckc_fmha_kernel_builder_builder(kb);

        /* tensor / ptr / scalar / coord lookups */
        Q = ckc_fmha_kernel_builder_tensor(kb, "Q");
        K = ckc_fmha_kernel_builder_tensor(kb, "K");
        V = ckc_fmha_kernel_builder_tensor(kb, "V");
        dO = ckc_fmha_kernel_builder_tensor(kb, "dO");
        M_saved = ckc_fmha_kernel_builder_ptr(kb, "M_saved");
        L_saved = ckc_fmha_kernel_builder_ptr(kb, "L_saved");
        dQ = ckc_fmha_kernel_builder_ptr(kb, "dQ");
        dK = ckc_fmha_kernel_builder_ptr(kb, "dK");
        dV = ckc_fmha_kernel_builder_ptr(kb, "dV");
        scale_log2 = ckc_fmha_kernel_builder_scalar(kb, "scale_log2");
        scale_inv = ckc_fmha_kernel_builder_scalar(kb, "scale_inv");
        seqlen_k = ckc_fmha_kernel_builder_scalar(kb, "seqlen_k");
        q_token = kb->q_token;
        head_idx = kb->head_idx;
        kv_head_idx = kb->kv_head_idx;

        /* tid = b.thread_id_x(); lane_d_base = b.mul(tid, b.const_i32(ept)) */
        tid = ckc_b_thread_id_x(b);
        lane_d_base = ckc_b_mul(b, tid, ckc_b_const_i32(b, ept));

        /* q_row = kb.q_row_base(); do_row = kb.row_base("do", q_token, head_idx) */
        q_row = ckc_fmha_kernel_builder_q_row_base(kb);
        do_row = ckc_fmha_kernel_builder_row_base(kb, "do", q_token, head_idx);

        /* ml_row = b.add(b.mul(q_token, const(HQ)), head_idx) */
        ml_row = ckc_b_add(b,
                           ckc_b_mul(b, q_token,
                                     ckc_b_const_i32(b, s->shape.num_query_heads)),
                           head_idx);
        /* m_qt / l_qt = b.global_load_f32(M_saved/L_saved, ml_row); inv_l = rcp(l_qt) */
        m_qt = ckc_b_global_load_f32(b, M_saved, ml_row, 0);
        l_qt = ckc_b_global_load_f32(b, L_saved, ml_row, 0);
        inv_l = ckc_b_rcp(b, l_qt);

        /* q_lane / do_lane = load_lane_slice_f32(b, Q/dO, row, lane_d_base, dtype, ept) */
        q_lane = (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
        do_lane = (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
        if (q_lane == NULL || do_lane == NULL)
        {
            return NULL;
        }
        if (!ckc_b_load_lane_slice_f32(b, Q, q_row, lane_d_base, s->dtype, ept, q_lane))
        {
            return NULL;
        }
        if (!ckc_b_load_lane_slice_f32(b, dO, do_row, lane_d_base, s->dtype, ept, do_lane))
        {
            return NULL;
        }

        zero_f = ckc_b_const_f32(b, 0.0);

        /* ---------------- First K-loop: rowsum_dp via iter_arg ---------------- */
        {
            ckc_iter_arg_t rs_arg;
            rs_arg.name = "rs";
            rs_arg.init = zero_f;
            {
                ckc_value_t* lp_start = ckc_b_const_i32(b, 0);
                ckc_value_t* lp_step  = ckc_b_const_i32(b, 1);
                k_loop_1 = ckc_b_scf_for_iter(b, lp_start, seqlen_k,
                                              lp_step, &rs_arg, 1, "k_idx",
                                              false, true);
            }
        }
        ckc_b_region_enter(b, k_loop_1.body);
        {
            ckc_value_t* k_idx = k_loop_1.iv;
            ckc_value_t* rowsum_carry = k_loop_1.iter_vars[0];
            ckc_value_t* k_row = ckc_fmha_kernel_builder_k_row_base(kb, k_idx);
            ckc_value_t* v_row = ckc_fmha_kernel_builder_v_row_base(kb, k_idx);
            ckc_value_t** k_lane =
                (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
            ckc_value_t** v_lane =
                (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
            ckc_value_t *partial_qk, *partial_dov, *dot_qk, *dot_dov, *s_log2, *p, *y;

            if (k_lane == NULL || v_lane == NULL)
            {
                return NULL;
            }
            ckc_b_load_lane_slice_f32(b, K, k_row, lane_d_base, s->dtype, ept, k_lane);
            ckc_b_load_lane_slice_f32(b, V, v_row, lane_d_base, s->dtype, ept, v_lane);
            partial_qk = zero_f;
            partial_dov = zero_f;
            for (kk = 0; kk < ept; ++kk)
            {
                partial_qk = ckc_b_fma(b, q_lane[kk], k_lane[kk], partial_qk);
                partial_dov = ckc_b_fma(b, do_lane[kk], v_lane[kk], partial_dov);
            }
            dot_qk = ckc_warp_xor_reduce_sum(b, partial_qk, 6);
            dot_dov = ckc_warp_xor_reduce_sum(b, partial_dov, 6);
            s_log2 = ckc_b_fmul(b, dot_qk, scale_log2);
            s_log2 = ckc_apply_attention_mask(b, s_log2, attn_mask, k_idx, q_token,
                                              s->sliding_window, NULL, NULL);
            /* p = b.fmul(b.exp2(b.fsub(s_log2, m_qt)), inv_l) */
            p = ckc_b_fmul(b, ckc_b_exp2(b, ckc_b_fsub(b, s_log2, m_qt)), inv_l);
            /* b.scf_yield(b.fma(p, dot_dov, rowsum_carry)) */
            y = ckc_b_fma(b, p, dot_dov, rowsum_carry);
            ckc_b_scf_yield(b, &y, 1);
        }
        ckc_b_region_leave(b);
        /* rowsum_dp = k_loop_1.results[0] */
        rowsum_dp = NULL;
        if (k_loop_1.op != NULL && k_loop_1.op->num_results > 0)
        {
            rowsum_dp = k_loop_1.op->results[0];
        }

        /* ---------------- Second K-loop: atomic dV/dK, per-lane dQ regs --------- */
        {
            ckc_iter_arg_t* iter_args2 =
                (ckc_iter_arg_t*)ckc_arena_alloc(&b->arena,
                                                 sizeof(ckc_iter_arg_t) * (size_t)ept);
            if (iter_args2 == NULL)
            {
                return NULL;
            }
            for (kk = 0; kk < ept; ++kk)
            {
                char* nm = ckc_arena_printf(&b->arena, "dq%d", kk);
                iter_args2[kk].name = nm;
                iter_args2[kk].init = zero_f;
            }
            {
                ckc_value_t* lp_start = ckc_b_const_i32(b, 0);
                ckc_value_t* lp_step  = ckc_b_const_i32(b, 1);
                k_loop_2 = ckc_b_scf_for_iter(b, lp_start, seqlen_k,
                                              lp_step, iter_args2, ept,
                                              "k_idx2", false, true);
            }
        }
        ckc_b_region_enter(b, k_loop_2.body);
        {
            ckc_value_t* k_idx = k_loop_2.iv;
            ckc_value_t* k_row = ckc_fmha_kernel_builder_k_row_base(kb, k_idx);
            ckc_value_t* v_row = ckc_fmha_kernel_builder_v_row_base(kb, k_idx);
            ckc_value_t* dk_row;
            ckc_value_t* dv_row;
            ckc_value_t** k_lane =
                (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
            ckc_value_t** v_lane =
                (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
            ckc_value_t** new_dq =
                (ckc_value_t**)ckc_arena_alloc(&b->arena, sizeof(ckc_value_t*) * (size_t)ept);
            ckc_value_t *partial_qk, *partial_dov, *dot_qk, *dot_dov, *s_log2, *p, *dp,
                *dp_scale;

            if (k_lane == NULL || v_lane == NULL || new_dq == NULL)
            {
                return NULL;
            }

            /* dk_row = b.add(b.mul(k_idx, stride_dk_token), b.mul(kv_head_idx, stride_k_head))
             * Sequence the two mul() emissions left-to-right (Python order); C
             * argument evaluation order is unspecified. */
            {
                ckc_value_t* dk_tok =
                    ckc_b_mul(b, k_idx, ckc_fmha_kernel_builder_scalar(kb, "stride_dk_token"));
                ckc_value_t* dk_hd =
                    ckc_b_mul(b, kv_head_idx, ckc_fmha_kernel_builder_stride_head(kb, "k"));
                dk_row = ckc_b_add(b, dk_tok, dk_hd);
            }
            /* dv_row = b.add(b.mul(k_idx, stride_dv_token), b.mul(kv_head_idx, stride_v_head)) */
            {
                ckc_value_t* dv_tok =
                    ckc_b_mul(b, k_idx, ckc_fmha_kernel_builder_scalar(kb, "stride_dv_token"));
                ckc_value_t* dv_hd =
                    ckc_b_mul(b, kv_head_idx, ckc_fmha_kernel_builder_stride_head(kb, "v"));
                dv_row = ckc_b_add(b, dv_tok, dv_hd);
            }

            ckc_b_load_lane_slice_f32(b, K, k_row, lane_d_base, s->dtype, ept, k_lane);
            ckc_b_load_lane_slice_f32(b, V, v_row, lane_d_base, s->dtype, ept, v_lane);
            partial_qk = zero_f;
            partial_dov = zero_f;
            for (kk = 0; kk < ept; ++kk)
            {
                partial_qk = ckc_b_fma(b, q_lane[kk], k_lane[kk], partial_qk);
                partial_dov = ckc_b_fma(b, do_lane[kk], v_lane[kk], partial_dov);
            }
            dot_qk = ckc_warp_xor_reduce_sum(b, partial_qk, 6);
            dot_dov = ckc_warp_xor_reduce_sum(b, partial_dov, 6);
            s_log2 = ckc_b_fmul(b, dot_qk, scale_log2);
            s_log2 = ckc_apply_attention_mask(b, s_log2, attn_mask, k_idx, q_token,
                                              s->sliding_window, NULL, NULL);
            p = ckc_b_fmul(b, ckc_b_exp2(b, ckc_b_fsub(b, s_log2, m_qt)), inv_l);
            /* dp = b.fmul(p, b.fsub(dot_dov, rowsum_dp)) */
            dp = ckc_b_fmul(b, p, ckc_b_fsub(b, dot_dov, rowsum_dp));
            /* dp_scale = b.fmul(dp, scale_inv) */
            dp_scale = ckc_b_fmul(b, dp, scale_inv);

            for (kk = 0; kk < ept; ++kk)
            {
                ckc_value_t* d = ckc_b_add(b, lane_d_base, ckc_b_const_i32(b, kk));
                /* b.global_atomic_add(dV, b.add(dv_row, d), b.fmul(p, do_lane[k]))
                 * Python evaluates the call args left-to-right: the address
                 * b.add(...) is emitted before the value b.fmul(...). C argument
                 * evaluation order is unspecified, so sequence them. */
                {
                    ckc_value_t* dv_addr = ckc_b_add(b, dv_row, d);
                    ckc_value_t* dv_val = ckc_b_fmul(b, p, do_lane[kk]);
                    ckc_b_global_atomic_add(b, dV, dv_addr, dv_val, NULL);
                }
                /* b.global_atomic_add(dK, b.add(dk_row, d), b.fmul(dp_scale, q_lane[k])) */
                {
                    ckc_value_t* dk_addr = ckc_b_add(b, dk_row, d);
                    ckc_value_t* dk_val = ckc_b_fmul(b, dp_scale, q_lane[kk]);
                    ckc_b_global_atomic_add(b, dK, dk_addr, dk_val, NULL);
                }
                /* new_dq.append(b.fma(dp_scale, k_lane[k], dq_state[k])) */
                new_dq[kk] = ckc_b_fma(b, dp_scale, k_lane[kk], k_loop_2.iter_vars[kk]);
            }
            /* b.scf_yield(*new_dq) */
            ckc_b_scf_yield(b, new_dq, ept);
        }
        ckc_b_region_leave(b);

        /* ---------------- Final: write dQ via per-lane atomic_add --------------- */
        {
            /* dq_row = b.add(b.mul(q_token, stride_dq_token), b.mul(head_idx, stride_q_head))
             * Sequence the two mul() emissions left-to-right (Python order). */
            ckc_value_t* dq_tok =
                ckc_b_mul(b, q_token, ckc_fmha_kernel_builder_scalar(kb, "stride_dq_token"));
            ckc_value_t* dq_hd =
                ckc_b_mul(b, head_idx, ckc_fmha_kernel_builder_stride_head(kb, "q"));
            ckc_value_t* dq_row = ckc_b_add(b, dq_tok, dq_hd);
            for (kk = 0; kk < ept; ++kk)
            {
                ckc_value_t* d = ckc_b_add(b, lane_d_base, ckc_b_const_i32(b, kk));
                ckc_value_t* res = NULL;
                if (k_loop_2.op != NULL && kk < k_loop_2.op->num_results)
                {
                    res = k_loop_2.op->results[kk];
                }
                ckc_b_global_atomic_add(b, dQ, ckc_b_add(b, dq_row, d), res, NULL);
            }
        }

        /* b.ret() */
        ckc_b_ret(b);
        return ckc_fmha_kernel_builder_kernel(kb);
    });
}

/* ===================================================================== *
 *  grid / signature
 * ===================================================================== */

void ckc_fmha_bwd_grid(const ckc_fmha_bwd_spec_t* spec, int* gx, int* gy, int* gz)
{
    if (spec == NULL)
    {
        return;
    }
    if (gx != NULL)
    {
        *gx = spec->seqlen_q;
    }
    if (gy != NULL)
    {
        *gy = spec->common.shape.num_query_heads;
    }
    if (gz != NULL)
    {
        *gz = 1;
    }
}

ckc_status_t ckc_fmha_bwd_signature(ckc_fmha_kernel_builder_t* kb,
                                    const ckc_fmha_bwd_spec_t* spec,
                                    ckc_arena_t* arena,
                                    const ckc_sig_entry_t** out_items,
                                    size_t* out_count)
{
    ckc_status_t st;

    if (kb == NULL || spec == NULL || arena == NULL || out_items == NULL ||
        out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    /* kb = FmhaKernelBuilder("ck_dsl_fmha_bwd_sig_probe", spec.common) */
    st = ckc_fmha_kernel_builder_init(kb, "ck_dsl_fmha_bwd_sig_probe",
                                      &spec->common);
    if (st != CKC_OK)
    {
        return st;
    }
    /* _declare_params(kb) */
    fmha_bwd_declare_params(kb);
    /* return kb.signature() */
    return ckc_fmha_kernel_builder_signature(kb, arena, out_items, out_count);
}

/* ===================================================================== *
 *  ckc_fmha_bwd_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own FmhaKernelBuilder.
 * ===================================================================== */

ckc_status_t ckc_fmha_bwd_lower_to_llvm(const ckc_fmha_bwd_spec_t* spec,
                                        const char* arch,
                                        ckc_llvm_flavor_t flavor,
                                        char** out_ll,
                                        char* err,
                                        size_t err_cap)
{
    ckc_fmha_kernel_builder_t kb;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        if (err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_fmha_bwd(&kb, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&kb.b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&kb.b);
            if (m == NULL)
            {
                m = "build_fmha_bwd failed";
            }
            snprintf(err, err_cap, "%s", m);
        }
        ckc_fmha_kernel_builder_free(&kb);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_fmha_kernel_builder_free(&kb);
    return st;
}
