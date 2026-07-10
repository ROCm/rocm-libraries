// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/_fmha_warp_body.py
 * (WARP_SIZE, fmha_warp_fwd_inner_body). See the header for the mapping.
 */

#include "rocke/helper_rocke.instances.common._fmha_warp_body.h"

#include <stdio.h> /* snprintf (iter-arg name formatting) */
#include <string.h>

#include "rocke/helper_rocke.helpers.attention.h" /* rocke_causal_mask, rocke_sliding_window_mask, rocke_warp_xor_reduce_sum */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type, rocke_b_load_scalar_as_f32, rocke_b_load_vec_as_f32 */
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* ------------------------------------------------------------------ peers *
 *
 * The io / attention helpers are now ported. load_scalar_as_f32 /
 * load_vec_as_f32 live in helper_rocke.helpers.io.h under the rocke_b_ prefix
 * (the builder-bound surface); alias them to the unprefixed names this body was
 * written against. The vec variant returns int (1 ok / 0 fail) but is used here
 * for its `out` side effect, matching the Python list-return contract. The
 * masks and the warp xor-reduce come straight from the attention helper header.
 */
#define rocke_load_scalar_as_f32 rocke_b_load_scalar_as_f32
#define rocke_load_vec_as_f32 rocke_b_load_vec_as_f32

/* Largest supported EPT slice (head_size 256 / WARP_SIZE 64 == 4). Used to size
 * the on-stack list-path scratch arrays. */
#define ROCKE_FMHA_MAX_EPT 8

int rocke_fmha_warp_size(void)
{
    return ROCKE_FMHA_WARP_SIZE;
}

/* _row_bases closure -> helper. Reproduces the Python addressing verbatim. */
static void rocke_fmha_row_bases(rocke_ir_builder_t* b,
                                 const rocke_fmha_warp_fwd_opts_t* o,
                                 rocke_value_t* k_idx,
                                 rocke_value_t* k_off,
                                 rocke_value_t* v_off,
                                 rocke_value_t** out_kbase,
                                 rocke_value_t** out_vbase)
{
    rocke_value_t* kbase;
    rocke_value_t* vbase;

    if(o->k_row_base_fn != NULL)
    {
        kbase = o->k_row_base_fn(b, k_idx, o->user);
    }
    else
    {
        /* Python: b.add(b.add(b.mul(k_idx, stride_k_token),
         *                     b.mul(kv_head_idx, stride_k_head)), k_off).
         * The token-stride mul is emitted BEFORE the head-stride mul. C leaves
         * function-argument evaluation order unspecified (clang/gcc emit
         * right-to-left), which would emit the head mul first and renumber the
         * SSA values. Pin Python order with explicit temporaries. */
        rocke_value_t* k_tok = rocke_b_mul(b, k_idx, o->stride_k_token);
        rocke_value_t* k_hd = rocke_b_mul(b, o->kv_head_idx, o->stride_k_head);
        kbase = rocke_b_add(b, rocke_b_add(b, k_tok, k_hd), k_off);
    }
    if(o->v_row_base_fn != NULL)
    {
        vbase = o->v_row_base_fn(b, k_idx, o->user);
    }
    else
    {
        /* Same token-before-head pinning as kbase above. */
        rocke_value_t* v_tok = rocke_b_mul(b, k_idx, o->stride_v_token);
        rocke_value_t* v_hd = rocke_b_mul(b, o->kv_head_idx, o->stride_v_head);
        vbase = rocke_b_add(b, rocke_b_add(b, v_tok, v_hd), v_off);
    }
    *out_kbase = kbase;
    *out_vbase = vbase;
}

/* _apply_mask_and_score closure -> helper. neg_inf is the loop-invariant -1e30
 * constant created once by the caller. */
static rocke_value_t* rocke_fmha_apply_mask_and_score(rocke_ir_builder_t* b,
                                                      const rocke_fmha_warp_fwd_opts_t* o,
                                                      rocke_value_t* score_log2,
                                                      rocke_value_t* k_idx,
                                                      rocke_value_t* neg_inf)
{
    const char* mask_mode = (o->mask_mode != NULL) ? o->mask_mode : "none";

    if(o->extra_score_transform != NULL)
    {
        score_log2 = o->extra_score_transform(b, score_log2, k_idx, o->user);
    }

    if(o->extra_mask_predicate != NULL)
    {
        rocke_value_t* keep_extra = o->extra_mask_predicate(b, k_idx, o->user);
        score_log2 = rocke_b_select(b, keep_extra, score_log2, neg_inf);
    }

    if(strcmp(mask_mode, "causal") == 0 && o->causal_ctx_len != NULL)
    {
        rocke_value_t* keep
            = rocke_causal_mask(b, k_idx, rocke_b_const_i32(b, 0), o->causal_ctx_len);
        score_log2 = rocke_b_select(b, keep, score_log2, neg_inf);
    }
    else if(strcmp(mask_mode, "sliding_window") == 0 && o->causal_ctx_len != NULL)
    {
        rocke_value_t* keep = rocke_sliding_window_mask(
            b, k_idx, rocke_b_const_i32(b, 0), o->causal_ctx_len, o->sliding_window);
        score_log2 = rocke_b_select(b, keep, score_log2, neg_inf);
    }
    return score_log2;
}

void rocke_fmha_warp_fwd_inner_body(rocke_ir_builder_t* b, const rocke_fmha_warp_fwd_opts_t* o)
{
    const char* dtype;
    int ept;
    const rocke_type_t* dtype_ir;
    rocke_value_t* tid;
    rocke_value_t* c_ept;
    rocke_value_t* lane_d_base;
    rocke_value_t* q_row_base;
    rocke_value_t* o_row_base;
    rocke_value_t* k_off;
    rocke_value_t* v_off;
    rocke_value_t* neg_inf;
    rocke_value_t* zero_f;
    rocke_value_t* q_lane_addr;
    rocke_value_t* o_lane_addr;
    int slot;

    if(b == NULL || !rocke_i_live(b) || o == NULL)
    {
        return;
    }

    dtype = o->dtype;

    /* if dtype not in ("f16","fp16","bf16"): raise ValueError(...)
     * Python f-string uses {dtype!r} -> single-quoted repr. */
    if(dtype == NULL
       || (strcmp(dtype, "f16") != 0 && strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0))
    {
        (void)rocke_i_set_err(b,
                              ROCKE_ERR_VALUE,
                              "fmha_warp_fwd_inner_body dtype %s%s%s not supported",
                              dtype ? "'" : "",
                              dtype ? dtype : "None",
                              dtype ? "'" : "");
        return;
    }

    /* if head_size % WARP_SIZE != 0: raise ValueError(...) */
    if(o->head_size % ROCKE_FMHA_WARP_SIZE != 0)
    {
        (void)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "fmha_warp_fwd_inner_body needs head_size %% %d == 0; got head_size=%d",
            ROCKE_FMHA_WARP_SIZE,
            o->head_size);
        return;
    }

    ept = o->head_size / ROCKE_FMHA_WARP_SIZE;

    dtype_ir = rocke_b_io_ir_type(b, dtype);

    tid = rocke_b_thread_id_x(b);
    c_ept = rocke_b_const_i32(b, ept);
    lane_d_base = rocke_b_mul(b, tid, c_ept); /* tid * EPT */

    /* Python evaluates add() args left-to-right: the token-mul is emitted
     * before the head-mul. C argument evaluation order is unspecified, so
     * sequence the two muls into temporaries to pin IR emission order. */
    {
        rocke_value_t* q_tok_mul = rocke_b_mul(b, o->q_token, o->stride_q_token);
        rocke_value_t* q_head_mul = rocke_b_mul(b, o->head_idx, o->stride_q_head);
        q_row_base = rocke_b_add(b, q_tok_mul, q_head_mul);
    }
    {
        rocke_value_t* o_tok_mul = rocke_b_mul(b, o->q_token, o->stride_o_token);
        rocke_value_t* o_head_mul = rocke_b_mul(b, o->head_idx, o->stride_o_head);
        o_row_base = rocke_b_add(b, o_tok_mul, o_head_mul);
    }

    k_off = (o->k_token_offset_elems != NULL) ? o->k_token_offset_elems : rocke_b_const_i32(b, 0);
    v_off = (o->v_token_offset_elems != NULL) ? o->v_token_offset_elems : rocke_b_const_i32(b, 0);

    neg_inf = rocke_b_const_f32(b, -1e30);
    zero_f = rocke_b_const_f32(b, 0.0);

    q_lane_addr = rocke_b_add(b, q_row_base, lane_d_base);
    o_lane_addr = rocke_b_add(b, o_row_base, lane_d_base);

    /* ============================================================= list path */
    if(o->kv_lane_loader != NULL)
    {
        rocke_value_t* q_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_iter_arg_t iter_args[2 + ROCKE_FMHA_MAX_EPT];
        char acc_names[ROCKE_FMHA_MAX_EPT][16]; /* "a" + worst-case %d + NUL */
        int num_iter_args;
        rocke_for_t k_loop;
        rocke_value_t* k_idx;
        rocke_value_t* m;
        rocke_value_t* lse;
        rocke_value_t** acc_iter;
        rocke_value_t* k_row_base;
        rocke_value_t* v_row_base;
        rocke_value_t* k_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* v_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* partial;
        rocke_value_t* dot;
        rocke_value_t* score_log2;
        rocke_value_t* m_new;
        rocke_value_t* alpha;
        rocke_value_t* p;
        rocke_value_t* lse_new;
        rocke_value_t* new_yields[2 + ROCKE_FMHA_MAX_EPT];
        rocke_value_t* l_final;
        rocke_value_t* acc_final[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* inv_l;

        if(o->q_lane_loader != NULL)
        {
            o->q_lane_loader(b, q_row_base, lane_d_base, ept, q_f32_list, o->user);
        }
        else
        {
            rocke_load_vec_as_f32(b, o->Q, q_lane_addr, dtype, ept, q_f32_list);
        }

        /* iter_args = [("m", neg_inf), ("l", zero_f)] + [("aSLOT", zero_f)...] */
        iter_args[0].name = "m";
        iter_args[0].init = neg_inf;
        iter_args[1].name = "l";
        iter_args[1].init = zero_f;
        for(slot = 0; slot < ept; ++slot)
        {
            snprintf(acc_names[slot], sizeof(acc_names[slot]), "a%d", slot);
            iter_args[2 + slot].name = acc_names[slot];
            iter_args[2 + slot].init = zero_f;
        }
        num_iter_args = 2 + ept;

        {
            rocke_value_t* loop_start = rocke_b_const_i32(b, 0);
            rocke_value_t* loop_step = rocke_b_const_i32(b, 1);
            k_loop = rocke_b_scf_for_iter(b,
                                          loop_start,
                                          o->seqlen_k,
                                          loop_step,
                                          iter_args,
                                          num_iter_args,
                                          "k_idx",
                                          false,
                                          true);
        }

        rocke_b_region_enter(b, k_loop.body);
        k_idx = k_loop.iv;
        m = k_loop.iter_vars[0];
        lse = k_loop.iter_vars[1];
        acc_iter = &k_loop.iter_vars[2];

        rocke_fmha_row_bases(b, o, k_idx, k_off, v_off, &k_row_base, &v_row_base);

        o->kv_lane_loader(
            b, k_idx, k_row_base, v_row_base, lane_d_base, ept, k_f32_list, v_f32_list, o->user);

        partial = rocke_b_const_f32(b, 0.0);
        for(slot = 0; slot < ept; ++slot)
        {
            partial = rocke_b_fadd(b, partial, rocke_b_fmul(b, q_f32_list[slot], k_f32_list[slot]));
        }

        dot = rocke_warp_xor_reduce_sum(b, partial, 6);
        score_log2 = rocke_fmha_apply_mask_and_score(
            b, o, rocke_b_fmul(b, dot, o->scale_log2), k_idx, neg_inf);

        m_new = rocke_b_fmax(b, m, score_log2);
        alpha = rocke_b_exp2(b, rocke_b_fsub(b, m, m_new));
        p = rocke_b_exp2(b, rocke_b_fsub(b, score_log2, m_new));
        lse_new = rocke_b_fadd(b, rocke_b_fmul(b, lse, alpha), p);

        new_yields[0] = m_new;
        new_yields[1] = lse_new;
        for(slot = 0; slot < ept; ++slot)
        {
            /* Python: b.fadd(b.fmul(acc, alpha), b.fmul(p, v)) -- the
             * acc-mul is emitted before the p-mul. Sequence into temps. */
            {
                rocke_value_t* acc_mul = rocke_b_fmul(b, acc_iter[slot], alpha);
                rocke_value_t* pv_mul = rocke_b_fmul(b, p, v_f32_list[slot]);
                new_yields[2 + slot] = rocke_b_fadd(b, acc_mul, pv_mul);
            }
        }
        rocke_b_scf_yield(b, new_yields, num_iter_args);
        rocke_b_region_leave(b);

        /* results = k_loop.results; l_final = results[1]; acc_final = results[2:] */
        l_final = (k_loop.op != NULL) ? k_loop.op->results[1] : NULL;
        for(slot = 0; slot < ept; ++slot)
        {
            acc_final[slot] = (k_loop.op != NULL) ? k_loop.op->results[2 + slot] : NULL;
        }
        inv_l = rocke_b_rcp(b, l_final);
        for(slot = 0; slot < ept; ++slot)
        {
            /* Python: store_scalar_from_f32(b, O, b.add(o_lane_addr, slot),
             *                               b.fmul(acc_final[slot], inv_l)).
             * The output-offset add is emitted BEFORE the fmul. C arg-eval
             * order is unspecified (right-to-left on clang/gcc), so pin the
             * Python order with explicit temporaries. */
            rocke_value_t* o_addr = rocke_b_add(b, o_lane_addr, rocke_b_const_i32(b, slot));
            rocke_value_t* o_val = rocke_b_fmul(b, acc_final[slot], inv_l);
            rocke_b_store_scalar_from_f32(b, o->O, o_addr, o_val, dtype);
        }
        return;
    }

    /* =========================================================== scalar path */
    if(ept == 1)
    {
        rocke_value_t* q_scalar;
        rocke_iter_arg_t iter_args[3];
        rocke_for_t k_loop;
        rocke_value_t* k_idx;
        rocke_value_t* m;
        rocke_value_t* lse;
        rocke_value_t* acc0;
        rocke_value_t* k_row_base;
        rocke_value_t* v_row_base;
        rocke_value_t* kd;
        rocke_value_t* vd;
        rocke_value_t* partial;
        rocke_value_t* dot;
        rocke_value_t* score_log2;
        rocke_value_t* m_new;
        rocke_value_t* alpha;
        rocke_value_t* p;
        rocke_value_t* lse_new;
        rocke_value_t* new_acc0;
        rocke_value_t* yields[3];
        rocke_value_t* l_final;
        rocke_value_t* acc0_final;
        rocke_value_t* out_f32;

        q_scalar = rocke_load_scalar_as_f32(b, o->Q, q_lane_addr, dtype);

        iter_args[0].name = "m";
        iter_args[0].init = neg_inf;
        iter_args[1].name = "l";
        iter_args[1].init = zero_f;
        iter_args[2].name = "a0";
        iter_args[2].init = zero_f;

        {
            rocke_value_t* loop_start = rocke_b_const_i32(b, 0);
            rocke_value_t* loop_step = rocke_b_const_i32(b, 1);
            k_loop = rocke_b_scf_for_iter(
                b, loop_start, o->seqlen_k, loop_step, iter_args, 3, "k_idx", false, true);
        }

        rocke_b_region_enter(b, k_loop.body);
        k_idx = k_loop.iv;
        m = k_loop.iter_vars[0];
        lse = k_loop.iter_vars[1];
        acc0 = k_loop.iter_vars[2];

        rocke_fmha_row_bases(b, o, k_idx, k_off, v_off, &k_row_base, &v_row_base);

        kd = rocke_load_scalar_as_f32(b, o->K, rocke_b_add(b, k_row_base, lane_d_base), dtype);
        vd = rocke_load_scalar_as_f32(b, o->V, rocke_b_add(b, v_row_base, lane_d_base), dtype);
        partial = rocke_b_fmul(b, q_scalar, kd);

        dot = rocke_warp_xor_reduce_sum(b, partial, 6);
        score_log2 = rocke_fmha_apply_mask_and_score(
            b, o, rocke_b_fmul(b, dot, o->scale_log2), k_idx, neg_inf);

        m_new = rocke_b_fmax(b, m, score_log2);
        alpha = rocke_b_exp2(b, rocke_b_fsub(b, m, m_new));
        p = rocke_b_exp2(b, rocke_b_fsub(b, score_log2, m_new));
        lse_new = rocke_b_fadd(b, rocke_b_fmul(b, lse, alpha), p);
        /* Python: b.fadd(b.fmul(acc0, alpha), b.fmul(p, vd)) -- acc0-mul
         * emitted before the p-mul. Sequence into temps for emission order. */
        {
            rocke_value_t* acc_mul = rocke_b_fmul(b, acc0, alpha);
            rocke_value_t* pv_mul = rocke_b_fmul(b, p, vd);
            new_acc0 = rocke_b_fadd(b, acc_mul, pv_mul);
        }

        yields[0] = m_new;
        yields[1] = lse_new;
        yields[2] = new_acc0;
        rocke_b_scf_yield(b, yields, 3);
        rocke_b_region_leave(b);

        l_final = (k_loop.op != NULL) ? k_loop.op->results[1] : NULL;
        acc0_final = (k_loop.op != NULL) ? k_loop.op->results[2] : NULL;
        out_f32 = rocke_b_fmul(b, acc0_final, rocke_b_rcp(b, l_final));
        rocke_b_store_scalar_from_f32(b, o->O, o_lane_addr, out_f32, dtype);
        return;
    }

    /* =========================================================== vector path */
    {
        rocke_value_t* q_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* q_vec_f32;
        rocke_iter_arg_t iter_args[3];
        rocke_for_t k_loop;
        rocke_value_t* k_idx;
        rocke_value_t* m;
        rocke_value_t* lse;
        rocke_value_t* acc_v;
        rocke_value_t* k_row_base;
        rocke_value_t* v_row_base;
        rocke_value_t* k_lane_addr;
        rocke_value_t* k_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* k_vec_f32;
        rocke_value_t* v_lane_addr;
        rocke_value_t* v_f32_list[ROCKE_FMHA_MAX_EPT];
        rocke_value_t* v_vec_f32;
        rocke_value_t* partial;
        rocke_value_t* dot;
        rocke_value_t* score_log2;
        rocke_value_t* m_new;
        rocke_value_t* alpha;
        rocke_value_t* p;
        rocke_value_t* lse_new;
        rocke_value_t* alpha_v;
        rocke_value_t* p_v;
        rocke_value_t* new_acc_v;
        rocke_value_t* yields[3];
        rocke_value_t* l_final;
        rocke_value_t* acc_final_v;
        rocke_value_t* inv_l;
        rocke_value_t* inv_l_v;
        rocke_value_t* out_v_f32;
        rocke_value_t* out_v_dtype;

        rocke_load_vec_as_f32(b, o->Q, q_lane_addr, dtype, ept, q_f32_list);
        q_vec_f32 = rocke_b_vec_pack(b, q_f32_list, ept, rocke_f32());

        /* Python: b.scf_for_iter(b.const_i32(0), seqlen_k, b.const_i32(1),
         *                        iter_args=[("m",..),("l",..),
         *                                   ("acc", b.zero_vec_f32(ept))]).
         * Args are evaluated left-to-right: the start/step consts are emitted
         * before the iter_args list (hence the acc zero_vec). C evaluates the
         * call args in unspecified order, so build the start/step consts first
         * and the zero_vec iter_arg last to pin IR emission order. */
        {
            rocke_value_t* loop_start = rocke_b_const_i32(b, 0);
            rocke_value_t* loop_step = rocke_b_const_i32(b, 1);

            iter_args[0].name = "m";
            iter_args[0].init = neg_inf;
            iter_args[1].name = "l";
            iter_args[1].init = zero_f;
            iter_args[2].name = "acc";
            iter_args[2].init = rocke_b_zero_vec_f32(b, ept);

            k_loop = rocke_b_scf_for_iter(
                b, loop_start, o->seqlen_k, loop_step, iter_args, 3, "k_idx", false, true);
        }

        /* A failed earlier op (e.g. load_vec n=ept not in {2,4,8}, which
         * Python raises before the loop is ever built) leaves the builder
         * non-live and scf_for_iter returns a zeroed rocke_for_t with a NULL
         * iter_vars array. Mirror Python's immediate abort instead of
         * dereferencing the NULL iter_vars[] below. */
        if(!rocke_i_live(b) || k_loop.iter_vars == NULL)
        {
            return;
        }

        rocke_b_region_enter(b, k_loop.body);
        k_idx = k_loop.iv;
        m = k_loop.iter_vars[0];
        lse = k_loop.iter_vars[1];
        acc_v = k_loop.iter_vars[2];

        rocke_fmha_row_bases(b, o, k_idx, k_off, v_off, &k_row_base, &v_row_base);

        k_lane_addr = rocke_b_add(b, k_row_base, lane_d_base);
        rocke_load_vec_as_f32(b, o->K, k_lane_addr, dtype, ept, k_f32_list);
        k_vec_f32 = rocke_b_vec_pack(b, k_f32_list, ept, rocke_f32());

        v_lane_addr = rocke_b_add(b, v_row_base, lane_d_base);
        rocke_load_vec_as_f32(b, o->V, v_lane_addr, dtype, ept, v_f32_list);
        v_vec_f32 = rocke_b_vec_pack(b, v_f32_list, ept, rocke_f32());

        partial = rocke_b_vector_sum(b, rocke_b_vector_mul(b, q_vec_f32, k_vec_f32));

        dot = rocke_warp_xor_reduce_sum(b, partial, 6);
        score_log2 = rocke_fmha_apply_mask_and_score(
            b, o, rocke_b_fmul(b, dot, o->scale_log2), k_idx, neg_inf);

        m_new = rocke_b_fmax(b, m, score_log2);
        alpha = rocke_b_exp2(b, rocke_b_fsub(b, m, m_new));
        p = rocke_b_exp2(b, rocke_b_fsub(b, score_log2, m_new));
        lse_new = rocke_b_fadd(b, rocke_b_fmul(b, lse, alpha), p);

        alpha_v = rocke_b_vector_splat(b, alpha, ept);
        p_v = rocke_b_vector_splat(b, p, ept);
        /* Python: vector_add(vector_mul(acc_v, alpha_v),
         *                     vector_mul(p_v, v_vec_f32)) -- acc-mul emitted
         * before the p-mul. Sequence into temps for emission order. */
        {
            rocke_value_t* acc_mul = rocke_b_vector_mul(b, acc_v, alpha_v);
            rocke_value_t* pv_mul = rocke_b_vector_mul(b, p_v, v_vec_f32);
            new_acc_v = rocke_b_vector_add(b, acc_mul, pv_mul);
        }

        yields[0] = m_new;
        yields[1] = lse_new;
        yields[2] = new_acc_v;
        rocke_b_scf_yield(b, yields, 3);
        rocke_b_region_leave(b);

        l_final = (k_loop.op != NULL) ? k_loop.op->results[1] : NULL;
        acc_final_v = (k_loop.op != NULL) ? k_loop.op->results[2] : NULL;

        inv_l = rocke_b_rcp(b, l_final);
        inv_l_v = rocke_b_vector_splat(b, inv_l, ept);
        out_v_f32 = rocke_b_vector_mul(b, acc_final_v, inv_l_v);
        out_v_dtype = rocke_b_vec_cast_f32_to(b, out_v_f32, dtype_ir);
        rocke_b_global_store_vN(b, o->O, o_lane_addr, out_v_dtype, ept, ept * 2);
    }
}
