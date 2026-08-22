// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the split-KV decode FMHA forward instance builder
 * rocke/instances/common/fmha_splitkv_decode.py.
 *
 * See rocke/instance_fmha_splitkv_decode.h for the public symbol map.
 *
 * The build entries reproduce the Python builder-call sequence byte-faithfully:
 * the FmhaKernelBuilder param declarations (in declaration order), the GQA
 * magic-division head map, the power-of-two segment-length right-shift, the
 * scf.for online-softmax accumulation in the segment kernel, the two-pass /
 * per-lane scf.for reduction in the reduce kernel, and the scf.if lead-lane
 * (m, l) writeback. The shared scaffolding (FmhaKernelBuilder, common-spec
 * validation, the io / attention / transforms helpers) is reused from the
 * sibling C ports.
 */
#include "rocke/instance_fmha_splitkv_decode.h"

#include <stdio.h> /* snprintf */
#include <string.h> /* memset, memcpy, strlen */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.attention.h" /* apply_attention_mask, warp_xor_reduce_sum */
#include "rocke/helper_rocke.helpers.io.h" /* load_lane_slice_f32, pack_f32_to, store_vec */
#include "rocke/helper_rocke.helpers.spec.h" /* kernel_name_join */
#include "rocke/helper_rocke.helpers.transforms.h" /* calculate_magic_numbers, do_magic_division */
#include "rocke/helper_rocke.instances.common._fmha_warp_body.h" /* ROCKE_FMHA_WARP_SIZE */
#include "rocke/helper_rocke.instances.common.fmha_arch.h" /* validate_fmha_mfma_atom */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* WARP_SIZE = 64 (one warp per CTA, lane t owns head-dim slice). */
#define ROCKE_SPLITKV_WARP_SIZE ROCKE_FMHA_WARP_SIZE

/* Power-of-two vector widths the DSL's global_load_vN / packed stores
 * understand. EPT == 1 (head_size=64) and EPT == 3 (head_size=192) fall back to
 * per-element scalar paths.  Python: _VEC_WIDTHS = (2, 4, 8). */
static bool rocke_splitkv_ept_in_vec_widths(int ept)
{
    return ept == 2 || ept == 4 || ept == 8;
}

/* int.bit_length() - 1 for a positive power of two (== log2). */
static int rocke_splitkv_log2_pow2(int v)
{
    int s = 0;
    while((1 << (s + 1)) <= v)
    {
        ++s;
    }
    return s;
}

/* Map the FmhaCommonSpec mask-mode enum to the attention-helper mask-mode enum.
 * Python passes the mask_mode *string* through to apply_attention_mask, which
 * accepts "none"/"causal"/"sliding_window"; alibi/custom would raise there (and
 * never reach the warp body in a valid spec). */
static rocke_attn_mask_mode_t rocke_splitkv_attn_mask_mode(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return ROCKE_ATTN_MASK_SLIDING_WINDOW;
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* ===================================================================== *
 *  FmhaFwdSplitKvDecodeSpec value helpers
 * ===================================================================== */

rocke_fmha_splitkv_decode_spec_t rocke_fmha_splitkv_decode_spec_default(
    rocke_fmha_common_spec_t common, int batch, int num_segments)
{
    rocke_fmha_splitkv_decode_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.common = common;
    spec.batch = batch;
    spec.num_segments = num_segments;
    spec.name = "rocke_fmha_fwd_splitkv_decode";
    spec.use_mfma_body = false;
    spec.prune_sliding_window = false;
    return spec;
}

rocke_status_t rocke_fmha_splitkv_decode_kernel_name(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                     const char* phase,
                                                     char* out,
                                                     size_t out_cap)
{
    char h[32], hq[32], hk[32], bb[32], ss[32];
    const char* parts[7];
    const rocke_fmha_shape_t* s;
    const char* name;

    if(spec == NULL || phase == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    s = &spec->common.shape;
    name = (spec->name != NULL) ? spec->name : "rocke_fmha_fwd_splitkv_decode";

    /* kernel_name_join(name, phase, f"H{head_size}", f"HQ{num_query_heads}",
     *                  f"HK{num_kv_heads}", dtype, f"B{batch}", f"S{num_segments}") */
    snprintf(h, sizeof(h), "H%d", s->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", s->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", s->num_kv_heads);
    snprintf(bb, sizeof(bb), "B%d", spec->batch);
    snprintf(ss, sizeof(ss), "S%d", spec->num_segments);

    parts[0] = phase;
    parts[1] = h;
    parts[2] = hq;
    parts[3] = hk;
    parts[4] = spec->common.dtype;
    parts[5] = bb;
    parts[6] = ss;

    return rocke_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason)
 * ===================================================================== */

static void rocke_splitkv_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_fmha_splitkv_decode_is_valid_spec(const rocke_fmha_splitkv_decode_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap)
{
    rocke_arena_t arena;
    const char* why = NULL;
    bool ok;
    char buf[192];

    if(spec == NULL)
    {
        rocke_splitkv_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = validate_common_spec(spec.common) */
    if(rocke_arena_init(&arena, 0) != 0)
    {
        rocke_splitkv_set_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    ok = rocke_fmha_validate_common_spec(&arena, &spec->common, &why);
    if(!ok)
    {
        rocke_splitkv_set_reason(reason, reason_cap, (why != NULL) ? why : "invalid common spec");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch) */
    if(!rocke_validate_fmha_mfma_atom(spec->common.dtype, arch, buf, sizeof(buf)))
    {
        rocke_splitkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.batch <= 0: return False, f"batch must be > 0 (got {batch})" */
    if(spec->batch <= 0)
    {
        snprintf(buf, sizeof(buf), "batch must be > 0 (got %d)", spec->batch);
        rocke_splitkv_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.num_segments not in (1, 2, 4, 8, 16, 32, 64, 128): ... */
    {
        int n = spec->num_segments;
        bool good
            = (n == 1 || n == 2 || n == 4 || n == 8 || n == 16 || n == 32 || n == 64 || n == 128);
        if(!good)
        {
            snprintf(buf, sizeof(buf), "num_segments %d not in {1, 2, ..., 128}", n);
            rocke_splitkv_set_reason(reason, reason_cap, buf);
            return false;
        }
    }

    rocke_splitkv_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  param declarations
 * ===================================================================== */

static void rocke_splitkv_declare_segment_params(rocke_fmha_kernel_builder_t* kb)
{
    const char* kv_names[2];

    /* kb.add_tensor("Q", readonly=True) etc. (align default 16). */
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    /* kb.add_ptr("seqlens_k", dtype="i32", readonly=True) (align default 4). */
    rocke_fmha_kernel_builder_add_ptr(kb, "seqlens_k", "i32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_m", "f32", false, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_l", "f32", false, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_acc", "f32", false, 4);
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "batch", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_q_seq", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_q_head", "i32");
    /* kb.add_strides("k", "v") */
    kv_names[0] = "k";
    kv_names[1] = "v";
    rocke_fmha_kernel_builder_add_strides(kb, kv_names, 2);
}

static void rocke_splitkv_declare_reduce_params(rocke_fmha_kernel_builder_t* kb)
{
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_m", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_l", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "ws_acc", "f32", true, 4);
    /* kb.add_tensor("O", readonly=False, writeonly=True) */
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, false, true, 16);
    rocke_fmha_kernel_builder_add_scalar(kb, "batch", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_o_seq", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_o_head", "i32");
}

/* ===================================================================== *
 *  _store_lane_slice_f32_packed
 * ===================================================================== *
 *
 * One packed global_store_vN for EPT in {2,4,8}; per-element scalar stores
 * otherwise. */
static void rocke_splitkv_store_lane_slice_f32_packed(rocke_ir_builder_t* b,
                                                      rocke_value_t* ptr,
                                                      rocke_value_t* row_base,
                                                      rocke_value_t* lane_d_base,
                                                      rocke_value_t* const* values_f32,
                                                      const char* dtype,
                                                      int ept)
{
    if(rocke_splitkv_ept_in_vec_widths(ept))
    {
        rocke_value_t* packed = rocke_b_pack_f32_to(b, values_f32, ept, dtype);
        rocke_b_store_vec(b, ptr, rocke_b_add(b, row_base, lane_d_base), packed, ept);
        return;
    }
    {
        int k;
        for(k = 0; k < ept; ++k)
        {
            rocke_b_store_scalar_from_f32(
                b,
                ptr,
                rocke_b_add(b, row_base, rocke_b_add(b, lane_d_base, rocke_b_const_i32(b, k))),
                values_f32[k],
                dtype);
        }
    }
}

/* ===================================================================== *
 *  build_fmha_fwd_splitkv_decode_segment
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_fmha_fwd_splitkv_decode_segment(
    rocke_fmha_kernel_builder_t* kb, const rocke_fmha_splitkv_decode_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        char reason[192];
        char kname[256];
        const rocke_fmha_common_spec_t* s;
        int H, ept, nqkv, num_seg, num_seg_log2;
        rocke_status_t st;
        rocke_ir_builder_t* b;

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_fmha_splitkv_decode_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            /* Init a throwaway builder so the caller can read the error / free. */
            if(rocke_fmha_splitkv_decode_kernel_name(spec, "seg", kname, sizeof(kname)) != ROCKE_OK)
            {
                return NULL;
            }
            if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) != ROCKE_OK)
            {
                return NULL;
            }
            rocke_i_set_err(rocke_fmha_kernel_builder_builder(kb),
                            ROCKE_ERR_VALUE,
                            "invalid splitkv_decode spec: %s",
                            reason);
            return NULL;
        }

        s = &spec->common;
        H = s->shape.head_size;
        if(H % ROCKE_SPLITKV_WARP_SIZE != 0)
        {
            if(rocke_fmha_splitkv_decode_kernel_name(spec, "seg", kname, sizeof(kname)) != ROCKE_OK)
            {
                return NULL;
            }
            if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) != ROCKE_OK)
            {
                return NULL;
            }
            rocke_i_set_err(rocke_fmha_kernel_builder_builder(kb),
                            ROCKE_ERR_VALUE,
                            "splitkv_decode warp body needs H %% %d == 0; got %d",
                            ROCKE_SPLITKV_WARP_SIZE,
                            H);
            return NULL;
        }
        ept = H / ROCKE_SPLITKV_WARP_SIZE;

        /* kb = FmhaKernelBuilder(spec.kernel_name("seg"), s) */
        if(rocke_fmha_splitkv_decode_kernel_name(spec, "seg", kname, sizeof(kname)) != ROCKE_OK)
        {
            return NULL;
        }
        st = rocke_fmha_kernel_builder_init(kb, kname, s);
        if(st != ROCKE_OK)
        {
            return NULL;
        }

        /* kb.block_size(WARP_SIZE) */
        rocke_fmha_kernel_builder_block_size(kb, ROCKE_SPLITKV_WARP_SIZE);
        rocke_splitkv_declare_segment_params(kb);
        b = rocke_fmha_kernel_builder_builder(kb);

        {
            rocke_value_t* Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
            rocke_value_t* K = rocke_fmha_kernel_builder_tensor(kb, "K");
            rocke_value_t* V = rocke_fmha_kernel_builder_tensor(kb, "V");
            rocke_value_t* seqlens_k_ptr = rocke_fmha_kernel_builder_ptr(kb, "seqlens_k");
            rocke_value_t* ws_m = rocke_fmha_kernel_builder_ptr(kb, "ws_m");
            rocke_value_t* ws_l = rocke_fmha_kernel_builder_ptr(kb, "ws_l");
            rocke_value_t* ws_acc = rocke_fmha_kernel_builder_ptr(kb, "ws_acc");
            rocke_value_t* scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
            rocke_value_t* stride_q_seq = rocke_fmha_kernel_builder_scalar(kb, "stride_q_seq");
            rocke_value_t* stride_q_head = rocke_fmha_kernel_builder_scalar(kb, "stride_q_head");

            rocke_value_t* seq_idx;
            rocke_value_t* head_idx;
            rocke_value_t* segment_idx;
            rocke_value_t* kv_head_idx;
            uint64_t nqkv_mult;
            int nqkv_shift;

            rocke_value_t* seqlen_k;
            rocke_value_t* seg_len_base;
            rocke_value_t* seg_start;
            rocke_value_t* seg_end_raw;
            rocke_value_t* seg_end;

            rocke_value_t* q_row;
            rocke_value_t* tid;
            rocke_value_t* lane_d_base;
            rocke_value_t* kv_head_k_off;
            rocke_value_t* kv_head_v_off;
            rocke_value_t* stride_k_tok;
            rocke_value_t* stride_v_tok;
            rocke_value_t* neg_inf;
            rocke_value_t* zero_f;
            rocke_value_t* q_lane[8];

            rocke_iter_arg_t iter_args[2 + 8];
            char acc_names[8][16]; /* "a" + worst-case %d + NUL */
            int num_iter_args;
            int k;

            rocke_for_t k_loop;
            rocke_value_t* m_final;
            rocke_value_t* l_final;
            rocke_value_t* acc_final[8];

            rocke_value_t* seg_stride;
            rocke_value_t* ws_idx;
            rocke_value_t* ws_idx_acc_base;
            rocke_value_t* is_lead;

            seq_idx = rocke_b_block_id_x(b);
            head_idx = rocke_b_block_id_y(b);
            segment_idx = rocke_b_block_id_z(b);

            /* nqkv = s.shape.num_queries_per_kv */
            nqkv = 0;
            if(rocke_fmha_shape_num_queries_per_kv(&s->shape, &nqkv) != ROCKE_OK)
            {
                rocke_i_set_err(b,
                                ROCKE_ERR_VALUE,
                                "splitkv_decode: num_query_heads not divisible by num_kv_heads");
                return NULL;
            }
            /* nqkv_mult, nqkv_shift = calculate_magic_numbers(nqkv)
             * kv_head_idx = do_magic_division(b, head_idx, nqkv_mult, nqkv_shift) */
            if(!rocke_calculate_magic_numbers(b, nqkv, &nqkv_mult, &nqkv_shift))
            {
                return NULL;
            }
            kv_head_idx = rocke_do_magic_division(b, head_idx, nqkv_mult, nqkv_shift);

            /* num_seg = num_segments; num_seg_log2 = bit_length-1 */
            num_seg = spec->num_segments;
            num_seg_log2 = rocke_splitkv_log2_pow2(num_seg);

            /* seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx) */
            seqlen_k = rocke_b_global_load_i32(b, seqlens_k_ptr, seq_idx, 0);
            /* seg_len_base = b.lshr(seqlen_k, const(num_seg_log2)) */
            seg_len_base = rocke_b_lshr(b, seqlen_k, rocke_b_const_i32(b, num_seg_log2));
            /* seg_start = b.mul(segment_idx, seg_len_base) */
            seg_start = rocke_b_mul(b, segment_idx, seg_len_base);
            /* seg_end_raw = b.add(seg_start, seg_len_base) */
            seg_end_raw = rocke_b_add(b, seg_start, seg_len_base);
            /* seg_end = select(cmp_ge(segment_idx+1, num_seg), seqlen_k, seg_end_raw)
             * Python evaluates the cmp_ge operands left-to-right: the
             * add(segment_idx, 1) sub-expression is built before the const(num_seg)
             * right operand. C function-argument evaluation order is unspecified,
             * so sequence the sub-expressions into temporaries to match the
             * Python op-emission (and SSA-numbering) order exactly. */
            {
                rocke_value_t* seg_p1 = rocke_b_add(b, segment_idx, rocke_b_const_i32(b, 1));
                rocke_value_t* seg_ge = rocke_b_cmp_ge(b, seg_p1, rocke_b_const_i32(b, num_seg));
                seg_end = rocke_b_select(b, seg_ge, seqlen_k, seg_end_raw);
            }

            /* q_row = seq_idx*stride_q_seq + head_idx*stride_q_head
             * Left-to-right: mul(seq_idx, stride_q_seq) is emitted before
             * mul(head_idx, stride_q_head) in Python; force that order here. */
            {
                rocke_value_t* q_mul_seq = rocke_b_mul(b, seq_idx, stride_q_seq);
                rocke_value_t* q_mul_head = rocke_b_mul(b, head_idx, stride_q_head);
                q_row = rocke_b_add(b, q_mul_seq, q_mul_head);
            }
            tid = rocke_b_thread_id_x(b);
            lane_d_base = rocke_b_mul(b, tid, rocke_b_const_i32(b, ept));

            /* kv_head_k_off = kv_head_idx*stride_head("k") + lane_d_base */
            kv_head_k_off = rocke_b_add(
                b,
                rocke_b_mul(b, kv_head_idx, rocke_fmha_kernel_builder_stride_head(kb, "k")),
                lane_d_base);
            kv_head_v_off = rocke_b_add(
                b,
                rocke_b_mul(b, kv_head_idx, rocke_fmha_kernel_builder_stride_head(kb, "v")),
                lane_d_base);
            stride_k_tok = rocke_fmha_kernel_builder_stride_token(kb, "k");
            stride_v_tok = rocke_fmha_kernel_builder_stride_token(kb, "v");

            neg_inf = rocke_b_const_f32(b, -1e30);
            zero_f = rocke_b_const_f32(b, 0.0);

            /* q_lane = load_lane_slice_f32(b, Q, q_row, lane_d_base, dtype, ept) */
            if(!rocke_b_load_lane_slice_f32(b, Q, q_row, lane_d_base, s->dtype, ept, q_lane))
            {
                return NULL;
            }

            /* iter_args = [("m", neg_inf), ("l", zero_f)] + [(f"a{k}", zero_f)] */
            iter_args[0].name = "m";
            iter_args[0].init = neg_inf;
            iter_args[1].name = "l";
            iter_args[1].init = zero_f;
            for(k = 0; k < ept; ++k)
            {
                snprintf(acc_names[k], sizeof(acc_names[k]), "a%d", k);
                iter_args[2 + k].name = acc_names[k];
                iter_args[2 + k].init = zero_f;
            }
            num_iter_args = 2 + ept;

            /* k_loop = b.scf_for_iter(seg_start, seg_end, const(1), iter_args,
             *                         iv_name="k_idx") */
            k_loop = rocke_b_scf_for_iter(b,
                                          seg_start,
                                          seg_end,
                                          rocke_b_const_i32(b, 1),
                                          iter_args,
                                          num_iter_args,
                                          "k_idx",
                                          false,
                                          true);
            rocke_b_region_enter(b, k_loop.body);
            {
                rocke_value_t* k_idx = k_loop.iv;
                rocke_value_t* m = k_loop.iter_vars[0];
                rocke_value_t* l = k_loop.iter_vars[1];
                rocke_value_t** acc_iter = &k_loop.iter_vars[2];

                rocke_value_t* k_row;
                rocke_value_t* v_row;
                rocke_value_t* k_lane[8];
                rocke_value_t* v_lane[8];
                rocke_value_t* partial;
                rocke_value_t* dot;
                rocke_value_t* score_log2;
                rocke_value_t* m_new;
                rocke_value_t* alpha;
                rocke_value_t* p;
                rocke_value_t* l_new;
                rocke_value_t* new_yields[2 + 8];

                /* k_row = k_idx*stride_k_tok + kv_head_k_off */
                k_row = rocke_b_add(b, rocke_b_mul(b, k_idx, stride_k_tok), kv_head_k_off);
                v_row = rocke_b_add(b, rocke_b_mul(b, k_idx, stride_v_tok), kv_head_v_off);

                /* k_lane / v_lane via load_lane_slice_f32 with zero lane offset. */
                if(!rocke_b_load_lane_slice_f32(
                       b, K, k_row, rocke_b_const_i32(b, 0), s->dtype, ept, k_lane))
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }
                if(!rocke_b_load_lane_slice_f32(
                       b, V, v_row, rocke_b_const_i32(b, 0), s->dtype, ept, v_lane))
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }

                /* partial = zero_f; for k: partial = fma(q_lane[k], k_lane[k], partial) */
                partial = zero_f;
                for(k = 0; k < ept; ++k)
                {
                    partial = rocke_b_fma(b, q_lane[k], k_lane[k], partial);
                }
                /* dot = warp_xor_reduce_sum(b, partial, stages=6) */
                dot = rocke_warp_xor_reduce_sum(b, partial, 6);
                /* score_log2 = b.fmul(dot, scale_log2) */
                score_log2 = rocke_b_fmul(b, dot, scale_log2);
                /* score_log2 = apply_attention_mask(b, score_log2, mask_mode,
                 *     k_idx=k_idx, query_pos=const(0), sliding_window=s.sliding_window)
                 * context_len/neg_inf default to None -> NULL. */
                score_log2 = rocke_apply_attention_mask(b,
                                                        score_log2,
                                                        rocke_splitkv_attn_mask_mode(s->mask_mode),
                                                        k_idx,
                                                        rocke_b_const_i32(b, 0),
                                                        s->sliding_window,
                                                        NULL,
                                                        NULL);
                if(score_log2 == NULL)
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }

                /* m_new = fmax(m, score_log2); alpha = exp2(m - m_new);
                 * p = exp2(score_log2 - m_new); l_new = fma(l, alpha, p) */
                m_new = rocke_b_fmax(b, m, score_log2);
                alpha = rocke_b_exp2(b, rocke_b_fsub(b, m, m_new));
                p = rocke_b_exp2(b, rocke_b_fsub(b, score_log2, m_new));
                l_new = rocke_b_fma(b, l, alpha, p);

                new_yields[0] = m_new;
                new_yields[1] = l_new;
                for(k = 0; k < ept; ++k)
                {
                    /* acc' = fma(p, v_lane[k], fmul(acc_iter[k], alpha)) */
                    new_yields[2 + k]
                        = rocke_b_fma(b, p, v_lane[k], rocke_b_fmul(b, acc_iter[k], alpha));
                }
                rocke_b_scf_yield(b, new_yields, num_iter_args);
            }
            rocke_b_region_leave(b);

            /* m_final = results[0]; l_final = results[1]; acc_final = results[2:] */
            m_final = (k_loop.op != NULL) ? k_loop.op->results[0] : NULL;
            l_final = (k_loop.op != NULL) ? k_loop.op->results[1] : NULL;
            for(k = 0; k < ept; ++k)
            {
                acc_final[k] = (k_loop.op != NULL) ? k_loop.op->results[2 + k] : NULL;
            }

            /* seg_stride = num_query_heads * batch */
            {
                rocke_value_t* ss_nqh = rocke_b_const_i32(b, s->shape.num_query_heads);
                rocke_value_t* ss_bat = rocke_b_const_i32(b, spec->batch);
                seg_stride = rocke_b_mul(b, ss_nqh, ss_bat);
            }
            /* ws_idx = segment_idx*seg_stride + (seq_idx*num_query_heads + head_idx)
             * Left-to-right: the mul(segment_idx, seg_stride) left operand is
             * emitted before the add(mul(seq_idx, num_query_heads), head_idx) right
             * operand in Python; sequence into temporaries to match. */
            {
                rocke_value_t* ws_seg = rocke_b_mul(b, segment_idx, seg_stride);
                rocke_value_t* ws_sh = rocke_b_add(
                    b,
                    rocke_b_mul(b, seq_idx, rocke_b_const_i32(b, s->shape.num_query_heads)),
                    head_idx);
                ws_idx = rocke_b_add(b, ws_seg, ws_sh);
            }
            /* if H pow2: ws_idx_acc_base = shl(ws_idx, log2(H)) else mul(ws_idx, H) */
            if((H & (H - 1)) == 0)
            {
                ws_idx_acc_base
                    = rocke_b_shl(b, ws_idx, rocke_b_const_i32(b, rocke_splitkv_log2_pow2(H)));
            }
            else
            {
                ws_idx_acc_base = rocke_b_mul(b, ws_idx, rocke_b_const_i32(b, H));
            }

            /* is_lead = cmp_eq(tid, 0); with scf_if(is_lead): store m, l (align=4) */
            is_lead = rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0));
            {
                rocke_if_t gate = rocke_b_scf_if(b, is_lead);
                rocke_b_region_enter(b, gate.then_region);
                rocke_b_global_store(b, ws_m, ws_idx, m_final, 4);
                rocke_b_global_store(b, ws_l, ws_idx, l_final, 4);
                rocke_b_region_leave(b);
            }

            /* for k: d = lane_d_base + k; global_store(ws_acc, ws_idx_acc_base + d,
             *                                          acc_final[k], align=4) */
            for(k = 0; k < ept; ++k)
            {
                rocke_value_t* d = rocke_b_add(b, lane_d_base, rocke_b_const_i32(b, k));
                rocke_b_global_store(
                    b, ws_acc, rocke_b_add(b, ws_idx_acc_base, d), acc_final[k], 4);
            }

            rocke_b_ret(b);
        }

        return rocke_fmha_kernel_builder_kernel(kb);
    });
}

/* ===================================================================== *
 *  build_fmha_fwd_splitkv_decode_reduce
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_fmha_fwd_splitkv_decode_reduce(
    rocke_fmha_kernel_builder_t* kb, const rocke_fmha_splitkv_decode_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        char reason[192];
        char kname[256];
        const rocke_fmha_common_spec_t* s;
        int H, ept;
        rocke_status_t st;
        rocke_ir_builder_t* b;

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_fmha_splitkv_decode_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            if(rocke_fmha_splitkv_decode_kernel_name(spec, "reduce", kname, sizeof(kname))
               != ROCKE_OK)
            {
                return NULL;
            }
            if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) != ROCKE_OK)
            {
                return NULL;
            }
            rocke_i_set_err(rocke_fmha_kernel_builder_builder(kb),
                            ROCKE_ERR_VALUE,
                            "invalid splitkv_decode spec: %s",
                            reason);
            return NULL;
        }

        s = &spec->common;
        H = s->shape.head_size;
        if(H % ROCKE_SPLITKV_WARP_SIZE != 0)
        {
            if(rocke_fmha_splitkv_decode_kernel_name(spec, "reduce", kname, sizeof(kname))
               != ROCKE_OK)
            {
                return NULL;
            }
            if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) != ROCKE_OK)
            {
                return NULL;
            }
            rocke_i_set_err(rocke_fmha_kernel_builder_builder(kb),
                            ROCKE_ERR_VALUE,
                            "splitkv_decode reduce needs H %% %d == 0; got %d",
                            ROCKE_SPLITKV_WARP_SIZE,
                            H);
            return NULL;
        }
        ept = H / ROCKE_SPLITKV_WARP_SIZE;

        /* kb = FmhaKernelBuilder(spec.kernel_name("reduce"), s) */
        if(rocke_fmha_splitkv_decode_kernel_name(spec, "reduce", kname, sizeof(kname)) != ROCKE_OK)
        {
            return NULL;
        }
        st = rocke_fmha_kernel_builder_init(kb, kname, s);
        if(st != ROCKE_OK)
        {
            return NULL;
        }

        rocke_fmha_kernel_builder_block_size(kb, ROCKE_SPLITKV_WARP_SIZE);
        rocke_splitkv_declare_reduce_params(kb);
        b = rocke_fmha_kernel_builder_builder(kb);

        {
            rocke_value_t* ws_m = rocke_fmha_kernel_builder_ptr(kb, "ws_m");
            rocke_value_t* ws_l = rocke_fmha_kernel_builder_ptr(kb, "ws_l");
            rocke_value_t* ws_acc = rocke_fmha_kernel_builder_ptr(kb, "ws_acc");
            rocke_value_t* O = rocke_fmha_kernel_builder_tensor(kb, "O");
            rocke_value_t* stride_o_seq = rocke_fmha_kernel_builder_scalar(kb, "stride_o_seq");
            rocke_value_t* stride_o_head = rocke_fmha_kernel_builder_scalar(kb, "stride_o_head");

            rocke_value_t* seq_idx;
            rocke_value_t* head_idx;
            rocke_value_t* seg_stride;
            rocke_value_t* tid;
            rocke_value_t* lane_d_base;
            rocke_value_t* neg_inf;
            rocke_value_t* zero_f;
            rocke_value_t* base_ml;

            rocke_for_t mx_loop;
            rocke_value_t* overall_max;
            rocke_for_t sum_loop;
            rocke_value_t* overall_expsum;
            rocke_value_t* safe_expsum;
            rocke_value_t* inv_l;

            int H_pow2;
            int H_shift;
            rocke_value_t* acc_per_lane[8];
            rocke_value_t* o_row;
            int k;

            seq_idx = rocke_b_block_id_x(b);
            head_idx = rocke_b_block_id_y(b);
            /* seg_stride = num_query_heads * batch */
            {
                rocke_value_t* ss_nqh = rocke_b_const_i32(b, s->shape.num_query_heads);
                rocke_value_t* ss_bat = rocke_b_const_i32(b, spec->batch);
                seg_stride = rocke_b_mul(b, ss_nqh, ss_bat);
            }
            tid = rocke_b_thread_id_x(b);
            lane_d_base = rocke_b_mul(b, tid, rocke_b_const_i32(b, ept));

            /* neg_inf = const_f32(-inf); zero_f = const_f32(0.0) */
            neg_inf = rocke_b_const_f32(b, -1.0 / 0.0);
            zero_f = rocke_b_const_f32(b, 0.0);

            /* base_ml = seq_idx*num_query_heads + head_idx */
            base_ml = rocke_b_add(
                b,
                rocke_b_mul(b, seq_idx, rocke_b_const_i32(b, s->shape.num_query_heads)),
                head_idx);

            /* Pass 1: overall_max. */
            {
                rocke_iter_arg_t ia[1];
                ia[0].name = "mx";
                ia[0].init = neg_inf;
                mx_loop = rocke_b_scf_for_iter(b,
                                               rocke_b_const_i32(b, 0),
                                               rocke_b_const_i32(b, spec->num_segments),
                                               rocke_b_const_i32(b, 1),
                                               ia,
                                               1,
                                               "s_mx",
                                               false,
                                               true);
                rocke_b_region_enter(b, mx_loop.body);
                {
                    rocke_value_t* sv = mx_loop.iv;
                    rocke_value_t* mx = mx_loop.iter_vars[0];
                    rocke_value_t* ws_idx = rocke_b_add(b, rocke_b_mul(b, sv, seg_stride), base_ml);
                    rocke_value_t* ms = rocke_b_global_load_f32(b, ws_m, ws_idx, 0);
                    rocke_value_t* yld = rocke_b_fmax(b, mx, ms);
                    rocke_b_scf_yield(b, &yld, 1);
                }
                rocke_b_region_leave(b);
                overall_max = (mx_loop.op != NULL) ? mx_loop.op->results[0] : NULL;
            }

            /* Pass 2: overall_expsum. */
            {
                rocke_iter_arg_t ia[1];
                ia[0].name = "den";
                ia[0].init = zero_f;
                sum_loop = rocke_b_scf_for_iter(b,
                                                rocke_b_const_i32(b, 0),
                                                rocke_b_const_i32(b, spec->num_segments),
                                                rocke_b_const_i32(b, 1),
                                                ia,
                                                1,
                                                "s_sum",
                                                false,
                                                true);
                rocke_b_region_enter(b, sum_loop.body);
                {
                    rocke_value_t* sv = sum_loop.iv;
                    rocke_value_t* den = sum_loop.iter_vars[0];
                    rocke_value_t* ws_idx = rocke_b_add(b, rocke_b_mul(b, sv, seg_stride), base_ml);
                    rocke_value_t* ms = rocke_b_global_load_f32(b, ws_m, ws_idx, 0);
                    rocke_value_t* ls = rocke_b_global_load_f32(b, ws_l, ws_idx, 0);
                    rocke_value_t* ms_finite = rocke_b_fcmp(b, "ogt", ms, neg_inf);
                    rocke_value_t* factor_raw = rocke_b_exp2(b, rocke_b_fsub(b, ms, overall_max));
                    rocke_value_t* factor = rocke_b_select(b, ms_finite, factor_raw, zero_f);
                    rocke_value_t* yld = rocke_b_fadd(b, den, rocke_b_fmul(b, ls, factor));
                    rocke_b_scf_yield(b, &yld, 1);
                }
                rocke_b_region_leave(b);
                overall_expsum = (sum_loop.op != NULL) ? sum_loop.op->results[0] : NULL;
            }
            /* safe_expsum = fcmp(oeq, overall_expsum, 0);
             * inv_l = select(safe_expsum, 0, rcp(overall_expsum)) */
            safe_expsum = rocke_b_fcmp(b, "oeq", overall_expsum, zero_f);
            inv_l = rocke_b_select(b, safe_expsum, zero_f, rocke_b_rcp(b, overall_expsum));

            /* Pass 3: per-lane reduce + normalise + write. */
            H_pow2 = (H % ROCKE_SPLITKV_WARP_SIZE == 0 && (H & (H - 1)) == 0);
            H_shift = rocke_splitkv_log2_pow2(H);
            for(k = 0; k < ept; ++k)
            {
                rocke_for_t acc_loop;
                char ivname[24]; /* "s_acc" + worst-case %d + NUL */
                rocke_iter_arg_t ia[1];
                char accname[16]; /* "ac"    + worst-case %d + NUL */
                snprintf(accname, sizeof(accname), "ac%d", k);
                snprintf(ivname, sizeof(ivname), "s_acc%d", k);
                ia[0].name = accname;
                ia[0].init = zero_f;
                acc_loop = rocke_b_scf_for_iter(b,
                                                rocke_b_const_i32(b, 0),
                                                rocke_b_const_i32(b, spec->num_segments),
                                                rocke_b_const_i32(b, 1),
                                                ia,
                                                1,
                                                ivname,
                                                false,
                                                true);
                rocke_b_region_enter(b, acc_loop.body);
                {
                    rocke_value_t* sv = acc_loop.iv;
                    rocke_value_t* ac = acc_loop.iter_vars[0];
                    rocke_value_t* ws_idx = rocke_b_add(b, rocke_b_mul(b, sv, seg_stride), base_ml);
                    rocke_value_t* ms = rocke_b_global_load_f32(b, ws_m, ws_idx, 0);
                    rocke_value_t* ms_finite = rocke_b_fcmp(b, "ogt", ms, neg_inf);
                    rocke_value_t* factor_raw = rocke_b_exp2(b, rocke_b_fsub(b, ms, overall_max));
                    rocke_value_t* factor = rocke_b_select(b, ms_finite, factor_raw, zero_f);
                    rocke_value_t* d = rocke_b_add(b, lane_d_base, rocke_b_const_i32(b, k));
                    /* ws_idx_acc_base_fn(sv): shl/mul of (sv*seg_stride + base_ml). */
                    rocke_value_t* acc_base_idx
                        = rocke_b_add(b, rocke_b_mul(b, sv, seg_stride), base_ml);
                    rocke_value_t* acc_base
                        = H_pow2 ? rocke_b_shl(b, acc_base_idx, rocke_b_const_i32(b, H_shift))
                                 : rocke_b_mul(b, acc_base_idx, rocke_b_const_i32(b, H));
                    rocke_value_t* ov
                        = rocke_b_global_load_f32(b, ws_acc, rocke_b_add(b, acc_base, d), 0);
                    rocke_value_t* yld = rocke_b_fadd(b, ac, rocke_b_fmul(b, ov, factor));
                    rocke_b_scf_yield(b, &yld, 1);
                }
                rocke_b_region_leave(b);
                /* acc_per_lane.append(fmul(acc_loop.results[0], inv_l)) */
                acc_per_lane[k] = rocke_b_fmul(
                    b, (acc_loop.op != NULL) ? acc_loop.op->results[0] : NULL, inv_l);
            }

            /* o_row = seq_idx*stride_o_seq + head_idx*stride_o_head
             * Left-to-right: mul(seq_idx, stride_o_seq) before
             * mul(head_idx, stride_o_head) to match Python op-emission order. */
            {
                rocke_value_t* o_mul_seq = rocke_b_mul(b, seq_idx, stride_o_seq);
                rocke_value_t* o_mul_head = rocke_b_mul(b, head_idx, stride_o_head);
                o_row = rocke_b_add(b, o_mul_seq, o_mul_head);
            }
            /* _store_lane_slice_f32_packed(b, O, o_row, lane_d_base, acc_per_lane,
             *                              dtype=s.dtype, ept=ept) */
            rocke_splitkv_store_lane_slice_f32_packed(
                b, O, o_row, lane_d_base, acc_per_lane, s->dtype, ept);

            rocke_b_ret(b);
        }

        return rocke_fmha_kernel_builder_kernel(kb);
    });
}

/* ===================================================================== *
 *  grids
 * ===================================================================== */

rocke_status_t rocke_fmha_splitkv_decode_segment_grid(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                      int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = spec->batch;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = spec->num_segments;
    return ROCKE_OK;
}

rocke_status_t rocke_fmha_splitkv_decode_reduce_grid(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                     int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = spec->batch;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  lower-to-.ll convenience
 * ===================================================================== */

static void rocke_splitkv_copy_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(m == NULL)
    {
        m = "splitkv_decode lower failed";
    }
    n = strlen(m);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, m, n);
    err[n] = '\0';
}

/* Shared driver: build (segment or reduce) then lower. is_segment selects which
 * kernel to build. */
static rocke_status_t rocke_splitkv_lower_to_llvm(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  bool is_segment,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* kernel;
    rocke_ir_builder_t* b;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_splitkv_copy_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = is_segment ? rocke_build_fmha_fwd_splitkv_decode_segment(&kb, spec, arch)
                        : rocke_build_fmha_fwd_splitkv_decode_reduce(&kb, spec, arch);
    b = rocke_fmha_kernel_builder_builder(&kb);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(b);
        rocke_splitkv_copy_err(err, err_cap, rocke_ir_builder_error(b));
        rocke_fmha_kernel_builder_free(&kb);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}

rocke_status_t
    rocke_fmha_splitkv_decode_segment_lower_to_llvm(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap)
{
    return rocke_splitkv_lower_to_llvm(spec, arch, flavor, true, out_ll, err, err_cap);
}

rocke_status_t
    rocke_fmha_splitkv_decode_reduce_lower_to_llvm(const rocke_fmha_splitkv_decode_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    return rocke_splitkv_lower_to_llvm(spec, arch, flavor, false, out_ll, err, err_cap);
}
