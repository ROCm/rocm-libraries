// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helper_ck_dsl.instances.common.attention_unified_selectors.c --
 *   C99 port of selected SELECTOR + descriptor + emit symbols from
 *   ck_dsl/instances/common/attention_unified.py.
 *
 * The host-side selectors reproduce the Python branch structure exactly
 * (same comparisons, same order, same gate predicates). The IR-emitting
 * helpers reproduce the Python ckc_b_* builder-call sequence byte-faithfully
 * (same ops, same order, same operands).
 *
 * Lifetime: every emitted node is arena-owned (ckc_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "ckc/helper_helper_ck_dsl.instances.common.attention_unified_selectors.h"
#include "ckc/helper_ck_dsl.helpers.transforms.h"
#include "ckc/ir.h"

/* ------------------------------------------------------- arch resolution */

/* Python: _resolve_attention_arch() -- query the running device arch, memoized
 * process-wide, falling back to "gfx950" when the device arch is unavailable
 * (CPU-only / cross-compile harnesses, which is exactly this static-port
 * context).
 *
 * TODO(port): wire a real runtime.hip_module.get_device_arch() query when the
 * C runtime surface exposes one. Until then this faithfully reproduces the
 * documented fallback (the only arch the tiled MFMA path supports by default)
 * and lets a host override the resolution via the setter below. */
static const char* g_resolved_attention_arch = NULL;

void ckc_unified_attn_set_resolved_arch(const char* arch) { g_resolved_attention_arch = arch; }

static const char* ckc_unified_attn_resolve_arch(void)
{
    return (g_resolved_attention_arch != NULL) ? g_resolved_attention_arch : "gfx950";
}

static bool arch_is(const char* want) { return strcmp(ckc_unified_attn_resolve_arch(), want) == 0; }

/* ----------------------------------------------------- num_queries_per_kv */

int ckc_unified_attn_num_queries_per_kv(ckc_ir_builder_t* b, const ckc_unified_attn_problem_t* p)
{
    if(p->num_kv_heads == 0 || (p->num_query_heads % p->num_kv_heads) != 0)
    {
        if(b != NULL && b->status == CKC_OK)
        {
            b->status = CKC_ERR_VALUE;
            (void)snprintf(b->err,
                           (size_t)CKC_ERR_MSG_CAP,
                           "num_query_heads must be divisible by num_kv_heads");
        }
        return 0;
    }
    return p->num_query_heads / p->num_kv_heads;
}

/* Internal convenience: same value without a builder (the selectors call the
 * property freely; the divisibility precondition is guaranteed for any problem
 * that reached the spec stage). */
static int nqpk(const ckc_unified_attn_problem_t* p)
{
    if(p->num_kv_heads == 0)
    {
        return 0;
    }
    return p->num_query_heads / p->num_kv_heads;
}

/* ----------------------------------------------------- gate predicates */
/* These mirror the Python private predicates (_enable_*). They are static so
 * the selectors stay byte-faithful; not part of the public ABI. */

static int select_2d_tile_size(const ckc_unified_attn_problem_t* p); /* fwd */

/* Python: _enable_combo_2d(problem). */
static bool enable_combo_2d(const ckc_unified_attn_problem_t* p)
{
    if(!arch_is("gfx950"))
    {
        return false;
    }
    if(strcmp(p->dtype, "bf16") != 0)
    {
        return false;
    }
    if(p->use_alibi || p->use_qq_bias || p->softcap > 0)
    {
        return false;
    }
    if(p->head_size != 64 || p->block_size != 32)
    {
        return false;
    }
    if(nqpk(p) != 8)
    {
        return false;
    }
    if(p->max_seqlen_q <= 256)
    {
        return false;
    }
    return true;
}

/* Python: _enable_transposed_qk_32x32(problem). */
static bool enable_transposed_qk_32x32(const ckc_unified_attn_problem_t* p)
{
    if(!arch_is("gfx950"))
    {
        return false;
    }
    if(enable_combo_2d(p))
    {
        return true;
    }
    /* fp16 and bf16 both ride the validated transposed-32x32 path. Gating this
       to bf16 only left fp16 d128/d64 prefill on a legacy path that is
       numerically wrong at d128 with KV >= 1024 (mirrors the Python selector
       _enable_transposed_qk_32x32). */
    if(strcmp(p->dtype, "bf16") != 0 && strcmp(p->dtype, "fp16") != 0)
    {
        return false;
    }
    if(p->use_fp8)
    {
        return false;
    }
    if(p->use_alibi || p->use_qq_bias)
    {
        return false;
    }
    if(p->softcap > 0 || p->use_sinks)
    {
        return false;
    }
    if(p->head_size != 64 && p->head_size != 128)
    {
        return false;
    }
    bool multi_batch     = (p->max_seqlen_q > 256) && (p->num_seqs >= 2);
    bool single_seq_hd64 = (p->head_size == 64) && (p->block_size == 16) && (p->num_seqs <= 1) &&
                           (nqpk(p) >= 4) && (p->max_seqlen_q > 768);
    if(!(multi_batch || single_seq_hd64))
    {
        return false;
    }
    if(p->sliding_window > 0 && !p->use_fp8)
    {
        return false;
    }
    return true;
}

/* Python: _enable_gfx942_small_q_narrow(problem). */
static bool enable_gfx942_small_q_narrow(const ckc_unified_attn_problem_t* p)
{
    return arch_is("gfx942") && (strcmp(p->dtype, "fp16") == 0 || strcmp(p->dtype, "bf16") == 0) &&
           !p->use_fp8 && (p->head_size == 64 || p->head_size == 128) &&
           (p->max_seqlen_q > 1 && p->max_seqlen_q <= 768) && p->sliding_window == 0 &&
           !p->use_sinks && p->softcap == 0 && !p->use_alibi && !p->use_qq_bias;
}

/* Python: _enable_gfx942_fp16_flash(problem). */
static bool enable_gfx942_fp16_flash(const ckc_unified_attn_problem_t* p)
{
    return arch_is("gfx942") && (p->head_size == 64 || p->head_size == 128) &&
           strcmp(p->dtype, "fp16") == 0 && !p->use_fp8 && p->sliding_window == 0 &&
           !p->use_sinks && p->softcap == 0 && !p->use_alibi && !p->use_qq_bias &&
           !enable_gfx942_small_q_narrow(p);
}

/* Python: _enable_gfx942_d128_fp16_flash(problem). */
static bool enable_gfx942_d128_fp16_flash(const ckc_unified_attn_problem_t* p)
{
    return enable_gfx942_fp16_flash(p) && p->head_size == 128;
}

/* Python: _enable_gfx942_l4(problem) -- alias for the D128 fp16 flash family. */
static bool enable_gfx942_l4(const ckc_unified_attn_problem_t* p)
{
    return enable_gfx942_d128_fp16_flash(p);
}

/* Python: _gfx942_flash_wide_setting(). The HIPDNN_GFX942_FLASH_WIDE env knob
 * defaults to 4 (off/2/4 overrides). The static port honours the default; env
 * override is a host-runtime concern.
 *
 * TODO(port): consult getenv("HIPDNN_GFX942_FLASH_WIDE") for off/2/4 once the
 * port wires environment knobs. Until then returns the documented default. */
static int gfx942_flash_wide_setting(void) { return 4; }

/* Python: _select_gfx942_flash_num_warps(problem). */
static int select_gfx942_flash_num_warps(const ckc_unified_attn_problem_t* p)
{
    (void)p;
    int wide = gfx942_flash_wide_setting();
    return (wide == 2 || wide == 4) ? wide : 1;
}

/* ----------------------------------------------------- select_2d_tile_size */

/* Python: _enable_gfx942_fp16_flash gate inside _select_2d_tile_size for the
 * D64 force-T=64 branch. */
static int select_2d_tile_size(const ckc_unified_attn_problem_t* p)
{
    /* Sliding-window long-prefill FP8 exception. */
    if(p->use_fp8 && p->sliding_window > 0 && p->max_seqlen_q > 256)
    {
        return p->block_size;
    }
    /* gfx942 D64. */
    if(arch_is("gfx942") && p->head_size == 64)
    {
        if(enable_gfx942_fp16_flash(p))
        {
            return 64;
        }
        return p->block_size;
    }
    /* gfx942 D128 (ALL dtypes): T=64. */
    if(arch_is("gfx942") && p->head_size == 128)
    {
        return 64;
    }
    /* bf16 transposed-combo sliding-window. */
    if(enable_combo_2d(p) && p->sliding_window > 0)
    {
        return p->block_size;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8 &&
       strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4)
    {
        if(p->max_seqlen_q >= 512 && p->max_seqlen_q <= 768)
        {
            return 128;
        }
        if(p->max_seqlen_q > 64)
        {
            return 64;
        }
    }
    return 2 * p->block_size;
}

int ckc_unified_attn_select_2d_tile_size(const ckc_unified_attn_problem_t* p)
{
    return select_2d_tile_size(p);
}

/* ----------------------------------------------------- select_2d_num_warps */

int ckc_unified_attn_select_2d_num_warps(const ckc_unified_attn_problem_t* p)
{
    int target;

    /* Small/medium gfx942 prefill light narrow path. */
    if(enable_gfx942_small_q_narrow(p))
    {
        return (nqpk(p) == 1) ? 1 : 2;
    }
    /* gfx942 D128 fp16 flash/L4. */
    if(enable_gfx942_l4(p))
    {
        return select_gfx942_flash_num_warps(p);
    }
    /* gfx942 D64 oracle. */
    if(arch_is("gfx942") && p->head_size == 64)
    {
        return 4;
    }
    if(enable_combo_2d(p))
    {
        int t2 = (p->sliding_window > 0 && !p->use_fp8) ? 2 : 4;
        int HD = p->head_size;
        int BS = p->block_size;
        int T  = select_2d_tile_size(p);
        while(t2 > 1)
        {
            if((T * HD) < 64 * t2 * 8)
            {
                t2 /= 2;
                continue;
            }
            if((64 * 8) / HD > BS)
            {
                t2 /= 2;
                continue;
            }
            break;
        }
        return (t2 > 1) ? t2 : 1;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8 &&
       strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4)
    {
        if(p->max_seqlen_q <= 128)
        {
            target = 1;
        }
        else if(p->max_seqlen_q <= 768)
        {
            target = 2;
        }
        else
        {
            target = 4;
        }
    }
    else if(p->max_seqlen_q <= 64)
    {
        target = 1;
    }
    else if(p->max_seqlen_q <= 128)
    {
        target = 2;
    }
    else if(p->max_seqlen_q <= 256)
    {
        target = 4;
    }
    else if(p->num_seqs <= 1)
    {
        target = 2;
    }
    else if(nqpk(p) == 1 && p->head_size == 64 && !enable_combo_2d(p))
    {
        target = (p->max_seqlen_q <= 512 || p->max_seqlen_q >= 1536) ? 2 : 4;
    }
    else
    {
        target = 4;
    }

    {
        int HD               = p->head_size;
        int BS               = p->block_size;
        int T                = select_2d_tile_size(p);
        const int WORK_BYTES = 2;
        /* Step down until all constraints are satisfied. */
        while(target > 1)
        {
            int THREADS = 64 * target;
            int BLOCK_M = 16 * target;
            int per_wave_tokens;
            int lds_bytes;
            if((T * HD) < THREADS * 8)
            {
                target /= 2;
                continue;
            }
            per_wave_tokens = (64 * 8) / HD;
            if(per_wave_tokens > BS)
            {
                target /= 2;
                continue;
            }
            lds_bytes = BLOCK_M * HD * WORK_BYTES + 2 * T * HD * WORK_BYTES +
                        2 * T * HD * WORK_BYTES + BLOCK_M * T * WORK_BYTES + BLOCK_M * HD * 4;
            if(lds_bytes <= 96 * 1024)
            {
                break;
            }
            target /= 2;
        }
    }
    return (target > 1) ? target : 1;
}

/* ------------------------------------------------ select_2d_block_m_per_warp */

int ckc_unified_attn_select_2d_block_m_per_warp(const ckc_unified_attn_problem_t* p)
{
    if(enable_gfx942_small_q_narrow(p))
    {
        return 16;
    }
    if(arch_is("gfx942") && p->head_size == 64)
    {
        return 32;
    }
    if(enable_gfx942_l4(p))
    {
        return 32;
    }
    if(enable_transposed_qk_32x32(p)) /* includes _enable_combo_2d */
    {
        return 32;
    }
    if(p->use_fp8 && p->max_seqlen_q > 256 && p->num_seqs >= 2)
    {
        return 32;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8 &&
       strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4 && p->max_seqlen_q > 768 &&
       p->sliding_window == 0 && p->softcap == 0 && !p->use_sinks && !p->use_alibi &&
       !p->use_qq_bias)
    {
        return 32;
    }
    return 16;
}

/* ----------------------------------------------------- kv_storage_dtype */

const char* ckc_unified_attn_kv_storage_dtype(const ckc_unified_attn_problem_t* p)
{
    return p->use_fp8 ? "fp8e4m3" : NULL;
}

/* ----------------------------------------------------------- magic div */

ckc_value_t* ckc_unified_attn_magic_div(ckc_ir_builder_t* b, ckc_value_t* dividend, int divisor)
{
    uint64_t mult = 0;
    int shift     = 0;
    if(!ckc_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        return NULL;
    }
    return ckc_do_magic_division(b, dividend, mult, shift);
}

bool ckc_unified_attn_magic_div_mod(ckc_ir_builder_t* b,
                                    ckc_value_t* dividend,
                                    int divisor,
                                    ckc_value_t** out_quotient,
                                    ckc_value_t** out_remainder)
{
    ckc_value_t* quotient = ckc_unified_attn_magic_div(b, dividend, divisor);
    ckc_value_t* remainder =
        ckc_b_sub(b, dividend, ckc_b_mul(b, quotient, ckc_b_const_i32(b, divisor)));
    if(out_quotient != NULL)
    {
        *out_quotient = quotient;
    }
    if(out_remainder != NULL)
    {
        *out_remainder = remainder;
    }
    return b != NULL && b->status == CKC_OK;
}

/* --------------------------------------------------------- descriptors */

ckc_tensor_descriptor_t* ckc_unified_attn_q_descriptor(ckc_ir_builder_t* b,
                                                       const ckc_unified_attn_problem_t* p)
{
    int lengths[3];
    static const char* const coord_names[3] = {"token", "head", "dim"};
    lengths[0]                              = p->max_seqlen_q + 1;
    lengths[1]                              = p->num_query_heads;
    lengths[2]                              = p->head_size;
    return ckc_tensor_descriptor_naive(b, "Q", lengths, 3, NULL, coord_names, 3);
}

ckc_unified_attn_paged_kv_descriptor_t
ckc_unified_attn_paged_kv_descriptor(const ckc_unified_attn_problem_t* p)
{
    ckc_unified_attn_paged_kv_descriptor_t d;
    d.block_size = p->block_size;
    d.stride_0   = p->block_size * p->num_kv_heads * p->head_size;
    d.stride_1   = p->num_kv_heads * p->head_size;
    d.stride_2   = p->head_size;
    d.stride_3   = 1;
    return d;
}

ckc_value_t* ckc_unified_attn_paged_kv_offset(ckc_ir_builder_t* b,
                                              const ckc_unified_attn_paged_kv_descriptor_t* d,
                                              ckc_value_t* physical_block,
                                              ckc_value_t* token_in_block,
                                              ckc_value_t* kv_head,
                                              ckc_value_t* dim)
{
    ckc_value_t* off = ckc_b_mul(b, physical_block, ckc_b_const_i32(b, d->stride_0));
    off = ckc_b_add(b, off, ckc_b_mul(b, token_in_block, ckc_b_const_i32(b, d->stride_1)));
    off = ckc_b_add(b, off, ckc_b_mul(b, kv_head, ckc_b_const_i32(b, d->stride_2)));
    off = ckc_b_add(b, off, ckc_b_mul(b, dim, ckc_b_const_i32(b, d->stride_3)));
    return off;
}

bool ckc_unified_attn_segm_descriptors(ckc_ir_builder_t* b,
                                       const ckc_unified_attn_problem_t* p,
                                       int num_segments,
                                       ckc_tensor_descriptor_t** out_ml,
                                       ckc_tensor_descriptor_t** out_output)
{
    int ml_lengths[3];
    int out_lengths[4];
    static const char* const ml_coords[3]  = {"token", "head", "seg"};
    static const char* const out_coords[4] = {"token", "head", "seg", "dim"};
    ckc_tensor_descriptor_t* ml;
    ckc_tensor_descriptor_t* out;

    ml_lengths[0] = p->max_seqlen_q + 1;
    ml_lengths[1] = p->num_query_heads;
    ml_lengths[2] = num_segments;
    ml            = ckc_tensor_descriptor_naive(b, "segm_ml", ml_lengths, 3, NULL, ml_coords, 3);

    out_lengths[0] = p->max_seqlen_q + 1;
    out_lengths[1] = p->num_query_heads;
    out_lengths[2] = num_segments;
    out_lengths[3] = p->head_size;
    out = ckc_tensor_descriptor_naive(b, "segm_output", out_lengths, 4, NULL, out_coords, 4);

    if(out_ml != NULL)
    {
        *out_ml = ml;
    }
    if(out_output != NULL)
    {
        *out_output = out;
    }
    return ml != NULL && out != NULL;
}

/* ------------------------------------------------------- IR emit helpers */

bool ckc_unified_attn_physical_block_and_token(ckc_ir_builder_t* b,
                                               const ckc_unified_attn_problem_t* p,
                                               ckc_value_t* block_tables,
                                               ckc_value_t* seq_idx,
                                               ckc_value_t* kpos,
                                               ckc_value_t** out_physical,
                                               ckc_value_t** out_token_in_block)
{
    ckc_value_t* block_idx      = NULL;
    ckc_value_t* token_in_block = NULL;
    int max_blocks;
    ckc_value_t* physical;

    if(!ckc_unified_attn_magic_div_mod(b, kpos, p->block_size, &block_idx, &token_in_block))
    {
        return false;
    }
    max_blocks = (p->max_seqlen_k + p->block_size - 1) / p->block_size;
    physical   = ckc_b_global_load_i32(
        b,
        block_tables,
        ckc_b_add(b, ckc_b_mul(b, seq_idx, ckc_b_const_i32(b, max_blocks)), block_idx),
        0 /* align default */);

    if(out_physical != NULL)
    {
        *out_physical = physical;
    }
    if(out_token_in_block != NULL)
    {
        *out_token_in_block = token_in_block;
    }
    return b != NULL && b->status == CKC_OK;
}

ckc_value_t* ckc_unified_attn_emit_qk_score(ckc_ir_builder_t* b,
                                            const ckc_unified_attn_problem_t* p,
                                            const ckc_type_t* dtype,
                                            ckc_value_t* query,
                                            ckc_value_t* key,
                                            ckc_value_t* block_tables,
                                            ckc_value_t* seq_idx,
                                            ckc_value_t* q_tok,
                                            ckc_value_t* q_head,
                                            ckc_value_t* kv_head,
                                            ckc_value_t* kpos,
                                            ckc_value_t* scale,
                                            ckc_value_t* rcp_ln2)
{
    const int VEC               = 8;
    ckc_value_t* score          = ckc_b_const_f32(b, 0.0);
    ckc_value_t* physical       = NULL;
    ckc_value_t* token_in_block = NULL;
    ckc_tensor_descriptor_t* q_desc;
    ckc_unified_attn_paged_kv_descriptor_t kv_desc;
    ckc_value_t* q_off_base = NULL;
    ckc_value_t* k_off_base;
    int n_vec;
    int d8;
    int d;

    if(!ckc_unified_attn_physical_block_and_token(
           b, p, block_tables, seq_idx, kpos, &physical, &token_in_block))
    {
        return NULL;
    }
    q_desc  = ckc_unified_attn_q_descriptor(b, p);
    kv_desc = ckc_unified_attn_paged_kv_descriptor(p);

    /* q_off_base, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=const_i32(0)) */
    {
        const char* in_names[3] = {"token", "head", "dim"};
        ckc_value_t* in_values[3];
        ckc_value_t* valid = NULL;
        in_values[0]       = q_tok;
        in_values[1]       = q_head;
        in_values[2]       = ckc_b_const_i32(b, 0);
        if(!ckc_transforms_descriptor_offset(
               b, q_desc, in_names, in_values, 3, &q_off_base, &valid))
        {
            return NULL;
        }
    }
    k_off_base = ckc_unified_attn_paged_kv_offset(
        b, &kv_desc, physical, token_in_block, kv_head, ckc_b_const_i32(b, 0));

    n_vec = p->head_size / VEC;
    for(d8 = 0; d8 < n_vec; ++d8)
    {
        ckc_value_t* d_base = ckc_b_const_i32(b, d8 * VEC);
        ckc_value_t* qv =
            ckc_b_global_load_vN(b, query, ckc_b_add(b, q_off_base, d_base), dtype, VEC, 16);
        ckc_value_t* kv =
            ckc_b_global_load_vN(b, key, ckc_b_add(b, k_off_base, d_base), dtype, VEC, 16);
        int i;
        for(i = 0; i < VEC; ++i)
        {
            score = ckc_b_fadd(b,
                               score,
                               ckc_b_fmul(b,
                                          ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, qv, i)),
                                          ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, kv, i))));
        }
    }
    /* Tail scalar fold for head_size not a multiple of VEC (empty for the
     * supported {64,128,256} head sizes). */
    for(d = n_vec * VEC; d < p->head_size; ++d)
    {
        ckc_value_t* d_v   = ckc_b_const_i32(b, d);
        ckc_value_t* q_off = NULL;
        ckc_value_t* k_off;
        ckc_value_t* qv_s;
        ckc_value_t* kv_s;
        {
            const char* in_names[3] = {"token", "head", "dim"};
            ckc_value_t* in_values[3];
            ckc_value_t* valid = NULL;
            in_values[0]       = q_tok;
            in_values[1]       = q_head;
            in_values[2]       = d_v;
            if(!ckc_transforms_descriptor_offset(b, q_desc, in_names, in_values, 3, &q_off, &valid))
            {
                return NULL;
            }
        }
        k_off =
            ckc_unified_attn_paged_kv_offset(b, &kv_desc, physical, token_in_block, kv_head, d_v);
        qv_s  = ckc_b_cast_to_f32(b, ckc_b_global_load(b, query, q_off, dtype, 2));
        kv_s  = ckc_b_cast_to_f32(b, ckc_b_global_load(b, key, k_off, dtype, 2));
        score = ckc_b_fadd(b, score, ckc_b_fmul(b, qv_s, kv_s));
    }
    return ckc_b_fmul(b, ckc_b_fmul(b, score, scale), rcp_ln2);
}

ckc_value_t* ckc_unified_attn_emit_v_load(ckc_ir_builder_t* b,
                                          const ckc_unified_attn_problem_t* p,
                                          const ckc_type_t* dtype,
                                          ckc_value_t* value,
                                          ckc_value_t* block_tables,
                                          ckc_value_t* seq_idx,
                                          ckc_value_t* kv_head,
                                          ckc_value_t* kpos,
                                          ckc_value_t* dim)
{
    ckc_value_t* physical       = NULL;
    ckc_value_t* token_in_block = NULL;
    ckc_unified_attn_paged_kv_descriptor_t kv_desc;
    ckc_value_t* v_off;

    if(!ckc_unified_attn_physical_block_and_token(
           b, p, block_tables, seq_idx, kpos, &physical, &token_in_block))
    {
        return NULL;
    }
    kv_desc = ckc_unified_attn_paged_kv_descriptor(p);
    v_off   = ckc_unified_attn_paged_kv_offset(b, &kv_desc, physical, token_in_block, kv_head, dim);
    return ckc_b_cast_to_f32(b, ckc_b_global_load(b, value, v_off, dtype, 2));
}
