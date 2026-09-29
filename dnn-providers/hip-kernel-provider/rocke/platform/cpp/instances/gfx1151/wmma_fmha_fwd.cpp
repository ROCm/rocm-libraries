// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_gfx1151_wmma_fmha_fwd.c -- C99 port of
 * rocke/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * Byte-identical builder-call sequence vs the Python build_wmma_fmha_fwd: a raw
 * IRBuilder declares the same params in the same order (_declare_params), bakes
 * the same max_workgroup_size attr, decodes the (seqlen_q//16, num_query_heads,
 * batch) grid the same way, computes the same GQA kv_head + per-batch offsets,
 * and calls the already-ported helper rocke_mfma_attention_fwd_inner_body with the
 * same operands / attrs (incl. wmma_v_lds_stage), then b.ret(). All the wave32
 * QK->softmax->PV IR emission is delegated to that helper (which dispatches to
 * the WMMA wave32 inner body on the RDNA target); this file is the thin
 * spec->kernel adapter plus a lower-to-.ll convenience.
 */

#include "rocke/instance_gfx1151_wmma_fmha_fwd.h"

#include <bit>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.wmma_swapqk.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define WMMA_FMHA_DEFAULT_NAME "rocke_wmma_fmha_fwd"
#define WMMA_FMHA_DEFAULT_ARCH "gfx1151"

/* ----- small helpers ----- */

static void wmma_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

/* Canonical helper dtype, or NULL for an unsupported spec dtype. */
static const char* wmma_dtype(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    if(spec == NULL || spec->dtype == NULL)
        return NULL;
    if(strcmp(spec->dtype, "fp16") == 0 || strcmp(spec->dtype, "f16") == 0)
        return "f16";
    if(strcmp(spec->dtype, "bf16") == 0)
        return "bf16";
    return NULL;
}

static bool wmma_valid_layout(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    if(spec == NULL || spec->layout == NULL)
        return false;
    if(strcmp(spec->layout, "paged") == 0)
        return spec->page_block_size > 0
               && (spec->page_block_size & (spec->page_block_size - 1)) == 0;
    return (strcmp(spec->layout, "dense") == 0 || strcmp(spec->layout, "ragged") == 0)
           && spec->page_block_size == 0;
}

static bool wmma_valid_kv_dtype(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return spec != NULL && spec->kv_dtype != NULL
           && (spec->kv_dtype[0] == '\0' || strcmp(spec->kv_dtype, "fp8e4m3") == 0);
}

/* WmmaFmhaFwdSpec.kv_heads property: num_kv_heads or num_query_heads. */
static int wmma_kv_heads(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return spec->num_kv_heads != 0 ? spec->num_kv_heads : spec->num_query_heads;
}

/* Map the shared FMHA mask enum to the attention-helper mask enum. WMMA FMHA
 * supports only NONE / CAUSAL (validated up front); anything else => NONE. */
static rocke_attn_mask_mode_t wmma_to_attn_mask(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_NONE:
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* --------------------------------------------------------------------------- *
 * rocke_wmma_fmha_fwd_spec_default
 * --------------------------------------------------------------------------- */
rocke_wmma_fmha_fwd_spec_t rocke_wmma_fmha_fwd_spec_default(void)
{
    rocke_wmma_fmha_fwd_spec_t s;
    s.head_size = 0;
    s.num_query_heads = 0;
    s.num_kv_heads = 0;
    s.mask_mode = ROCKE_FMHA_MASK_NONE;
    s.v_lds_stage = false;
    s.sliding_window = 0;
    s.name = WMMA_FMHA_DEFAULT_NAME;
    s.dtype = "fp16";
    s.causal_bottom_right = false;
    s.query_tail = false;
    s.kv_tail = false;
    s.use_softcap = false;
    s.use_sinks = false;
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.layout = "dense";
    s.page_block_size = 0;
    s.kv_dtype = "";
    s.transposed_qk = false;
    s.block_n = 32;
    s.num_waves = 1;
    return s;
}

/* --------------------------------------------------------------------------- *
 * WmmaFmhaFwdSpec.kernel_name()
 *
 * kernel_name_join(name, "wmma16x16x16", "H{hd}", "HQ{hq}", "HK{kv_heads}",
 *   canonical dtype, mask_mode, "vlds" if v_lds_stage else "vgather").
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap)
{
    const char* name;
    const char* mask;
    char h[32], hq[32], hk[32], window[32], page[32], block_n[32], waves[32];
    const char* parts[13];
    const char* dtype = wmma_dtype(spec);

    if(spec == NULL || out == NULL || dtype == NULL
       || !wmma_valid_layout(spec) || !wmma_valid_kv_dtype(spec))
    {
        return ROCKE_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : WMMA_FMHA_DEFAULT_NAME;
    mask = rocke_fmha_mask_mode_name(spec->mask_mode);
    if(mask == NULL)
    {
        mask = "none";
    }

    snprintf(h, sizeof(h), "H%d", spec->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", spec->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", wmma_kv_heads(spec));

    parts[0] = spec->transposed_qk ? "wmma_swapqk" : "wmma16x16x16";
    parts[1] = h;
    parts[2] = hq;
    parts[3] = hk;
    parts[4] = strcmp(dtype, "bf16") == 0 ? "bf16" : "fp16";
    parts[5] = spec->causal_bottom_right ? "causal_br" : mask;
    parts[6] = spec->v_lds_stage ? "vlds" : "vgather";

    size_t num_parts = 7;
    if(spec->sliding_window > 0)
    {
        snprintf(window, sizeof(window), "sw%d", spec->sliding_window);
        parts[num_parts++] = window;
    }
    const bool packed = strcmp(spec->layout, "dense") != 0;
    if(packed)
        parts[num_parts++] = spec->layout;
    if(strcmp(spec->layout, "paged") == 0)
    {
        snprintf(page, sizeof(page), "bs%d", spec->page_block_size);
        parts[num_parts++] = page;
    }
    if(spec->kv_dtype[0] != '\0')
        parts[num_parts++] = "kvfp8e4m3";
    if(spec->transposed_qk)
    {
        snprintf(block_n, sizeof(block_n), "bn%d", spec->block_n);
        snprintf(waves, sizeof(waves), "w%d", spec->num_waves);
        parts[num_parts++] = block_n;
        parts[num_parts++] = waves;
    }
    const char* flag_names[] = {"qtail", "kvtail", "softcap", "sinks", "alibi", "qqbias"};
    const int flag_on[] = {spec->query_tail || packed, spec->kv_tail || packed,
                          spec->use_softcap, spec->use_sinks, spec->use_alibi, spec->use_qq_bias};
    return rocke_kernel_name_join(name, parts, num_parts, flag_names, flag_on, 6, out, out_cap, NULL);
}

/* --------------------------------------------------------------------------- *
 * is_valid_spec(spec, arch)
 *
 * Python:
 *   target = ArchTarget.from_gfx(arch)              # KeyError -> reject
 *   op = target.mma.by_op_id(_WMMA_OP_ID)
 *   if op is None or op.family != "wmma": reject
 *   if target.wave_size != op.wave_size: reject
 *   if spec.head_size % 16 != 0: reject
 *   if num_kv_heads and num_query_heads % num_kv_heads != 0: reject
 *   bytes_lds = BLOCK_M*BLOCK_K*2 (+ BLOCK_M*head_size*2 if v_lds_stage)
 *   if not target.fits_lds(bytes_lds): reject
 *   return True, "ok"
 * --------------------------------------------------------------------------- */
bool rocke_wmma_fmha_fwd_is_valid_spec(const rocke_wmma_fmha_fwd_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_mmaop_t* op;
    long bytes_lds;
    char buf[256];

    if(spec == NULL)
    {
        wmma_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = WMMA_FMHA_DEFAULT_ARCH;
    }
    const char* dtype = wmma_dtype(spec);
    if(dtype == NULL)
    {
        wmma_set_reason(reason, reason_cap, "WMMA FMHA dtype must be fp16/f16 or bf16");
        return false;
    }
    if(!wmma_valid_kv_dtype(spec))
    {
        wmma_set_reason(reason, reason_cap, "KV storage must match Q or use OCP fp8e4m3");
        return false;
    }
    if(spec->mask_mode != ROCKE_FMHA_MASK_NONE && spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL)
    {
        wmma_set_reason(reason, reason_cap, "WMMA FMHA supports none/causal masking");
        return false;
    }
    if(spec->causal_bottom_right && spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL)
    {
        wmma_set_reason(reason, reason_cap, "bottom-right alignment requires causal masking");
        return false;
    }
    if(spec->sliding_window < 0
       || (spec->sliding_window > 0 && spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL))
    {
        wmma_set_reason(reason, reason_cap, "sliding-window attention requires a nonnegative width and causal masking");
        return false;
    }
    if(!wmma_valid_layout(spec))
    {
        wmma_set_reason(reason, reason_cap, "layout must be dense/ragged with page size zero, or paged with a positive power-of-two page size");
        return false;
    }
    if(spec->transposed_qk)
    {
        if(strcmp(arch, "gfx1151") != 0 || strcmp(dtype, "f16") != 0
           || (spec->head_size != 64 && spec->head_size != 128)
           || (spec->block_n != 32 && spec->block_n != 64)
           || (spec->num_waves != 1 && spec->num_waves != 2))
        {
            wmma_set_reason(reason, reason_cap, "transposed QK requires gfx1151 FP16 D64/D128, block_n 32/64 and one or two waves");
            return false;
        }
        if(strcmp(spec->layout, "dense") != 0 || spec->kv_dtype[0] != '\0'
           || spec->query_tail || spec->kv_tail
           || spec->v_lds_stage || spec->sliding_window || spec->use_softcap
           || spec->use_sinks || spec->use_alibi || spec->use_qq_bias)
        {
            wmma_set_reason(reason, reason_cap, "transposed QK requires aligned dense inputs without extra score features");
            return false;
        }
    }
    else if(spec->block_n != 32 || spec->num_waves != 1)
    {
        wmma_set_reason(reason, reason_cap, "block_n and num_waves are transposed-QK options");
        return false;
    }

    /* target = ArchTarget.from_gfx(arch) -- KeyError path. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown arch '%s'", arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    const bool bf16 = strcmp(dtype, "bf16") == 0;
    const char* op_id = strcmp(arch, "gfx1201") == 0
                           ? (bf16 ? "wmma_gfx12_f32_16x16x16_bf16"
                                   : "wmma_gfx12_f32_16x16x16_f16")
                           : (bf16 ? "wmma_f32_16x16x16_bf16"
                                   : "wmma_f32_16x16x16_f16");
    op = rocke_archtarget_by_op_id(target, op_id);
    if(op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WMMA %s atom absent on %s (WMMA is an RDNA gfx11/gfx12 "
                 "instruction; this kernel needs a wave32 RDNA target)",
                 op_id,
                 arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* wave-size agreement (WMMA atom wave32 vs the target). */
    if(target->wave_size != op->wave_size)
    {
        snprintf(buf,
                 sizeof(buf),
                 "arch wave size %d != WMMA atom wave size %d on %s",
                 target->wave_size,
                 op->wave_size,
                 arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* head_size % 16 != 0 */
    if(spec->head_size % 16 != 0)
    {
        snprintf(buf, sizeof(buf), "head_size must be a multiple of 16 (got %d)", spec->head_size);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* GQA requires an integral number of query heads per KV head. Otherwise
     * kv_head = head / (num_query_heads / num_kv_heads) can address beyond the
     * available KV heads. num_kv_heads == 0 denotes MHA. */
    if(spec->num_kv_heads != 0 && (spec->num_query_heads % spec->num_kv_heads) != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "num_query_heads must be a multiple of num_kv_heads for GQA "
                 "(got num_query_heads=%d, num_kv_heads=%d)",
                 spec->num_query_heads,
                 spec->num_kv_heads);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* Both fp16 and bf16 use two bytes per P/V staging element. */
    bytes_lds = (long)ROCKE_WMMA_FMHA_FWD_BLOCK_M * ROCKE_WMMA_FMHA_FWD_BLOCK_K * 2;
    if(spec->v_lds_stage)
    {
        bytes_lds += (long)ROCKE_WMMA_FMHA_FWD_BLOCK_M * spec->head_size * 2;
    }
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        snprintf(buf, sizeof(buf), "LDS budget %ld > cap on %s", bytes_lds, arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    wmma_set_reason(reason, reason_cap, "ok");
    return true;
}

/* --------------------------------------------------------------------------- *
 * _declare_params(b): the gfx1151 WMMA FMHA kernel ABI.
 *
 * Q/K/V/O ptrs, scale_log2/seqlen_q/seqlen_k scalars, then the four (token,
 * head) element-stride pairs, in the exact Python declaration order. The named
 * params are recovered later via rocke_b_get_param. */
static void wmma_declare_params(rocke_ir_builder_t* b, const rocke_wmma_fmha_fwd_spec_t* spec)
{
    const char* dtype = wmma_dtype(spec);
    rocke_param_opts_t opts;
    const rocke_type_t* ptr_io
        = rocke_ptr_type(b, rocke_mfma_attn_ir_type_for_dtype(b, dtype), "global");
    const rocke_type_t* ptr_kv = spec->kv_dtype[0] != '\0'
                                   ? rocke_ptr_type(b, rocke_fp8e4m3(), "global") : ptr_io;

    /* Q uses the selected 16-bit type; FP8 K/V are byte storage. */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)rocke_b_param(b, "Q", ptr_io, &opts);
    (void)rocke_b_param(b, "K", ptr_kv, &opts);
    (void)rocke_b_param(b, "V", ptr_kv, &opts);

    /* O has the same storage type as Q. */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)rocke_b_param(b, "O", ptr_io, &opts);

    /* scalars */
    (void)rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
    (void)rocke_b_param(b, "seqlen_q", rocke_i32(), NULL);
    (void)rocke_b_param(b, "seqlen_k", rocke_i32(), NULL);

    /* element strides (token, head) per tensor, in Python order. */
    (void)rocke_b_param(b, "stride_q_token", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_q_head", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_k_token", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_k_head", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_v_token", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_v_head", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_o_token", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_o_head", rocke_i32(), NULL);
    if(spec->use_softcap)
        (void)rocke_b_param(b, "softcap", rocke_f32(), NULL);
    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align_set = true;
    if(spec->use_sinks)
    {
        opts.align = 2;
        (void)rocke_b_param(b, "sink_ptr", ptr_io, &opts);
    }
    if(spec->use_alibi || spec->use_qq_bias)
    {
        const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");
        opts.align = 4;
        if(spec->use_alibi)
            (void)rocke_b_param(b, "alibi_slopes_ptr", ptr_f32, &opts);
        if(spec->use_qq_bias)
        {
            (void)rocke_b_param(b, "qq_bias_ptr", ptr_f32, &opts);
            (void)rocke_b_param(b, "qq_bias_rows", rocke_i32(), NULL);
            (void)rocke_b_param(b, "qq_bias_cols", rocke_i32(), NULL);
            (void)rocke_b_param(b, "qq_bias_stride", rocke_i32(), NULL);
        }
    }
    if(strcmp(spec->layout, "dense") != 0)
    {
        const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
        opts.align = 4;
        (void)rocke_b_param(b, "cu_seqlens_q", ptr_i32, &opts);
        if(strcmp(spec->layout, "ragged") == 0)
            (void)rocke_b_param(b, "cu_seqlens_k", ptr_i32, &opts);
        else
        {
            (void)rocke_b_param(b, "seqused_k", ptr_i32, &opts);
            (void)rocke_b_param(b, "block_table", ptr_i32, &opts);
            (void)rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
            (void)rocke_b_param(b, "stride_k_block", rocke_i32(), NULL);
            (void)rocke_b_param(b, "stride_v_block", rocke_i32(), NULL);
        }
    }
    if(spec->kv_dtype[0] != '\0')
    {
        (void)rocke_b_param(b, "k_scale", rocke_f32(), NULL);
        (void)rocke_b_param(b, "v_scale", rocke_f32(), NULL);
    }
}

struct wmma_score_context
{
    rocke_value_t* context;
    rocke_value_t* log2e;
    rocke_value_t* cap;
    rocke_value_t* slope;
    rocke_value_t* qq_bias;
    rocke_value_t* qq_rows;
    rocke_value_t* qq_cols;
    rocke_value_t* qq_stride;
    rocke_value_t* zero_f;
    rocke_value_t* zero_i;
};

static rocke_value_t* wmma_score_transform(
    rocke_ir_builder_t* b, rocke_value_t* score, rocke_value_t* kt, int row,
    rocke_value_t* query_pos, rocke_value_t* key_pos, void* user)
{
    (void)kt;
    (void)row;
    const auto* c = static_cast<const wmma_score_context*>(user);
    if(c->cap != NULL)
        score = rocke_b_fmul(b, c->cap, rocke_b_tanh(b, rocke_b_fdiv(b, score, c->cap)));
    rocke_value_t* relative_k = NULL;
    if(c->slope != NULL || c->qq_bias != NULL)
        relative_k = rocke_b_sub(b, key_pos, c->context);
    if(c->slope != NULL)
        score = rocke_b_fadd(b, score, rocke_b_fmul(b, c->slope, rocke_b_sitofp_f32(b, relative_k)));
    if(c->qq_bias != NULL)
    {
        rocke_value_t* q_ok = rocke_b_cmp_lt(b, query_pos, c->qq_rows);
        rocke_value_t* k_lo = rocke_b_cmp_ge(b, relative_k, c->zero_i);
        rocke_value_t* k_hi = rocke_b_cmp_lt(b, relative_k, c->qq_cols);
        rocke_value_t* keep = rocke_b_land(b, q_ok, rocke_b_land(b, k_lo, k_hi));
        rocke_value_t* index = rocke_b_add(b, rocke_b_mul(b, query_pos, c->qq_stride), relative_k);
        rocke_value_t* bias = rocke_b_masked_global_load(
            b, c->qq_bias, index, keep, c->zero_f, rocke_f32(), 4);
        score = rocke_b_fadd(b, score, rocke_b_fmul(b, bias, c->log2e));
    }
    return score;
}

struct wmma_paged_row_context
{
    rocke_value_t* block_table;
    rocke_value_t* table_row;
    rocke_value_t* page_log2;
    rocke_value_t* page_mask;
    rocke_value_t* stride_block;
    rocke_value_t* stride_token;
    rocke_value_t* head_offset;
};

static rocke_value_t* wmma_paged_row(rocke_ir_builder_t* b, rocke_value_t* token, void* user)
{
    const auto* c = static_cast<const wmma_paged_row_context*>(user);
    rocke_value_t* logical_block = rocke_b_lshr(b, token, c->page_log2);
    rocke_value_t* page_token = rocke_b_land(b, token, c->page_mask);
    rocke_value_t* physical_block = rocke_b_global_load_i32(
        b, c->block_table, rocke_b_add(b, c->table_row, logical_block), 4);
    rocke_value_t* block_offset = rocke_b_mul(b, physical_block, c->stride_block);
    rocke_value_t* token_offset = rocke_b_mul(b, page_token, c->stride_token);
    return rocke_b_add(b, rocke_b_add(b, block_offset, token_offset), c->head_offset);
}

/* --------------------------------------------------------------------------- *
 * The shared Python build body: emit the adapter into an already-initialised
 * builder `b` (kernel name already set). Returns ROCKE_OK or the sticky status.
 *
 * Mirrors build_wmma_fmha_fwd op-for-op:
 *   b.kernel.attrs["max_workgroup_size"] = wave
 *   _declare_params(b)
 *   c16 = const_i32(16)
 *   q_tile = block_id_x; head = block_id_y; batch = block_id_z
 *   kv_head = head if kvh==qh else div(head, const(qh//kvh))
 *   q_row0       = q_tile * 16
 *   batch_row_q  = batch  * seqlen_q
 *   batch_off_k  = batch * seqlen_k * stride_k_token
 *   batch_off_v  = batch * seqlen_k * stride_v_token
 *   mfma_attention_fwd_inner_body(...) ; b.ret()
 * --------------------------------------------------------------------------- */
static rocke_status_t
    wmma_emit_body(rocke_ir_builder_t* b, const rocke_wmma_fmha_fwd_spec_t* spec, const char* arch)
{
    const rocke_archtarget_t* target;
    int wave;
    int qh, kvh;
    rocke_value_t* c16;
    rocke_value_t* q_tile;
    rocke_value_t* head;
    rocke_value_t* batch;
    rocke_value_t* kv_head;
    rocke_value_t* seqlen_q;
    rocke_value_t* seqlen_k;
    rocke_value_t* q_row0;
    rocke_value_t* batch_row_q;
    rocke_value_t* batch_off_k;
    rocke_value_t* batch_off_v;
    rocke_mfma_attn_params_t p;

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_fmha_fwd: unknown arch '%s'", arch);
        return ROCKE_ERR_VALUE;
    }
    wave = target->wave_size; /* 32 for WMMA */

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size",
                      wave * (spec->transposed_qk ? spec->num_waves : 1));

    /* _declare_params(b) */
    wmma_declare_params(b, spec);

    c16 = rocke_b_const_i32(b, ROCKE_WMMA_FMHA_FWD_BLOCK_M);

    /* grid decode */
    q_tile = rocke_b_block_id_x(b); /* Q-tile index (16 rows) */
    head = rocke_b_block_id_y(b); /* query head             */
    batch = rocke_b_block_id_z(b); /* batch index            */

    /* GQA: kv_head = head // (num_query_heads // kv_heads). */
    qh = spec->num_query_heads;
    kvh = wmma_kv_heads(spec);
    if(kvh == qh)
    {
        kv_head = head;
    }
    else
    {
        kv_head = rocke_b_div(b, head, rocke_b_const_i32(b, qh / kvh));
    }

    seqlen_q = rocke_b_get_param(b, "seqlen_q");
    seqlen_k = rocke_b_get_param(b, "seqlen_k");

    /* per-batch shifts (Python op order). */
    rocke_value_t* q_step = spec->transposed_qk
                               ? rocke_b_const_i32(b, ROCKE_WMMA_FMHA_FWD_BLOCK_M * spec->num_waves)
                               : c16;
    q_row0 = rocke_b_mul(b, q_tile, q_step);
    const bool packed = strcmp(spec->layout, "dense") != 0;
    if(!packed)
    {
        batch_row_q = rocke_b_mul(b, batch, seqlen_q);
        batch_off_k
            = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), rocke_b_get_param(b, "stride_k_token"));
        batch_off_v
            = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), rocke_b_get_param(b, "stride_v_token"));
    }
    else
    {
        rocke_value_t* next_batch = rocke_b_add(b, batch, rocke_b_const_i32(b, 1));
        rocke_value_t* cu_q = rocke_b_get_param(b, "cu_seqlens_q");
        batch_row_q = rocke_b_global_load_i32(b, cu_q, batch, 4);
        rocke_value_t* q_end = rocke_b_global_load_i32(b, cu_q, next_batch, 4);
        seqlen_q = rocke_b_sub(b, q_end, batch_row_q);
        rocke_if_t guard = rocke_b_scf_if(b, rocke_b_cmp_ge(b, q_row0, seqlen_q));
        rocke_b_region_enter(b, guard.then_region);
        rocke_b_ret(b);
        rocke_b_region_leave(b);
        if(strcmp(spec->layout, "ragged") == 0)
        {
            rocke_value_t* cu_k = rocke_b_get_param(b, "cu_seqlens_k");
            rocke_value_t* k_start = rocke_b_global_load_i32(b, cu_k, batch, 4);
            rocke_value_t* k_end = rocke_b_global_load_i32(b, cu_k, next_batch, 4);
            seqlen_k = rocke_b_sub(b, k_end, k_start);
            batch_off_k = rocke_b_mul(b, k_start, rocke_b_get_param(b, "stride_k_token"));
            batch_off_v = rocke_b_mul(b, k_start, rocke_b_get_param(b, "stride_v_token"));
        }
        else
        {
            seqlen_k = rocke_b_global_load_i32(b, rocke_b_get_param(b, "seqused_k"), batch, 4);
            batch_off_k = batch_off_v = rocke_b_const_i32(b, 0);
        }
    }

    /* mfma_attention_fwd_inner_body(...) with the WMMA v-LDS staging flag. */
    memset(&p, 0, sizeof(p));
    p.Q = rocke_b_get_param(b, "Q");
    p.K = rocke_b_get_param(b, "K");
    p.V = rocke_b_get_param(b, "V");
    p.O = rocke_b_get_param(b, "O");
    p.head_size = spec->head_size;
    p.seqlen_k = seqlen_k;
    /* global Q/O row index folds the batch shift in; within-batch q position for
     * the mask is q_pos_base = q_row0. */
    p.q_tile_base = rocke_b_add(b, q_row0, batch_row_q);
    p.head_idx = head;
    p.kv_head_idx = kv_head;
    p.q_pos_base = q_row0;
    p.stride_q_token = rocke_b_get_param(b, "stride_q_token");
    p.stride_q_head = rocke_b_get_param(b, "stride_q_head");
    p.stride_k_token = rocke_b_get_param(b, "stride_k_token");
    p.stride_k_head = rocke_b_get_param(b, "stride_k_head");
    p.stride_v_token = rocke_b_get_param(b, "stride_v_token");
    p.stride_v_head = rocke_b_get_param(b, "stride_v_head");
    p.stride_o_token = rocke_b_get_param(b, "stride_o_token");
    p.stride_o_head = rocke_b_get_param(b, "stride_o_head");
    p.scale_log2 = rocke_b_get_param(b, "scale_log2");
    p.dtype = wmma_dtype(spec);
    p.mask_mode = wmma_to_attn_mask(spec->mask_mode);
    p.sliding_window = spec->sliding_window;
    p.causal_ctx_offset = spec->causal_bottom_right
                              ? rocke_b_sub(b, seqlen_k, seqlen_q)
                              : rocke_b_const_i32(b, 0);
    const bool score_features = spec->use_softcap || spec->use_alibi || spec->use_qq_bias;
    const bool strict = spec->causal_bottom_right || strcmp(p.dtype, "bf16") == 0
                        || spec->kv_tail || spec->sliding_window > 0
                        || score_features || spec->use_sinks || packed;
    p.mask_neg_inf = strict ? rocke_b_const_f32(b, -INFINITY) : NULL;
    wmma_score_context features{};
    features.context = p.causal_ctx_offset;
    if(score_features || spec->use_sinks)
    {
        features.log2e = rocke_b_const_f32(b, 1.4426950408889634);
        if(spec->use_softcap)
            features.cap = rocke_b_fmul(b, rocke_b_get_param(b, "softcap"), features.log2e);
        if(spec->use_sinks)
        {
            const rocke_type_t* elem = rocke_mfma_attn_ir_type_for_dtype(b, p.dtype);
            rocke_value_t* value = rocke_b_global_load(
                b, rocke_b_get_param(b, "sink_ptr"), head, elem, 2);
            p.sink_log2 = rocke_b_fmul(b, rocke_b_cast_to_f32(b, value), features.log2e);
        }
        if(spec->use_alibi)
        {
            rocke_value_t* value = rocke_b_global_load(
                b, rocke_b_get_param(b, "alibi_slopes_ptr"), head, rocke_f32(), 4);
            features.slope = rocke_b_fmul(b, value, features.log2e);
        }
        if(spec->use_qq_bias)
        {
            features.zero_f = rocke_b_const_f32(b, 0.0);
            features.zero_i = rocke_b_const_i32(b, 0);
            features.qq_bias = rocke_b_get_param(b, "qq_bias_ptr");
            features.qq_rows = rocke_b_get_param(b, "qq_bias_rows");
            features.qq_cols = rocke_b_get_param(b, "qq_bias_cols");
            features.qq_stride = rocke_b_get_param(b, "qq_bias_stride");
        }
    }
    if(score_features)
    {
        p.extra_score_transform = wmma_score_transform;
        p.extra_score_transform_user = &features;
    }
    if(spec->sliding_window > 0)
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* one = rocke_b_const_i32(b, 1);
        rocke_value_t* last = rocke_b_const_i32(b, 15);
        rocke_value_t* start_pos = rocke_b_add(b, q_row0, p.causal_ctx_offset);
        rocke_value_t* width = rocke_b_const_i32(b, spec->sliding_window - 1);
        rocke_value_t* lower = rocke_b_sub(b, start_pos, width);
        lower = rocke_b_select(b, rocke_b_cmp_lt(b, lower, zero), zero, lower);
        rocke_value_t* q_last = rocke_b_add(b, q_row0, last);
        rocke_value_t* q_limit = rocke_b_sub(b, seqlen_q, one);
        q_last = rocke_b_select(b, rocke_b_cmp_gt(b, q_last, q_limit), q_limit, q_last);
        rocke_value_t* upper = rocke_b_add(b, rocke_b_add(b, q_last, p.causal_ctx_offset), one);
        upper = rocke_b_select(b, rocke_b_cmp_gt(b, upper, seqlen_k), seqlen_k, upper);
        upper = rocke_b_select(b, rocke_b_cmp_lt(b, upper, zero), zero, upper);
        p.k_tile_start = rocke_b_div(b, lower, c16);
        p.k_tile_stop = rocke_b_div(b, rocke_b_add(b, upper, last), c16);
    }
    wmma_paged_row_context paged_k{}, paged_v{};
    if(strcmp(spec->layout, "paged") == 0)
    {
        rocke_value_t* page_log2 = rocke_b_const_i32(
            b, std::countr_zero(static_cast<unsigned int>(spec->page_block_size)));
        rocke_value_t* page_mask = rocke_b_const_i32(b, spec->page_block_size - 1);
        rocke_value_t* table_row = rocke_b_mul(b, batch, rocke_b_get_param(b, "block_table_stride"));
        rocke_value_t* table = rocke_b_get_param(b, "block_table");
        paged_k = {table, table_row, page_log2, page_mask,
                   rocke_b_get_param(b, "stride_k_block"), p.stride_k_token,
                   rocke_b_mul(b, kv_head, p.stride_k_head)};
        paged_v = {table, table_row, page_log2, page_mask,
                   rocke_b_get_param(b, "stride_v_block"), p.stride_v_token,
                   rocke_b_mul(b, kv_head, p.stride_v_head)};
        p.k_row_base_fn = wmma_paged_row;
        p.k_row_base_user = &paged_k;
        p.v_row_base_fn = wmma_paged_row;
        p.v_row_base_user = &paged_v;
    }
    p.k_token_offset_elems = batch_off_k;
    p.v_token_offset_elems = batch_off_v;
    p.wmma_v_lds_stage = spec->v_lds_stage;
    p.arch = arch;
    p.wmma_seqlen_q = spec->query_tail || packed ? seqlen_q : NULL;
    p.wmma_kv_tail = spec->kv_tail || packed;
    if(spec->kv_dtype[0] != '\0')
    {
        p.kv_dtype = spec->kv_dtype;
        p.k_scale = rocke_b_get_param(b, "k_scale");
        p.v_scale = rocke_b_get_param(b, "v_scale");
    }

    if(spec->transposed_qk)
    {
        if(!spec->causal_bottom_right)
            p.causal_ctx_offset = NULL;
        (void)rocke_wmma_swapqk_fwd_inner_body(b, &p, spec->block_n, spec->num_waves);
    }
    else
        (void)rocke_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    rocke_b_ret(b);

    return rocke_ir_builder_status(b);
}

/* --------------------------------------------------------------------------- *
 * build_wmma_fmha_fwd(spec, arch)
 *
 * `b` is the destination builder, assumed already initialised by the caller
 * with spec.kernel_name() (the gfx1201 WMMA GEMM call contract). Validates,
 * emits the adapter body, and returns b.kernel (NULL on validation / IR error).
 * --------------------------------------------------------------------------- */
rocke_kernel_def_t* rocke_build_wmma_fmha_fwd(rocke_ir_builder_t* b,
                                              const rocke_wmma_fmha_fwd_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = WMMA_FMHA_DEFAULT_ARCH;
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_wmma_fmha_fwd_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid wmma_fmha_fwd spec: %s", reason);
            return NULL;
        }

        if(wmma_emit_body(b, spec, arch) != ROCKE_OK)
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* --------------------------------------------------------------------------- *
 * wmma_fmha_fwd_grid(spec, seqlen_q, batch)
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_grid(const rocke_wmma_fmha_fwd_spec_t* spec,
                                        int seqlen_q,
                                        int batch,
                                        int out[3])
{
    if(spec == NULL || out == NULL || !wmma_valid_layout(spec))
    {
        return ROCKE_ERR_VALUE;
    }
    if(spec->transposed_qk && spec->num_waves != 1 && spec->num_waves != 2)
        return ROCKE_ERR_VALUE;
    int block_m = ROCKE_WMMA_FMHA_FWD_BLOCK_M * (spec->transposed_qk ? spec->num_waves : 1);
    if(strcmp(spec->layout, "dense") == 0 && !spec->query_tail
       && seqlen_q % block_m != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = (seqlen_q + block_m - 1) / block_m;
    out[1] = spec->num_query_heads;
    out[2] = batch;
    return ROCKE_OK;
}

/* --------------------------------------------------------------------------- *
 * wmma_fmha_fwd_signature(spec): the kernel ABI (Q/K/V/O ptrs, scale_log2/
 * seqlen_q/seqlen_k scalars, q/k/v/o stride pairs), via a transient probe
 * builder that runs _declare_params and reads the param order from the kernel.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_signature(const rocke_wmma_fmha_fwd_spec_t* spec,
                                             rocke_arena_t* arena,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count)
{
    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL
       || wmma_dtype(spec) == NULL || !wmma_valid_layout(spec) || !wmma_valid_kv_dtype(spec))
        return ROCKE_ERR_VALUE;
    rocke_ir_builder_t b;
    rocke_status_t status = rocke_ir_builder_init(&b, "rocke_wmma_fmha_fwd_sig_probe");
    if(status != ROCKE_OK)
        return status;
    wmma_declare_params(&b, spec);
    status = rocke_ir_builder_status(&b);
    if(status != ROCKE_OK)
    {
        rocke_ir_builder_free(&b);
        return status;
    }
    const int count = b.kernel->num_params;
    auto* items = static_cast<rocke_sig_entry_t*>(
        rocke_arena_alloc(arena, static_cast<size_t>(count) * sizeof(rocke_sig_entry_t)));
    if(items == NULL)
    {
        rocke_ir_builder_free(&b);
        return ROCKE_ERR_OOM;
    }
    for(int i = 0; i < count; ++i)
    {
        const rocke_param_t* param = b.kernel->params[i];
        const rocke_type_t* type = param->type;
        if(type->kind == ROCKE_TYPE_PTR)
            status = rocke_sig_param(arena, param->name, type->pointee->name, type->space, &items[i]);
        else
            status = rocke_sig_scalar(arena, param->name, type->name, &items[i]);
        if(status != ROCKE_OK)
        {
            rocke_ir_builder_free(&b);
            return status;
        }
    }
    rocke_ir_builder_free(&b);
    *out_items = items;
    *out_count = static_cast<size_t>(count);
    return ROCKE_OK;
}

/* --------------------------------------------------------------------------- *
 * rocke_wmma_fmha_fwd_lower_to_llvm -- build + lower to .ll convenience.
 *
 * Owns and frees its own IRBuilder for the whole lower so the kernel stays
 * alive through lowering.
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_lower_to_llvm(const rocke_wmma_fmha_fwd_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    char name_buf[256];
    rocke_ir_builder_t b;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = WMMA_FMHA_DEFAULT_ARCH;
    }

    if(!rocke_wmma_fmha_fwd_is_valid_spec(spec, arch, err, err_cap))
    {
        return ROCKE_ERR_VALUE;
    }

    if(rocke_wmma_fmha_fwd_kernel_name(spec, name_buf, sizeof(name_buf)) != ROCKE_OK)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: kernel name too long");
        return ROCKE_ERR_VALUE;
    }

    st = rocke_ir_builder_init(&b, name_buf);
    if(st != ROCKE_OK)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: builder init failed");
        return st;
    }

    st = wmma_emit_body(&b, spec, arch);
    if(st != ROCKE_OK || b.kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        wmma_set_reason(err, err_cap, m != NULL ? m : "build_wmma_fmha_fwd failed");
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }

    st = rocke_lower_kernel_to_llvm_ex(b.kernel, flavor, arch, out_ll, err, err_cap);
    return st;
}
