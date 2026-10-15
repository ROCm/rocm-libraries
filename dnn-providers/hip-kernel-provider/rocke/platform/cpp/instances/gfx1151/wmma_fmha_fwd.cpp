// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_gfx1151_wmma_fmha_fwd.c -- C99 port of
 * rocke/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * Byte-identical builder-call sequence vs the Python build_wmma_fmha_fwd: a raw
 * IRBuilder declares the same params in the same order (_declare_params), bakes
 * the same max_workgroup_size attr, decodes runtime query/head/batch-value-tile
 * coordinates, computes the same GQA KV head and tensor offsets, and calls the
 * already-ported rocke_mfma_attention_fwd_inner_body with the same operands and
 * attributes, including wmma_v_lds_stage.
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

static bool wmma_valid_bias_dtype(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return spec->bias_dtype != NULL
           && (strcmp(spec->bias_dtype, "f32") == 0 || strcmp(spec->bias_dtype, "q") == 0);
}

static bool wmma_bias_is_q(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return strcmp(spec->bias_dtype, "q") == 0;
}

/* WmmaFmhaFwdSpec.v_dim: the V/output head width (v_head_size or head_size). */
static int wmma_v_dim(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return spec->v_head_size != 0 ? spec->v_head_size : spec->head_size;
}

static bool wmma_valid_v_head(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    return spec->v_head_size == 0
           || (spec->v_head_size >= 16 && spec->v_head_size <= 256 && spec->v_head_size % 16 == 0
               && spec->v_head_size != spec->head_size && !spec->transposed_qk);
}

static bool wmma_valid_value_tile(const rocke_wmma_fmha_fwd_spec_t* spec)
{
    int tile = spec->value_tile_size;
    int v_dim = wmma_v_dim(spec);
    return wmma_valid_v_head(spec)
           && (tile == 0
               || (tile >= 16 && tile % 16 == 0 && tile < v_dim && v_dim % tile == 0
                   && !spec->transposed_qk));
}

/* Window masking is emitted by this adapter's runtime-bound score transform. */
static rocke_attn_mask_mode_t wmma_to_attn_mask(rocke_fmha_mask_mode_t m)
{
    return m == ROCKE_FMHA_MASK_CAUSAL ? ROCKE_ATTN_MASK_CAUSAL : ROCKE_ATTN_MASK_NONE;
}

/* --------------------------------------------------------------------------- *
 * rocke_wmma_fmha_fwd_spec_default
 * --------------------------------------------------------------------------- */
rocke_wmma_fmha_fwd_spec_t rocke_wmma_fmha_fwd_spec_default(void)
{
    rocke_wmma_fmha_fwd_spec_t s;
    s.head_size = 0;
    s.mask_mode = ROCKE_FMHA_MASK_NONE;
    s.v_lds_stage = false;
    s.name = WMMA_FMHA_DEFAULT_NAME;
    s.dtype = "fp16";
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
    s.scheduler_strategy = NULL;
    s.value_tile_size = 0;
    s.causal_tile_skip = false;
    s.v_head_size = 0;
    s.use_attn_bias = false;
    s.bias_dtype = "f32";
    return s;
}

/* --------------------------------------------------------------------------- *
 * WmmaFmhaFwdSpec.kernel_name()
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap)
{
    const char* name;
    const char* mask;
    char h[32], page[32], block_n[32], waves[32], scheduler[32], value_tile[32], v_head[32];
    const char* parts[16];
    const char* dtype = wmma_dtype(spec);

    if(spec == NULL || out == NULL || dtype == NULL || !wmma_valid_layout(spec)
       || !wmma_valid_kv_dtype(spec) || !rocke_scheduler_strategy_is_valid(spec->scheduler_strategy)
       || !wmma_valid_value_tile(spec))
    {
        return ROCKE_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : WMMA_FMHA_DEFAULT_NAME;
    mask = spec->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW
               ? "window"
               : rocke_fmha_mask_mode_name(spec->mask_mode);
    if(mask == NULL)
    {
        mask = "none";
    }

    snprintf(h, sizeof(h), "H%d", spec->head_size);
    parts[0] = spec->transposed_qk ? "wmma_swapqk" : "wmma16x16x16";
    parts[1] = h;
    parts[2] = strcmp(dtype, "bf16") == 0 ? "bf16" : "fp16";
    parts[3] = mask;
    parts[4] = spec->v_lds_stage ? "vlds" : "vgather";

    size_t num_parts = 5;
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
    if(spec->scheduler_strategy != NULL)
    {
        snprintf(scheduler, sizeof(scheduler), "sched_%s", spec->scheduler_strategy);
        for(char* c = scheduler; *c; ++c)
            if(*c == '-')
                *c = '_';
        parts[num_parts++] = scheduler;
    }
    if(spec->value_tile_size != 0)
    {
        snprintf(value_tile, sizeof(value_tile), "dv%d", spec->value_tile_size);
        parts[num_parts++] = value_tile;
    }
    if(spec->v_head_size != 0)
    {
        snprintf(v_head, sizeof(v_head), "vh%d", spec->v_head_size);
        parts[num_parts++] = v_head;
    }
    if(spec->use_attn_bias)
        parts[num_parts++] = wmma_bias_is_q(spec) ? "abias_q" : "abias_f32";
    const char* flag_names[] = {"qtail", "kvtail", "softcap", "sinks", "alibi", "qqbias", "cskip"};
    const int flag_on[] = {spec->query_tail || packed,
                           spec->kv_tail || packed,
                           spec->use_softcap,
                           spec->use_sinks,
                           spec->use_alibi,
                           spec->use_qq_bias,
                           spec->causal_tile_skip};
    return rocke_kernel_name_join(
        name, parts, num_parts, flag_names, flag_on, 7, out, out_cap, NULL);
}

/* --------------------------------------------------------------------------- *
 * is_valid_spec(spec, arch)
 *
 * Python validates the target WMMA atom, wave size, head-size granularity,
 * compile-time option compatibility and LDS footprint. Runtime head-count
 * divisibility is a launch-contract check, not a code-object specialization.
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
    if(!rocke_scheduler_strategy_is_valid(spec->scheduler_strategy))
    {
        wmma_set_reason(reason, reason_cap, "unsupported scheduler_strategy");
        return false;
    }
    if(arch == NULL)
    {
        arch = WMMA_FMHA_DEFAULT_ARCH;
    }
    if(!wmma_valid_value_tile(spec)
       || ((spec->value_tile_size != 0 || spec->v_head_size != 0) && strcmp(arch, "gfx1151") != 0))
    {
        wmma_set_reason(reason, reason_cap, "invalid gfx1151 output-column tile or V head size");
        return false;
    }
    const char* dtype = wmma_dtype(spec);
    if(dtype == NULL)
    {
        wmma_set_reason(reason, reason_cap, "WMMA FMHA dtype must be fp16/f16 or bf16");
        return false;
    }
    if(!wmma_valid_bias_dtype(spec))
    {
        wmma_set_reason(reason, reason_cap, "bias_dtype must be 'f32' or 'q'");
        return false;
    }
    if(!wmma_valid_kv_dtype(spec))
    {
        wmma_set_reason(reason, reason_cap, "KV storage must match Q or use OCP fp8e4m3");
        return false;
    }
    if(spec->mask_mode != ROCKE_FMHA_MASK_NONE && spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL
       && spec->mask_mode != ROCKE_FMHA_MASK_SLIDING_WINDOW)
    {
        wmma_set_reason(reason, reason_cap, "WMMA FMHA supports none/causal/window masking");
        return false;
    }
    if(spec->causal_tile_skip && (spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL || spec->transposed_qk))
    {
        wmma_set_reason(reason,
                        reason_cap,
                        "causal_tile_skip requires causal masking on the standard "
                        "(non-transposed) path");
        return false;
    }
    if(!wmma_valid_layout(spec))
    {
        wmma_set_reason(reason,
                        reason_cap,
                        "layout must be dense/ragged with page size zero, or paged with a positive "
                        "power-of-two page size");
        return false;
    }
    if(spec->transposed_qk)
    {
        if(strcmp(arch, "gfx1151") != 0 || (spec->head_size != 64 && spec->head_size != 128)
           || (spec->block_n != 32 && spec->block_n != 64)
           || (spec->num_waves != 1 && spec->num_waves != 2))
        {
            wmma_set_reason(reason,
                            reason_cap,
                            "transposed QK requires gfx1151 FP16/BF16 D64/D128, block_n 32/64 "
                            "and one or two waves");
            return false;
        }
        if(strcmp(spec->layout, "dense") != 0 || spec->kv_dtype[0] != '\0' || spec->query_tail
           || spec->kv_tail || spec->v_lds_stage
           || spec->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW || spec->use_softcap
           || spec->use_sinks || spec->use_alibi || spec->use_qq_bias || spec->use_attn_bias)
        {
            wmma_set_reason(reason,
                            reason_cap,
                            "transposed QK requires aligned dense inputs without windows or extra "
                            "score features");
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
    const char* op_id
        = strcmp(arch, "gfx1201") == 0
              ? (bf16 ? "wmma_gfx12_f32_16x16x16_bf16" : "wmma_gfx12_f32_16x16x16_f16")
              : (bf16 ? "wmma_f32_16x16x16_bf16" : "wmma_f32_16x16x16_f16");
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

    if(spec->head_size < 16 || spec->head_size > 256 || spec->head_size % 16 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "head_size must be a multiple of 16 in [16, 256] (got %d)",
                 spec->head_size);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* Both fp16 and bf16 use two bytes per P/V staging element. */
    bytes_lds = (long)ROCKE_WMMA_FMHA_FWD_BLOCK_M * ROCKE_WMMA_FMHA_FWD_BLOCK_K * 2;
    if(spec->v_lds_stage)
    {
        int value_size = spec->value_tile_size != 0 ? spec->value_tile_size : wmma_v_dim(spec);
        bytes_lds += (long)ROCKE_WMMA_FMHA_FWD_BLOCK_M * value_size * 2;
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
 * _declare_params(b): fixed runtime-shape ABI plus layout-specific metadata.
 * --------------------------------------------------------------------------- */
static void wmma_declare_params(rocke_ir_builder_t* b, const rocke_wmma_fmha_fwd_spec_t* spec)
{
    const char* dtype = wmma_dtype(spec);
    rocke_param_opts_t opts;
    const rocke_type_t* ptr_io
        = rocke_ptr_type(b, rocke_mfma_attn_ir_type_for_dtype(b, dtype), "global");
    const rocke_type_t* ptr_kv
        = spec->kv_dtype[0] != '\0' ? rocke_ptr_type(b, rocke_fp8e4m3(), "global") : ptr_io;
    const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 2;
    opts.align_set = true;
    (void)rocke_b_param(b, "Q", ptr_io, &opts);
    opts.align = spec->kv_dtype[0] != '\0' ? 1 : 2;
    (void)rocke_b_param(b, "K", ptr_kv, &opts);
    (void)rocke_b_param(b, "V", ptr_kv, &opts);

    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 2;
    opts.align_set = true;
    (void)rocke_b_param(b, "O", ptr_io, &opts);
    opts.noalias = false;
    opts.noalias_set = false;
    opts.align = 4;
    (void)rocke_b_param(b, "LSE", ptr_f32, &opts);

    (void)rocke_b_param(b, "scale_log2", rocke_f32(), NULL);
    (void)rocke_b_param(b, "seqlen_q", rocke_i32(), NULL);
    (void)rocke_b_param(b, "seqlen_k", rocke_i32(), NULL);
    (void)rocke_b_param(b, "num_query_heads", rocke_i32(), NULL);
    (void)rocke_b_param(b, "num_kv_heads", rocke_i32(), NULL);
    (void)rocke_b_param(b, "bottom_right", rocke_i32(), NULL);
    (void)rocke_b_param(b, "window_left", rocke_i32(), NULL);
    (void)rocke_b_param(b, "window_right", rocke_i32(), NULL);
    (void)rocke_b_param(b, "write_lse", rocke_i32(), NULL);

    const char* tensors[] = {"q", "k", "v", "o"};
    for(const char* tensor : tensors)
    {
        char name[32];
        snprintf(name, sizeof(name), "stride_%s_batch", tensor);
        (void)rocke_b_param(b, name, rocke_i32(), NULL);
        snprintf(name, sizeof(name), "stride_%s_token", tensor);
        (void)rocke_b_param(b, name, rocke_i32(), NULL);
        snprintf(name, sizeof(name), "stride_%s_head", tensor);
        (void)rocke_b_param(b, name, rocke_i32(), NULL);
    }
    (void)rocke_b_param(b, "stride_lse_batch", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_lse_token", rocke_i32(), NULL);
    (void)rocke_b_param(b, "stride_lse_head", rocke_i32(), NULL);

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
    if(spec->use_attn_bias)
    {
        const bool q_bias = wmma_bias_is_q(spec);
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = q_bias ? 2 : 4;
        opts.align_set = true;
        const rocke_type_t* elem
            = q_bias ? rocke_mfma_attn_ir_type_for_dtype(b, wmma_dtype(spec)) : rocke_f32();
        (void)rocke_b_param(b, "attn_bias_ptr", rocke_ptr_type(b, elem, "global"), &opts);
        (void)rocke_b_param(b, "bias_stride_b", rocke_i32(), NULL);
        (void)rocke_b_param(b, "bias_stride_h", rocke_i32(), NULL);
        (void)rocke_b_param(b, "bias_stride_q", rocke_i32(), NULL);
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
    rocke_value_t* attn_bias;
    const rocke_type_t* bias_elem;
    bool bias_q;
    rocke_value_t* bias_base;
    rocke_value_t* bias_stride_q;
    rocke_value_t* bias_zero;
    rocke_value_t* seqlen_q;
    rocke_value_t* seqlen_k;
    rocke_value_t* window_left;
    rocke_value_t* window_right;
    rocke_value_t* masked;
    bool windowed;
};

static rocke_value_t* wmma_score_transform(rocke_ir_builder_t* b,
                                           rocke_value_t* score,
                                           rocke_value_t* kt,
                                           int row,
                                           rocke_value_t* query_pos,
                                           rocke_value_t* key_pos,
                                           void* user)
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
        score
            = rocke_b_fadd(b, score, rocke_b_fmul(b, c->slope, rocke_b_sitofp_f32(b, relative_k)));
    if(c->qq_bias != NULL)
    {
        rocke_value_t* q_ok = rocke_b_cmp_lt(b, query_pos, c->qq_rows);
        rocke_value_t* k_lo = rocke_b_cmp_ge(b, relative_k, c->zero_i);
        rocke_value_t* k_hi = rocke_b_cmp_lt(b, relative_k, c->qq_cols);
        rocke_value_t* both = rocke_b_land(b, k_lo, k_hi);
        rocke_value_t* keep = rocke_b_land(b, q_ok, both);
        rocke_value_t* q_base = rocke_b_mul(b, query_pos, c->qq_stride);
        rocke_value_t* index = rocke_b_add(b, q_base, relative_k);
        rocke_value_t* bias
            = rocke_b_masked_global_load(b, c->qq_bias, index, keep, c->zero_f, rocke_f32(), 4);
        rocke_value_t* scaled_bias = rocke_b_fmul(b, bias, c->log2e);
        score = rocke_b_fadd(b, score, scaled_bias);
    }
    if(c->windowed)
    {
        rocke_value_t* center = rocke_b_add(b, query_pos, c->context);
        rocke_value_t* left_unbounded = rocke_b_cmp_lt(b, c->window_left, c->zero_i);
        rocke_value_t* left_edge = rocke_b_sub(b, center, c->window_left);
        rocke_value_t* left_inside = rocke_b_cmp_ge(b, key_pos, left_edge);
        rocke_value_t* keep_left = rocke_b_lor(b, left_unbounded, left_inside);
        rocke_value_t* right_unbounded = rocke_b_cmp_lt(b, c->window_right, c->zero_i);
        rocke_value_t* right_edge = rocke_b_add(b, center, c->window_right);
        rocke_value_t* right_inside = rocke_b_cmp_le(b, key_pos, right_edge);
        rocke_value_t* keep_right = rocke_b_lor(b, right_unbounded, right_inside);
        rocke_value_t* keep = rocke_b_land(b, keep_left, keep_right);
        score = rocke_b_select(b, keep, score, c->masked);
    }
    if(c->attn_bias != NULL)
    {
        rocke_value_t* bias_q_ok = rocke_b_cmp_lt(b, query_pos, c->seqlen_q);
        rocke_value_t* bias_k_ok = rocke_b_cmp_lt(b, key_pos, c->seqlen_k);
        rocke_value_t* keep = rocke_b_land(b, bias_q_ok, bias_k_ok);
        rocke_value_t* index = rocke_b_add(
            b, rocke_b_add(b, c->bias_base, rocke_b_mul(b, query_pos, c->bias_stride_q)), key_pos);
        rocke_value_t* bias;
        if(!c->bias_q)
        {
            bias = rocke_b_masked_global_load(
                b, c->attn_bias, index, keep, c->bias_zero, rocke_f32(), 4);
        }
        else
        {
            rocke_value_t* safe = rocke_b_select(b, keep, index, rocke_b_const_i32(b, 0));
            rocke_value_t* loaded = rocke_b_cast_to_f32(
                b, rocke_b_global_load(b, c->attn_bias, safe, c->bias_elem, 2));
            bias = rocke_b_select(b, keep, loaded, c->bias_zero);
        }
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
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_fmha_fwd: unknown arch '%s'", arch);
        return ROCKE_ERR_VALUE;
    }
    const int wave = target->wave_size;
    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       wave * (spec->transposed_qk ? spec->num_waves : 1));
    if(spec->scheduler_strategy != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->scheduler_strategy);
    wmma_declare_params(b, spec);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c16 = rocke_b_const_i32(b, ROCKE_WMMA_FMHA_FWD_BLOCK_M);
    rocke_value_t* q_tile = rocke_b_block_id_x(b);
    rocke_value_t* head = rocke_b_block_id_y(b);
    rocke_value_t* batch = rocke_b_block_id_z(b);
    rocke_value_t* value_offset = NULL;
    if(spec->value_tile_size != 0)
    {
        rocke_value_t* batch_tile = batch;
        rocke_value_t* tiles = rocke_b_const_i32(b, wmma_v_dim(spec) / spec->value_tile_size);
        batch = rocke_b_div(b, batch_tile, tiles);
        rocke_value_t* value_tile = rocke_b_mod(b, batch_tile, tiles);
        value_offset = rocke_b_mul(b, value_tile, rocke_b_const_i32(b, spec->value_tile_size));
    }

    rocke_value_t* group_size = rocke_b_div(
        b, rocke_b_get_param(b, "num_query_heads"), rocke_b_get_param(b, "num_kv_heads"));
    rocke_value_t* kv_head = rocke_b_div(b, head, group_size);
    rocke_value_t* seqlen_q = rocke_b_get_param(b, "seqlen_q");
    rocke_value_t* seqlen_k = rocke_b_get_param(b, "seqlen_k");
    rocke_value_t* q_step
        = spec->transposed_qk ? rocke_b_const_i32(b, ROCKE_WMMA_FMHA_FWD_BLOCK_M * spec->num_waves)
                              : c16;
    rocke_value_t* q_row0 = rocke_b_mul(b, q_tile, q_step);

    rocke_value_t* Q = rocke_b_get_param(b, "Q");
    rocke_value_t* K = rocke_b_get_param(b, "K");
    rocke_value_t* V = rocke_b_get_param(b, "V");
    rocke_value_t* O = rocke_b_get_param(b, "O");
    rocke_value_t* LSE = rocke_b_get_param(b, "LSE");
    rocke_value_t* q_global = NULL;
    rocke_value_t* batch_off_k = NULL;
    rocke_value_t* batch_off_v = NULL;
    const bool packed = strcmp(spec->layout, "dense") != 0;
    if(!packed)
    {
        rocke_value_t* q_elem_bytes = rocke_b_const_i32(b, 2);
        rocke_value_t* kv_elem_bytes = rocke_b_const_i32(b, spec->kv_dtype[0] != '\0' ? 1 : 2);
        rocke_value_t* lse_elem_bytes = rocke_b_const_i32(b, 4);
        const auto batchPtr
            = [&](rocke_value_t* pointer, rocke_value_t* stride, rocke_value_t* elemBytes) {
                  rocke_value_t* elements = rocke_b_mul(b, batch, stride);
                  rocke_value_t* bytes = rocke_b_mul(b, elements, elemBytes);
                  return rocke_b_global_ptr_add(b, pointer, bytes);
              };
        Q = batchPtr(Q, rocke_b_get_param(b, "stride_q_batch"), q_elem_bytes);
        K = batchPtr(K, rocke_b_get_param(b, "stride_k_batch"), kv_elem_bytes);
        V = batchPtr(V, rocke_b_get_param(b, "stride_v_batch"), kv_elem_bytes);
        O = batchPtr(O, rocke_b_get_param(b, "stride_o_batch"), q_elem_bytes);
        LSE = batchPtr(LSE, rocke_b_get_param(b, "stride_lse_batch"), lse_elem_bytes);
        q_global = q_row0;
        batch_off_k = batch_off_v = c0;
    }
    else
    {
        rocke_value_t* next_batch = rocke_b_add(b, batch, rocke_b_const_i32(b, 1));
        rocke_value_t* cu_q = rocke_b_get_param(b, "cu_seqlens_q");
        rocke_value_t* batch_row_q = rocke_b_global_load_i32(b, cu_q, batch, 4);
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
            batch_off_k = batch_off_v = c0;
        }
        q_global = rocke_b_add(b, q_row0, batch_row_q);
    }

    rocke_value_t* bottom_right = rocke_b_get_param(b, "bottom_right");
    rocke_value_t* use_bottom_right = rocke_b_cmp_ne(b, bottom_right, c0);
    rocke_value_t* bottom_right_offset = rocke_b_sub(b, seqlen_k, seqlen_q);
    rocke_value_t* context = rocke_b_select(b, use_bottom_right, bottom_right_offset, c0);
    const bool windowed = spec->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW;
    const bool score_features = windowed || spec->use_softcap || spec->use_alibi
                                || spec->use_qq_bias || spec->use_attn_bias;
    const bool strict = spec->mask_mode != ROCKE_FMHA_MASK_NONE
                        || strcmp(wmma_dtype(spec), "bf16") == 0 || spec->kv_tail
                        || spec->causal_tile_skip || spec->use_softcap || spec->use_sinks
                        || spec->use_alibi || spec->use_qq_bias || spec->use_attn_bias || packed;
    rocke_value_t* masked = strict ? rocke_b_const_f32(b, -INFINITY) : NULL;

    rocke_mfma_attn_params_t p;
    memset(&p, 0, sizeof(p));
    p.Q = Q;
    p.K = K;
    p.V = V;
    p.O = O;
    p.head_size = spec->head_size;
    p.seqlen_k = seqlen_k;
    p.q_tile_base = q_global;
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
    p.sliding_window = 0;
    p.causal_ctx_offset = context;
    p.mask_neg_inf = masked;
    p.k_token_offset_elems = batch_off_k;
    p.v_token_offset_elems = batch_off_v;
    p.wmma_v_lds_stage = spec->v_lds_stage;
    p.arch = arch;
    p.wmma_seqlen_q = spec->query_tail || packed ? seqlen_q : NULL;
    p.wmma_kv_tail = spec->kv_tail || packed;
    p.wmma_value_tile_size = spec->value_tile_size;
    p.wmma_value_offset = value_offset;
    p.wmma_v_head_size = spec->v_head_size;
    p.lse = LSE;
    p.write_lse = rocke_b_get_param(b, "write_lse");
    p.stride_lse_token = rocke_b_get_param(b, "stride_lse_token");
    p.stride_lse_head = rocke_b_get_param(b, "stride_lse_head");

    wmma_score_context features{};
    features.context = context;
    features.windowed = windowed;
    if(score_features || spec->use_sinks)
    {
        features.log2e = rocke_b_const_f32(b, 1.4426950408889634);
        if(spec->use_softcap)
            features.cap = rocke_b_fmul(b, rocke_b_get_param(b, "softcap"), features.log2e);
        if(spec->use_sinks)
        {
            const rocke_type_t* elem = rocke_mfma_attn_ir_type_for_dtype(b, p.dtype);
            rocke_value_t* value
                = rocke_b_global_load(b, rocke_b_get_param(b, "sink_ptr"), head, elem, 2);
            p.sink_log2 = rocke_b_fmul(b, rocke_b_cast_to_f32(b, value), features.log2e);
        }
        if(spec->use_alibi)
        {
            rocke_value_t* value = rocke_b_global_load(
                b, rocke_b_get_param(b, "alibi_slopes_ptr"), head, rocke_f32(), 4);
            features.slope = rocke_b_fmul(b, value, features.log2e);
        }
        if(spec->use_qq_bias)
            features.zero_f = rocke_b_const_f32(b, 0.0);
        if(spec->use_qq_bias || windowed)
            features.zero_i = rocke_b_const_i32(b, 0);
        if(spec->use_qq_bias)
        {
            features.qq_bias = rocke_b_get_param(b, "qq_bias_ptr");
            features.qq_rows = rocke_b_get_param(b, "qq_bias_rows");
            features.qq_cols = rocke_b_get_param(b, "qq_bias_cols");
            features.qq_stride = rocke_b_get_param(b, "qq_bias_stride");
        }
        if(windowed)
        {
            features.window_left = rocke_b_get_param(b, "window_left");
            features.window_right = rocke_b_get_param(b, "window_right");
            features.masked = masked;
        }
        if(spec->use_attn_bias)
        {
            features.bias_q = wmma_bias_is_q(spec);
            features.bias_elem
                = features.bias_q ? rocke_mfma_attn_ir_type_for_dtype(b, p.dtype) : rocke_f32();
            features.bias_zero = rocke_b_const_f32(b, 0.0);
            features.attn_bias = rocke_b_get_param(b, "attn_bias_ptr");
            features.bias_stride_q = rocke_b_get_param(b, "bias_stride_q");
            rocke_value_t* bias_b = rocke_b_mul(b, batch, rocke_b_get_param(b, "bias_stride_b"));
            rocke_value_t* bias_h = rocke_b_mul(b, head, rocke_b_get_param(b, "bias_stride_h"));
            features.bias_base = rocke_b_add(b, bias_b, bias_h);
            features.seqlen_q = seqlen_q;
            features.seqlen_k = seqlen_k;
        }
    }
    if(score_features)
    {
        p.extra_score_transform = wmma_score_transform;
        p.extra_score_transform_user = &features;
    }
    if(windowed || spec->causal_tile_skip)
    {
        rocke_value_t* left
            = windowed ? rocke_b_get_param(b, "window_left") : rocke_b_const_i32(b, -1);
        rocke_value_t* right = windowed ? rocke_b_get_param(b, "window_right") : c0;
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* one = rocke_b_const_i32(b, 1);
        rocke_value_t* last = rocke_b_const_i32(b, 15);
        rocke_value_t* first_center = rocke_b_add(b, q_row0, context);
        rocke_value_t* lower = rocke_b_sub(b, first_center, left);
        lower = rocke_b_select(b, rocke_b_cmp_lt(b, left, zero), zero, lower);
        lower = rocke_b_select(b, rocke_b_cmp_lt(b, lower, zero), zero, lower);
        rocke_value_t* q_last = rocke_b_add(b, q_row0, last);
        rocke_value_t* q_limit = rocke_b_sub(b, seqlen_q, one);
        q_last = rocke_b_select(b, rocke_b_cmp_gt(b, q_last, q_limit), q_limit, q_last);
        rocke_value_t* centered_last = rocke_b_add(b, q_last, context);
        rocke_value_t* right_plus_one = rocke_b_add(b, right, one);
        rocke_value_t* upper = rocke_b_add(b, centered_last, right_plus_one);
        upper = rocke_b_select(b, rocke_b_cmp_lt(b, right, zero), seqlen_k, upper);
        upper = rocke_b_select(b, rocke_b_cmp_gt(b, upper, seqlen_k), seqlen_k, upper);
        upper = rocke_b_select(b, rocke_b_cmp_lt(b, upper, zero), zero, upper);
        if(lower != NULL)
            p.k_tile_start = rocke_b_div(b, lower, c16);
        p.k_tile_stop = rocke_b_div(b, rocke_b_add(b, upper, last), c16);
    }

    wmma_paged_row_context paged_k{}, paged_v{};
    if(strcmp(spec->layout, "paged") == 0)
    {
        rocke_value_t* page_log2 = rocke_b_const_i32(
            b, std::countr_zero(static_cast<unsigned int>(spec->page_block_size)));
        rocke_value_t* page_mask = rocke_b_const_i32(b, spec->page_block_size - 1);
        rocke_value_t* table_row
            = rocke_b_mul(b, batch, rocke_b_get_param(b, "block_table_stride"));
        rocke_value_t* table = rocke_b_get_param(b, "block_table");
        paged_k = {table,
                   table_row,
                   page_log2,
                   page_mask,
                   rocke_b_get_param(b, "stride_k_block"),
                   p.stride_k_token,
                   rocke_b_mul(b, kv_head, p.stride_k_head)};
        paged_v = {table,
                   table_row,
                   page_log2,
                   page_mask,
                   rocke_b_get_param(b, "stride_v_block"),
                   p.stride_v_token,
                   rocke_b_mul(b, kv_head, p.stride_v_head)};
        p.k_row_base_fn = wmma_paged_row;
        p.k_row_base_user = &paged_k;
        p.v_row_base_fn = wmma_paged_row;
        p.v_row_base_user = &paged_v;
    }

    if(spec->kv_dtype[0] != '\0')
    {
        p.kv_dtype = spec->kv_dtype;
        p.k_scale = rocke_b_get_param(b, "k_scale");
        p.v_scale = rocke_b_get_param(b, "v_scale");
    }

    if(spec->transposed_qk)
    {
        if(spec->mask_mode != ROCKE_FMHA_MASK_CAUSAL)
            p.causal_ctx_offset = NULL;
        (void)rocke_wmma_swapqk_fwd_inner_body(b, &p, spec->block_n, spec->num_waves);
    }
    else
        (void)rocke_mfma_attention_fwd_inner_body(b, &p);

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
 * wmma_fmha_fwd_grid(spec, seqlen_q, num_query_heads, batch)
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_grid(const rocke_wmma_fmha_fwd_spec_t* spec,
                                        int seqlen_q,
                                        int num_query_heads,
                                        int batch,
                                        int out[3])
{
    if(spec == NULL || out == NULL || !wmma_valid_layout(spec) || !wmma_valid_value_tile(spec))
    {
        return ROCKE_ERR_VALUE;
    }
    if(spec->transposed_qk && spec->num_waves != 1 && spec->num_waves != 2)
        return ROCKE_ERR_VALUE;
    int block_m = ROCKE_WMMA_FMHA_FWD_BLOCK_M * (spec->transposed_qk ? spec->num_waves : 1);
    if(strcmp(spec->layout, "dense") == 0 && !spec->query_tail && seqlen_q % block_m != 0)
        return ROCKE_ERR_VALUE;
    if(num_query_heads <= 0)
        return ROCKE_ERR_VALUE;
    int value_tiles = spec->value_tile_size != 0 ? wmma_v_dim(spec) / spec->value_tile_size : 1;
    if(spec->value_tile_size != 0 && (batch < 0 || batch > 0x7FFFFFFF / value_tiles))
        return ROCKE_ERR_VALUE;
    out[0] = (seqlen_q + block_m - 1) / block_m;
    out[1] = num_query_heads;
    out[2] = batch * value_tiles;
    return ROCKE_OK;
}

/* --------------------------------------------------------------------------- *
 * wmma_fmha_fwd_signature(spec): fixed runtime-shape ABI, via a transient
 * probe builder that runs _declare_params and reads the parameter order.
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
            status
                = rocke_sig_param(arena, param->name, type->pointee->name, type->space, &items[i]);
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
