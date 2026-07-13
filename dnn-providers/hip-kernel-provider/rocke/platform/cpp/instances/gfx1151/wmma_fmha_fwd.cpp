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

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define WMMA_FMHA_DEFAULT_NAME "rocke_wmma_fmha_fwd"
#define WMMA_FMHA_DEFAULT_ARCH "gfx1151"

/* ----- small helpers ----- */

static void wmma_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
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
    return s;
}

/* --------------------------------------------------------------------------- *
 * WmmaFmhaFwdSpec.kernel_name()
 *
 * kernel_name_join(name, "wmma16x16x16", "H{hd}", "HQ{hq}", "HK{kv_heads}",
 *   "fp16", mask_mode, "vlds" if v_lds_stage else "vgather").
 * --------------------------------------------------------------------------- */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap)
{
    const char* name;
    const char* mask;
    char h[32], hq[32], hk[32];
    const char* parts[7];

    if(spec == NULL || out == NULL)
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

    parts[0] = "wmma16x16x16";
    parts[1] = h;
    parts[2] = hq;
    parts[3] = hk;
    parts[4] = "fp16";
    parts[5] = mask;
    parts[6] = spec->v_lds_stage ? "vlds" : "vgather";

    return rocke_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
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

    /* target = ArchTarget.from_gfx(arch) -- KeyError path. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown arch '%s'", arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* op = target.mma.by_op_id(_wmma_op_id_for_arch(arch)); reject if absent or
     * not "wmma". gfx1201 (RDNA4) selects the split-K wmma_gfx12_* atom; gfx11
     * (RDNA3/3.5) the cross-half-duplicated atom. */
    const char* op_id = (strcmp(arch, "gfx1201") == 0) ? "wmma_gfx12_f32_16x16x16_f16"
                                                       : ROCKE_WMMA_FMHA_FWD_OP_ID;
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

    /* LDS budget: one 16x16 f16 P-staging tile, plus (with V-LDS staging) one
     * 16 x head_size f16 V tile. */
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
static void wmma_declare_params(rocke_ir_builder_t* b)
{
    rocke_param_opts_t opts;
    const rocke_type_t* ptr_f16 = rocke_ptr_type(b, rocke_f16(), "global");

    /* Q/K/V = param(ptr<f16,global>, noalias, readonly, align16). */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)rocke_b_param(b, "Q", ptr_f16, &opts);
    (void)rocke_b_param(b, "K", ptr_f16, &opts);
    (void)rocke_b_param(b, "V", ptr_f16, &opts);

    /* O = param(ptr<f16,global>, noalias, writeonly, align16). */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)rocke_b_param(b, "O", ptr_f16, &opts);

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

    /* b.kernel.attrs["max_workgroup_size"] = wave */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", wave);

    /* _declare_params(b) */
    wmma_declare_params(b);

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
    q_row0 = rocke_b_mul(b, q_tile, c16); /* first Q row of this tile      */
    batch_row_q = rocke_b_mul(b, batch, seqlen_q); /* batch shift in Q rows         */
    batch_off_k
        = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), rocke_b_get_param(b, "stride_k_token"));
    batch_off_v
        = rocke_b_mul(b, rocke_b_mul(b, batch, seqlen_k), rocke_b_get_param(b, "stride_v_token"));

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
    p.dtype = "f16";
    p.mask_mode = wmma_to_attn_mask(spec->mask_mode);
    p.sliding_window = spec->sliding_window;
    p.causal_ctx_offset = rocke_b_const_i32(b, 0);
    p.k_token_offset_elems = batch_off_k;
    p.v_token_offset_elems = batch_off_v;
    p.wmma_v_lds_stage = spec->v_lds_stage;
    p.arch = arch;

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
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* if seqlen_q % BLOCK_M != 0: raise ValueError(...) */
    if(seqlen_q % ROCKE_WMMA_FMHA_FWD_BLOCK_M != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = seqlen_q / ROCKE_WMMA_FMHA_FWD_BLOCK_M;
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
    rocke_ir_builder_t b;
    rocke_status_t st;
    rocke_sig_entry_t* items;
    int n;
    int i;
    int k;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_ir_builder_init(&b, "rocke_wmma_fmha_fwd_sig_probe");
    if(st != ROCKE_OK)
    {
        return st;
    }
    wmma_declare_params(&b);

    n = b.kernel->num_params;
    items = (rocke_sig_entry_t*)rocke_arena_alloc(arena, (size_t)n * sizeof(rocke_sig_entry_t));
    if(items == NULL)
    {
        return ROCKE_ERR_OOM;
    }

    /* The first four params are the Q/K/V/O global pointers (ptr<f16,global>);
     * the rest are scalars (f32 scale_log2, i32 seqlen/strides). */
    k = 0;
    for(i = 0; i < n; ++i)
    {
        const rocke_param_t* pr = b.kernel->params[i];
        items[k].name = pr->name;
        if(i < 4)
        {
            items[k].type = "ptr<f16, global>";
        }
        else if(i == 4)
        {
            items[k].type = "f32"; /* scale_log2 */
        }
        else
        {
            items[k].type = "i32";
        }
        ++k;
    }

    *out_items = items;
    *out_count = (size_t)k;
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
