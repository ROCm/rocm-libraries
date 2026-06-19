/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gfx1151_wmma_fmha_fwd.c -- C99 port of
 * ck_dsl/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * Byte-identical builder-call sequence vs the Python build_wmma_fmha_fwd: a raw
 * IRBuilder declares the same params in the same order (_declare_params), bakes
 * the same max_workgroup_size attr, decodes the (seqlen_q//16, num_query_heads,
 * batch) grid the same way, computes the same GQA kv_head + per-batch offsets,
 * and calls the already-ported helper ckc_mfma_attention_fwd_inner_body with the
 * same operands / attrs (incl. wmma_v_lds_stage), then b.ret(). All the wave32
 * QK->softmax->PV IR emission is delegated to that helper (which dispatches to
 * the WMMA wave32 inner body on the RDNA target); this file is the thin
 * spec->kernel adapter plus a lower-to-.ll convenience.
 */

#include "ckc/instance_gfx1151_wmma_fmha_fwd.h"

#include <stdio.h>
#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/arch_target.h"
#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

#define WMMA_FMHA_DEFAULT_NAME "ck_dsl_wmma_fmha_fwd"
#define WMMA_FMHA_DEFAULT_ARCH "gfx1151"

/* ----- small helpers ----- */

static void wmma_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if (reason != NULL && reason_cap > 0)
    {
        size_t n = strlen(msg);
        if (n >= reason_cap)
        {
            n = reason_cap - 1;
        }
        memcpy(reason, msg, n);
        reason[n] = '\0';
    }
}

/* WmmaFmhaFwdSpec.kv_heads property: num_kv_heads or num_query_heads. */
static int wmma_kv_heads(const ckc_wmma_fmha_fwd_spec_t* spec)
{
    return spec->num_kv_heads != 0 ? spec->num_kv_heads : spec->num_query_heads;
}

/* Map the shared FMHA mask enum to the attention-helper mask enum. WMMA FMHA
 * supports only NONE / CAUSAL (validated up front); anything else => NONE. */
static ckc_attn_mask_mode_t wmma_to_attn_mask(ckc_fmha_mask_mode_t m)
{
    switch (m)
    {
        case CKC_FMHA_MASK_CAUSAL:
            return CKC_ATTN_MASK_CAUSAL;
        case CKC_FMHA_MASK_NONE:
        default:
            return CKC_ATTN_MASK_NONE;
    }
}

/* --------------------------------------------------------------------------- *
 * ckc_wmma_fmha_fwd_spec_default
 * --------------------------------------------------------------------------- */
ckc_wmma_fmha_fwd_spec_t ckc_wmma_fmha_fwd_spec_default(void)
{
    ckc_wmma_fmha_fwd_spec_t s;
    s.head_size = 0;
    s.num_query_heads = 0;
    s.num_kv_heads = 0;
    s.mask_mode = CKC_FMHA_MASK_NONE;
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
ckc_status_t
ckc_wmma_fmha_fwd_kernel_name(const ckc_wmma_fmha_fwd_spec_t* spec, char* out, size_t out_cap)
{
    const char* name;
    const char* mask;
    char h[32], hq[32], hk[32];
    const char* parts[7];

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    name = (spec->name != NULL) ? spec->name : WMMA_FMHA_DEFAULT_NAME;
    mask = ckc_fmha_mask_mode_name(spec->mask_mode);
    if (mask == NULL)
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

    return ckc_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
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
bool ckc_wmma_fmha_fwd_is_valid_spec(const ckc_wmma_fmha_fwd_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const ckc_archtarget_t* target;
    const ckc_mmaop_t* op;
    long bytes_lds;
    char buf[256];

    if (spec == NULL)
    {
        wmma_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = WMMA_FMHA_DEFAULT_ARCH;
    }

    /* target = ArchTarget.from_gfx(arch) -- KeyError path. */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown arch '%s'", arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* op = target.mma.by_op_id(_WMMA_OP_ID); reject if absent or not "wmma". */
    op = ckc_archtarget_by_op_id(target, CKC_WMMA_FMHA_FWD_OP_ID);
    if (op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0)
    {
        snprintf(buf, sizeof(buf),
                 "WMMA %s atom absent on %s (WMMA is an RDNA/gfx11 instruction; "
                 "this kernel is gfx1151-only)",
                 CKC_WMMA_FMHA_FWD_OP_ID, arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* wave-size agreement (WMMA atom wave32 vs the target). */
    if (target->wave_size != op->wave_size)
    {
        snprintf(buf, sizeof(buf),
                 "arch wave size %d != WMMA atom wave size %d on %s",
                 target->wave_size, op->wave_size, arch);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* head_size % 16 != 0 */
    if (spec->head_size % 16 != 0)
    {
        snprintf(buf, sizeof(buf), "head_size must be a multiple of 16 (got %d)",
                 spec->head_size);
        wmma_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* LDS budget: one 16x16 f16 P-staging tile, plus (with V-LDS staging) one
     * 16 x head_size f16 V tile. */
    bytes_lds = (long)CKC_WMMA_FMHA_FWD_BLOCK_M * CKC_WMMA_FMHA_FWD_BLOCK_K * 2;
    if (spec->v_lds_stage)
    {
        bytes_lds += (long)CKC_WMMA_FMHA_FWD_BLOCK_M * spec->head_size * 2;
    }
    if (!ckc_archtarget_fits_lds(target, bytes_lds))
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
 * params are recovered later via ckc_b_get_param. */
static void wmma_declare_params(ckc_ir_builder_t* b)
{
    ckc_param_opts_t opts;
    const ckc_type_t* ptr_f16 = ckc_ptr_type(b, ckc_f16(), "global");

    /* Q/K/V = param(ptr<f16,global>, noalias, readonly, align16). */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)ckc_b_param(b, "Q", ptr_f16, &opts);
    (void)ckc_b_param(b, "K", ptr_f16, &opts);
    (void)ckc_b_param(b, "V", ptr_f16, &opts);

    /* O = param(ptr<f16,global>, noalias, writeonly, align16). */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    (void)ckc_b_param(b, "O", ptr_f16, &opts);

    /* scalars */
    (void)ckc_b_param(b, "scale_log2", ckc_f32(), NULL);
    (void)ckc_b_param(b, "seqlen_q", ckc_i32(), NULL);
    (void)ckc_b_param(b, "seqlen_k", ckc_i32(), NULL);

    /* element strides (token, head) per tensor, in Python order. */
    (void)ckc_b_param(b, "stride_q_token", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_q_head", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_k_token", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_k_head", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_v_token", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_v_head", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_o_token", ckc_i32(), NULL);
    (void)ckc_b_param(b, "stride_o_head", ckc_i32(), NULL);
}

/* --------------------------------------------------------------------------- *
 * The shared Python build body: emit the adapter into an already-initialised
 * builder `b` (kernel name already set). Returns CKC_OK or the sticky status.
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
static ckc_status_t wmma_emit_body(ckc_ir_builder_t* b,
                                   const ckc_wmma_fmha_fwd_spec_t* spec,
                                   const char* arch)
{
    const ckc_archtarget_t* target;
    int wave;
    int qh, kvh;
    ckc_value_t* c16;
    ckc_value_t* q_tile;
    ckc_value_t* head;
    ckc_value_t* batch;
    ckc_value_t* kv_head;
    ckc_value_t* seqlen_q;
    ckc_value_t* seqlen_k;
    ckc_value_t* q_row0;
    ckc_value_t* batch_row_q;
    ckc_value_t* batch_off_k;
    ckc_value_t* batch_off_v;
    ckc_mfma_attn_params_t p;

    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "wmma_fmha_fwd: unknown arch '%s'", arch);
        return CKC_ERR_VALUE;
    }
    wave = target->wave_size; /* 32 for WMMA */

    /* b.kernel.attrs["max_workgroup_size"] = wave */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", wave);

    /* _declare_params(b) */
    wmma_declare_params(b);

    c16 = ckc_b_const_i32(b, CKC_WMMA_FMHA_FWD_BLOCK_M);

    /* grid decode */
    q_tile = ckc_b_block_id_x(b); /* Q-tile index (16 rows) */
    head = ckc_b_block_id_y(b);   /* query head             */
    batch = ckc_b_block_id_z(b);  /* batch index            */

    /* GQA: kv_head = head // (num_query_heads // kv_heads). */
    qh = spec->num_query_heads;
    kvh = wmma_kv_heads(spec);
    if (kvh == qh)
    {
        kv_head = head;
    }
    else
    {
        kv_head = ckc_b_div(b, head, ckc_b_const_i32(b, qh / kvh));
    }

    seqlen_q = ckc_b_get_param(b, "seqlen_q");
    seqlen_k = ckc_b_get_param(b, "seqlen_k");

    /* per-batch shifts (Python op order). */
    q_row0 = ckc_b_mul(b, q_tile, c16);          /* first Q row of this tile      */
    batch_row_q = ckc_b_mul(b, batch, seqlen_q); /* batch shift in Q rows         */
    batch_off_k = ckc_b_mul(b, ckc_b_mul(b, batch, seqlen_k),
                            ckc_b_get_param(b, "stride_k_token"));
    batch_off_v = ckc_b_mul(b, ckc_b_mul(b, batch, seqlen_k),
                            ckc_b_get_param(b, "stride_v_token"));

    /* mfma_attention_fwd_inner_body(...) with the WMMA v-LDS staging flag. */
    memset(&p, 0, sizeof(p));
    p.Q = ckc_b_get_param(b, "Q");
    p.K = ckc_b_get_param(b, "K");
    p.V = ckc_b_get_param(b, "V");
    p.O = ckc_b_get_param(b, "O");
    p.head_size = spec->head_size;
    p.seqlen_k = seqlen_k;
    /* global Q/O row index folds the batch shift in; within-batch q position for
     * the mask is q_pos_base = q_row0. */
    p.q_tile_base = ckc_b_add(b, q_row0, batch_row_q);
    p.head_idx = head;
    p.kv_head_idx = kv_head;
    p.q_pos_base = q_row0;
    p.stride_q_token = ckc_b_get_param(b, "stride_q_token");
    p.stride_q_head = ckc_b_get_param(b, "stride_q_head");
    p.stride_k_token = ckc_b_get_param(b, "stride_k_token");
    p.stride_k_head = ckc_b_get_param(b, "stride_k_head");
    p.stride_v_token = ckc_b_get_param(b, "stride_v_token");
    p.stride_v_head = ckc_b_get_param(b, "stride_v_head");
    p.stride_o_token = ckc_b_get_param(b, "stride_o_token");
    p.stride_o_head = ckc_b_get_param(b, "stride_o_head");
    p.scale_log2 = ckc_b_get_param(b, "scale_log2");
    p.dtype = "f16";
    p.mask_mode = wmma_to_attn_mask(spec->mask_mode);
    p.sliding_window = spec->sliding_window;
    p.causal_ctx_offset = ckc_b_const_i32(b, 0);
    p.k_token_offset_elems = batch_off_k;
    p.v_token_offset_elems = batch_off_v;
    p.wmma_v_lds_stage = spec->v_lds_stage;
    p.arch = arch;

    (void)ckc_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    ckc_b_ret(b);

    return ckc_ir_builder_status(b);
}

/* --------------------------------------------------------------------------- *
 * build_wmma_fmha_fwd(spec, arch)
 *
 * `b` is the destination builder, assumed already initialised by the caller
 * with spec.kernel_name() (the gfx1201 WMMA GEMM call contract). Validates,
 * emits the adapter body, and returns b.kernel (NULL on validation / IR error).
 * --------------------------------------------------------------------------- */
ckc_kernel_def_t* ckc_build_wmma_fmha_fwd(ckc_ir_builder_t* b,
                                          const ckc_wmma_fmha_fwd_spec_t* spec,
                                          const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char reason[CKC_ERR_MSG_CAP];

        if (b == NULL || spec == NULL)
        {
            return NULL;
        }
        if (arch == NULL)
        {
            arch = WMMA_FMHA_DEFAULT_ARCH;
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if (!ckc_wmma_fmha_fwd_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)ckc_i_set_err(b, CKC_ERR_VALUE, "invalid wmma_fmha_fwd spec: %s", reason);
            return NULL;
        }

        if (wmma_emit_body(b, spec, arch) != CKC_OK)
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* --------------------------------------------------------------------------- *
 * wmma_fmha_fwd_grid(spec, seqlen_q, batch)
 * --------------------------------------------------------------------------- */
ckc_status_t
ckc_wmma_fmha_fwd_grid(const ckc_wmma_fmha_fwd_spec_t* spec, int seqlen_q, int batch, int out[3])
{
    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* if seqlen_q % BLOCK_M != 0: raise ValueError(...) */
    if (seqlen_q % CKC_WMMA_FMHA_FWD_BLOCK_M != 0)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = seqlen_q / CKC_WMMA_FMHA_FWD_BLOCK_M;
    out[1] = spec->num_query_heads;
    out[2] = batch;
    return CKC_OK;
}

/* --------------------------------------------------------------------------- *
 * wmma_fmha_fwd_signature(spec): the kernel ABI (Q/K/V/O ptrs, scale_log2/
 * seqlen_q/seqlen_k scalars, q/k/v/o stride pairs), via a transient probe
 * builder that runs _declare_params and reads the param order from the kernel.
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_wmma_fmha_fwd_signature(const ckc_wmma_fmha_fwd_spec_t* spec,
                                         ckc_arena_t* arena,
                                         const ckc_sig_entry_t** out_items,
                                         size_t* out_count)
{
    ckc_ir_builder_t b;
    ckc_status_t st;
    ckc_sig_entry_t* items;
    int n;
    int i;
    int k;

    if (spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_ir_builder_init(&b, "ck_dsl_wmma_fmha_fwd_sig_probe");
    if (st != CKC_OK)
    {
        return st;
    }
    wmma_declare_params(&b);

    n = b.kernel->num_params;
    items = (ckc_sig_entry_t*)ckc_arena_alloc(arena, (size_t)n * sizeof(ckc_sig_entry_t));
    if (items == NULL)
    {
        return CKC_ERR_OOM;
    }

    /* The first four params are the Q/K/V/O global pointers (ptr<f16,global>);
     * the rest are scalars (f32 scale_log2, i32 seqlen/strides). */
    k = 0;
    for (i = 0; i < n; ++i)
    {
        const ckc_param_t* pr = b.kernel->params[i];
        items[k].name = pr->name;
        if (i < 4)
        {
            items[k].type = "ptr<f16, global>";
        }
        else if (i == 4)
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
    return CKC_OK;
}

/* --------------------------------------------------------------------------- *
 * ckc_wmma_fmha_fwd_lower_to_llvm -- build + lower to .ll convenience.
 *
 * Owns and frees its own IRBuilder for the whole lower so the kernel stays
 * alive through lowering.
 * --------------------------------------------------------------------------- */
ckc_status_t ckc_wmma_fmha_fwd_lower_to_llvm(const ckc_wmma_fmha_fwd_spec_t* spec,
                                             const char* arch,
                                             ckc_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap)
{
    char name_buf[256];
    ckc_ir_builder_t b;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = WMMA_FMHA_DEFAULT_ARCH;
    }

    if (!ckc_wmma_fmha_fwd_is_valid_spec(spec, arch, err, err_cap))
    {
        return CKC_ERR_VALUE;
    }

    if (ckc_wmma_fmha_fwd_kernel_name(spec, name_buf, sizeof(name_buf)) != CKC_OK)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: kernel name too long");
        return CKC_ERR_VALUE;
    }

    st = ckc_ir_builder_init(&b, name_buf);
    if (st != CKC_OK)
    {
        wmma_set_reason(err, err_cap, "lower_to_llvm: builder init failed");
        return st;
    }

    st = wmma_emit_body(&b, spec, arch);
    if (st != CKC_OK || b.kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        wmma_set_reason(err, err_cap, m != NULL ? m : "build_wmma_fmha_fwd failed");
        return st != CKC_OK ? st : CKC_ERR_VALUE;
    }

    st = ckc_lower_kernel_to_llvm_ex(b.kernel, flavor, arch, out_ll, err, err_cap);
    return st;
}
