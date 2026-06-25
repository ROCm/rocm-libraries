// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ckc/instance_fmha_fwd_fp8.c -- C99 port of ck_dsl/instances/common/fmha_fwd_fp8.py.
 *
 * Byte-identical builder-call sequence vs the Python build_fmha_fwd_fp8: same
 * param declaration order, same grid decode, same waves_per_eu attr, same
 * fmul/mul/const op order, and the same single call into the already-ported
 * helper ckc_mfma_attention_fwd_inner_body. See the header for the symbol map.
 */
#include "ckc/instance_fmha_fwd_fp8.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arch_target.h" /* ckc_arch_target_t, has_shape */
#include "ckc/arena.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/helper_ck_dsl.core.arch.h" /* ckc_archtarget_from_gfx      */
#include "ckc/helper_ck_dsl.helpers.attention.h" /* ckc_attn_mask_mode_t         */
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h" /* CKC_MFMA_ATTN_BLOCK_M, body  */
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_kernel_name_join         */
#include "ckc/helper_ck_dsl.instances.common._fmha_common.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/lower_llvm.h"

/* ------------------------------------------------------------------ *
 * _FNUZ_FP8_TARGET_FAMILIES (Python frozenset({"gfx9_mfma"}))
 * ------------------------------------------------------------------ */
static bool fp8_fwd_is_fnuz_family(const char* target_family)
{
    return target_family != NULL && strcmp(target_family, "gfx9_mfma") == 0;
}

/* ------------------------------------------------------------------ *
 * KvFp8DType
 * ------------------------------------------------------------------ */
const char* ckc_kv_fp8_dtype_name(ckc_kv_fp8_dtype_t d)
{
    switch(d)
    {
    case CKC_KV_FP8_E4M3:
        return "fp8e4m3";
    case CKC_KV_BF8_E5M2:
        return "bf8e5m2";
    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ *
 * FmhaFwdFp8Spec defaults + kernel_name
 * ------------------------------------------------------------------ */
ckc_fmha_fwd_fp8_spec_t ckc_fmha_fwd_fp8_spec_default(void)
{
    ckc_fmha_fwd_fp8_spec_t s;
    memset(&s, 0, sizeof(s));
    /* common is left zero-initialised; the caller must fill it. */
    s.kv_dtype = CKC_KV_FP8_E4M3;
    s.seqlen_q = 1;
    s.seqlen_k = 0;
    s.fp8_fnuz = false;
    s.has_waves_per_eu = true; /* Python default Optional[int] = 4 */
    s.waves_per_eu = 4;
    s.name = "ck_dsl_fmha_fwd_fp8";
    return s;
}

/* Common-spec dtype spelling (the Python self.common.dtype string). */
static const char* fp8_fwd_common_dtype(const ckc_fmha_fwd_fp8_spec_t* spec)
{
    return spec->common.dtype != NULL ? spec->common.dtype : "f16";
}

ckc_status_t
    ckc_fmha_fwd_fp8_kernel_name(const ckc_fmha_fwd_fp8_spec_t* spec, char* out, size_t out_cap)
{
    char hbuf[32];
    char hqbuf[32];
    char hkbuf[32];
    char qbuf[32];
    const char* parts[7];
    const char* name;
    const char* mask;
    const char* kvname;

    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    name = spec->name != NULL ? spec->name : "ck_dsl_fmha_fwd_fp8";
    mask = ckc_fmha_mask_mode_name(spec->common.mask_mode);
    kvname = ckc_kv_fp8_dtype_name(spec->kv_dtype);
    if(mask == NULL || kvname == NULL)
    {
        return CKC_ERR_VALUE;
    }

    snprintf(hbuf, sizeof(hbuf), "H%d", spec->common.shape.head_size);
    snprintf(hqbuf, sizeof(hqbuf), "HQ%d", spec->common.shape.num_query_heads);
    snprintf(hkbuf, sizeof(hkbuf), "HK%d", spec->common.shape.num_kv_heads);
    snprintf(qbuf, sizeof(qbuf), "Q%d", spec->seqlen_q);

    parts[0] = hbuf;
    parts[1] = hqbuf;
    parts[2] = hkbuf;
    parts[3] = fp8_fwd_common_dtype(spec);
    parts[4] = kvname;
    parts[5] = qbuf;
    parts[6] = mask;

    return ckc_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

/* ------------------------------------------------------------------ *
 * is_valid_spec
 * ------------------------------------------------------------------ */
static void fp8_fwd_set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != NULL && cap > 0)
    {
        snprintf(reason, cap, "%s", msg);
    }
}

bool ckc_fmha_fwd_fp8_is_valid_spec(const ckc_fmha_fwd_fp8_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap)
{
    const ckc_arch_target_t* target;
    const ckc_fmha_common_spec_t* cs;
    const char* common_reason = NULL;
    const char* kvname;
    ckc_arena_t arena;
    long bytes_lds;

    if(spec == NULL)
    {
        fp8_fwd_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    cs = &spec->common;

    /* target = ArchTarget.from_gfx(arch); except KeyError -> reason */
    target = ckc_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    /* ok, why = validate_common_spec(spec.common) */
    ckc_arena_init(&arena, 0);
    if(!ckc_fmha_validate_common_spec(&arena, cs, &common_reason))
    {
        fp8_fwd_set_reason(
            reason, reason_cap, common_reason != NULL ? common_reason : "invalid common spec");
        ckc_arena_destroy(&arena);
        return false;
    }
    ckc_arena_destroy(&arena);

    /* kv_dtype must be 'fp8e4m3' or 'bf8e5m2' */
    kvname = ckc_kv_fp8_dtype_name(spec->kv_dtype);
    if(kvname == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "kv_dtype must be 'fp8e4m3' or 'bf8e5m2', got %d",
                     (int)spec->kv_dtype);
        }
        return false;
    }

    /* G3: OCP-fp8 K/V on a fnuz-decoding target is silently wrong. */
    if(fp8_fwd_is_fnuz_family(target->target_family) && !spec->fp8_fnuz)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "fp8 K/V on %s (target_family=%s) decodes via the native "
                     "e4m3fnuz/e5m2fnuz format, not OCP e4m3fn/e5m2; the default "
                     "%s path assumes OCP bytes and would silently mis-decode "
                     "K/V. Quantise K/V to fnuz and set "
                     "FmhaFwdFp8Spec(fp8_fnuz=True), or run the OCP-fp8 "
                     "attention on gfx950 / gfx11.",
                     arch,
                     target->target_family != NULL ? target->target_family : "?",
                     kvname);
        }
        return false;
    }

    if(spec->seqlen_q <= 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "seqlen_q must be > 0 (got %d)", spec->seqlen_q);
        }
        return false;
    }
    if(spec->seqlen_q % CKC_MFMA_ATTN_BLOCK_M != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "MFMA fp8 attention needs seqlen_q (%d) to be a multiple of "
                     "BLOCK_M (%d)",
                     spec->seqlen_q,
                     CKC_MFMA_ATTN_BLOCK_M);
        }
        return false;
    }
    if(cs->shape.head_size % 16 != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "MFMA fp8 attention needs head_size %% 16 == 0 (got %d)",
                     cs->shape.head_size);
        }
        return false;
    }

    /* The dequant-on-load path emits the f16 16x16x16 atom. */
    if(!ckc_arch_supports_dtype_combo(target, "f16", "f16", "fp32", NULL))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unsupported f16 MFMA dtype combo on %s", arch);
        }
        return false;
    }
    if(!ckc_mma_catalog_has_shape(&target->mma,
                                  NULL,
                                  "f16",
                                  "f16",
                                  "fp32",
                                  CKC_MFMA_ATTN_BLOCK_M,
                                  CKC_MFMA_ATTN_BLOCK_M,
                                  CKC_MFMA_ATTN_BLOCK_M))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "unsupported f16 warp_tile (%d,%d,%d) on %s",
                     CKC_MFMA_ATTN_BLOCK_M,
                     CKC_MFMA_ATTN_BLOCK_M,
                     CKC_MFMA_ATTN_BLOCK_M,
                     arch);
        }
        return false;
    }

    /* LDS budget: one BLOCK_M x BLOCK_M f16 P-staging buffer. */
    bytes_lds = (long)CKC_MFMA_ATTN_BLOCK_M * CKC_MFMA_ATTN_BLOCK_M * 2;
    if(!ckc_arch_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "LDS budget %ld > %d cap on %s",
                     bytes_lds,
                     target->lds_capacity_bytes,
                     arch);
        }
        return false;
    }

    fp8_fwd_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ------------------------------------------------------------------ *
 * _declare_params(kb, spec)  (shared between build + signature)
 * ------------------------------------------------------------------ *
 *
 *     kb.add_tensor("Q", readonly=True)
 *     kb.add_tensor("K", dtype=spec.kv_dtype, readonly=True, align=8)
 *     kb.add_tensor("V", dtype=spec.kv_dtype, readonly=True, align=8)
 *     kb.add_tensor("O", readonly=False, writeonly=True)
 *     kb.add_scalar("k_scale", "f32")
 *     kb.add_scalar("v_scale", "f32")
 *     kb.add_scalar("scale_log2", "f32")
 *     kb.add_scalar("seqlen_q", "i32")
 *     kb.add_scalar("seqlen_k", "i32")
 *     kb.add_strides("q", "k", "v", "o")
 */
static void fp8_fwd_declare_params(ckc_fmha_kernel_builder_t* kb,
                                   const ckc_fmha_fwd_fp8_spec_t* spec)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};
    const char* kv = ckc_kv_fp8_dtype_name(spec->kv_dtype);

    ckc_fmha_kernel_builder_add_tensor(kb,
                                       "Q",
                                       NULL,
                                       /*readonly*/ true,
                                       /*writeonly*/ false,
                                       /*align*/ 16);
    ckc_fmha_kernel_builder_add_tensor(kb,
                                       "K",
                                       kv,
                                       /*readonly*/ true,
                                       /*writeonly*/ false,
                                       /*align*/ 8);
    ckc_fmha_kernel_builder_add_tensor(kb,
                                       "V",
                                       kv,
                                       /*readonly*/ true,
                                       /*writeonly*/ false,
                                       /*align*/ 8);
    ckc_fmha_kernel_builder_add_tensor(kb,
                                       "O",
                                       NULL,
                                       /*readonly*/ false,
                                       /*writeonly*/ true,
                                       /*align*/ 16);
    ckc_fmha_kernel_builder_add_scalar(kb, "k_scale", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "v_scale", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    ckc_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    ckc_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ------------------------------------------------------------------ *
 * Map the FmhaCommonSpec mask mode -> the helper's attention mask enum.
 * (The helper only knows none / causal / sliding_window; alibi / custom
 * never reach this fp8 instance because validate_common_spec gates them.)
 * ------------------------------------------------------------------ */
static ckc_attn_mask_mode_t fp8_fwd_attn_mask(ckc_fmha_mask_mode_t m)
{
    switch(m)
    {
    case CKC_FMHA_MASK_CAUSAL:
        return CKC_ATTN_MASK_CAUSAL;
    case CKC_FMHA_MASK_SLIDING_WINDOW:
        return CKC_ATTN_MASK_SLIDING_WINDOW;
    default:
        return CKC_ATTN_MASK_NONE;
    }
}

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_fp8
 * ------------------------------------------------------------------ */
ckc_kernel_def_t* ckc_build_fmha_fwd_fp8(ckc_fmha_kernel_builder_t* kb,
                                         const ckc_fmha_fwd_fp8_spec_t* spec,
                                         const char* arch)
{
    const ckc_fmha_common_spec_t* s;
    ckc_ir_builder_t* b;
    ckc_kernel_def_t* kernel;
    ckc_value_t* Q;
    ckc_value_t* K;
    ckc_value_t* V;
    ckc_value_t* O;
    ckc_value_t* k_scale;
    ckc_value_t* v_scale;
    ckc_value_t* scale_log2_raw;
    ckc_value_t* scale_log2;
    ckc_value_t* seqlen_k;
    ckc_value_t* q_tile_idx;
    ckc_value_t* head_idx;
    ckc_value_t* kv_head_idx;
    ckc_value_t* q_tile_base;
    ckc_value_t* causal_ctx;
    char reason[512];
    ckc_mfma_attn_params_t p;

    if(kb == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!ckc_fmha_fwd_fp8_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        b = ckc_fmha_kernel_builder_builder(kb);
        if(b != NULL)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fmha_fwd_fp8 spec: %s", reason);
        }
        return NULL;
    }
    s = &spec->common;
    b = ckc_fmha_kernel_builder_builder(kb);

    /* kb.block_size(64) */
    ckc_fmha_kernel_builder_block_size(kb, 64);

    /* _declare_params(kb, spec) */
    fp8_fwd_declare_params(kb, spec);

    /* kb.decode_grid() */
    ckc_fmha_kernel_builder_decode_grid(kb,
                                        /*num_queries_per_kv=None*/ -1,
                                        /*has_batch_axis*/ false,
                                        NULL,
                                        NULL,
                                        NULL);

    /* Occupancy hint: b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu */
    if(spec->has_waves_per_eu)
    {
        kernel = ckc_fmha_kernel_builder_kernel(kb);
        if(kernel != NULL)
        {
            ckc_attr_set_int(b, &kernel->attrs, "waves_per_eu", (int64_t)spec->waves_per_eu);
        }
    }

    Q = ckc_fmha_kernel_builder_tensor(kb, "Q");
    K = ckc_fmha_kernel_builder_tensor(kb, "K");
    V = ckc_fmha_kernel_builder_tensor(kb, "V");
    O = ckc_fmha_kernel_builder_tensor(kb, "O");

    k_scale = ckc_fmha_kernel_builder_scalar(kb, "k_scale");
    v_scale = ckc_fmha_kernel_builder_scalar(kb, "v_scale");
    scale_log2_raw = ckc_fmha_kernel_builder_scalar(kb, "scale_log2");
    /* scale_log2 = b.fmul(scale_log2_raw, k_scale) */
    scale_log2 = ckc_b_fmul(b, scale_log2_raw, k_scale);
    seqlen_k = ckc_fmha_kernel_builder_scalar(kb, "seqlen_k");

    q_tile_idx = kb->q_token;
    head_idx = kb->head_idx;
    kv_head_idx = kb->kv_head_idx;
    /* q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)) */
    q_tile_base = ckc_b_mul(b, q_tile_idx, ckc_b_const_i32(b, CKC_MFMA_ATTN_BLOCK_M));

    /* causal_ctx = b.const_i32(0) if mask in (causal, sliding_window) else None */
    if(s->mask_mode == CKC_FMHA_MASK_CAUSAL || s->mask_mode == CKC_FMHA_MASK_SLIDING_WINDOW)
    {
        causal_ctx = ckc_b_const_i32(b, 0);
    }
    else
    {
        causal_ctx = NULL;
    }

    /* mfma_attention_fwd_inner_body(b, Q=..., K=..., ..., arch=arch) */
    memset(&p, 0, sizeof(p));
    p.Q = Q;
    p.K = K;
    p.V = V;
    p.O = O;
    p.head_size = s->shape.head_size;
    p.seqlen_k = seqlen_k;
    p.q_tile_base = q_tile_base;
    p.head_idx = head_idx;
    p.kv_head_idx = kv_head_idx;
    p.q_pos_base = NULL; /* default => q_tile_base */

    p.stride_q_token = ckc_fmha_kernel_builder_stride_token(kb, "q");
    p.stride_q_head = ckc_fmha_kernel_builder_stride_head(kb, "q");
    p.stride_k_token = ckc_fmha_kernel_builder_stride_token(kb, "k");
    p.stride_k_head = ckc_fmha_kernel_builder_stride_head(kb, "k");
    p.stride_v_token = ckc_fmha_kernel_builder_stride_token(kb, "v");
    p.stride_v_head = ckc_fmha_kernel_builder_stride_head(kb, "v");
    p.stride_o_token = ckc_fmha_kernel_builder_stride_token(kb, "o");
    p.stride_o_head = ckc_fmha_kernel_builder_stride_head(kb, "o");

    p.scale_log2 = scale_log2;
    p.dtype = fp8_fwd_common_dtype(spec);
    p.mask_mode = fp8_fwd_attn_mask(s->mask_mode);
    p.sliding_window = s->sliding_window;
    p.causal_ctx_offset = causal_ctx;
    p.k_token_offset_elems = NULL;
    p.v_token_offset_elems = NULL;

    p.k_row_base_fn = NULL;
    p.k_row_base_user = NULL;
    p.v_row_base_fn = NULL;
    p.v_row_base_user = NULL;

    p.k_tile_start = NULL;
    p.k_tile_stop = NULL;

    p.extra_score_transform = NULL;
    p.extra_score_transform_user = NULL;
    p.extra_mask_predicate = NULL;
    p.extra_mask_predicate_user = NULL;
    p.extra_skip_predicate = NULL;
    p.extra_skip_predicate_user = NULL;
    p.k_block_iter_fn = NULL;
    p.k_block_iter_user = NULL;

    /* fp8 / bf8 K/V dequant on load. */
    p.kv_dtype = ckc_kv_fp8_dtype_name(spec->kv_dtype);
    p.v_scale = v_scale;
    p.use_wider_atom = false;
    p.native_fp8_path = false;
    p.use_async_kv = false;
    p.codebook_ptr = NULL;
    p.wmma_v_lds_stage = false;
    p.arch = arch;

    ckc_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    ckc_b_ret(b);

    if(ckc_ir_builder_status(b) != CKC_OK)
    {
        return NULL;
    }
    return ckc_fmha_kernel_builder_kernel(kb);
}

ckc_kernel_def_t* ckc_build_fmha_fwd_fp8_new(ckc_fmha_kernel_builder_t* kb,
                                             const ckc_fmha_fwd_fp8_spec_t* spec,
                                             const char* arch)
{
    return ckc::guard_builder(ckc_fmha_kernel_builder_builder(kb), [&]() -> ckc_kernel_def_t* {
        char name[256];

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_fmha_fwd_fp8_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_fmha_kernel_builder_init(kb, name, &spec->common) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_fmha_fwd_fp8(kb, spec, arch);
    });
}

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_grid
 * ------------------------------------------------------------------ */
ckc_status_t ckc_fmha_fwd_fp8_grid(const ckc_fmha_fwd_fp8_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = spec->seqlen_q / CKC_MFMA_ATTN_BLOCK_M;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_signature
 * ------------------------------------------------------------------ */
ckc_status_t ckc_fmha_fwd_fp8_signature(const ckc_fmha_fwd_fp8_spec_t* spec,
                                        ckc_arena_t* arena,
                                        const ckc_sig_entry_t** out_items,
                                        size_t* out_count)
{
    ckc_fmha_kernel_builder_t kb;
    ckc_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_fmha_kernel_builder_init(&kb, "ck_dsl_fmha_fwd_fp8_sig_probe", &spec->common);
    if(st != CKC_OK)
    {
        return st;
    }

    fp8_fwd_declare_params(&kb, spec);

    st = ckc_fmha_kernel_builder_signature(&kb, arena, out_items, out_count);
    ckc_fmha_kernel_builder_free(&kb);
    return st;
}

/* ------------------------------------------------------------------ *
 * convenience lower-to-.ll
 * ------------------------------------------------------------------ */
ckc_status_t ckc_fmha_fwd_fp8_lower_to_llvm(const ckc_fmha_fwd_fp8_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap)
{
    ckc_fmha_kernel_builder_t kb;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_fmha_fwd_fp8_new(&kb, spec, arch);
    if(kernel == NULL)
    {
        ckc_ir_builder_t* b = ckc_fmha_kernel_builder_builder(&kb);
        st = b != NULL ? ckc_ir_builder_status(b) : CKC_ERR_VALUE;
        if(err != NULL && err_cap > 0)
        {
            const char* m = b != NULL ? ckc_ir_builder_error(b) : NULL;
            snprintf(err, err_cap, "%s", m != NULL ? m : "build_fmha_fwd_fp8 failed");
        }
        ckc_fmha_kernel_builder_free(&kb);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_fmha_kernel_builder_free(&kb);
    return st;
}
