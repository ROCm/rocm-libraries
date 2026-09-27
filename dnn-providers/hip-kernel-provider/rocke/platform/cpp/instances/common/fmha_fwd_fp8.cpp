// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_fmha_fwd_fp8.c -- C99 port of rocke/instances/common/fmha_fwd_fp8.py.
 *
 * Byte-identical builder-call sequence vs the Python build_fmha_fwd_fp8: same
 * param declaration order, same grid decode, same waves_per_eu attr, same
 * fmul/mul/const op order, and the same single call into the already-ported
 * helper rocke_mfma_attention_fwd_inner_body. See the header for the symbol map.
 */
#include "rocke/instance_fmha_fwd_fp8.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h" /* rocke_arch_target_t, has_shape */
#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx      */
#include "rocke/helper_rocke.helpers.attention.h" /* rocke_attn_mask_mode_t         */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* ROCKE_MFMA_ATTN_BLOCK_M, body  */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join         */
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

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
const char* rocke_kv_fp8_dtype_name(rocke_kv_fp8_dtype_t d)
{
    switch(d)
    {
    case ROCKE_KV_FP8_E4M3:
        return "fp8e4m3";
    case ROCKE_KV_BF8_E5M2:
        return "bf8e5m2";
    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ *
 * FmhaFwdFp8Spec defaults + kernel_name
 * ------------------------------------------------------------------ */
rocke_fmha_fwd_fp8_spec_t rocke_fmha_fwd_fp8_spec_default(void)
{
    rocke_fmha_fwd_fp8_spec_t s;
    memset(&s, 0, sizeof(s));
    /* common is left zero-initialised; the caller must fill it. */
    s.kv_dtype = ROCKE_KV_FP8_E4M3;
    s.seqlen_q = 1;
    s.seqlen_k = 0;
    s.fp8_fnuz = false;
    s.has_waves_per_eu = true; /* Python default Optional[int] = 4 */
    s.waves_per_eu = 4;
    s.name = "rocke_fmha_fwd_fp8";
    return s;
}

/* Common-spec dtype spelling (the Python self.common.dtype string). */
static const char* fp8_fwd_common_dtype(const rocke_fmha_fwd_fp8_spec_t* spec)
{
    return spec->common.dtype != NULL ? spec->common.dtype : "f16";
}

rocke_status_t
    rocke_fmha_fwd_fp8_kernel_name(const rocke_fmha_fwd_fp8_spec_t* spec, char* out, size_t out_cap)
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
        return ROCKE_ERR_VALUE;
    }
    name = spec->name != NULL ? spec->name : "rocke_fmha_fwd_fp8";
    mask = rocke_fmha_mask_mode_name(spec->common.mask_mode);
    kvname = rocke_kv_fp8_dtype_name(spec->kv_dtype);
    if(mask == NULL || kvname == NULL)
    {
        return ROCKE_ERR_VALUE;
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

    return rocke_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
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

bool rocke_fmha_fwd_fp8_is_valid_spec(const rocke_fmha_fwd_fp8_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const rocke_fmha_common_spec_t* cs;
    const char* common_reason = NULL;
    const char* kvname;
    rocke_arena_t arena;
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
    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown arch %s", arch);
        }
        return false;
    }

    /* ok, why = validate_common_spec(spec.common) */
    rocke_arena_init(&arena, 0);
    if(!rocke_fmha_validate_common_spec(&arena, cs, &common_reason))
    {
        fp8_fwd_set_reason(
            reason, reason_cap, common_reason != NULL ? common_reason : "invalid common spec");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* kv_dtype must be 'fp8e4m3' or 'bf8e5m2' */
    kvname = rocke_kv_fp8_dtype_name(spec->kv_dtype);
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
    if(spec->seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "MFMA fp8 attention needs seqlen_q (%d) to be a multiple of "
                     "BLOCK_M (%d)",
                     spec->seqlen_q,
                     ROCKE_MFMA_ATTN_BLOCK_M);
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
    if(!rocke_arch_supports_dtype_combo(target, "f16", "f16", "fp32", NULL))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unsupported f16 MFMA dtype combo on %s", arch);
        }
        return false;
    }
    if(!rocke_mma_catalog_has_shape(&target->mma,
                                    NULL,
                                    "f16",
                                    "f16",
                                    "fp32",
                                    ROCKE_MFMA_ATTN_BLOCK_M,
                                    ROCKE_MFMA_ATTN_BLOCK_M,
                                    ROCKE_MFMA_ATTN_BLOCK_M))
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "unsupported f16 warp_tile (%d,%d,%d) on %s",
                     ROCKE_MFMA_ATTN_BLOCK_M,
                     ROCKE_MFMA_ATTN_BLOCK_M,
                     ROCKE_MFMA_ATTN_BLOCK_M,
                     arch);
        }
        return false;
    }

    /* LDS budget: one BLOCK_M x BLOCK_M f16 P-staging buffer. */
    bytes_lds = (long)ROCKE_MFMA_ATTN_BLOCK_M * ROCKE_MFMA_ATTN_BLOCK_M * 2;
    if(!rocke_arch_fits_lds(target, bytes_lds))
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
static void fp8_fwd_declare_params(rocke_fmha_kernel_builder_t* kb,
                                   const rocke_fmha_fwd_fp8_spec_t* spec)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};
    const char* kv = rocke_kv_fp8_dtype_name(spec->kv_dtype);

    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "Q",
                                         NULL,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 16);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "K",
                                         kv,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 8);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "V",
                                         kv,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 8);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "O",
                                         NULL,
                                         /*readonly*/ false,
                                         /*writeonly*/ true,
                                         /*align*/ 16);
    rocke_fmha_kernel_builder_add_scalar(kb, "k_scale", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "v_scale", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    rocke_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ------------------------------------------------------------------ *
 * Map the FmhaCommonSpec mask mode -> the helper's attention mask enum.
 * (The helper only knows none / causal / sliding_window; alibi / custom
 * never reach this fp8 instance because validate_common_spec gates them.)
 * ------------------------------------------------------------------ */
static rocke_attn_mask_mode_t fp8_fwd_attn_mask(rocke_fmha_mask_mode_t m)
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

/* ------------------------------------------------------------------ *
 * build_fmha_fwd_fp8
 * ------------------------------------------------------------------ */
rocke_kernel_def_t* rocke_build_fmha_fwd_fp8(rocke_fmha_kernel_builder_t* kb,
                                             const rocke_fmha_fwd_fp8_spec_t* spec,
                                             const char* arch)
{
    const rocke_fmha_common_spec_t* s;
    rocke_ir_builder_t* b;
    rocke_kernel_def_t* kernel;
    rocke_value_t* Q;
    rocke_value_t* K;
    rocke_value_t* V;
    rocke_value_t* O;
    rocke_value_t* k_scale;
    rocke_value_t* v_scale;
    rocke_value_t* scale_log2_raw;
    rocke_value_t* scale_log2;
    rocke_value_t* seqlen_k;
    rocke_value_t* q_tile_idx;
    rocke_value_t* head_idx;
    rocke_value_t* kv_head_idx;
    rocke_value_t* q_tile_base;
    rocke_value_t* causal_ctx;
    char reason[512];
    rocke_mfma_attn_params_t p;

    if(kb == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!rocke_fmha_fwd_fp8_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        b = rocke_fmha_kernel_builder_builder(kb);
        if(b != NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fmha_fwd_fp8 spec: %s", reason);
        }
        return NULL;
    }
    s = &spec->common;
    b = rocke_fmha_kernel_builder_builder(kb);

    /* kb.block_size(64) */
    rocke_fmha_kernel_builder_block_size(kb, 64);

    /* _declare_params(kb, spec) */
    fp8_fwd_declare_params(kb, spec);

    /* kb.decode_grid() */
    rocke_fmha_kernel_builder_decode_grid(kb,
                                          /*num_queries_per_kv=None*/ -1,
                                          /*has_batch_axis*/ false,
                                          NULL,
                                          NULL,
                                          NULL);

    /* Occupancy hint: b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu */
    if(spec->has_waves_per_eu)
    {
        kernel = rocke_fmha_kernel_builder_kernel(kb);
        if(kernel != NULL)
        {
            rocke_attr_set_int(b, &kernel->attrs, "waves_per_eu", (int64_t)spec->waves_per_eu);
        }
    }

    Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
    K = rocke_fmha_kernel_builder_tensor(kb, "K");
    V = rocke_fmha_kernel_builder_tensor(kb, "V");
    O = rocke_fmha_kernel_builder_tensor(kb, "O");

    k_scale = rocke_fmha_kernel_builder_scalar(kb, "k_scale");
    v_scale = rocke_fmha_kernel_builder_scalar(kb, "v_scale");
    scale_log2_raw = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
    /* scale_log2 = b.fmul(scale_log2_raw, k_scale) */
    scale_log2 = rocke_b_fmul(b, scale_log2_raw, k_scale);
    seqlen_k = rocke_fmha_kernel_builder_scalar(kb, "seqlen_k");

    q_tile_idx = kb->q_token;
    head_idx = kb->head_idx;
    kv_head_idx = kb->kv_head_idx;
    /* q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)) */
    q_tile_base = rocke_b_mul(b, q_tile_idx, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M));

    /* causal_ctx = b.const_i32(0) if mask in (causal, sliding_window) else None */
    if(s->mask_mode == ROCKE_FMHA_MASK_CAUSAL || s->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW)
    {
        causal_ctx = rocke_b_const_i32(b, 0);
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

    p.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
    p.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
    p.stride_k_token = rocke_fmha_kernel_builder_stride_token(kb, "k");
    p.stride_k_head = rocke_fmha_kernel_builder_stride_head(kb, "k");
    p.stride_v_token = rocke_fmha_kernel_builder_stride_token(kb, "v");
    p.stride_v_head = rocke_fmha_kernel_builder_stride_head(kb, "v");
    p.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
    p.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");

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
    p.kv_dtype = rocke_kv_fp8_dtype_name(spec->kv_dtype);
    p.v_scale = v_scale;
    p.use_wider_atom = false;
    p.native_fp8_path = false;
    p.use_async_kv = false;
    p.codebook_ptr = NULL;
    p.wmma_v_lds_stage = false;
    p.arch = arch;

    rocke_mfma_attention_fwd_inner_body(b, &p);

    /* b.ret() */
    rocke_b_ret(b);

    if(rocke_ir_builder_status(b) != ROCKE_OK)
    {
        return NULL;
    }
    return rocke_fmha_kernel_builder_kernel(kb);
}

rocke_kernel_def_t* rocke_build_fmha_fwd_fp8_new(rocke_fmha_kernel_builder_t* kb,
                                                 const rocke_fmha_fwd_fp8_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_fmha_fwd_fp8_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_fmha_kernel_builder_init(kb, name, &spec->common) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_fmha_fwd_fp8(kb, spec, arch);
    });
}

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_grid
 * ------------------------------------------------------------------ */
rocke_status_t rocke_fmha_fwd_fp8_grid(const rocke_fmha_fwd_fp8_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = spec->seqlen_q / ROCKE_MFMA_ATTN_BLOCK_M;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * fmha_fwd_fp8_signature
 * ------------------------------------------------------------------ */
rocke_status_t rocke_fmha_fwd_fp8_signature(const rocke_fmha_fwd_fp8_spec_t* spec,
                                            rocke_arena_t* arena,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_fmha_kernel_builder_init(&kb, "rocke_fmha_fwd_fp8_sig_probe", &spec->common);
    if(st != ROCKE_OK)
    {
        return st;
    }

    fp8_fwd_declare_params(&kb, spec);

    st = rocke_fmha_kernel_builder_signature(&kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}

/* ------------------------------------------------------------------ *
 * convenience lower-to-.ll
 * ------------------------------------------------------------------ */
rocke_status_t rocke_fmha_fwd_fp8_lower_to_llvm(const rocke_fmha_fwd_fp8_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

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
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_fmha_fwd_fp8_new(&kb, spec, arch);
    if(kernel == NULL)
    {
        rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(&kb);
        st = b != NULL ? rocke_ir_builder_status(b) : ROCKE_ERR_VALUE;
        if(err != NULL && err_cap > 0)
        {
            const char* m = b != NULL ? rocke_ir_builder_error(b) : NULL;
            snprintf(err, err_cap, "%s", m != NULL ? m : "build_fmha_fwd_fp8 failed");
        }
        rocke_fmha_kernel_builder_free(&kb);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}
