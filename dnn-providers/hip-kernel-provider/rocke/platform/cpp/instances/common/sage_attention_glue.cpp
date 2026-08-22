// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_sage_attention_sage_attention_glue.c -- PUBLIC entry + dispatch glue
 * for the C99 chunked port of rocke/instances/common/sage_attention.py.
 *
 * SCOPE (this TU only): every symbol declared in instance_sage_attention.h that
 * is NOT a phase body, plus the shared ABI primitives both physical builders and
 * the signature probe consume:
 *
 *   - rocke_sage_quant_mode_name / _is_mfma / _is_codebook         (lines 98-101)
 *   - rocke_sage_attention_spec_default                            (dataclass defaults)
 *   - rocke_sage_attention_kernel_name        (kernel_name -> kernel_name_join, 139-150)
 *   - rocke_sage_attention_mfma_dimensions_ok (_mfma_dimensions_ok, 153-181)
 *   - rocke_sage_attention_uses_mfma_path     (_uses_mfma_path, 184-190)
 *   - rocke_sage_attention_is_valid_spec      (is_valid_spec, 193-228)
 *   - rocke_build_sage_attention / _new       (build_sage_attention, 820-838)
 *   - rocke_sage_attention_grid               (sage_attention_grid, 841-855)
 *   - rocke_sage_attention_signature          (sage_attention_signature, 858-861)
 *   - rocke_sage_attention_lower_to_llvm      (convenience build -> lower .ll)
 *   - rocke_sage_declare_params               (_declare_params, 248-264)
 *   - rocke_sage_kv_pointee_for_quant_mode    (_kv_pointee_for_quant_mode, 231-236)
 *   - rocke_sage_kv_dtype_str                 (_kv_dtype_str, 239-245)
 *
 * The public build entry reproduces build_sage_attention's control flow op-for-op:
 *   ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError
 *   if _uses_mfma_path(spec): return _build_sage_mfma(spec, arch)
 *   return _build_sage_warp(spec)
 * where each Python builder body is decomposed into the prologue/fold/emit (MFMA)
 * or prologue/stage/preload/emit (warp) phase functions declared in
 * instance_sage_attention_internal.h, called here in Python order.
 *
 * This TU binds peers via the internal header only and edits no headers. The
 * phase bodies + the shared codebook / lane-load primitives are peer TUs.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_sage_attention.h"
#include "rocke/instance_sage_attention_internal.h"

#include "rocke/arch_target.h" /* rocke_arch_target_t      */
#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_arch_target_from_gfx */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_io_ir_type          */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* ROCKE_MFMA_ATTN_BLOCK_M/K */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join    */
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/helper_rocke.instances.common._fmha_warp_body.h" /* WARP_SIZE     */
#include "rocke/helper_rocke.instances.common.fmha_arch.h" /* validate_fmha_mfma_atom */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  SageQuantMode helpers
 *  Python: SageQuantMode = Literal["fp16_bf16","fp8_bf16","i8_fp8_bf16",
 *                                  "i4_fp8_bf16"]
 *          _MFMA_QUANT_MODES     = ("fp16_bf16", "fp8_bf16")
 *          _CODEBOOK_QUANT_MODES = ("i8_fp8_bf16", "i4_fp8_bf16")
 * ===================================================================== */
const char* rocke_sage_quant_mode_name(rocke_sage_quant_mode_t m)
{
    switch(m)
    {
    case ROCKE_SAGE_QUANT_FP16_BF16:
        return "fp16_bf16";
    case ROCKE_SAGE_QUANT_FP8_BF16:
        return "fp8_bf16";
    case ROCKE_SAGE_QUANT_I8_FP8_BF16:
        return "i8_fp8_bf16";
    case ROCKE_SAGE_QUANT_I4_FP8_BF16:
        return "i4_fp8_bf16";
    default:
        return NULL;
    }
}

bool rocke_sage_quant_mode_is_mfma(rocke_sage_quant_mode_t m)
{
    return m == ROCKE_SAGE_QUANT_FP16_BF16 || m == ROCKE_SAGE_QUANT_FP8_BF16;
}

bool rocke_sage_quant_mode_is_codebook(rocke_sage_quant_mode_t m)
{
    return m == ROCKE_SAGE_QUANT_I8_FP8_BF16 || m == ROCKE_SAGE_QUANT_I4_FP8_BF16;
}

/* ===================================================================== *
 *  SageAttentionSpec defaults
 *  Python dataclass defaults: name="rocke_sage_attention",
 *                             use_outer_scale_loop=False.
 *  The caller still fills common / quant_mode / q_scale / k_scale /
 *  seqlen_q / seqlen_k.
 * ===================================================================== */
rocke_sage_attention_spec_t rocke_sage_attention_spec_default(void)
{
    rocke_sage_attention_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "rocke_sage_attention";
    s.use_outer_scale_loop = false;
    return s;
}

/* Common-spec dtype spelling (the Python self.common.dtype string). */
static const char* sage_common_dtype(const rocke_sage_attention_spec_t* spec)
{
    return spec->common.dtype != NULL ? spec->common.dtype : "f16";
}

/* ===================================================================== *
 *  kernel_name -> kernel_name_join (lines 139-150)
 *
 *    s = self.common.shape
 *    return kernel_name_join(self.name, self.quant_mode, f"H{s.head_size}",
 *        f"HQ{s.num_query_heads}", f"HK{s.num_kv_heads}", self.common.dtype,
 *        f"Q{self.seqlen_q}", f"K{self.seqlen_k}")
 * ===================================================================== */
rocke_status_t rocke_sage_attention_kernel_name(const rocke_sage_attention_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char hbuf[32];
    char hqbuf[32];
    char hkbuf[32];
    char qbuf[32];
    char kbuf[32];
    const char* parts[7];
    const char* name;
    const char* mode;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    name = spec->name != NULL ? spec->name : "rocke_sage_attention";
    mode = rocke_sage_quant_mode_name(spec->quant_mode);
    if(mode == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(hbuf, sizeof(hbuf), "H%d", spec->common.shape.head_size);
    snprintf(hqbuf, sizeof(hqbuf), "HQ%d", spec->common.shape.num_query_heads);
    snprintf(hkbuf, sizeof(hkbuf), "HK%d", spec->common.shape.num_kv_heads);
    snprintf(qbuf, sizeof(qbuf), "Q%d", spec->seqlen_q);
    snprintf(kbuf, sizeof(kbuf), "K%d", spec->seqlen_k);

    parts[0] = mode;
    parts[1] = hbuf;
    parts[2] = hqbuf;
    parts[3] = hkbuf;
    parts[4] = sage_common_dtype(spec);
    parts[5] = qbuf;
    parts[6] = kbuf;

    return rocke_kernel_name_join(name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  _mfma_dimensions_ok (lines 153-181)
 * ===================================================================== */
bool rocke_sage_attention_mfma_dimensions_ok(const rocke_sage_attention_spec_t* spec)
{
    if(spec == NULL)
    {
        return false;
    }
    if(spec->seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        return false;
    }
    if(spec->seqlen_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        return false;
    }
    if(spec->common.shape.head_size % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        return false;
    }
    if(spec->q_scale.layout == ROCKE_QK_SCALE_PER_BLOCK
       && spec->q_scale.scale_block % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        return false;
    }
    if(spec->k_scale.layout == ROCKE_QK_SCALE_PER_BLOCK
       && spec->k_scale.scale_block % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        return false;
    }
    return true;
}

/* ===================================================================== *
 *  _uses_mfma_path (lines 184-190)
 *    spec.quant_mode in _MFMA_QUANT_MODES and _mfma_dimensions_ok(spec)
 * ===================================================================== */
bool rocke_sage_attention_uses_mfma_path(const rocke_sage_attention_spec_t* spec)
{
    if(spec == NULL)
    {
        return false;
    }
    return rocke_sage_quant_mode_is_mfma(spec->quant_mode)
           && rocke_sage_attention_mfma_dimensions_ok(spec);
}

/* ===================================================================== *
 *  is_valid_spec (lines 193-228)
 * ===================================================================== */
static void sage_set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != NULL && cap > 0)
    {
        snprintf(reason, cap, "%s", msg);
    }
}

bool rocke_sage_attention_is_valid_spec(const rocke_sage_attention_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_fmha_common_spec_t* cs;
    const char* common_reason = NULL;
    char atom_reason[256];
    rocke_arena_t arena;

    if(spec == NULL)
    {
        sage_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    cs = &spec->common;

    /* ok, why = validate_common_spec(spec.common) */
    rocke_arena_init(&arena, 0);
    if(!rocke_fmha_validate_common_spec(&arena, cs, &common_reason))
    {
        sage_set_reason(
            reason, reason_cap, common_reason != NULL ? common_reason : "invalid common spec");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch) */
    atom_reason[0] = '\0';
    if(!rocke_validate_fmha_mfma_atom(
           sage_common_dtype(spec), arch, atom_reason, sizeof(atom_reason)))
    {
        sage_set_reason(
            reason, reason_cap, atom_reason[0] != '\0' ? atom_reason : "invalid MFMA atom");
        return false;
    }

    /* if spec.quant_mode not in (the four literals): unknown quant_mode */
    if(rocke_sage_quant_mode_name(spec->quant_mode) == NULL)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason, reason_cap, "unknown quant_mode %d", (int)spec->quant_mode);
        }
        return false;
    }

    /* if spec.seqlen_q <= 0 or spec.seqlen_k <= 0 */
    if(spec->seqlen_q <= 0 || spec->seqlen_k <= 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "seqlen_q / seqlen_k must be > 0 (got %d, %d)",
                     spec->seqlen_q,
                     spec->seqlen_k);
        }
        return false;
    }

    /* if quant_mode == "i4_fp8_bf16" and (head_size % 2) != 0 */
    if(spec->quant_mode == ROCKE_SAGE_QUANT_I4_FP8_BF16 && (cs->shape.head_size % 2) != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "i4 sage requires head_size divisible by 2 (got %d)",
                     cs->shape.head_size);
        }
        return false;
    }

    /* if head_size % WARP_SIZE != 0 (the warp body's universal constraint) */
    if(cs->shape.head_size % ROCKE_FMHA_WARP_SIZE != 0)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "sage warp body needs head_size %% %d == 0 (got %d)",
                     ROCKE_FMHA_WARP_SIZE,
                     cs->shape.head_size);
        }
        return false;
    }

    sage_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  SHARED ABI: _kv_pointee_for_quant_mode (lines 231-236)
 *
 *    if quant_mode == "fp16_bf16": return io_ir_type(dtype)
 *    if quant_mode == "fp8_bf16":  return FP8E4M3
 *    return I8
 * ===================================================================== */
const rocke_type_t* rocke_sage_kv_pointee_for_quant_mode(rocke_sage_quant_mode_t quant_mode,
                                                         const char* dtype)
{
    if(quant_mode == ROCKE_SAGE_QUANT_FP16_BF16)
    {
        return rocke_io_ir_type(dtype);
    }
    if(quant_mode == ROCKE_SAGE_QUANT_FP8_BF16)
    {
        return rocke_fp8e4m3();
    }
    return rocke_i8();
}

/* ===================================================================== *
 *  SHARED ABI: _kv_dtype_str (lines 239-245)
 *
 *    if quant_mode == "fp16_bf16": return dtype
 *    if quant_mode == "fp8_bf16":  return "fp8e4m3"
 *    return "i8"
 * ===================================================================== */
const char* rocke_sage_kv_dtype_str(rocke_sage_quant_mode_t quant_mode, const char* dtype)
{
    if(quant_mode == ROCKE_SAGE_QUANT_FP16_BF16)
    {
        return dtype;
    }
    if(quant_mode == ROCKE_SAGE_QUANT_FP8_BF16)
    {
        return "fp8e4m3";
    }
    return "i8";
}

/* ===================================================================== *
 *  SHARED ABI: _declare_params (lines 248-264)
 *
 *    kv_dtype = _kv_dtype_str(spec.quant_mode, kb.common.dtype)
 *    kb.add_tensor("Q", readonly=True)
 *    kb.add_tensor("K", dtype=kv_dtype, readonly=True, align=8)
 *    kb.add_tensor("V", dtype=kv_dtype, readonly=True, align=8)
 *    kb.add_tensor("O", readonly=False, writeonly=True)
 *    kb.add_ptr("q_scale", dtype="f32", readonly=True)
 *    kb.add_ptr("k_scale", dtype="f32", readonly=True)
 *    if spec.quant_mode in _CODEBOOK_QUANT_MODES:
 *        kb.add_ptr("codebook_k", dtype="f32", readonly=True)
 *        kb.add_ptr("codebook_v", dtype="f32", readonly=True)
 *    kb.add_scalar("scale_log2", "f32")
 *    kb.add_scalar("seqlen_q", "i32")
 *    kb.add_scalar("seqlen_k", "i32")
 *    kb.add_strides("q", "k", "v", "o")
 * ===================================================================== */
void rocke_sage_declare_params(rocke_fmha_kernel_builder_t* kb,
                               const rocke_sage_attention_spec_t* spec)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};
    const char* kv_dtype;

    if(kb == NULL || spec == NULL)
    {
        return;
    }
    /* kb.common.dtype (the FmhaKernelBuilder's copied common spec). */
    kv_dtype = rocke_sage_kv_dtype_str(spec->quant_mode,
                                       kb->common.dtype != NULL ? kb->common.dtype : "f16");

    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "Q",
                                         NULL,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 16);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "K",
                                         kv_dtype,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 8);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "V",
                                         kv_dtype,
                                         /*readonly*/ true,
                                         /*writeonly*/ false,
                                         /*align*/ 8);
    rocke_fmha_kernel_builder_add_tensor(kb,
                                         "O",
                                         NULL,
                                         /*readonly*/ false,
                                         /*writeonly*/ true,
                                         /*align*/ 16);
    rocke_fmha_kernel_builder_add_ptr(kb, "q_scale", "f32", /*readonly*/ true, /*align*/ 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "k_scale", "f32", /*readonly*/ true, /*align*/ 4);
    if(rocke_sage_quant_mode_is_codebook(spec->quant_mode))
    {
        rocke_fmha_kernel_builder_add_ptr(kb,
                                          "codebook_k",
                                          "f32",
                                          /*readonly*/ true,
                                          /*align*/ 4);
        rocke_fmha_kernel_builder_add_ptr(kb,
                                          "codebook_v",
                                          "f32",
                                          /*readonly*/ true,
                                          /*align*/ 4);
    }
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    rocke_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ===================================================================== *
 *  build_sage_attention (lines 820-838)
 *
 *    ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...)
 *    if _uses_mfma_path(spec): return _build_sage_mfma(spec, arch)
 *    return _build_sage_warp(spec)
 *
 *  In C the two physical builders are decomposed into the phase functions
 *  declared in the internal header; this driver runs them in Python order on
 *  the caller-supplied (already initialised) FmhaKernelBuilder.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_sage_attention(rocke_fmha_kernel_builder_t* kb,
                                               const rocke_sage_attention_spec_t* spec,
                                               const char* arch)
{
    rocke_ir_builder_t* b;
    char reason[512];

    if(kb == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if(!rocke_sage_attention_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        b = rocke_fmha_kernel_builder_builder(kb);
        if(b != NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid sage_attention spec: %s", reason);
        }
        return NULL;
    }

    /* if _uses_mfma_path(spec): return _build_sage_mfma(spec, arch) */
    if(rocke_sage_attention_uses_mfma_path(spec))
    {
        rocke_sage_mfma_ctx_t ctx;

        memset(&ctx, 0, sizeof(ctx));
        ctx.kb = kb;
        ctx.spec = spec;
        ctx.arch = arch;
        ctx.s = spec->common;

        /* _build_sage_mfma: prologue -> Q-scale preload + scale fold -> body. */
        rocke_sage_mfma_prologue(&ctx);
        rocke_sage_mfma_fold_scales(&ctx);
        return rocke_sage_mfma_emit_body(&ctx);
    }

    /* return _build_sage_warp(spec) */
    {
        rocke_sage_warp_ctx_t ctx;

        memset(&ctx, 0, sizeof(ctx));
        ctx.kb = kb;
        ctx.spec = spec;
        ctx.s = spec->common;

        /* _build_sage_warp: prologue (i4 geometry guard) -> codebook staging ->
         * Q-scale preload -> body. The prologue returns false on the i4 geometry
         * ValueError, with the builder sticky error already set. */
        if(!rocke_sage_warp_prologue(&ctx))
        {
            return NULL;
        }
        rocke_sage_warp_stage_codebooks(&ctx);
        rocke_sage_warp_preload_q_scale(&ctx);
        return rocke_sage_warp_emit_body(&ctx);
    }
}

rocke_kernel_def_t* rocke_build_sage_attention_new(rocke_fmha_kernel_builder_t* kb,
                                                   const rocke_sage_attention_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_sage_attention_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_fmha_kernel_builder_init(kb, name, &spec->common) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_sage_attention(kb, spec, arch);
    });
}

/* ===================================================================== *
 *  sage_attention_grid (lines 841-855)
 *
 *    if _uses_mfma_path(spec):
 *        return (seqlen_q // BLOCK_M, num_query_heads, 1)
 *    return (seqlen_q, num_query_heads, 1)
 * ===================================================================== */
rocke_status_t rocke_sage_attention_grid(const rocke_sage_attention_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_sage_attention_uses_mfma_path(spec))
    {
        out[0] = spec->seqlen_q / ROCKE_MFMA_ATTN_BLOCK_M;
    }
    else
    {
        out[0] = spec->seqlen_q;
    }
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  sage_attention_signature (lines 858-861)
 *
 *    kb = FmhaKernelBuilder("rocke_sage_attention_sig_probe", spec.common)
 *    _declare_params(kb, spec)
 *    return kb.signature()
 * ===================================================================== */
rocke_status_t rocke_sage_attention_signature(const rocke_sage_attention_spec_t* spec,
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

    st = rocke_fmha_kernel_builder_init(&kb, "rocke_sage_attention_sig_probe", &spec->common);
    if(st != ROCKE_OK)
    {
        return st;
    }

    rocke_sage_declare_params(&kb, spec);

    st = rocke_fmha_kernel_builder_signature(&kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}

/* ===================================================================== *
 *  convenience lower-to-.ll
 *
 *  Build (via the same dispatch as rocke_build_sage_attention) then lower to
 *  LLVM .ll text. Owns + frees its FmhaKernelBuilder.
 * ===================================================================== */
rocke_status_t rocke_sage_attention_lower_to_llvm(const rocke_sage_attention_spec_t* spec,
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

    kernel = rocke_build_sage_attention_new(&kb, spec, arch);
    if(kernel == NULL)
    {
        rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(&kb);
        st = b != NULL ? rocke_ir_builder_status(b) : ROCKE_ERR_VALUE;
        if(err != NULL && err_cap > 0)
        {
            const char* m = b != NULL ? rocke_ir_builder_error(b) : NULL;
            snprintf(err, err_cap, "%s", m != NULL ? m : "build_sage_attention failed");
        }
        rocke_fmha_kernel_builder_free(&kb);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}
