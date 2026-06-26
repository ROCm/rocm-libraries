// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_block_scaled_gemm.c -- C99 port of
 * rocke/instances/gfx1250/block_scaled_gemm.py.
 *
 * gfx1250 K=64 FP8/BF8 block-scaled dense GEMM (RCR, C = A @ B^T). One wave (32
 * lanes) computes one 16x16 output tile, no LDS; the K loop runs in block_k
 * groups, each accumulating block_k/64 WMMA K=64 steps into a fresh <8 x f32>
 * acc then applying per-block scales. The build op order tracks
 * build_block_scaled_gemm() top-to-bottom; emitted IR is byte-identical to the
 * Python lowerer (args sequenced left-to-right, const_i32 never deduped).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_block_scaled_gemm.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_BSG_BLOCK_M 16
#define ROCKE_BSG_BLOCK_N 16
#define ROCKE_BSG_WMMA_K 64
#define ROCKE_BSG_WAVE 32
#define ROCKE_BSG_HALF_K 32 /* K-elements per lane-half for the K=64 atom */
#define ROCKE_BSG_ACC 8 /* accumulator slots per lane (<8 x f32>) */

/* _canon_lowbit: fp8/fp8e4m3 -> "fp8"; bf8/bf8e5m2 -> "bf8"; else NULL. */
static const char* rocke_bsg_canon_lowbit(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "fp8") == 0 || strcmp(dtype, "fp8e4m3") == 0)
    {
        return "fp8";
    }
    if(strcmp(dtype, "bf8") == 0 || strcmp(dtype, "bf8e5m2") == 0)
    {
        return "bf8";
    }
    return NULL;
}

/* _wire_scale_dtype: fp16/f16 -> "f16"; fp32/f32 -> "f32"; else NULL. */
static const char* rocke_bsg_wire_scale_dtype(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
    {
        return "f16";
    }
    if(strcmp(dtype, "fp32") == 0 || strcmp(dtype, "f32") == 0)
    {
        return "f32";
    }
    return NULL;
}

/* _storage_type: fp16/f16 -> f16; bf16 -> bf16; else quant_ir_type(dtype). */
static const rocke_type_t* rocke_bsg_storage_type(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    return rocke_quant_ir_type(dtype);
}

/* _scale_type: f16 if wire==f16 else f32. */
static const rocke_type_t* rocke_bsg_scale_type(const char* dtype)
{
    const char* wire = rocke_bsg_wire_scale_dtype(dtype);
    if(wire != NULL && strcmp(wire, "f16") == 0)
    {
        return rocke_f16();
    }
    return rocke_f32();
}

const char* rocke_block_scaled_gemm_gfx1250_resolved_path(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec)
{
    if(spec == NULL || spec->matrix_path == NULL)
    {
        return "wmma";
    }
    if(strcmp(spec->matrix_path, "auto") == 0 || strcmp(spec->matrix_path, "wmma_scaffold") == 0)
    {
        return "wmma";
    }
    return spec->matrix_path;
}

/* _wmma_op_id: wmma_gfx1250_f32_16x16x64_{a}_{b}. Writes into out (cap out_cap). */
static void rocke_bsg_wmma_op_id(const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
                                 char* out,
                                 size_t out_cap)
{
    snprintf(out,
             out_cap,
             "wmma_gfx1250_f32_16x16x64_%s_%s",
             rocke_bsg_canon_lowbit(spec->dtype_a),
             rocke_bsg_canon_lowbit(spec->dtype_b));
}

rocke_block_scaled_gemm_gfx1250_spec_t rocke_block_scaled_gemm_gfx1250_spec_default(void)
{
    rocke_block_scaled_gemm_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL; /* required */
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.dtype_a = "fp8";
    s.dtype_b = "fp8";
    s.dtype_c = "bf16";
    s.dtype_acc = "fp32";
    s.scale_dtype = "fp32";
    s.block_k = 128;
    s.layout = "RCR";
    s.matrix_path = "auto";
    s.tile_m = 16;
    s.tile_n = 16;
    s.tile_k = 128;
    return s;
}

int rocke_block_scaled_gemm_gfx1250_block_size(const rocke_block_scaled_gemm_gfx1250_spec_t* spec)
{
    (void)spec;
    return ROCKE_BSG_WAVE;
}

rocke_status_t rocke_block_scaled_gemm_gfx1250_kernel_name(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char ab_part[32];
    char mnk_part[64];
    char bk_part[32];
    char tile_part[48];
    const char* parts[5];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(ab_part,
             sizeof(ab_part),
             "%s_%s",
             rocke_bsg_canon_lowbit(spec->dtype_a),
             rocke_bsg_canon_lowbit(spec->dtype_b));
    snprintf(mnk_part, sizeof(mnk_part), "M%dN%dK%d", spec->M, spec->N, spec->K);
    snprintf(bk_part, sizeof(bk_part), "bk%d", spec->block_k);
    snprintf(tile_part, sizeof(tile_part), "t%dx%dx%d", spec->tile_m, spec->tile_n, spec->tile_k);

    parts[0] = "block_scaled";
    parts[1] = ab_part;
    parts[2] = mnk_part;
    parts[3] = bk_part;
    parts[4] = tile_part;
    flag_names[0] = "wmma";
    flag_on[0] = (strcmp(rocke_block_scaled_gemm_gfx1250_resolved_path(spec), "wmma") == 0) ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap, NULL);
}

static bool rocke_bsg_lowbit_ok(const char* dtype)
{
    return dtype != NULL
           && (strcmp(dtype, "fp8") == 0 || strcmp(dtype, "fp8e4m3") == 0
               || strcmp(dtype, "bf8") == 0 || strcmp(dtype, "bf8e5m2") == 0);
}

/* The C arch_target struct has no has_wmma/has_mfma fields (unlike Python, which
 * derives them as any(op.family=="wmma"/"mma")). Mirror that derivation here by
 * scanning the catalog. */
static bool rocke_bsg_catalog_has_family(const rocke_arch_target_t* t, const char* family)
{
    int n = 0;
    const rocke_mma_op_t* ops = rocke_mma_catalog_ops(&t->mma, &n);
    int i;
    for(i = 0; i < n; ++i)
    {
        if(ops[i].family != NULL && strcmp(ops[i].family, family) == 0)
        {
            return true;
        }
    }
    return false;
}

bool rocke_block_scaled_gemm_gfx1250_is_valid_spec(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const rocke_arch_target_t* target;
    const char* resolved;
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(strcmp(arch, "gfx1250") != 0)
    {
        snprintf(buf, sizeof(buf), "block_scaled_gemm scaffold is gfx1250-only (got '%s')", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* matrix_path must be one of auto/wmma/wmma_scaffold/mfma. */
    if(spec->matrix_path == NULL
       || (strcmp(spec->matrix_path, "auto") != 0 && strcmp(spec->matrix_path, "wmma") != 0
           && strcmp(spec->matrix_path, "wmma_scaffold") != 0
           && strcmp(spec->matrix_path, "mfma") != 0))
    {
        rocke_spec_set_reason(
            reason,
            reason_cap,
            "matrix_path must be one of ['auto', 'mfma', 'wmma', 'wmma_scaffold']");
        return false;
    }
    resolved = rocke_block_scaled_gemm_gfx1250_resolved_path(spec);
    if(strcmp(spec->matrix_path, "mfma") == 0 || strcmp(resolved, "mfma") == 0)
    {
        rocke_spec_set_reason(reason,
                              reason_cap,
                              "gfx1250 has no MFMA block_scale path; use matrix_path='wmma' (the "
                              "K=64 FP8/BF8 WMMA atom)");
        return false;
    }
    /* has_wmma = any(op.family=="wmma"); has_mfma = any(op.family=="mma"). */
    if(!rocke_bsg_catalog_has_family(target, "wmma"))
    {
        snprintf(buf, sizeof(buf), "%s does not expose WMMA for block-scaled GEMM", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(rocke_bsg_catalog_has_family(target, "mma"))
    {
        snprintf(buf, sizeof(buf), "%s unexpectedly exposes MFMA; expected WMMA-only", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->M <= 0 || spec->N <= 0 || spec->K <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "M/N/K must be positive (got M=%d, N=%d, K=%d)",
                 spec->M,
                 spec->N,
                 spec->K);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!rocke_bsg_lowbit_ok(spec->dtype_a) || !rocke_bsg_lowbit_ok(spec->dtype_b))
    {
        snprintf(buf,
                 sizeof(buf),
                 "A/B must be fp8 or bf8 (got A='%s', B='%s')",
                 spec->dtype_a ? spec->dtype_a : "None",
                 spec->dtype_b ? spec->dtype_b : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(rocke_archtarget_op_for_shape(target,
                                     "wmma",
                                     rocke_bsg_canon_lowbit(spec->dtype_a),
                                     rocke_bsg_canon_lowbit(spec->dtype_b),
                                     "fp32",
                                     ROCKE_BSG_BLOCK_M,
                                     ROCKE_BSG_BLOCK_N,
                                     ROCKE_BSG_WMMA_K)
       == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "no gfx1250 16x16x64 WMMA atom for %s/%s",
                 rocke_bsg_canon_lowbit(spec->dtype_a),
                 rocke_bsg_canon_lowbit(spec->dtype_b));
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->dtype_c == NULL
       || (strcmp(spec->dtype_c, "bf16") != 0 && strcmp(spec->dtype_c, "fp16") != 0
           && strcmp(spec->dtype_c, "f16") != 0))
    {
        snprintf(buf,
                 sizeof(buf),
                 "day-0 gfx1250 output must be bf16/fp16 (got '%s')",
                 spec->dtype_c ? spec->dtype_c : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->dtype_acc == NULL
       || (strcmp(spec->dtype_acc, "fp32") != 0 && strcmp(spec->dtype_acc, "f32") != 0))
    {
        snprintf(buf,
                 sizeof(buf),
                 "accumulator dtype must be fp32 (got '%s')",
                 spec->dtype_acc ? spec->dtype_acc : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->layout == NULL || strcmp(spec->layout, "RCR") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_scaled_gemm supports RCR only (got '%s')",
                 spec->layout ? spec->layout : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(rocke_bsg_wire_scale_dtype(spec->scale_dtype) == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "scale_dtype must be fp16/fp32, got '%s'",
                 spec->scale_dtype ? spec->scale_dtype : "None");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->block_k <= 0 || (spec->K % spec->block_k) != 0)
    {
        snprintf(
            buf, sizeof(buf), "K (%d) must be divisible by block_k (%d)", spec->K, spec->block_k);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if((spec->block_k % ROCKE_BSG_WMMA_K) != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_k (%d) must be a multiple of %d",
                 spec->block_k,
                 ROCKE_BSG_WMMA_K);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if((spec->K % ROCKE_BSG_WMMA_K) != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "K (%d) must be a multiple of the WMMA K=%d",
                 spec->K,
                 ROCKE_BSG_WMMA_K);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->tile_m != ROCKE_BSG_BLOCK_M || spec->tile_n != ROCKE_BSG_BLOCK_N)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "gfx1250 block_scaled_gemm uses fixed 16x16 output tiles");
        return false;
    }
    if((spec->M % ROCKE_BSG_BLOCK_M) != 0 || (spec->N % ROCKE_BSG_BLOCK_N) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "M and N must be multiples of 16");
        return false;
    }

    rocke_spec_set_reason(reason, reason_cap, "ok: gfx1250 K=64 FP8/BF8 WMMA block-scaled GEMM");
    return true;
}

/* _as_f32(ir, v): v if v is f32 else cast_to_f32(v). The caller passes a value
 * whose type is known to be scale_ty; we mirror by checking against f32. */
static rocke_value_t*
    rocke_bsg_as_f32(rocke_ir_builder_t* b, rocke_value_t* v, const rocke_type_t* scale_ty)
{
    if(scale_ty == rocke_f32())
    {
        return v;
    }
    return rocke_b_cast_to_f32(b, v);
}

rocke_kernel_def_t* rocke_build_block_scaled_gemm_gfx1250(
    rocke_ir_builder_t* b, const rocke_block_scaled_gemm_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* a_ty;
        const rocke_type_t* b_ty;
        const rocke_type_t* c_ty;
        const rocke_type_t* scale_ty;
        const rocke_type_t* a_frag_ty;
        char op_id[96];
        int groups;
        int steps_per_group;
        int kg;
        int step;
        int i;
        rocke_value_t* A;
        rocke_value_t* B;
        rocke_value_t* AScale;
        rocke_value_t* BScale;
        rocke_value_t* C;
        rocke_value_t* cK;
        rocke_value_t* cN;
        rocke_value_t* c16;
        rocke_value_t* c32;
        rocke_value_t* lane;
        rocke_value_t* frag;
        rocke_value_t* half;
        rocke_value_t* half_k;
        rocke_value_t* m0;
        rocke_value_t* n0;
        rocke_value_t* a_row;
        rocke_value_t* b_row;
        rocke_value_t* a_base;
        rocke_value_t* b_base;
        rocke_value_t* out_col;
        rocke_value_t* row_base;
        rocke_value_t** outer;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        if(!rocke_block_scaled_gemm_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            char msg[ROCKE_ERR_MSG_CAP];
            ROCKE_ERR_SNPRINTF(
                msg, sizeof(msg), "invalid block_scaled_gemm spec for %s: %s", arch, reason);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }

        a_ty = rocke_bsg_storage_type(spec->dtype_a);
        b_ty = rocke_bsg_storage_type(spec->dtype_b);
        c_ty = rocke_bsg_storage_type(spec->dtype_c);
        scale_ty = rocke_bsg_scale_type(spec->scale_dtype);
        rocke_bsg_wmma_op_id(spec, op_id, sizeof(op_id));
        a_frag_ty = rocke_vector_type(b, rocke_i32(), ROCKE_BSG_ACC);

        groups = spec->K / spec->block_k;
        steps_per_group = spec->block_k / ROCKE_BSG_WMMA_K;

        /* ir.kernel.attrs["max_workgroup_size"] = block_size */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ROCKE_BSG_WAVE);

        /* ---- kernel params -- */
        {
            rocke_param_opts_t ro16; /* noalias readonly align16 */
            rocke_param_opts_t ro4; /* noalias readonly align4  */
            rocke_param_opts_t wo16; /* noalias writeonly align16 */
            const rocke_type_t* ptr_a = rocke_ptr_type(b, a_ty, "global");
            const rocke_type_t* ptr_b = rocke_ptr_type(b, b_ty, "global");
            const rocke_type_t* ptr_sc = rocke_ptr_type(b, scale_ty, "global");
            const rocke_type_t* ptr_c = rocke_ptr_type(b, c_ty, "global");

            memset(&ro16, 0, sizeof(ro16));
            ro16.noalias = true;
            ro16.noalias_set = true;
            ro16.readonly = true;
            ro16.readonly_set = true;
            ro16.align = 16;
            ro16.align_set = true;

            memset(&ro4, 0, sizeof(ro4));
            ro4.noalias = true;
            ro4.noalias_set = true;
            ro4.readonly = true;
            ro4.readonly_set = true;
            ro4.align = 4;
            ro4.align_set = true;

            memset(&wo16, 0, sizeof(wo16));
            wo16.noalias = true;
            wo16.noalias_set = true;
            wo16.writeonly = true;
            wo16.writeonly_set = true;
            wo16.align = 16;
            wo16.align_set = true;

            A = rocke_b_param(b, "A", ptr_a, &ro16);
            B = rocke_b_param(b, "B", ptr_b, &ro16);
            AScale = rocke_b_param(b, "A_scale", ptr_sc, &ro4);
            BScale = rocke_b_param(b, "B_scale", ptr_sc, &ro4);
            C = rocke_b_param(b, "C", ptr_c, &wo16);
            (void)rocke_b_param(b, "M", rocke_i32(), NULL);
            (void)rocke_b_param(b, "N", rocke_i32(), NULL);
            (void)rocke_b_param(b, "K", rocke_i32(), NULL);
        }

        /* cK = const(K); cN = const(N); c16 = const(_BLOCK_M); c32 = const(_WAVE) */
        cK = rocke_b_const_i32(b, spec->K);
        cN = rocke_b_const_i32(b, spec->N);
        c16 = rocke_b_const_i32(b, ROCKE_BSG_BLOCK_M);
        c32 = rocke_b_const_i32(b, ROCKE_BSG_WAVE);

        /* lane = mod(thread_id_x(), c32); frag = mod(lane, c16); half = div(lane, c16) */
        lane = rocke_b_mod(b, rocke_b_thread_id_x(b), c32);
        frag = rocke_b_mod(b, lane, c16);
        half = rocke_b_div(b, lane, c16);
        /* half_k = mul(half, const(_HALF_K)) */
        half_k = rocke_b_mul(b, half, rocke_b_const_i32(b, ROCKE_BSG_HALF_K));

        /* m0 = mul(block_id_y(), c16); n0 = mul(block_id_x(), c16) */
        m0 = rocke_b_mul(b, rocke_b_block_id_y(b), c16);
        n0 = rocke_b_mul(b, rocke_b_block_id_x(b), c16);
        /* a_row = add(m0, frag); b_row = add(n0, frag) */
        a_row = rocke_b_add(b, m0, frag);
        b_row = rocke_b_add(b, n0, frag);
        /* a_base = mul(a_row, cK); b_base = mul(b_row, cK) */
        a_base = rocke_b_mul(b, a_row, cK);
        b_base = rocke_b_mul(b, b_row, cK);

        /* outer = [ir.const_f32(0.0) for _ in range(_ACC)] */
        outer = (rocke_value_t**)calloc((size_t)ROCKE_BSG_ACC, sizeof(rocke_value_t*));
        if(outer == NULL)
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }
        for(i = 0; i < ROCKE_BSG_ACC; ++i)
        {
            outer[i] = rocke_b_const_f32(b, 0.0);
        }

        for(kg = 0; kg < groups; ++kg)
        {
            rocke_value_t* acc = rocke_b_zero_vec_f32(b, ROCKE_BSG_ACC);
            rocke_value_t* b_scale_off;
            rocke_value_t* b_scale;

            for(step = 0; step < steps_per_group; ++step)
            {
                int k0 = kg * spec->block_k + step * ROCKE_BSG_WMMA_K;
                rocke_value_t* a_frag;
                rocke_value_t* b_frag;
                /* _load_frag(A, a_base, a_ty, k0) */
                {
                    rocke_value_t* off0
                        = rocke_b_add(b, rocke_b_add(b, a_base, rocke_b_const_i32(b, k0)), half_k);
                    rocke_value_t* off1 = rocke_b_add(b, off0, rocke_b_const_i32(b, 16));
                    rocke_value_t* lo = rocke_b_global_load_vN(b, A, off0, a_ty, 16, /*align=*/16);
                    rocke_value_t* hi = rocke_b_global_load_vN(b, A, off1, a_ty, 16, /*align=*/16);
                    a_frag = rocke_b_bitcast(b, rocke_b_vec_concat(b, lo, hi), a_frag_ty);
                }
                /* _load_frag(B, b_base, b_ty, k0) */
                {
                    rocke_value_t* off0
                        = rocke_b_add(b, rocke_b_add(b, b_base, rocke_b_const_i32(b, k0)), half_k);
                    rocke_value_t* off1 = rocke_b_add(b, off0, rocke_b_const_i32(b, 16));
                    rocke_value_t* lo = rocke_b_global_load_vN(b, B, off0, b_ty, 16, /*align=*/16);
                    rocke_value_t* hi = rocke_b_global_load_vN(b, B, off1, b_ty, 16, /*align=*/16);
                    b_frag = rocke_b_bitcast(b, rocke_b_vec_concat(b, lo, hi), a_frag_ty);
                }
                /* acc = ir.mma(op_id, a_frag, b_frag, acc) */
                acc = rocke_b_mma(b, op_id, a_frag, b_frag, acc, NULL, 0);
            }

            /* b_scale_off = add(mul(const(kg), cN), b_row) */
            b_scale_off = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, kg), cN), b_row);
            /* b_scale = _as_f32(ir.global_load(BScale, b_scale_off, scale_ty, align=4)) */
            b_scale = rocke_bsg_as_f32(
                b, rocke_b_global_load(b, BScale, b_scale_off, scale_ty, /*align=*/4), scale_ty);

            for(i = 0; i < ROCKE_BSG_ACC; ++i)
            {
                rocke_value_t* out_row;
                rocke_value_t* a_scale_off;
                rocke_value_t* a_scale;
                rocke_value_t* ab;
                rocke_value_t* prod;

                /* out_row = add(m0, add(mul(half, const(_ACC)), const(i)))
                 * inner add has two side-effecting args; sequence L-to-R. */
                {
                    rocke_value_t* hm = rocke_b_mul(b, half, rocke_b_const_i32(b, ROCKE_BSG_ACC));
                    rocke_value_t* ci = rocke_b_const_i32(b, i);
                    out_row = rocke_b_add(b, m0, rocke_b_add(b, hm, ci));
                }
                /* a_scale_off = add(mul(out_row, const(groups)), const(kg))
                 * arg-1 (mul) and arg-2 (const(kg)) both side-effecting; sequence. */
                {
                    rocke_value_t* om = rocke_b_mul(b, out_row, rocke_b_const_i32(b, groups));
                    rocke_value_t* ckg = rocke_b_const_i32(b, kg);
                    a_scale_off = rocke_b_add(b, om, ckg);
                }
                /* a_scale = _as_f32(ir.global_load(AScale, a_scale_off, scale_ty, align=4)) */
                a_scale = rocke_bsg_as_f32(
                    b,
                    rocke_b_global_load(b, AScale, a_scale_off, scale_ty, /*align=*/4),
                    scale_ty);
                /* ab = ir.fmul(a_scale, b_scale) */
                ab = rocke_b_fmul(b, a_scale, b_scale);
                /* outer[i] = ir.fadd(outer[i], ir.fmul(ir.vec_extract(acc, i), ab)) */
                prod = rocke_b_fmul(b, rocke_b_vec_extract(b, acc, i), ab);
                outer[i] = rocke_b_fadd(b, outer[i], prod);
            }
        }

        /* out_col = add(n0, frag); row_base = add(m0, mul(half, const(_ACC))) */
        out_col = rocke_b_add(b, n0, frag);
        row_base = rocke_b_add(b, m0, rocke_b_mul(b, half, rocke_b_const_i32(b, ROCKE_BSG_ACC)));
        for(i = 0; i < ROCKE_BSG_ACC; ++i)
        {
            /* out_row = add(row_base, const(i)) */
            rocke_value_t* out_row = rocke_b_add(b, row_base, rocke_b_const_i32(b, i));
            /* idx = add(mul(out_row, cN), out_col) */
            rocke_value_t* idx = rocke_b_add(b, rocke_b_mul(b, out_row, cN), out_col);
            /* ir.global_store(C, idx, ir.cast_f32_to(outer[i], c_ty), align=2) */
            rocke_value_t* sval = rocke_b_cast_f32_to(b, outer[i], c_ty);
            rocke_b_global_store(b, C, idx, sval, /*align=*/2);
        }

        free(outer);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_block_scaled_gemm_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_block_scaled_gemm_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_block_scaled_gemm_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_block_scaled_gemm_gfx1250(b, spec, arch);
    });
}

rocke_status_t
    rocke_block_scaled_gemm_gfx1250_grid(const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
                                         int out[3])
{
    int totals[2];
    int tiles[2];
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    totals[0] = spec->N;
    tiles[0] = spec->tile_n;
    totals[1] = spec->M;
    tiles[1] = spec->tile_m;
    return rocke_ceil_div_grid(totals, tiles, 2, out);
}

rocke_status_t rocke_block_scaled_gemm_gfx1250_lower_to_llvm(
    const rocke_block_scaled_gemm_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }

    kernel = rocke_build_block_scaled_gemm_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_block_scaled_gemm_gfx1250 failed";
                n = strlen(m);
            }
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
