// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_wmma_gemm_int8.c -- C99 port of
 * rocke/instances/gfx1151/wmma_gemm_int8.py.
 *
 * The gfx1151 (RDNA3.5 / Strix Halo) INT8-storage / f16-compute WMMA GEMM
 * kernel: one wave32 per 16x16 output tile, no LDS, RCR layout (C = A @ B.T).
 * Operands are stored as symmetric per-tensor int8 and dequantized to f16 on
 * chip; the combined scale (scale_a*scale_b) is folded into the epilogue. The
 * build op order tracks build_wmma_gemm_int8() top-to-bottom so a reviewer can
 * diff line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_gemm_int8.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

/* wmma_gemm_int8.py module constants. */
#define ROCKE_WMMA_GEMM_INT8_DEFAULT_NAME "rocke_wmma_gemm_int8"
#define ROCKE_WMMA_GEMM_INT8_DEFAULT_DTYPE "i8"

#define ROCKE_WMMA_M 16 /* _WMMA_M */
#define ROCKE_WMMA_N 16 /* _WMMA_N */
#define ROCKE_WMMA_K 16 /* _WMMA_K */
#define ROCKE_WMMA_WAVE 32 /* _WAVE */

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

rocke_wmma_gemm_int8_spec_t rocke_wmma_gemm_int8_spec_default(void)
{
    rocke_wmma_gemm_int8_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = ROCKE_WMMA_GEMM_INT8_DEFAULT_NAME;
    s.dtype = ROCKE_WMMA_GEMM_INT8_DEFAULT_DTYPE;
    return s;
}

/* WmmaGemmInt8Spec.block_size: one wave32. */
int rocke_wmma_gemm_int8_block_size(const rocke_wmma_gemm_int8_spec_t* spec)
{
    (void)spec;
    return ROCKE_WMMA_WAVE;
}

/* WmmaGemmInt8Spec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x16", "i8_f16", "rcr"). */
rocke_status_t rocke_wmma_gemm_int8_kernel_name(const rocke_wmma_gemm_int8_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    parts[0] = "wmma16x16x16";
    parts[1] = "i8_f16";
    parts[2] = "rcr";

    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void rocke_wmma_gemm_int8_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_wmma_gemm_int8_is_valid_spec(const rocke_wmma_gemm_int8_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_arch_target_t* target;
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_wmma_gemm_int8_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1151";
    }

    /* __post_init__: WmmaGemmInt8Spec stores operands as int8 only. */
    if(spec->dtype == NULL || strcmp(spec->dtype, "i8") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WmmaGemmInt8Spec stores operands as int8 only, got %s%s%s",
                 spec->dtype ? "'" : "",
                 spec->dtype ? spec->dtype : "None",
                 spec->dtype ? "'" : "");
        rocke_wmma_gemm_int8_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: ... */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unknown gfx target %s%s%s",
                 arch ? "'" : "",
                 arch ? arch : "None",
                 arch ? "'" : "");
        rocke_wmma_gemm_int8_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if not target.mma.has_shape(family="wmma", a=fp16, b=fp16, c=fp32,
     *                             m=16, n=16, k=16): ...
     * Operands are int8 in memory but dequantized to f16 before the MMA, so the
     * *compute* atom we gate on is the fp16 WMMA 16x16x16 one.
     * op_for_shape returns NULL when the shape/dtype combo is absent. */
    if(rocke_archtarget_op_for_shape(
           target, "wmma", "fp16", "fp16", "fp32", ROCKE_WMMA_M, ROCKE_WMMA_N, ROCKE_WMMA_K)
       == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WMMA %dx%dx%d f16 compute atom absent on %s "
                 "(WMMA is an RDNA/gfx11 instruction)",
                 ROCKE_WMMA_M,
                 ROCKE_WMMA_N,
                 ROCKE_WMMA_K,
                 arch);
        rocke_wmma_gemm_int8_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.wave_size != _WAVE: ... */
    if(target->wave_size != ROCKE_WMMA_WAVE)
    {
        snprintf(
            buf, sizeof(buf), "this WMMA kernel is wave32; %s is wave%d", arch, target->wave_size);
        rocke_wmma_gemm_int8_set_reason(reason, reason_cap, buf);
        return false;
    }

    rocke_wmma_gemm_int8_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  _i8_to_f16(b, vec, i):
 *    extract lane-fragment slot i (int8), sign-extend to i32, sitofp to f32,
 *    then round to f16:
 *      b.cast_f32_to(b.sitofp_f32(b.sext(b.vec_extract(vec, i), I32)), F16)
 * ===================================================================== */
static rocke_value_t* rocke_wmma_gemm_int8_i8_to_f16(rocke_ir_builder_t* b,
                                                     rocke_value_t* vec,
                                                     int i,
                                                     const rocke_type_t* f16)
{
    rocke_value_t* slot = rocke_b_vec_extract(b, vec, i);
    rocke_value_t* wide = rocke_b_sext(b, slot, rocke_i32());
    rocke_value_t* asf32 = rocke_b_sitofp_f32(b, wide);
    return rocke_b_cast_f32_to(b, asf32, f16);
}

/* ===================================================================== *
 *  build_wmma_gemm_int8(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_wmma_gemm_int8(rocke_ir_builder_t* b,
                                               const rocke_wmma_gemm_int8_spec_t* spec,
                                               const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* f16;
        const rocke_type_t* i8;
        rocke_value_t* A;
        rocke_value_t* Bp;
        rocke_value_t* C;
        rocke_value_t* scale_a;
        rocke_value_t* scale_b;
        rocke_value_t* scale;
        rocke_value_t* c0;
        rocke_value_t* c16;
        rocke_value_t* c32;
        rocke_value_t* Kparam;
        rocke_value_t* lane;
        rocke_value_t* frag;
        rocke_value_t* half;
        rocke_value_t* m0;
        rocke_value_t* n0;
        rocke_value_t* a_base;
        rocke_value_t* b_base;
        rocke_value_t* acc0;
        rocke_value_t* acc;
        rocke_value_t* out_col;
        rocke_for_t loop;
        rocke_iter_arg_t iter_args[1];
        int i;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1151";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_wmma_gemm_int8_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            char msg[ROCKE_ERR_MSG_CAP];
            ROCKE_ERR_SNPRINTF(msg, sizeof(msg), "invalid WMMA int8 GEMM spec: %s", reason);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }

        /* The builder `b` is assumed already initialised by the caller with
         * spec.kernel_name() (per the public header contract). Set the attr the
         * Python bakes in: b.kernel.attrs["max_workgroup_size"] = _WAVE. */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ROCKE_WMMA_WAVE);

        f16 = rocke_f16();
        i8 = rocke_i8();

        /* ---- kernel params -- */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_i8 = rocke_ptr_type(b, i8, "global");
            const rocke_type_t* ptr_f16 = rocke_ptr_type(b, f16, "global");

            /* A = b.param("A", PtrType(I8,"global"), noalias, readonly, align16)
             * Bp = b.param("B", PtrType(I8,"global"), noalias, readonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            A = rocke_b_param(b, "A", ptr_i8, &opts);
            Bp = rocke_b_param(b, "B", ptr_i8, &opts);

            /* C = b.param("C", PtrType(F16,"global"), noalias, writeonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            C = rocke_b_param(b, "C", ptr_f16, &opts);

            /* M / N / K : i32. M unused after declare (kept for ABI parity); N used
             * for the row-major output index; K is the loop bound + A/B row stride. */
            (void)rocke_b_param(b, "M", rocke_i32(), NULL);
            (void)rocke_b_param(b, "N", rocke_i32(), NULL);
            Kparam = rocke_b_param(b, "K", rocke_i32(), NULL);

            /* scale_a / scale_b : f32 (per-tensor symmetric dequant scales). */
            scale_a = rocke_b_param(b, "scale_a", rocke_f32(), NULL);
            scale_b = rocke_b_param(b, "scale_b", rocke_f32(), NULL);
        }

        /* scale = b.fmul(scale_a, scale_b) */
        scale = rocke_b_fmul(b, scale_a, scale_b);

        /* c0 = b.const_i32(0); c16 = b.const_i32(_WMMA_K); c32 = b.const_i32(_WAVE) */
        c0 = rocke_b_const_i32(b, 0);
        c16 = rocke_b_const_i32(b, ROCKE_WMMA_K);
        c32 = rocke_b_const_i32(b, ROCKE_WMMA_WAVE);

        /* lane = b.mod(b.thread_id_x(), c32) */
        lane = rocke_b_mod(b, rocke_b_thread_id_x(b), c32);
        /* frag = b.mod(lane, c16)  # lane%16 */
        frag = rocke_b_mod(b, lane, c16);
        /* half = b.div(lane, c16)  # lane/16: 0 or 1 */
        half = rocke_b_div(b, lane, c16);

        /* m0 = b.mul(b.block_id_x(), c16); n0 = b.mul(b.block_id_y(), c16) */
        m0 = rocke_b_mul(b, rocke_b_block_id_x(b), c16);
        n0 = rocke_b_mul(b, rocke_b_block_id_y(b), c16);

        /* a_base = b.mul(b.add(m0, frag), K); b_base = b.mul(b.add(n0, frag), K) */
        a_base = rocke_b_mul(b, rocke_b_add(b, m0, frag), Kparam);
        b_base = rocke_b_mul(b, rocke_b_add(b, n0, frag), Kparam);

        /* acc0 = b.zero_vec_f32(8) */
        acc0 = rocke_b_zero_vec_f32(b, 8);

        /* loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0") */
        iter_args[0].name = "acc";
        iter_args[0].init = acc0;
        loop = rocke_b_scf_for_iter(b,
                                    c0,
                                    Kparam,
                                    c16,
                                    iter_args,
                                    1,
                                    "k0",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* k0 = loop.iv;
            rocke_value_t* acc_v = loop.iter_vars[0];
            rocke_value_t* a_i8;
            rocke_value_t* b_i8;
            rocke_value_t* a_comps[16];
            rocke_value_t* b_comps[16];
            rocke_value_t* a_frag;
            rocke_value_t* b_frag;
            rocke_value_t* nacc;
            rocke_value_t* yield_vals[1];
            int j;

            /* a_i8 = b.global_load_vN(A, b.add(a_base, k0), I8, 16, align=16) */
            a_i8 = rocke_b_global_load_vN(b, A, rocke_b_add(b, a_base, k0), i8, 16, /*align=*/16);
            /* b_i8 = b.global_load_vN(Bp, b.add(b_base, k0), I8, 16, align=16) */
            b_i8 = rocke_b_global_load_vN(b, Bp, rocke_b_add(b, b_base, k0), i8, 16, /*align=*/16);

            /* a_frag = b.vec_pack([_i8_to_f16(b, a_i8, i) for i in range(16)], F16)
             * b_frag = b.vec_pack([_i8_to_f16(b, b_i8, i) for i in range(16)], F16) */
            for(j = 0; j < 16; ++j)
            {
                a_comps[j] = rocke_wmma_gemm_int8_i8_to_f16(b, a_i8, j, f16);
            }
            a_frag = rocke_b_vec_pack(b, a_comps, 16, f16);
            for(j = 0; j < 16; ++j)
            {
                b_comps[j] = rocke_wmma_gemm_int8_i8_to_f16(b, b_i8, j, f16);
            }
            b_frag = rocke_b_vec_pack(b, b_comps, 16, f16);

            /* nacc = b.wmma_f32_16x16x16_f16(a_frag, b_frag, acc) */
            nacc = rocke_b_wmma_f32_16x16x16_f16(b, a_frag, b_frag, acc_v);
            /* b.scf_yield(nacc) */
            yield_vals[0] = nacc;
            rocke_b_scf_yield(b, yield_vals, 1);
        }
        rocke_b_region_leave(b);

        /* acc = loop.results[0] */
        if(!rocke_ir_builder_ok(b) || loop.op == NULL || loop.op->num_results < 1)
        {
            return NULL;
        }
        acc = loop.op->results[0];

        /* Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
         * Fold the dequant scale into the f32 accumulator before truncating to f16. */
        /* out_col = b.add(n0, frag) */
        out_col = rocke_b_add(b, n0, frag);
        for(i = 0; i < 8; ++i)
        {
            rocke_value_t* elem;
            rocke_value_t* h;
            rocke_value_t* out_row;
            rocke_value_t* idx;
            rocke_value_t* Nparam = rocke_b_get_param(b, "N");

            /* elem = b.fmul(b.vec_extract(acc, i), scale) */
            elem = rocke_b_fmul(b, rocke_b_vec_extract(b, acc, i), scale);
            /* h = b.trunc_f32_to_f16(elem) */
            h = rocke_b_trunc_f32_to_f16(b, elem);
            /* out_row = b.add(m0, b.add(b.const_i32(2*i), half)) */
            out_row = rocke_b_add(b, m0, rocke_b_add(b, rocke_b_const_i32(b, 2 * i), half));
            /* idx = b.add(b.mul(out_row, N), out_col) */
            idx = rocke_b_add(b, rocke_b_mul(b, out_row, Nparam), out_col);
            /* b.global_store(C, idx, h) */
            rocke_b_global_store(b, C, idx, h, /*align=*/-1);
        }

        /* return b.kernel -- no explicit cf.return (added at lowering); matches Python IR. */

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* ===================================================================== *
 *  rocke_build_wmma_gemm_int8_new -- init builder with spec.kernel_name()
 *  then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_wmma_gemm_int8_new(rocke_ir_builder_t* b,
                                                   const rocke_wmma_gemm_int8_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_wmma_gemm_int8_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_wmma_gemm_int8(b, spec, arch);
    });
}

/* ===================================================================== *
 *  wmma_gemm_int8_grid(M, N) -> ((M+15)//16, (N+15)//16, 1)
 * ===================================================================== */
rocke_status_t rocke_wmma_gemm_int8_grid(int M, int N, int out[3])
{
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = (M + ROCKE_WMMA_M - 1) / ROCKE_WMMA_M;
    out[1] = (N + ROCKE_WMMA_N - 1) / ROCKE_WMMA_N;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  rocke_wmma_gemm_int8_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_wmma_gemm_int8_lower_to_llvm(const rocke_wmma_gemm_int8_spec_t* spec,
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
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1151";
    }

    kernel = rocke_build_wmma_gemm_int8_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_wmma_gemm_int8 failed";
            }
            n = strlen(m);
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
