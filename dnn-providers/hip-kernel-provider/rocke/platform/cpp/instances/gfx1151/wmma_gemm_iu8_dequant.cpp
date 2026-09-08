// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1151_wmma_gemm_iu8_dequant.c -- C99 port of
 * rocke/instances/gfx1151/wmma_gemm_iu8_dequant.py.
 *
 * The gfx1151 (RDNA3.5 / Strix Halo) true-INT8 WMMA GEMM with f16 dequant
 * output: one wave32 per 16x16 output tile, no LDS, RCR layout (C = A @ B.T).
 * Runs the hardware wmma_i32_16x16x16_iu8 instruction (int8 x int8 -> int32
 * accumulate); A/B are passed packed as i32 (4 int8 per i32), the <8 x i32>
 * accumulator is loop-carried, and the epilogue dequantizes each i32 slot
 * (sitofp -> * (scale_a*scale_b)) before truncating to f16. The build op order
 * tracks build_wmma_gemm_iu8_dequant() top-to-bottom so a reviewer can diff
 * line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_gemm_iu8_dequant.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

/* wmma_gemm_iu8_dequant.py module constants. */
#define ROCKE_WMMA_IU8_DEFAULT_NAME "rocke_wmma_gemm_iu8_dequant"

#define ROCKE_WMMA_IU8_M 16 /* _WMMA_M */
#define ROCKE_WMMA_IU8_N 16 /* _WMMA_N */
#define ROCKE_WMMA_IU8_K 16 /* _WMMA_K */
#define ROCKE_WMMA_IU8_WAVE 32 /* _WAVE */
#define ROCKE_WMMA_IU8_K_PER_I32 4 /* _K_PER_I32: int8 K-values packed per i32 slot */
#define ROCKE_WMMA_IU8_OP_ID "wmma_i32_16x16x16_iu8" /* _OP_ID */

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

rocke_wmma_gemm_iu8_dequant_spec_t rocke_wmma_gemm_iu8_dequant_spec_default(void)
{
    rocke_wmma_gemm_iu8_dequant_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = ROCKE_WMMA_IU8_DEFAULT_NAME;
    return s;
}

/* WmmaGemmIu8DequantSpec.block_size: one wave32. */
int rocke_wmma_gemm_iu8_dequant_block_size(const rocke_wmma_gemm_iu8_dequant_spec_t* spec)
{
    (void)spec;
    return ROCKE_WMMA_IU8_WAVE;
}

/* WmmaGemmIu8DequantSpec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x16", "iu8_f16", "rcr"). */
rocke_status_t rocke_wmma_gemm_iu8_dequant_kernel_name(
    const rocke_wmma_gemm_iu8_dequant_spec_t* spec, char* out, size_t out_cap)
{
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    parts[0] = "wmma16x16x16";
    parts[1] = "iu8_f16";
    parts[2] = "rcr";

    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void rocke_wmma_gemm_iu8_dequant_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_wmma_gemm_iu8_dequant_is_valid_spec(const rocke_wmma_gemm_iu8_dequant_spec_t* spec,
                                               const char* arch,
                                               char* reason,
                                               size_t reason_cap)
{
    const rocke_arch_target_t* target;
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1151";
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
        rocke_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.mma.by_op_id(_OP_ID) is None: return False, f"{_OP_ID} atom
     * absent on {arch}" */
    if(rocke_archtarget_by_op_id(target, ROCKE_WMMA_IU8_OP_ID) == NULL)
    {
        snprintf(buf, sizeof(buf), "%s atom absent on %s", ROCKE_WMMA_IU8_OP_ID, arch);
        rocke_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.wave_size != _WAVE: ... */
    if(target->wave_size != ROCKE_WMMA_IU8_WAVE)
    {
        snprintf(
            buf, sizeof(buf), "this WMMA kernel is wave32; %s is wave%d", arch, target->wave_size);
        rocke_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    rocke_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_wmma_gemm_iu8_dequant(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_wmma_gemm_iu8_dequant(
    rocke_ir_builder_t* b, const rocke_wmma_gemm_iu8_dequant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* f16;
        const rocke_type_t* i32;
        const rocke_arch_target_t* target;
        const rocke_mmaop_t* op;
        const char* op_id;
        rocke_value_t* A;
        rocke_value_t* Bp;
        rocke_value_t* C;
        rocke_value_t* scale_a;
        rocke_value_t* scale_b;
        rocke_value_t* scale;
        rocke_value_t* c0;
        rocke_value_t* c4;
        rocke_value_t* c16;
        rocke_value_t* c32;
        rocke_value_t* Kparam;
        rocke_value_t* lane;
        rocke_value_t* frag;
        rocke_value_t* half;
        rocke_value_t* m0;
        rocke_value_t* n0;
        rocke_value_t* k4;
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
        if(!rocke_wmma_gemm_iu8_dequant_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "invalid iu8 dequant WMMA GEMM spec: %s", reason);
            return NULL;
        }

        /* target = ArchTarget.from_gfx(arch); op = target.mma.by_op_id(_OP_ID) */
        target = rocke_archtarget_from_gfx(arch);
        op = rocke_archtarget_by_op_id(target, ROCKE_WMMA_IU8_OP_ID);
        if(op == NULL)
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "%s atom absent on %s", ROCKE_WMMA_IU8_OP_ID, arch);
            return NULL;
        }
        op_id = op->op_id; /* the iu8 atom handle fed to rocke_b_mma */

        /* The builder `b` is assumed already initialised by the caller with
         * spec.kernel_name() (per the public header contract). Set the attr the
         * Python bakes in: b.kernel.attrs["max_workgroup_size"] = _WAVE. */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ROCKE_WMMA_IU8_WAVE);

        f16 = rocke_f16();
        i32 = rocke_i32();

        /* ---- kernel params --
         * A/B are int8 logically but passed packed as i32 (4 int8/i32). C is f16. */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_i32 = rocke_ptr_type(b, i32, "global");
            const rocke_type_t* ptr_f16 = rocke_ptr_type(b, f16, "global");

            /* A = b.param("A", PtrType(I32,"global"), noalias, readonly, align16)
             * Bp = b.param("B", PtrType(I32,"global"), noalias, readonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            A = rocke_b_param(b, "A", ptr_i32, &opts);
            Bp = rocke_b_param(b, "B", ptr_i32, &opts);

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

            /* scale_a = b.param("scale_a", F32); scale_b = b.param("scale_b", F32) */
            scale_a = rocke_b_param(b, "scale_a", rocke_f32(), NULL);
            scale_b = rocke_b_param(b, "scale_b", rocke_f32(), NULL);
        }

        /* scale = b.fmul(scale_a, scale_b)  # combined per-tensor dequant scale */
        scale = rocke_b_fmul(b, scale_a, scale_b);

        /* c0 = const_i32(0); c4 = const_i32(_K_PER_I32); c16 = const_i32(_WMMA_K);
         * c32 = const_i32(_WAVE) */
        c0 = rocke_b_const_i32(b, 0);
        c4 = rocke_b_const_i32(b, ROCKE_WMMA_IU8_K_PER_I32);
        c16 = rocke_b_const_i32(b, ROCKE_WMMA_IU8_K);
        c32 = rocke_b_const_i32(b, ROCKE_WMMA_IU8_WAVE);

        /* lane = b.mod(b.thread_id_x(), c32) */
        lane = rocke_b_mod(b, rocke_b_thread_id_x(b), c32);
        /* frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag col, output col */
        frag = rocke_b_mod(b, lane, c16);
        /* half = b.div(lane, c16)  # lane/16: even/odd output row selector */
        half = rocke_b_div(b, lane, c16);

        /* m0 = b.mul(b.block_id_x(), c16); n0 = b.mul(b.block_id_y(), c16) */
        m0 = rocke_b_mul(b, rocke_b_block_id_x(b), c16);
        n0 = rocke_b_mul(b, rocke_b_block_id_y(b), c16);

        /* k4 = b.div(K, c4)  # i32 columns per row */
        k4 = rocke_b_div(b, Kparam, c4);
        /* a_base = b.mul(b.add(m0, frag), k4); b_base = b.mul(b.add(n0, frag), k4) */
        a_base = rocke_b_mul(b, rocke_b_add(b, m0, frag), k4);
        b_base = rocke_b_mul(b, rocke_b_add(b, n0, frag), k4);

        /* acc0 = b.zero_vec(I32, 8) */
        acc0 = rocke_b_zero_vec(b, i32, 8);

        /* loop = b.scf_for_iter(c0, k4, c4, [("acc", acc0)], iv_name="k0") */
        iter_args[0].name = "acc";
        iter_args[0].init = acc0;
        loop = rocke_b_scf_for_iter(b,
                                    c0,
                                    k4,
                                    c4,
                                    iter_args,
                                    1,
                                    "k0",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* k0 = loop.iv;
            rocke_value_t* acc_v = loop.iter_vars[0];
            rocke_value_t* a_frag;
            rocke_value_t* b_frag;
            rocke_value_t* nacc;
            rocke_value_t* yield_vals[1];

            /* a_frag = b.global_load_vN(A, b.add(a_base, k0), I32, _K_PER_I32) */
            a_frag = rocke_b_global_load_vN(
                b, A, rocke_b_add(b, a_base, k0), i32, ROCKE_WMMA_IU8_K_PER_I32, /*align=*/-1);
            /* b_frag = b.global_load_vN(Bp, b.add(b_base, k0), I32, _K_PER_I32) */
            b_frag = rocke_b_global_load_vN(
                b, Bp, rocke_b_add(b, b_base, k0), i32, ROCKE_WMMA_IU8_K_PER_I32, /*align=*/-1);
            /* nacc = b.mma(op, a_frag, b_frag, acc) */
            nacc = rocke_b_mma(b, op_id, a_frag, b_frag, acc_v, NULL, 0);
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
         * Dequantize the i32 accumulator (-> f32 -> * scale) before the f16 store. */
        /* out_col = b.add(n0, frag) */
        out_col = rocke_b_add(b, n0, frag);
        for(i = 0; i < 8; ++i)
        {
            rocke_value_t* elem;
            rocke_value_t* deq;
            rocke_value_t* h;
            rocke_value_t* out_row;
            rocke_value_t* idx;
            rocke_value_t* Nparam = rocke_b_get_param(b, "N");

            /* deq = b.fmul(b.sitofp_f32(b.vec_extract(acc, i)), scale) */
            elem = rocke_b_vec_extract(b, acc, i);
            deq = rocke_b_fmul(b, rocke_b_sitofp_f32(b, elem), scale);
            /* h = b.trunc_f32_to_f16(deq) */
            h = rocke_b_trunc_f32_to_f16(b, deq);
            /* out_row = b.add(m0, b.add(b.const_i32(2 * i), half)) */
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
 *  rocke_build_wmma_gemm_iu8_dequant_new -- init builder with spec.kernel_name()
 *  then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_wmma_gemm_iu8_dequant_new(
    rocke_ir_builder_t* b, const rocke_wmma_gemm_iu8_dequant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_wmma_gemm_iu8_dequant_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_wmma_gemm_iu8_dequant(b, spec, arch);
    });
}

/* ===================================================================== *
 *  wmma_gemm_iu8_dequant_grid(M, N) -> ((M+15)//16, (N+15)//16, 1)
 * ===================================================================== */
rocke_status_t rocke_wmma_gemm_iu8_dequant_grid(int M, int N, int out[3])
{
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = (M + ROCKE_WMMA_IU8_M - 1) / ROCKE_WMMA_IU8_M;
    out[1] = (N + ROCKE_WMMA_IU8_N - 1) / ROCKE_WMMA_IU8_N;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  rocke_wmma_gemm_iu8_dequant_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t
    rocke_wmma_gemm_iu8_dequant_lower_to_llvm(const rocke_wmma_gemm_iu8_dequant_spec_t* spec,
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

    kernel = rocke_build_wmma_gemm_iu8_dequant_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_wmma_gemm_iu8_dequant failed";
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
