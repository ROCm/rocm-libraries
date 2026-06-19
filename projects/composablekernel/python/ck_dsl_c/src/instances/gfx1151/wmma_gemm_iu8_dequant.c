/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx1151_wmma_gemm_iu8_dequant.c -- C99 port of
 * ck_dsl/instances/gfx1151/wmma_gemm_iu8_dequant.py.
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

#include "ckc/instance_gfx1151_wmma_gemm_iu8_dequant.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */

#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/lower_llvm.h"

/* wmma_gemm_iu8_dequant.py module constants. */
#define CKC_WMMA_IU8_DEFAULT_NAME "ck_dsl_wmma_gemm_iu8_dequant"

#define CKC_WMMA_IU8_M 16     /* _WMMA_M */
#define CKC_WMMA_IU8_N 16     /* _WMMA_N */
#define CKC_WMMA_IU8_K 16     /* _WMMA_K */
#define CKC_WMMA_IU8_WAVE 32  /* _WAVE */
#define CKC_WMMA_IU8_K_PER_I32 4 /* _K_PER_I32: int8 K-values packed per i32 slot */
#define CKC_WMMA_IU8_OP_ID "wmma_i32_16x16x16_iu8" /* _OP_ID */

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

ckc_wmma_gemm_iu8_dequant_spec_t ckc_wmma_gemm_iu8_dequant_spec_default(void)
{
    ckc_wmma_gemm_iu8_dequant_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = CKC_WMMA_IU8_DEFAULT_NAME;
    return s;
}

/* WmmaGemmIu8DequantSpec.block_size: one wave32. */
int ckc_wmma_gemm_iu8_dequant_block_size(const ckc_wmma_gemm_iu8_dequant_spec_t* spec)
{
    (void)spec;
    return CKC_WMMA_IU8_WAVE;
}

/* WmmaGemmIu8DequantSpec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x16", "iu8_f16", "rcr"). */
ckc_status_t ckc_wmma_gemm_iu8_dequant_kernel_name(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    const char* parts[3];

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    parts[0] = "wmma16x16x16";
    parts[1] = "iu8_f16";
    parts[2] = "rcr";

    return ckc_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void ckc_wmma_gemm_iu8_dequant_set_reason(char* reason, size_t reason_cap, const char* msg)
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

bool ckc_wmma_gemm_iu8_dequant_is_valid_spec(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap)
{
    const ckc_arch_target_t* target;
    char buf[CKC_ERR_MSG_CAP];

    if (spec == NULL)
    {
        ckc_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx1151";
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: ... */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s%s%s",
                 arch ? "'" : "", arch ? arch : "None", arch ? "'" : "");
        ckc_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.mma.by_op_id(_OP_ID) is None: return False, f"{_OP_ID} atom
     * absent on {arch}" */
    if (ckc_archtarget_by_op_id(target, CKC_WMMA_IU8_OP_ID) == NULL)
    {
        snprintf(buf, sizeof(buf), "%s atom absent on %s", CKC_WMMA_IU8_OP_ID, arch);
        ckc_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.wave_size != _WAVE: ... */
    if (target->wave_size != CKC_WMMA_IU8_WAVE)
    {
        snprintf(buf, sizeof(buf),
                 "this WMMA kernel is wave32; %s is wave%d", arch, target->wave_size);
        ckc_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, buf);
        return false;
    }

    ckc_wmma_gemm_iu8_dequant_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_wmma_gemm_iu8_dequant(spec, arch)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_wmma_gemm_iu8_dequant(ckc_ir_builder_t* b,
                                                  const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                  const char* arch)
{
    const ckc_type_t* f16;
    const ckc_type_t* i32;
    const ckc_arch_target_t* target;
    const ckc_mmaop_t* op;
    const char* op_id;
    ckc_value_t* A;
    ckc_value_t* Bp;
    ckc_value_t* C;
    ckc_value_t* scale_a;
    ckc_value_t* scale_b;
    ckc_value_t* scale;
    ckc_value_t* c0;
    ckc_value_t* c4;
    ckc_value_t* c16;
    ckc_value_t* c32;
    ckc_value_t* Kparam;
    ckc_value_t* lane;
    ckc_value_t* frag;
    ckc_value_t* half;
    ckc_value_t* m0;
    ckc_value_t* n0;
    ckc_value_t* k4;
    ckc_value_t* a_base;
    ckc_value_t* b_base;
    ckc_value_t* acc0;
    ckc_value_t* acc;
    ckc_value_t* out_col;
    ckc_for_t loop;
    ckc_iter_arg_t iter_args[1];
    int i;
    char reason[CKC_ERR_MSG_CAP];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (arch == NULL)
    {
        arch = "gfx1151";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if (!ckc_wmma_gemm_iu8_dequant_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "invalid iu8 dequant WMMA GEMM spec: %s", reason);
        return NULL;
    }

    /* target = ArchTarget.from_gfx(arch); op = target.mma.by_op_id(_OP_ID) */
    target = ckc_archtarget_from_gfx(arch);
    op = ckc_archtarget_by_op_id(target, CKC_WMMA_IU8_OP_ID);
    if (op == NULL)
    {
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "%s atom absent on %s", CKC_WMMA_IU8_OP_ID, arch);
        return NULL;
    }
    op_id = op->op_id; /* the iu8 atom handle fed to ckc_b_mma */

    /* The builder `b` is assumed already initialised by the caller with
     * spec.kernel_name() (per the public header contract). Set the attr the
     * Python bakes in: b.kernel.attrs["max_workgroup_size"] = _WAVE. */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", CKC_WMMA_IU8_WAVE);

    f16 = ckc_f16();
    i32 = ckc_i32();

    /* ---- kernel params --
     * A/B are int8 logically but passed packed as i32 (4 int8/i32). C is f16. */
    {
        ckc_param_opts_t opts;
        const ckc_type_t* ptr_i32 = ckc_ptr_type(b, i32, "global");
        const ckc_type_t* ptr_f16 = ckc_ptr_type(b, f16, "global");

        /* A = b.param("A", PtrType(I32,"global"), noalias, readonly, align16)
         * Bp = b.param("B", PtrType(I32,"global"), noalias, readonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = ckc_b_param(b, "A", ptr_i32, &opts);
        Bp = ckc_b_param(b, "B", ptr_i32, &opts);

        /* C = b.param("C", PtrType(F16,"global"), noalias, writeonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = ckc_b_param(b, "C", ptr_f16, &opts);

        /* M / N / K : i32. M unused after declare (kept for ABI parity); N used
         * for the row-major output index; K is the loop bound + A/B row stride. */
        (void)ckc_b_param(b, "M", ckc_i32(), NULL);
        (void)ckc_b_param(b, "N", ckc_i32(), NULL);
        Kparam = ckc_b_param(b, "K", ckc_i32(), NULL);

        /* scale_a = b.param("scale_a", F32); scale_b = b.param("scale_b", F32) */
        scale_a = ckc_b_param(b, "scale_a", ckc_f32(), NULL);
        scale_b = ckc_b_param(b, "scale_b", ckc_f32(), NULL);
    }

    /* scale = b.fmul(scale_a, scale_b)  # combined per-tensor dequant scale */
    scale = ckc_b_fmul(b, scale_a, scale_b);

    /* c0 = const_i32(0); c4 = const_i32(_K_PER_I32); c16 = const_i32(_WMMA_K);
     * c32 = const_i32(_WAVE) */
    c0 = ckc_b_const_i32(b, 0);
    c4 = ckc_b_const_i32(b, CKC_WMMA_IU8_K_PER_I32);
    c16 = ckc_b_const_i32(b, CKC_WMMA_IU8_K);
    c32 = ckc_b_const_i32(b, CKC_WMMA_IU8_WAVE);

    /* lane = b.mod(b.thread_id_x(), c32) */
    lane = ckc_b_mod(b, ckc_b_thread_id_x(b), c32);
    /* frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag col, output col */
    frag = ckc_b_mod(b, lane, c16);
    /* half = b.div(lane, c16)  # lane/16: even/odd output row selector */
    half = ckc_b_div(b, lane, c16);

    /* m0 = b.mul(b.block_id_x(), c16); n0 = b.mul(b.block_id_y(), c16) */
    m0 = ckc_b_mul(b, ckc_b_block_id_x(b), c16);
    n0 = ckc_b_mul(b, ckc_b_block_id_y(b), c16);

    /* k4 = b.div(K, c4)  # i32 columns per row */
    k4 = ckc_b_div(b, Kparam, c4);
    /* a_base = b.mul(b.add(m0, frag), k4); b_base = b.mul(b.add(n0, frag), k4) */
    a_base = ckc_b_mul(b, ckc_b_add(b, m0, frag), k4);
    b_base = ckc_b_mul(b, ckc_b_add(b, n0, frag), k4);

    /* acc0 = b.zero_vec(I32, 8) */
    acc0 = ckc_b_zero_vec(b, i32, 8);

    /* loop = b.scf_for_iter(c0, k4, c4, [("acc", acc0)], iv_name="k0") */
    iter_args[0].name = "acc";
    iter_args[0].init = acc0;
    loop = ckc_b_scf_for_iter(b, c0, k4, c4, iter_args, 1, "k0",
                              /*unroll=*/false, /*elide_trailing_barrier=*/true);

    ckc_b_region_enter(b, loop.body);
    {
        ckc_value_t* k0 = loop.iv;
        ckc_value_t* acc_v = loop.iter_vars[0];
        ckc_value_t* a_frag;
        ckc_value_t* b_frag;
        ckc_value_t* nacc;
        ckc_value_t* yield_vals[1];

        /* a_frag = b.global_load_vN(A, b.add(a_base, k0), I32, _K_PER_I32) */
        a_frag = ckc_b_global_load_vN(b, A, ckc_b_add(b, a_base, k0), i32,
                                      CKC_WMMA_IU8_K_PER_I32, /*align=*/-1);
        /* b_frag = b.global_load_vN(Bp, b.add(b_base, k0), I32, _K_PER_I32) */
        b_frag = ckc_b_global_load_vN(b, Bp, ckc_b_add(b, b_base, k0), i32,
                                      CKC_WMMA_IU8_K_PER_I32, /*align=*/-1);
        /* nacc = b.mma(op, a_frag, b_frag, acc) */
        nacc = ckc_b_mma(b, op_id, a_frag, b_frag, acc_v, NULL, 0);
        /* b.scf_yield(nacc) */
        yield_vals[0] = nacc;
        ckc_b_scf_yield(b, yield_vals, 1);
    }
    ckc_b_region_leave(b);

    /* acc = loop.results[0] */
    if (!ckc_ir_builder_ok(b) || loop.op == NULL || loop.op->num_results < 1)
    {
        return NULL;
    }
    acc = loop.op->results[0];

    /* Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
     * Dequantize the i32 accumulator (-> f32 -> * scale) before the f16 store. */
    /* out_col = b.add(n0, frag) */
    out_col = ckc_b_add(b, n0, frag);
    for (i = 0; i < 8; ++i)
    {
        ckc_value_t* elem;
        ckc_value_t* deq;
        ckc_value_t* h;
        ckc_value_t* out_row;
        ckc_value_t* idx;
        ckc_value_t* Nparam = ckc_b_get_param(b, "N");

        /* deq = b.fmul(b.sitofp_f32(b.vec_extract(acc, i)), scale) */
        elem = ckc_b_vec_extract(b, acc, i);
        deq = ckc_b_fmul(b, ckc_b_sitofp_f32(b, elem), scale);
        /* h = b.trunc_f32_to_f16(deq) */
        h = ckc_b_trunc_f32_to_f16(b, deq);
        /* out_row = b.add(m0, b.add(b.const_i32(2 * i), half)) */
        out_row = ckc_b_add(b, m0, ckc_b_add(b, ckc_b_const_i32(b, 2 * i), half));
        /* idx = b.add(b.mul(out_row, N), out_col) */
        idx = ckc_b_add(b, ckc_b_mul(b, out_row, Nparam), out_col);
        /* b.global_store(C, idx, h) */
        ckc_b_global_store(b, C, idx, h, /*align=*/-1);
    }

    /* return b.kernel -- no explicit cf.return (added at lowering); matches Python IR. */

    if (!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  ckc_build_wmma_gemm_iu8_dequant_new -- init builder with spec.kernel_name()
 *  then build.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_wmma_gemm_iu8_dequant_new(ckc_ir_builder_t* b,
                                                      const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                      const char* arch)
{
    char name[256];
    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_wmma_gemm_iu8_dequant_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_wmma_gemm_iu8_dequant(b, spec, arch);
}

/* ===================================================================== *
 *  wmma_gemm_iu8_dequant_grid(M, N) -> ((M+15)//16, (N+15)//16, 1)
 * ===================================================================== */
ckc_status_t ckc_wmma_gemm_iu8_dequant_grid(int M, int N, int out[3])
{
    if (out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = (M + CKC_WMMA_IU8_M - 1) / CKC_WMMA_IU8_M;
    out[1] = (N + CKC_WMMA_IU8_N - 1) / CKC_WMMA_IU8_N;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  ckc_wmma_gemm_iu8_dequant_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
ckc_status_t ckc_wmma_gemm_iu8_dequant_lower_to_llvm(const ckc_wmma_gemm_iu8_dequant_spec_t* spec,
                                                     const char* arch,
                                                     ckc_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if (out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if (spec == NULL || out_ll == NULL)
    {
        if (err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if (n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx1151";
    }

    kernel = ckc_build_wmma_gemm_iu8_dequant_new(&b, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if (m == NULL)
            {
                m = "build_wmma_gemm_iu8_dequant failed";
            }
            n = strlen(m);
            if (n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
