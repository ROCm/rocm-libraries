/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_gfx1201_wmma_gemm.c -- C99 port of
 * ck_dsl/instances/gfx1201/wmma_gemm.py.
 *
 * The gfx1201 (RDNA4 / Navi48) WMMA GEMM kernel: one wave32 per 16x16 output
 * tile, no LDS, RCR layout (C = A @ B.T). The build op order tracks
 * build_wmma_gemm() top-to-bottom so a reviewer can diff line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_gfx1201_wmma_gemm.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */

#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/lower_llvm.h"

/* wmma_gemm.py module constants. */
#define CKC_WMMA_GEMM_GFX1201_DEFAULT_NAME "ck_dsl_wmma_gemm_gfx12"
#define CKC_WMMA_GEMM_GFX1201_DEFAULT_DTYPE "fp16"

#define CKC_WMMA_M 16  /* _WMMA_M */
#define CKC_WMMA_N 16  /* _WMMA_N */
#define CKC_WMMA_K 16  /* _WMMA_K */
#define CKC_WMMA_WAVE 32  /* _WAVE */
#define CKC_WMMA_HALF_K 8 /* _HALF_K: K-elements per lane-half (gfx12) */

/* ===================================================================== *
 *  Spec value accessors (the Python @property methods)
 * ===================================================================== */

ckc_wmma_gemm_gfx1201_spec_t ckc_wmma_gemm_gfx1201_spec_default(void)
{
    ckc_wmma_gemm_gfx1201_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = CKC_WMMA_GEMM_GFX1201_DEFAULT_NAME;
    s.dtype = CKC_WMMA_GEMM_GFX1201_DEFAULT_DTYPE;
    return s;
}

/* WmmaGemmSpec.block_size: one wave32. */
int ckc_wmma_gemm_gfx1201_block_size(const ckc_wmma_gemm_gfx1201_spec_t* spec)
{
    (void)spec;
    return CKC_WMMA_WAVE;
}

/* WmmaGemmSpec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x16", self.dtype, "rcr"). */
ckc_status_t ckc_wmma_gemm_gfx1201_kernel_name(const ckc_wmma_gemm_gfx1201_spec_t* spec,
                                               char* out,
                                               size_t out_cap)
{
    const char* parts[3];

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    parts[0] = "wmma16x16x16";
    parts[1] = spec->dtype;
    parts[2] = "rcr";

    return ckc_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

/* Write `msg` into reason (capacity reason_cap), NUL-terminated. */
static void ckc_wmma_gemm_gfx1201_set_reason(char* reason, size_t reason_cap, const char* msg)
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

bool ckc_wmma_gemm_gfx1201_is_valid_spec(const ckc_wmma_gemm_gfx1201_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    const ckc_arch_target_t* target;
    char buf[CKC_ERR_MSG_CAP];

    if (spec == NULL)
    {
        ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx1201";
    }

    /* __post_init__: WmmaGemmSpec currently supports fp16 only. */
    if (spec->dtype == NULL || strcmp(spec->dtype, "fp16") != 0)
    {
        snprintf(buf, sizeof(buf),
                 "WmmaGemmSpec currently supports fp16 only, got %s%s%s",
                 spec->dtype ? "'" : "", spec->dtype ? spec->dtype : "None",
                 spec->dtype ? "'" : "");
        ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* try: target = ArchTarget.from_gfx(arch) except KeyError as e: ... */
    target = ckc_archtarget_from_gfx(arch);
    if (target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s%s%s",
                 arch ? "'" : "", arch ? arch : "None", arch ? "'" : "");
        ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if not target.mma.has_shape(family="wmma", a=fp16, b=fp16, c=fp32,
     *                             m=16, n=16, k=16): ...
     * op_for_shape returns NULL when the shape/dtype combo is absent. */
    if (ckc_archtarget_op_for_shape(target, "wmma", spec->dtype, spec->dtype, "fp32",
                                    CKC_WMMA_M, CKC_WMMA_N, CKC_WMMA_K) == NULL)
    {
        snprintf(buf, sizeof(buf),
                 "WMMA %dx%dx%d %s atom absent on %s",
                 CKC_WMMA_M, CKC_WMMA_N, CKC_WMMA_K, spec->dtype, arch);
        ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.wave_size != _WAVE: ... */
    if (target->wave_size != CKC_WMMA_WAVE)
    {
        snprintf(buf, sizeof(buf),
                 "this WMMA kernel is wave32; %s is wave%d", arch, target->wave_size);
        ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, buf);
        return false;
    }

    ckc_wmma_gemm_gfx1201_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_wmma_gemm(spec, arch)
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_wmma_gemm_gfx1201(ckc_ir_builder_t* b,
                                              const ckc_wmma_gemm_gfx1201_spec_t* spec,
                                              const char* arch)
{
    const ckc_type_t* f16;
    ckc_value_t* A;
    ckc_value_t* Bp;
    ckc_value_t* C;
    ckc_value_t* c0;
    ckc_value_t* c16;
    ckc_value_t* c32;
    ckc_value_t* Kparam;
    ckc_value_t* lane;
    ckc_value_t* frag;
    ckc_value_t* half;
    ckc_value_t* half_k;
    ckc_value_t* m0;
    ckc_value_t* n0;
    ckc_value_t* a_base;
    ckc_value_t* b_base;
    ckc_value_t* acc0;
    ckc_value_t* acc;
    ckc_value_t* out_col;
    ckc_value_t* row_base;
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
        arch = "gfx1201";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
    if (!ckc_wmma_gemm_gfx1201_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        char msg[CKC_ERR_MSG_CAP];
        snprintf(msg, sizeof(msg), "invalid WMMA GEMM spec: %s", reason);
        (void)ckc_i_set_err(b, CKC_ERR_VALUE, "%s", msg);
        return NULL;
    }

    /* The builder `b` is assumed already initialised by the caller with
     * spec.kernel_name() (per the public header contract). Set the attr the
     * Python bakes in: b.kernel.attrs["max_workgroup_size"] = _WAVE. */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", CKC_WMMA_WAVE);

    f16 = ckc_f16();

    /* ---- kernel params -- */
    {
        ckc_param_opts_t opts;
        const ckc_type_t* ptr_f16 = ckc_ptr_type(b, f16, "global");

        /* A = b.param("A", PtrType(F16,"global"), noalias, readonly, align16)
         * Bp = b.param("B", ..., noalias, readonly, align16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = ckc_b_param(b, "A", ptr_f16, &opts);
        Bp = ckc_b_param(b, "B", ptr_f16, &opts);

        /* C = b.param("C", ..., noalias, writeonly, align16) */
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
    }

    /* c0 = b.const_i32(0); c16 = b.const_i32(_WMMA_K); c32 = b.const_i32(_WAVE) */
    c0 = ckc_b_const_i32(b, 0);
    c16 = ckc_b_const_i32(b, CKC_WMMA_K);
    c32 = ckc_b_const_i32(b, CKC_WMMA_WAVE);

    /* lane = b.mod(b.thread_id_x(), c32) */
    lane = ckc_b_mod(b, ckc_b_thread_id_x(b), c32);
    /* frag = b.mod(lane, c16)  # lane%16 */
    frag = ckc_b_mod(b, lane, c16);
    /* half = b.div(lane, c16)  # lane/16: 0 or 1 */
    half = ckc_b_div(b, lane, c16);
    /* half_k = b.mul(half, b.const_i32(_HALF_K)) */
    half_k = ckc_b_mul(b, half, ckc_b_const_i32(b, CKC_WMMA_HALF_K));

    /* m0 = b.mul(b.block_id_y(), c16); n0 = b.mul(b.block_id_x(), c16) */
    m0 = ckc_b_mul(b, ckc_b_block_id_y(b), c16);
    n0 = ckc_b_mul(b, ckc_b_block_id_x(b), c16);

    /* a_base = b.mul(b.add(m0, frag), K); b_base = b.mul(b.add(n0, frag), K) */
    a_base = ckc_b_mul(b, ckc_b_add(b, m0, frag), Kparam);
    b_base = ckc_b_mul(b, ckc_b_add(b, n0, frag), Kparam);

    /* acc0 = b.zero_vec_f32(8) */
    acc0 = ckc_b_zero_vec_f32(b, 8);

    /* loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0") */
    iter_args[0].name = "acc";
    iter_args[0].init = acc0;
    loop = ckc_b_scf_for_iter(b, c0, Kparam, c16, iter_args, 1, "k0",
                              /*unroll=*/false, /*elide_trailing_barrier=*/false);

    ckc_b_region_enter(b, loop.body);
    {
        ckc_value_t* k0 = loop.iv;
        ckc_value_t* acc_v = loop.iter_vars[0];
        ckc_value_t* a_off;
        ckc_value_t* b_off;
        ckc_value_t* a_frag;
        ckc_value_t* b_frag;
        ckc_value_t* nacc;
        ckc_value_t* yield_vals[1];

        /* a_off = b.add(b.add(a_base, k0), half_k) */
        a_off = ckc_b_add(b, ckc_b_add(b, a_base, k0), half_k);
        /* b_off = b.add(b.add(b_base, k0), half_k) */
        b_off = ckc_b_add(b, ckc_b_add(b, b_base, k0), half_k);
        /* a_frag = b.global_load_vN_f16(A, a_off, 8) */
        a_frag = ckc_b_global_load_vN_f16(b, A, a_off, 8, /*align=*/-1);
        /* b_frag = b.global_load_vN_f16(Bp, b_off, 8) */
        b_frag = ckc_b_global_load_vN_f16(b, Bp, b_off, 8, /*align=*/-1);
        /* nacc = b.wmma_gfx12_f32_16x16x16_f16(a_frag, b_frag, acc) */
        nacc = ckc_b_wmma_gfx12_f32_16x16x16_f16(b, a_frag, b_frag, acc_v);
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

    /* Epilogue (column-distributed): slot i of lane l ->
     *   (row = m0 + (l//16)*8 + i, col = n0 + l%16). */
    /* out_col = b.add(n0, frag) */
    out_col = ckc_b_add(b, n0, frag);
    /* row_base = b.add(m0, half_k) */
    row_base = ckc_b_add(b, m0, half_k);
    for (i = 0; i < 8; ++i)
    {
        ckc_value_t* elem;
        ckc_value_t* h;
        ckc_value_t* out_row;
        ckc_value_t* idx;
        ckc_value_t* Nparam = ckc_b_get_param(b, "N");

        /* elem = b.vec_extract(acc, i) */
        elem = ckc_b_vec_extract(b, acc, i);
        /* h = b.trunc_f32_to_f16(elem) */
        h = ckc_b_trunc_f32_to_f16(b, elem);
        /* out_row = b.add(row_base, b.const_i32(i)) */
        out_row = ckc_b_add(b, row_base, ckc_b_const_i32(b, i));
        /* idx = b.add(b.mul(out_row, N), out_col) */
        idx = ckc_b_add(b, ckc_b_mul(b, out_row, Nparam), out_col);
        /* b.global_store(C, idx, h) */
        ckc_b_global_store(b, C, idx, h, /*align=*/-1);
    }

    /* return b.kernel */
    ckc_b_ret(b);

    if (!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  ckc_build_wmma_gemm_gfx1201_new -- init builder with spec.kernel_name()
 *  then build.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_wmma_gemm_gfx1201_new(ckc_ir_builder_t* b,
                                                  const ckc_wmma_gemm_gfx1201_spec_t* spec,
                                                  const char* arch)
{
    char name[256];
    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_wmma_gemm_gfx1201_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_wmma_gemm_gfx1201(b, spec, arch);
}

/* ===================================================================== *
 *  wmma_gemm_grid(M, N) -> ((N+15)//16, (M+15)//16, 1)
 * ===================================================================== */
ckc_status_t ckc_wmma_gemm_gfx1201_grid(int M, int N, int out[3])
{
    if (out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = (N + CKC_WMMA_N - 1) / CKC_WMMA_N;
    out[1] = (M + CKC_WMMA_M - 1) / CKC_WMMA_M;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  ckc_wmma_gemm_gfx1201_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
ckc_status_t ckc_wmma_gemm_gfx1201_lower_to_llvm(const ckc_wmma_gemm_gfx1201_spec_t* spec,
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
        arch = "gfx1201";
    }

    kernel = ckc_build_wmma_gemm_gfx1201_new(&b, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            size_t n;
            if (m == NULL)
            {
                m = "build_wmma_gemm_gfx1201 failed";
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
