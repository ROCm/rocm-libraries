// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_wmma_gemm.c -- C99 port of rocke/instances/gfx1250/wmma_gemm.py.
 *
 * The gfx1250 (CDNA, GFX12 programming model) WMMA GEMM: one wave32 per 16x16
 * output tile, no LDS, RCR layout (C = A @ B.T), K=32 fragment ABI. The build op
 * order tracks build_wmma_gemm() top-to-bottom so a reviewer can diff line by
 * line; emitted IR is byte-identical to the Python lowerer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_wmma_gemm.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_WMMA_GEMM_GFX1250_DEFAULT_NAME "rocke_wmma_gemm_gfx1250"
#define ROCKE_WMMA_GEMM_GFX1250_DEFAULT_DTYPE "fp16"

#define ROCKE_WMMA_M 16 /* _WMMA_M */
#define ROCKE_WMMA_N 16 /* _WMMA_N */
#define ROCKE_WMMA_K 32 /* _WMMA_K (gfx1250: K=32) */
#define ROCKE_WMMA_WAVE 32 /* _WAVE */
#define ROCKE_WMMA_HALF_K 16 /* _HALF_K: K-elements per lane-half (gfx1250: 16) */

rocke_wmma_gemm_gfx1250_spec_t rocke_wmma_gemm_gfx1250_spec_default(void)
{
    rocke_wmma_gemm_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = ROCKE_WMMA_GEMM_GFX1250_DEFAULT_NAME;
    s.dtype = ROCKE_WMMA_GEMM_GFX1250_DEFAULT_DTYPE;
    return s;
}

int rocke_wmma_gemm_gfx1250_block_size(const rocke_wmma_gemm_gfx1250_spec_t* spec)
{
    (void)spec;
    return ROCKE_WMMA_WAVE;
}

/* WmmaGemmSpec.kernel_name():
 *   kernel_name_join(self.name, "wmma16x16x32", self.dtype, "rcr"). */
rocke_status_t rocke_wmma_gemm_gfx1250_kernel_name(const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    parts[0] = "wmma16x16x32";
    parts[1] = spec->dtype;
    parts[2] = "rcr";
    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_wmma_gemm_gfx1250_is_valid_spec(const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    const rocke_arch_target_t* target;
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

    /* __post_init__: WmmaGemmSpec currently supports fp16 only. */
    if(spec->dtype == NULL || strcmp(spec->dtype, "fp16") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WmmaGemmSpec currently supports fp16 only, got %s%s%s",
                 spec->dtype ? "'" : "",
                 spec->dtype ? spec->dtype : "None",
                 spec->dtype ? "'" : "");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* try: target = ArchTarget.from_gfx(arch) */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unknown gfx target %s%s%s",
                 arch ? "'" : "",
                 arch ? arch : "None",
                 arch ? "'" : "");
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if not target.mma.has_shape(family="wmma", a/b=fp16, c=fp32, 16x16x32). */
    if(rocke_archtarget_op_for_shape(target,
                                     "wmma",
                                     spec->dtype,
                                     spec->dtype,
                                     "fp32",
                                     ROCKE_WMMA_M,
                                     ROCKE_WMMA_N,
                                     ROCKE_WMMA_K)
       == NULL)
    {
        snprintf(buf,
                 sizeof(buf),
                 "WMMA %dx%dx%d %s atom absent on %s",
                 ROCKE_WMMA_M,
                 ROCKE_WMMA_N,
                 ROCKE_WMMA_K,
                 spec->dtype,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    /* if target.wave_size != _WAVE. */
    if(target->wave_size != ROCKE_WMMA_WAVE)
    {
        snprintf(
            buf, sizeof(buf), "this WMMA kernel is wave32; %s is wave%d", arch, target->wave_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1250(rocke_ir_builder_t* b,
                                                  const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                  const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* f16;
        rocke_value_t* A;
        rocke_value_t* Bp;
        rocke_value_t* C;
        rocke_value_t* c0;
        rocke_value_t* c16;
        rocke_value_t* c32;
        rocke_value_t* cK;
        rocke_value_t* Kparam;
        rocke_value_t* lane;
        rocke_value_t* frag;
        rocke_value_t* half;
        rocke_value_t* half_k;
        rocke_value_t* m0;
        rocke_value_t* n0;
        rocke_value_t* a_base;
        rocke_value_t* b_base;
        rocke_value_t* acc0;
        rocke_value_t* acc;
        rocke_value_t* out_col;
        rocke_value_t* row_base;
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
            arch = "gfx1250";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_wmma_gemm_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            char msg[ROCKE_ERR_MSG_CAP];
            ROCKE_ERR_SNPRINTF(msg, sizeof(msg), "invalid WMMA GEMM spec: %s", reason);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }

        /* b.kernel.attrs["max_workgroup_size"] = _WAVE */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ROCKE_WMMA_WAVE);

        f16 = rocke_f16();

        /* ---- kernel params -- */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_f16 = rocke_ptr_type(b, f16, "global");

            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            A = rocke_b_param(b, "A", ptr_f16, &opts);
            Bp = rocke_b_param(b, "B", ptr_f16, &opts);

            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            C = rocke_b_param(b, "C", ptr_f16, &opts);

            (void)rocke_b_param(b, "M", rocke_i32(), NULL);
            (void)rocke_b_param(b, "N", rocke_i32(), NULL);
            Kparam = rocke_b_param(b, "K", rocke_i32(), NULL);
        }

        /* c0 = const(0); c16 = const(_WMMA_M); c32 = const(_WAVE); cK = const(_WMMA_K) */
        c0 = rocke_b_const_i32(b, 0);
        c16 = rocke_b_const_i32(b, ROCKE_WMMA_M);
        c32 = rocke_b_const_i32(b, ROCKE_WMMA_WAVE);
        cK = rocke_b_const_i32(b, ROCKE_WMMA_K);

        /* lane = b.mod(b.thread_id_x(), c32) */
        lane = rocke_b_mod(b, rocke_b_thread_id_x(b), c32);
        /* frag = b.mod(lane, c16) */
        frag = rocke_b_mod(b, lane, c16);
        /* half = b.div(lane, c16) */
        half = rocke_b_div(b, lane, c16);
        /* half_k = b.mul(half, b.const_i32(_HALF_K)) */
        half_k = rocke_b_mul(b, half, rocke_b_const_i32(b, ROCKE_WMMA_HALF_K));

        /* m0 = b.mul(b.block_id_y(), c16); n0 = b.mul(b.block_id_x(), c16) */
        m0 = rocke_b_mul(b, rocke_b_block_id_y(b), c16);
        n0 = rocke_b_mul(b, rocke_b_block_id_x(b), c16);

        /* a_base = b.mul(b.add(m0, frag), K); b_base = b.mul(b.add(n0, frag), K) */
        a_base = rocke_b_mul(b, rocke_b_add(b, m0, frag), Kparam);
        b_base = rocke_b_mul(b, rocke_b_add(b, n0, frag), Kparam);

        /* acc0 = b.zero_vec_f32(8) */
        acc0 = rocke_b_zero_vec_f32(b, 8);

        /* loop = b.scf_for_iter(c0, K, cK, [("acc", acc0)], iv_name="k0") */
        iter_args[0].name = "acc";
        iter_args[0].init = acc0;
        loop = rocke_b_scf_for_iter(b,
                                    c0,
                                    Kparam,
                                    cK,
                                    iter_args,
                                    1,
                                    "k0",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* k0 = loop.iv;
            rocke_value_t* acc_v = loop.iter_vars[0];
            rocke_value_t* a_off;
            rocke_value_t* b_off;
            rocke_value_t* a_frag;
            rocke_value_t* b_frag;
            rocke_value_t* nacc;
            rocke_value_t* yield_vals[1];

            /* a_off = b.add(b.add(a_base, k0), half_k) */
            a_off = rocke_b_add(b, rocke_b_add(b, a_base, k0), half_k);
            /* b_off = b.add(b.add(b_base, k0), half_k) */
            b_off = rocke_b_add(b, rocke_b_add(b, b_base, k0), half_k);
            /* a_frag = b.global_load_vN_f16(A, a_off, 16) */
            a_frag = rocke_b_global_load_vN_f16(b, A, a_off, 16, /*align=*/-1);
            /* b_frag = b.global_load_vN_f16(Bp, b_off, 16) */
            b_frag = rocke_b_global_load_vN_f16(b, Bp, b_off, 16, /*align=*/-1);
            /* nacc = b.wmma_gfx1250_f32_16x16x32_f16(a_frag, b_frag, acc) */
            nacc = rocke_b_wmma_gfx1250_f32_16x16x32_f16(b, a_frag, b_frag, acc_v);
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

        /* Epilogue (column-distributed): slot i of lane l ->
         *   (row = m0 + (l//16)*8 + i, col = n0 + l%16). */
        /* out_col = b.add(n0, frag) */
        out_col = rocke_b_add(b, n0, frag);
        /* row_base = b.add(m0, b.mul(half, b.const_i32(8))) */
        row_base = rocke_b_add(b, m0, rocke_b_mul(b, half, rocke_b_const_i32(b, 8)));
        for(i = 0; i < 8; ++i)
        {
            rocke_value_t* elem;
            rocke_value_t* h;
            rocke_value_t* out_row;
            rocke_value_t* idx;
            rocke_value_t* Nparam = rocke_b_get_param(b, "N");

            /* elem = b.vec_extract(acc, i); h = b.trunc_f32_to_f16(elem) */
            elem = rocke_b_vec_extract(b, acc, i);
            h = rocke_b_trunc_f32_to_f16(b, elem);
            /* out_row = b.add(row_base, b.const_i32(i)) */
            out_row = rocke_b_add(b, row_base, rocke_b_const_i32(b, i));
            /* idx = b.add(b.mul(out_row, N), out_col) */
            idx = rocke_b_add(b, rocke_b_mul(b, out_row, Nparam), out_col);
            /* b.global_store(C, idx, h) */
            rocke_b_global_store(b, C, idx, h, /*align=*/-1);
        }

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_wmma_gemm_gfx1250_new(rocke_ir_builder_t* b,
                                                      const rocke_wmma_gemm_gfx1250_spec_t* spec,
                                                      const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_wmma_gemm_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_wmma_gemm_gfx1250(b, spec, arch);
    });
}

rocke_status_t rocke_wmma_gemm_gfx1250_grid(int M, int N, int out[3])
{
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = (N + ROCKE_WMMA_N - 1) / ROCKE_WMMA_N;
    out[1] = (M + ROCKE_WMMA_M - 1) / ROCKE_WMMA_M;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_wmma_gemm_gfx1250_lower_to_llvm(const rocke_wmma_gemm_gfx1250_spec_t* spec,
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

    kernel = rocke_build_wmma_gemm_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_wmma_gemm_gfx1250 failed";
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
