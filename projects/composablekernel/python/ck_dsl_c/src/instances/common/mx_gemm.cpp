// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_mx_gemm.c -- C99 port of ck_dsl/instances/common/mx_gemm.py
 * (build_mx_gemm + is_valid_spec + grid + spec helpers).
 *
 * The build entry mirrors the Python build_mx_gemm one op at a time so a
 * reviewer can diff line by line: same param order/attrs, same K-group
 * scf.for_iter, same per-group E8M0 decode + scale-fold, same store epilogue.
 *
 * Builder primitives are ckc/ir.h's ckc_b_*; the MX scale decode, quant type
 * map, MFMA atom, and GEMM load/store helpers are the already-ported C helpers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/instance_mx_gemm.h"

#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/helper_ck_dsl.helpers.atoms.h"
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h"
#include "ckc/helper_ck_dsl.helpers.mx_scale.h"
#include "ckc/helper_ck_dsl.helpers.quant.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */
#include "ckc/lower_llvm.h"
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  ckc_mx_gemm_spec_default -- Python MxGemmSpec dataclass defaults.
 * ===================================================================== */
ckc_mx_gemm_spec_t ckc_mx_gemm_spec_default(void)
{
    ckc_mx_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M              = 0;
    s.N              = 0;
    s.K              = 0;
    s.mantissa_dtype = "fp8e4m3";
    s.group_k        = 32;
    s.block_tile_m   = 16;
    s.block_tile_n   = 16;
    s.name           = "ck_dsl_mx_gemm";
    s.per_input_row  = true;
    return s;
}

/* ===================================================================== *
 *  MxGemmSpec.block_size property: one wave64 warp per CTA = 64.
 * ===================================================================== */
int ckc_mx_gemm_block_size(const ckc_mx_gemm_spec_t* spec)
{
    (void)spec;
    return 64;
}

/* ===================================================================== *
 *  MxGemmSpec.atom property.
 *
 *     if (block_tile_m, block_tile_n) != (16, 16): raise ValueError
 *     return MfmaAtom.fp8_16x16x32() if mantissa_dtype == "fp8e4m3"
 *            else MfmaAtom.bf8_16x16x32()
 *
 * The atoms-port exposes the fp8/bf8 16x16x32 atoms via ckc_mfma_atom() over the
 * static catalog: fp8e4m3/bf8e5m2 (16,16,32) both exist as entries.
 * ===================================================================== */
const struct ckc_mfma_atom* ckc_mx_gemm_atom(const ckc_mx_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(spec->block_tile_m != 16 || spec->block_tile_n != 16)
    {
        return NULL; /* Python ValueError: 16x16 tiles only */
    }
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") == 0)
    {
        return ckc_mfma_atom("fp8e4m3", 16, 16, 32);
    }
    return ckc_mfma_atom("bf8e5m2", 16, 16, 32);
}

/* ===================================================================== *
 *  MxGemmSpec.kernel_name()
 *
 *     kernel_name_join(name, f"M{M}N{N}K{K}", mantissa_dtype,
 *                      f"gk{group_k}", f"t{block_tile_m}x{block_tile_n}")
 * ===================================================================== */
ckc_status_t ckc_mx_gemm_kernel_name(const ckc_mx_gemm_spec_t* spec, char* out, size_t out_cap)
{
    char shape[64];
    char gk[32];
    char tile[48];
    const char* parts[4];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }
    snprintf(shape, sizeof(shape), "M%dN%dK%d", spec->M, spec->N, spec->K);
    snprintf(gk, sizeof(gk), "gk%d", spec->group_k);
    snprintf(tile, sizeof(tile), "t%dx%d", spec->block_tile_m, spec->block_tile_n);

    parts[0] = shape;
    parts[1] = spec->mantissa_dtype;
    parts[2] = gk;
    parts[3] = tile;

    return ckc_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason)
 *
 * Mirrors validate_arch_and_block_size (arch resolve + block-size cap) plus the
 * MX-specific checks. The reason strings surface only through ValueError
 * messages (never into IR), so the exact spelling is non-load-bearing for
 * byte-identical emitted code; we keep them Python-shaped for parity.
 * ===================================================================== */
static void ckc_mxg_setreason(char* reason, size_t cap, const char* msg)
{
    ckc_spec_set_reason(reason, cap, msg);
}

bool ckc_mx_gemm_is_valid_spec(const ckc_mx_gemm_spec_t* spec,
                               const char* arch,
                               char* reason,
                               size_t reason_cap)
{
    const ckc_archtarget_t* target;
    int block_size;
    char buf[160];

    if(spec == NULL)
    {
        ckc_mxg_setreason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* validate_arch_and_block_size: ArchTarget.from_gfx(arch) + block cap. */
    target = ckc_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s", arch);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    block_size = ckc_mx_gemm_block_size(spec);
    if(block_size > ckc_archtarget_max_threads_per_block(target))
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d > %d (hardware cap) on %s",
                 block_size,
                 ckc_archtarget_max_threads_per_block(target),
                 arch);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }

    if(strcmp(spec->mantissa_dtype, "fp8e4m3") != 0 && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unsupported mantissa_dtype %s; v1 ships fp8e4m3 / bf8e5m2 only "
                 "(fp4 / fp6 are v2)",
                 spec->mantissa_dtype);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->group_k != 32)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MX spec requires group_k = 32 (shared exponent per 32 mantissa "
                 "elements); got group_k=%d",
                 spec->group_k);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->K % spec->group_k)
    {
        snprintf(
            buf, sizeof(buf), "K (%d) must be divisible by group_k (%d)", spec->K, spec->group_k);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->M % spec->block_tile_m || spec->N % spec->block_tile_n)
    {
        ckc_mxg_setreason(
            reason, reason_cap, "M / N must divide their tile sizes (v1 no partial tiles)");
        return false;
    }
    if(spec->block_tile_m != 16 || spec->block_tile_n != 16)
    {
        snprintf(buf,
                 sizeof(buf),
                 "mx_gemm MFMA path supports 16x16 tiles only (got %dx%d)",
                 spec->block_tile_m,
                 spec->block_tile_n);
        ckc_mxg_setreason(reason, reason_cap, buf);
        return false;
    }

    ckc_mxg_setreason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_mx_gemm(spec, arch)
 *
 * One op at a time mirroring the Python. The builder `b` is assumed already
 * initialised by the caller with spec.kernel_name() (public-header contract).
 * ===================================================================== */
ckc_kernel_def_t*
ckc_build_mx_gemm(ckc_ir_builder_t* b, const ckc_mx_gemm_spec_t* spec, const char* arch)
{
    char reason[160];
    const ckc_type_t* mantissa_ty;
    const ckc_mfma_atom_t* atom;
    int BS;
    int k_scale_count;
    int atoms_per_group;
    int kt_local;

    ckc_value_t* A;
    ckc_value_t* AScale;
    ckc_value_t* Bp;
    ckc_value_t* BScale;
    ckc_value_t* C;

    ckc_value_t* lane;
    ckc_value_t* bid_n;
    ckc_value_t* bid_m;
    ckc_value_t* m_tile_base;
    ckc_value_t* n_tile_base;
    ckc_lane_decode_t lane_decode;
    ckc_value_t* m_global_row;
    ckc_value_t* n_global_col;
    ckc_value_t* a_scale_row_base;

    ckc_iter_arg_t iter_arg;
    ckc_for_t outer;
    ckc_value_t* acc_final;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch=arch); if not ok: raise ValueError */
    if(!ckc_mx_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid mx_gemm spec for %s: %s", arch, reason);
    }

    atom = ckc_mx_gemm_atom(spec);
    if(atom == NULL)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "mx_gemm: no MFMA atom for mantissa_dtype %s", spec->mantissa_dtype);
    }
    /* validate_mfma_atom_in_catalog(spec.atom, arch, where="mx_gemm") */
    if(ckc_validate_mfma_atom_in_catalog(b, atom, arch, "mx_gemm") != CKC_OK)
    {
        return NULL;
    }

    /* mantissa_ty = quant_ir_type(spec.mantissa_dtype) */
    mantissa_ty = ckc_b_quant_ir_type(b, spec->mantissa_dtype);
    if(mantissa_ty == NULL)
    {
        return NULL;
    }
    BS = ckc_mx_gemm_block_size(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* ---- kernel params (mirror Python order/attrs) ---- */
    {
        ckc_param_opts_t opts;
        const ckc_type_t* ptr_mant = ckc_ptr_type(b, mantissa_ty, "global");
        const ckc_type_t* ptr_i8   = ckc_ptr_type(b, ckc_i8(), "global");
        const ckc_type_t* ptr_f32  = ckc_ptr_type(b, ckc_f32(), "global");

        /* A = b.param("A", PtrType(mantissa_ty,"global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 16;
        opts.align_set    = true;
        A                 = ckc_b_param(b, "A", ptr_mant, &opts);

        /* AScale = b.param("AScale", PtrType(I8,"global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 1;
        opts.align_set    = true;
        AScale            = ckc_b_param(b, "AScale", ptr_i8, &opts);

        /* B = b.param("B", PtrType(mantissa_ty,"global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 16;
        opts.align_set    = true;
        Bp                = ckc_b_param(b, "B", ptr_mant, &opts);

        /* BScale = b.param("BScale", PtrType(I8,"global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 1;
        opts.align_set    = true;
        BScale            = ckc_b_param(b, "BScale", ptr_i8, &opts);

        /* C = b.param("C", PtrType(F32,"global"), writeonly=True, align=4) */
        memset(&opts, 0, sizeof(opts));
        opts.writeonly     = true;
        opts.writeonly_set = true;
        opts.align         = 4;
        opts.align_set     = true;
        C                  = ckc_b_param(b, "C", ptr_f32, &opts);

        /* _M = b.param("M", I32); _N; _K  (ABI scalars) */
        (void)ckc_b_param(b, "M", ckc_i32(), NULL);
        (void)ckc_b_param(b, "N", ckc_i32(), NULL);
        (void)ckc_b_param(b, "K", ckc_i32(), NULL);
    }

    /* lane = b.thread_id_x(); bid_n = b.block_id_x(); bid_m = b.block_id_y() */
    lane  = ckc_b_thread_id_x(b);
    bid_n = ckc_b_block_id_x(b);
    bid_m = ckc_b_block_id_y(b);

    /* m_tile_base = b.mul(bid_m, b.const_i32(block_tile_m)) */
    m_tile_base = ckc_b_mul(b, bid_m, ckc_b_const_i32(b, spec->block_tile_m));
    /* n_tile_base = b.mul(bid_n, b.const_i32(block_tile_n)) */
    n_tile_base = ckc_b_mul(b, bid_n, ckc_b_const_i32(b, spec->block_tile_n));

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = ckc_decode_mfma_lanes(b, atom, lane);

    /* if spec.group_k % atom.k != 0: raise ValueError */
    if(spec->group_k % atom->k != 0)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b,
            CKC_ERR_VALUE,
            "MX group_k (%d) must be a multiple of atom.k (%d) so the per-group "
            "scale apply aligns with whole MFMA invocations",
            spec->group_k,
            atom->k);
    }
    k_scale_count   = spec->K / spec->group_k;
    atoms_per_group = spec->group_k / atom->k;

    /* Loop-invariant scale-address bases (hoisted out of the K-group loop). */
    /* m_global_row = b.add(m_tile_base, lane_decode.m_in_atom) */
    m_global_row = ckc_b_add(b, m_tile_base, lane_decode.m_in_atom);
    /* n_global_col = b.add(n_tile_base, lane_decode.n_in_atom) */
    n_global_col = ckc_b_add(b, n_tile_base, lane_decode.n_in_atom);
    /* a_scale_row_base = b.mul(m_global_row, b.const_i32(k_scale_count)) */
    a_scale_row_base = ckc_b_mul(b, m_global_row, ckc_b_const_i32(b, k_scale_count));

    /* outer = b.scf_for_iter(0, k_scale_count, 1, [("oacc", atom.zero_acc(b))],
     *                        iv_name="kg")
     *
     * Python evaluates the call arguments left-to-right: the three
     * ``const_i32`` bounds (lb/ub/step) are emitted *before* the iter_args
     * list, whose only entry's init is ``atom.zero_acc(b)`` (the
     * zero-vector). Each ``const_i32`` and the zero-vector consume one SSA
     * value-counter tick, so the order is load-bearing for byte-identical
     * value numbering -- the three constants MUST be emitted before the
     * zero-vector accumulator. */
    {
        ckc_value_t* lb   = ckc_b_const_i32(b, 0);
        ckc_value_t* ub   = ckc_b_const_i32(b, k_scale_count);
        ckc_value_t* step = ckc_b_const_i32(b, 1);
        iter_arg.name     = "oacc";
        iter_arg.init     = ckc_b_zero_vec_f32(b, atom->c_per_lane); /* atom.zero_acc(b) */
        /* Python scf_for_iter signature defaults elide_trailing_barrier=True
         * (unroll=False); build_mx_gemm relies on those defaults. The C wrapper
         * takes them as explicit trailing args, so pass unroll=false,
         * elide_trailing_barrier=true to match the emitted scf.for attribute. */
        outer = ckc_b_scf_for_iter(b, lb, ub, step, &iter_arg, 1, "kg", false, true);
    }

    /* with outer as (kg, (outer_acc,)): */
    ckc_b_region_enter(b, outer.body);
    {
        ckc_value_t* kg        = outer.iv;
        ckc_value_t* outer_acc = outer.iter_vars[0];

        ckc_value_t* a_scale_off;
        ckc_value_t* b_scale_off;
        ckc_value_t* a_scale;
        ckc_value_t* b_scale;
        ckc_value_t* ab_scale;
        ckc_value_t* k_group_base;
        ckc_value_t* group_acc;
        ckc_value_t* ab_scale_vec;
        ckc_value_t* new_outer;

        /* a_scale_off = b.add(a_scale_row_base, kg) */
        a_scale_off = ckc_b_add(b, a_scale_row_base, kg);
        /* b_scale_off = b.add(b.mul(kg, b.const_i32(N)), n_global_col) */
        b_scale_off = ckc_b_add(b, ckc_b_mul(b, kg, ckc_b_const_i32(b, spec->N)), n_global_col);

        /* a_scale = decode_mx_scale_e8m0(b, b.global_load(AScale, a_scale_off, I8, align=1)) */
        a_scale =
            ckc_decode_mx_scale_e8m0(b, ckc_b_global_load(b, AScale, a_scale_off, ckc_i8(), 1));
        /* b_scale = decode_mx_scale_e8m0(b, b.global_load(BScale, b_scale_off, I8, align=1)) */
        b_scale =
            ckc_decode_mx_scale_e8m0(b, ckc_b_global_load(b, BScale, b_scale_off, ckc_i8(), 1));
        /* ab_scale = b.fmul(a_scale, b_scale) */
        ab_scale = ckc_b_fmul(b, a_scale, b_scale);

        /* k_group_base = b.mul(kg, b.const_i32(group_k)) */
        k_group_base = ckc_b_mul(b, kg, ckc_b_const_i32(b, spec->group_k));

        /* group_acc = atom.zero_acc(b) */
        group_acc = ckc_b_zero_vec_f32(b, atom->c_per_lane);
        /* for kt_local in range(atoms_per_group): */
        for(kt_local = 0; kt_local < atoms_per_group; ++kt_local)
        {
            /* k_tile_base = b.add(k_group_base, b.const_i32(kt_local * atom.k)) */
            ckc_value_t* k_tile_base =
                ckc_b_add(b, k_group_base, ckc_b_const_i32(b, kt_local * atom->k));
            ckc_value_t* a_vec;
            ckc_value_t* b_vec;

            /* a_vec = load_a_row_major_contiguous(b, A=A, atom, lane_decode,
             *             m_tile_base, k_tile_base, K=spec.K) */
            a_vec = ckc_load_a_row_major_contiguous(
                b, A, atom, &lane_decode, m_tile_base, k_tile_base, spec->K);
            /* b_vec = load_b_col_strided_scalars(b, B=Bp, atom, lane_decode,
             *             n_tile_base, k_tile_base, N=spec.N) */
            b_vec = ckc_load_b_col_strided_scalars(
                b, Bp, atom, &lane_decode, n_tile_base, k_tile_base, spec->N);
            /* group_acc = atom.emit(b, a_vec, b_vec, group_acc) -> b.mma(name,...) */
            group_acc = ckc_b_mma(b, atom->name, a_vec, b_vec, group_acc, NULL, 0);
        }

        /* ab_scale_vec = b.vector_splat(ab_scale, atom.c_per_lane) */
        ab_scale_vec = ckc_b_vector_splat(b, ab_scale, atom->c_per_lane);
        /* new_outer = b.vector_fma(group_acc, ab_scale_vec, outer_acc) */
        new_outer = ckc_b_vector_fma(b, group_acc, ab_scale_vec, outer_acc);
        /* b.scf_yield(new_outer) */
        {
            ckc_value_t* yvals[1];
            yvals[0] = new_outer;
            ckc_b_scf_yield(b, yvals, 1);
        }
    }
    ckc_b_region_leave(b);

    /* acc_final = outer.results[0] */
    acc_final = (outer.op != NULL && outer.op->num_results > 0) ? outer.op->results[0] : NULL;

    /* store_acc_to_global(b, C=C, atom, lane_decode, m_tile_base, n_tile_base,
     *                     acc=acc_final, N=spec.N, out_dtype="f32") */
    ckc_store_acc_to_global(b,
                            C,
                            atom,
                            &lane_decode,
                            m_tile_base,
                            n_tile_base,
                            acc_final,
                            spec->N,
                            "f32",
                            /*atomic_add=*/false,
                            /*epilogue=*/NULL,
                            /*user=*/NULL);

    /* b.ret() */
    ckc_b_ret(b);
    return b->kernel;
}

/* ===================================================================== *
 *  ckc_build_mx_gemm_new -- init the builder with spec.kernel_name(), then build.
 * ===================================================================== */
ckc_kernel_def_t*
ckc_build_mx_gemm_new(ckc_ir_builder_t* b, const ckc_mx_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_mx_gemm_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_mx_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  ckc_mx_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
ckc_status_t ckc_mx_gemm_lower_to_llvm(const ckc_mx_gemm_spec_t* spec,
                                       const char* arch,
                                       ckc_llvm_flavor_t flavor,
                                       char** out_ll,
                                       char* err,
                                       size_t err_cap)
{
    ckc_ir_builder_t b;
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
            ckc_mxg_setreason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_mx_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_mx_gemm failed";
            }
            ckc_mxg_setreason(err, err_cap, m);
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  mx_gemm_grid(spec) -> (n_tiles, m_tiles, 1)
 * ===================================================================== */
void ckc_mx_gemm_grid(const ckc_mx_gemm_spec_t* spec, int* out_gx, int* out_gy, int* out_gz)
{
    int n_tiles;
    int m_tiles;
    if(spec == NULL)
    {
        return;
    }
    n_tiles = (spec->N + spec->block_tile_n - 1) / spec->block_tile_n;
    m_tiles = (spec->M + spec->block_tile_m - 1) / spec->block_tile_m;
    if(out_gx != NULL)
    {
        *out_gx = n_tiles;
    }
    if(out_gy != NULL)
    {
        *out_gy = m_tiles;
    }
    if(out_gz != NULL)
    {
        *out_gz = 1;
    }
}
