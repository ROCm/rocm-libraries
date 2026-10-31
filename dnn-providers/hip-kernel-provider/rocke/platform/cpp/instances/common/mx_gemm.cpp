// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_mx_gemm.c -- C99 port of rocke/instances/common/mx_gemm.py
 * (build_mx_gemm + is_valid_spec + grid + spec helpers).
 *
 * The build entry mirrors the Python build_mx_gemm one op at a time so a
 * reviewer can diff line by line: same param order/attrs, same K-group
 * scf.for_iter, same per-group E8M0 scale-fold, same store epilogue.
 *
 * Builder primitives are rocke/ir.h's rocke_b_*; the MX scale decode, quant type
 * map, MFMA atom, and GEMM load/store helpers are the already-ported C helpers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>
#include <vector>

#include "rocke/instance_mx_gemm.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.mx_scale.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */
#include "rocke/lower_llvm.h"

/* Python _SCALE_LOADS_FIRST: sched_barrier mask, every instruction class may
 * be scheduled across the barrier except VMEM (0x10) and VMEM read (0x20). */
#define ROCKE_MX_GEMM_SCALE_LOADS_FIRST 0x7CF
/* Python _E8M0_CODE_FOR_ONE: the E8M0 scale code of 2^0. */
#define ROCKE_MX_GEMM_E8M0_CODE_FOR_ONE 127

/* ===================================================================== *
 *  mx_gemm_element_format(mantissa_dtype, arch)
 *
 * Python _MFMA_ELEMENT_FORMATS: the fp8 / bf8 encoding the MFMA of each
 * target reads, the FNUZ encodings on gfx942 and the OCP ones on gfx950.
 * ===================================================================== */
const rocke_mx_element_format_t* rocke_mx_gemm_element_format(const char* mantissa_dtype,
                                                              const char* arch,
                                                              char* err,
                                                              size_t err_cap)
{
    bool fp8;
    bool bf8;

    if(mantissa_dtype == NULL || arch == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "mx_gemm_element_format: NULL argument");
        }
        return NULL;
    }
    fp8 = strcmp(mantissa_dtype, "fp8e4m3") == 0;
    bf8 = strcmp(mantissa_dtype, "bf8e5m2") == 0;
    if(strcmp(arch, "gfx942") == 0 && (fp8 || bf8))
    {
        return fp8 ? &ROCKE_FP8_E4M3_FNUZ : &ROCKE_BF8_E5M2_FNUZ;
    }
    if(strcmp(arch, "gfx950") == 0 && (fp8 || bf8))
    {
        return fp8 ? &ROCKE_FP8_E4M3_OCP : &ROCKE_BF8_E5M2_OCP;
    }
    if(err != NULL && err_cap > 0)
    {
        snprintf(err,
                 err_cap,
                 "mx_gemm: no %s element format known for %s; the MX GEMM reads "
                 "fp8e4m3 / bf8e5m2 on gfx942 and gfx950",
                 mantissa_dtype,
                 arch);
    }
    return NULL;
}

/* ===================================================================== *
 *  rocke_mx_gemm_spec_default -- Python MxGemmSpec dataclass defaults.
 * ===================================================================== */
rocke_mx_gemm_spec_t rocke_mx_gemm_spec_default(void)
{
    rocke_mx_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.mantissa_dtype = "fp8e4m3";
    s.group_k = 32;
    s.block_tile_m = 16;
    s.block_tile_n = 16;
    s.name = "rocke_mx_gemm";
    s.per_input_row = true;
    return s;
}

/* ===================================================================== *
 *  MxGemmSpec.block_size property: one wave64 warp per CTA = 64.
 * ===================================================================== */
int rocke_mx_gemm_block_size(const rocke_mx_gemm_spec_t* spec)
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
 * The atoms-port exposes the fp8/bf8 16x16x32 atoms via rocke_mfma_atom() over the
 * static catalog: fp8e4m3/bf8e5m2 (16,16,32) both exist as entries.
 * ===================================================================== */
const struct rocke_mfma_atom* rocke_mx_gemm_atom(const rocke_mx_gemm_spec_t* spec)
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
        return rocke_mfma_atom("fp8e4m3", 16, 16, 32);
    }
    return rocke_mfma_atom("bf8e5m2", 16, 16, 32);
}

/* ===================================================================== *
 *  MxGemmSpec.kernel_name()
 *
 *     kernel_name_join(name, f"M{M}N{N}K{K}", mantissa_dtype,
 *                      f"gk{group_k}", f"t{block_tile_m}x{block_tile_n}")
 * ===================================================================== */
rocke_status_t
    rocke_mx_gemm_kernel_name(const rocke_mx_gemm_spec_t* spec, char* out, size_t out_cap)
{
    char shape[64];
    char gk[32];
    char tile[48];
    const char* parts[4];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(shape, sizeof(shape), "M%dN%dK%d", spec->M, spec->N, spec->K);
    snprintf(gk, sizeof(gk), "gk%d", spec->group_k);
    snprintf(tile, sizeof(tile), "t%dx%d", spec->block_tile_m, spec->block_tile_n);

    parts[0] = shape;
    parts[1] = spec->mantissa_dtype;
    parts[2] = gk;
    parts[3] = tile;

    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch) -> (ok, reason)
 *
 * Mirrors validate_arch_and_block_size (arch resolve + block-size cap) plus the
 * MX-specific checks. The reason strings surface only through ValueError
 * messages (never into IR), so the exact spelling is non-load-bearing for
 * byte-identical emitted code; we keep them Python-shaped for parity.
 * ===================================================================== */
static void rocke_mxg_setreason(char* reason, size_t cap, const char* msg)
{
    rocke_spec_set_reason(reason, cap, msg);
}

bool rocke_mx_gemm_is_valid_spec(const rocke_mx_gemm_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    const rocke_archtarget_t* target;
    int block_size;
    char buf[160];

    if(spec == NULL)
    {
        rocke_mxg_setreason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* validate_arch_and_block_size: ArchTarget.from_gfx(arch) + block cap. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target %s", arch);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    block_size = rocke_mx_gemm_block_size(spec);
    if(block_size > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d > %d (hardware cap) on %s",
                 block_size,
                 rocke_archtarget_max_threads_per_block(target),
                 arch);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }

    if(strcmp(spec->mantissa_dtype, "fp8e4m3") != 0 && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "unsupported mantissa_dtype %s; v1 ships fp8e4m3 / bf8e5m2 only "
                 "(fp4 / fp6 are v2)",
                 spec->mantissa_dtype);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->group_k != 32)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MX spec requires group_k = 32 (shared exponent per 32 mantissa "
                 "elements); got group_k=%d",
                 spec->group_k);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->K % spec->group_k)
    {
        snprintf(
            buf, sizeof(buf), "K (%d) must be divisible by group_k (%d)", spec->K, spec->group_k);
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }
    if(spec->M % spec->block_tile_m || spec->N % spec->block_tile_n)
    {
        rocke_mxg_setreason(
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
        rocke_mxg_setreason(reason, reason_cap, buf);
        return false;
    }

    rocke_mxg_setreason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_mx_gemm(spec, arch)
 *
 * One op at a time mirroring the Python. The builder `b` is assumed already
 * initialised by the caller with spec.kernel_name() (public-header contract).
 * ===================================================================== */
rocke_kernel_def_t*
    rocke_build_mx_gemm(rocke_ir_builder_t* b, const rocke_mx_gemm_spec_t* spec, const char* arch)
{
    char reason[160];
    const rocke_type_t* mantissa_ty;
    const rocke_mfma_atom_t* atom;
    int BS;
    int k_scale_count;
    int atoms_per_group;
    int kt_local;

    rocke_value_t* A;
    rocke_value_t* AScale;
    rocke_value_t* Bp;
    rocke_value_t* BScale;
    rocke_value_t* C;

    rocke_value_t* lane;
    rocke_value_t* bid_n;
    rocke_value_t* bid_m;
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_lane_decode_t lane_decode;
    rocke_value_t* m_global_row;
    rocke_value_t* n_global_col;
    rocke_value_t* a_scale_row_base;

    rocke_value_t* unit_scale_code;
    rocke_for_t outer;
    rocke_value_t* acc_final;
    const rocke_mx_element_format_t* element_format;
    rocke_mx_scale_split_t scale_split;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch=arch); if not ok: raise ValueError */
    if(!rocke_mx_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid mx_gemm spec for %s: %s", arch, reason);
    }

    atom = rocke_mx_gemm_atom(spec);
    if(atom == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(b,
                                                    ROCKE_ERR_VALUE,
                                                    "mx_gemm: no MFMA atom for mantissa_dtype %s",
                                                    spec->mantissa_dtype);
    }
    /* validate_mfma_atom_in_catalog(spec.atom, arch, where="mx_gemm") */
    if(rocke_validate_mfma_atom_in_catalog(b, atom, arch, "mx_gemm") != ROCKE_OK)
    {
        return NULL;
    }

    /* mantissa_ty = quant_ir_type(spec.mantissa_dtype) */
    mantissa_ty = rocke_b_quant_ir_type(b, spec->mantissa_dtype);
    if(mantissa_ty == NULL)
    {
        return NULL;
    }
    BS = rocke_mx_gemm_block_size(spec);
    /* The split that makes each group's scale update exact, derived from the
     * bounds of a group's dot product in the elements the MFMA reads:
     *     element_format = mx_gemm_element_format(spec.mantissa_dtype, arch)
     *     scale_split = exact_mx_scale_split(element_format, element_format,
     *                                        spec.group_k) */
    element_format
        = rocke_mx_gemm_element_format(spec->mantissa_dtype, arch, reason, sizeof(reason));
    if(element_format == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
    }
    {
        char split_err[512];
        if(rocke_exact_mx_scale_split(element_format,
                                      element_format,
                                      spec->group_k,
                                      &scale_split,
                                      split_err,
                                      sizeof(split_err))
           != ROCKE_OK)
        {
            return (rocke_kernel_def_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", split_err);
        }
    }

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* ---- kernel params (mirror Python order/attrs) ---- */
    {
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_mant = rocke_ptr_type(b, mantissa_ty, "global");
        const rocke_type_t* ptr_i8 = rocke_ptr_type(b, rocke_i8(), "global");
        const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

        /* A = b.param("A", PtrType(mantissa_ty,"global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", ptr_mant, &opts);

        /* AScale = b.param("AScale", PtrType(I8,"global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 1;
        opts.align_set = true;
        AScale = rocke_b_param(b, "AScale", ptr_i8, &opts);

        /* B = b.param("B", PtrType(mantissa_ty,"global"), readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = rocke_b_param(b, "B", ptr_mant, &opts);

        /* BScale = b.param("BScale", PtrType(I8,"global"), readonly=True, align=1) */
        memset(&opts, 0, sizeof(opts));
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 1;
        opts.align_set = true;
        BScale = rocke_b_param(b, "BScale", ptr_i8, &opts);

        /* C = b.param("C", PtrType(F32,"global"), writeonly=True, align=4) */
        memset(&opts, 0, sizeof(opts));
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 4;
        opts.align_set = true;
        C = rocke_b_param(b, "C", ptr_f32, &opts);

        /* _M = b.param("M", I32); _N; _K  (ABI scalars) */
        (void)rocke_b_param(b, "M", rocke_i32(), NULL);
        (void)rocke_b_param(b, "N", rocke_i32(), NULL);
        (void)rocke_b_param(b, "K", rocke_i32(), NULL);
    }

    /* lane = b.thread_id_x(); bid_n = b.block_id_x(); bid_m = b.block_id_y() */
    lane = rocke_b_thread_id_x(b);
    bid_n = rocke_b_block_id_x(b);
    bid_m = rocke_b_block_id_y(b);

    /* m_tile_base = b.mul(bid_m, b.const_i32(block_tile_m)) */
    m_tile_base = rocke_b_mul(b, bid_m, rocke_b_const_i32(b, spec->block_tile_m));
    /* n_tile_base = b.mul(bid_n, b.const_i32(block_tile_n)) */
    n_tile_base = rocke_b_mul(b, bid_n, rocke_b_const_i32(b, spec->block_tile_n));

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = rocke_decode_mfma_lanes(b, atom, lane);

    /* if spec.group_k % atom.k != 0: raise ValueError */
    if(spec->group_k % atom->k != 0)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "MX group_k (%d) must be a multiple of atom.k (%d) so the per-group "
            "scale apply aligns with whole MFMA invocations",
            spec->group_k,
            atom->k);
    }
    k_scale_count = spec->K / spec->group_k;
    atoms_per_group = spec->group_k / atom->k;

    /* Loop-invariant scale-address bases (hoisted out of the K-group loop). */
    /* m_global_row = b.add(m_tile_base, lane_decode.m_in_atom) */
    m_global_row = rocke_b_add(b, m_tile_base, lane_decode.m_in_atom);
    /* n_global_col = b.add(n_tile_base, lane_decode.n_in_atom) */
    n_global_col = rocke_b_add(b, n_tile_base, lane_decode.n_in_atom);
    /* a_scale_row_base = b.mul(m_global_row, b.const_i32(k_scale_count)) */
    a_scale_row_base = rocke_b_mul(b, m_global_row, rocke_b_const_i32(b, k_scale_count));

    /* The fold runs one group late (see the Python comment): iteration kg
     * issues its loads, folds group kg - 1 (prev_group with its two scale
     * codes) while they are in flight, then runs its own MFMA; the last group
     * is folded after the loop.
     *
     * unit_scale_code = b.trunc(b.const_i32(_E8M0_CODE_FOR_ONE), I8) */
    unit_scale_code
        = rocke_b_trunc(b, rocke_b_const_i32(b, ROCKE_MX_GEMM_E8M0_CODE_FOR_ONE), rocke_i8());

    /* outer = b.scf_for_iter(0, k_scale_count, 1,
     *     [("oacc", atom.zero_acc(b)), ("prev_group", atom.zero_acc(b)),
     *      ("prev_a_code", unit_scale_code), ("prev_b_code", unit_scale_code)],
     *     iv_name="kg")
     *
     * Python evaluates the call arguments left-to-right: the three
     * ``const_i32`` bounds (lb/ub/step) are emitted *before* the iter_args
     * list, whose two zero-vector inits come next. Each ``const_i32`` and
     * zero-vector consumes one SSA value-counter tick, so the order is
     * load-bearing for byte-identical value numbering. */
    {
        rocke_value_t* lb = rocke_b_const_i32(b, 0);
        rocke_value_t* ub = rocke_b_const_i32(b, k_scale_count);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        rocke_iter_arg_t iter_args[4];
        iter_args[0].name = "oacc";
        iter_args[0].init = rocke_b_zero_vec_f32(b, atom->c_per_lane); /* atom.zero_acc(b) */
        iter_args[1].name = "prev_group";
        iter_args[1].init = rocke_b_zero_vec_f32(b, atom->c_per_lane); /* atom.zero_acc(b) */
        iter_args[2].name = "prev_a_code";
        iter_args[2].init = unit_scale_code;
        iter_args[3].name = "prev_b_code";
        iter_args[3].init = unit_scale_code;
        /* Python scf_for_iter signature defaults elide_trailing_barrier=True
         * (unroll=False); build_mx_gemm relies on those defaults. The C wrapper
         * takes them as explicit trailing args, so pass unroll=false,
         * elide_trailing_barrier=true to match the emitted scf.for attribute. */
        outer = rocke_b_scf_for_iter(b, lb, ub, step, iter_args, 4, "kg", false, true);
    }

    /* with outer as (kg, (outer_acc, prev_group, prev_a_code, prev_b_code)): */
    rocke_b_region_enter(b, outer.body);
    {
        rocke_value_t* kg = outer.iv;
        rocke_value_t* outer_acc = outer.iter_vars[0];
        rocke_value_t* prev_group = outer.iter_vars[1];
        rocke_value_t* prev_a_code = outer.iter_vars[2];
        rocke_value_t* prev_b_code = outer.iter_vars[3];

        rocke_value_t* a_scale_off;
        rocke_value_t* b_scale_off;
        rocke_value_t* a_scale_code;
        rocke_value_t* b_scale_code;
        rocke_value_t* k_group_base;
        rocke_value_t* group_acc;
        rocke_value_t* new_outer;
        /* operands = [] (one (a_vec, b_vec) pair per MFMA atom) */
        std::vector<std::pair<rocke_value_t*, rocke_value_t*>> operands;

        /* a_scale_off = b.add(a_scale_row_base, kg) */
        a_scale_off = rocke_b_add(b, a_scale_row_base, kg);
        /* b_scale_off = b.add(b.mul(kg, b.const_i32(N)), n_global_col) */
        b_scale_off
            = rocke_b_add(b, rocke_b_mul(b, kg, rocke_b_const_i32(b, spec->N)), n_global_col);

        /* a_scale_code = b.global_load_i8(AScale, a_scale_off) */
        a_scale_code = rocke_b_global_load_i8(b, AScale, a_scale_off, 0);
        /* b_scale_code = b.global_load_i8(BScale, b_scale_off) */
        b_scale_code = rocke_b_global_load_i8(b, BScale, b_scale_off, 0);
        /* b.sched_barrier(_SCALE_LOADS_FIRST) */
        rocke_b_sched_barrier(b, ROCKE_MX_GEMM_SCALE_LOADS_FIRST);

        /* k_group_base = b.mul(kg, b.const_i32(group_k)) */
        k_group_base = rocke_b_mul(b, kg, rocke_b_const_i32(b, spec->group_k));

        /* for kt_local in range(atoms_per_group): */
        for(kt_local = 0; kt_local < atoms_per_group; ++kt_local)
        {
            /* k_tile_base = b.add(k_group_base, b.const_i32(kt_local * atom.k)) */
            rocke_value_t* k_tile_base
                = rocke_b_add(b, k_group_base, rocke_b_const_i32(b, kt_local * atom->k));
            rocke_value_t* a_vec;
            rocke_value_t* b_vec;

            /* a_vec = load_a_row_major_contiguous(b, A=A, atom, lane_decode,
             *             m_tile_base, k_tile_base, K=spec.K) */
            a_vec = rocke_load_a_row_major_contiguous(
                b, A, atom, &lane_decode, m_tile_base, k_tile_base, spec->K);
            /* b_vec = load_b_col_strided_scalars(b, B=Bp, atom, lane_decode,
             *             n_tile_base, k_tile_base, N=spec.N) */
            b_vec = rocke_load_b_col_strided_scalars(
                b, Bp, atom, &lane_decode, n_tile_base, k_tile_base, spec->N);
            /* operands.append((a_vec, b_vec)) */
            operands.emplace_back(a_vec, b_vec);
        }

        /* new_outer = fma_with_mx_scale_pair(b, prev_group, prev_a_code,
         *                                    prev_b_code, outer_acc,
         *                                    split=scale_split) */
        new_outer = rocke_fma_with_mx_scale_pair(
            b, prev_group, prev_a_code, prev_b_code, outer_acc, &scale_split);

        /* group_acc = atom.zero_acc(b) */
        group_acc = rocke_b_zero_vec_f32(b, atom->c_per_lane);
        /* for a_vec, b_vec in operands:
         *     group_acc = atom.emit(b, a_vec, b_vec, group_acc) -> b.mma(name,...) */
        for(const auto& [a_vec, b_vec] : operands)
        {
            group_acc = rocke_b_mma(b, atom->name, a_vec, b_vec, group_acc, NULL, 0);
        }
        /* b.scf_yield(new_outer, group_acc, a_scale_code, b_scale_code) */
        {
            rocke_value_t* yvals[4];
            yvals[0] = new_outer;
            yvals[1] = group_acc;
            yvals[2] = a_scale_code;
            yvals[3] = b_scale_code;
            rocke_b_scf_yield(b, yvals, 4);
        }
    }
    rocke_b_region_leave(b);

    /* outer_acc, last_group, last_a_code, last_b_code = outer.results
     * acc_final = fma_with_mx_scale_pair(b, last_group, last_a_code,
     *                                    last_b_code, outer_acc) */
    if(!rocke_i_live(b) || outer.op == NULL || outer.op->num_results < 4)
    {
        return NULL;
    }
    acc_final = rocke_fma_with_mx_scale_pair(b,
                                             outer.op->results[1],
                                             outer.op->results[2],
                                             outer.op->results[3],
                                             outer.op->results[0],
                                             &scale_split);

    /* store_acc_to_global(b, C=C, atom, lane_decode, m_tile_base, n_tile_base,
     *                     acc=acc_final, N=spec.N, out_dtype="f32") */
    rocke_store_acc_to_global(b,
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
    rocke_b_ret(b);
    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_mx_gemm_new -- init the builder with spec.kernel_name(), then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_mx_gemm_new(rocke_ir_builder_t* b,
                                            const rocke_mx_gemm_spec_t* spec,
                                            const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_mx_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_mx_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_mx_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_mx_gemm_lower_to_llvm(const rocke_mx_gemm_spec_t* spec,
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
            rocke_mxg_setreason(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_mx_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_mx_gemm failed";
            }
            rocke_mxg_setreason(err, err_cap, m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  mx_gemm_grid(spec) -> (n_tiles, m_tiles, 1)
 * ===================================================================== */
void rocke_mx_gemm_grid(const rocke_mx_gemm_spec_t* spec, int* out_gx, int* out_gy, int* out_gz)
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
