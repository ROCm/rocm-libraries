/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_block_scale_gemm.c -- C99 port of
 * ck_dsl/instances/common/block_scale_gemm.py.
 *
 * Byte-identical builder-call sequence vs the Python build_block_scale_gemm.
 * See instance_block_scale_gemm.h for the symbol map.
 */
#include "ckc/instance_block_scale_gemm.h"

#include <stdio.h>
#include <string.h>

#include "ckc/arena.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */
#include "ckc/helper_ck_dsl.helpers.atoms.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h"
#include "ckc/helper_ck_dsl.helpers.quant.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"

/* ===================================================================== *
 *  spec defaults / new
 * ===================================================================== */
ckc_block_scale_gemm_spec_t ckc_block_scale_gemm_spec_default(void)
{
    ckc_block_scale_gemm_spec_t s;
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.quant_mode = "bquant";
    s.mantissa_dtype = "fp8e4m3";
    s.preshuffle_b = false;
    s.group_m = 1;
    s.group_n = 1;
    s.group_k = 128;
    s.block_tile_m = 16;
    s.block_tile_n = 16;
    s.name = "ck_dsl_block_scale_gemm";
    s.per_input_row = true;
    return s;
}

ckc_block_scale_gemm_spec_t ckc_block_scale_gemm_spec_new(int M,
                                                          int N,
                                                          int K,
                                                          const char* quant_mode,
                                                          const char* mantissa_dtype,
                                                          int group_m,
                                                          int group_n,
                                                          int group_k)
{
    ckc_block_scale_gemm_spec_t s = ckc_block_scale_gemm_spec_default();
    s.M = M;
    s.N = N;
    s.K = K;
    if (quant_mode != NULL)
    {
        s.quant_mode = quant_mode;
    }
    if (mantissa_dtype != NULL)
    {
        s.mantissa_dtype = mantissa_dtype;
    }
    s.group_m = group_m;
    s.group_n = group_n;
    s.group_k = group_k;
    return s;
}

/* ===================================================================== *
 *  spec.atom (@property)
 *
 *  if (block_tile_m, block_tile_n) != (16, 16): raise ValueError
 *  fp8e4m3 -> MfmaAtom.fp8_16x16x32()  == mfma_atom("fp8", 16,16,32)
 *  bf8e5m2 -> MfmaAtom.bf8_16x16x32()  == mfma_atom("bf8", 16,16,32)
 *  i4_fp8  -> fp8 atom ; i4_bf8 -> bf8 atom
 * ===================================================================== */
const ckc_mfma_atom_t*
ckc_block_scale_gemm_spec_atom(const ckc_block_scale_gemm_spec_t* spec)
{
    if (spec == NULL)
    {
        return NULL;
    }
    if (spec->block_tile_m != 16 || spec->block_tile_n != 16)
    {
        return NULL; /* Python ValueError (16x16 tiles only) */
    }
    if (strcmp(spec->mantissa_dtype, "fp8e4m3") == 0)
    {
        return ckc_mfma_atom("fp8", 16, 16, 32);
    }
    if (strcmp(spec->mantissa_dtype, "bf8e5m2") == 0)
    {
        return ckc_mfma_atom("bf8", 16, 16, 32);
    }
    if (strcmp(spec->mantissa_dtype, "i4_fp8") == 0)
    {
        return ckc_mfma_atom("fp8", 16, 16, 32);
    }
    if (strcmp(spec->mantissa_dtype, "i4_bf8") == 0)
    {
        return ckc_mfma_atom("bf8", 16, 16, 32);
    }
    return NULL; /* Python: no atom for mantissa */
}

int ckc_block_scale_gemm_spec_block_size(const ckc_block_scale_gemm_spec_t* spec)
{
    (void)spec;
    return 64;
}

/* ===================================================================== *
 *  spec.kernel_name()
 *
 *  kernel_name_join(self.name,
 *      f"M{M}N{N}K{K}", quant_mode, mantissa_dtype,
 *      f"g{gm}x{gn}x{gk}", f"t{tm}x{tn}",
 *      flags={"psb": preshuffle_b})
 * ===================================================================== */
ckc_status_t ckc_block_scale_gemm_kernel_name(const ckc_block_scale_gemm_spec_t* spec,
                                              char* out,
                                              size_t out_cap)
{
    char mnk[96];
    char gpart[96];
    char tpart[64];
    const char* parts[5];
    const char* flag_names[1];
    int flag_on[1];

    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    if (snprintf(mnk, sizeof(mnk), "M%dN%dK%d", spec->M, spec->N, spec->K) < 0)
    {
        return CKC_ERR_VALUE;
    }
    if (snprintf(gpart, sizeof(gpart), "g%dx%dx%d", spec->group_m, spec->group_n,
                 spec->group_k) < 0)
    {
        return CKC_ERR_VALUE;
    }
    if (snprintf(tpart, sizeof(tpart), "t%dx%d", spec->block_tile_m, spec->block_tile_n) < 0)
    {
        return CKC_ERR_VALUE;
    }

    parts[0] = mnk;
    parts[1] = spec->quant_mode;
    parts[2] = spec->mantissa_dtype;
    parts[3] = gpart;
    parts[4] = tpart;

    flag_names[0] = "psb";
    flag_on[0] = spec->preshuffle_b ? 1 : 0;

    return ckc_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap,
                                NULL);
}

/* ===================================================================== *
 *  _mantissa_storage_dtype / _storage_ir_type
 * ===================================================================== */
const char* ckc_block_scale_gemm_mantissa_store(const ckc_block_scale_gemm_spec_t* spec)
{
    if (spec == NULL)
    {
        return NULL;
    }
    if (strcmp(spec->mantissa_dtype, "fp8e4m3") == 0
        || strcmp(spec->mantissa_dtype, "bf8e5m2") == 0)
    {
        return spec->mantissa_dtype;
    }
    /* i4_fp8 / i4_bf8: packed two-per-byte storage. */
    return "i8";
}

const ckc_type_t* ckc_block_scale_gemm_storage_ir_type(const char* store)
{
    if (store == NULL)
    {
        return NULL;
    }
    if (strcmp(store, "f16") == 0)
    {
        return ckc_io_ir_type("f16");
    }
    if (strcmp(store, "i8") == 0)
    {
        return ckc_i8();
    }
    return ckc_quant_ir_type(store);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */
static void ckc_i_write_reason(char* reason, size_t reason_cap, const char* msg)
{
    size_t n;
    if (reason == NULL || reason_cap == 0 || msg == NULL)
    {
        return;
    }
    n = strlen(msg);
    if (n >= reason_cap)
    {
        n = reason_cap - 1;
    }
    memcpy(reason, msg, n);
    reason[n] = '\0';
}

bool ckc_block_scale_gemm_is_valid_spec(ckc_ir_builder_t* b,
                                        const ckc_block_scale_gemm_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    int bs;
    const char* arch_reason = NULL;
    const ckc_archtarget_t* target = NULL;
    char buf[160];
    int i;
    int gms[3];

    if (spec == NULL)
    {
        ckc_i_write_reason(reason, reason_cap, "null spec");
        return false;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    bs = ckc_block_scale_gemm_spec_block_size(spec);

    /* validate_arch_and_block_size(arch, spec.block_size). The reason string is
     * either a static "ok"/"unknown gfx..." or an arena-owned formatted cap
     * message; `b` provides the arena. */
    if (!ckc_validate_arch_and_block_size(b, arch, bs, &arch_reason, &target))
    {
        ckc_i_write_reason(reason, reason_cap,
                           arch_reason != NULL ? arch_reason : "invalid arch/block_size");
        return false;
    }
    (void)target;

    /* quant_mode in ("aquant","bquant","abquant") */
    if (strcmp(spec->quant_mode, "aquant") != 0 && strcmp(spec->quant_mode, "bquant") != 0
        && strcmp(spec->quant_mode, "abquant") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported quant_mode %s", spec->quant_mode);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* MFMA block_scale_gemm currently ships quant_mode='abquant' only */
    if (strcmp(spec->quant_mode, "abquant") != 0)
    {
        snprintf(buf, sizeof(buf),
                 "MFMA block_scale_gemm currently ships quant_mode='abquant' only; got %s",
                 spec->quant_mode);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* mantissa_dtype in (fp8e4m3,bf8e5m2,i4_fp8,i4_bf8) */
    if (strcmp(spec->mantissa_dtype, "fp8e4m3") != 0
        && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0
        && strcmp(spec->mantissa_dtype, "i4_fp8") != 0
        && strcmp(spec->mantissa_dtype, "i4_bf8") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported mantissa_dtype %s", spec->mantissa_dtype);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* currently ships fp8e4m3 / bf8e5m2 mantissas only */
    if (strcmp(spec->mantissa_dtype, "fp8e4m3") != 0
        && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0)
    {
        snprintf(buf, sizeof(buf),
                 "MFMA block_scale_gemm currently ships fp8e4m3 / bf8e5m2 mantissas only; "
                 "got %s",
                 spec->mantissa_dtype);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    if (spec->preshuffle_b)
    {
        ckc_i_write_reason(reason, reason_cap,
                           "preshuffle_b=True requires the MFMA-based kernel body "
                           "( follow-on); v1 ships the scalar inner only");
        return false;
    }
    if (bs > 1024)
    {
        snprintf(buf, sizeof(buf), "block_size %d > 1024 hardware cap", bs);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* any(g <= 0 for g in group_size_mnk) */
    gms[0] = spec->group_m;
    gms[1] = spec->group_n;
    gms[2] = spec->group_k;
    for (i = 0; i < 3; ++i)
    {
        if (gms[i] <= 0)
        {
            snprintf(buf, sizeof(buf), "group_size_mnk must be positive, got (%d, %d, %d)",
                     spec->group_m, spec->group_n, spec->group_k);
            ckc_i_write_reason(reason, reason_cap, buf);
            return false;
        }
    }
    /* K % gk */
    if (spec->group_k != 0 && (spec->K % spec->group_k) != 0)
    {
        snprintf(buf, sizeof(buf), "K (%d) must be divisible by group_k (%d)", spec->K,
                 spec->group_k);
        ckc_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* M % block_tile_m or N % block_tile_n */
    if ((spec->block_tile_m != 0 && (spec->M % spec->block_tile_m) != 0)
        || (spec->block_tile_n != 0 && (spec->N % spec->block_tile_n) != 0))
    {
        ckc_i_write_reason(reason, reason_cap,
                           "M / N must be divisible by their tile sizes "
                           "(v1 doesn't handle partial tiles)");
        return false;
    }

    ckc_i_write_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_block_scale_gemm
 *
 *  Closures _load_a_in_group / _load_b_in_group capture the build context;
 *  in C that environment is this struct, threaded through the load callbacks'
 *  `user` pointer.
 * ===================================================================== */
typedef struct ckc_bsg_load_ctx
{
    ckc_value_t* A;
    ckc_value_t* Bp;
    const ckc_mfma_atom_t* atom;
    const ckc_lane_decode_t* lane_decode;
    ckc_value_t* m_tile_base;
    ckc_value_t* n_tile_base;
    ckc_value_t* k_group_base;
    ckc_value_t* c_atom_k;
    int K;
    int N;
} ckc_bsg_load_ctx_t;

/* _load_a_in_group(b, kt_local): k_tile_base = base + kt_local*c_atom_k. */
static ckc_value_t* ckc_bsg_load_a_in_group(ckc_ir_builder_t* b,
                                            ckc_value_t* kt_local,
                                            void* user)
{
    ckc_bsg_load_ctx_t* c = (ckc_bsg_load_ctx_t*)user;
    ckc_value_t* k_tile_base =
        ckc_b_add(b, c->k_group_base, ckc_b_mul(b, kt_local, c->c_atom_k));
    return ckc_load_a_row_major_contiguous(b, c->A, c->atom, c->lane_decode, c->m_tile_base,
                                           k_tile_base, c->K);
}

/* _load_b_in_group(b, kt_local): k_tile_base = base + kt_local*c_atom_k. */
static ckc_value_t* ckc_bsg_load_b_in_group(ckc_ir_builder_t* b,
                                            ckc_value_t* kt_local,
                                            void* user)
{
    ckc_bsg_load_ctx_t* c = (ckc_bsg_load_ctx_t*)user;
    ckc_value_t* k_tile_base =
        ckc_b_add(b, c->k_group_base, ckc_b_mul(b, kt_local, c->c_atom_k));
    return ckc_load_b_col_strided_scalars(b, c->Bp, c->atom, c->lane_decode, c->n_tile_base,
                                          k_tile_base, c->N);
}

ckc_kernel_def_t* ckc_build_block_scale_gemm(ckc_ir_builder_t* b,
                                             const ckc_block_scale_gemm_spec_t* spec,
                                             const char* arch)
{
    char reason[160];
    const ckc_mfma_atom_t* atom;
    const char* mantissa_store;
    int BS;
    const char* a_store;
    const char* b_store;
    ckc_value_t* A;
    ckc_value_t* Bp;
    ckc_value_t* AScale = NULL;
    ckc_value_t* BScale = NULL;
    ckc_value_t* C;
    ckc_value_t* lane;
    ckc_value_t* bid_n;
    ckc_value_t* bid_m;
    ckc_value_t* m_tile_base;
    ckc_value_t* n_tile_base;
    ckc_lane_decode_t lane_decode;
    int gm, gn, gk;
    int n_scale_count_b;
    int k_scale_count_a;
    int num_groups;
    ckc_value_t* c_atom_k;
    ckc_param_opts_t opts;
    const ckc_type_t* a_store_ty;
    const ckc_type_t* b_store_ty;
    ckc_iter_arg_t outer_args[1];
    ckc_value_t* loop_lb;
    ckc_value_t* loop_ub;
    ckc_value_t* loop_step;
    ckc_for_t outer;
    ckc_value_t* acc_final;
    ckc_bsg_load_ctx_t lctx;

    if (!ckc_i_live(b))
    {
        return NULL;
    }
    if (spec == NULL)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(b, CKC_ERR_VALUE,
                                                "build_block_scale_gemm: null spec");
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    if (!ckc_block_scale_gemm_is_valid_spec(b, spec, arch, reason, sizeof(reason)))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "invalid block_scale_gemm spec for %s: %s", arch, reason);
    }

    atom = ckc_block_scale_gemm_spec_atom(spec);
    if (atom == NULL)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(b, CKC_ERR_VALUE,
                                                "block_scale_gemm: no MFMA atom for spec");
    }
    /* validate_mfma_atom_in_catalog(spec.atom, arch, where="block_scale_gemm") */
    if (ckc_validate_mfma_atom_in_catalog(b, atom, arch, "block_scale_gemm") != CKC_OK)
    {
        return NULL;
    }

    mantissa_store = ckc_block_scale_gemm_mantissa_store(spec);
    BS = ckc_block_scale_gemm_spec_block_size(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    if (ckc_i_live(b) && b->kernel != NULL)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);
    }

    /* a_store = mantissa_store if quant_mode in (aquant,abquant) else "f16"
     * b_store = mantissa_store if quant_mode in (bquant,abquant) else "f16" */
    a_store = (strcmp(spec->quant_mode, "aquant") == 0
               || strcmp(spec->quant_mode, "abquant") == 0)
                  ? mantissa_store
                  : "f16";
    b_store = (strcmp(spec->quant_mode, "bquant") == 0
               || strcmp(spec->quant_mode, "abquant") == 0)
                  ? mantissa_store
                  : "f16";

    a_store_ty = ckc_block_scale_gemm_storage_ir_type(a_store);
    b_store_ty = ckc_block_scale_gemm_storage_ir_type(b_store);
    if (a_store_ty == NULL || b_store_ty == NULL)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(b, CKC_ERR_VALUE,
                                                "block_scale_gemm: bad storage dtype");
    }

    /* A = b.param("A", PtrType(_storage_ir_type(a_store),"global"),
     *             noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    A = ckc_b_param(b, "A", ckc_ptr_type(b, a_store_ty, "global"), &opts);
    Bp = ckc_b_param(b, "B", ckc_ptr_type(b, b_store_ty, "global"), &opts);

    /* Scale pointers per quant_mode. */
    if (strcmp(spec->quant_mode, "aquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        ckc_param_opts_t sopts;
        memset(&sopts, 0, sizeof(sopts));
        sopts.readonly = true;
        sopts.readonly_set = true;
        sopts.align = 4;
        sopts.align_set = true;
        AScale = ckc_b_param(b, "AScale", ckc_ptr_type(b, ckc_f32(), "global"), &sopts);
    }
    if (strcmp(spec->quant_mode, "bquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        ckc_param_opts_t sopts;
        memset(&sopts, 0, sizeof(sopts));
        sopts.readonly = true;
        sopts.readonly_set = true;
        sopts.align = 4;
        sopts.align_set = true;
        BScale = ckc_b_param(b, "BScale", ckc_ptr_type(b, ckc_f32(), "global"), &sopts);
    }

    /* C = b.param("C", PtrType(F32,"global"), writeonly=True, align=4) */
    {
        ckc_param_opts_t copts;
        memset(&copts, 0, sizeof(copts));
        copts.writeonly = true;
        copts.writeonly_set = true;
        copts.align = 4;
        copts.align_set = true;
        C = ckc_b_param(b, "C", ckc_ptr_type(b, ckc_f32(), "global"), &copts);
    }

    /* M/N/K i32 ABI params (unused values, like Python's noqa F841). */
    (void)ckc_b_param(b, "M", ckc_i32(), NULL);
    (void)ckc_b_param(b, "N", ckc_i32(), NULL);
    (void)ckc_b_param(b, "K", ckc_i32(), NULL);

    lane = ckc_b_thread_id_x(b);
    bid_n = ckc_b_block_id_x(b);
    bid_m = ckc_b_block_id_y(b);
    m_tile_base = ckc_b_mul(b, bid_m, ckc_b_const_i32(b, spec->block_tile_m));
    n_tile_base = ckc_b_mul(b, bid_n, ckc_b_const_i32(b, spec->block_tile_n));

    /* if quant_mode != "abquant": raise NotImplementedError */
    if (strcmp(spec->quant_mode, "abquant") != 0)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_NOTIMPL,
            "MFMA block_scale_gemm v1 ships abquant only; aquant/bquant "
            "variants are a follow-on (same MFMA inner, asymmetric scale apply).");
    }
    /* if a_store != b_store or a_store not in (fp8e4m3,bf8e5m2): raise */
    if (strcmp(a_store, b_store) != 0
        || (strcmp(a_store, "fp8e4m3") != 0 && strcmp(a_store, "bf8e5m2") != 0))
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_NOTIMPL,
            "MFMA path needs A and B in fp8e4m3 or bf8e5m2 (got A=%s, B=%s)", a_store,
            b_store);
    }

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = ckc_decode_mfma_lanes(b, atom, lane);

    gm = spec->group_m;
    gn = spec->group_n;
    gk = spec->group_k;
    /* if gk % atom.k != 0: raise ValueError */
    if (atom->k == 0 || (gk % atom->k) != 0)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE,
            "group_k (%d) must be a multiple of atom.k (%d) so the per-group scale "
            "apply aligns with a whole number of MFMA invocations",
            gk, atom->k);
    }

    n_scale_count_b = (spec->N + gn - 1) / gn;
    k_scale_count_a = (spec->K + gk - 1) / gk;

    num_groups = spec->K / gk;
    c_atom_k = ckc_b_const_i32(b, atom->k);

    /* outer = b.scf_for_iter(0, num_groups, 1, [("acc", atom.zero_acc(b))], "kg")
     *   atom.zero_acc(b) -> b.zero_vec_f32(atom.c_per_lane)
     * Python evaluates scf_for_iter's positional args left-to-right, so the
     * three bound constants (0, num_groups, 1) are emitted BEFORE the
     * zero_vec inside the iter-arg list. Emit them first here so the global
     * value counter for the zero_vec matches Python (C arg-eval order is
     * unspecified). */
    loop_lb = ckc_b_const_i32(b, 0);
    loop_ub = ckc_b_const_i32(b, num_groups);
    loop_step = ckc_b_const_i32(b, 1);
    outer_args[0].name = "acc";
    outer_args[0].init = ckc_b_zero_vec_f32(b, atom->c_per_lane);
    outer = ckc_b_scf_for_iter(b, loop_lb, loop_ub, loop_step, outer_args, 1, "kg",
                               /*unroll=*/false, /*elide_trailing_barrier=*/false);

    ckc_b_region_enter(b, outer.body);
    {
        ckc_value_t* kg = outer.iv;
        ckc_value_t* outer_acc = outer.iter_vars[0];
        ckc_value_t* a_scale_off;
        ckc_value_t* b_scale_off;
        ckc_value_t* a_scale_v;
        ckc_value_t* b_scale_v;
        ckc_value_t* ab_scale;
        ckc_value_t* k_group_base;
        ckc_value_t* a_scale_inner_add;
        ckc_value_t* a_scale_div;
        ckc_value_t* a_scale_mul;
        ckc_value_t* b_scale_mul;
        ckc_value_t* b_scale_inner_add;
        ckc_value_t* b_scale_div;
        ckc_value_t* group_acc;
        ckc_value_t* ab_scale_vec;
        ckc_value_t* scaled_group;
        ckc_value_t* new_outer;
        ckc_value_t* yield_vals[1];

        /* a_scale_off = ((m_tile_base + m_in_atom)/gm) * k_scale_count_a + kg
         * Python evaluates the nested b.add/b.mul/b.div args left-to-right, so
         * the emission order is: add(m_tile_base,m_in_atom), const(gm), div,
         * const(k_scale_count_a), mul, then kg (already a value). C arg-eval
         * order is unspecified, so bind each sub-expression to a temporary in
         * that exact order to match Python's value numbering. */
        a_scale_inner_add = ckc_b_add(b, m_tile_base, lane_decode.m_in_atom);
        a_scale_div = ckc_b_div(b, a_scale_inner_add, ckc_b_const_i32(b, gm));
        a_scale_mul = ckc_b_mul(b, a_scale_div, ckc_b_const_i32(b, k_scale_count_a));
        a_scale_off = ckc_b_add(b, a_scale_mul, kg);
        /* b_scale_off = kg * n_scale_count_b + ((n_tile_base + n_in_atom)/gn)
         * Python evaluates b.add's args left-to-right, so the mul (kg *
         * n_scale_count_b) is emitted BEFORE the div(add) operand. C arg
         * evaluation order is unspecified, so bind the mul to a temporary
         * first to force the same emission order. */
        b_scale_mul = ckc_b_mul(b, kg, ckc_b_const_i32(b, n_scale_count_b));
        b_scale_inner_add = ckc_b_add(b, n_tile_base, lane_decode.n_in_atom);
        b_scale_div = ckc_b_div(b, b_scale_inner_add, ckc_b_const_i32(b, gn));
        b_scale_off = ckc_b_add(b, b_scale_mul, b_scale_div);

        a_scale_v = ckc_b_global_load_f32(b, AScale, a_scale_off, /*align=*/0);
        b_scale_v = ckc_b_global_load_f32(b, BScale, b_scale_off, /*align=*/0);
        ab_scale = ckc_b_fmul(b, a_scale_v, b_scale_v);

        /* k_group_base = kg * gk */
        k_group_base = ckc_b_mul(b, kg, ckc_b_const_i32(b, gk));

        /* group_acc = mfma_k_loop(b, K=gk, atom, _load_a_in_group,
         *     _load_b_in_group, iv_name="kk", acc_name="gacc") */
        lctx.A = A;
        lctx.Bp = Bp;
        lctx.atom = atom;
        lctx.lane_decode = &lane_decode;
        lctx.m_tile_base = m_tile_base;
        lctx.n_tile_base = n_tile_base;
        lctx.k_group_base = k_group_base;
        lctx.c_atom_k = c_atom_k;
        lctx.K = spec->K;
        lctx.N = spec->N;

        group_acc = ckc_mfma_k_loop(b, gk, atom, ckc_bsg_load_a_in_group,
                                    ckc_bsg_load_b_in_group, /*per_tile_post_mfma=*/NULL,
                                    /*initial_acc=*/NULL, "kk", "gacc", &lctx);

        /* ab_scale_vec = b.vector_splat(ab_scale, atom.c_per_lane)
         * scaled_group = b.vector_mul(group_acc, ab_scale_vec)
         * new_outer    = b.vector_add(outer_acc, scaled_group) */
        ab_scale_vec = ckc_b_vector_splat(b, ab_scale, atom->c_per_lane);
        scaled_group = ckc_b_vector_mul(b, group_acc, ab_scale_vec);
        new_outer = ckc_b_vector_add(b, outer_acc, scaled_group);

        yield_vals[0] = new_outer;
        ckc_b_scf_yield(b, yield_vals, 1);
    }
    ckc_b_region_leave(b);

    /* acc_final = outer.results[0] */
    if (!ckc_i_live(b) || outer.op == NULL || outer.op->num_results < 1)
    {
        return NULL;
    }
    acc_final = outer.op->results[0];

    /* store_acc_to_global(b, C, atom, lane_decode, m_tile_base, n_tile_base,
     *     acc_final, N=spec.N, out_dtype="f32") */
    if (ckc_store_acc_to_global(b, C, atom, &lane_decode, m_tile_base, n_tile_base, acc_final,
                                spec->N, "f32", /*atomic_add=*/false, /*epilogue=*/NULL,
                                NULL)
        != CKC_OK)
    {
        return NULL;
    }

    ckc_b_ret(b);
    if (!ckc_i_live(b))
    {
        return NULL;
    }
    return b->kernel;
}

ckc_kernel_def_t* ckc_build_block_scale_gemm_new(ckc_ir_builder_t* b,
                                                 const ckc_block_scale_gemm_spec_t* spec,
                                                 const char* arch)
{
    char name[256];

    if (b == NULL || spec == NULL)
    {
        return NULL;
    }
    if (ckc_block_scale_gemm_kernel_name(spec, name, sizeof(name)) != CKC_OK)
    {
        return NULL;
    }
    if (ckc_ir_builder_init(b, name) != CKC_OK)
    {
        return NULL;
    }
    return ckc_build_block_scale_gemm(b, spec, arch);
}

/* ===================================================================== *
 *  block_scale_gemm_grid
 * ===================================================================== */
ckc_status_t ckc_block_scale_gemm_grid(const ckc_block_scale_gemm_spec_t* spec, int out[3])
{
    if (spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    out[0] = (spec->N + spec->block_tile_n - 1) / spec->block_tile_n;
    out[1] = (spec->M + spec->block_tile_m - 1) / spec->block_tile_m;
    out[2] = 1;
    return CKC_OK;
}

/* ===================================================================== *
 *  block_scale_gemm_signature
 *
 *  a_dtype = "f16" if quant_mode=="bquant"
 *            else ("i8" if mantissa.startswith("i4_") else mantissa)
 *  b_dtype = "f16" if quant_mode=="aquant"
 *            else ("i8" if mantissa.startswith("i4_") else mantissa)
 *  sb.ptr(A,a).ptr(B,b)[.ptr(AScale,f32)][.ptr(BScale,f32)]
 *    .ptr(C,f32).scalar(M,i32).scalar(N,i32).scalar(K,i32)
 * ===================================================================== */
static const char* ckc_bsg_side_dtype(const ckc_block_scale_gemm_spec_t* spec,
                                       const char* f16_when_mode)
{
    if (strcmp(spec->quant_mode, f16_when_mode) == 0)
    {
        return "f16";
    }
    if (strncmp(spec->mantissa_dtype, "i4_", 3) == 0)
    {
        return "i8";
    }
    return spec->mantissa_dtype;
}

ckc_status_t ckc_block_scale_gemm_signature(struct ckc_arena* arena,
                                            const ckc_block_scale_gemm_spec_t* spec,
                                            const ckc_sig_entry_t** out_items,
                                            size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;
    const char* a_dtype;
    const char* b_dtype;

    if (arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return CKC_ERR_VALUE;
    }

    st = ckc_signature_builder_init(&sb, arena);
    if (st != CKC_OK)
    {
        return st;
    }

    a_dtype = ckc_bsg_side_dtype(spec, "bquant");
    b_dtype = ckc_bsg_side_dtype(spec, "aquant");

    ckc_signature_builder_ptr(&sb, "A", a_dtype, NULL);
    ckc_signature_builder_ptr(&sb, "B", b_dtype, NULL);
    if (strcmp(spec->quant_mode, "aquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        ckc_signature_builder_ptr(&sb, "AScale", "f32", NULL);
    }
    if (strcmp(spec->quant_mode, "bquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        ckc_signature_builder_ptr(&sb, "BScale", "f32", NULL);
    }
    ckc_signature_builder_ptr(&sb, "C", "f32", NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  lower_to_llvm convenience
 * ===================================================================== */
ckc_status_t ckc_block_scale_gemm_lower_to_llvm(const ckc_block_scale_gemm_spec_t* spec,
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
        ckc_i_write_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if (arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_block_scale_gemm_new(&b, spec, arch);
    if (kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if (err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
            ckc_i_write_reason(err, err_cap,
                               (m != NULL) ? m : "build_block_scale_gemm failed");
        }
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
