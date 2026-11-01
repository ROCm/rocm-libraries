// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_block_scale_gemm.c -- C99 port of
 * rocke/instances/common/block_scale_gemm.py.
 *
 * Byte-identical builder-call sequence vs the Python build_block_scale_gemm.
 * See instance_block_scale_gemm.h for the symbol map.
 */
#include "rocke/instance_block_scale_gemm.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

/* ===================================================================== *
 *  spec defaults / new
 * ===================================================================== */
rocke_block_scale_gemm_spec_t rocke_block_scale_gemm_spec_default(void)
{
    rocke_block_scale_gemm_spec_t s;
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
    s.name = "rocke_block_scale_gemm";
    return s;
}

rocke_block_scale_gemm_spec_t rocke_block_scale_gemm_spec_new(int M,
                                                              int N,
                                                              int K,
                                                              const char* quant_mode,
                                                              const char* mantissa_dtype,
                                                              int group_m,
                                                              int group_n,
                                                              int group_k)
{
    rocke_block_scale_gemm_spec_t s = rocke_block_scale_gemm_spec_default();
    s.M = M;
    s.N = N;
    s.K = K;
    if(quant_mode != NULL)
    {
        s.quant_mode = quant_mode;
    }
    if(mantissa_dtype != NULL)
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
const rocke_mfma_atom_t* rocke_block_scale_gemm_spec_atom(const rocke_block_scale_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(spec->block_tile_m != 16 || spec->block_tile_n != 16)
    {
        return NULL; /* Python ValueError (16x16 tiles only) */
    }
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") == 0)
    {
        return rocke_mfma_atom("fp8", 16, 16, 32);
    }
    if(strcmp(spec->mantissa_dtype, "bf8e5m2") == 0)
    {
        return rocke_mfma_atom("bf8", 16, 16, 32);
    }
    if(strcmp(spec->mantissa_dtype, "i4_fp8") == 0)
    {
        return rocke_mfma_atom("fp8", 16, 16, 32);
    }
    if(strcmp(spec->mantissa_dtype, "i4_bf8") == 0)
    {
        return rocke_mfma_atom("bf8", 16, 16, 32);
    }
    return NULL; /* Python: no atom for mantissa */
}

int rocke_block_scale_gemm_spec_block_size(const rocke_block_scale_gemm_spec_t* spec)
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
rocke_status_t rocke_block_scale_gemm_kernel_name(const rocke_block_scale_gemm_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char mnk[96];
    char gpart[96];
    char tpart[64];
    const char* parts[5];
    const char* flag_names[1];
    int flag_on[1];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    if(snprintf(mnk, sizeof(mnk), "M%dN%dK%d", spec->M, spec->N, spec->K) < 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(snprintf(gpart, sizeof(gpart), "g%dx%dx%d", spec->group_m, spec->group_n, spec->group_k) < 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(snprintf(tpart, sizeof(tpart), "t%dx%d", spec->block_tile_m, spec->block_tile_n) < 0)
    {
        return ROCKE_ERR_VALUE;
    }

    parts[0] = mnk;
    parts[1] = spec->quant_mode;
    parts[2] = spec->mantissa_dtype;
    parts[3] = gpart;
    parts[4] = tpart;

    flag_names[0] = "psb";
    flag_on[0] = spec->preshuffle_b ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 5, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  _mantissa_storage_dtype / _storage_ir_type
 * ===================================================================== */
const char* rocke_block_scale_gemm_mantissa_store(const rocke_block_scale_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") == 0 || strcmp(spec->mantissa_dtype, "bf8e5m2") == 0)
    {
        return spec->mantissa_dtype;
    }
    /* i4_fp8 / i4_bf8: packed two-per-byte storage. */
    return "i8";
}

const rocke_type_t* rocke_block_scale_gemm_storage_ir_type(const char* store)
{
    if(store == NULL)
    {
        return NULL;
    }
    if(strcmp(store, "f16") == 0)
    {
        return rocke_io_ir_type("f16");
    }
    if(strcmp(store, "i8") == 0)
    {
        return rocke_i8();
    }
    return rocke_quant_ir_type(store);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */
static void rocke_i_write_reason(char* reason, size_t reason_cap, const char* msg)
{
    size_t n;
    if(reason == NULL || reason_cap == 0 || msg == NULL)
    {
        return;
    }
    n = strlen(msg);
    if(n >= reason_cap)
    {
        n = reason_cap - 1;
    }
    memcpy(reason, msg, n);
    reason[n] = '\0';
}

bool rocke_block_scale_gemm_is_valid_spec(rocke_ir_builder_t* b,
                                          const rocke_block_scale_gemm_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    int bs;
    const char* arch_reason = NULL;
    const rocke_archtarget_t* target = NULL;
    char buf[160];
    int i;
    int gms[3];

    if(spec == NULL)
    {
        rocke_i_write_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    bs = rocke_block_scale_gemm_spec_block_size(spec);

    /* validate_arch_and_block_size(arch, spec.block_size). The reason string is
     * either a static "ok"/"unknown gfx..." or an arena-owned formatted cap
     * message; `b` provides the arena. */
    if(!rocke_validate_arch_and_block_size(b, arch, bs, &arch_reason, &target))
    {
        rocke_i_write_reason(
            reason, reason_cap, arch_reason != NULL ? arch_reason : "invalid arch/block_size");
        return false;
    }
    (void)target;

    /* quant_mode in ("aquant","bquant","abquant") */
    if(strcmp(spec->quant_mode, "aquant") != 0 && strcmp(spec->quant_mode, "bquant") != 0
       && strcmp(spec->quant_mode, "abquant") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported quant_mode %s", spec->quant_mode);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* MFMA block_scale_gemm currently ships quant_mode='abquant' only */
    if(strcmp(spec->quant_mode, "abquant") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA block_scale_gemm currently ships quant_mode='abquant' only; got %s",
                 spec->quant_mode);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* mantissa_dtype in (fp8e4m3,bf8e5m2,i4_fp8,i4_bf8) */
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") != 0 && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0
       && strcmp(spec->mantissa_dtype, "i4_fp8") != 0
       && strcmp(spec->mantissa_dtype, "i4_bf8") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported mantissa_dtype %s", spec->mantissa_dtype);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* currently ships fp8e4m3 / bf8e5m2 mantissas only */
    if(strcmp(spec->mantissa_dtype, "fp8e4m3") != 0 && strcmp(spec->mantissa_dtype, "bf8e5m2") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA block_scale_gemm currently ships fp8e4m3 / bf8e5m2 mantissas only; "
                 "got %s",
                 spec->mantissa_dtype);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->preshuffle_b)
    {
        rocke_i_write_reason(reason,
                             reason_cap,
                             "preshuffle_b=True requires the MFMA-based kernel body "
                             "( follow-on); v1 ships the scalar inner only");
        return false;
    }
    if(bs > 1024)
    {
        snprintf(buf, sizeof(buf), "block_size %d > 1024 hardware cap", bs);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* any(g <= 0 for g in group_size_mnk) */
    gms[0] = spec->group_m;
    gms[1] = spec->group_n;
    gms[2] = spec->group_k;
    for(i = 0; i < 3; ++i)
    {
        if(gms[i] <= 0)
        {
            snprintf(buf,
                     sizeof(buf),
                     "group_size_mnk must be positive, got (%d, %d, %d)",
                     spec->group_m,
                     spec->group_n,
                     spec->group_k);
            rocke_i_write_reason(reason, reason_cap, buf);
            return false;
        }
    }
    /* K % gk */
    if(spec->group_k != 0 && (spec->K % spec->group_k) != 0)
    {
        snprintf(
            buf, sizeof(buf), "K (%d) must be divisible by group_k (%d)", spec->K, spec->group_k);
        rocke_i_write_reason(reason, reason_cap, buf);
        return false;
    }
    /* M % block_tile_m or N % block_tile_n */
    if((spec->block_tile_m != 0 && (spec->M % spec->block_tile_m) != 0)
       || (spec->block_tile_n != 0 && (spec->N % spec->block_tile_n) != 0))
    {
        rocke_i_write_reason(reason,
                             reason_cap,
                             "M / N must be divisible by their tile sizes "
                             "(v1 doesn't handle partial tiles)");
        return false;
    }

    rocke_i_write_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_block_scale_gemm
 *
 *  Closures _load_a_in_group / _load_b_in_group capture the build context;
 *  in C that environment is this struct, threaded through the load callbacks'
 *  `user` pointer.
 * ===================================================================== */

/* The most accumulator slots per lane the builder handles (the 16x16 atom
 * holds 4). */
#define ROCKE_BSG_MAX_C_PER_LANE 4

typedef struct rocke_bsg_load_ctx
{
    rocke_value_t* A;
    rocke_value_t* Bp;
    const rocke_mfma_atom_t* atom;
    const rocke_lane_decode_t* lane_decode;
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_value_t* k_group_base;
    rocke_value_t* c_atom_k;
    int K;
    int N;
} rocke_bsg_load_ctx_t;

/* _load_a_in_group(b, kt_local): k_tile_base = base + kt_local*c_atom_k. */
static rocke_value_t*
    rocke_bsg_load_a_in_group(rocke_ir_builder_t* b, rocke_value_t* kt_local, void* user)
{
    rocke_bsg_load_ctx_t* c = (rocke_bsg_load_ctx_t*)user;
    rocke_value_t* k_tile_base
        = rocke_b_add(b, c->k_group_base, rocke_b_mul(b, kt_local, c->c_atom_k));
    return rocke_load_a_row_major_contiguous(
        b, c->A, c->atom, c->lane_decode, c->m_tile_base, k_tile_base, c->K);
}

/* _load_b_in_group(b, kt_local): k_tile_base = base + kt_local*c_atom_k. */
static rocke_value_t*
    rocke_bsg_load_b_in_group(rocke_ir_builder_t* b, rocke_value_t* kt_local, void* user)
{
    rocke_bsg_load_ctx_t* c = (rocke_bsg_load_ctx_t*)user;
    rocke_value_t* k_tile_base
        = rocke_b_add(b, c->k_group_base, rocke_b_mul(b, kt_local, c->c_atom_k));
    return rocke_load_b_col_strided_scalars(
        b, c->Bp, c->atom, c->lane_decode, c->n_tile_base, k_tile_base, c->N);
}

/* K groups whose A scales the K-group loop reads with one load per output row
 * (Python _GROUPS_PER_A_SCALE_LOAD): a row's scales of consecutive K groups
 * are adjacent, so two groups take one 8-byte load. Each group of a load is
 * one more copy of the group's MFMA loop in the kernel; four groups per
 * 16-byte load measured slower than two on some shapes. */
#define ROCKE_BSG_GROUPS_PER_A_SCALE_LOAD 2

/* What _accumulate_group reads from build_block_scale_gemm's scope. */
typedef struct rocke_bsg_group_ctx
{
    rocke_bsg_load_ctx_t load; /* k_group_base is set per group */
    rocke_value_t* b_scale_rsrc;
    rocke_value_t* b_scale_col_offset;
    int n_scale_count_b;
    int gk;
} rocke_bsg_group_ctx_t;

/* _accumulate_group(b, kg, a_scales, acc, loop_suffix): add K group kg to acc.
 * Load the group's B scale, run its gk / atom.k MFMAs on a fresh accumulator,
 * scale the result by a_scales[i] * b_scale per output i and add it to acc.
 * loop_suffix keeps the names of the group's MFMA loop unique when a K-group
 * loop iteration runs more than one group. Each builder call is bound to a
 * temporary in Python's evaluation order:
 *   b_group_offset = kg * (n_scale_count_b * 4)
 *   b_scale_v      = buffer_load(b_scale_rsrc, b_scale_col_offset, b_group_offset)
 *   ab_scale_vec   = vec_pack([a_scales[i] * b_scale_v ...], F32)
 *   k_group_base   = kg * gk
 *   group_acc      = mfma_k_loop(b, K=gk, ..., "kk<suffix>", "gacc<suffix>")
 *   result         = vector_add(acc, vector_mul(group_acc, ab_scale_vec)) */
static rocke_value_t* rocke_bsg_accumulate_group(rocke_ir_builder_t* b,
                                                 rocke_bsg_group_ctx_t* g,
                                                 rocke_value_t* kg,
                                                 rocke_value_t* const* a_scales,
                                                 rocke_value_t* acc,
                                                 const char* loop_suffix)
{
    const rocke_mfma_atom_t* atom = g->load.atom;
    rocke_value_t* ab_scales[ROCKE_BSG_MAX_C_PER_LANE];
    rocke_value_t* b_group_offset;
    rocke_value_t* b_scale_v;
    rocke_value_t* ab_scale_vec;
    rocke_value_t* group_acc;
    rocke_value_t* scaled_group;
    char iv_name[16];
    char acc_name[16];
    int i;

    b_group_offset = rocke_b_mul(b, kg, rocke_b_const_i32(b, g->n_scale_count_b * 4));
    b_scale_v = rocke_b_buffer_load(
        b, g->b_scale_rsrc, g->b_scale_col_offset, b_group_offset, rocke_f32());
    for(i = 0; i < atom->c_per_lane; ++i)
    {
        ab_scales[i] = rocke_b_fmul(b, a_scales[i], b_scale_v);
    }
    ab_scale_vec = rocke_b_vec_pack(b, ab_scales, atom->c_per_lane, rocke_f32());

    /* The group's K range is [kg * gk, (kg + 1) * gk). */
    g->load.k_group_base = rocke_b_mul(b, kg, rocke_b_const_i32(b, g->gk));

    snprintf(iv_name, sizeof(iv_name), "kk%s", loop_suffix);
    snprintf(acc_name, sizeof(acc_name), "gacc%s", loop_suffix);
    group_acc = rocke_mfma_k_loop(b,
                                  g->gk,
                                  atom,
                                  rocke_bsg_load_a_in_group,
                                  rocke_bsg_load_b_in_group,
                                  /*per_tile_post_mfma=*/NULL,
                                  /*initial_acc=*/NULL,
                                  iv_name,
                                  acc_name,
                                  &g->load);

    /* group_acc, acc and ab_scale_vec are <c_per_lane x f32> per-lane vectors
     * (element i belongs to output i); one vector fmul and one vector fadd
     * lower to packed v_pk_mul_f32 / v_pk_add_f32 on AMDGPU. */
    scaled_group = rocke_b_vector_mul(b, group_acc, ab_scale_vec);
    return rocke_b_vector_add(b, acc, scaled_group);
}

rocke_kernel_def_t* rocke_build_block_scale_gemm(rocke_ir_builder_t* b,
                                                 const rocke_block_scale_gemm_spec_t* spec,
                                                 const char* arch)
{
    char reason[160];
    const rocke_mfma_atom_t* atom;
    const char* mantissa_store;
    int BS;
    const char* a_store;
    const char* b_store;
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* AScale = NULL;
    rocke_value_t* BScale = NULL;
    rocke_value_t* C;
    rocke_value_t* lane;
    rocke_value_t* bid_n;
    rocke_value_t* bid_m;
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_lane_decode_t lane_decode;
    int gm, gn, gk;
    int n_scale_count_b;
    int k_scale_count_a;
    int num_groups;
    rocke_value_t* c_atom_k;
    rocke_param_opts_t opts;
    const rocke_type_t* a_store_ty;
    const rocke_type_t* b_store_ty;
    rocke_value_t* acc_final;
    rocke_bsg_group_ctx_t gctx;
    int groups_per_load;
    int num_group_runs;
    /* One A-scale row base per accumulator slot; c_per_lane is 4 for the 16x16
     * atom, the only atom rocke_block_scale_gemm_spec_atom returns. */
    rocke_value_t* a_scale_rsrc;
    rocke_value_t* b_scale_rsrc;
    rocke_value_t* a_scale_row_offsets[ROCKE_BSG_MAX_C_PER_LANE];
    rocke_value_t* b_scale_col_offset;
    int i;

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(spec == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "build_block_scale_gemm: null spec");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    if(!rocke_block_scale_gemm_is_valid_spec(b, spec, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid block_scale_gemm spec for %s: %s", arch, reason);
    }

    atom = rocke_block_scale_gemm_spec_atom(spec);
    if(atom == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "block_scale_gemm: no MFMA atom for spec");
    }
    /* validate_mfma_atom_in_catalog(spec.atom, arch, where="block_scale_gemm") */
    if(rocke_validate_mfma_atom_in_catalog(b, atom, arch, "block_scale_gemm") != ROCKE_OK)
    {
        return NULL;
    }

    mantissa_store = rocke_block_scale_gemm_mantissa_store(spec);
    BS = rocke_block_scale_gemm_spec_block_size(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    if(rocke_i_live(b) && b->kernel != NULL)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);
    }

    /* a_store = mantissa_store if quant_mode in (aquant,abquant) else "f16"
     * b_store = mantissa_store if quant_mode in (bquant,abquant) else "f16" */
    a_store = (strcmp(spec->quant_mode, "aquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
                  ? mantissa_store
                  : "f16";
    b_store = (strcmp(spec->quant_mode, "bquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
                  ? mantissa_store
                  : "f16";

    a_store_ty = rocke_block_scale_gemm_storage_ir_type(a_store);
    b_store_ty = rocke_block_scale_gemm_storage_ir_type(b_store);
    if(a_store_ty == NULL || b_store_ty == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "block_scale_gemm: bad storage dtype");
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
    A = rocke_b_param(b, "A", rocke_ptr_type(b, a_store_ty, "global"), &opts);
    Bp = rocke_b_param(b, "B", rocke_ptr_type(b, b_store_ty, "global"), &opts);

    /* Scale pointers per quant_mode. */
    if(strcmp(spec->quant_mode, "aquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        rocke_param_opts_t sopts;
        memset(&sopts, 0, sizeof(sopts));
        sopts.readonly = true;
        sopts.readonly_set = true;
        sopts.align = 4;
        sopts.align_set = true;
        AScale = rocke_b_param(b, "AScale", rocke_ptr_type(b, rocke_f32(), "global"), &sopts);
    }
    if(strcmp(spec->quant_mode, "bquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        rocke_param_opts_t sopts;
        memset(&sopts, 0, sizeof(sopts));
        sopts.readonly = true;
        sopts.readonly_set = true;
        sopts.align = 4;
        sopts.align_set = true;
        BScale = rocke_b_param(b, "BScale", rocke_ptr_type(b, rocke_f32(), "global"), &sopts);
    }

    /* C = b.param("C", PtrType(F32,"global"), writeonly=True, align=4) */
    {
        rocke_param_opts_t copts;
        memset(&copts, 0, sizeof(copts));
        copts.writeonly = true;
        copts.writeonly_set = true;
        copts.align = 4;
        copts.align_set = true;
        C = rocke_b_param(b, "C", rocke_ptr_type(b, rocke_f32(), "global"), &copts);
    }

    /* M/N/K i32 ABI params (unused values, like Python's noqa F841). */
    (void)rocke_b_param(b, "M", rocke_i32(), NULL);
    (void)rocke_b_param(b, "N", rocke_i32(), NULL);
    (void)rocke_b_param(b, "K", rocke_i32(), NULL);

    lane = rocke_b_thread_id_x(b);
    bid_n = rocke_b_block_id_x(b);
    bid_m = rocke_b_block_id_y(b);
    m_tile_base = rocke_b_mul(b, bid_m, rocke_b_const_i32(b, spec->block_tile_m));
    n_tile_base = rocke_b_mul(b, bid_n, rocke_b_const_i32(b, spec->block_tile_n));

    /* if quant_mode != "abquant": raise NotImplementedError */
    if(strcmp(spec->quant_mode, "abquant") != 0)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_NOTIMPL,
            "MFMA block_scale_gemm v1 ships abquant only; aquant/bquant "
            "variants are a follow-on (same MFMA inner, asymmetric scale apply).");
    }
    /* if a_store != b_store or a_store not in (fp8e4m3,bf8e5m2): raise */
    if(strcmp(a_store, b_store) != 0
       || (strcmp(a_store, "fp8e4m3") != 0 && strcmp(a_store, "bf8e5m2") != 0))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_NOTIMPL,
            "MFMA path needs A and B in fp8e4m3 or bf8e5m2 (got A=%s, B=%s)",
            a_store,
            b_store);
    }

    /* lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane_decode = rocke_decode_mfma_lanes(b, atom, lane);

    gm = spec->group_m;
    gn = spec->group_n;
    gk = spec->group_k;
    /* if gk % atom.k != 0: raise ValueError */
    if(atom->k == 0 || (gk % atom->k) != 0)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "group_k (%d) must be a multiple of atom.k (%d) so the per-group scale "
            "apply aligns with a whole number of MFMA invocations",
            gk,
            atom->k);
    }

    n_scale_count_b = (spec->N + gn - 1) / gn;
    k_scale_count_a = (spec->K + gk - 1) / gk;

    num_groups = spec->K / gk;
    c_atom_k = rocke_b_const_i32(b, atom->k);

    /* Each lane scales the outputs it holds, so it needs the A scale of each
     * output's row and the B scale of each output's column. In the 16x16 atom a
     * lane's c_per_lane outputs sit in c_per_lane rows of one column
     * (rocke_mfma_atom_lane_to_output, the mapping the store uses): the rows are
     * not the row the lane loads A from (m_in_atom), so the lane loads one A
     * scale per output, while its column is the one it loads B from
     * (n_in_atom), so one B scale serves all its outputs.
     *
     * The scales are read with buffer loads: the per-lane byte offset of each
     * scale row (A) or column (B) is computed once, before the K-group loop, and
     * the K group's byte offset, the same for every lane, goes in the load's
     * scalar offset, so a K group adds no per-lane address arithmetic for its
     * scales. The scale tensors are (ceil(M / gm), K / gk)
     * (A) and (K / gk, ceil(N / gn)) (B), f32, row-major. gk is a multiple of
     * atom->k (32), so each is at most an eighth of the size in bytes of A or
     * B: they fit a buffer's 32-bit byte range whenever A and B fit the 32-bit
     * element index the kernel already uses for them.
     *
     * Each builder call is bound to a temporary in Python's left-to-right
     * evaluation order (C argument order is unspecified):
     *   a_scale_rsrc           = buffer_rsrc(AScale, ceil(M / gm) * k_scale_count_a * 4)
     *   b_scale_rsrc           = buffer_rsrc(BScale, num_groups * n_scale_count_b * 4)
     *   a_scale_row_offsets[i] = ((m_tile_base + row_in_atom(i)) / gm)
     *                            * (k_scale_count_a * 4)
     *   b_scale_col_offset     = ((n_tile_base + n_in_atom) / gn) * 4 */
    if(atom->c_per_lane > ROCKE_BSG_MAX_C_PER_LANE)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "block_scale_gemm: atom c_per_lane %d exceeds %d",
            atom->c_per_lane,
            ROCKE_BSG_MAX_C_PER_LANE);
    }
    a_scale_rsrc = rocke_b_buffer_rsrc(
        b, AScale, rocke_b_const_i32(b, ((spec->M + gm - 1) / gm) * k_scale_count_a * 4));
    b_scale_rsrc
        = rocke_b_buffer_rsrc(b, BScale, rocke_b_const_i32(b, num_groups * n_scale_count_b * 4));
    for(i = 0; i < atom->c_per_lane; ++i)
    {
        rocke_value_t* row_in_atom;
        rocke_value_t* col_in_atom;
        rocke_value_t* row;
        rocke_value_t* row_group;

        if(!rocke_mfma_atom_lane_to_output(
               b, atom, lane_decode.lane, i, &row_in_atom, &col_in_atom))
        {
            return NULL;
        }
        row = rocke_b_add(b, m_tile_base, row_in_atom);
        row_group = rocke_b_div(b, row, rocke_b_const_i32(b, gm));
        a_scale_row_offsets[i]
            = rocke_b_mul(b, row_group, rocke_b_const_i32(b, k_scale_count_a * 4));
    }
    {
        rocke_value_t* col = rocke_b_add(b, n_tile_base, lane_decode.n_in_atom);
        rocke_value_t* col_group = rocke_b_div(b, col, rocke_b_const_i32(b, gn));
        b_scale_col_offset = rocke_b_mul(b, col_group, rocke_b_const_i32(b, 4));
    }

    gctx.load.A = A;
    gctx.load.Bp = Bp;
    gctx.load.atom = atom;
    gctx.load.lane_decode = &lane_decode;
    gctx.load.m_tile_base = m_tile_base;
    gctx.load.n_tile_base = n_tile_base;
    gctx.load.k_group_base = NULL;
    gctx.load.c_atom_k = c_atom_k;
    gctx.load.K = spec->K;
    gctx.load.N = spec->N;
    gctx.b_scale_rsrc = b_scale_rsrc;
    gctx.b_scale_col_offset = b_scale_col_offset;
    gctx.n_scale_count_b = n_scale_count_b;
    gctx.gk = gk;

    /* A row's A scales of consecutive K groups are adjacent, so the main
     * K-group loop runs groups_per_load groups per iteration and reads each
     * output row's A scales for all of them with one load. The last
     * num_groups % groups_per_load groups run one per iteration, with one
     * 4-byte load per row.
     *   acc = atom.zero_acc(b)    (emitted before the loops' bound constants) */
    groups_per_load = ROCKE_BSG_GROUPS_PER_A_SCALE_LOAD;
    num_group_runs = num_groups / groups_per_load;
    acc_final = rocke_b_zero_vec_f32(b, atom->c_per_lane);
    if(num_group_runs > 0)
    {
        /* run_loop = b.scf_for_iter(0, num_group_runs, 1, [("acc", acc)], "kr") */
        rocke_value_t* lb = rocke_b_const_i32(b, 0);
        rocke_value_t* ub = rocke_b_const_i32(b, num_group_runs);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        rocke_iter_arg_t run_args[1];
        rocke_for_t run_loop;

        run_args[0].name = "acc";
        run_args[0].init = acc_final;
        run_loop = rocke_b_scf_for_iter(b,
                                        lb,
                                        ub,
                                        step,
                                        run_args,
                                        1,
                                        "kr",
                                        /*unroll=*/false,
                                        /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, run_loop.body);
        {
            rocke_value_t* kr = run_loop.iv;
            rocke_value_t* run_acc = run_loop.iter_vars[0];
            rocke_value_t* a_scale_runs[ROCKE_BSG_MAX_C_PER_LANE];
            rocke_value_t* a_scales[ROCKE_BSG_MAX_C_PER_LANE];
            rocke_value_t* a_run_offset;
            rocke_value_t* first_kg;
            rocke_value_t* yield_vals[1];
            int j;

            /* a_run_offset    = kr * (groups_per_load * 4)
             * a_scale_runs[i] = buffer_load_vN(a_scale_rsrc, a_scale_row_offsets[i],
             *                                  a_run_offset, F32, groups_per_load)
             * first_kg        = kr * groups_per_load */
            a_run_offset = rocke_b_mul(b, kr, rocke_b_const_i32(b, groups_per_load * 4));
            for(i = 0; i < atom->c_per_lane; ++i)
            {
                a_scale_runs[i] = rocke_b_buffer_load_vN(b,
                                                         a_scale_rsrc,
                                                         a_scale_row_offsets[i],
                                                         a_run_offset,
                                                         rocke_f32(),
                                                         groups_per_load);
            }
            first_kg = rocke_b_mul(b, kr, rocke_b_const_i32(b, groups_per_load));
            for(j = 0; j < groups_per_load; ++j)
            {
                /* kg = first_kg + j; a_scales[i] = vec_extract(a_scale_runs[i], j) */
                char loop_suffix[8];
                rocke_value_t* kg = rocke_b_add(b, first_kg, rocke_b_const_i32(b, j));

                for(i = 0; i < atom->c_per_lane; ++i)
                {
                    a_scales[i] = rocke_b_vec_extract(b, a_scale_runs[i], j);
                }
                snprintf(loop_suffix, sizeof(loop_suffix), "%d", j);
                run_acc = rocke_bsg_accumulate_group(b, &gctx, kg, a_scales, run_acc, loop_suffix);
            }
            yield_vals[0] = run_acc;
            rocke_b_scf_yield(b, yield_vals, 1);
        }
        rocke_b_region_leave(b);
        if(!rocke_i_live(b) || run_loop.op == NULL || run_loop.op->num_results < 1)
        {
            return NULL;
        }
        acc_final = run_loop.op->results[0];
    }
    if(num_groups % groups_per_load != 0)
    {
        /* tail_loop = b.scf_for_iter(num_group_runs * groups_per_load, num_groups, 1,
         *                            [("tail_acc", acc)], "kg") */
        rocke_value_t* lb = rocke_b_const_i32(b, num_group_runs * groups_per_load);
        rocke_value_t* ub = rocke_b_const_i32(b, num_groups);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        rocke_iter_arg_t tail_args[1];
        rocke_for_t tail_loop;

        tail_args[0].name = "tail_acc";
        tail_args[0].init = acc_final;
        tail_loop = rocke_b_scf_for_iter(b,
                                         lb,
                                         ub,
                                         step,
                                         tail_args,
                                         1,
                                         "kg",
                                         /*unroll=*/false,
                                         /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, tail_loop.body);
        {
            rocke_value_t* kg = tail_loop.iv;
            rocke_value_t* a_scales[ROCKE_BSG_MAX_C_PER_LANE];
            rocke_value_t* a_group_offset;
            rocke_value_t* yield_vals[1];

            /* a_group_offset = kg * 4
             * a_scales[i]    = buffer_load(a_scale_rsrc, a_scale_row_offsets[i],
             *                              a_group_offset) */
            a_group_offset = rocke_b_mul(b, kg, rocke_b_const_i32(b, 4));
            for(i = 0; i < atom->c_per_lane; ++i)
            {
                a_scales[i] = rocke_b_buffer_load(
                    b, a_scale_rsrc, a_scale_row_offsets[i], a_group_offset, rocke_f32());
            }
            yield_vals[0]
                = rocke_bsg_accumulate_group(b, &gctx, kg, a_scales, tail_loop.iter_vars[0], "");
            rocke_b_scf_yield(b, yield_vals, 1);
        }
        rocke_b_region_leave(b);
        if(!rocke_i_live(b) || tail_loop.op == NULL || tail_loop.op->num_results < 1)
        {
            return NULL;
        }
        acc_final = tail_loop.op->results[0];
    }

    /* store_acc_to_global(b, C, atom, lane_decode, m_tile_base, n_tile_base,
     *     acc_final, N=spec.N, out_dtype="f32") */
    if(rocke_store_acc_to_global(b,
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
                                 NULL)
       != ROCKE_OK)
    {
        return NULL;
    }

    rocke_b_ret(b);
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_block_scale_gemm_new(rocke_ir_builder_t* b,
                                                     const rocke_block_scale_gemm_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_block_scale_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_block_scale_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  block_scale_gemm_grid
 * ===================================================================== */
rocke_status_t rocke_block_scale_gemm_grid(const rocke_block_scale_gemm_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = (spec->N + spec->block_tile_n - 1) / spec->block_tile_n;
    out[1] = (spec->M + spec->block_tile_m - 1) / spec->block_tile_m;
    out[2] = 1;
    return ROCKE_OK;
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
static const char* rocke_bsg_side_dtype(const rocke_block_scale_gemm_spec_t* spec,
                                        const char* f16_when_mode)
{
    if(strcmp(spec->quant_mode, f16_when_mode) == 0)
    {
        return "f16";
    }
    if(strncmp(spec->mantissa_dtype, "i4_", 3) == 0)
    {
        return "i8";
    }
    return spec->mantissa_dtype;
}

rocke_status_t rocke_block_scale_gemm_signature(struct rocke_arena* arena,
                                                const rocke_block_scale_gemm_spec_t* spec,
                                                const rocke_sig_entry_t** out_items,
                                                size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    const char* a_dtype;
    const char* b_dtype;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }

    a_dtype = rocke_bsg_side_dtype(spec, "bquant");
    b_dtype = rocke_bsg_side_dtype(spec, "aquant");

    rocke_signature_builder_ptr(&sb, "A", a_dtype, NULL);
    rocke_signature_builder_ptr(&sb, "B", b_dtype, NULL);
    if(strcmp(spec->quant_mode, "aquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        rocke_signature_builder_ptr(&sb, "AScale", "f32", NULL);
    }
    if(strcmp(spec->quant_mode, "bquant") == 0 || strcmp(spec->quant_mode, "abquant") == 0)
    {
        rocke_signature_builder_ptr(&sb, "BScale", "f32", NULL);
    }
    rocke_signature_builder_ptr(&sb, "C", "f32", NULL);
    rocke_signature_builder_scalar(&sb, "M", "i32");
    rocke_signature_builder_scalar(&sb, "N", "i32");
    rocke_signature_builder_scalar(&sb, "K", "i32");

    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  lower_to_llvm convenience
 * ===================================================================== */
rocke_status_t rocke_block_scale_gemm_lower_to_llvm(const rocke_block_scale_gemm_spec_t* spec,
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
        rocke_i_write_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_block_scale_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            rocke_i_write_reason(err, err_cap, (m != NULL) ? m : "build_block_scale_gemm failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
