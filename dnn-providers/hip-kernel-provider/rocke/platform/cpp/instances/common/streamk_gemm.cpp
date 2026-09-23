// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_streamk_gemm.c -- C99 port of rocke/instances/common/streamk_gemm.py
 * (CK Tile 40_streamk_gemm parity).
 *
 * The builder-call sequence in rocke_build_streamk_gemm mirrors build_streamk_gemm
 * one-for-one: param order, thread_id_x / block_id_x, decode_mfma_lanes, the
 * per-macro-tile MFMA + atomic-store body (SGPR-pinned bases), and the
 * persistent vs non-persistent dispatch. The macro-tile body is shared between
 * both dispatch modes through a small closure-environment struct passed to the
 * persistent helper's body callback (the C analog of the Python closure).
 */
#include "rocke/instance_streamk_gemm.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h" /* mma has_shape       */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* ArchTarget.from_gfx */
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h" /* decode/load/k_loop/store */
#include "rocke/helper_rocke.helpers.persistent.h" /* persistent_tile_for_each */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/helper_rocke.helpers.streamk.h" /* partition / decode  */
#include "rocke/ir_internal.h" /* rocke_i_set_err       */

/* ===================================================================== *
 *  Spec defaults + derived @property accessors
 * ===================================================================== */

rocke_streamk_gemm_spec_t rocke_streamk_gemm_spec_default(void)
{
    rocke_streamk_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.M = 0;
    s.N = 0;
    s.K = 0;
    s.tile_m = 16;
    s.tile_n = 16;
    s.tile_k = 16;
    s.dtype = "f16";
    s.num_cus = 304;
    s.blocks_per_cu = 1;
    s.reduction = ROCKE_STREAMK_REDUCTION_ATOMIC;
    s.persistent = false;
    s.name = "rocke_streamk_gemm";
    return s;
}

bool rocke_streamk_gemm_partition(const rocke_streamk_gemm_spec_t* spec,
                                  rocke_streamk_partition_t* out)
{
    if(spec == NULL || out == NULL)
    {
        return false;
    }
    /* Python: raise ValueError if M/N/K not divisible by their tile sizes. */
    if(spec->tile_m == 0 || spec->tile_n == 0 || spec->tile_k == 0)
    {
        return false;
    }
    if((spec->M % spec->tile_m) || (spec->N % spec->tile_n) || (spec->K % spec->tile_k))
    {
        return false;
    }
    out->m_tiles = spec->M / spec->tile_m;
    out->n_tiles = spec->N / spec->tile_n;
    out->k_iters = spec->K / spec->tile_k;
    return true;
}

const rocke_mfma_atom_t* rocke_streamk_gemm_atom(const rocke_streamk_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    /* Python: (16,16) -> f16_16x16x16; (32,32) -> f16_32x32x8; else ValueError.
     * The two factory atoms resolve through the static catalog lookup. */
    if(spec->tile_m == 16 && spec->tile_n == 16)
    {
        return rocke_mfma_atom("f16", 16, 16, 16);
    }
    if(spec->tile_m == 32 && spec->tile_n == 32)
    {
        return rocke_mfma_atom("f16", 32, 32, 8);
    }
    return NULL;
}

int rocke_streamk_gemm_grid_size(const rocke_streamk_gemm_spec_t* spec)
{
    rocke_streamk_partition_t part;
    if(!rocke_streamk_gemm_partition(spec, &part))
    {
        return -1;
    }
    return rocke_compute_streamk_grid_size(&part, spec->num_cus, spec->blocks_per_cu, NULL);
}

int rocke_streamk_gemm_block_size(const rocke_streamk_gemm_spec_t* spec)
{
    (void)spec;
    return 64;
}

int rocke_streamk_gemm_persistent_max_iters(const rocke_streamk_gemm_spec_t* spec)
{
    rocke_streamk_partition_t part;
    int nm;
    int gs;
    if(!rocke_streamk_gemm_partition(spec, &part))
    {
        return -1;
    }
    nm = rocke_streamk_partition_num_macro_tiles(&part);
    gs = rocke_streamk_gemm_grid_size(spec);
    if(gs <= 0)
    {
        return -1;
    }
    /* Python: (nm + gs - 1) // gs */
    return (nm + gs - 1) / gs;
}

/* ===================================================================== *
 *  kernel_name
 *
 *  Python:
 *    kernel_name_join(self.name,
 *        f"M{M}N{N}K{K}", f"t{tm}x{tn}x{tk}",
 *        f"r{reduction.value}", f"g{grid_size}",
 *        flags={"pers": self.persistent})
 * ===================================================================== */
rocke_status_t
    rocke_streamk_gemm_kernel_name(const rocke_streamk_gemm_spec_t* spec, char* out, size_t out_cap)
{
    char part_mnk[64];
    char part_t[48];
    char part_r[32];
    char part_g[32];
    const char* parts[4];
    const char* flag_names[1];
    int flag_on[1];
    const char* rval;
    int gs;

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }

    gs = rocke_streamk_gemm_grid_size(spec);
    rval = rocke_streamk_reduction_strategy_value(spec->reduction);
    if(rval == NULL)
    {
        rval = "atomic";
    }

    snprintf(part_mnk, sizeof(part_mnk), "M%dN%dK%d", spec->M, spec->N, spec->K);
    snprintf(part_t, sizeof(part_t), "t%dx%dx%d", spec->tile_m, spec->tile_n, spec->tile_k);
    snprintf(part_r, sizeof(part_r), "r%s", rval);
    snprintf(part_g, sizeof(part_g), "g%d", gs);

    parts[0] = part_mnk;
    parts[1] = part_t;
    parts[2] = part_r;
    parts[3] = part_g;

    flag_names[0] = "pers";
    flag_on[0] = spec->persistent ? 1 : 0;

    return rocke_kernel_name_join(spec->name, parts, 4, flag_names, flag_on, 1, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */
static bool set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != NULL && cap > 0)
    {
        size_t n = strlen(msg);
        if(n >= cap)
        {
            n = cap - 1;
        }
        memcpy(reason, msg, n);
        reason[n] = '\0';
    }
    return false;
}

bool rocke_streamk_gemm_is_valid_spec(const rocke_streamk_gemm_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    const rocke_mfma_atom_t* atom;
    const rocke_archtarget_t* target;
    const rocke_mma_catalog_t* mma;

    if(spec == NULL)
    {
        return set_reason(reason, reason_cap, "null spec");
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* Python: v1 ships f16 only. */
    if(spec->dtype == NULL || strcmp(spec->dtype, "f16") != 0)
    {
        return set_reason(reason, reason_cap, "v1 ships f16 only");
    }
    /* Python: v1 ships the Atomic reduction strategy only. */
    if(spec->reduction != ROCKE_STREAMK_REDUCTION_ATOMIC)
    {
        return set_reason(reason, reason_cap, "v1 ships the Atomic reduction strategy only");
    }
    /* Python: MFMA path supports tile (16,16) or (32,32). */
    if(!((spec->tile_m == 16 && spec->tile_n == 16) || (spec->tile_m == 32 && spec->tile_n == 32)))
    {
        return set_reason(reason, reason_cap, "MFMA path supports tile (16,16) or (32,32)");
    }
    /* Python: M/N/K must be divisible by their tile sizes. */
    if(spec->tile_m == 0 || spec->tile_n == 0 || spec->tile_k == 0 || (spec->M % spec->tile_m)
       || (spec->N % spec->tile_n) || (spec->K % spec->tile_k))
    {
        return set_reason(reason, reason_cap, "M / N / K must be divisible by their tile sizes");
    }

    atom = rocke_streamk_gemm_atom(spec);
    if(atom == NULL)
    {
        return set_reason(reason, reason_cap, "no MFMA atom for tile shape");
    }
    /* Python: tile_k must be a multiple of atom.k. */
    if(atom->k == 0 || (spec->tile_k % atom->k) != 0)
    {
        return set_reason(reason, reason_cap, "tile_k must be a multiple of atom.k");
    }

    /* Arch gating: validate against the target MMA catalog (fp16/fp16/fp32). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        return set_reason(reason, reason_cap, "unknown gfx target");
    }
    mma = rocke_archtarget_mma(target);
    if(!rocke_mma_catalog_has_shape(mma, NULL, "fp16", "fp16", "fp32", atom->m, atom->n, atom->k))
    {
        return set_reason(reason, reason_cap, "f16 MFMA atom not available on target");
    }

    /* Python: reject persistent=True when persistent_max_iters > 1 (P35 bug). */
    if(spec->persistent)
    {
        int mi = rocke_streamk_gemm_persistent_max_iters(spec);
        if(mi > 1)
        {
            return set_reason(reason,
                              reason_cap,
                              "persistent=True with persistent_max_iters > 1 "
                              "hits a known bug in persistent_tile_for_each");
        }
    }

    if(reason != NULL && reason_cap > 0)
    {
        size_t n = 2; /* "ok" */
        if(n >= reason_cap)
        {
            n = reason_cap - 1;
        }
        memcpy(reason, "ok", n);
        reason[n] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  Macro-tile body -- the Python `_process_macro_tile` closure.
 *
 *  The captured environment (builder-local values + spec geometry) is bundled
 *  into rocke_streamk_body_env_t. _process_macro_tile is the body; the two
 *  load callbacks recompute the per-K-atom k_tile_base from the SGPR-pinned
 *  k_macro_base (set once per macro tile in _process_macro_tile).
 * ===================================================================== */
typedef struct rocke_streamk_body_env
{
    const rocke_streamk_gemm_spec_t* spec;
    rocke_streamk_partition_t partition;
    const rocke_mfma_atom_t* atom;
    rocke_lane_decode_t lane_decode;
    rocke_value_t* A;
    rocke_value_t* B;
    rocke_value_t* Cf32;
    /* Per-macro-tile SGPR-pinned bases, refreshed by _process_macro_tile and
     * consumed by the two load callbacks within the same macro tile. */
    rocke_value_t* m_tile_base;
    rocke_value_t* n_tile_base;
    rocke_value_t* k_macro_base;
} rocke_streamk_body_env_t;

/* Python _load_a closure:
 *   k_tile_base = b.add(k_macro_base, b.mul(kt, b.const_i32(atom.k)))
 *   return load_a_row_major_contiguous(b, A=A, atom=atom, lane_decode=...,
 *       m_tile_base=m_tile_base, k_tile_base=k_tile_base, K=spec.K) */
static rocke_value_t* streamk_load_a(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    rocke_streamk_body_env_t* env = (rocke_streamk_body_env_t*)user;
    rocke_value_t* k_tile_base
        = rocke_b_add(b, env->k_macro_base, rocke_b_mul(b, kt, rocke_b_const_i32(b, env->atom->k)));
    return rocke_load_a_row_major_contiguous(
        b, env->A, env->atom, &env->lane_decode, env->m_tile_base, k_tile_base, env->spec->K);
}

/* Python _load_b closure:
 *   k_tile_base = b.add(k_macro_base, b.mul(kt, b.const_i32(atom.k)))
 *   return load_b_col_strided_scalars(b, B=Bp, atom=atom, lane_decode=...,
 *       n_tile_base=n_tile_base, k_tile_base=k_tile_base, N=spec.N) */
static rocke_value_t* streamk_load_b(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    rocke_streamk_body_env_t* env = (rocke_streamk_body_env_t*)user;
    rocke_value_t* k_tile_base
        = rocke_b_add(b, env->k_macro_base, rocke_b_mul(b, kt, rocke_b_const_i32(b, env->atom->k)));
    return rocke_load_b_col_strided_scalars(
        b, env->B, env->atom, &env->lane_decode, env->n_tile_base, k_tile_base, env->spec->N);
}

/* Python _process_macro_tile(linear_id). */
static void streamk_process_macro_tile(rocke_ir_builder_t* b, rocke_value_t* linear_id, void* user)
{
    rocke_streamk_body_env_t* env = (rocke_streamk_body_env_t*)user;
    const rocke_streamk_gemm_spec_t* spec = env->spec;
    const rocke_mfma_atom_t* atom = env->atom;
    rocke_streamk_decoded_tile_t decoded;
    rocke_value_t* m_tile;
    rocke_value_t* n_tile;
    rocke_value_t* k_iter;
    rocke_value_t* acc_final;

    /* decoded = emit_streamk_decode(b, linear_id, partition) */
    decoded = rocke_emit_streamk_decode(b, linear_id, &env->partition);

    /* SGPR-pin m_tile / n_tile / k_iter (CTA-uniform). */
    m_tile = rocke_b_to_sgpr_u32(b, decoded.m_tile);
    n_tile = rocke_b_to_sgpr_u32(b, decoded.n_tile);
    k_iter = rocke_b_to_sgpr_u32(b, decoded.k_iter);

    /* Per-tile bases (pinned too):
     *   m_tile_base = to_sgpr_u32(mul(m_tile, tile_m))
     *   n_tile_base = to_sgpr_u32(mul(n_tile, tile_n))
     *   k_macro_base = to_sgpr_u32(mul(k_iter, tile_k)) */
    env->m_tile_base
        = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, m_tile, rocke_b_const_i32(b, spec->tile_m)));
    env->n_tile_base
        = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, n_tile, rocke_b_const_i32(b, spec->tile_n)));
    env->k_macro_base
        = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, k_iter, rocke_b_const_i32(b, spec->tile_k)));

    /* acc_final = mfma_k_loop(b, K=tile_k, atom=atom, load_a=_load_a,
     *     load_b=_load_b)  -- initial_acc=None, iv_name="kt", acc_name="acc" */
    acc_final = rocke_mfma_k_loop(
        b, spec->tile_k, atom, streamk_load_a, streamk_load_b, NULL, NULL, "kt", "acc", env);

    /* store_acc_to_global(b, C=Cf32, atom=atom, lane_decode=..., m_tile_base=...,
     *     n_tile_base=..., acc=acc_final, N=spec.N, out_dtype="f32",
     *     atomic_add=True) */
    rocke_store_acc_to_global(b,
                              env->Cf32,
                              atom,
                              &env->lane_decode,
                              env->m_tile_base,
                              env->n_tile_base,
                              acc_final,
                              spec->N,
                              "f32",
                              /*atomic_add=*/true,
                              /*epilogue=*/NULL,
                              /*epilogue_user=*/NULL);
}

/* ===================================================================== *
 *  build_streamk_gemm
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_streamk_gemm(rocke_ir_builder_t* b,
                                             const rocke_streamk_gemm_spec_t* spec,
                                             const char* arch)
{
    char reason[256];
    int BS;
    rocke_streamk_body_env_t env;
    rocke_param_opts_t opts;
    const rocke_type_t* ptr_f16;
    const rocke_type_t* ptr_f32;
    const rocke_type_t* ptr_i32;
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* Cf32;
    rocke_value_t* Counter;
    rocke_value_t* lane;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* Python: ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError */
    if(!rocke_streamk_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid streamk_gemm spec for %s: %s", arch, reason);
        return NULL;
    }

    memset(&env, 0, sizeof(env));
    env.spec = spec;
    if(!rocke_streamk_gemm_partition(spec, &env.partition))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "streamk_gemm: degenerate partition");
        return NULL;
    }
    BS = rocke_streamk_gemm_block_size(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* params:
     *   A = param("A", PtrType(F16,"global"), noalias, readonly, align=16)
     *   B = param("B", PtrType(F16,"global"), noalias, readonly, align=16)
     *   Cf32 = param("Cf32", PtrType(F32,"global"), align=4)
     *   Counter = param("Counter", PtrType(I32,"global"), align=4) */
    ptr_f16 = rocke_ptr_type(b, rocke_f16(), "global");
    ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");
    ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");

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
    opts.align = 4;
    opts.align_set = true;
    Cf32 = rocke_b_param(b, "Cf32", ptr_f32, &opts);

    memset(&opts, 0, sizeof(opts));
    opts.align = 4;
    opts.align_set = true;
    Counter = rocke_b_param(b, "Counter", ptr_i32, &opts);

    /* lane = b.thread_id_x(); atom = spec.atom;
     * lane_decode = decode_mfma_lanes(b, atom, lane) */
    lane = rocke_b_thread_id_x(b);
    env.atom = rocke_streamk_gemm_atom(spec);
    env.lane_decode = rocke_decode_mfma_lanes(b, env.atom, lane);

    env.A = A;
    env.B = Bp;
    env.Cf32 = Cf32;

    if(spec->persistent)
    {
        /* persistent_tile_for_each(b, counter=Counter,
         *     num_tiles=b.const_i32(partition.num_macro_tiles),
         *     max_iters=spec.persistent_max_iters, body=_process_macro_tile) */
        int nm = rocke_streamk_partition_num_macro_tiles(&env.partition);
        int max_iters = rocke_streamk_gemm_persistent_max_iters(spec);
        rocke_persistent_tile_for_each(b,
                                       Counter,
                                       rocke_b_const_i32(b, nm),
                                       max_iters,
                                       streamk_process_macro_tile,
                                       &env,
                                       /*counter_idx=*/NULL,
                                       /*cooperative=*/true,
                                       /*wave_size=*/64,
                                       /*block_size=*/64);
    }
    else
    {
        /* _ = Counter  (kept in ABI); _process_macro_tile(b.block_id_x()) */
        (void)Counter;
        streamk_process_macro_tile(b, rocke_b_block_id_x(b), &env);
    }

    /* b.ret() */
    rocke_b_ret(b);
    return b->kernel;
}

/* ===================================================================== *
 *  build_streamk_gemm_new -- init builder w/ spec.kernel_name() then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_streamk_gemm_new(rocke_ir_builder_t* b,
                                                 const rocke_streamk_gemm_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_streamk_gemm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_streamk_gemm(b, spec, arch);
    });
}

/* ===================================================================== *
 *  build_streamk_gemm_block_tile -- replace(spec, persistent=True) -> build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_streamk_gemm_block_tile(rocke_ir_builder_t* b,
                                                        const rocke_streamk_gemm_spec_t* spec,
                                                        const char* arch)
{
    rocke_streamk_gemm_spec_t block_tile_spec;
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    block_tile_spec = *spec;
    block_tile_spec.persistent = true;
    return rocke_build_streamk_gemm(b, &block_tile_spec, arch);
}

/* ===================================================================== *
 *  streamk_gemm_grid / streamk_gemm_workspace_bytes
 * ===================================================================== */
rocke_status_t rocke_streamk_gemm_grid(const rocke_streamk_gemm_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(spec->persistent)
    {
        int gs = rocke_streamk_gemm_grid_size(spec);
        if(gs < 0)
        {
            return ROCKE_ERR_VALUE;
        }
        out[0] = gs;
        out[1] = 1;
        out[2] = 1;
        return ROCKE_OK;
    }
    else
    {
        rocke_streamk_partition_t part;
        if(!rocke_streamk_gemm_partition(spec, &part))
        {
            return ROCKE_ERR_VALUE;
        }
        out[0] = rocke_streamk_partition_num_macro_tiles(&part);
        out[1] = 1;
        out[2] = 1;
        return ROCKE_OK;
    }
}

long rocke_streamk_gemm_workspace_bytes(const rocke_streamk_gemm_spec_t* spec)
{
    if(spec == NULL)
    {
        return -1;
    }
    /* Python: 4 * M * N + 4 */
    return (long)4 * (long)spec->M * (long)spec->N + 4L;
}

/* ===================================================================== *
 *  rocke_streamk_gemm_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_streamk_gemm_lower_to_llvm(const rocke_streamk_gemm_spec_t* spec,
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
        arch = "gfx950";
    }

    kernel = rocke_build_streamk_gemm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_streamk_gemm failed";
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
