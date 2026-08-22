// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the public surface + family dispatcher of the MatMulNBits
 * gfx1151 instance (rocke/instances/common/matmul_nbits.py), plus the baseline
 * (non-optimized) large-N WMMA tiled body
 * (rocke/instances/common/_matmul_nbits_large_n.build_large_n_matmul_nbits).
 *
 * Every IR-emitting routine reproduces the Python IRBuilder call sequence
 * op-for-op so the produced rocke_kernel_def_t is byte-faithful to the Python
 * output. Every node is arena-owned (b->arena); nothing is freed individually.
 *
 * The opt large-N body and the decode-GEMV body are peer ports (their own
 * translation units); this file dispatches into them and contributes only the
 * baseline large-N body + the validating dispatcher + conveniences.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.i4_dequant.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_common.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_large_n.h"
#include "rocke/helper_rocke.instances.common._matmul_nbits_large_n_opt.h"
#include "rocke/instance_matmul_nbits.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

/* The decode-GEMV peer header
 * (helper_rocke.instances.common._matmul_nbits_decode_gemv.h) is intentionally
 * NOT included here: it re-declares rocke.helpers.i4_dequant's
 * unpack_i4_byte_to_pair_f32 with a stale (void-returning) signature as a
 * standalone-compilation fallback, which conflicts with the canonical
 * (rocke_status_t-returning) declaration in helper_rocke.helpers.i4_dequant.h
 * that the large-N body uses. We mirror the two decode-GEMV public symbols
 * below; the definitions live in the peer .c and are resolved at link time. */
typedef enum rocke_matmul_nbits_scale_wire_dtype
{
    ROCKE_NBITS_SCALE_F16 = 0,
    ROCKE_NBITS_SCALE_F32 = 1
} rocke_matmul_nbits_scale_wire_dtype_t;

typedef struct rocke_matmul_nbits_decode_gemv_spec
{
    int N;
    int K;
    int group_size;
    int block_size;
    rocke_matmul_nbits_scale_wire_dtype_t scale_wire;
} rocke_matmul_nbits_decode_gemv_spec_t;

/* C++ build: cross-TU C-ABI helper (defined with C linkage in
 * _matmul_nbits_decode_gemv.c); forward decl must be extern "C". No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
rocke_kernel_def_t* rocke_build_decode_gemv_matmul_nbits(
    rocke_ir_builder_t* b, const rocke_matmul_nbits_decode_gemv_spec_t* spec, const char* arch);
#ifdef __cplusplus
}
#endif

/* ===================================================================== *
 *  small helpers
 * ===================================================================== */

/* Copy a NUL-terminated message into a (cap-bounded) reason/err buffer. No-op
 * if buf is NULL or cap is 0. */
static void rocke_mnb_copy_msg(char* buf, size_t cap, const char* msg)
{
    size_t n;
    if(buf == NULL || cap == 0 || msg == NULL)
    {
        return;
    }
    n = strlen(msg);
    if(n >= cap)
    {
        n = cap - 1;
    }
    memcpy(buf, msg, n);
    buf[n] = '\0';
}

/* getattr(b, wp.wmma_op) -- dispatch the WMMA atom by its IRBuilder method name.
 * Only the two names _wmma_params can produce are handled; unknown => NULL. */
static rocke_value_t* rocke_mnb_wmma(rocke_ir_builder_t* b,
                                     const char* wmma_op,
                                     rocke_value_t* a,
                                     rocke_value_t* bb,
                                     rocke_value_t* c)
{
    if(wmma_op != NULL && strcmp(wmma_op, "wmma_gfx12_f32_16x16x16_f16") == 0)
    {
        return rocke_b_wmma_gfx12_f32_16x16x16_f16(b, a, bb, c);
    }
    if(wmma_op != NULL && strcmp(wmma_op, "wmma_f32_16x16x16_f16") == 0)
    {
        return rocke_b_wmma_f32_16x16x16_f16(b, a, bb, c);
    }
    return NULL;
}

/* ===================================================================== *
 *  is_valid_spec  (matmul_nbits.py::is_valid_spec)
 * ===================================================================== */
bool rocke_matmul_nbits_is_valid_spec(const rocke_matmul_nbits_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    if(spec == NULL)
    {
        rocke_mnb_copy_msg(reason, reason_cap, "spec is NULL");
        return false;
    }

    /* ok, reason = validate_common_spec(spec, arch) */
    if(!rocke_matmul_nbits_validate_common_spec(spec, arch, reason, reason_cap))
    {
        return false;
    }

    /* Per-family geometry extras. The skinny-N specialization exists to avoid
     * wasting a wide N tile on the N=32 linear-attention projection; route
     * anything wider through the large-N family.
     *
     *   if spec.family == "skinny_n" and spec.N > 64:
     *       return False, (
     *           f"skinny_n is for narrow N (<= 64); N={spec.N} should use "
     *           "family='large_n'")
     */
    if(spec->family != NULL && strcmp(spec->family, "skinny_n") == 0 && spec->N > 64)
    {
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "skinny_n is for narrow N (<= 64); N=%d should use "
                     "family='large_n'",
                     spec->N);
        }
        return false;
    }

    rocke_mnb_copy_msg(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  rocke_build_large_n_matmul_nbits -- baseline large-N WMMA tiled body
 *  (C99 port of _matmul_nbits_large_n.build_large_n_matmul_nbits)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_large_n_matmul_nbits(rocke_ir_builder_t* b,
                                                     const rocke_matmul_nbits_spec_t* spec,
                                                     const char* arch)
{
    rocke_wmma_params_t wp;
    const rocke_gemm_tile_spec_t* t;
    int N, K, group;
    int wtk, wave;
    int rows_per_wave, cols_per_wave, n_sub_m, n_sub_n, n_acc;
    int k_packed_stride, k_group_stride;
    const char* scale_wire = NULL;
    const rocke_type_t* scale_t;
    const rocke_type_t* F16t;
    const rocke_type_t* I8t;
    const rocke_type_t* I32t;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MATMUL_NBITS_V1_ARCH; /* "gfx1151" */
    }

    /* wp = _wmma_params(arch) */
    if(rocke_wmma_params(arch, &wp) != ROCKE_OK)
    {
        return NULL;
    }

    t = &spec->tile;
    N = spec->N;
    K = spec->K;
    group = spec->group_size;
    /* wtm/wtn read off the tile via the loop indices below; wtk drives the step. */
    wtk = t->warp_tile_k;
    wave = spec->wave_size;

    rows_per_wave = t->tile_m / t->warp_m;
    cols_per_wave = t->tile_n / t->warp_n;
    n_sub_m = rows_per_wave / t->warp_tile_m;
    n_sub_n = cols_per_wave / t->warp_tile_n;
    k_packed_stride = K / 2; /* packed-byte row stride for B (two int4 per byte) */
    k_group_stride = K / group; /* scale row stride */

    /* scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32 */
    if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
    {
        return NULL;
    }
    F16t = rocke_f16();
    I8t = rocke_i8();
    I32t = rocke_i32();
    scale_t = (strcmp(scale_wire, "f16") == 0) ? rocke_f16() : rocke_f32();

    /* b = IRBuilder(spec.kernel_name()) is the caller's responsibility (the
     * builder is already init'd; see rocke_build_matmul_nbits_new).
     * b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* --- kernel params --- */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* Sp;
    rocke_value_t* C;
    rocke_value_t* M;
    {
        rocke_param_opts_t opts;

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", rocke_ptr_type(b, F16t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = rocke_b_param(b, "B", rocke_ptr_type(b, I8t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 8;
        opts.align_set = true;
        Sp = rocke_b_param(b, "Scales", rocke_ptr_type(b, scale_t, "global"), &opts);

        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = rocke_b_param(b, "C", rocke_ptr_type(b, F16t, "global"), &opts);

        M = rocke_b_param(b, "M", I32t, NULL);
    }

    /* --- constants --- */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* cK = rocke_b_const_i32(b, K);
    rocke_value_t* cN = rocke_b_const_i32(b, N);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* cwtk = rocke_b_const_i32(b, wtk);
    rocke_value_t* cwave = rocke_b_const_i32(b, wave);
    rocke_value_t* cgroup = rocke_b_const_i32(b, group);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, cwave);
    rocke_value_t* lane = rocke_b_mod(b, tid, cwave);
    rocke_value_t* frag = rocke_b_mod(b, lane, c16); /* lane%16 */
    rocke_value_t* half = rocke_b_div(b, lane, c16); /* lane/16 */

    /* gfx12 splits the WMMA's K=16 across lane-halves; per-lane K offset within a
     * step is half*frag_k. gfx11 carries the full K in every lane. */
    rocke_value_t* half_k_elem
        = wp.split_k_by_half ? rocke_b_mul(b, half, rocke_b_const_i32(b, wp.frag_k)) : c0;
    rocke_value_t* half_k_byte
        = wp.split_k_by_half ? rocke_b_mul(b, half, rocke_b_const_i32(b, wp.frag_k / 2)) : c0;

    rocke_value_t* wave_m = rocke_b_div(b, wave_id, rocke_b_const_i32(b, t->warp_n));
    rocke_value_t* wave_n = rocke_b_mod(b, wave_id, rocke_b_const_i32(b, t->warp_n));

    /* Python evaluates block_id_y()/block_id_x() BEFORE const_i32(); C arg
     * evaluation is right-to-left, so bind the block-id to a temp first to keep
     * SSA-value-id creation order identical to Python. */
    rocke_value_t* bid_y = rocke_b_block_id_y(b);
    rocke_value_t* m0 = rocke_b_mul(b, bid_y, rocke_b_const_i32(b, t->tile_m));
    rocke_value_t* bid_x = rocke_b_block_id_x(b);
    rocke_value_t* n0 = rocke_b_mul(b, bid_x, rocke_b_const_i32(b, t->tile_n));
    rocke_value_t* wm_base
        = rocke_b_add(b, m0, rocke_b_mul(b, wave_m, rocke_b_const_i32(b, rows_per_wave)));
    rocke_value_t* wn_base
        = rocke_b_add(b, n0, rocke_b_mul(b, wave_n, rocke_b_const_i32(b, cols_per_wave)));

    /* Loop-invariant per-lane row bases (element / byte / group offsets). */
    rocke_value_t** a_row_off
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_m * sizeof(*a_row_off));
    rocke_value_t** b_byte_off
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*b_byte_off));
    rocke_value_t** b_scale_off
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*b_scale_off));
    if(a_row_off == NULL || b_byte_off == NULL || b_scale_off == NULL)
    {
        return NULL;
    }
    {
        int sm, sn;
        for(sm = 0; sm < n_sub_m; ++sm)
        {
            rocke_value_t* a_row = rocke_b_add(
                b, wm_base, rocke_b_add(b, rocke_b_const_i32(b, sm * t->warp_tile_m), frag));
            a_row_off[sm] = rocke_b_mul(b, a_row, cK);
        }
        for(sn = 0; sn < n_sub_n; ++sn)
        {
            rocke_value_t* b_row = rocke_b_add(
                b, wn_base, rocke_b_add(b, rocke_b_const_i32(b, sn * t->warp_tile_n), frag));
            b_byte_off[sn] = rocke_b_mul(b, b_row, rocke_b_const_i32(b, k_packed_stride));
            b_scale_off[sn] = rocke_b_mul(b, b_row, rocke_b_const_i32(b, k_group_stride));
        }
    }

    /* --- acc0 = [zero_vec_f32(8) for _ in range(n_acc)] --- */
    n_acc = n_sub_m * n_sub_n;
    rocke_iter_arg_t* iter_args
        = (rocke_iter_arg_t*)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(*iter_args));
    if(iter_args == NULL)
    {
        return NULL;
    }
    {
        int i;
        for(i = 0; i < n_acc; ++i)
        {
            char* nm = (char*)rocke_arena_alloc(&b->arena, 24);
            if(nm == NULL)
            {
                return NULL;
            }
            snprintf(nm, 24, "acc%d", i);
            iter_args[i].name = nm;
            iter_args[i].init = rocke_b_zero_vec_f32(b, 8);
        }
    }

    /* loop = b.scf_for_iter(c0, cK, cwtk, [(acc{i}, acc0[i])...], iv_name="k0") */
    rocke_for_t loop = rocke_b_scf_for_iter(b, c0, cK, cwtk, iter_args, n_acc, "k0", false, true);
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* k0 = loop.iv;
        rocke_value_t* const* accs = loop.iter_vars;

        rocke_value_t* k_half = rocke_b_div(b, k0, c2); /* packed-byte K offset */
        rocke_value_t* k_grp = rocke_b_div(b, k0, cgroup); /* scale group index    */

        /* a_frag = [global_load_vN_f16(A, a_row_off[sm]+k0+half_k_elem, frag_k)] */
        rocke_value_t** a_frag
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_m * sizeof(*a_frag));
        if(a_frag == NULL)
        {
            return NULL;
        }
        {
            int sm;
            for(sm = 0; sm < n_sub_m; ++sm)
            {
                rocke_value_t* idx = rocke_b_add(b, rocke_b_add(b, a_row_off[sm], k0), half_k_elem);
                a_frag[sm] = rocke_b_global_load_vN_f16(b, A, idx, wp.frag_k, 0);
            }
        }

        int n_bytes = wp.frag_k / 2; /* packed B bytes feeding one fragment */
        rocke_value_t** b_frag
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_sub_n * sizeof(*b_frag));
        if(b_frag == NULL)
        {
            return NULL;
        }
        {
            int sn;
            for(sn = 0; sn < n_sub_n; ++sn)
            {
                rocke_value_t* packed = rocke_b_global_load_vN(
                    b,
                    Bp,
                    rocke_b_add(b, rocke_b_add(b, b_byte_off[sn], k_half), half_k_byte),
                    I8t,
                    n_bytes,
                    0);
                rocke_value_t* scale = rocke_b_global_load(
                    b, Sp, rocke_b_add(b, b_scale_off[sn], k_grp), scale_t, 1);
                rocke_value_t* scale_f32 = rocke_b_cast_to_f32(b, scale);
                rocke_value_t** comps = (rocke_value_t**)rocke_arena_alloc(
                    &b->arena, (size_t)(2 * n_bytes) * sizeof(*comps));
                if(comps == NULL)
                {
                    return NULL;
                }
                {
                    int j;
                    for(j = 0; j < n_bytes; ++j)
                    {
                        rocke_value_t* byte = rocke_b_vec_extract(b, packed, j);
                        rocke_value_t* lo = NULL;
                        rocke_value_t* hi = NULL;
                        if(rocke_dequant_i4_byte_to_f16_pair(b, byte, scale_f32, &lo, &hi)
                           != ROCKE_OK)
                        {
                            return NULL;
                        }
                        comps[2 * j] = lo;
                        comps[2 * j + 1] = hi;
                    }
                }
                b_frag[sn] = rocke_b_vec_pack(b, comps, 2 * n_bytes, F16t);
            }
        }

        /* wmma = getattr(b, wp.wmma_op); nacc[idx] = wmma(a_frag[sm], b_frag[sn], accs[idx]) */
        rocke_value_t** nacc
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)n_acc * sizeof(*nacc));
        if(nacc == NULL)
        {
            return NULL;
        }
        {
            int sm, sn;
            for(sm = 0; sm < n_sub_m; ++sm)
            {
                for(sn = 0; sn < n_sub_n; ++sn)
                {
                    int idx = sm * n_sub_n + sn;
                    nacc[idx] = rocke_mnb_wmma(b, wp.wmma_op, a_frag[sm], b_frag[sn], accs[idx]);
                    if(nacc[idx] == NULL)
                    {
                        return NULL;
                    }
                }
            }
        }
        rocke_b_scf_yield(b, nacc, n_acc);
    }
    rocke_b_region_leave(b);

    rocke_value_t* const* results = loop.op->results;

    /* Epilogue: scatter each <8 x float> sub-tile accumulator (row-guarded on M). */
    {
        int sm, sn, i;
        for(sm = 0; sm < n_sub_m; ++sm)
        {
            rocke_value_t* r0 = rocke_b_add(b, wm_base, rocke_b_const_i32(b, sm * t->warp_tile_m));
            for(sn = 0; sn < n_sub_n; ++sn)
            {
                rocke_value_t* acc = results[sm * n_sub_n + sn];
                rocke_value_t* out_col = rocke_b_add(
                    b, rocke_b_add(b, wn_base, rocke_b_const_i32(b, sn * t->warp_tile_n)), frag);
                for(i = 0; i < 8; ++i)
                {
                    rocke_value_t* h = rocke_b_trunc_f32_to_f16(b, rocke_b_vec_extract(b, acc, i));
                    rocke_value_t* out_row;
                    if(wp.split_k_by_half)
                    {
                        /* gfx12 column-distributed: out_row = r0 + (lane//16)*8 + i. */
                        out_row = rocke_b_add(
                            b, r0, rocke_b_add(b, half_k_elem, rocke_b_const_i32(b, i)));
                    }
                    else
                    {
                        /* gfx11 row-distributed: out_row = r0 + 2*i + (lane//16). */
                        out_row
                            = rocke_b_add(b, r0, rocke_b_add(b, rocke_b_const_i32(b, 2 * i), half));
                    }
                    {
                        rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_lt(b, out_row, M));
                        rocke_b_region_enter(b, iff.then_region);
                        rocke_b_global_store(
                            b, C, rocke_b_add(b, rocke_b_mul(b, out_row, cN), out_col), h, 1);
                        rocke_b_region_leave(b);
                    }
                }
            }
        }
    }

    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_matmul_nbits -- the validating dispatcher
 *  (matmul_nbits.py::build_matmul_nbits)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_matmul_nbits(rocke_ir_builder_t* b,
                                             const rocke_matmul_nbits_spec_t* spec,
                                             const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = ROCKE_MATMUL_NBITS_V1_ARCH; /* "gfx1151" */
        }

        /* ok, reason = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_matmul_nbits_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            char msg[384];
            snprintf(msg, sizeof(msg), "invalid matmul_nbits spec for %s: %s", arch, reason);
            /* Surface the Python ValueError on the sticky-error builder. */
            if(b->status == ROCKE_OK)
            {
                b->status = ROCKE_ERR_VALUE;
                rocke_mnb_copy_msg(b->err, sizeof(b->err), msg);
            }
            return NULL;
        }

        /* if spec.family in ("large_n", "skinny_n"): */
        if(spec->family != NULL
           && (strcmp(spec->family, "large_n") == 0 || strcmp(spec->family, "skinny_n") == 0))
        {
            if(spec->optimized)
            {
                return rocke_build_large_n_opt_matmul_nbits(b, spec, arch);
            }
            return rocke_build_large_n_matmul_nbits(b, spec, arch);
        }

        /* if spec.family == "decode_gemv": */
        if(spec->family != NULL && strcmp(spec->family, "decode_gemv") == 0)
        {
            rocke_matmul_nbits_decode_gemv_spec_t dgv;
            const char* scale_wire = NULL;

            if(rocke_matmul_nbits_scale_wire_dtype(spec->scale_dtype, &scale_wire) != ROCKE_OK)
            {
                return NULL;
            }
            memset(&dgv, 0, sizeof(dgv));
            dgv.N = spec->N;
            dgv.K = spec->K;
            dgv.group_size = spec->group_size;
            dgv.block_size = spec->block_size;
            dgv.scale_wire
                = (strcmp(scale_wire, "f16") == 0) ? ROCKE_NBITS_SCALE_F16 : ROCKE_NBITS_SCALE_F32;
            return rocke_build_decode_gemv_matmul_nbits(b, &dgv, arch);
        }

        /* Unreachable: validate_common_spec already rejects unknown families. */
        if(b->status == ROCKE_OK)
        {
            char msg[256];
            snprintf(msg,
                     sizeof(msg),
                     "unknown family %s",
                     spec->family != NULL ? spec->family : "(null)");
            b->status = ROCKE_ERR_VALUE;
            rocke_mnb_copy_msg(b->err, sizeof(b->err), msg);
        }
        return NULL;
    });
}

/* ===================================================================== *
 *  rocke_build_matmul_nbits_new -- init the builder with spec.kernel_name()
 *  then dispatch+build. (public convenience.)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_matmul_nbits_new(rocke_ir_builder_t* b,
                                                 const rocke_matmul_nbits_spec_t* spec,
                                                 const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_matmul_nbits_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_matmul_nbits(b, spec, arch);
    });
}

/* ===================================================================== *
 *  rocke_matmul_nbits_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_matmul_nbits_lower_to_llvm(const rocke_matmul_nbits_spec_t* spec,
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
        rocke_mnb_copy_msg(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MATMUL_NBITS_V1_ARCH; /* "gfx1151" */
    }

    kernel = rocke_build_matmul_nbits_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_matmul_nbits failed";
            }
            rocke_mnb_copy_msg(err, err_cap, m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
