// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_gemm_fused_epilogue-closures.c -- C99 port of the two MoE GEMM
 * fusion epilogue closures that are NOT carried by the value-type helper header
 * (helper_ck_dsl.instances.common.moe_gemm_fused.h already ports
 * _emit_down_reduce_epilogue_atomic + _emit_cshuffle_stage):
 *
 *   _emit_gate_up_silu_epilogue_default   (moe_gemm_fused.py lines 912-1016)
 *       -> ckc_moe_emit_gate_up_silu_epilogue_default
 *   _emit_interleaved_silu_epilogue       (moe_gemm_fused.py lines 1394-1527)
 *       -> ckc_moe_emit_interleaved_silu_epilogue
 *
 * Both are FREE helpers taking explicit Values (the gate-up / interleaved
 * emit_compute drivers supply them from their ctx). The builder-call sequence is
 * byte-identical to the Python: per-lane silu/acc cell -> ckc_moe_emit_cshuffle_stage
 * into LDS, sync, then the vectorised global-store loop with the pad mask.
 *
 * Binds to the private internal header (prototypes + the value-type leaf/cshuffle
 * helpers it re-exports), instance_gemm_internal.h (_load_smem_scalar/_vec), and
 * ckc/ir.h (the C IRBuilder).
 */
#include <string.h>

#include "ckc/instance_gemm_internal.h" /* ckc_gemm_load_smem_scalar / _vec */
#include "ckc/instance_moe_gemm_fused_internal.h"
#include "ckc/ir.h"

/* _storage_dtype(spec): homogeneous A/B/C dtype -> ckc_type_t. Mirrors the
 * static helper in helper_ck_dsl.instances.common.moe_gemm_fused.c (which is not
 * exported); re-derived here from the spec's A dtype string. */
static const ckc_type_t* ckc_moe_ep_storage_dtype(const ckc_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if(d == NULL)
    {
        return ckc_f16();
    }
    if(strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return ckc_f16();
    }
    if(strcmp(d, "bf16") == 0)
    {
        return ckc_bf16();
    }
    return ckc_scalar_by_name(d);
}

/* ====================================================================== *
 *  _emit_gate_up_silu_epilogue_default  (lines 912-1016)
 * ====================================================================== */

/* Closure context for the `_silu_cell` cell-value callback (lines 961-968). */
typedef struct ckc_moe_silu_cell_ctx
{
    ckc_ir_builder_t* b;
    ckc_value_t* const* gate_accs;
    ckc_value_t* const* up_accs;
    int mfmas_n;
    const ckc_type_t* storage_dtype;
    ckc_value_t* one_f32;
    ckc_value_t* c_neg_log2e;
} ckc_moe_silu_cell_ctx_t;

/* _silu_cell(mi, ni, i): silu(gate)*up -> storage_dtype. */
static ckc_value_t* ckc_moe_silu_cell(int mi, int ni, int i, void* user)
{
    ckc_moe_silu_cell_ctx_t* c = (ckc_moe_silu_cell_ctx_t*)user;
    int flat = mi * c->mfmas_n + ni;
    ckc_value_t* g = ckc_b_vec_extract(c->b, c->gate_accs[flat], i);
    ckc_value_t* up = ckc_b_vec_extract(c->b, c->up_accs[flat], i);
    ckc_value_t* sm = ckc_moe_gemm_fused_silu_mul_f32(c->b, g, up, c->one_f32, c->c_neg_log2e);
    return ckc_b_cast_f32_to(c->b, sm, c->storage_dtype);
}

void ckc_moe_emit_gate_up_silu_epilogue_default(ckc_ir_builder_t* b,
                                                const ckc_gemm_universal_spec_t* spec,
                                                ckc_value_t* const* gate_accs,
                                                ckc_value_t* const* up_accs,
                                                int num_accs,
                                                ckc_value_t* warp_m_idx,
                                                ckc_value_t* warp_n_idx,
                                                ckc_value_t* lane,
                                                ckc_value_t* block_m_off,
                                                ckc_value_t* block_n_off,
                                                ckc_value_t* M,
                                                ckc_value_t* N,
                                                ckc_value_t* Hidden,
                                                int c_per_lane,
                                                ckc_value_t* batch_off_c)
{
    (void)num_accs;
    const ckc_gemm_tile_spec_t* t = &spec->tile;
    const ckc_type_t* storage_dtype = ckc_moe_ep_storage_dtype(spec);
    int mfmas_m = ckc_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = ckc_gemm_tile_mfmas_per_warp_n(t);
    ckc_value_t* c_neg_log2e = ckc_b_const_f32(b, -1.4426950408889634);
    ckc_value_t* one_f32 = ckc_b_const_f32(b, 1.0);
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;

    ckc_value_t* warp_m_off
        = ckc_b_mul(b, warp_m_idx, ckc_b_const_i32(b, mfmas_m * t->warp_tile_m));
    ckc_value_t* warp_n_off
        = ckc_b_mul(b, warp_n_idx, ckc_b_const_i32(b, mfmas_n * t->warp_tile_n));

    int hs[2] = {t->tile_m, t->tile_n};
    ckc_value_t* Cs = ckc_b_smem_alloc(b, storage_dtype, hs, 2, "Hidden_smem");

    /* MFMA-output (lane, slot) -> (ld_m, ld_n) via the C-warp tile distribution
     * (CWarpDstrEncoding); the silu-mul results stage into LDS via the cshuffle
     * stage. */
    ckc_moe_cwarp_decode_t cdec;
    if(!ckc_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }

    ckc_moe_silu_cell_ctx_t cell = {0};
    cell.b = b;
    cell.gate_accs = gate_accs;
    cell.up_accs = up_accs;
    cell.mfmas_n = mfmas_n;
    cell.storage_dtype = storage_dtype;
    cell.one_f32 = one_f32;
    cell.c_neg_log2e = c_neg_log2e;

    ckc_moe_emit_cshuffle_stage(
        b, spec, &cdec, Cs, storage_dtype, c_per_lane, ckc_moe_silu_cell, &cell);

    ckc_b_sync(b);

    /* Wide global stores from LDS in output layout. */
    int threads = spec->block_size;
    int store_vec = 8;
    while(store_vec > 1
          && ((t->tile_n % store_vec != 0) || ((t->tile_m * t->tile_n) / store_vec < threads)
              || (((t->tile_m * t->tile_n) / store_vec) % threads)))
    {
        store_vec /= 2;
    }

    ckc_value_t* tid = ckc_b_thread_id_x(b);
    ckc_value_t* c_threads = ckc_b_const_i32(b, threads);
    int tile_n_div_vec = t->tile_n / store_vec;
    int vecs_per_thread = (t->tile_m * t->tile_n / store_vec) / threads;
    for(int e = 0; e < vecs_per_thread; ++e)
    {
        ckc_value_t* vec_idx = ckc_b_add(b, ckc_b_mul(b, ckc_b_const_i32(b, e), c_threads), tid);
        /* vec_idx -> (row, col_v) via magic-division unmerge (tile_n_div_vec is
         * the compile-time inner extent). */
        ckc_value_t* row = NULL;
        ckc_value_t* col_v = NULL;
        ckc_moe_magic_div_mod(b, vec_idx, tile_n_div_vec, &row, &col_v);
        ckc_value_t* col
            = (store_vec > 1) ? ckc_b_mul(b, col_v, ckc_b_const_i32(b, store_vec)) : col_v;

        ckc_value_t* c_m = ckc_b_add(b, block_m_off, row);
        ckc_value_t* c_n = ckc_b_add(b, block_n_off, col);
        ckc_value_t* c_off = ckc_b_add(b, batch_off_c, ckc_b_add(b, ckc_b_mul(b, c_m, N), c_n));

        ckc_value_t* in_bounds = ckc_moe_pad_in_bounds(b, c_m, c_n, M, N, pad_m, pad_n, store_vec);

        if(store_vec == 1)
        {
            ckc_value_t* h = ckc_gemm_load_smem_scalar(b, Cs, row, col, storage_dtype);
            if(in_bounds != NULL)
            {
                ckc_if_t g = ckc_b_scf_if(b, in_bounds);
                ckc_b_region_enter(b, g.then_region);
                ckc_b_global_store(b, Hidden, c_off, h, 2);
                ckc_b_region_leave(b);
            }
            else
            {
                ckc_b_global_store(b, Hidden, c_off, h, 2);
            }
        }
        else
        {
            ckc_value_t* hv = ckc_gemm_load_smem_vec(b, Cs, row, col, store_vec, storage_dtype);
            if(in_bounds != NULL)
            {
                ckc_if_t g = ckc_b_scf_if(b, in_bounds);
                ckc_b_region_enter(b, g.then_region);
                ckc_b_global_store_vN(b, Hidden, c_off, hv, store_vec, 0);
                ckc_b_region_leave(b);
            }
            else
            {
                ckc_b_global_store_vN(b, Hidden, c_off, hv, store_vec, 0);
            }
        }
    }
}

/* ====================================================================== *
 *  _emit_interleaved_silu_epilogue  (lines 1394-1527)
 * ====================================================================== */

/* Closure context for the `_acc_cell` cell-value callback (lines 1428-1430). */
typedef struct ckc_moe_acc_cell_ctx
{
    ckc_ir_builder_t* b;
    ckc_value_t* const* accs;
    int mfmas_n;
    const ckc_type_t* storage_dtype;
} ckc_moe_acc_cell_ctx_t;

/* _acc_cell(mi, ni, i): cast f32 acc slot -> storage_dtype. */
static ckc_value_t* ckc_moe_acc_cell(int mi, int ni, int i, void* user)
{
    ckc_moe_acc_cell_ctx_t* c = (ckc_moe_acc_cell_ctx_t*)user;
    ckc_value_t* acc = c->accs[mi * c->mfmas_n + ni];
    return ckc_b_cast_f32_to(c->b, ckc_b_vec_extract(c->b, acc, i), c->storage_dtype);
}

void ckc_moe_emit_interleaved_silu_epilogue(ckc_ir_builder_t* b,
                                            const ckc_gemm_universal_spec_t* spec,
                                            ckc_value_t* const* accs,
                                            int num_accs,
                                            ckc_value_t* C_smem,
                                            ckc_value_t* warp_m_idx,
                                            ckc_value_t* warp_n_idx,
                                            ckc_value_t* lane,
                                            ckc_value_t* block_m_off,
                                            ckc_value_t* block_n_off,
                                            ckc_value_t* M,
                                            ckc_value_t* N,
                                            ckc_value_t* Hidden,
                                            int c_per_lane,
                                            ckc_value_t* batch_off_c)
{
    (void)num_accs;
    const ckc_gemm_tile_spec_t* t = &spec->tile;
    const ckc_type_t* storage_dtype = ckc_moe_ep_storage_dtype(spec);
    int mfmas_m = ckc_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = ckc_gemm_tile_mfmas_per_warp_n(t);
    ckc_value_t* c_neg_log2e = ckc_b_const_f32(b, -1.4426950408889634);
    ckc_value_t* one_f32 = ckc_b_const_f32(b, 1.0);
    ckc_value_t* warp_m_off
        = ckc_b_mul(b, warp_m_idx, ckc_b_const_i32(b, mfmas_m * t->warp_tile_m));
    ckc_value_t* warp_n_off
        = ckc_b_mul(b, warp_n_idx, ckc_b_const_i32(b, mfmas_n * t->warp_tile_n));

    /* 1) Accumulator -> LDS in normal output layout (M x 2I tile). The
     * MFMA-output (lane, slot) -> (ld_m, ld_n) decode is the C-warp tile
     * distribution; staging goes through the cshuffle stage. */
    ckc_moe_cwarp_decode_t cdec;
    if(!ckc_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }

    ckc_moe_acc_cell_ctx_t cell = {0};
    cell.b = b;
    cell.accs = accs;
    cell.mfmas_n = mfmas_n;
    cell.storage_dtype = storage_dtype;

    ckc_moe_emit_cshuffle_stage(
        b, spec, &cdec, C_smem, storage_dtype, c_per_lane, ckc_moe_acc_cell, &cell);

    ckc_b_sync(b);

    /* 2) LDS interleaved pairs -> Hidden. Vectorised over vec_h adjacent hidden
     * columns per thread per chunk. */
    int threads = spec->block_size;
    int hidden_cols_per_tile = t->tile_n / 2;
    int total_hidden = t->tile_m * hidden_cols_per_tile;
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;

    /* Largest power-of-two vec_h s.t. hidden_cols_per_tile % vec_h == 0 and
     * total_hidden % (threads*vec_h) == 0; 2*vec_h capped at smem_load_vN width. */
    int vec_h = 4;
    while(vec_h > 1 && (hidden_cols_per_tile % vec_h != 0 || total_hidden % (threads * vec_h) != 0))
    {
        vec_h /= 2;
    }

    int units_per_thread = total_hidden / (threads * vec_h);
    ckc_value_t* c_vec_h = ckc_b_const_i32(b, vec_h);
    ckc_value_t* n_base = NULL;
    ckc_value_t* n_base_rem = NULL;
    ckc_moe_magic_div_mod(b, block_n_off, 2, &n_base, &n_base_rem);
    for(int u = 0; u < units_per_thread; ++u)
    {
        ckc_value_t* linear_base = ckc_b_const_i32(b, u * threads * vec_h);
        ckc_value_t* linear_mul = ckc_b_mul(b, ckc_b_thread_id_x(b), c_vec_h);
        ckc_value_t* linear_h = ckc_b_add(b, linear_base, linear_mul);
        /* linear_h -> (row, hcol_local) via magic-division unmerge
         * (hidden_cols_per_tile is the compile-time inner extent). */
        ckc_value_t* row = NULL;
        ckc_value_t* hcol_local = NULL;
        ckc_moe_magic_div_mod(b, linear_h, hidden_cols_per_tile, &row, &hcol_local);
        ckc_value_t* pair_col = ckc_b_mul(b, hcol_local, ckc_b_const_i32(b, 2));
        ckc_value_t* c_m = ckc_b_add(b, block_m_off, row);
        ckc_value_t* c_n_start = ckc_b_add(b, n_base, hcol_local);
        ckc_value_t* off = ckc_b_add(b, batch_off_c, ckc_b_add(b, ckc_b_mul(b, c_m, N), c_n_start));

        if(vec_h == 1)
        {
            ckc_value_t* gate_h
                = ckc_gemm_load_smem_scalar(b, C_smem, row, pair_col, storage_dtype);
            ckc_value_t* up_h = ckc_gemm_load_smem_scalar(
                b, C_smem, row, ckc_b_add(b, pair_col, ckc_b_const_i32(b, 1)), storage_dtype);
            ckc_value_t* g = ckc_b_cast_to_f32(b, gate_h);
            ckc_value_t* up = ckc_b_cast_to_f32(b, up_h);
            ckc_value_t* out_v = ckc_b_cast_f32_to(
                b, ckc_moe_gemm_fused_silu_mul_f32(b, g, up, one_f32, c_neg_log2e), storage_dtype);

            ckc_value_t* in_bounds
                = ckc_moe_pad_in_bounds(b, c_m, c_n_start, M, N, pad_m, pad_n, 1);
            if(in_bounds != NULL)
            {
                ckc_if_t gd = ckc_b_scf_if(b, in_bounds);
                ckc_b_region_enter(b, gd.then_region);
                ckc_b_global_store(b, Hidden, off, out_v, 2);
                ckc_b_region_leave(b);
            }
            else
            {
                ckc_b_global_store(b, Hidden, off, out_v, 2);
            }
        }
        else
        {
            /* One wide LDS read returning <2*vec_h x dtype> with (gate_0, up_0,
             * ..., gate_{vh-1}, up_{vh-1}) interleaved. */
            ckc_value_t* gu_vec
                = ckc_gemm_load_smem_vec(b, C_smem, row, pair_col, 2 * vec_h, storage_dtype);
            ckc_value_t* h_scalars[CKC_MOE_MAX_VECS];
            for(int i = 0; i < vec_h; ++i)
            {
                ckc_value_t* g = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, gu_vec, 2 * i));
                ckc_value_t* up = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, gu_vec, 2 * i + 1));
                h_scalars[i] = ckc_b_cast_f32_to(
                    b,
                    ckc_moe_gemm_fused_silu_mul_f32(b, g, up, one_f32, c_neg_log2e),
                    storage_dtype);
            }
            ckc_value_t* h_packed = ckc_b_vec_pack(b, h_scalars, vec_h, storage_dtype);

            /* vec_h consecutive columns; bounds-check the last one. */
            ckc_value_t* in_bounds
                = ckc_moe_pad_in_bounds(b, c_m, c_n_start, M, N, pad_m, pad_n, vec_h);
            if(in_bounds != NULL)
            {
                ckc_if_t gd = ckc_b_scf_if(b, in_bounds);
                ckc_b_region_enter(b, gd.then_region);
                ckc_b_global_store_vN(b, Hidden, off, h_packed, vec_h, 0);
                ckc_b_region_leave(b);
            }
            else
            {
                ckc_b_global_store_vN(b, Hidden, off, h_packed, vec_h, 0);
            }
        }
    }
}
