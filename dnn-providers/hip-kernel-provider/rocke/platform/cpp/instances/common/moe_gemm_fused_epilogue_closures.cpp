// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_gemm_fused_epilogue-closures.c -- C99 port of the two MoE GEMM
 * fusion epilogue closures that are NOT carried by the value-type helper header
 * (helper_rocke.instances.common.moe_gemm_fused.h already ports
 * _emit_down_reduce_epilogue_atomic + _emit_cshuffle_stage):
 *
 *   _emit_gate_up_silu_epilogue_default   (moe_gemm_fused.py lines 912-1016)
 *       -> rocke_moe_emit_gate_up_silu_epilogue_default
 *   _emit_interleaved_silu_epilogue       (moe_gemm_fused.py lines 1394-1527)
 *       -> rocke_moe_emit_interleaved_silu_epilogue
 *
 * Both are FREE helpers taking explicit Values (the gate-up / interleaved
 * emit_compute drivers supply them from their ctx). The builder-call sequence is
 * byte-identical to the Python: per-lane silu/acc cell -> rocke_moe_emit_cshuffle_stage
 * into LDS, sync, then the vectorised global-store loop with the pad mask.
 *
 * Binds to the private internal header (prototypes + the value-type leaf/cshuffle
 * helpers it re-exports), instance_gemm_internal.h (_load_smem_scalar/_vec), and
 * rocke/ir.h (the C IRBuilder).
 */
#include <string.h>

#include "rocke/instance_gemm_internal.h" /* rocke_gemm_load_smem_scalar / _vec */
#include "rocke/instance_moe_gemm_fused_internal.h"
#include "rocke/ir.h"

/* _storage_dtype(spec): homogeneous A/B/C dtype -> rocke_type_t. Mirrors the
 * static helper in helper_rocke.instances.common.moe_gemm_fused.c (which is not
 * exported); re-derived here from the spec's A dtype string. */
static const rocke_type_t* rocke_moe_ep_storage_dtype(const rocke_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if(d == NULL)
    {
        return rocke_f16();
    }
    if(strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(d, "bf16") == 0)
    {
        return rocke_bf16();
    }
    return rocke_scalar_by_name(d);
}

/* ====================================================================== *
 *  _emit_gate_up_silu_epilogue_default  (lines 912-1016)
 * ====================================================================== */

/* Closure context for the `_silu_cell` cell-value callback (lines 961-968). */
typedef struct rocke_moe_silu_cell_ctx
{
    rocke_ir_builder_t* b;
    rocke_value_t* const* gate_accs;
    rocke_value_t* const* up_accs;
    int mfmas_n;
    const rocke_type_t* storage_dtype;
    rocke_value_t* one_f32;
    rocke_value_t* c_neg_log2e;
} rocke_moe_silu_cell_ctx_t;

/* _silu_cell(mi, ni, i): silu(gate)*up -> storage_dtype. */
static rocke_value_t* rocke_moe_silu_cell(int mi, int ni, int i, void* user)
{
    rocke_moe_silu_cell_ctx_t* c = (rocke_moe_silu_cell_ctx_t*)user;
    int flat = mi * c->mfmas_n + ni;
    rocke_value_t* g = rocke_b_vec_extract(c->b, c->gate_accs[flat], i);
    rocke_value_t* up = rocke_b_vec_extract(c->b, c->up_accs[flat], i);
    rocke_value_t* sm = rocke_moe_gemm_fused_silu_mul_f32(c->b, g, up, c->one_f32, c->c_neg_log2e);
    return rocke_b_cast_f32_to(c->b, sm, c->storage_dtype);
}

void rocke_moe_emit_gate_up_silu_epilogue_default(rocke_ir_builder_t* b,
                                                  const rocke_gemm_universal_spec_t* spec,
                                                  rocke_value_t* const* gate_accs,
                                                  rocke_value_t* const* up_accs,
                                                  int num_accs,
                                                  rocke_value_t* warp_m_idx,
                                                  rocke_value_t* warp_n_idx,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* block_m_off,
                                                  rocke_value_t* block_n_off,
                                                  rocke_value_t* M,
                                                  rocke_value_t* N,
                                                  rocke_value_t* Hidden,
                                                  int c_per_lane,
                                                  rocke_value_t* batch_off_c)
{
    (void)num_accs;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* storage_dtype = rocke_moe_ep_storage_dtype(spec);
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    rocke_value_t* c_neg_log2e = rocke_b_const_f32(b, -1.4426950408889634);
    rocke_value_t* one_f32 = rocke_b_const_f32(b, 1.0);
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;

    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    int hs[2] = {t->tile_m, t->tile_n};
    rocke_value_t* Cs = rocke_b_smem_alloc(b, storage_dtype, hs, 2, "Hidden_smem");

    /* MFMA-output (lane, slot) -> (ld_m, ld_n) via the C-warp tile distribution
     * (CWarpDstrEncoding); the silu-mul results stage into LDS via the cshuffle
     * stage. */
    rocke_moe_cwarp_decode_t cdec;
    if(!rocke_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }

    rocke_moe_silu_cell_ctx_t cell = {0};
    cell.b = b;
    cell.gate_accs = gate_accs;
    cell.up_accs = up_accs;
    cell.mfmas_n = mfmas_n;
    cell.storage_dtype = storage_dtype;
    cell.one_f32 = one_f32;
    cell.c_neg_log2e = c_neg_log2e;

    rocke_moe_emit_cshuffle_stage(
        b, spec, &cdec, Cs, storage_dtype, c_per_lane, rocke_moe_silu_cell, &cell);

    rocke_b_sync(b);

    /* Wide global stores from LDS in output layout. */
    int threads = spec->block_size;
    int store_vec = 8;
    while(store_vec > 1
          && ((t->tile_n % store_vec != 0) || ((t->tile_m * t->tile_n) / store_vec < threads)
              || (((t->tile_m * t->tile_n) / store_vec) % threads)))
    {
        store_vec /= 2;
    }

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* c_threads = rocke_b_const_i32(b, threads);
    int tile_n_div_vec = t->tile_n / store_vec;
    int vecs_per_thread = (t->tile_m * t->tile_n / store_vec) / threads;
    for(int e = 0; e < vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
        /* vec_idx -> (row, col_v) via magic-division unmerge (tile_n_div_vec is
         * the compile-time inner extent). */
        rocke_value_t* row = NULL;
        rocke_value_t* col_v = NULL;
        rocke_moe_magic_div_mod(b, vec_idx, tile_n_div_vec, &row, &col_v);
        rocke_value_t* col
            = (store_vec > 1) ? rocke_b_mul(b, col_v, rocke_b_const_i32(b, store_vec)) : col_v;

        rocke_value_t* c_m = rocke_b_add(b, block_m_off, row);
        rocke_value_t* c_n = rocke_b_add(b, block_n_off, col);
        rocke_value_t* c_off
            = rocke_b_add(b, batch_off_c, rocke_b_add(b, rocke_b_mul(b, c_m, N), c_n));

        rocke_value_t* in_bounds
            = rocke_moe_pad_in_bounds(b, c_m, c_n, M, N, pad_m, pad_n, store_vec);

        if(store_vec == 1)
        {
            rocke_value_t* h = rocke_gemm_load_smem_scalar(b, Cs, row, col, storage_dtype);
            if(in_bounds != NULL)
            {
                rocke_if_t g = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, g.then_region);
                rocke_b_global_store(b, Hidden, c_off, h, 2);
                rocke_b_region_leave(b);
            }
            else
            {
                rocke_b_global_store(b, Hidden, c_off, h, 2);
            }
        }
        else
        {
            rocke_value_t* hv = rocke_gemm_load_smem_vec(b, Cs, row, col, store_vec, storage_dtype);
            if(in_bounds != NULL)
            {
                rocke_if_t g = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, g.then_region);
                rocke_b_global_store_vN(b, Hidden, c_off, hv, store_vec, 0);
                rocke_b_region_leave(b);
            }
            else
            {
                rocke_b_global_store_vN(b, Hidden, c_off, hv, store_vec, 0);
            }
        }
    }
}

/* ====================================================================== *
 *  _emit_interleaved_silu_epilogue  (lines 1394-1527)
 * ====================================================================== */

/* Closure context for the `_acc_cell` cell-value callback (lines 1428-1430). */
typedef struct rocke_moe_acc_cell_ctx
{
    rocke_ir_builder_t* b;
    rocke_value_t* const* accs;
    int mfmas_n;
    const rocke_type_t* storage_dtype;
} rocke_moe_acc_cell_ctx_t;

/* _acc_cell(mi, ni, i): cast f32 acc slot -> storage_dtype. */
static rocke_value_t* rocke_moe_acc_cell(int mi, int ni, int i, void* user)
{
    rocke_moe_acc_cell_ctx_t* c = (rocke_moe_acc_cell_ctx_t*)user;
    rocke_value_t* acc = c->accs[mi * c->mfmas_n + ni];
    return rocke_b_cast_f32_to(c->b, rocke_b_vec_extract(c->b, acc, i), c->storage_dtype);
}

void rocke_moe_emit_interleaved_silu_epilogue(rocke_ir_builder_t* b,
                                              const rocke_gemm_universal_spec_t* spec,
                                              rocke_value_t* const* accs,
                                              int num_accs,
                                              rocke_value_t* C_smem,
                                              rocke_value_t* warp_m_idx,
                                              rocke_value_t* warp_n_idx,
                                              rocke_value_t* lane,
                                              rocke_value_t* block_m_off,
                                              rocke_value_t* block_n_off,
                                              rocke_value_t* M,
                                              rocke_value_t* N,
                                              rocke_value_t* Hidden,
                                              int c_per_lane,
                                              rocke_value_t* batch_off_c)
{
    (void)num_accs;
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_type_t* storage_dtype = rocke_moe_ep_storage_dtype(spec);
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    rocke_value_t* c_neg_log2e = rocke_b_const_f32(b, -1.4426950408889634);
    rocke_value_t* one_f32 = rocke_b_const_f32(b, 1.0);
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    /* 1) Accumulator -> LDS in normal output layout (M x 2I tile). The
     * MFMA-output (lane, slot) -> (ld_m, ld_n) decode is the C-warp tile
     * distribution; staging goes through the cshuffle stage. */
    rocke_moe_cwarp_decode_t cdec;
    if(!rocke_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }

    rocke_moe_acc_cell_ctx_t cell = {0};
    cell.b = b;
    cell.accs = accs;
    cell.mfmas_n = mfmas_n;
    cell.storage_dtype = storage_dtype;

    rocke_moe_emit_cshuffle_stage(
        b, spec, &cdec, C_smem, storage_dtype, c_per_lane, rocke_moe_acc_cell, &cell);

    rocke_b_sync(b);

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
    rocke_value_t* c_vec_h = rocke_b_const_i32(b, vec_h);
    rocke_value_t* n_base = NULL;
    rocke_value_t* n_base_rem = NULL;
    rocke_moe_magic_div_mod(b, block_n_off, 2, &n_base, &n_base_rem);
    for(int u = 0; u < units_per_thread; ++u)
    {
        rocke_value_t* linear_base = rocke_b_const_i32(b, u * threads * vec_h);
        rocke_value_t* linear_mul = rocke_b_mul(b, rocke_b_thread_id_x(b), c_vec_h);
        rocke_value_t* linear_h = rocke_b_add(b, linear_base, linear_mul);
        /* linear_h -> (row, hcol_local) via magic-division unmerge
         * (hidden_cols_per_tile is the compile-time inner extent). */
        rocke_value_t* row = NULL;
        rocke_value_t* hcol_local = NULL;
        rocke_moe_magic_div_mod(b, linear_h, hidden_cols_per_tile, &row, &hcol_local);
        rocke_value_t* pair_col = rocke_b_mul(b, hcol_local, rocke_b_const_i32(b, 2));
        rocke_value_t* c_m = rocke_b_add(b, block_m_off, row);
        rocke_value_t* c_n_start = rocke_b_add(b, n_base, hcol_local);
        rocke_value_t* off
            = rocke_b_add(b, batch_off_c, rocke_b_add(b, rocke_b_mul(b, c_m, N), c_n_start));

        if(vec_h == 1)
        {
            rocke_value_t* gate_h
                = rocke_gemm_load_smem_scalar(b, C_smem, row, pair_col, storage_dtype);
            rocke_value_t* up_h = rocke_gemm_load_smem_scalar(
                b, C_smem, row, rocke_b_add(b, pair_col, rocke_b_const_i32(b, 1)), storage_dtype);
            rocke_value_t* g = rocke_b_cast_to_f32(b, gate_h);
            rocke_value_t* up = rocke_b_cast_to_f32(b, up_h);
            rocke_value_t* out_v = rocke_b_cast_f32_to(
                b,
                rocke_moe_gemm_fused_silu_mul_f32(b, g, up, one_f32, c_neg_log2e),
                storage_dtype);

            rocke_value_t* in_bounds
                = rocke_moe_pad_in_bounds(b, c_m, c_n_start, M, N, pad_m, pad_n, 1);
            if(in_bounds != NULL)
            {
                rocke_if_t gd = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, gd.then_region);
                rocke_b_global_store(b, Hidden, off, out_v, 2);
                rocke_b_region_leave(b);
            }
            else
            {
                rocke_b_global_store(b, Hidden, off, out_v, 2);
            }
        }
        else
        {
            /* One wide LDS read returning <2*vec_h x dtype> with (gate_0, up_0,
             * ..., gate_{vh-1}, up_{vh-1}) interleaved. */
            rocke_value_t* gu_vec
                = rocke_gemm_load_smem_vec(b, C_smem, row, pair_col, 2 * vec_h, storage_dtype);
            rocke_value_t* h_scalars[ROCKE_MOE_MAX_VECS];
            for(int i = 0; i < vec_h; ++i)
            {
                rocke_value_t* g = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, gu_vec, 2 * i));
                rocke_value_t* up
                    = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, gu_vec, 2 * i + 1));
                h_scalars[i] = rocke_b_cast_f32_to(
                    b,
                    rocke_moe_gemm_fused_silu_mul_f32(b, g, up, one_f32, c_neg_log2e),
                    storage_dtype);
            }
            rocke_value_t* h_packed = rocke_b_vec_pack(b, h_scalars, vec_h, storage_dtype);

            /* vec_h consecutive columns; bounds-check the last one. */
            rocke_value_t* in_bounds
                = rocke_moe_pad_in_bounds(b, c_m, c_n_start, M, N, pad_m, pad_n, vec_h);
            if(in_bounds != NULL)
            {
                rocke_if_t gd = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, gd.then_region);
                rocke_b_global_store_vN(b, Hidden, off, h_packed, vec_h, 0);
                rocke_b_region_leave(b);
            }
            else
            {
                rocke_b_global_store_vN(b, Hidden, off, h_packed, vec_h, 0);
            }
        }
    }
}
