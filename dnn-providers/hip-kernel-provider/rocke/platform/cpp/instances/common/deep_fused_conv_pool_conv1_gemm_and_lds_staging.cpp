// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_deep_fused_conv_pool_conv1_gemm_and_lds_staging.c -- chunked C99 port
 * of three builder-emit helpers from
 * rocke/instances/common/deep_fused_conv_pool.py:
 *
 *   _stage_accumulators_to_cshuffle_lds  (py 388-456)
 *   _load_conv1_weights_to_lds           (py 459-498)
 *   _emit_conv1_1x1                      (py 816-910)
 *
 * Scope: the conv1 1x1 GEMM and the two disjoint LDS producers it consumes (the
 * conv0 accumulator C-shuffle stage and the W1 weight load). The merged barrier,
 * the maxpool reductions, the conv0 A-loaders and the driver live in sibling
 * part-files / the public driver and are reached through the shared internal
 * header.
 *
 * BYTE-IDENTICAL BUILDER SEQUENCE. Every rocke_b_* call below mirrors the Python
 * builder call order one-for-one. Where the Python reaches a peer helper that is
 * NOT directly includable as a public C symbol the body either (a) inlines the
 * peer's exact builder-op sequence with ported primitives (this is possible for
 * the cshuffle store loop and load_smem_frag_contiguous_f16, whose op sequences
 * use only ported rocke_b_* ops) or (b) calls the peer by its stable C-ABI symbol
 * via a local forward declaration (resolved at link; see the NAMED GAP below).
 *
 * PEER NOTES.
 *   - _cshuffle_acc_distribution(c_frag_len) + LoadStoreTraits(vector_dim_y=1,
 *     scalar_per_vector=1) + store_tile_cshuffle(coord_fn=...): for THIS
 *     distribution iterate_accesses() yields exactly c_frag_len accesses with
 *     y_base==(i,0) (i in [0,c_frag_len)) and one scalar per access, so the whole
 *     producer collapses to the per-slot loop reproduced inline below -- no
 *     distribution object is materialised and the emitted op stream is identical.
 *   - make_lds_view / CoalescedTileLoader / op layout coords / IRBuilder.mma are
 *     ported (tensor_view / loads / arch / ir headers).
 *   - load_smem_frag_contiguous_f16 (mfma_gemm_inner.py) is inlined: its op
 *     sequence is pure ported primitives.
 *   - _apply_accumulator_epilogue is a conv_implicit_gemm peer; its C symbol
 *     rocke_conv_apply_accumulator_epilogue lives in a peer-private header whose
 *     ConvAccumulatorEpilogue struct collides with this TU's. NAMED GAP (blocked
 *     on a non-conflicting shared header): it is called via a local forward
 *     declaration (link-resolved) until the peer epilogue type is exposed through
 *     a header that does not clash with this TU's struct of the same name.
 *   - conv_spec.warp_tile_k is read from spec->warp_tile_k: spec.conv_spec()
 *     constructs the ImplicitGemmConvSpec with warp_tile_k=self.warp_tile_k, so
 *     the two are equal by construction and spec is the non-opaque handle here.
 */
#include "rocke/instance_deep_fused_conv_pool_internal.h"

#include "rocke/helper_rocke.core.arch.h" /* rocke_mmaop_*_layout, layout coord */
#include "rocke/helper_rocke.helpers.epilogues.h" /* rocke_warp_grid_t accessors */
#include "rocke/helper_rocke.helpers.loads.h" /* CoalescedTileLoader */
#include "rocke/helper_rocke.helpers.tensor_view.h" /* make_lds_view, rocke_tensor_view_t */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include <stddef.h>

/* ------------------------------------------------------------------ *
 * Peer forward declaration (link-resolved; see PEER NOTES).
 * _apply_accumulator_epilogue(b, epilogue, accs[n], out_accs[n]).
 * ------------------------------------------------------------------ */
/* NAMED GAP: expose this through a non-conflicting public/internal header so the
 * forward declaration can be dropped (the peer ConvAccumulatorEpilogue struct
 * currently clashes with this TU's struct of the same name). Link-resolved. */
/* C++ build: cross-TU C-ABI helper; forward decl must be extern "C" so the
 * reference is not mangled. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
void rocke_conv_apply_accumulator_epilogue(rocke_ir_builder_t* b,
                                           const rocke_conv_acc_epilogue_t* epilogue,
                                           rocke_value_t* const* accs,
                                           int num_accs,
                                           rocke_value_t** out_accs);
#ifdef __cplusplus
}
#endif

/* ================================================================== *
 * load_smem_frag_contiguous_f16 (mfma_gemm_inner.py 322-365), inlined.
 *
 * Read a contiguous f16 operand fragment smem[row, col_base:col_base+frag_len].
 * needs_mask=false && frag_len in {2,4,8}: one wide vector ds_read.
 * else: frag_len scalar reads with a per-element (col < valid_k) zero-fill mask.
 * ================================================================== */
static rocke_value_t* dfcp_load_smem_frag_contiguous_f16(rocke_ir_builder_t* b,
                                                         rocke_value_t* smem,
                                                         rocke_value_t* row,
                                                         rocke_value_t* col_base,
                                                         int frag_len,
                                                         bool needs_mask,
                                                         int valid_k)
{
    if(!needs_mask && (frag_len == 2 || frag_len == 4 || frag_len == 8))
    {
        rocke_value_t* idx[2];
        idx[0] = row;
        idx[1] = col_base;
        return rocke_b_smem_load_vN_f16(b, smem, idx, 2, frag_len);
    }

    {
        rocke_value_t* zero_h = rocke_b_trunc_f32_to_f16(b, rocke_b_const_f32(b, 0.0));
        rocke_value_t* c_valid_k = rocke_b_const_i32(b, valid_k);
        rocke_value_t* elems[64]; /* frag_len bounded by op.{a,b}_frag_len (tiny) */
        rocke_value_t* first_type_src = NULL;
        int i;
        if(frag_len <= 0 || frag_len > (int)(sizeof(elems) / sizeof(elems[0])))
        {
            (void)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "deep_fused_conv_pool: frag_len %d out of range", frag_len);
            return NULL;
        }
        for(i = 0; i < frag_len; ++i)
        {
            rocke_value_t* col = rocke_b_add(b, col_base, rocke_b_const_i32(b, i));
            rocke_value_t* idx[2];
            rocke_value_t* raw;
            rocke_value_t* ok;
            idx[0] = row;
            idx[1] = col;
            raw = rocke_b_vec_extract(b, rocke_b_smem_load_vN_f16(b, smem, idx, 2, 1), 0);
            ok = rocke_b_cmp_lt(b, col, c_valid_k);
            elems[i] = rocke_b_select(b, ok, raw, zero_h);
        }
        first_type_src = elems[0];
        return rocke_b_vec_pack(
            b, elems, frag_len, first_type_src != NULL ? first_type_src->type : NULL);
    }
}

/* ================================================================== *
 * _stage_accumulators_to_cshuffle_lds (py 388-456)
 *
 * Publish MMA accumulators to a row-major [tile_m, tile_n] LDS tile, fully
 * op-driven via op->c_frag_len + op->c_layout()->coord + the per-warp tile
 * geometry. sync=false defers the trailing barrier so the caller can merge it
 * with the disjoint W1 producer barrier.
 * ================================================================== */
rocke_value_t* rocke_dfcp_stage_accumulators_to_cshuffle_lds(rocke_ir_builder_t* b,
                                                             const rocke_mma_op_t* op,
                                                             rocke_value_t* const* accs,
                                                             size_t num_accs,
                                                             const rocke_warp_grid_t* grid,
                                                             bool sync)
{
    int c_frag_len;
    int mfmas_m;
    int mfmas_n;
    int shape[2];
    rocke_tensor_view_t c_view;
    rocke_value_t* c_smem;
    const rocke_arch_layout_map_t* c_map;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    int mi;
    int ni;

    if(b == NULL || op == NULL || grid == NULL)
    {
        return NULL;
    }

    c_frag_len = op->c_frag_len;
    mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    if((int)num_accs != mfmas_m * mfmas_n)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "expected %d accs, got %d", mfmas_m * mfmas_n, (int)num_accs);
    }

    /* LdsLayout.cshuffle(tile_m, tile_n): row-major [tile_m, tile_n], no swizzle,
     * k_pad=0. storage_shape(tile_m) == (tile_m, tile_n). make_lds_view allocates
     * the addrspace(3) tile; c_smem == c_view.base (the store target). */
    shape[0] = grid->tile_m;
    shape[1] = grid->tile_n;
    if(rocke_make_lds_view(b, &c_view, rocke_f16(), shape, 2, "DeepFusionC_smem", NULL) != ROCKE_OK)
    {
        return NULL;
    }
    c_smem = c_view.base;

    /* c_window = c_view.tile(storage_shape, [b.const_i32(0), b.const_i32(0)])
     * (py 425-428). The window itself is unused (the stores below go through
     * coord_fn on c_smem directly), but Python still emits the two origin
     * const_i32(0) values, each consuming an SSA counter (they are DCE'd before
     * printing). Mirror them so the @DeepFusionC_smem global id and every later
     * numbered SSA name line up with the Python reference. */
    (void)rocke_b_const_i32(b, 0);
    (void)rocke_b_const_i32(b, 0);

    warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    c_map = rocke_mmaop_c_layout(op, b);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    for(mi = 0; mi < mfmas_m; ++mi)
    {
        for(ni = 0; ni < mfmas_n; ++ni)
        {
            rocke_value_t* acc = accs[mi * mfmas_n + ni];
            rocke_value_t* acc_h = rocke_b_vec_trunc_f32_to_f16(b, acc);
            rocke_value_t* tile_m_base;
            rocke_value_t* tile_n_base;
            int i;

            /* make_static_distributed_tensor + store_tile_cshuffle(coord_fn) over
             * _cshuffle_acc_distribution(c_frag_len) with vector_dim_y=1,
             * scalar_per_vector=1: iterate_accesses() yields y_base==(i,0) for
             * i in [0,c_frag_len), one scalar each. The store target is the LDS
             * view base; coords come from coord_fn.
             *
             * Python fills the distributed tensor FIRST (the dt.set loop at
             * py 441-442 emits all c_frag_len `vec_extract`s up front), THEN
             * store_tile_cshuffle consumes it (emitting the address math +
             * stores). The two phases must stay separate so the emitted IR is
             * byte-identical: all extractelements precede the stores, not
             * interleaved one-per-store. */
            rocke_value_t* scalars[16];
            if(c_frag_len > (int)(sizeof(scalars) / sizeof(scalars[0])))
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "cshuffle stage: c_frag_len=%d exceeds cap", c_frag_len);
                return NULL;
            }
            /* Phase 1: dt.set([i,0], vec_extract(acc_h, i)) for all i. */
            for(i = 0; i < c_frag_len; ++i)
            {
                scalars[i] = rocke_b_vec_extract(b, acc_h, i);
            }
            /* tile_*_base (py 444-445) are computed AFTER the dt.set extracts and
             * BEFORE store_tile_cshuffle -- keep that ordering for byte parity. */
            tile_m_base = rocke_b_add(b, warp_m_off, rocke_b_const_i32(b, mi * op->m));
            tile_n_base = rocke_b_add(b, warp_n_off, rocke_b_const_i32(b, ni * op->n));
            /* Phase 2: store_tile_cshuffle -- per access, address math + store. */
            for(i = 0; i < c_frag_len; ++i)
            {
                rocke_value_t* row_in_atom = NULL;
                rocke_value_t* col_in_atom = NULL;
                rocke_value_t* coords[2];
                if(!rocke_arch_layout_map_coord(
                       c_map, b, grid->lane, i, &row_in_atom, &col_in_atom))
                {
                    return NULL;
                }
                coords[0] = rocke_b_add(b, tile_m_base, row_in_atom);
                coords[1] = rocke_b_add(b, tile_n_base, col_in_atom);
                rocke_b_smem_store_vN(b, c_smem, coords, 2, scalars[i], 1);
            }
        }
    }

    if(sync)
    {
        rocke_b_sync(b);
    }
    return c_smem;
}

/* ================================================================== *
 * _load_conv1_weights_to_lds (py 459-498)
 *
 * Load W1[K1, K0] into a padded row-major LDS tile [tile_n, K] via the shared
 * CoalescedTileLoader. sync=false defers the trailing barrier so it merges with
 * the conv0 cshuffle producer barrier.
 * ================================================================== */

/* descriptor closure environment for the W1 loader. */
typedef struct dfcp_w1_desc_env
{
    rocke_value_t* c_k0; /* const_i32(conv.K)              */
    rocke_value_t* c_k1; /* const_i32(conv1_channels)     */
} dfcp_w1_desc_env_t;

/* descriptor(b, row, col) -> (off, valid):
 *   row_ok = row < c_k1; col_ok = col < c_k0; valid = row_ok && col_ok
 *   off = row * c_k0 + col */
static rocke_value_t* dfcp_w1_descriptor(rocke_ir_builder_t* b,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         rocke_value_t** out_valid,
                                         void* user)
{
    dfcp_w1_desc_env_t* env = (dfcp_w1_desc_env_t*)user;
    rocke_value_t* row_ok = rocke_b_cmp_lt(b, row, env->c_k1);
    rocke_value_t* col_ok = rocke_b_cmp_lt(b, col, env->c_k0);
    rocke_value_t* valid = rocke_b_land(b, row_ok, col_ok);
    rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, row, env->c_k0), col);
    if(out_valid != NULL)
    {
        *out_valid = valid;
    }
    return off;
}

rocke_value_t* rocke_dfcp_load_conv1_weights_to_lds(rocke_ir_builder_t* b,
                                                    const rocke_deep_fused_conv_pool_spec_t* spec,
                                                    rocke_value_t* w1_rsrc,
                                                    const rocke_warp_grid_t* grid,
                                                    bool sync)
{
    int shape[2];
    rocke_value_t* w1_smem;
    rocke_coalesced_tile_loader_t loader;
    dfcp_w1_desc_env_t env;
    int K0;
    int K1;
    int block_size;

    if(b == NULL || spec == NULL || grid == NULL)
    {
        return NULL;
    }

    K0 = spec->problem.conv.K;
    K1 = rocke_fused_conv_pool_problem_conv1_channels(&spec->problem);
    block_size = rocke_deep_fused_conv_pool_spec_block_size(spec);

    /* b.smem_alloc(F16, [tile_n, K], "W1_smem"). */
    shape[0] = spec->tile_n;
    shape[1] = K0;
    w1_smem = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "W1_smem");
    if(w1_smem == NULL)
    {
        return NULL;
    }

    /* CoalescedTileLoader.from_tile(tile_rows=tile_n, tile_cols=K,
     *   block_size=block_size, max_vec=8). use_buffer_rsrc default True. */
    if(rocke_coalesced_tile_loader_from_tile(spec->tile_n, K0, block_size, 8, true, &loader)
       != ROCKE_OK)
    {
        (void)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "deep_fused_conv_pool: W1 loader from_tile failed");
        return NULL;
    }

    env.c_k0 = rocke_b_const_i32(b, K0);
    env.c_k1 = rocke_b_const_i32(b, K1);

    /* loader.load(tid=grid.tid, smem_dst=w1_smem, descriptor=..., rsrc=w1_rsrc). */
    rocke_coalesced_tile_loader_load(
        b, &loader, grid->tid, w1_smem, dfcp_w1_descriptor, &env, w1_rsrc, NULL);

    if(sync)
    {
        rocke_b_sync(b);
    }
    return w1_smem;
}

/* ================================================================== *
 * _emit_conv1_1x1 (py 816-910)
 *
 * Compute conv1 as a 1x1 GEMM over the staged conv0 activations, op-driven:
 * operand lane->(mn,k) decomposition from op->a_layout()/b_layout()->coord,
 * fragment widths op->{a,b}_frag_len, matmul via the target-neutral mma. The
 * K-tail mask is handled by load_smem_frag_contiguous_f16. defer_epilogue=true
 * returns the raw fp32 accumulators; else applies spec.conv1_epilogue.
 * ================================================================== */
rocke_status_t rocke_dfcp_emit_conv1_1x1(rocke_ir_builder_t* b,
                                         const rocke_deep_fused_conv_pool_spec_t* spec,
                                         const rocke_implicit_gemm_conv_spec_t* conv_spec,
                                         const rocke_mma_op_t* op,
                                         rocke_value_t* c0_smem,
                                         rocke_value_t* w1_smem,
                                         const rocke_warp_grid_t* grid,
                                         bool defer_epilogue,
                                         rocke_value_t** out_accs,
                                         size_t out_cap,
                                         size_t* out_n)
{
    const rocke_arch_layout_map_t* a_map;
    const rocke_arch_layout_map_t* b_map;
    rocke_value_t* a_mn_in_atom = NULL;
    rocke_value_t* a_k_base = NULL;
    rocke_value_t* b_k_base = NULL;
    rocke_value_t* b_mn_in_atom = NULL;
    int a_frag;
    int b_frag;
    int mfmas_m;
    int mfmas_n;
    int K0;
    int conv1_tile_k;
    int warp_tile_k;
    int k_atoms;
    int k_chunks;
    bool needs_mask;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* accs[ROCKE_DFCP_MAX_ACCS];
    int num_accs;
    int flat;
    int k_chunk;

    (void)conv_spec; /* warp_tile_k read from spec (equal by construction). */

    if(out_n != NULL)
    {
        *out_n = 0;
    }
    if(b == NULL || spec == NULL || op == NULL || grid == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    a_map = rocke_mmaop_a_layout(op, b);
    b_map = rocke_mmaop_b_layout(op, b);
    if(!rocke_ir_builder_ok(b))
    {
        return rocke_ir_builder_status(b);
    }
    /* a_map.coord(lane, 0) -> (a_mn_in_atom, a_k_base);
     * b_map.coord(lane, 0) -> (b_k_base, b_mn_in_atom). */
    if(!rocke_arch_layout_map_coord(a_map, b, grid->lane, 0, &a_mn_in_atom, &a_k_base))
    {
        return rocke_ir_builder_status(b);
    }
    if(!rocke_arch_layout_map_coord(b_map, b, grid->lane, 0, &b_k_base, &b_mn_in_atom))
    {
        return rocke_ir_builder_status(b);
    }

    a_frag = op->a_frag_len;
    b_frag = op->b_frag_len;
    mfmas_m = rocke_warp_grid_mfmas_per_warp_m(b, grid);
    mfmas_n = rocke_warp_grid_mfmas_per_warp_n(b, grid);
    if(!rocke_ir_builder_ok(b))
    {
        return rocke_ir_builder_status(b);
    }

    K0 = spec->problem.conv.K;
    conv1_tile_k = rocke_deep_fused_conv_pool_spec_effective_conv1_tile_k(spec);
    /* conv_spec.warp_tile_k == spec.warp_tile_k (see header note). */
    warp_tile_k = spec->warp_tile_k;
    if(warp_tile_k <= 0 || conv1_tile_k <= 0)
    {
        return rocke_i_set_err(b, ROCKE_ERR_VALUE, "deep_fused_conv_pool: bad conv1 tile_k") == NULL
                   ? ROCKE_ERR_VALUE
                   : ROCKE_ERR_VALUE;
    }
    k_atoms = conv1_tile_k / warp_tile_k;
    k_chunks = (K0 + conv1_tile_k - 1) / conv1_tile_k;
    /* The valid_k mask only guards a K tail; statically dead when the tiling
     * covers K exactly. */
    needs_mask = (k_chunks * conv1_tile_k != K0);

    warp_m_off = rocke_warp_grid_warp_m_off(b, grid);
    warp_n_off = rocke_warp_grid_warp_n_off(b, grid);
    if(!rocke_ir_builder_ok(b))
    {
        return rocke_ir_builder_status(b);
    }

    num_accs = mfmas_m * mfmas_n;
    if(num_accs < 0 || num_accs > ROCKE_DFCP_MAX_ACCS)
    {
        return rocke_i_set_err(b,
                               ROCKE_ERR_VALUE,
                               "deep_fused_conv_pool: conv1 acc count %d out of "
                               "range",
                               num_accs)
                       == NULL
                   ? ROCKE_ERR_VALUE
                   : ROCKE_ERR_VALUE;
    }
    {
        int idx;
        for(idx = 0; idx < num_accs; ++idx)
        {
            accs[idx] = rocke_b_zero_vec_f32(b, op->c_frag_len);
        }
    }

    for(k_chunk = 0; k_chunk < k_chunks; ++k_chunk)
    {
        int chunk_base = k_chunk * conv1_tile_k;
        int kk;
        for(kk = 0; kk < k_atoms; ++kk)
        {
            rocke_value_t* tile_off = rocke_b_const_i32(b, chunk_base + kk * warp_tile_k);
            rocke_value_t* a_col_base = rocke_b_add(b, a_k_base, tile_off);
            rocke_value_t* b_col_base = rocke_b_add(b, b_k_base, tile_off);
            rocke_value_t* a_rows[ROCKE_DFCP_MAX_ACCS];
            rocke_value_t* b_cols[ROCKE_DFCP_MAX_ACCS];
            int mi;
            int ni;

            for(mi = 0; mi < mfmas_m; ++mi)
            {
                rocke_value_t* a_row = rocke_b_add(
                    b, warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * op->m), a_mn_in_atom));
                a_rows[mi] = dfcp_load_smem_frag_contiguous_f16(
                    b, c0_smem, a_row, a_col_base, a_frag, needs_mask, K0);
            }

            for(ni = 0; ni < mfmas_n; ++ni)
            {
                rocke_value_t* b_row = rocke_b_add(
                    b, warp_n_off, rocke_b_add(b, rocke_b_const_i32(b, ni * op->n), b_mn_in_atom));
                b_cols[ni] = dfcp_load_smem_frag_contiguous_f16(
                    b, w1_smem, b_row, b_col_base, b_frag, needs_mask, K0);
            }

            flat = 0;
            for(mi = 0; mi < mfmas_m; ++mi)
            {
                for(ni = 0; ni < mfmas_n; ++ni)
                {
                    accs[flat]
                        = rocke_b_mma(b, op->op_id, a_rows[mi], b_cols[ni], accs[flat], NULL, 0);
                    ++flat;
                }
            }
        }
    }

    if(!rocke_ir_builder_ok(b))
    {
        return rocke_ir_builder_status(b);
    }
    if(out_accs == NULL || out_cap < (size_t)num_accs)
    {
        return rocke_i_set_err(b,
                               ROCKE_ERR_VALUE,
                               "deep_fused_conv_pool: conv1 out_accs cap %d < %d",
                               (int)out_cap,
                               num_accs)
                       == NULL
                   ? ROCKE_ERR_VALUE
                   : ROCKE_ERR_VALUE;
    }

    if(defer_epilogue)
    {
        int idx;
        for(idx = 0; idx < num_accs; ++idx)
        {
            out_accs[idx] = accs[idx];
        }
    }
    else
    {
        /* _apply_accumulator_epilogue(b, spec.conv1_epilogue, accs). */
        rocke_conv_apply_accumulator_epilogue(b, &spec->conv1_epilogue, accs, num_accs, out_accs);
        if(!rocke_ir_builder_ok(b))
        {
            return rocke_ir_builder_status(b);
        }
    }

    if(out_n != NULL)
    {
        *out_n = (size_t)num_accs;
    }
    return ROCKE_OK;
}
