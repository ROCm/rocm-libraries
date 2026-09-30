// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.sweep.c -- C99 port of rocke.helpers.sweep.
 *
 * Ports sweep_row_chunks, pass2_row_chunks and RowChunkSweepResult. The
 * builder-call sequence is byte-identical to the Python so the emitted IR op
 * stream matches exactly: per chunk both functions emit
 *
 *   n_off = b.add(b.mul(b.const_i32(k * block_size), c_vec),
 *                 b.mul(tid, c_vec))
 *
 * with c_vec = b.const_i32(vec) hoisted once before the loop, exactly as the
 * Python does.
 */
#include "rocke/helper_rocke.helpers.sweep.h"

#include "rocke/arena.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err for Python-ValueError parity */

/* ------------------------------------------------------------------------
 * sweep_row_chunks
 *
 * Python:
 *   if tile.rank != 2: raise ValueError(...)
 *   if elems_per_thread % vec: raise ValueError(...)
 *   chunks_per_thread = elems_per_thread // vec
 *   if row is not None:
 *       old_origin = tile.origin
 *       tile = TileWindow(view=tile.view, lengths=tile.lengths,
 *                         origin=(row, old_origin[1]))
 *   cached = []
 *   c_vec = b.const_i32(vec)
 *   for k in range(chunks_per_thread):
 *       n_off = b.add(b.mul(b.const_i32(k * block_size), c_vec),
 *                     b.mul(tid, c_vec))
 *       x_scalars = tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=vec)
 *       if cache: cached.extend(x_scalars)
 *       if body is not None: body(n_off, x_scalars)
 *   return RowChunkSweepResult(cached=cached, chunks_per_thread=...)
 * ------------------------------------------------------------------------ */
rocke_row_chunk_sweep_result_t rocke_sweep_row_chunks(rocke_ir_builder_t* b,
                                                      const rocke_tile_window_t* tile,
                                                      rocke_value_t* tid,
                                                      int block_size,
                                                      int vec,
                                                      int elems_per_thread,
                                                      rocke_value_t* row,
                                                      rocke_sweep_row_body_fn body,
                                                      void* user,
                                                      bool cache)
{
    rocke_row_chunk_sweep_result_t res;
    rocke_tile_window_t shifted;
    const rocke_tile_window_t* t;
    int chunks_per_thread;
    rocke_value_t* c_vec;
    rocke_value_t** scratch;
    int k;

    res.cached = NULL;
    res.num_cached = 0;
    res.chunks_per_thread = 0;

    if(tile->rank != 2)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "sweep_row_chunks expects a 2D TileWindow (got rank %d)",
                        tile->rank);
        return res;
    }
    if(vec != 0 && (elems_per_thread % vec) != 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "elems_per_thread (%d) not divisible by vec (%d)",
                        elems_per_thread,
                        vec);
        return res;
    }

    chunks_per_thread = elems_per_thread / vec;
    res.chunks_per_thread = chunks_per_thread;

    /* row is not None: shift the tile origin to (row, old_origin[1]). */
    t = tile;
    if(row != NULL)
    {
        shifted = *tile;
        shifted.origin[0] = row;
        shifted.origin[1] = tile->origin[1];
        t = &shifted;
    }

    /* cached: when caching, sized chunks_per_thread * vec (element order
     * across chunks), arena-owned to outlive this frame. */
    if(cache && chunks_per_thread > 0 && vec > 0)
    {
        res.cached = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)chunks_per_thread * (size_t)vec * sizeof(rocke_value_t*));
    }

    /* Per-chunk scratch buffer for the vec f32 scalars from the load. */
    scratch = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, (size_t)(vec > 0 ? vec : 1) * sizeof(rocke_value_t*));

    c_vec = rocke_b_const_i32(b, vec);
    for(k = 0; k < chunks_per_thread; ++k)
    {
        rocke_value_t* n_off;
        rocke_value_t* mul_kbs;
        rocke_value_t* mul_tid;
        rocke_value_t* local_indices[2];
        int i;

        /* Python: n_off = b.add(b.mul(b.const_i32(k*block_size), c_vec),
         *                       b.mul(tid, c_vec))
         * Python evaluates the two mul() args left-to-right, so the
         * (k*block_size)*vec multiply is emitted BEFORE the tid*vec multiply.
         * C function-argument evaluation order is unspecified, so the two
         * multiplies are hoisted into temporaries to pin Python's order. */
        mul_kbs = rocke_b_mul(b, rocke_b_const_i32(b, (int64_t)k * (int64_t)block_size), c_vec);
        mul_tid = rocke_b_mul(b, tid, c_vec);
        n_off = rocke_b_add(b, mul_kbs, mul_tid);

        /* tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=vec)
         * The b.const_i32(0) is allocated at the call site each iteration
         * (NOT hoisted) so the SSA-id counter advances exactly as in Python,
         * which spells the literal-0 column index inline at the load. */
        local_indices[0] = rocke_b_const_i32(b, 0);
        local_indices[1] = n_off;
        rocke_tile_window_load_vec_as_f32(b, t, local_indices, 2, vec, scratch);

        if(cache && res.cached != NULL)
        {
            for(i = 0; i < vec; ++i)
            {
                res.cached[res.num_cached++] = scratch[i];
            }
        }
        if(body != NULL)
        {
            body(b, n_off, scratch, vec, user);
        }
    }

    return res;
}

/* ------------------------------------------------------------------------
 * pass2_row_chunks
 *
 * Python:
 *   if tile.rank != 2: raise ValueError(...)
 *   if body is None: raise ValueError(...)
 *   if elems_per_thread % vec: raise ValueError(...)
 *   if row is not None:
 *       old_origin = tile.origin
 *       tile = TileWindow(view=..., lengths=..., origin=(row, old_origin[1]))
 *   chunks_per_thread = elems_per_thread // vec
 *   c_vec = b.const_i32(vec)
 *   for k in range(chunks_per_thread):
 *       n_off = b.add(b.mul(b.const_i32(k * block_size), c_vec),
 *                     b.mul(tid, c_vec))
 *       if cached_f32: x_scalars = list(cached_f32[k*vec:(k+1)*vec])
 *       else:          x_scalars = []
 *       out = body(n_off, k, x_scalars)
 *       if len(out) != vec: raise ValueError(...)
 *       tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=out)
 * ------------------------------------------------------------------------ */
void rocke_pass2_row_chunks(rocke_ir_builder_t* b,
                            const rocke_tile_window_t* tile,
                            rocke_value_t* tid,
                            int block_size,
                            int vec,
                            int elems_per_thread,
                            rocke_value_t* row,
                            rocke_pass2_row_body_fn body,
                            void* user,
                            rocke_value_t* const* cached_f32,
                            int num_cached_f32)
{
    rocke_tile_window_t shifted;
    const rocke_tile_window_t* t;
    int chunks_per_thread;
    rocke_value_t* c_vec;
    rocke_value_t** out;
    int k;

    if(tile->rank != 2)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "pass2_row_chunks expects a 2D TileWindow (got rank %d)",
                        tile->rank);
        return;
    }
    if(body == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "pass2_row_chunks requires a body callback");
        return;
    }
    if(vec != 0 && (elems_per_thread % vec) != 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "elems_per_thread (%d) not divisible by vec (%d)",
                        elems_per_thread,
                        vec);
        return;
    }

    /* row is not None: shift the tile origin to (row, old_origin[1]). */
    t = tile;
    if(row != NULL)
    {
        shifted = *tile;
        shifted.origin[0] = row;
        shifted.origin[1] = tile->origin[1];
        t = &shifted;
    }

    chunks_per_thread = elems_per_thread / vec;

    /* Per-chunk out buffer for the vec f32 scalars the body returns. */
    out = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                             (size_t)(vec > 0 ? vec : 1) * sizeof(rocke_value_t*));

    c_vec = rocke_b_const_i32(b, vec);
    for(k = 0; k < chunks_per_thread; ++k)
    {
        rocke_value_t* n_off;
        rocke_value_t* mul_kbs;
        rocke_value_t* mul_tid;
        rocke_value_t* const* x_scalars;
        int num_x;
        rocke_value_t* local_indices[2];

        /* Python: n_off = b.add(b.mul(b.const_i32(k*block_size), c_vec),
         *                       b.mul(tid, c_vec)) -- left-to-right arg eval
         * emits the (k*block_size)*vec multiply before the tid*vec multiply.
         * Hoist to temporaries to pin that order in C. */
        mul_kbs = rocke_b_mul(b, rocke_b_const_i32(b, (int64_t)k * (int64_t)block_size), c_vec);
        mul_tid = rocke_b_mul(b, tid, c_vec);
        n_off = rocke_b_add(b, mul_kbs, mul_tid);

        /* x_scalars = cached_f32[k*vec:(k+1)*vec] when cache present, else []. */
        if(num_cached_f32 > 0 && cached_f32 != NULL)
        {
            x_scalars = cached_f32 + (size_t)k * (size_t)vec;
            num_x = vec;
        }
        else
        {
            x_scalars = NULL;
            num_x = 0;
        }

        /* out = body(n_off, k, x_scalars); the callback fills out[0..vec). */
        body(b, n_off, k, x_scalars, num_x, out, vec, user);

        /* tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=out)
         * b.const_i32(0) allocated at the call site (NOT hoisted) so the
         * SSA-id counter advances exactly as in Python. */
        local_indices[0] = rocke_b_const_i32(b, 0);
        local_indices[1] = n_off;
        rocke_tile_window_store_vec_from_f32(b, t, local_indices, 2, out, vec);
    }
}
