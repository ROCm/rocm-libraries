// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.persistent.c -- C99 port of rocke.helpers.persistent.
 *
 * Ports build_persistent_counter_init and persistent_tile_for_each, plus the
 * persistent_tile_loop @contextmanager body (reproduced as a file-local
 * static). The builder-call sequence is byte-identical to the Python so the
 * emitted IR op stream matches exactly.
 */
#include "rocke/helper_rocke.helpers.persistent.h"

/* ------------------------------------------------------------------------
 * build_persistent_counter_init (public)
 *
 *   from ..core.ir import I32
 *   if counter_idx is None:
 *       counter_idx = b.const_i32(0)
 *   if not cooperative:
 *       return b.global_atomic_add(counter, counter_idx, b.const_i32(increment))
 *
 *   tid = b.thread_id_x()
 *   is_lead = b.cmp_eq(tid, b.const_i32(0))
 *
 *   if block_size <= wave_size:
 *       inc_per_lane = b.select(is_lead, b.const_i32(increment), b.const_i32(0))
 *       fetched = b.global_atomic_add(counter, counter_idx, inc_per_lane)
 *       return b.ds_bpermute(b.const_i32(0), fetched)
 *
 *   if broadcast_slot is None:
 *       broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
 *   with b.scf_if(is_lead):
 *       v = b.global_atomic_add(counter, counter_idx, b.const_i32(increment))
 *       b.smem_store_vN(broadcast_slot, [b.const_i32(0)], v, 1)
 *   b.sync()
 *   return b.vec_extract(
 *       b.smem_load_vN(broadcast_slot, b.const_i32(0), dtype=I32, n=1), 0)
 * ------------------------------------------------------------------------ */
rocke_value_t* rocke_build_persistent_counter_init(rocke_ir_builder_t* b,
                                                   rocke_value_t* counter,
                                                   rocke_value_t* counter_idx,
                                                   int increment,
                                                   bool cooperative,
                                                   rocke_value_t* broadcast_slot,
                                                   int wave_size,
                                                   int block_size)
{
    const rocke_type_t* I32 = rocke_i32();
    rocke_value_t* tid;
    rocke_value_t* is_lead;

    if(counter_idx == NULL)
        counter_idx = rocke_b_const_i32(b, 0);

    if(!cooperative)
        return rocke_b_global_atomic_add(
            b, counter, counter_idx, rocke_b_const_i32(b, (int64_t)increment), NULL);

    tid = rocke_b_thread_id_x(b);
    is_lead = rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0));

    /* Single-wave CTA: ds_bpermute broadcast, no s_barrier. */
    if(block_size <= wave_size)
    {
        /* Python b.select evaluates its operands left-to-right: the true-value
         * const (increment) is emitted BEFORE the false-value const (0). Hoist
         * to pin that order -- C arg-eval order is unspecified (GCC is r-to-l). */
        rocke_value_t* sel_t = rocke_b_const_i32(b, (int64_t)increment);
        rocke_value_t* sel_f = rocke_b_const_i32(b, 0);
        rocke_value_t* inc_per_lane = rocke_b_select(b, is_lead, sel_t, sel_f);
        rocke_value_t* fetched
            = rocke_b_global_atomic_add(b, counter, counter_idx, inc_per_lane, NULL);
        return rocke_b_ds_bpermute(b, rocke_b_const_i32(b, 0), fetched);
    }

    /* Multi-wave CTA: LDS slot + s_barrier (real barrier; other waves observe). */
    if(broadcast_slot == NULL)
    {
        int shape[1] = {1};
        broadcast_slot = rocke_b_smem_alloc(b, I32, shape, 1, "pers_brd");
    }
    {
        rocke_if_t gate = rocke_b_scf_if(b, is_lead);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* v = rocke_b_global_atomic_add(
                b, counter, counter_idx, rocke_b_const_i32(b, (int64_t)increment), NULL);
            rocke_value_t* store_idx[1] = {rocke_b_const_i32(b, 0)};
            rocke_b_smem_store_vN(b, broadcast_slot, store_idx, 1, v, 1);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    {
        rocke_value_t* load_idx[1] = {rocke_b_const_i32(b, 0)};
        rocke_value_t* loaded = rocke_b_smem_load_vN(b, broadcast_slot, load_idx, 1, I32, 1);
        return rocke_b_vec_extract(b, loaded, 0);
    }
}

/* ------------------------------------------------------------------------
 * persistent_tile_loop (file-local; Python @contextmanager)
 *
 * The Python contextmanager:
 *   if counter_idx is None:
 *       counter_idx = b.const_i32(0)
 *   for_op = b.scf_for_iter(
 *       b.const_i32(0), b.const_i32(max_iters), b.const_i32(1),
 *       [("tile_idx", tile_idx_init)], iv_name="pers_iter")
 *   with for_op as (_iter_v, (tile_idx,)):
 *       in_range = b.cmp_lt(tile_idx, num_tiles)
 *       yield tile_idx, in_range            # caller body runs here
 *       next_tile = build_persistent_counter_init(
 *           b, counter, counter_idx=counter_idx, increment=1,
 *           cooperative=cooperative, broadcast_slot=broadcast_slot,
 *           wave_size=wave_size, block_size=block_size)
 *       b.scf_yield(next_tile)
 *
 * In C, the "yield to the caller body" is realized by invoking `body`
 * (already wrapped in the in_range scf.if guard, exactly as
 * persistent_tile_for_each's call site does after the yield). This keeps the
 * single C consumer (persistent_tile_for_each) byte-identical to the Python
 * composition. body/user_data is the per-iteration tile callback.
 * ------------------------------------------------------------------------ */
static void rocke_i_persistent_tile_loop(rocke_ir_builder_t* b,
                                         rocke_value_t* counter,
                                         rocke_value_t* num_tiles,
                                         int max_iters,
                                         rocke_value_t* tile_idx_init,
                                         rocke_value_t* counter_idx,
                                         bool cooperative,
                                         rocke_value_t* broadcast_slot,
                                         int wave_size,
                                         int block_size,
                                         rocke_persistent_tile_body_fn body,
                                         void* user_data)
{
    rocke_iter_arg_t loop_args[1];
    rocke_for_t for_op;
    rocke_value_t* lb;
    rocke_value_t* ub;
    rocke_value_t* step;

    if(counter_idx == NULL)
        counter_idx = rocke_b_const_i32(b, 0);

    loop_args[0].name = "tile_idx";
    loop_args[0].init = tile_idx_init;

    /* Python scf_for_iter defaults: unroll=False, elide_trailing_barrier=True.
     * Emit the bound constants in positional (lb, ub, step) order; C arg-eval
     * order is unspecified (GCC is right-to-left), so hoist to match Python. */
    lb = rocke_b_const_i32(b, 0);
    ub = rocke_b_const_i32(b, (int64_t)max_iters);
    step = rocke_b_const_i32(b, 1);
    for_op = rocke_b_scf_for_iter(b,
                                  lb,
                                  ub,
                                  step,
                                  loop_args,
                                  1,
                                  "pers_iter",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* tile_idx = for_op.iter_vars[0];
        rocke_value_t* in_range = rocke_b_cmp_lt(b, tile_idx, num_tiles);
        rocke_value_t* next_tile;
        rocke_value_t* yield_vals[1];

        /* Yield point: persistent_tile_for_each wraps the caller body in the
         * in_range guard (with b.scf_if(in_range): body(tile_idx)). */
        {
            rocke_if_t gate = rocke_b_scf_if(b, in_range);
            rocke_b_region_enter(b, gate.then_region);
            body(b, tile_idx, user_data);
            rocke_b_region_leave(b);
        }

        /* After the caller's body, fetch the next tile id for this CTA and
         * yield it as the loop-carried value. */
        next_tile = rocke_build_persistent_counter_init(b,
                                                        counter,
                                                        counter_idx,
                                                        /*increment=*/1,
                                                        cooperative,
                                                        broadcast_slot,
                                                        wave_size,
                                                        block_size);
        yield_vals[0] = next_tile;
        rocke_b_scf_yield(b, yield_vals, 1);
    }
    rocke_b_region_leave(b);
}

/* ------------------------------------------------------------------------
 * persistent_tile_for_each (public)
 *
 *   from ..core.ir import I32
 *   broadcast_slot = None
 *   if cooperative and block_size > wave_size:
 *       broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
 *   tile_idx0 = build_persistent_counter_init(
 *       b, counter, counter_idx=counter_idx, cooperative=cooperative,
 *       broadcast_slot=broadcast_slot, wave_size=wave_size,
 *       block_size=block_size)
 *   with persistent_tile_loop(
 *       b, counter=counter, num_tiles=num_tiles, max_iters=max_iters,
 *       tile_idx_init=tile_idx0, counter_idx=counter_idx,
 *       cooperative=cooperative, broadcast_slot=broadcast_slot,
 *       wave_size=wave_size, block_size=block_size) as (tile_idx, in_range):
 *       with b.scf_if(in_range):
 *           body(tile_idx)
 * ------------------------------------------------------------------------ */
void rocke_persistent_tile_for_each(rocke_ir_builder_t* b,
                                    rocke_value_t* counter,
                                    rocke_value_t* num_tiles,
                                    int max_iters,
                                    rocke_persistent_tile_body_fn body,
                                    void* user_data,
                                    rocke_value_t* counter_idx,
                                    bool cooperative,
                                    int wave_size,
                                    int block_size)
{
    const rocke_type_t* I32 = rocke_i32();
    rocke_value_t* broadcast_slot = NULL;
    rocke_value_t* tile_idx0;

    /* Multi-wave CTA still uses LDS broadcast; allocate the slot once at CTA
     * scope so every iteration reuses it. */
    if(cooperative && block_size > wave_size)
    {
        int shape[1] = {1};
        broadcast_slot = rocke_b_smem_alloc(b, I32, shape, 1, "pers_brd");
    }

    /* Note: build_persistent_counter_init's increment defaults to 1 here
     * (the Python call omits it). */
    tile_idx0 = rocke_build_persistent_counter_init(b,
                                                    counter,
                                                    counter_idx,
                                                    /*increment=*/1,
                                                    cooperative,
                                                    broadcast_slot,
                                                    wave_size,
                                                    block_size);

    rocke_i_persistent_tile_loop(b,
                                 counter,
                                 num_tiles,
                                 max_iters,
                                 tile_idx0,
                                 counter_idx,
                                 cooperative,
                                 broadcast_slot,
                                 wave_size,
                                 block_size,
                                 body,
                                 user_data);
}
