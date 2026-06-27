/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.loads.h -- C99 port of two loader symbols from
 * rocke/helpers/loads.py:
 *
 *   Python                          C99 (this header)
 *   ------------------------------  -----------------------------------------
 *   CoalescedTileLoader (frozen dc) rocke_coalesced_tile_loader_t
 *     .choose_vec  (classmethod)      rocke_coalesced_tile_loader_choose_vec()
 *     .from_tile   (classmethod)      rocke_coalesced_tile_loader_from_tile()
 *     .vecs_per_thread (property)     rocke_coalesced_tile_loader_vecs_per_thread()
 *     .cols_per_vec    (property)     rocke_coalesced_tile_loader_cols_per_vec()
 *     .load        (method)          rocke_coalesced_tile_loader_load()
 *   AsyncTileLoader (frozen dc)     rocke_async_tile_loader_t
 *     .choose_dwords (classmethod)    rocke_async_tile_loader_choose_dwords()
 *     .from_tile     (classmethod)    rocke_async_tile_loader_from_tile()
 *     .halves_per_chunk (property)    rocke_async_tile_loader_halves_per_chunk()
 *     .bytes_per_chunk  (property)    rocke_async_tile_loader_bytes_per_chunk()
 *     .cols_per_chunk   (property)    rocke_async_tile_loader_cols_per_chunk()
 *     .wave_bytes       (property)    rocke_async_tile_loader_wave_bytes()
 *     .pass_bytes       (property)    rocke_async_tile_loader_pass_bytes()
 *     .bind          (method)         rocke_async_tile_loader_bind()
 *   AsyncTileLoaderSlot (frozen dc) rocke_async_tile_loader_slot_t
 *     .issue         (method)         rocke_async_tile_loader_slot_issue()
 *     .required_lds_bytes (method)    rocke_async_tile_loader_slot_required_lds_bytes()
 *
 * Tile loaders: global memory -> LDS. Both share the same authoring contract:
 *
 *   LDS[row, col] = global[block_row_off + row, block_col_off + col]
 *
 * The (row, col) -> (linear_element_offset, valid_predicate) mapping is supplied
 * by a `descriptor` callback so the loader stays convolution-aware. In Python
 * the descriptor returns `(off, valid)` where `valid` may be `None` ("always
 * in-bounds"). In C the callback writes the offset Value as its return and
 * stores the valid predicate through `*out_valid`; a NULL `*out_valid` means the
 * Python `None` ("always in-bounds").
 *
 *   CoalescedTileLoader -- the classic two-step pattern: each lane issues a
 *     buffer_load / global_load into a register then a matching smem_store
 *     writes that register to LDS.
 *   AsyncTileLoader -- direct DRAM->LDS via raw_ptr_buffer_load_lds (no register
 *     intermediate); completion is signalled via the VMEM counter so consumers
 *     must place an s_waitcnt(vmcnt=0) before reading the LDS.
 *
 * AsyncPingPongLoader, lane_contiguous_descriptor and the bare DescriptorFn
 * documentation alias from the Python module are NOT ported here (only
 * CoalescedTileLoader + AsyncTileLoader were requested); the descriptor typedef
 * below is the C spelling of the callback both loaders consume.
 *
 * Error model mirrors the rest of the C port: builder-emitting symbols are
 * no-ops returning NULL/void when `b` is already in error and record the
 * Python-matching message on the builder sticky error; pure value-type spellings
 * use a rocke_status_t out-param contract for the ValueError paths.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_LOADS_H
#define ROCKE_HELPER_ROCKE_HELPERS_LOADS_H

#include <stdbool.h>

#include "rocke/ir.h" /* rocke_status_t, rocke_ir_builder_t, rocke_value_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ DescriptorFn *
 *
 * Python:
 *   DescriptorFn = Callable[[IRBuilder, Value, Value],
 *                           Tuple[Value, Optional[Value]]]
 *
 * Maps (row, col) in the tile-local coordinate system to
 * (element_offset_in_global_array, valid_predicate). The returned offset is in
 * fp16 elements; the loader scales by 2 (sizeof half) when feeding it to a
 * buffer_load voffset. `valid` is an i1 Value, or "None" -- expressed here by
 * the callback storing NULL through *out_valid.
 *
 * Contract:
 *   - return value: the offset-in-elements Value (or NULL on error)
 *   - *out_valid:   the i1 valid Value, or NULL meaning "always in-bounds"
 *   - user:         opaque pointer threaded back to the callback unchanged
 *
 * `out_valid` is always non-NULL on entry; the callback must write to it. */
typedef rocke_value_t* (*rocke_loads_descriptor_fn)(rocke_ir_builder_t* b,
                                                    rocke_value_t* row,
                                                    rocke_value_t* col,
                                                    rocke_value_t** out_valid,
                                                    void* user);

/* ============================================================================
 * CoalescedTileLoader
 * ========================================================================== */

/* Value type mirroring rocke.helpers.loads.CoalescedTileLoader (frozen
 * dataclass). Fields are 1:1 with the Python declaration order. */
typedef struct rocke_coalesced_tile_loader
{
    int tile_rows;
    int tile_cols;
    int block_size;
    int load_vec; /* halves per thread per chunk */
    bool use_buffer_rsrc; /* Python default True */
    int oob_sentinel; /* Python default (1 << 31) - 1 */
    /* Python `inner_dim: Optional[int] = None`. Carried for field-parity;
     * inner_dim is consumer-side documentation only (the loader body never
     * reads it). has_inner_dim distinguishes None (false) from a set value. */
    bool has_inner_dim;
    int inner_dim;
} rocke_coalesced_tile_loader_t;

/* CoalescedTileLoader.choose_vec classmethod.
 *
 * Picks the widest `load_vec` that distributes evenly. Writes the chosen vec to
 * *out_vec and returns ROCKE_OK, or ROCKE_ERR_VALUE on the Python ValueError path
 * ("no usable load_vec ...") leaving *out_vec untouched. max_vec default is 8
 * (pass 8 to match the Python default). */
rocke_status_t rocke_coalesced_tile_loader_choose_vec(
    int tile_rows, int tile_cols, int block_size, int max_vec, int* out_vec);

/* CoalescedTileLoader.from_tile classmethod.
 *
 * Builds a loader by running choose_vec. On success writes the loader to *out
 * and returns ROCKE_OK; propagates ROCKE_ERR_VALUE from choose_vec. max_vec default
 * 8, use_buffer_rsrc default true. */
rocke_status_t rocke_coalesced_tile_loader_from_tile(int tile_rows,
                                                     int tile_cols,
                                                     int block_size,
                                                     int max_vec,
                                                     bool use_buffer_rsrc,
                                                     rocke_coalesced_tile_loader_t* out);

/* CoalescedTileLoader.vecs_per_thread property.
 *
 *   total_vecs = tile_rows * tile_cols / load_vec
 *   return total_vecs / block_size
 *
 * Writes the result to *out and returns ROCKE_OK, or ROCKE_ERR_VALUE on the Python
 * ValueError path (total_vecs not divisible by block_size). */
rocke_status_t
    rocke_coalesced_tile_loader_vecs_per_thread(const rocke_coalesced_tile_loader_t* self,
                                                int* out);

/* CoalescedTileLoader.cols_per_vec property: tile_cols / load_vec. */
int rocke_coalesced_tile_loader_cols_per_vec(const rocke_coalesced_tile_loader_t* self);

/* CoalescedTileLoader.load method.
 *
 * Emits the per-thread load loop. `descriptor` (+ its `user`) defines the tile
 * -> global mapping. If use_buffer_rsrc, `rsrc` must be non-NULL; otherwise
 * `ptr` must be non-NULL (the unused one is NULL). On a NULL-arg precondition
 * miss it records the Python ValueError message on `b` and returns. No-op if `b`
 * is already in error. */
void rocke_coalesced_tile_loader_load(rocke_ir_builder_t* b,
                                      const rocke_coalesced_tile_loader_t* self,
                                      rocke_value_t* tid,
                                      rocke_value_t* smem_dst,
                                      rocke_loads_descriptor_fn descriptor,
                                      void* descriptor_user,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* ptr);

/* ============================================================================
 * AsyncTileLoader / AsyncTileLoaderSlot
 * ========================================================================== */

/* Value type mirroring rocke.helpers.loads.AsyncTileLoader (frozen dataclass).
 * Fields are 1:1 with the Python declaration order. */
typedef struct rocke_async_tile_loader
{
    int tile_rows;
    int tile_cols;
    int block_size;
    int wave_size;
    int dwords; /* 1, 3, or 4 */
    int chunks_total; /* tile_rows * tile_cols / (dwords * 2) */
    int chunks_per_pass; /* = block_size */
    int passes; /* ceil(chunks_total / block_size) */
} rocke_async_tile_loader_t;

/* Bound AsyncTileLoader (rocke.helpers.loads.AsyncTileLoaderSlot). */
typedef struct rocke_async_tile_loader_slot
{
    rocke_async_tile_loader_t loader; /* by value (frozen dataclass copy) */
    rocke_value_t* smem_dst;
    rocke_value_t* per_wave_lds_base; /* i64; lane 0 of the wave writes here */
} rocke_async_tile_loader_slot_t;

/* AsyncTileLoader.choose_dwords classmethod.
 *
 * Picks the widest `dwords` value (4, 3, or 1) that divides the tile evenly.
 * Writes it to *out and returns ROCKE_OK, or ROCKE_ERR_VALUE on the Python
 * ValueError path. max_dwords default 4 (values > 4 are clamped to 4, matching
 * Python). */
rocke_status_t rocke_async_tile_loader_choose_dwords(
    int tile_rows, int tile_cols, int block_size, int max_dwords, int* out);

/* AsyncTileLoader.from_tile classmethod.
 *
 * Builds a loader by running choose_dwords then deriving chunks/passes. On
 * success writes the loader to *out and returns ROCKE_OK; propagates
 * ROCKE_ERR_VALUE. wave_size default 64, max_dwords default 4. */
rocke_status_t rocke_async_tile_loader_from_tile(int tile_rows,
                                                 int tile_cols,
                                                 int block_size,
                                                 int wave_size,
                                                 int max_dwords,
                                                 rocke_async_tile_loader_t* out);

/* AsyncTileLoader properties (pure int arithmetic). */
int rocke_async_tile_loader_halves_per_chunk(const rocke_async_tile_loader_t* self);
int rocke_async_tile_loader_bytes_per_chunk(const rocke_async_tile_loader_t* self);
int rocke_async_tile_loader_cols_per_chunk(const rocke_async_tile_loader_t* self);
int rocke_async_tile_loader_wave_bytes(const rocke_async_tile_loader_t* self);
int rocke_async_tile_loader_pass_bytes(const rocke_async_tile_loader_t* self);

/* AsyncTileLoader.bind method.
 *
 * Materialises the SSA values needed by issue (the per-wave LDS base offset,
 * hoisted into an SGPR via to_sgpr_u32). On success writes the slot to *out_slot
 * and returns ROCKE_OK. No-op returning b->status if `b` is already in error. */
rocke_status_t rocke_async_tile_loader_bind(rocke_ir_builder_t* b,
                                            const rocke_async_tile_loader_t* self,
                                            rocke_value_t* smem_dst,
                                            rocke_value_t* wave_id,
                                            rocke_async_tile_loader_slot_t* out_slot);

/* AsyncTileLoaderSlot.issue method.
 *
 * Fires all `passes * threads` async loads for this iteration. After issue the
 * LDS contents are in flight; consumers must place an s_waitcnt(vmcnt=0) before
 * reading. oob_sentinel default (1 << 31) - 1, coherency default 0. No-op if `b`
 * is already in error. */
void rocke_async_tile_loader_slot_issue(rocke_ir_builder_t* b,
                                        const rocke_async_tile_loader_slot_t* self,
                                        rocke_value_t* tid,
                                        rocke_value_t* rsrc,
                                        rocke_loads_descriptor_fn descriptor,
                                        void* descriptor_user,
                                        int oob_sentinel,
                                        int coherency);

/* AsyncTileLoaderSlot.required_lds_bytes method: passes * pass_bytes. */
int rocke_async_tile_loader_slot_required_lds_bytes(const rocke_async_tile_loader_slot_t* self);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_LOADS_H */
