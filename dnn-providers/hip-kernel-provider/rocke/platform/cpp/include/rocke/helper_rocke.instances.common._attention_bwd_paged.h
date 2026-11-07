/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_paged.h -- C port of
 * library/kernels/common/_attention_bwd_paged.py: the paged address generator
 * of the unified attention backward main kernel.
 *
 *   Python (_attention_bwd_paged.py)    C (this header)
 *   ----------------------------------  ----------------------------------------
 *   live_blocks                         rocke_attn_bwd_live_blocks
 *   PagedKvTile (+ methods)             rocke_attn_bwd_paged_kv_tile_t (+ _init,
 *                                       _free, _locate, _checked, _store_rule,
 *                                       _workspace_row)
 *   PagedKvTile.workspace_row's         rocke_attn_bwd_paged_ws_row_t +
 *     row_slab(lr) closure                rocke_attn_bwd_paged_ws_row_slab
 *   PagedSlab (+ methods)               rocke_attn_bwd_paged_slab_t (+ _init,
 *                                       _load_vec, _store, _store_vec)
 *   PagedCacheRows (+ slab)             rocke_attn_bwd_paged_cache_rows_t (+ _slab)
 *   PagedKv (+ tile, cache)             rocke_attn_bwd_paged_kv_t (+ _init, _free,
 *                                       _guarded, _tile, _cache)
 *
 * The Python IntOrValue arguments are rocke_attn_bwd_iv_t. sub_page: int | None
 * is an `int*` (NULL is None); an explicit sub_page of 0 or less is refused
 * with a ValueError (Python hits ZeroDivisionError, or builds an empty page
 * list for a negative one). `reads` (Python's itertools.count of the
 * block-table reads; unique loop names per kernel) is an `int*` counter, NULL
 * for a fresh count of 0 as in Python.
 *
 * Ownership: a tile owns its page_ids array (release with _tile_free); a
 * PagedKv owns its tiles (release with _free). Do not copy those structs. A
 * tile keeps a copy of the SeqInfo; PagedKv keys its tiles on the k0 Value
 * pointer, as Python keys on id(k0).
 *
 * Errors (Python ValueError) are recorded on the builder's sticky error as
 * ROCKE_ERR_VALUE; entries return false (or NULL) after one. Builder calls
 * are made in Python's order.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_PAGED_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_PAGED_H

#include <stdbool.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_addr.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* live_blocks: ceil(length / page) (i32). */
rocke_value_t* rocke_attn_bwd_live_blocks(rocke_ir_builder_t* b,
                                          rocke_value_t* length,
                                          rocke_attn_bwd_iv_t page);

/* PagedKvTile. in0 is NULL in the "sub" class. */
typedef struct rocke_attn_bwd_paged_kv_tile
{
    rocke_ir_builder_t* b;
    rocke_value_t* k0;
    int k_n0;
    bool has_sub_page;
    int sub_page;
    rocke_value_t* page;
    rocke_attn_bwd_seq_info_t seq;
    rocke_value_t* n_blocks;
    rocke_value_t* cover;
    rocke_value_t* in0;
    bool guarded;
    rocke_value_t** page_ids; /* n_pieces entries */
    int n_pieces;
} rocke_attn_bwd_paged_kv_tile_t;

/* PagedKvTile.__init__ (sub_page NULL is None; num_blocks NULL is None; reads
 * NULL is a fresh counter). false on a ValueError. */
bool rocke_attn_bwd_paged_kv_tile_init(rocke_attn_bwd_paged_kv_tile_t* t,
                                       rocke_ir_builder_t* b,
                                       rocke_value_t* bt_ptr,
                                       rocke_attn_bwd_iv_t bt_stride,
                                       rocke_value_t* seq_idx,
                                       const rocke_attn_bwd_seq_info_t* seq,
                                       rocke_value_t* k0,
                                       int k_n0,
                                       rocke_attn_bwd_iv_t page,
                                       const int* sub_page,
                                       rocke_value_t* num_blocks,
                                       int* reads);
void rocke_attn_bwd_paged_kv_tile_free(rocke_attn_bwd_paged_kv_tile_t* t);

/* locate(lr): (page id, in-page slot) of tile row lr. */
void rocke_attn_bwd_paged_kv_tile_locate(const rocke_attn_bwd_paged_kv_tile_t* t,
                                         rocke_value_t* lr,
                                         rocke_value_t** out_pid,
                                         rocke_value_t** out_slot);
/* checked(pid): (address id, ok); ok is NULL (None) without the guard. */
void rocke_attn_bwd_paged_kv_tile_checked(const rocke_attn_bwd_paged_kv_tile_t* t,
                                          rocke_value_t* pid,
                                          rocke_value_t** out_id,
                                          rocke_value_t** out_ok);
/* store_rule(k): (store, row_live) of logical key row k. */
void rocke_attn_bwd_paged_kv_tile_store_rule(const rocke_attn_bwd_paged_kv_tile_t* t,
                                             rocke_value_t* k,
                                             rocke_value_t** out_store,
                                             rocke_value_t** out_row_live);

/* workspace_row(ws_ptr, page_rows=, width=, head=): the row_slab(lr) closure
 * as a struct (the "multi" class holds the one slab formed at this call). */
typedef struct rocke_attn_bwd_paged_ws_row
{
    const rocke_attn_bwd_paged_kv_tile_t* tile;
    rocke_value_t* ws_ptr;
    rocke_attn_bwd_iv_t page_rows;
    int width;
    rocke_attn_bwd_iv_t head;
    rocke_value_t* page64;
    rocke_attn_bwd_ws_slab_t tile_slab; /* "multi" class */
    rocke_value_t* zero; /* "sub" class: the constant row_slab returns */
} rocke_attn_bwd_paged_ws_row_t;

bool rocke_attn_bwd_paged_kv_tile_workspace_row(const rocke_attn_bwd_paged_kv_tile_t* t,
                                                rocke_value_t* ws_ptr,
                                                rocke_attn_bwd_iv_t page_rows,
                                                int width,
                                                rocke_attn_bwd_iv_t head,
                                                rocke_attn_bwd_paged_ws_row_t* out);
/* row_slab(lr) -> (WsSlab, local row). */
bool rocke_attn_bwd_paged_ws_row_slab(const rocke_attn_bwd_paged_ws_row_t* w,
                                      rocke_value_t* lr,
                                      rocke_attn_bwd_ws_slab_t* out_slab,
                                      rocke_value_t** out_row);

/* PagedSlab. last NULL is None. */
typedef struct rocke_attn_bwd_paged_slab
{
    rocke_ir_builder_t* b;
    const rocke_attn_bwd_paged_kv_tile_t* tile;
    const rocke_type_t* dtype;
    rocke_value_t* last;
    int stage_vec;
    int esize;
    rocke_attn_bwd_iv_t s_blk;
    rocke_attn_bwd_iv_t s_tok;
    rocke_value_t* base;
    rocke_value_t* tile_ok;
    rocke_value_t* tile_ptr; /* NULL in the "sub" class */
} rocke_attn_bwd_paged_slab_t;

bool rocke_attn_bwd_paged_slab_init(rocke_attn_bwd_paged_slab_t* s,
                                    rocke_ir_builder_t* b,
                                    const rocke_attn_bwd_paged_kv_tile_t* tile,
                                    rocke_value_t* ptr,
                                    const rocke_type_t* dtype,
                                    const rocke_attn_bwd_iv_t strides[3],
                                    rocke_attn_bwd_iv_t head,
                                    rocke_value_t* last,
                                    int stage_vec);
/* load_vec(row, col0, n=8). */
rocke_value_t* rocke_attn_bwd_paged_slab_load_vec(const rocke_attn_bwd_paged_slab_t* s,
                                                  rocke_value_t* row,
                                                  rocke_attn_bwd_iv_t col0,
                                                  int n);
void rocke_attn_bwd_paged_slab_store(const rocke_attn_bwd_paged_slab_t* s,
                                     rocke_value_t* row,
                                     rocke_attn_bwd_iv_t col,
                                     rocke_value_t* value);
void rocke_attn_bwd_paged_slab_store_vec(const rocke_attn_bwd_paged_slab_t* s,
                                         rocke_value_t* row,
                                         rocke_attn_bwd_iv_t col0,
                                         rocke_value_t* value);

/* PagedKv. */
typedef struct rocke_attn_bwd_paged_kv_cache_entry
{
    rocke_value_t* k0;
    rocke_attn_bwd_paged_kv_tile_t* tile;
} rocke_attn_bwd_paged_kv_cache_entry_t;

typedef struct rocke_attn_bwd_paged_kv
{
    rocke_ir_builder_t* b;
    rocke_value_t* bt_ptr;
    rocke_attn_bwd_iv_t bt_stride;
    rocke_value_t* seq_idx;
    rocke_attn_bwd_seq_info_t seq;
    int k_n0;
    rocke_attn_bwd_iv_t page;
    bool has_sub_page;
    int sub_page;
    rocke_value_t* num_blocks; /* NULL: None */
    int reads;
    rocke_attn_bwd_paged_kv_cache_entry_t* tiles;
    int n_tiles;
    int cap_tiles;
} rocke_attn_bwd_paged_kv_t;

/* PagedKv.__init__ (sub_page NULL is None; page_rows NULL is None). */
bool rocke_attn_bwd_paged_kv_init(rocke_attn_bwd_paged_kv_t* p,
                                  rocke_ir_builder_t* b,
                                  rocke_value_t* bt_ptr,
                                  rocke_attn_bwd_iv_t bt_stride,
                                  rocke_value_t* seq_idx,
                                  const rocke_attn_bwd_seq_info_t* seq,
                                  int k_n0,
                                  rocke_attn_bwd_iv_t page,
                                  const int* sub_page,
                                  const rocke_attn_bwd_iv_t* page_rows);
void rocke_attn_bwd_paged_kv_free(rocke_attn_bwd_paged_kv_t* p);
bool rocke_attn_bwd_paged_kv_guarded(const rocke_attn_bwd_paged_kv_t* p);
/* tile(k0): built once per k0 Value; NULL after an error. The pointer stays
 * valid until _free. */
const rocke_attn_bwd_paged_kv_tile_t* rocke_attn_bwd_paged_kv_tile(rocke_attn_bwd_paged_kv_t* p,
                                                                   rocke_value_t* k0);

/* PagedCacheRows: a paged [num_blocks, page, h, D] cache by logical kv tile
 * (strides: blk, tok, h). paged is borrowed. */
typedef struct rocke_attn_bwd_paged_cache_rows
{
    rocke_attn_bwd_paged_kv_t* paged;
    rocke_value_t* ptr;
    const rocke_type_t* dtype;
    rocke_attn_bwd_iv_t strides[3];
    int stage_vec;
} rocke_attn_bwd_paged_cache_rows_t;

/* PagedKv.cache(ptr, dtype=, strides=, stage_vec=). */
rocke_attn_bwd_paged_cache_rows_t
    rocke_attn_bwd_paged_kv_cache(rocke_attn_bwd_paged_kv_t* p,
                                  rocke_value_t* ptr,
                                  const rocke_type_t* dtype,
                                  const rocke_attn_bwd_iv_t strides[3],
                                  int stage_vec);
/* PagedCacheRows.slab(h, k0, clamp=True). */
bool rocke_attn_bwd_paged_cache_rows_slab(const rocke_attn_bwd_paged_cache_rows_t* c,
                                          rocke_attn_bwd_iv_t h,
                                          rocke_value_t* k0,
                                          bool clamp,
                                          rocke_attn_bwd_paged_slab_t* out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_PAGED_H */
