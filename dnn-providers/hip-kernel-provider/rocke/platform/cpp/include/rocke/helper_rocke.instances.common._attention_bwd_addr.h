/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_addr.h -- C port of the
 * address generators of the attention backward kernels
 * (library/kernels/common/_attention_bwd_addr.py).
 *
 *   Python (_attention_bwd_addr.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   IntOrValue                          rocke_attn_bwd_iv_t
 *   as_i32 / as_i64                     rocke_attn_bwd_as_i32 / _as_i64
 *   last_row                            rocke_attn_bwd_last_row
 *   SeqInfo                             rocke_attn_bwd_seq_info_t
 *   load_offset                         rocke_attn_bwd_load_offset
 *   decode_batched / decode_thd         rocke_attn_bwd_decode_batched / _thd
 *   RowSlab (+ methods)                 rocke_attn_bwd_row_slab_t (+ _row_slab_*)
 *   StridedTensorAddr (init, slab)      rocke_attn_bwd_strided_tensor_addr_t
 *   WsSlab (+ methods)                  rocke_attn_bwd_ws_slab_t (+ _ws_slab_*)
 *   WorkspaceAddr (init, slab)          rocke_attn_bwd_workspace_addr_t
 *   block_table_entry                   rocke_attn_bwd_block_table_entry
 *   PagedCacheAddr (init, slab)         rocke_attn_bwd_paged_cache_addr_t
 *
 * Byte-identity rules every function here follows:
 *   - ops are created in the Python evaluation order (left to right, inner
 *     call before the outer one), every intermediate bound to a local first;
 *   - an IntOrValue argument is a host int (v == NULL) or an SSA value, and an
 *     int emits exactly what Python emits for it (a constant, or nothing for
 *     the literal-0 / literal-1 shortcuts);
 *   - the per-builder counter of the guarded SEQ reads (Python _SEQ_GUARDS,
 *     keyed weakly on the builder) is an explicit `int* seq_guards` owned by
 *     the kernel builder: start it at 0 for every new kernel.
 *
 * Errors (the Python ValueErrors): rocke_i_set_err(b, ROCKE_ERR_VALUE, ...)
 * raises ckc::Error, so call these helpers under ckc::guard_builder (the
 * kernel builders do). The bool returns never answer false to a caller.
 *
 * Not ported yet (later units): deal_chunks, chunk_coords, sgpr_ptr,
 * RowSlab.sgpr_ptr, StridedTensorAddr.offset_bytes / stride_bytes / slab_at,
 * StridedBatchedAddr, ThdOffsetAddr, WsSlab.sgpr_ptr / load / load_raw /
 * neutral / atomic_add / atomic_add_row / atomic_add_row_select / store_row /
 * store_vec_in_row, and PackedRows. The slab structs already carry the
 * (base, offset) parts sgpr_ptr needs.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_ADDR_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_ADDR_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/helper_rocke.helpers.attention_band.h" /* ROCKE_ATTN_I32_FAR */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ IntOrValue */

/* A host int (v == NULL, value in i) or an SSA value (v != NULL). */
typedef struct rocke_attn_bwd_iv
{
    rocke_value_t* v;
    int64_t i;
} rocke_attn_bwd_iv_t;

static inline rocke_attn_bwd_iv_t rocke_attn_bwd_iv_int(int64_t i)
{
    rocke_attn_bwd_iv_t x;
    x.v = NULL;
    x.i = i;
    return x;
}

static inline rocke_attn_bwd_iv_t rocke_attn_bwd_iv_val(rocke_value_t* v)
{
    rocke_attn_bwd_iv_t x;
    x.v = v;
    x.i = 0;
    return x;
}

/* Python `isinstance(x, int) and x == i`. */
static inline bool rocke_attn_bwd_iv_is_int(rocke_attn_bwd_iv_t x, int64_t i)
{
    return x.v == NULL && x.i == i;
}

/* as_i32: an int as an i32 constant; an SSA value unchanged. */
rocke_value_t* rocke_attn_bwd_as_i32(rocke_ir_builder_t* b, rocke_attn_bwd_iv_t x);
/* as_i64: an int as an i64 constant; an i64 value unchanged; an i32 value
 * zext (sext when is_signed). */
rocke_value_t* rocke_attn_bwd_as_i64(rocke_ir_builder_t* b, rocke_attn_bwd_iv_t x, bool is_signed);
/* last_row: max(length - 1, 0). */
rocke_value_t* rocke_attn_bwd_last_row(rocke_ir_builder_t* b, rocke_value_t* length);

/* ------------------------------------------------------- sequence decode */

typedef enum rocke_attn_bwd_seq_mode
{
    ROCKE_ATTN_BWD_SEQ_BATCHED = 0,
    ROCKE_ATTN_BWD_SEQ_THD = 1
} rocke_attn_bwd_seq_mode_t;

/* SeqInfo: tok0 is NULL in batched mode. */
typedef struct rocke_attn_bwd_seq_info
{
    rocke_attn_bwd_seq_mode_t mode;
    rocke_value_t* batch;
    rocke_value_t* length;
    rocke_value_t* last;
    rocke_value_t* s_max;
    rocke_value_t* ws_row0;
    rocke_value_t* tok0;
} rocke_attn_bwd_seq_info_t;

/* load_offset: entry idx of an int32 / int64 offset table, as i64. */
rocke_value_t* rocke_attn_bwd_load_offset(rocke_ir_builder_t* b,
                                          rocke_value_t* off_ptr,
                                          rocke_value_t* idx,
                                          rocke_attn_bwd_iv_t off64);

/* Keyword arguments of decode_batched / decode_thd. Fill the defaults with
 * rocke_attn_bwd_decode_args_default() (Python's keyword defaults: off64 0,
 * mult 1, div 1, has_len 0, len_stride 1, ws_seg 0, no pointers), then set the
 * fields the call passes. s_max has no default. decode_batched reads only
 * s_max, seq_ptr, has_len and len_stride. */
typedef struct rocke_attn_bwd_decode_args
{
    rocke_attn_bwd_iv_t s_max;
    rocke_value_t* off_ptr;
    rocke_attn_bwd_iv_t off64;
    rocke_attn_bwd_iv_t mult;
    rocke_attn_bwd_iv_t div;
    rocke_value_t* seq_ptr;
    rocke_attn_bwd_iv_t has_len;
    rocke_attn_bwd_iv_t len_stride;
    rocke_attn_bwd_iv_t ws_seg;
} rocke_attn_bwd_decode_args_t;

rocke_attn_bwd_decode_args_t rocke_attn_bwd_decode_args_default(void);

rocke_attn_bwd_seq_info_t rocke_attn_bwd_decode_batched(rocke_ir_builder_t* b,
                                                        int* seq_guards,
                                                        rocke_value_t* batch,
                                                        const rocke_attn_bwd_decode_args_t* args);
rocke_attn_bwd_seq_info_t rocke_attn_bwd_decode_thd(rocke_ir_builder_t* b,
                                                    int* seq_guards,
                                                    rocke_value_t* batch,
                                                    const rocke_attn_bwd_decode_args_t* args);

/* ------------------------------------------------------- strided tensors */

/* RowSlab. last == NULL: rows are not clamped. parts_* mirror Python's
 * `parts` tuple (base, byte offset or NULL, ptr); has_parts false is None. */
typedef struct rocke_attn_bwd_row_slab
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    const rocke_type_t* dtype;
    rocke_attn_bwd_iv_t t_stride;
    rocke_value_t* base_row;
    rocke_value_t* last;
    int stage_vec;
    int narrow;
    bool has_parts;
    rocke_value_t* parts_base;
    rocke_value_t* parts_off;
} rocke_attn_bwd_row_slab_t;

rocke_value_t* rocke_attn_bwd_row_slab_local_row(const rocke_attn_bwd_row_slab_t* s,
                                                 rocke_value_t* row);
rocke_value_t* rocke_attn_bwd_row_slab_index(const rocke_attn_bwd_row_slab_t* s,
                                             rocke_value_t* row,
                                             rocke_attn_bwd_iv_t col);
rocke_value_t* rocke_attn_bwd_row_slab_load(const rocke_attn_bwd_row_slab_t* s,
                                            rocke_value_t* row,
                                            rocke_attn_bwd_iv_t col);
void rocke_attn_bwd_row_slab_store(const rocke_attn_bwd_row_slab_t* s,
                                   rocke_value_t* row,
                                   rocke_attn_bwd_iv_t col,
                                   rocke_value_t* value);
rocke_value_t* rocke_attn_bwd_row_slab_load_vec(const rocke_attn_bwd_row_slab_t* s,
                                                rocke_value_t* row,
                                                rocke_attn_bwd_iv_t col0,
                                                int n);
void rocke_attn_bwd_row_slab_store_vec(const rocke_attn_bwd_row_slab_t* s,
                                       rocke_value_t* row,
                                       rocke_attn_bwd_iv_t col0,
                                       rocke_value_t* value);

/* StridedTensorAddr: strides are (batch, head, token) element strides. */
typedef struct rocke_attn_bwd_strided_tensor_addr
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    const rocke_type_t* dtype;
    rocke_attn_bwd_iv_t strides[3];
    rocke_attn_bwd_seq_info_t seq;
    int stage_vec;
    int narrow_vec;
    int esize;
    rocke_value_t* base;
    rocke_value_t* seq_base; /* NULL in THD mode (folded into base) */
} rocke_attn_bwd_strided_tensor_addr_t;

/* StridedTensorAddr.__init__. A Python ValueError raises (see Errors). */
bool rocke_attn_bwd_strided_tensor_addr_init(rocke_attn_bwd_strided_tensor_addr_t* t,
                                             rocke_ir_builder_t* b,
                                             rocke_value_t* ptr,
                                             const rocke_type_t* dtype,
                                             const rocke_attn_bwd_iv_t strides[3],
                                             const rocke_attn_bwd_seq_info_t* seq,
                                             int stage_vec,
                                             int narrow_vec);
rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_strided_tensor_addr_slab(const rocke_attn_bwd_strided_tensor_addr_t* t,
                                            rocke_attn_bwd_iv_t h,
                                            rocke_attn_bwd_iv_t row0,
                                            bool clamp);

/* -------------------------------------------------------- fp32 workspace */

typedef enum rocke_attn_bwd_ws_layout
{
    ROCKE_ATTN_BWD_WS_HEAD_MAJOR = 0,
    ROCKE_ATTN_BWD_WS_TOKEN_MAJOR = 1
} rocke_attn_bwd_ws_layout_t;

typedef struct rocke_attn_bwd_ws_slab
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    rocke_value_t* rows_left;
    rocke_attn_bwd_iv_t row_stride;
    bool has_parts;
    rocke_value_t* parts_base;
    rocke_value_t* parts_off;
} rocke_attn_bwd_ws_slab_t;

rocke_value_t* rocke_attn_bwd_ws_slab_row_ok(const rocke_attn_bwd_ws_slab_t* s, rocke_value_t* r);
rocke_value_t* rocke_attn_bwd_ws_slab_local_row(const rocke_attn_bwd_ws_slab_t* s,
                                                rocke_value_t* r);
rocke_value_t* rocke_attn_bwd_ws_slab_index(const rocke_attn_bwd_ws_slab_t* s,
                                            rocke_value_t* r,
                                            rocke_attn_bwd_iv_t x);
void rocke_attn_bwd_ws_slab_store(const rocke_attn_bwd_ws_slab_t* s,
                                  rocke_value_t* r,
                                  rocke_attn_bwd_iv_t x,
                                  rocke_value_t* v);
void rocke_attn_bwd_ws_slab_store_vec(const rocke_attn_bwd_ws_slab_t* s,
                                      rocke_value_t* r,
                                      rocke_attn_bwd_iv_t x,
                                      rocke_value_t* v);
rocke_value_t* rocke_attn_bwd_ws_slab_load_vec4(const rocke_attn_bwd_ws_slab_t* s,
                                                rocke_value_t* r,
                                                rocke_attn_bwd_iv_t x);

/* WorkspaceAddr. has_n_heads false is Python's n_heads=None. */
typedef struct rocke_attn_bwd_workspace_addr
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    rocke_value_t* ws_rows;
    int width;
    rocke_value_t* row0;
    rocke_attn_bwd_ws_layout_t layout;
    bool has_n_heads;
    rocke_attn_bwd_iv_t n_heads;
} rocke_attn_bwd_workspace_addr_t;

/* WorkspaceAddr.__init__ (n_heads NULL is None). A Python ValueError raises
 * (see Errors). */
bool rocke_attn_bwd_workspace_addr_init(rocke_attn_bwd_workspace_addr_t* w,
                                        rocke_ir_builder_t* b,
                                        rocke_value_t* ptr,
                                        rocke_attn_bwd_iv_t ws_rows,
                                        int width,
                                        rocke_value_t* row0,
                                        rocke_attn_bwd_ws_layout_t layout,
                                        const rocke_attn_bwd_iv_t* n_heads);
rocke_attn_bwd_ws_slab_t rocke_attn_bwd_workspace_addr_slab(
    const rocke_attn_bwd_workspace_addr_t* w, rocke_attn_bwd_iv_t h, rocke_attn_bwd_iv_t tile_row0);

/* ---------------------------------------------------------- paged caches */

rocke_value_t* rocke_attn_bwd_block_table_entry(rocke_ir_builder_t* b,
                                                rocke_value_t* bt_ptr,
                                                rocke_value_t* seq_idx,
                                                rocke_value_t* j,
                                                rocke_attn_bwd_iv_t bt_stride);

/* PagedCacheAddr: strides are (blk, tok, h) element strides. */
typedef struct rocke_attn_bwd_paged_cache_addr
{
    rocke_ir_builder_t* b;
    rocke_value_t* ptr;
    const rocke_type_t* dtype;
    rocke_attn_bwd_iv_t strides[3];
    int stage_vec;
    int esize;
} rocke_attn_bwd_paged_cache_addr_t;

bool rocke_attn_bwd_paged_cache_addr_init(rocke_attn_bwd_paged_cache_addr_t* p,
                                          rocke_ir_builder_t* b,
                                          rocke_value_t* ptr,
                                          const rocke_type_t* dtype,
                                          const rocke_attn_bwd_iv_t strides[3],
                                          int stage_vec);
rocke_attn_bwd_row_slab_t
    rocke_attn_bwd_paged_cache_addr_slab(const rocke_attn_bwd_paged_cache_addr_t* p,
                                         rocke_value_t* page_id,
                                         rocke_attn_bwd_iv_t h,
                                         rocke_attn_bwd_iv_t tok0);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_ADDR_H */
