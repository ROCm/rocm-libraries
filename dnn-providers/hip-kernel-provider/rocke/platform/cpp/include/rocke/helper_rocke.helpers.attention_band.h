/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.attention_band.h -- C port of
 * rocke.helpers.attention_band (the runtime two-sided diagonal band of the
 * attention shells).
 *
 * Band semantics (the keep rule):
 *
 *     keep(q, k) = k < seqlen_k
 *              and (right < 0 or k <= q + diag_offset + right)
 *              and (left  < 0 or k >= q + diag_offset - left)
 *
 *   * AttnRuntimeBounds -> rocke_attn_runtime_bounds_t (+ _make / _validate)
 *   * RowBand           -> rocke_row_band_t
 *   * BandEmitter       -> rocke_band_emitter_t and rocke_band_*
 *   * I32_FAR           -> ROCKE_ATTN_I32_FAR
 *
 * Python takes "int or i32 Value" in several places; the C port spells that as
 * rocke_int_or_value_t. A field Python holds as None is an unset
 * rocke_band_limit_t (has == false) or a NULL Value pointer.
 *
 * Errors: the Python ValueError cases record ROCKE_ERR_VALUE and the Python
 * message on the builder (sticky error); Value-returning entries return NULL
 * and status-returning entries return the status code.
 *
 * Every emitter reproduces the Python builder-call order exactly (the .ll must
 * match byte for byte); sub-emissions Python nests as call arguments are bound
 * to locals in Python's left-to-right order.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_BAND_H
#define ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_BAND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A limit past every row, key, length and band bound (all stay below 2**30). */
#define ROCKE_ATTN_I32_FAR (1 << 30)

/* IntOrValue: a build-time int or an i32 SSA value. */
typedef struct rocke_int_or_value
{
    bool is_value; /* true: `v` is set; false: `i` is set */
    int64_t i;
    rocke_value_t* v;
} rocke_int_or_value_t;

rocke_int_or_value_t rocke_iov_int(int64_t i);
rocke_int_or_value_t rocke_iov_value(rocke_value_t* v);

/* AttnRuntimeBounds (frozen dataclass). Defaults: diag_offset 0, left_bound -1,
 * right_bound -1, skip_tiles true. */
typedef struct rocke_attn_runtime_bounds
{
    rocke_int_or_value_t seqlen_q;
    rocke_int_or_value_t seqlen_k;
    rocke_int_or_value_t diag_offset;
    rocke_int_or_value_t left_bound;
    rocke_int_or_value_t right_bound;
    bool skip_tiles;
} rocke_attn_runtime_bounds_t;

/* The dataclass with the field defaults and the two required lengths. */
rocke_attn_runtime_bounds_t rocke_attn_runtime_bounds_make(rocke_int_or_value_t seqlen_q,
                                                           rocke_int_or_value_t seqlen_k);

/* __post_init__: an int left_bound / right_bound below -1 is a ValueError.
 * Returns 1 if valid, else 0 with the message in `reason` (may be NULL). */
int rocke_attn_runtime_bounds_validate(const rocke_attn_runtime_bounds_t* bounds,
                                       char* reason,
                                       size_t reason_cap);

/* RowBand: loop-invariant band facts of one query row. lo / hi are NULL when
 * the side has no limit (Python None). */
typedef struct rocke_row_band
{
    rocke_value_t* q_pos;
    rocke_value_t* row_in;
    rocke_value_t* valid;
    rocke_value_t* lo;
    rocke_value_t* hi;
} rocke_row_band_t;

/* One side of the band as BandEmitter stores it: has == false is Python None
 * (a static unbounded side). */
typedef struct rocke_band_limit
{
    bool has;
    rocke_int_or_value_t bound;
} rocke_band_limit_t;

/* BandEmitter: the shared limits (seqlen - 1, zero) and the cached FAR. */
typedef struct rocke_band_emitter
{
    rocke_ir_builder_t* b;
    rocke_attn_runtime_bounds_t bounds;
    rocke_value_t* sq;
    rocke_value_t* sk;
    rocke_value_t* zero;
    rocke_value_t* sk_minus1; /* -1 when the key sequence is empty */
    rocke_value_t* off; /* NULL: a static zero diagonal offset */
    rocke_band_limit_t left;
    rocke_band_limit_t right;
    rocke_value_t* far_value; /* const_i32(I32_FAR), created on first use */
} rocke_band_emitter_t;

/* BandEmitter.__init__: validates `bounds`, then emits (in Python's order) the
 * length constants, one, zero, sk - 1 and the offset. Returns ROCKE_OK or the
 * error status (also recorded on `b`). */
rocke_status_t rocke_band_emitter_init(rocke_band_emitter_t* self,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_runtime_bounds_t* bounds);

/* _i32: const_i32 of an int, the Value unchanged. */
rocke_value_t* rocke_band_i32(rocke_ir_builder_t* b, rocke_int_or_value_t x);

/* row_band(q_pos). */
rocke_status_t
    rocke_band_row_band(rocke_band_emitter_t* self, rocke_value_t* q_pos, rocke_row_band_t* out);
/* col_in(k_pos): i1 k_pos < sk. */
rocke_value_t* rocke_band_col_in(rocke_band_emitter_t* self, rocke_value_t* k_pos);
/* cell_keep(rb, k_pos, col_in). */
rocke_value_t* rocke_band_cell_keep(rocke_band_emitter_t* self,
                                    const rocke_row_band_t* rb,
                                    rocke_value_t* k_pos,
                                    rocke_value_t* col_in);
/* k_tile_range(q_pos_first, block_m, block_k) -> [start, stop). */
rocke_status_t rocke_band_k_tile_range(rocke_band_emitter_t* self,
                                       rocke_value_t* q_pos_first,
                                       int block_m,
                                       int block_k,
                                       rocke_value_t** out_start,
                                       rocke_value_t** out_stop);
/* k_tile_count_ceil(block_k). */
rocke_value_t* rocke_band_k_tile_count_ceil(rocke_band_emitter_t* self, int block_k);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_BAND_H */
