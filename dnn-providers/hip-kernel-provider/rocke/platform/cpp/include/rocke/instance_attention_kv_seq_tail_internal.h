// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Paged K/V tail internals. Mirrors library/kernels/common/_attention_kv_seq_tail.py op for op
 * and in the same order, so the two engines emit byte-identical IR. Every builder call is its own
 * statement: C leaves the evaluation order of call arguments unspecified, and the order the ops
 * are created in decides their SSA names. Not a public ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H

#include <stdint.h>

#include "rocke/ir.h"

/* OUT_OF_RANGE_OFFSET: past any buffer resource the kernels build. */
#define ROCKE_KV_SEQ_TAIL_OUT_OF_RANGE 0x7FFFFFFF

/* rows_in_sequence: seq_len - tile * tile_size. */
static inline rocke_value_t* rocke_kv_rows_in_sequence(rocke_ir_builder_t* b,
                                                       rocke_value_t* seq_len,
                                                       rocke_value_t* tile,
                                                       int tile_size)
{
    rocke_value_t* size = rocke_b_const_i32(b, tile_size);
    rocke_value_t* tile_first = rocke_b_mul(b, tile, size);
    return rocke_b_sub(b, seq_len, tile_first);
}

/* buffer_offset_in_sequence: voff for row < rows, an out-of-range offset past it. */
static inline rocke_value_t* rocke_kv_buffer_offset_in_sequence(rocke_ir_builder_t* b,
                                                                rocke_value_t* voff,
                                                                rocke_value_t* row,
                                                                rocke_value_t* rows)
{
    rocke_value_t* inside = rocke_b_cmp_lt(b, row, rows);
    rocke_value_t* past = rocke_b_const_i32(b, ROCKE_KV_SEQ_TAIL_OUT_OF_RANGE);
    return rocke_b_select(b, inside, voff, past);
}

/* buffer_offset_in_rows_left: voff while the immediate row is below rows_left; past it, an
 * out-of-range offset (select), or 2**31 added (add nuw) when rows_left is wave-uniform. */
static inline rocke_value_t* rocke_kv_buffer_offset_in_rows_left(
    rocke_ir_builder_t* b, rocke_value_t* voff, rocke_value_t* rows_left, int row, bool uniform)
{
    rocke_value_t* row_c = rocke_b_const_i32(b, row);
    rocke_value_t* inside = rocke_b_cmp_lt(b, row_c, rows_left);
    if(uniform)
    {
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* bit31 = rocke_b_const_i32(b, INT32_MIN);
        rocke_value_t* penalty = rocke_b_select(b, inside, zero, bit31);
        return rocke_b_add_nuw(b, voff, penalty);
    }
    rocke_value_t* past = rocke_b_const_i32(b, ROCKE_KV_SEQ_TAIL_OUT_OF_RANGE);
    return rocke_b_select(b, inside, voff, past);
}

/* last_row_in_sequence: max(rows_in_sequence - 1, 0). */
static inline rocke_value_t* rocke_kv_last_row_in_sequence(rocke_ir_builder_t* b,
                                                           rocke_value_t* seq_len,
                                                           rocke_value_t* tile,
                                                           int tile_size)
{
    rocke_value_t* rows = rocke_kv_rows_in_sequence(b, seq_len, tile, tile_size);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* last = rocke_b_sub(b, rows, one);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    return rocke_b_smax(b, last, zero);
}

#endif /* ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H */
