// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Paged K/V tail internals: V rows past the sequence load zero, block-table reads stay inside the
 * sequence's entries. Mirrors library/kernels/common/_attention_kv_seq_tail.py op for op
 * and in the same order, so the two engines emit byte-identical IR. Every builder call is its own
 * statement: C leaves the evaluation order of call arguments unspecified, and the order the ops
 * are created in decides their SSA names. Not a public ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H

#include <stdbool.h>
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

/* block_table_max_idx: the end of the sequence's own block-table entries (seq_base +
 * ceil(seq_len / block_size)), and of the table; wave-uniform. */
static inline rocke_value_t* rocke_kv_block_table_max_idx(rocke_ir_builder_t* b,
                                                          rocke_value_t* seq_base,
                                                          rocke_value_t* seq_len,
                                                          int block_size,
                                                          rocke_value_t* num_seqs,
                                                          rocke_value_t* bt_stride)
{
    rocke_value_t* round_up = rocke_b_const_i32(b, block_size - 1);
    rocke_value_t* len_up = rocke_b_add(b, seq_len, round_up);
    rocke_value_t* bs = rocke_b_const_i32(b, block_size);
    rocke_value_t* n_blocks = rocke_b_div(b, len_up, bs);
    rocke_value_t* seq_end = rocke_b_add(b, seq_base, n_blocks);
    rocke_value_t* table_end = rocke_b_mul(b, num_seqs, bt_stride);
    rocke_value_t* end = rocke_b_smin(b, seq_end, table_end);
    return rocke_b_to_sgpr_u32(b, end);
}

/* tile_page_id: the page id of block `blk` of `tile` (or, when `blk_val` is not NULL, of the
 * wave-uniform block it holds): one block-table read, 0 past max_idx; wave-uniform. */
static inline rocke_value_t* rocke_kv_tile_page_id(rocke_ir_builder_t* b,
                                                   rocke_value_t* block_tables,
                                                   rocke_value_t* seq_base,
                                                   rocke_value_t* max_idx,
                                                   rocke_value_t* tile,
                                                   int blk,
                                                   rocke_value_t* blk_val,
                                                   int blocks_per_tile)
{
    rocke_value_t* tile_blk = tile;
    if(blocks_per_tile != 1)
    {
        rocke_value_t* c_nb = rocke_b_const_i32(b, blocks_per_tile);
        rocke_value_t* first = rocke_b_mul(b, tile, c_nb);
        rocke_value_t* c_blk = blk_val != NULL ? blk_val : rocke_b_const_i32(b, blk);
        tile_blk = rocke_b_add(b, first, c_blk);
    }
    rocke_value_t* idx = rocke_b_add(b, seq_base, tile_blk);
    rocke_value_t* mask = rocke_b_cmp_lt(b, idx, max_idx);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* page
        = rocke_b_masked_global_load(b, block_tables, idx, mask, zero, rocke_i32(), 4);
    return rocke_b_to_sgpr_u32(b, page);
}

/* VTileBound: the bound of each bf16/fp16 V copy of a tile. Init emits the per-tile part,
 * apply bounds one copy. `lane_row` NULL: built here (0 when uniform with one row per copy). */
typedef struct rocke_kv_v_tile_bound
{
    rocke_ir_builder_t* b;
    int head_size;
    int halves_per_call;
    rocke_value_t* lane_half_base;
    bool whole_rows;
    bool in_row;
    bool uniform;
    rocke_value_t* rows;
    rocke_value_t* rows_left;
} rocke_kv_v_tile_bound_t;

static inline void rocke_kv_v_tile_bound_init(rocke_kv_v_tile_bound_t* vb,
                                              rocke_ir_builder_t* b,
                                              rocke_value_t* seq_len,
                                              rocke_value_t* tile,
                                              int tile_size,
                                              int head_size,
                                              int halves_per_call,
                                              rocke_value_t* lane_half_base,
                                              rocke_value_t* lane_row,
                                              bool uniform,
                                              bool in_row)
{
    vb->b = b;
    vb->head_size = head_size;
    vb->halves_per_call = halves_per_call;
    vb->lane_half_base = lane_half_base;
    vb->whole_rows = halves_per_call % head_size == 0;
    vb->in_row = in_row && !vb->whole_rows && head_size % halves_per_call == 0;
    vb->uniform = false;
    vb->rows_left = NULL;
    vb->rows = rocke_kv_rows_in_sequence(b, seq_len, tile, tile_size);
    if(vb->whole_rows)
    {
        vb->uniform = uniform && lane_row == NULL && halves_per_call == head_size;
        if(lane_row == NULL)
        {
            if(vb->uniform)
                lane_row = rocke_b_const_i32(b, 0);
            else
            {
                rocke_value_t* hd = rocke_b_const_i32(b, head_size);
                lane_row = rocke_b_div(b, lane_half_base, hd);
            }
        }
        vb->rows_left = rocke_b_sub(b, vb->rows, lane_row);
    }
}

/* voff of copy `call` bounded to the sequence. A copy that ends mid-row takes the lane's `row`
 * if not NULL, else its row from `linear_half` (built here if NULL). */
static inline rocke_value_t* rocke_kv_v_tile_bound_apply(const rocke_kv_v_tile_bound_t* vb,
                                                         rocke_value_t* voff,
                                                         int call,
                                                         rocke_value_t* row,
                                                         rocke_value_t* linear_half)
{
    rocke_ir_builder_t* b = vb->b;
    if(vb->whole_rows)
        return rocke_kv_buffer_offset_in_rows_left(
            b, voff, vb->rows_left, call * (vb->halves_per_call / vb->head_size), vb->uniform);
    if(vb->in_row)
        return rocke_kv_buffer_offset_in_rows_left(
            b, voff, vb->rows, call / (vb->head_size / vb->halves_per_call), true);
    if(row == NULL)
    {
        if(linear_half == NULL)
        {
            rocke_value_t* first = rocke_b_const_i32(b, call * vb->halves_per_call);
            linear_half = rocke_b_add(b, first, vb->lane_half_base);
        }
        rocke_value_t* hd = rocke_b_const_i32(b, vb->head_size);
        row = rocke_b_div(b, linear_half, hd);
    }
    return rocke_kv_buffer_offset_in_sequence(b, voff, row, vb->rows);
}

#endif /* ROCKE_INSTANCE_ATTENTION_KV_SEQ_TAIL_INTERNAL_H */
