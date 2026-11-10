// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Paged K/V copies of the tiled 2D kernels: a tile's bf16/fp16 K/V copies addressed by page
 * (one wave-uniform page id per page, read before the copies). Mirrors
 * library/kernels/common/_attention_paged_kv_copies.py op for op and in the same order, so the
 * two engines emit byte-identical IR. Every builder call is its own statement: C leaves the
 * evaluation order of call arguments unspecified, and the order the ops are created in decides
 * their SSA names. Not a public ABI. */
#ifndef ROCKE_INSTANCE_ATTENTION_PAGED_KV_COPIES_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_PAGED_KV_COPIES_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "rocke/arena.h"
#include "rocke/instance_attention_kv_seq_tail_internal.h"
#include "rocke/ir.h"

/* PagedKVCopies. The caller fills the inputs and calls rocke_kv_page_copies_init (no ops);
 * then rocke_kv_page_copies_emit_offsets where the kernel builds its K/V offsets, and per
 * loader rocke_kv_page_copies_tile_pages once and rocke_kv_page_copies_copy per copy. */
typedef struct rocke_kv_page_copies
{
    /* inputs */
    rocke_ir_builder_t* b;
    int block_size;
    int head_size;
    int halves_per_call;
    int num_warps;
    int blocks_per_tile;
    int calls_per_tile;
    int kv_bytes;
    int head_stride_b;
    rocke_value_t* lane_half_base;
    rocke_value_t* wave_id; /* NULL with one warp */
    rocke_value_t* kv_head;
    rocke_value_t* token_stride_b;
    rocke_value_t* page_stride_b;
    rocke_value_t* page_bytes;
    rocke_value_t* zero_soff;
    bool i64;
    bool page_base_in_soffset;
    /* derived by init */
    int wave_halves;
    bool wave_part_in_page;
    int copy_tokens;
    bool whole_row_copies;
    bool uniform_blocks;
    /* emitted by emit_offsets */
    rocke_value_t* lane_within_b;
    rocke_value_t** copy_tok_b; /* by the copy's first token in its page */
    rocke_value_t* page_stride_i64;
} rocke_kv_page_copies_t;

/* A tile's pages, per copy: its block (a constant, or -1 and a wave-uniform value), its page
 * id, the first copy with the same page (the Python dict key), and the page's pointer and
 * resource (64-bit offsets; kept at the first copy of the page). */
typedef struct rocke_kv_tile_pages
{
    int n_calls;
    int* blk_const;
    rocke_value_t** blk_val;
    rocke_value_t** page;
    int* first;
    rocke_value_t** ptr;
    rocke_value_t** rsrc;
} rocke_kv_tile_pages_t;

/* uniform_blocks: every wave's part of each copy lies in the same block of the tile. A pure
 * function of the geometry, for callers that decide before the helper exists. */
static inline bool rocke_kv_page_copies_uniform_blocks(int block_size,
                                                       int head_size,
                                                       int halves_per_call,
                                                       int blocks_per_tile)
{
    const int page_halves = block_size * head_size;
    return blocks_per_tile == 1
           || (halves_per_call <= page_halves && page_halves % halves_per_call == 0);
}

static inline void rocke_kv_page_copies_init(rocke_kv_page_copies_t* pc)
{
    const int page_halves = pc->block_size * pc->head_size;
    pc->wave_halves = pc->halves_per_call / pc->num_warps;
    pc->wave_part_in_page = page_halves % pc->wave_halves == 0;
    pc->copy_tokens = pc->halves_per_call / pc->head_size;
    pc->whole_row_copies
        = pc->halves_per_call % pc->head_size == 0 && pc->copy_tokens > 0
          && (pc->block_size % pc->copy_tokens == 0 || pc->copy_tokens % pc->block_size == 0);
    pc->uniform_blocks = rocke_kv_page_copies_uniform_blocks(
        pc->block_size, pc->head_size, pc->halves_per_call, pc->blocks_per_tile);
    pc->lane_within_b = NULL;
    pc->copy_tok_b = NULL;
    pc->page_stride_i64 = NULL;
}

static inline void* rocke_kv_page_copies_calloc(rocke_ir_builder_t* b, size_t n, size_t size)
{
    void* p = rocke_arena_calloc(&b->arena, (n > 0 ? n : 1) * size);
    if(p == NULL)
    {
        b->status = ROCKE_ERR_OOM;
        snprintf(b->err, ROCKE_ERR_MSG_CAP, "out of memory for a %zu-entry K/V page table", n);
    }
    return p;
}

/* emit_offsets: the lane's offset in its page and the copies' token offsets (whole-row
 * copies), and the 64-bit page stride. False on OOM (the builder's error is set). */
static inline bool rocke_kv_page_copies_emit_offsets(rocke_kv_page_copies_t* pc)
{
    rocke_ir_builder_t* b = pc->b;
    if(pc->whole_row_copies)
    {
        pc->copy_tok_b = (rocke_value_t**)rocke_kv_page_copies_calloc(
            b, (size_t)pc->block_size, sizeof(rocke_value_t*));
        if(pc->copy_tok_b == NULL)
            return false;
        rocke_value_t* c_hd0 = rocke_b_const_i32(b, pc->head_size);
        rocke_value_t* lane_tok = rocke_b_div(b, pc->lane_half_base, c_hd0);
        if(pc->block_size % pc->copy_tokens != 0)
        {
            /* A copy covers whole pages: the lane's row within its page. */
            rocke_value_t* c_bs = rocke_b_const_i32(b, pc->block_size);
            lane_tok = rocke_b_mod(b, lane_tok, c_bs);
        }
        rocke_value_t* c_hd1 = rocke_b_const_i32(b, pc->head_size);
        rocke_value_t* lane_dim = rocke_b_mod(b, pc->lane_half_base, c_hd1);
        rocke_value_t* tok_b = rocke_b_mul(b, lane_tok, pc->token_stride_b);
        rocke_value_t* c_h = rocke_b_const_i32(b, pc->head_stride_b);
        rocke_value_t* head_b = rocke_b_mul(b, pc->kv_head, c_h);
        rocke_value_t* tok_head = rocke_b_add(b, tok_b, head_b);
        rocke_value_t* c_bytes = rocke_b_const_i32(b, pc->kv_bytes);
        rocke_value_t* dim_b = rocke_b_mul(b, lane_dim, c_bytes);
        pc->lane_within_b = rocke_b_add(b, tok_head, dim_b);
        for(int call = 0; call < pc->calls_per_tile; ++call)
        {
            const int tok0 = (call * pc->copy_tokens) % pc->block_size;
            if(tok0 == 0 || pc->copy_tok_b[tok0] != NULL)
                continue;
            rocke_value_t* c_tok = rocke_b_const_i32(b, tok0);
            pc->copy_tok_b[tok0] = rocke_b_mul(b, pc->token_stride_b, c_tok);
        }
    }
    if(pc->i64)
        pc->page_stride_i64 = rocke_b_zext(b, pc->page_stride_b, rocke_i64());
    return true;
}

/* copy_block: the tile's block that the calling wave's part of copy `call` lies in: a
 * constant (*blk_const >= 0) when every wave's part lies in the same block, else a
 * wave-uniform value (*blk_val). */
static inline void rocke_kv_page_copies_copy_block(const rocke_kv_page_copies_t* pc,
                                                   int call,
                                                   int* blk_const,
                                                   rocke_value_t** blk_val)
{
    rocke_ir_builder_t* b = pc->b;
    const int page_halves = pc->block_size * pc->head_size;
    if(pc->uniform_blocks)
    {
        *blk_const = (call * pc->halves_per_call) / page_halves;
        *blk_val = NULL;
        return;
    }
    rocke_value_t* first_c = rocke_b_const_i32(b, call * pc->halves_per_call);
    rocke_value_t* wave_halves = rocke_b_const_i32(b, pc->wave_halves);
    rocke_value_t* wave_mul = rocke_b_mul(b, pc->wave_id, wave_halves);
    rocke_value_t* wave_off = rocke_b_to_sgpr_u32(b, wave_mul);
    rocke_value_t* wave_first = rocke_b_add(b, first_c, wave_off);
    rocke_value_t* page_halves_c = rocke_b_const_i32(b, page_halves);
    *blk_const = -1;
    *blk_val = rocke_b_div(b, wave_first, page_halves_c);
}

/* tile_pages: each copy's block and the page ids of the tile's blocks: `carried` (by block,
 * read by the loop ahead), or read once here before the copies (rocke_kv_tile_page_id; a
 * block from the wave per copy). False on OOM. */
static inline bool rocke_kv_page_copies_tile_pages(const rocke_kv_page_copies_t* pc,
                                                   rocke_value_t* block_tables,
                                                   rocke_value_t* seq_base,
                                                   rocke_value_t* max_idx,
                                                   rocke_value_t* tile,
                                                   rocke_value_t* const* carried,
                                                   rocke_kv_tile_pages_t* out)
{
    rocke_ir_builder_t* b = pc->b;
    const size_t n = (size_t)pc->calls_per_tile;
    const size_t vp = sizeof(rocke_value_t*);
    out->n_calls = pc->calls_per_tile;
    out->blk_const = (int*)rocke_kv_page_copies_calloc(b, n, sizeof(int));
    out->first = (int*)rocke_kv_page_copies_calloc(b, n, sizeof(int));
    out->blk_val = (rocke_value_t**)rocke_kv_page_copies_calloc(b, n, vp);
    out->page = (rocke_value_t**)rocke_kv_page_copies_calloc(b, n, vp);
    out->ptr = (rocke_value_t**)rocke_kv_page_copies_calloc(b, n, vp);
    out->rsrc = (rocke_value_t**)rocke_kv_page_copies_calloc(b, n, vp);
    if(out->blk_const == NULL || out->first == NULL || out->blk_val == NULL || out->page == NULL
       || out->ptr == NULL || out->rsrc == NULL)
        return false;
    for(int call = 0; call < out->n_calls; ++call)
        rocke_kv_page_copies_copy_block(pc, call, &out->blk_const[call], &out->blk_val[call]);
    for(int call = 0; call < out->n_calls; ++call)
    {
        const int blk = out->blk_const[call];
        out->first[call] = call;
        if(blk >= 0)
            for(int c = 0; c < call; ++c)
                if(out->blk_const[c] == blk)
                {
                    out->first[call] = c;
                    break;
                }
        if(carried != NULL && blk < 0)
        {
            /* Python: KeyError (carried ids are by constant block only). */
            b->status = ROCKE_ERR_VALUE;
            snprintf(b->err, ROCKE_ERR_MSG_CAP, "carried page ids need constant copy blocks");
            return false;
        }
        if(carried != NULL)
            out->page[call] = carried[blk];
        else if(out->first[call] != call)
            out->page[call] = out->page[out->first[call]];
        else
            out->page[call] = rocke_kv_tile_page_id(b,
                                                    block_tables,
                                                    seq_base,
                                                    max_idx,
                                                    tile,
                                                    blk,
                                                    out->blk_val[call],
                                                    pc->blocks_per_tile);
    }
    return true;
}

/* copy: (pointer, resource, voffset, soffset) of copy `call`. */
static inline void rocke_kv_page_copies_copy(const rocke_kv_page_copies_t* pc,
                                             rocke_kv_tile_pages_t* tp,
                                             int call,
                                             rocke_value_t* src,
                                             rocke_value_t* src_rsrc,
                                             rocke_value_t** out_ptr,
                                             rocke_value_t** out_rsrc,
                                             rocke_value_t** out_voff,
                                             rocke_value_t** out_soff)
{
    rocke_ir_builder_t* b = pc->b;
    const int blk = tp->blk_const[call];
    rocke_value_t* page = tp->page[call];
    rocke_value_t* voff;
    if(pc->whole_row_copies)
    {
        voff = pc->lane_within_b;
        const int tok0 = (call * pc->copy_tokens) % pc->block_size;
        if(tok0 != 0)
            voff = rocke_b_add(b, voff, pc->copy_tok_b[tok0]);
    }
    else
    {
        const int page_halves = pc->block_size * pc->head_size;
        rocke_value_t* start;
        if(blk >= 0)
            start = rocke_b_const_i32(b, blk * page_halves);
        else
        {
            rocke_value_t* c_ph = rocke_b_const_i32(b, page_halves);
            start = rocke_b_mul(b, tp->blk_val[call], c_ph);
        }
        rocke_value_t* first = rocke_b_const_i32(b, call * pc->halves_per_call);
        rocke_value_t* linear = rocke_b_add(b, first, pc->lane_half_base);
        rocke_value_t* half = rocke_b_sub(b, linear, start);
        rocke_value_t* c_hd0 = rocke_b_const_i32(b, pc->head_size);
        rocke_value_t* tok = rocke_b_div(b, half, c_hd0);
        rocke_value_t* c_hd1 = rocke_b_const_i32(b, pc->head_size);
        rocke_value_t* dim = rocke_b_mod(b, half, c_hd1);
        rocke_value_t* tok_b = rocke_b_mul(b, tok, pc->token_stride_b);
        rocke_value_t* c_h = rocke_b_const_i32(b, pc->head_stride_b);
        rocke_value_t* head_b = rocke_b_mul(b, pc->kv_head, c_h);
        rocke_value_t* tok_head = rocke_b_add(b, tok_b, head_b);
        rocke_value_t* c_bytes = rocke_b_const_i32(b, pc->kv_bytes);
        rocke_value_t* dim_b = rocke_b_mul(b, dim, c_bytes);
        voff = rocke_b_add(b, tok_head, dim_b);
    }
    *out_voff = voff;
    if(pc->i64)
    {
        /* One pointer and resource per page, shared by its copies. */
        const int first = tp->first[call];
        if(tp->rsrc[first] == NULL)
        {
            rocke_value_t* page_i64 = rocke_b_zext(b, page, rocke_i64());
            rocke_value_t* base_i64 = rocke_b_mul(b, page_i64, pc->page_stride_i64);
            tp->ptr[first] = rocke_b_global_ptr_add(b, src, base_i64);
            tp->rsrc[first] = rocke_b_buffer_rsrc(b, tp->ptr[first], pc->page_bytes);
        }
        *out_ptr = tp->ptr[first];
        *out_rsrc = tp->rsrc[first];
        *out_soff = pc->zero_soff;
        return;
    }
    rocke_value_t* page_b = rocke_b_mul(b, page, pc->page_stride_b);
    *out_ptr = src;
    *out_rsrc = src_rsrc;
    if(pc->page_base_in_soffset)
    {
        *out_soff = page_b;
        return;
    }
    *out_voff = rocke_b_add(b, voff, page_b);
    *out_soff = pc->zero_soff;
}

#endif /* ROCKE_INSTANCE_ATTENTION_PAGED_KV_COPIES_INTERNAL_H */
