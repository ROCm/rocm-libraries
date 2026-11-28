// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/loads.py: CoalescedTileLoader, AsyncTileLoader
 * (and the AsyncTileLoaderSlot that AsyncTileLoader.bind returns).
 *
 * See the header for the original Python and the contract. The builder-emitting
 * methods (load / bind / issue) reproduce the const_i32 / arith / load / store
 * call sequence byte-faithfully so the downstream IR is identical to the Python.
 */
#include "rocke/helper_rocke.helpers.loads.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ============================================================================
 * CoalescedTileLoader
 * ========================================================================== */

rocke_status_t rocke_coalesced_tile_loader_choose_vec_axis(
    int tile_rows, int tile_cols, int block_size, int max_vec, bool vector_axis_row, int* out_vec)
{
    int v;
    int axis;

    /* Python:
     *   axis = tile_rows if vector_axis == "row" else tile_cols
     *   v = max_vec
     *   while v >= 1:
     *       if (axis % v == 0
     *           and (tile_rows*tile_cols)//v >= block_size
     *           and ((tile_rows*tile_cols)//v) % block_size == 0):
     *           return v
     *       v //= 2
     *   raise ValueError(...)
     */
    axis = vector_axis_row ? tile_rows : tile_cols;
    v = max_vec;
    while(v >= 1)
    {
        if(axis % v == 0 && (tile_rows * tile_cols) / v >= block_size
           && ((tile_rows * tile_cols) / v) % block_size == 0)
        {
            if(out_vec != NULL)
            {
                *out_vec = v;
            }
            return ROCKE_OK;
        }
        v /= 2;
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_coalesced_tile_loader_choose_vec(
    int tile_rows, int tile_cols, int block_size, int max_vec, int* out_vec)
{
    /* vector_axis="col" (default). */
    return rocke_coalesced_tile_loader_choose_vec_axis(
        tile_rows, tile_cols, block_size, max_vec, false, out_vec);
}

rocke_status_t rocke_coalesced_tile_loader_from_tile(int tile_rows,
                                                     int tile_cols,
                                                     int block_size,
                                                     int max_vec,
                                                     bool use_buffer_rsrc,
                                                     rocke_coalesced_tile_loader_t* out)
{
    int vec;
    rocke_status_t st;

    /* Python:
     *   vec = cls.choose_vec(...)
     *   return cls(tile_rows=..., tile_cols=..., block_size=...,
     *              load_vec=vec, use_buffer_rsrc=use_buffer_rsrc)
     */
    st = rocke_coalesced_tile_loader_choose_vec(tile_rows, tile_cols, block_size, max_vec, &vec);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(out != NULL)
    {
        out->tile_rows = tile_rows;
        out->tile_cols = tile_cols;
        out->block_size = block_size;
        out->load_vec = vec;
        out->use_buffer_rsrc = use_buffer_rsrc;
        /* Python (1 << 31) - 1 == 2147483647 (arbitrary-precision ints);
         * spell it as the literal to avoid the C int shift-into-sign overflow. */
        out->oob_sentinel = 2147483647; /* dataclass default */
        out->vector_axis_row = false; /* vector_axis="col" default */
        out->has_inner_dim = false; /* inner_dim default None */
        out->inner_dim = 0;
        out->has_col_xor = false; /* col_xor default None */
        memset(out->col_xor, 0, sizeof(out->col_xor));
        out->elem_dtype = NULL; /* elem_dtype default F16 */
    }
    return ROCKE_OK;
}

rocke_status_t rocke_lds_col_xor_params(int tile_cols, int chunk, int out[4])
{
    int c;
    int rows_per_cycle;
    int row_shift;
    int chunk_log2;

    /* Python:
     *   c = tile_cols // chunk
     *   if chunk < 1 or chunk & (chunk - 1): raise ValueError
     *   if c < 2 or tile_cols % chunk or c & (c - 1) or c > 16: raise ValueError
     *   rows_per_cycle = max(1, 8 // c)
     *   return rows_per_cycle.bit_length() - 1, c - 1, chunk.bit_length() - 1, 3
     */
    if(out == NULL || chunk < 1 || (chunk & (chunk - 1)) != 0 || tile_cols < 0)
    {
        return ROCKE_ERR_VALUE;
    }
    c = tile_cols / chunk;
    if(c < 2 || tile_cols % chunk != 0 || (c & (c - 1)) != 0 || c > 16)
    {
        return ROCKE_ERR_VALUE;
    }
    rows_per_cycle = 8 / c > 1 ? 8 / c : 1;
    row_shift = 0;
    while((1 << (row_shift + 1)) <= rows_per_cycle)
    {
        ++row_shift;
    }
    chunk_log2 = 0;
    while((1 << (chunk_log2 + 1)) <= chunk)
    {
        ++chunk_log2;
    }
    out[0] = row_shift;
    out[1] = c - 1;
    out[2] = chunk_log2;
    out[3] = 3;
    return ROCKE_OK;
}

rocke_value_t* rocke_swizzle_lds_col(rocke_ir_builder_t* b,
                                     rocke_value_t* row,
                                     rocke_value_t* col,
                                     const int col_xor[4])
{
    rocke_value_t* r;
    rocke_value_t* sel;
    rocke_value_t* flip;
    rocke_value_t* shifted;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return NULL;
    }
    if(col_xor == NULL)
    {
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "swizzle_lds_col: NULL col_xor");
    }
    /* Python (each const_i32 is created right before the op that takes it):
     *   r = b.lshr(row, b.const_i32(row_shift)) if row_shift else row
     *   sel = b.land(r, b.const_i32(chunk_mask))
     *   flip = b.land(b.lshr(row, b.const_i32(flip_shift)), b.const_i32(1))
     *   sel = b.xor(sel, flip)
     *   return b.xor(col, b.shl(sel, b.const_i32(chunk_log2)))
     */
    r = row;
    if(col_xor[0] != 0)
    {
        rocke_value_t* c_shift = rocke_b_const_i32(b, col_xor[0]);
        r = rocke_b_lshr(b, row, c_shift);
    }
    {
        rocke_value_t* c_mask = rocke_b_const_i32(b, col_xor[1]);
        sel = rocke_b_land(b, r, c_mask);
    }
    {
        rocke_value_t* c_flip = rocke_b_const_i32(b, col_xor[3]);
        rocke_value_t* hi = rocke_b_lshr(b, row, c_flip);
        rocke_value_t* c_one = rocke_b_const_i32(b, 1);
        flip = rocke_b_land(b, hi, c_one);
    }
    sel = rocke_b_xor(b, sel, flip);
    {
        rocke_value_t* c_log2 = rocke_b_const_i32(b, col_xor[2]);
        shifted = rocke_b_shl(b, sel, c_log2);
    }
    return rocke_b_xor(b, col, shifted);
}

rocke_status_t rocke_coalesced_tile_loader_set_col_xor(rocke_coalesced_tile_loader_t* self,
                                                       const int col_xor[4])
{
    int i;
    int chunk;
    int chunks;
    long long group;

    if(self == NULL || col_xor == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* Python __post_init__:
     *   four non-negative ints;
     *   row_shift <= 31, flip_shift <= 31, chunk_log2 <= 30 (i32 shifts);
     *   chunk = 1 << chunk_log2; chunks = tile_cols // chunk
     *   group = 1 << (chunk_mask | 1).bit_length()
     *   tile_cols % chunk == 0, chunks >= group, chunks % group == 0;
     *   vector_axis == "col" -> chunk % load_vec == 0. */
    for(i = 0; i < 4; ++i)
    {
        if(col_xor[i] < 0)
        {
            return ROCKE_ERR_VALUE;
        }
    }
    /* rocke_swizzle_lds_col shifts i32 values by these: 32 or more is poison. */
    if(col_xor[0] > 31 || col_xor[3] > 31 || col_xor[2] > 30)
    {
        return ROCKE_ERR_VALUE;
    }
    /* The chunk index is XORed with values below `group` (chunk_mask and the
     * flip bit), which permutes aligned groups of `group` chunks: each row must
     * hold whole groups, or a store leaves its row. */
    chunk = 1 << col_xor[2];
    chunks = self->tile_cols / chunk;
    group = 2;
    while(group <= (long long)(col_xor[1] | 1))
    {
        group <<= 1;
    }
    if(self->tile_cols % chunk != 0 || chunks < group || chunks % group != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(!self->vector_axis_row && (self->load_vec <= 0 || chunk % self->load_vec != 0))
    {
        return ROCKE_ERR_VALUE;
    }
    self->has_col_xor = true;
    for(i = 0; i < 4; ++i)
    {
        self->col_xor[i] = col_xor[i];
    }
    return ROCKE_OK;
}

rocke_status_t
    rocke_coalesced_tile_loader_vecs_per_thread(const rocke_coalesced_tile_loader_t* self, int* out)
{
    int total_vecs;

    if(self == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* Python:
     *   total_vecs = (tile_rows * tile_cols) // load_vec
     *   if total_vecs % block_size: raise ValueError(...)
     *   return total_vecs // block_size
     */
    total_vecs = (self->tile_rows * self->tile_cols) / self->load_vec;
    if(total_vecs % self->block_size)
    {
        return ROCKE_ERR_VALUE;
    }
    if(out != NULL)
    {
        *out = total_vecs / self->block_size;
    }
    return ROCKE_OK;
}

int rocke_coalesced_tile_loader_cols_per_vec(const rocke_coalesced_tile_loader_t* self)
{
    /* Python: return tile_cols // load_vec */
    if(self == NULL)
    {
        return 0;
    }
    return self->tile_cols / self->load_vec;
}

/* Python CoalescedTileLoader.rows_per_vec property: tile_rows // load_vec. */
static int ctl_rows_per_vec(const rocke_coalesced_tile_loader_t* self)
{
    return self->tile_rows / self->load_vec;
}

/* The ``c_span`` const value: rows_per_vec in row mode, cols_per_vec otherwise
 * (Python: chosen when building c_span in load / load_global). */
static int ctl_span(const rocke_coalesced_tile_loader_t* self)
{
    return self->vector_axis_row ? ctl_rows_per_vec(self)
                                 : rocke_coalesced_tile_loader_cols_per_vec(self);
}

/* Python _ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}: the byte
 * width of a loader element type, 0 for an unsupported one. NULL is the F16
 * default. */
static int loads_elem_bytes(const rocke_type_t* dtype)
{
    if(dtype == NULL || dtype == rocke_f16() || dtype == rocke_bf16())
        return 2;
    if(dtype == rocke_f32() || dtype == rocke_i32())
        return 4;
    return 0;
}

/* True on the historical fp16 path (elem_dtype None/F16), which keeps its
 * f16-specific builder calls; any other element type takes the dtype-generic
 * builder calls the Python loader makes. */
static bool ctl_is_f16(const rocke_coalesced_tile_loader_t* self)
{
    return self->elem_dtype == NULL || self->elem_dtype == rocke_f16();
}

/* Python: raise ValueError(f"CoalescedTileLoader: unsupported elem_dtype ...").
 * Returns the element byte width, or 0 after setting the builder error. */
static int ctl_elem_bytes_or_err(rocke_ir_builder_t* b, const rocke_coalesced_tile_loader_t* self)
{
    int eb = loads_elem_bytes(self->elem_dtype);
    if(eb == 0)
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "CoalescedTileLoader: unsupported elem_dtype '%s'",
                        self->elem_dtype->name);
    return eb;
}

/* One chunk read through the buffer resource: Python
 *   b.buffer_load(rsrc, safe, c0, dtype)            if load_vec == 1
 *   b.buffer_load_vN(rsrc, safe, c0, dtype, load_vec)  otherwise */
static rocke_value_t* ctl_buffer_read(rocke_ir_builder_t* b,
                                      const rocke_coalesced_tile_loader_t* self,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* safe,
                                      rocke_value_t* c0)
{
    if(ctl_is_f16(self))
        return (self->load_vec == 1)
                   ? rocke_b_buffer_load_f16(b, rsrc, safe, c0)
                   : rocke_b_buffer_load_vN_f16(b, rsrc, safe, c0, self->load_vec / 2);
    return (self->load_vec == 1)
               ? rocke_b_buffer_load(b, rsrc, safe, c0, self->elem_dtype)
               : rocke_b_buffer_load_vN(b, rsrc, safe, c0, self->elem_dtype, self->load_vec);
}

/* One chunk read through a raw global pointer: Python
 *   b.global_load(ptr, off_elems, dtype) / b.global_load_vN(ptr, off_elems, dtype, n) */
static rocke_value_t* ctl_global_read(rocke_ir_builder_t* b,
                                      const rocke_coalesced_tile_loader_t* self,
                                      rocke_value_t* ptr,
                                      rocke_value_t* off_elems)
{
    if(ctl_is_f16(self))
        return (self->load_vec == 1)
                   ? rocke_b_global_load_f16(b, ptr, off_elems, 0)
                   : rocke_b_global_load_vN_f16(b, ptr, off_elems, self->load_vec, 0);
    return (self->load_vec == 1)
               ? rocke_b_global_load(b, ptr, off_elems, self->elem_dtype, 0)
               : rocke_b_global_load_vN(b, ptr, off_elems, self->elem_dtype, self->load_vec, 0);
}

/* Python CoalescedTileLoader._decode_row_col: map vec_idx -> (row, col).
 *   idx0  = b.div(vec_idx, c_span)
 *   idx1  = b.mod(vec_idx, c_span)
 *   chunk = b.mul(idx1, c_load_vec) if load_vec > 1 else idx1
 *   row mode: (row, col) = (chunk, idx0);  col mode: (idx0, chunk)
 * The "col" branch reproduces the historical inline decode exactly. */
static void ctl_decode_row_col(rocke_ir_builder_t* b,
                               const rocke_coalesced_tile_loader_t* self,
                               rocke_value_t* vec_idx,
                               rocke_value_t* c_span,
                               rocke_value_t* c_load_vec,
                               rocke_value_t** out_row,
                               rocke_value_t** out_col)
{
    rocke_value_t* idx0 = rocke_b_div(b, vec_idx, c_span);
    rocke_value_t* idx1 = rocke_b_mod(b, vec_idx, c_span);
    rocke_value_t* chunk = (self->load_vec > 1) ? rocke_b_mul(b, idx1, c_load_vec) : idx1;
    if(self->vector_axis_row)
    {
        *out_row = chunk;
        *out_col = idx0;
    }
    else
    {
        *out_row = idx0;
        *out_col = chunk;
    }
}

/* Python CoalescedTileLoader._store_tile: write a loaded chunk to LDS.
 *   row mode + load_vec>1: scatter load_vec f16 elements to [row+i, col].
 *   otherwise: one smem_store (f16 scalar for load_vec==1, else vN).
 * With col_xor the LDS column goes through rocke_swizzle_lds_col first (per
 * scattered element in row mode). */
static void ctl_store_tile(rocke_ir_builder_t* b,
                           const rocke_coalesced_tile_loader_t* self,
                           rocke_value_t* smem_dst,
                           rocke_value_t* row,
                           rocke_value_t* col,
                           rocke_value_t* v)
{
    if(self->vector_axis_row && self->load_vec > 1)
    {
        int i;
        for(i = 0; i < self->load_vec; ++i)
        {
            rocke_value_t* r = (i != 0) ? rocke_b_add(b, row, rocke_b_const_i32(b, i)) : row;
            /* col_xor: each scattered element takes its own row's XOR. */
            rocke_value_t* c
                = self->has_col_xor ? rocke_swizzle_lds_col(b, r, col, self->col_xor) : col;
            rocke_value_t* el = rocke_b_vec_extract(b, v, i);
            rocke_value_t* indices[2];
            indices[0] = r;
            indices[1] = c;
            if(ctl_is_f16(self))
                rocke_b_smem_store_f16(b, smem_dst, indices, 2, el);
            else
                rocke_b_smem_store_vN(b, smem_dst, indices, 2, el, 1);
        }
    }
    else
    {
        rocke_value_t* indices[2];
        if(self->has_col_xor)
        {
            col = rocke_swizzle_lds_col(b, row, col, self->col_xor);
        }
        indices[0] = row;
        indices[1] = col;
        if(!ctl_is_f16(self))
        {
            /* b.smem_store_vN(smem_dst, [row, col], v, load_vec) */
            rocke_b_smem_store_vN(b, smem_dst, indices, 2, v, self->load_vec);
        }
        else if(self->load_vec == 1)
        {
            rocke_b_smem_store_f16(b, smem_dst, indices, 2, v);
        }
        else
        {
            rocke_b_smem_store_vN_f16(b, smem_dst, indices, 2, v, self->load_vec);
        }
    }
}

void rocke_coalesced_tile_loader_load(rocke_ir_builder_t* b,
                                      const rocke_coalesced_tile_loader_t* self,
                                      rocke_value_t* tid,
                                      rocke_value_t* smem_dst,
                                      rocke_loads_descriptor_fn descriptor,
                                      void* descriptor_user,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* ptr)
{
    rocke_value_t* c_threads;
    rocke_value_t* c_load_vec;
    rocke_value_t* c_span;
    rocke_value_t* c_half_bytes;
    rocke_value_t* c0;
    rocke_value_t* c_oob;
    int vecs_per_thread;
    int e;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader.load: NULL loader");
        return;
    }

    /* Python:
     *   if self.use_buffer_rsrc and rsrc is None:
     *       raise ValueError("... use_buffer_rsrc=True requires rsrc")
     *   if not self.use_buffer_rsrc and ptr is None:
     *       raise ValueError("... use_buffer_rsrc=False requires ptr")
     */
    if(self->use_buffer_rsrc && rsrc == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=True requires rsrc");
        return;
    }
    if(!self->use_buffer_rsrc && ptr == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=False requires ptr");
        return;
    }

    /* Python:
     *   elem_bytes     = _ELEM_BYTES.get(self.elem_dtype.name)  (ValueError if None)
     *   c_threads      = b.const_i32(self.block_size)
     *   c_load_vec     = b.const_i32(self.load_vec)
     *   c_span         = b.const_i32(rows_per_vec if row else cols_per_vec)
     *   c_elem_bytes   = b.const_i32(elem_bytes)
     *   c0             = b.const_i32(0)
     *   c_oob          = b.const_i32(self.oob_sentinel)
     */
    const int elem_bytes = ctl_elem_bytes_or_err(b, self);
    if(elem_bytes == 0)
        return;
    c_threads = rocke_b_const_i32(b, self->block_size);
    c_load_vec = rocke_b_const_i32(b, self->load_vec);
    c_span = rocke_b_const_i32(b, ctl_span(self));
    c_half_bytes = rocke_b_const_i32(b, elem_bytes);
    c0 = rocke_b_const_i32(b, 0);
    c_oob = rocke_b_const_i32(b, self->oob_sentinel);

    /* self.vecs_per_thread (Python property; raises ValueError on bad divide). */
    if(rocke_coalesced_tile_loader_vecs_per_thread(self, &vecs_per_thread) != ROCKE_OK)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "CoalescedTileLoader: tile %dx%d / %d not divisible by "
                        "block_size %d",
                        self->tile_rows,
                        self->tile_cols,
                        self->load_vec,
                        self->block_size);
        return;
    }

    for(e = 0; e < vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx;
        rocke_value_t* row;
        rocke_value_t* col;
        rocke_value_t* off_elems;
        rocke_value_t* valid;

        /* Python: vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
         *         row, col = self._decode_row_col(b, vec_idx, ...) */
        vec_idx = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
        ctl_decode_row_col(b, self, vec_idx, c_span, c_load_vec, &row, &col);

        /* Python: off_elems, valid = descriptor(b, row, col) */
        valid = NULL; /* default => "None" if callback leaves it NULL */
        off_elems = descriptor(b, row, col, &valid, descriptor_user);

        if(self->use_buffer_rsrc)
        {
            rocke_value_t* off_bytes;
            rocke_value_t* safe;

            /* Python:
             *   off_bytes = b.mul(off_elems, c_half_bytes)
             *   safe = b.select(valid, off_bytes, c_oob) if valid is not None
             *          else off_bytes
             */
            off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
            if(valid != NULL)
            {
                safe = rocke_b_select(b, valid, off_bytes, c_oob);
            }
            else
            {
                safe = off_bytes;
            }

            /* Python:
             *   v = b.buffer_load(rsrc, safe, c0, dtype)        (load_vec == 1)
             *   v = b.buffer_load_vN(rsrc, safe, c0, dtype, n)  (otherwise)
             *   self._store_tile(b, smem_dst, row, col, v)      # vN or row-scatter
             */
            rocke_value_t* v = ctl_buffer_read(b, self, rsrc, safe, c0);
            ctl_store_tile(b, self, smem_dst, row, col, v);
        }
        else
        {
            /* Python:
             *   v = b.global_load(ptr, off_elems, dtype)        (load_vec == 1)
             *   v = b.global_load_vN(ptr, off_elems, dtype, n)  (otherwise)
             *   self._store_tile(b, smem_dst, row, col, v)      # vN or row-scatter
             */
            rocke_value_t* v = ctl_global_read(b, self, ptr, off_elems);
            ctl_store_tile(b, self, smem_dst, row, col, v);
        }
    }
}

/* ============================================================================
 * CoalescedTileLoader split-load helpers (load_global / store_lds)
 *
 * Mirror of Python load_global() / store_lds() added for CK pipeline_basic.
 * load_global emits only buffer_load_vN; store_lds emits only smem_store_vN.
 * ========================================================================== */

void rocke_coalesced_tile_loader_load_global(rocke_ir_builder_t* b,
                                             const rocke_coalesced_tile_loader_t* self,
                                             rocke_value_t* tid,
                                             rocke_loads_descriptor_fn descriptor,
                                             void* descriptor_user,
                                             rocke_value_t* rsrc,
                                             rocke_value_t* ptr,
                                             rocke_ctl_staged_t* staged)
{
    rocke_value_t* c_threads;
    rocke_value_t* c_load_vec;
    rocke_value_t* c_span;
    rocke_value_t* c_half_bytes;
    rocke_value_t* c0;
    rocke_value_t* c_oob;
    int vecs_per_thread;
    int e;

    if(staged != NULL)
        staged->count = 0;

    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || staged == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_global: NULL loader or staged");
        return;
    }
    if(self->use_buffer_rsrc && rsrc == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=True requires rsrc");
        return;
    }
    if(!self->use_buffer_rsrc && ptr == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=False requires ptr");
        return;
    }

    /* Same constants as load() — byte-identical SSA prefix. */
    const int elem_bytes = ctl_elem_bytes_or_err(b, self);
    if(elem_bytes == 0)
        return;
    c_threads = rocke_b_const_i32(b, self->block_size);
    c_load_vec = rocke_b_const_i32(b, self->load_vec);
    c_span = rocke_b_const_i32(b, ctl_span(self));
    c_half_bytes = rocke_b_const_i32(b, elem_bytes);
    c0 = rocke_b_const_i32(b, 0);
    c_oob = rocke_b_const_i32(b, self->oob_sentinel);

    if(rocke_coalesced_tile_loader_vecs_per_thread(self, &vecs_per_thread) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader: bad vecs_per_thread");
        return;
    }
    if(vecs_per_thread > ROCKE_CTL_MAX_VECS_PER_THREAD)
    {
        rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "CoalescedTileLoader: vecs_per_thread %d > ROCKE_CTL_MAX_VECS_PER_THREAD %d",
            vecs_per_thread,
            ROCKE_CTL_MAX_VECS_PER_THREAD);
        return;
    }

    for(e = 0; e < vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx;
        rocke_value_t* row;
        rocke_value_t* col;
        rocke_value_t* off_elems;
        rocke_value_t* valid;
        rocke_value_t* v;

        vec_idx = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
        ctl_decode_row_col(b, self, vec_idx, c_span, c_load_vec, &row, &col);

        valid = NULL;
        off_elems = descriptor(b, row, col, &valid, descriptor_user);

        if(self->use_buffer_rsrc)
        {
            rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
            rocke_value_t* safe
                = (valid != NULL) ? rocke_b_select(b, valid, off_bytes, c_oob) : off_bytes;
            v = ctl_buffer_read(b, self, rsrc, safe, c0);
        }
        else
        {
            v = ctl_global_read(b, self, ptr, off_elems);
        }

        staged->vecs[e].row = row;
        staged->vecs[e].col = col;
        staged->vecs[e].v = v;
    }
    staged->count = vecs_per_thread;
}

void rocke_coalesced_tile_loader_store_lds(rocke_ir_builder_t* b,
                                           const rocke_coalesced_tile_loader_t* self,
                                           rocke_value_t* smem_dst,
                                           const rocke_ctl_staged_t* staged)
{
    int e;
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || staged == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "store_lds: NULL loader or staged");
        return;
    }
    for(e = 0; e < staged->count; ++e)
    {
        ctl_store_tile(
            b, self, smem_dst, staged->vecs[e].row, staged->vecs[e].col, staged->vecs[e].v);
    }
}

/* ============================================================================
 * CoalescedTileLoader address-prefetch split (plan_slots / slot_offsets /
 * issue_slots). Same per-chunk ops as load_global, regrouped so a K loop can
 * carry the byte offsets across iterations. Mirrors the Python methods.
 * ========================================================================== */

void rocke_coalesced_tile_loader_plan_slots(rocke_ir_builder_t* b,
                                            const rocke_coalesced_tile_loader_t* self,
                                            rocke_value_t* tid,
                                            rocke_ctl_staged_t* slots)
{
    if(slots != NULL)
        slots->count = 0;
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || slots == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "plan_slots: NULL loader or slots");
        return;
    }

    /* Python:
     *   c_threads  = b.const_i32(self.block_size)
     *   c_load_vec = b.const_i32(self.load_vec)
     *   c_span     = b.const_i32(rows_per_vec if row else cols_per_vec) */
    rocke_value_t* c_threads = rocke_b_const_i32(b, self->block_size);
    rocke_value_t* c_load_vec = rocke_b_const_i32(b, self->load_vec);
    rocke_value_t* c_span = rocke_b_const_i32(b, ctl_span(self));

    int vecs_per_thread = 0;
    if(rocke_coalesced_tile_loader_vecs_per_thread(self, &vecs_per_thread) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader: bad vecs_per_thread");
        return;
    }
    if(vecs_per_thread > ROCKE_CTL_MAX_VECS_PER_THREAD)
    {
        rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "CoalescedTileLoader: vecs_per_thread %d > ROCKE_CTL_MAX_VECS_PER_THREAD %d",
            vecs_per_thread,
            ROCKE_CTL_MAX_VECS_PER_THREAD);
        return;
    }

    for(int e = 0; e < vecs_per_thread; ++e)
    {
        /* Python: vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid) */
        rocke_value_t* c_e = rocke_b_const_i32(b, e);
        rocke_value_t* scaled = rocke_b_mul(b, c_e, c_threads);
        rocke_value_t* vec_idx = rocke_b_add(b, scaled, tid);
        ctl_decode_row_col(
            b, self, vec_idx, c_span, c_load_vec, &slots->vecs[e].row, &slots->vecs[e].col);
        slots->vecs[e].v = NULL;
    }
    slots->count = vecs_per_thread;
}

void rocke_coalesced_tile_loader_slot_offsets(rocke_ir_builder_t* b,
                                              const rocke_coalesced_tile_loader_t* self,
                                              const rocke_ctl_staged_t* slots,
                                              rocke_loads_descriptor_fn descriptor,
                                              void* descriptor_user,
                                              rocke_value_t** out_offsets)
{
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || slots == NULL || descriptor == NULL || out_offsets == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "slot_offsets: NULL argument");
        return;
    }
    if(!self->use_buffer_rsrc)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader.slot_offsets requires use_buffer_rsrc=True");
        return;
    }

    /* Python:
     *   elem_bytes   = _ELEM_BYTES.get(self.elem_dtype.name)  (ValueError if None)
     *   c_elem_bytes = b.const_i32(elem_bytes)
     *   c_oob        = b.const_i32(self.oob_sentinel) */
    const int elem_bytes = ctl_elem_bytes_or_err(b, self);
    if(elem_bytes == 0)
        return;
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, elem_bytes);
    rocke_value_t* c_oob = rocke_b_const_i32(b, self->oob_sentinel);

    for(int e = 0; e < slots->count; ++e)
    {
        /* Python: off_elems, valid = descriptor(b, row, col) */
        rocke_value_t* valid = NULL;
        rocke_value_t* off_elems
            = descriptor(b, slots->vecs[e].row, slots->vecs[e].col, &valid, descriptor_user);
        rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
        out_offsets[e] = (valid != NULL) ? rocke_b_select(b, valid, off_bytes, c_oob) : off_bytes;
    }
}

void rocke_coalesced_tile_loader_issue_slots(rocke_ir_builder_t* b,
                                             const rocke_coalesced_tile_loader_t* self,
                                             rocke_value_t* rsrc,
                                             const rocke_ctl_staged_t* slots,
                                             rocke_value_t* const* offsets,
                                             rocke_ctl_staged_t* staged)
{
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || slots == NULL || offsets == NULL || staged == NULL || rsrc == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "issue_slots: NULL argument");
        return;
    }

    /* Python: c0 = b.const_i32(0) */
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    int count = slots->count;
    for(int e = 0; e < count; ++e)
    {
        rocke_value_t* v = ctl_buffer_read(b, self, rsrc, offsets[e], c0);
        staged->vecs[e].row = slots->vecs[e].row;
        staged->vecs[e].col = slots->vecs[e].col;
        staged->vecs[e].v = v;
    }
    staged->count = count;
}

void rocke_coalesced_tile_loader_load_async_to_lds(rocke_ir_builder_t* b,
                                                   const rocke_coalesced_tile_loader_t* self,
                                                   rocke_value_t* tid,
                                                   rocke_value_t* smem_dst,
                                                   rocke_loads_descriptor_fn descriptor,
                                                   void* descriptor_user,
                                                   rocke_value_t* ptr,
                                                   rocke_value_t* num_bytes,
                                                   int coherency)
{
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || descriptor == NULL || ptr == NULL || smem_dst == NULL || num_bytes == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "load_async_to_lds: NULL argument");
        return;
    }
    if(self->vector_axis_row)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "CoalescedTileLoader.load_async_to_lds supports only vector_axis='col' "
                        "(the copy cannot transpose on store)");
        return;
    }
    if(self->has_col_xor)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader.load_async_to_lds does not support col_xor");
        return;
    }
    /* This port loads 16-bit elements (see slot_offsets). */
    const int width = self->load_vec * 2;
    if(width != 4 && width != 8 && width != 16)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "global_load_async_to_lds moves 4, 8 or 16 bytes per lane; a "
                        "%d x 2 B chunk is %d B",
                        self->load_vec,
                        width);
        return;
    }

    /* Python: zero = b.zero_vec(self.elem_dtype, self.load_vec). This port
     * takes the zero's type from the destination LDS tile (f16 or bf16: the
     * 16-bit elements this port loads). */
    const rocke_type_t* lds_elem
        = (smem_dst->type != NULL && smem_dst->type->kind == ROCKE_TYPE_SMEM) ? smem_dst->type->elem
                                                                              : NULL;
    if(lds_elem == NULL
       || (strcmp(lds_elem->name, "f16") != 0 && strcmp(lds_elem->name, "bf16") != 0))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "load_async_to_lds: the LDS tile must hold f16 or bf16 (got %s)",
                        lds_elem != NULL ? lds_elem->name : "a non-LDS value");
        return;
    }
    rocke_value_t* zero = rocke_b_zero_vec(b, lds_elem, self->load_vec);
    rocke_value_t* c_elem_bytes = rocke_b_const_i32(b, 2);
    rocke_ctl_staged_t slots;
    rocke_coalesced_tile_loader_plan_slots(b, self, tid, &slots);
    for(int e = 0; e < slots.count; ++e)
    {
        rocke_value_t* idx[2];
        idx[0] = slots.vecs[e].row;
        idx[1] = slots.vecs[e].col;
        /* Python: off_elems, valid = descriptor(b, row, col) */
        rocke_value_t* valid = NULL;
        rocke_value_t* off_elems = descriptor(b, idx[0], idx[1], &valid, descriptor_user);
        /* Python: in_bounds = b.cmp_ult(b.mul(off_elems, c_elem_bytes), num_bytes)
         * -- unsigned, as the buffer resource's range check: a byte offset
         * at or past 2 GiB (negative as i32) is out of range, not below. */
        rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c_elem_bytes);
        rocke_value_t* in_bounds = rocke_b_cmp_ult(b, off_bytes, num_bytes);
        valid = (valid == NULL) ? in_bounds : rocke_b_land(b, valid, in_bounds);
        rocke_if_else_t ie = rocke_b_scf_if_else(b, valid);
        rocke_b_region_enter(b, ie.then_region);
        rocke_b_global_load_async_to_lds(b, ptr, off_elems, smem_dst, idx, 2, width, coherency, 0);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        rocke_b_smem_store_vN_f16(b, smem_dst, idx, 2, zero, self->load_vec);
        rocke_b_region_leave(b);
    }
}

/* ============================================================================
 * RegTransposeTileLoader
 *
 * Mirror of Python RegTransposeTileLoader. Every builder call below is its
 * own statement in Python's left-to-right evaluation order: C leaves the
 * order of call arguments unspecified, and each call takes the next SSA id.
 * ========================================================================== */

static const int REGT_VECS[4] = {8, 4, 2, 1};

static void regt_why(char* why, size_t why_cap, const char* fmt, ...)
{
    if(why == NULL || why_cap == 0)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, why_cap, fmt, ap);
    va_end(ap);
}

rocke_status_t rocke_reg_transpose_tile_loader_plan(int tile_rows,
                                                    int tile_cols,
                                                    int block_size,
                                                    int wave_size,
                                                    int vec,
                                                    int* out_units,
                                                    int* out_active,
                                                    char* why,
                                                    size_t why_cap)
{
    bool known = false;
    for(int i = 0; i < 4; ++i)
        known = known || REGT_VECS[i] == vec;
    if(!known)
    {
        regt_why(why, why_cap, "vec %d not in (8, 4, 2, 1)", vec);
        return ROCKE_ERR_VALUE;
    }
    if(tile_cols % ROCKE_REGT_K_RUN)
    {
        regt_why(
            why, why_cap, "tile K extent %d is not a multiple of %d", tile_cols, ROCKE_REGT_K_RUN);
        return ROCKE_ERR_VALUE;
    }
    if(tile_rows % vec)
    {
        regt_why(why, why_cap, "tile free extent %d not divisible by vec %d", tile_rows, vec);
        return ROCKE_ERR_VALUE;
    }
    const int units = (tile_rows / vec) * (tile_cols / ROCKE_REGT_K_RUN);
    int active = 0;
    if(block_size > 0 && units % block_size == 0)
        active = block_size;
    else if(units < block_size && wave_size > 0 && units % wave_size == 0)
        active = units;
    else
    {
        regt_why(why,
                 why_cap,
                 "%d micro-tiles (%dx%d at %dx%d) neither divide over %d threads nor fill whole "
                 "waves of %d",
                 units,
                 tile_rows,
                 tile_cols,
                 vec,
                 ROCKE_REGT_K_RUN,
                 block_size,
                 wave_size);
        return ROCKE_ERR_VALUE;
    }
    if(out_units != NULL)
        *out_units = units;
    if(out_active != NULL)
        *out_active = active;
    if(why != NULL && why_cap != 0)
        why[0] = '\0';
    return ROCKE_OK;
}

rocke_status_t rocke_reg_transpose_tile_loader_choose_vec(
    int tile_rows, int tile_cols, int block_size, int wave_size, int max_vec, int* out_vec)
{
    for(int i = 0; i < 4; ++i)
    {
        const int v = REGT_VECS[i];
        if(v > max_vec)
            continue;
        if(rocke_reg_transpose_tile_loader_plan(
               tile_rows, tile_cols, block_size, wave_size, v, NULL, NULL, NULL, 0)
           == ROCKE_OK)
        {
            if(out_vec != NULL)
                *out_vec = v;
            return ROCKE_OK;
        }
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_reg_transpose_tile_loader_init(rocke_reg_transpose_tile_loader_t* self,
                                                    int tile_rows,
                                                    int tile_cols,
                                                    int block_size,
                                                    int wave_size,
                                                    int vec,
                                                    const rocke_type_t* elem_dtype,
                                                    char* why,
                                                    size_t why_cap)
{
    if(self == NULL)
        return ROCKE_ERR_VALUE;
    memset(self, 0, sizeof(*self));
    self->tile_rows = tile_rows;
    self->tile_cols = tile_cols;
    self->block_size = block_size;
    self->wave_size = wave_size;
    self->vec = vec;
    self->elem_dtype = elem_dtype;
    self->oob_sentinel = 2147483647;
    if(loads_elem_bytes(elem_dtype) != 2)
    {
        regt_why(why,
                 why_cap,
                 "RegTransposeTileLoader: 16-bit element types only (one 16-byte LDS store is "
                 "%d elements); got '%s'",
                 ROCKE_REGT_K_RUN,
                 elem_dtype != NULL ? elem_dtype->name : "f16");
        return ROCKE_ERR_VALUE;
    }
    char plan_why[256];
    if(rocke_reg_transpose_tile_loader_plan(
           tile_rows, tile_cols, block_size, wave_size, vec, NULL, NULL, plan_why, sizeof plan_why)
       != ROCKE_OK)
    {
        regt_why(why, why_cap, "RegTransposeTileLoader: %s", plan_why);
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* The plan of a loader that passed init (its plan is usable). */
static int regt_active_threads(const rocke_reg_transpose_tile_loader_t* self)
{
    int active = 0;
    rocke_reg_transpose_tile_loader_plan(self->tile_rows,
                                         self->tile_cols,
                                         self->block_size,
                                         self->wave_size,
                                         self->vec,
                                         NULL,
                                         &active,
                                         NULL,
                                         0);
    return active;
}

static int regt_micro_tiles(const rocke_reg_transpose_tile_loader_t* self)
{
    return (self->tile_rows / self->vec) * (self->tile_cols / ROCKE_REGT_K_RUN);
}

bool rocke_reg_transpose_tile_loader_partial(const rocke_reg_transpose_tile_loader_t* self)
{
    return self != NULL && regt_active_threads(self) < self->block_size;
}

bool rocke_reg_transpose_tile_loader_guard_applies(const rocke_reg_transpose_tile_loader_t* self)
{
    return rocke_reg_transpose_tile_loader_partial(self) && self->vec > 1;
}

static const rocke_type_t* regt_elem(const rocke_reg_transpose_tile_loader_t* self)
{
    return self->elem_dtype != NULL ? self->elem_dtype : rocke_f16();
}

/* Python _active: the wave-uniform "this wave carries a micro-tile" predicate. */
static rocke_value_t* regt_active(rocke_ir_builder_t* b,
                                  const rocke_reg_transpose_tile_loader_t* self,
                                  rocke_value_t* tid)
{
    rocke_value_t* c_wave = rocke_b_const_i32(b, self->wave_size);
    rocke_value_t* q = rocke_b_div(b, tid, c_wave);
    rocke_value_t* wave = rocke_b_to_sgpr_u32(b, q);
    rocke_value_t* c_waves = rocke_b_const_i32(b, regt_active_threads(self) / self->wave_size);
    return rocke_b_cmp_lt(b, wave, c_waves);
}

/* The per-unit (row, col) decode both read paths share. */
static void regt_unit_row_col(rocke_ir_builder_t* b,
                              const rocke_reg_transpose_tile_loader_t* self,
                              rocke_value_t* tid,
                              int e,
                              bool clamp,
                              rocke_value_t* c_kc,
                              rocke_value_t* c_vec,
                              rocke_value_t* c_k_run,
                              rocke_value_t** out_row,
                              rocke_value_t** out_col)
{
    const int mt = regt_micro_tiles(self);
    rocke_value_t* c_e = rocke_b_const_i32(b, e);
    rocke_value_t* c_at = rocke_b_const_i32(b, regt_active_threads(self));
    rocke_value_t* m = rocke_b_mul(b, c_e, c_at);
    rocke_value_t* u = rocke_b_add(b, m, tid);
    if(clamp)
    {
        /* Idle waves decode a clamped unit so every address stays in the tile. */
        rocke_value_t* c_mt = rocke_b_const_i32(b, mt);
        rocke_value_t* in_range = rocke_b_cmp_lt(b, u, c_mt);
        rocke_value_t* c_last = rocke_b_const_i32(b, mt - 1);
        u = rocke_b_select(b, in_range, u, c_last);
    }
    rocke_value_t* q = rocke_b_div(b, u, c_kc);
    *out_row = rocke_b_mul(b, q, c_vec);
    rocke_value_t* r = rocke_b_mod(b, u, c_kc);
    *out_col = rocke_b_mul(b, r, c_k_run);
}

/* (offset, valid) of K column j of a unit: the run descriptor's, or the
 * per-column descriptor at col + j. */
static rocke_value_t* regt_column(rocke_ir_builder_t* b,
                                  rocke_value_t* row,
                                  rocke_value_t* col,
                                  int j,
                                  rocke_value_t* const* run_off,
                                  rocke_value_t* const* run_valid,
                                  rocke_loads_descriptor_fn descriptor,
                                  void* descriptor_user,
                                  rocke_value_t** out_valid)
{
    if(run_off != NULL)
    {
        *out_valid = run_valid[j];
        return run_off[j];
    }
    rocke_value_t* cj = col;
    if(j != 0)
    {
        rocke_value_t* c_j = rocke_b_const_i32(b, j);
        cj = rocke_b_add(b, col, c_j);
    }
    *out_valid = NULL;
    return descriptor(b, row, cj, out_valid, descriptor_user);
}

static bool regt_check_units(rocke_ir_builder_t* b, int n)
{
    if(n > ROCKE_REGT_MAX_UNITS)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "RegTransposeTileLoader: %d micro-tiles per thread > "
                        "ROCKE_REGT_MAX_UNITS %d",
                        n,
                        ROCKE_REGT_MAX_UNITS);
        return false;
    }
    return true;
}

/* Python _load_global_guarded: the addresses in the enclosing block, only the
 * reads in an if/else on `active` (idle waves yield undef). */
static void regt_load_global_guarded(rocke_ir_builder_t* b,
                                     const rocke_reg_transpose_tile_loader_t* self,
                                     rocke_value_t* tid,
                                     rocke_loads_descriptor_fn descriptor,
                                     void* descriptor_user,
                                     rocke_value_t* rsrc,
                                     rocke_value_t* active,
                                     rocke_loads_run_descriptor_fn run_fn,
                                     void* run_user,
                                     rocke_regt_staged_t* staged)
{
    const rocke_type_t* et = regt_elem(self);
    const int kc = self->tile_cols / ROCKE_REGT_K_RUN;
    rocke_value_t* c_kc = rocke_b_const_i32(b, kc);
    rocke_value_t* c_vec = rocke_b_const_i32(b, self->vec);
    rocke_value_t* c_k_run = rocke_b_const_i32(b, ROCKE_REGT_K_RUN);
    rocke_value_t* c_elem_bytes = rocke_b_const_i32(b, loads_elem_bytes(self->elem_dtype));
    rocke_value_t* c_oob = rocke_b_const_i32(b, self->oob_sentinel);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);

    const int upt = regt_micro_tiles(self) / regt_active_threads(self);
    if(!regt_check_units(b, upt))
        return;
    /* offs[unit * 8 + j] */
    rocke_value_t** offs = (rocke_value_t**)rocke_arena_alloc(
        &b->arena, sizeof(rocke_value_t*) * (size_t)upt * ROCKE_REGT_K_RUN);
    if(offs == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader: arena alloc failed");
        return;
    }
    for(int e = 0; e < upt; ++e)
    {
        rocke_value_t* row;
        rocke_value_t* col;
        regt_unit_row_col(b, self, tid, e, true, c_kc, c_vec, c_k_run, &row, &col);
        rocke_value_t* run_off[ROCKE_REGT_K_RUN];
        rocke_value_t* run_valid[ROCKE_REGT_K_RUN];
        if(run_fn != NULL)
            run_fn(b, row, col, run_off, run_valid, run_user);
        for(int j = 0; j < ROCKE_REGT_K_RUN; ++j)
        {
            rocke_value_t* valid;
            rocke_value_t* off_elems = regt_column(b,
                                                   row,
                                                   col,
                                                   j,
                                                   run_fn != NULL ? run_off : NULL,
                                                   run_valid,
                                                   descriptor,
                                                   descriptor_user,
                                                   &valid);
            rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c_elem_bytes);
            offs[e * ROCKE_REGT_K_RUN + j]
                = valid == NULL ? off_bytes : rocke_b_select(b, valid, off_bytes, c_oob);
        }
        staged->units[e].row = row;
        staged->units[e].col = col;
    }
    const int n = upt * ROCKE_REGT_K_RUN;
    const rocke_type_t* vt = rocke_vector_type(b, et, self->vec);
    const rocke_type_t** types
        = (const rocke_type_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_type_t*) * (size_t)n);
    rocke_value_t** vals
        = (rocke_value_t**)rocke_arena_alloc(&b->arena, sizeof(rocke_value_t*) * (size_t)n);
    if(types == NULL || vals == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader: arena alloc failed");
        return;
    }
    for(int i = 0; i < n; ++i)
        types[i] = vt;
    rocke_if_else_t branch = rocke_b_scf_if_else_results(b, active, types, n);
    if(!rocke_ir_builder_ok(b))
        return;
    rocke_b_region_enter(b, branch.then_region);
    for(int i = 0; i < n; ++i)
        vals[i] = rocke_b_buffer_load_vN(b, rsrc, offs[i], c0, et, self->vec);
    rocke_b_scf_yield(b, vals, n);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, branch.else_region);
    rocke_value_t* idle = rocke_b_undef_vec(b, et, self->vec);
    for(int i = 0; i < n; ++i)
        vals[i] = idle;
    rocke_b_scf_yield(b, vals, n);
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return;
    for(int e = 0; e < upt; ++e)
        for(int j = 0; j < ROCKE_REGT_K_RUN; ++j)
            staged->units[e].vs[j] = branch.op->results[e * ROCKE_REGT_K_RUN + j];
    staged->count = upt;
    staged->active = active;
}

void rocke_reg_transpose_tile_loader_load_global(rocke_ir_builder_t* b,
                                                 const rocke_reg_transpose_tile_loader_t* self,
                                                 rocke_value_t* tid,
                                                 rocke_loads_descriptor_fn descriptor,
                                                 void* descriptor_user,
                                                 rocke_value_t* rsrc,
                                                 bool masked,
                                                 rocke_loads_run_descriptor_fn run_fn,
                                                 void* run_user,
                                                 bool guard,
                                                 rocke_regt_staged_t* staged)
{
    if(staged != NULL)
    {
        staged->count = 0;
        staged->active = NULL;
    }
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || staged == NULL || rsrc == NULL || (descriptor == NULL && run_fn == NULL))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader.load_global: NULL argument");
        return;
    }
    const rocke_type_t* et = regt_elem(self);
    const int kc = self->tile_cols / ROCKE_REGT_K_RUN;
    rocke_value_t* c_kc = rocke_b_const_i32(b, kc);
    rocke_value_t* c_vec = rocke_b_const_i32(b, self->vec);
    rocke_value_t* c_k_run = rocke_b_const_i32(b, ROCKE_REGT_K_RUN);
    rocke_value_t* c_elem_bytes = rocke_b_const_i32(b, loads_elem_bytes(self->elem_dtype));
    rocke_value_t* c_oob = rocke_b_const_i32(b, self->oob_sentinel);
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    const bool partial = rocke_reg_transpose_tile_loader_partial(self);
    if(guard && !(rocke_reg_transpose_tile_loader_guard_applies(self) && masked))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "RegTransposeTileLoader.load_global: guard needs partial "
                        "participation, masked reads and vec > 1");
        return;
    }
    rocke_value_t* active = NULL;
    if(partial && masked)
        active = regt_active(b, self, tid);
    if(active != NULL && guard)
    {
        regt_load_global_guarded(
            b, self, tid, descriptor, descriptor_user, rsrc, active, run_fn, run_user, staged);
        return;
    }
    staged->active = active;
    const int upt = regt_micro_tiles(self) / regt_active_threads(self);
    if(!regt_check_units(b, upt))
        return;
    for(int e = 0; e < upt; ++e)
    {
        rocke_value_t* row;
        rocke_value_t* col;
        regt_unit_row_col(b, self, tid, e, partial, c_kc, c_vec, c_k_run, &row, &col);
        rocke_value_t* run_off[ROCKE_REGT_K_RUN];
        rocke_value_t* run_valid[ROCKE_REGT_K_RUN];
        if(run_fn != NULL)
            run_fn(b, row, col, run_off, run_valid, run_user);
        for(int j = 0; j < ROCKE_REGT_K_RUN; ++j)
        {
            rocke_value_t* valid;
            rocke_value_t* off_elems = regt_column(b,
                                                   row,
                                                   col,
                                                   j,
                                                   run_fn != NULL ? run_off : NULL,
                                                   run_valid,
                                                   descriptor,
                                                   descriptor_user,
                                                   &valid);
            rocke_value_t* off_bytes = rocke_b_mul(b, off_elems, c_elem_bytes);
            if(active != NULL)
                valid = valid == NULL ? active : rocke_b_land(b, valid, active);
            rocke_value_t* safe
                = valid == NULL ? off_bytes : rocke_b_select(b, valid, off_bytes, c_oob);
            staged->units[e].vs[j] = self->vec == 1
                                         ? rocke_b_buffer_load(b, rsrc, safe, c0, et)
                                         : rocke_b_buffer_load_vN(b, rsrc, safe, c0, et, self->vec);
        }
        staged->units[e].row = row;
        staged->units[e].col = col;
    }
    staged->count = upt;
}

/* LDS column of a 16-byte unit store (Python: col if swz is None else swz(...)). */
static rocke_value_t* regt_store_col(rocke_ir_builder_t* b,
                                     const rocke_reg_transpose_tile_loader_t* self,
                                     rocke_value_t* row,
                                     rocke_value_t* col)
{
    if(self->col_swizzle == NULL)
        return col;
    return self->col_swizzle(b, row, col, self->col_swizzle_user);
}

/* Python _store_unit: regroup one unit's 8 x vec registers and write vec rows. */
static void regt_store_unit(rocke_ir_builder_t* b,
                            const rocke_reg_transpose_tile_loader_t* self,
                            rocke_value_t* smem_dst,
                            const rocke_regt_staged_unit_t* u)
{
    const rocke_type_t* et = regt_elem(self);
    rocke_value_t* idx[2];
    if(self->vec == 1)
    {
        rocke_value_t* packed = rocke_b_vec_pack(b, u->vs, ROCKE_REGT_K_RUN, et);
        idx[0] = u->row;
        idx[1] = regt_store_col(b, self, u->row, u->col);
        rocke_b_smem_store_vN(b, smem_dst, idx, 2, packed, ROCKE_REGT_K_RUN);
        return;
    }
    const int n_dw = self->vec / 2;
    rocke_value_t* words[ROCKE_REGT_K_RUN][4];
    for(int j = 0; j < ROCKE_REGT_K_RUN; ++j)
    {
        if(n_dw == 1)
        {
            words[j][0] = rocke_b_vec_bitcast(b, u->vs[j], rocke_i32());
            continue;
        }
        rocke_value_t* w
            = rocke_b_vec_bitcast(b, u->vs[j], rocke_vector_type(b, rocke_i32(), n_dw));
        for(int d = 0; d < n_dw; ++d)
            words[j][d] = rocke_b_vec_extract(b, w, d);
    }
    const rocke_type_t* out_t = rocke_vector_type(b, et, ROCKE_REGT_K_RUN);
    for(int i = 0; i < self->vec; ++i)
    {
        const int d = i / 2;
        const int64_t sel = (i % 2) ? 0x07060302 : 0x05040100;
        rocke_value_t* dwords[ROCKE_REGT_K_RUN / 2];
        for(int p = 0; p < ROCKE_REGT_K_RUN / 2; ++p)
            dwords[p] = rocke_b_byte_perm(b, words[2 * p + 1][d], words[2 * p][d], sel);
        rocke_value_t* pk = rocke_b_vec_pack(b, dwords, ROCKE_REGT_K_RUN / 2, rocke_i32());
        rocke_value_t* packed = rocke_b_vec_bitcast(b, pk, out_t);
        rocke_value_t* r = u->row;
        if(i != 0)
        {
            rocke_value_t* c_i = rocke_b_const_i32(b, i);
            r = rocke_b_add(b, u->row, c_i);
        }
        idx[0] = r;
        idx[1] = regt_store_col(b, self, r, u->col);
        rocke_b_smem_store_vN(b, smem_dst, idx, 2, packed, ROCKE_REGT_K_RUN);
    }
}

void rocke_reg_transpose_tile_loader_store_lds(rocke_ir_builder_t* b,
                                               const rocke_reg_transpose_tile_loader_t* self,
                                               rocke_value_t* smem_dst,
                                               const rocke_regt_staged_t* staged)
{
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL || staged == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader.store_lds: NULL argument");
        return;
    }
    if(rocke_reg_transpose_tile_loader_partial(self))
    {
        if(staged->active == NULL)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "RegTransposeTileLoader.store_lds: partial participation needs the "
                            "staged list of a masked load_global");
            return;
        }
        rocke_if_t g = rocke_b_scf_if(b, staged->active);
        if(!rocke_ir_builder_ok(b))
            return;
        rocke_b_region_enter(b, g.then_region);
        for(int e = 0; e < staged->count; ++e)
            regt_store_unit(b, self, smem_dst, &staged->units[e]);
        rocke_b_region_leave(b);
        return;
    }
    for(int e = 0; e < staged->count; ++e)
        regt_store_unit(b, self, smem_dst, &staged->units[e]);
}

void rocke_reg_transpose_tile_loader_load(rocke_ir_builder_t* b,
                                          const rocke_reg_transpose_tile_loader_t* self,
                                          rocke_value_t* tid,
                                          rocke_value_t* smem_dst,
                                          rocke_loads_descriptor_fn descriptor,
                                          void* descriptor_user,
                                          rocke_value_t* rsrc,
                                          rocke_loads_run_descriptor_fn run_fn,
                                          void* run_user)
{
    if(b != NULL && b->status != ROCKE_OK)
        return;
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader.load: NULL loader");
        return;
    }
    /* Heap, not stack: ROCKE_REGT_MAX_UNITS units of ten pointers each. */
    rocke_regt_staged_t* staged = (rocke_regt_staged_t*)malloc(sizeof(rocke_regt_staged_t));
    if(staged == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "RegTransposeTileLoader.load: out of memory");
        return;
    }
    if(rocke_reg_transpose_tile_loader_partial(self))
    {
        rocke_value_t* act = regt_active(b, self, tid);
        rocke_if_t g = rocke_b_scf_if(b, act);
        if(rocke_ir_builder_ok(b))
        {
            rocke_b_region_enter(b, g.then_region);
            rocke_reg_transpose_tile_loader_load_global(b,
                                                        self,
                                                        tid,
                                                        descriptor,
                                                        descriptor_user,
                                                        rsrc,
                                                        /*masked=*/false,
                                                        run_fn,
                                                        run_user,
                                                        /*guard=*/false,
                                                        staged);
            for(int e = 0; e < staged->count; ++e)
                regt_store_unit(b, self, smem_dst, &staged->units[e]);
            rocke_b_region_leave(b);
        }
        free(staged);
        return;
    }
    rocke_reg_transpose_tile_loader_load_global(b,
                                                self,
                                                tid,
                                                descriptor,
                                                descriptor_user,
                                                rsrc,
                                                /*masked=*/true,
                                                run_fn,
                                                run_user,
                                                /*guard=*/false,
                                                staged);
    rocke_reg_transpose_tile_loader_store_lds(b, self, smem_dst, staged);
    free(staged);
}

/* ============================================================================
 * AsyncTileLoader / AsyncTileLoaderSlot
 * ========================================================================== */

/* choose_dwords with Python's `elem_bytes` argument: each chunk carries
 * d * 4 / elem_bytes elements (2-byte elements: d * 2 halves). */
static rocke_status_t atl_choose_dwords(int tile_rows,
                                        int tile_cols,
                                        int block_size,
                                        int elem_bytes,
                                        int max_dwords,
                                        int contig_cols,
                                        int* out);

rocke_status_t rocke_async_tile_loader_choose_dwords(
    int tile_rows, int tile_cols, int block_size, int max_dwords, int contig_cols, int* out)
{
    return atl_choose_dwords(tile_rows, tile_cols, block_size, 2, max_dwords, contig_cols, out);
}

static rocke_status_t atl_choose_dwords(int tile_rows,
                                        int tile_cols,
                                        int block_size,
                                        int elem_bytes,
                                        int max_dwords,
                                        int contig_cols,
                                        int* out)
{
    /* Python:
     *   if max_dwords > 4: max_dwords = 4
     *   for d in (4, 3, 1):
     *       if d > max_dwords: continue
     *       halves = (d * 4) // elem_bytes
     *       if tile_cols % halves != 0: continue
     *       chunks = (tile_rows*tile_cols) // halves
     *       if chunks < block_size: continue
     *       return d
     *   raise ValueError(...)
     */
    static const int kCandidates[3] = {4, 3, 1};
    int i;

    if(max_dwords > 4)
    {
        max_dwords = 4;
    }
    for(i = 0; i < 3; ++i)
    {
        int d = kCandidates[i];
        int halves;
        int chunks;
        if(d > max_dwords)
        {
            continue;
        }
        halves = (d * 4) / elem_bytes;
        /* A chunk must lie entirely inside one contiguous run of columns
         * (Python: contig_cols is not None and (elems > contig_cols or
         * contig_cols % elems != 0) -> skip). contig_cols == 0 means None. */
        if(contig_cols > 0 && (halves > contig_cols || contig_cols % halves != 0))
        {
            continue;
        }
        if(tile_cols % halves != 0)
        {
            continue;
        }
        chunks = (tile_rows * tile_cols) / halves;
        if(chunks < block_size)
        {
            continue;
        }
        /* Every pass fires block_size chunks; chunks past chunks_total still
         * issue (OOB sentinel, zero-filled), so a partial last pass writes past
         * this tile's allocation into the next smem_alloc. Require whole
         * passes -- a narrower width is tried next. Mirrors the Python. */
        if(chunks % block_size != 0)
        {
            continue;
        }
        if(out != NULL)
        {
            *out = d;
        }
        return ROCKE_OK;
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_async_tile_loader_from_tile(int tile_rows,
                                                 int tile_cols,
                                                 int block_size,
                                                 int wave_size,
                                                 int max_dwords,
                                                 int contig_cols,
                                                 rocke_async_tile_loader_t* out)
{
    return rocke_async_tile_loader_from_tile_typed(
        tile_rows, tile_cols, block_size, wave_size, NULL, max_dwords, contig_cols, out);
}

rocke_status_t rocke_async_tile_loader_from_tile_typed(int tile_rows,
                                                       int tile_cols,
                                                       int block_size,
                                                       int wave_size,
                                                       const rocke_type_t* elem_dtype,
                                                       int max_dwords,
                                                       int contig_cols,
                                                       rocke_async_tile_loader_t* out)
{
    int d;
    int halves;
    int chunks;
    int passes;
    rocke_status_t st;

    /* Python:
     *   eb = _ELEM_BYTES.get(elem_dtype.name)  (ValueError if None)
     *   d = cls.choose_dwords(..., elem_bytes=eb, ...)
     *   halves = (d * 4) // eb
     *   chunks = (tile_rows * tile_cols) // halves
     *   passes = (chunks + block_size - 1) // block_size
     *   return cls(... dwords=d, chunks_total=chunks,
     *              chunks_per_pass=block_size, passes=passes)
     */
    const int elem_bytes = loads_elem_bytes(elem_dtype);
    if(elem_bytes == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    st = atl_choose_dwords(
        tile_rows, tile_cols, block_size, elem_bytes, max_dwords, contig_cols, &d);
    if(st != ROCKE_OK)
    {
        return st;
    }
    halves = (d * 4) / elem_bytes;
    chunks = (tile_rows * tile_cols) / halves;
    passes = (chunks + block_size - 1) / block_size;
    if(out != NULL)
    {
        out->tile_rows = tile_rows;
        out->tile_cols = tile_cols;
        out->block_size = block_size;
        out->wave_size = wave_size;
        out->dwords = d;
        out->chunks_total = chunks;
        out->chunks_per_pass = block_size;
        out->passes = passes;
        out->contig_cols = contig_cols;
        out->elem_dtype = elem_dtype;
    }
    return ROCKE_OK;
}

int rocke_async_tile_loader_halves_per_chunk(const rocke_async_tile_loader_t* self)
{
    /* Python elems_per_chunk: return (dwords * 4) // _ELEM_BYTES[elem_dtype] */
    if(self == NULL)
        return 0;
    const int eb = loads_elem_bytes(self->elem_dtype);
    return (eb == 0) ? 0 : (self->dwords * 4) / eb;
}

int rocke_async_tile_loader_bytes_per_chunk(const rocke_async_tile_loader_t* self)
{
    /* Python: return dwords * 4 */
    return (self == NULL) ? 0 : self->dwords * 4;
}

int rocke_async_tile_loader_chunks_per_row(const rocke_async_tile_loader_t* self)
{
    /* Python: return self.tile_cols // self.elems_per_chunk
     *
     * Chunks needed to cover one tile row.  NOT elems_per_chunk: the two
     * coincide only when tile_cols == elems_per_chunk**2 (tile_cols=64 for a
     * 2-byte dtype at dwords=4), which is why 64 used to be the only tile
     * width the async path loaded correctly.  choose_dwords() guarantees
     * tile_cols % elems_per_chunk == 0, so this division is exact. */
    if(self == NULL)
    {
        return 0;
    }
    const int elems = rocke_async_tile_loader_halves_per_chunk(self);
    return (elems == 0) ? 0 : self->tile_cols / elems;
}

int rocke_async_tile_loader_wave_bytes(const rocke_async_tile_loader_t* self)
{
    /* Python: return wave_size * self.bytes_per_chunk */
    if(self == NULL)
    {
        return 0;
    }
    return self->wave_size * rocke_async_tile_loader_bytes_per_chunk(self);
}

int rocke_async_tile_loader_pass_bytes(const rocke_async_tile_loader_t* self)
{
    /* Python: return block_size * self.bytes_per_chunk */
    if(self == NULL)
    {
        return 0;
    }
    return self->block_size * rocke_async_tile_loader_bytes_per_chunk(self);
}

rocke_status_t rocke_async_tile_loader_bind(rocke_ir_builder_t* b,
                                            const rocke_async_tile_loader_t* self,
                                            rocke_value_t* smem_dst,
                                            rocke_value_t* wave_id,
                                            rocke_async_tile_loader_slot_t* out_slot)
{
    rocke_value_t* lds_base;
    rocke_value_t* wave_byte_off_i32;
    rocke_value_t* wave_byte_off_i64;
    rocke_value_t* per_wave_lds;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return b->status; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "AsyncTileLoader.bind: NULL loader");
        return ROCKE_ERR_VALUE;
    }

    /* Python:
     *   lds_base = b.smem_addr_of(smem_dst)                       # i64
     *   wave_byte_off_i32 = b.mul(wave_id, b.const_i32(self.wave_bytes))
     *   wave_byte_off_i32 = b.to_sgpr_u32(wave_byte_off_i32)
     *   wave_byte_off_i64 = b.zext(wave_byte_off_i32, I64)
     *   per_wave_lds = b.smem_ptr_add(lds_base, wave_byte_off_i64)
     *   return AsyncTileLoaderSlot(loader=self, smem_dst=smem_dst,
     *                              per_wave_lds_base=per_wave_lds)
     */
    lds_base = rocke_b_smem_addr_of(b, smem_dst);
    wave_byte_off_i32
        = rocke_b_mul(b, wave_id, rocke_b_const_i32(b, rocke_async_tile_loader_wave_bytes(self)));
    wave_byte_off_i32 = rocke_b_to_sgpr_u32(b, wave_byte_off_i32);
    wave_byte_off_i64 = rocke_b_zext(b, wave_byte_off_i32, rocke_i64());
    per_wave_lds = rocke_b_smem_ptr_add(b, lds_base, wave_byte_off_i64);

    if(out_slot != NULL)
    {
        out_slot->loader = *self;
        out_slot->smem_dst = smem_dst;
        out_slot->per_wave_lds_base = per_wave_lds;
    }
    return (b != NULL) ? b->status : ROCKE_OK;
}

void rocke_async_tile_loader_slot_issue(rocke_ir_builder_t* b,
                                        const rocke_async_tile_loader_slot_t* self,
                                        rocke_value_t* tid,
                                        rocke_value_t* rsrc,
                                        rocke_loads_descriptor_fn descriptor,
                                        void* descriptor_user,
                                        int oob_sentinel,
                                        int coherency)
{
    const rocke_async_tile_loader_t* L;
    rocke_value_t* c_half_bytes;
    rocke_value_t* c_oob;
    rocke_value_t* c0;
    rocke_value_t* c_chunks_per_row;
    int p;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "AsyncTileLoaderSlot.issue: NULL slot");
        return;
    }

    /* Python:
     *   L = self.loader
     *   c_half_bytes     = b.const_i32(2)
     *   c_oob            = b.const_i32(oob_sentinel)
     *   c0               = b.const_i32(0)
     *   c_chunks_per_row = b.const_i32(L.chunks_per_row)
     */
    L = &self->loader;
    const int elem_bytes = loads_elem_bytes(L->elem_dtype);
    if(elem_bytes == 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "AsyncTileLoader: unsupported elem_dtype '%s'",
                        L->elem_dtype->name);
        return;
    }
    c_half_bytes = rocke_b_const_i32(b, elem_bytes);
    c_oob = rocke_b_const_i32(b, oob_sentinel);
    c0 = rocke_b_const_i32(b, 0);
    c_chunks_per_row = rocke_b_const_i32(b, rocke_async_tile_loader_chunks_per_row(L));

    for(p = 0; p < L->passes; ++p)
    {
        int pass_byte_off;
        rocke_value_t* pass_base;
        rocke_value_t* chunk_idx;
        rocke_value_t* row;
        rocke_value_t* col_v;
        rocke_value_t* col;
        rocke_value_t* off_elems;
        rocke_value_t* valid;
        rocke_value_t* off_bytes;
        rocke_value_t* in_pass;
        rocke_value_t* valid_final;
        rocke_value_t* safe;

        /* Python:
         *   pass_byte_off = p * L.pass_bytes
         *   pass_base = (b.smem_ptr_add(self.per_wave_lds_base,
         *                               b.zext(b.const_i32(pass_byte_off), I64))
         *                if p > 0 else self.per_wave_lds_base)
         */
        pass_byte_off = p * rocke_async_tile_loader_pass_bytes(L);
        if(p > 0)
        {
            pass_base = rocke_b_smem_ptr_add(
                b,
                self->per_wave_lds_base,
                rocke_b_zext(b, rocke_b_const_i32(b, pass_byte_off), rocke_i64()));
        }
        else
        {
            pass_base = self->per_wave_lds_base;
        }

        /* Python:
         *   chunk_idx = b.add(tid, b.const_i32(p * L.block_size))
         *   row   = b.div(chunk_idx, c_chunks_per_row)
         *   col_v = b.mod(chunk_idx, c_chunks_per_row)
         *   col   = b.mul(col_v, b.const_i32(L.halves_per_chunk))
         */
        chunk_idx = rocke_b_add(b, tid, rocke_b_const_i32(b, p * L->block_size));
        row = rocke_b_div(b, chunk_idx, c_chunks_per_row);
        col_v = rocke_b_mod(b, chunk_idx, c_chunks_per_row);
        col = rocke_b_mul(
            b, col_v, rocke_b_const_i32(b, rocke_async_tile_loader_halves_per_chunk(L)));

        /* Python:
         *   off_elems, valid = descriptor(b, row, col)
         *   off_bytes = b.mul(off_elems, c_half_bytes)
         *   in_pass = b.cmp_lt(chunk_idx, b.const_i32(L.chunks_total))
         *   valid_final = b.land(valid, in_pass) if valid is not None else in_pass
         *   safe = b.select(valid_final, off_bytes, c_oob)
         *   b.async_buffer_load_lds_addr(rsrc, pass_base, safe, c0, L.dwords,
         *                                coherency=coherency)
         */
        valid = NULL;
        off_elems = descriptor(b, row, col, &valid, descriptor_user);
        off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
        in_pass = rocke_b_cmp_lt(b, chunk_idx, rocke_b_const_i32(b, L->chunks_total));
        if(valid != NULL)
        {
            valid_final = rocke_b_land(b, valid, in_pass);
        }
        else
        {
            valid_final = in_pass;
        }
        safe = rocke_b_select(b, valid_final, off_bytes, c_oob);
        rocke_b_async_buffer_load_lds_addr(b, rsrc, pass_base, safe, c0, L->dwords, coherency);
    }
}

int rocke_async_tile_loader_slot_required_lds_bytes(const rocke_async_tile_loader_slot_t* self)
{
    /* Python: return self.loader.passes * self.loader.pass_bytes */
    if(self == NULL)
    {
        return 0;
    }
    return self->loader.passes * rocke_async_tile_loader_pass_bytes(&self->loader);
}
