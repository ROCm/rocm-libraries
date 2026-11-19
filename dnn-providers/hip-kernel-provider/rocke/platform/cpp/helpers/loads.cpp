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

#include <stdint.h>
#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

bool rocke_tdm_tile_loader_2d_prepare(rocke_ir_builder_t* b,
                                      const rocke_tdm_tile_loader_2d_t* loader,
                                      const rocke_tdm_source_window_2d_t* source,
                                      rocke_value_t* lds_addr,
                                      rocke_value_t* zero,
                                      rocke_tdm_tile_load_t* out)
{
    if(loader == NULL || source == NULL || out == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm tile load needs loader, source and output");
        return false;
    }
    if(!rocke_i_live(b))
        return false;
    const rocke_type_t* ptr = source->ptr ? source->ptr->type : NULL;
    if(ptr == NULL || ptr->kind != ROCKE_TYPE_PTR || strcmp(ptr->space, "global") != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "TDM source must be a global pointer");
        return false;
    }
    const int bytes = loader->elem_bytes;
    if((bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
       || rocke_i_type_bits(ptr->pointee) != 8 * bytes)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "TDM source pointee width must match elem_bytes (1, 2, 4, or 8)");
        return false;
    }
    /* Keep the builder-call order explicit to mirror Python. */
    rocke_value_t* row_off64 = rocke_b_zext(b, source->row_origin, rocke_i64());
    rocke_value_t* stride64 = rocke_b_zext(b, source->row_stride, rocke_i64());
    rocke_value_t* row64 = rocke_b_mul(b, row_off64, stride64);
    rocke_value_t* col_off64 = rocke_b_zext(b, source->col_origin, rocke_i64());
    rocke_value_t* in_window = rocke_b_add(b, row64, col_off64);
    rocke_value_t* origin = rocke_b_add(b, source->base_offset, in_window);
    rocke_value_t* addr = rocke_b_global_addr_of(b, source->ptr, origin);
    rocke_value_t* rows_left = rocke_b_sub(b, source->row_extent, source->row_origin);
    rocke_value_t* rows_remaining = rocke_b_smax(b, rows_left, zero);
    return rocke_b_tdm_descriptor_2d(b,
                                     addr,
                                     lds_addr,
                                     loader->elem_bytes,
                                     source->cols_remaining,
                                     rows_remaining,
                                     source->row_stride,
                                     loader->tile_cols,
                                     loader->tile_rows,
                                     0,
                                     loader->pad_interval,
                                     loader->pad_amount,
                                     out->groups);
}

void rocke_tdm_tile_load_issue(rocke_ir_builder_t* b,
                               const rocke_tdm_tile_load_t* load,
                               int cachepolicy)
{
    if(load == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm tile load needs a prepared descriptor");
        return;
    }
    const auto* g = load->groups;
    rocke_b_tensor_load_to_lds(b, g[0], g[1], g[2], g[3], g[4], cachepolicy);
}

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

/* Keep the legacy immediate order when the loader supplies a col_xor tuple.
 * The bound form supplies SSA values instead and has no extra flip bit. */
static rocke_value_t* emit_lds_col_xor(rocke_ir_builder_t* b,
                                       rocke_value_t* row,
                                       rocke_value_t* col,
                                       const int* immediate,
                                       rocke_value_t* row_shift,
                                       rocke_value_t* chunk_mask,
                                       rocke_value_t* chunk_log2)
{
    if(immediate != NULL && immediate[0] != 0)
        row_shift = rocke_b_const_i32(b, immediate[0]);
    rocke_value_t* r = row_shift ? rocke_b_lshr(b, row, row_shift) : row;
    if(immediate != NULL)
        chunk_mask = rocke_b_const_i32(b, immediate[1]);
    rocke_value_t* sel = rocke_b_land(b, r, chunk_mask);
    if(immediate != NULL && immediate[3] >= 0)
    {
        rocke_value_t* c_flip = rocke_b_const_i32(b, immediate[3]);
        rocke_value_t* hi = rocke_b_lshr(b, row, c_flip);
        rocke_value_t* c_one = rocke_b_const_i32(b, 1);
        sel = rocke_b_xor(b, sel, rocke_b_land(b, hi, c_one));
    }
    if(immediate != NULL)
        chunk_log2 = rocke_b_const_i32(b, immediate[2]);
    return rocke_b_xor(b, col, rocke_b_shl(b, sel, chunk_log2));
}

const rocke_lds_column_xor_t rocke_lds_no_swizzle{0, 0, 0, -1};
const rocke_lds_column_xor_t rocke_lds_row_xor_2x16b{0, 1, 4, -1};
const rocke_lds_column_xor_t rocke_lds_row_xor_4x16b{0, 2, 4, -1};
const rocke_lds_column_xor_t rocke_lds_row_xor_8x16b{0, 3, 4, -1};

bool rocke_lds_column_xor_is_identity(const rocke_lds_column_xor_t* layout)
{
    return layout != NULL && layout->row_bits == 0 && layout->flip_row_bit == -1;
}

int rocke_lds_column_xor_chunk_mask(const rocke_lds_column_xor_t* layout)
{
    if(layout == NULL || layout->row_bits < 0 || layout->row_bits > 31)
        return 0;
    return static_cast<int>((1u << layout->row_bits) - 1u);
}

/* The plain shift query preserves its -1 sentinel; only the builder API
 * reports the diagnostic that corresponds to the Python check. */
static const char*
    lds_column_xor_check_shift(const rocke_lds_column_xor_t* layout, int unit_bytes, int& shift)
{
    if(layout == NULL)
        return "LDS column XOR layout must not be null";
    if(layout->row_shift < 0 || layout->row_shift > 31 || layout->row_bits < 0
       || layout->row_bits > 31 || layout->chunk_log2_bytes < 0 || layout->chunk_log2_bytes > 31
       || layout->flip_row_bit < -1 || layout->flip_row_bit > 31)
        return "LDS column XOR terms must be integers in [0, 31]";
    if(rocke_lds_column_xor_is_identity(layout))
    {
        shift = 0;
        return NULL;
    }
    if(unit_bytes <= 0 || (unit_bytes & (unit_bytes - 1)) != 0)
        return "LDS column units must be a positive power of two bytes";
    shift = layout->chunk_log2_bytes;
    for(; unit_bytes > 1; unit_bytes >>= 1)
        --shift;
    return shift < 0 ? "LDS column unit is larger than a swizzle chunk" : NULL;
}

int rocke_lds_column_xor_shift_for(const rocke_lds_column_xor_t* layout, int unit_bytes)
{
    int shift;
    return lds_column_xor_check_shift(layout, unit_bytes, shift) == NULL ? shift : -1;
}

rocke_value_t* rocke_lds_column_xor_emit(rocke_ir_builder_t* b,
                                         rocke_value_t* row,
                                         rocke_value_t* col,
                                         const rocke_lds_column_xor_t* layout,
                                         int unit_bytes)
{
    if(!rocke_i_live(b))
        return NULL;
    int shift;
    if(const char* diagnostic = lds_column_xor_check_shift(layout, unit_bytes, shift))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", diagnostic);
    if(rocke_lds_column_xor_is_identity(layout))
        return col;
    const int terms[4]
        = {layout->row_shift, rocke_lds_column_xor_chunk_mask(layout), shift, layout->flip_row_bit};
    return emit_lds_col_xor(b, row, col, terms, NULL, NULL, NULL);
}

rocke_value_t* rocke_emit_lds_col_xor(rocke_ir_builder_t* b,
                                      rocke_value_t* row,
                                      rocke_value_t* col,
                                      rocke_value_t* row_shift,
                                      rocke_value_t* chunk_mask,
                                      rocke_value_t* chunk_log2)
{
    if(!rocke_i_live(b))
        return NULL;
    if(chunk_mask == NULL || chunk_log2 == NULL)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "emit_lds_col_xor: missing mask or shift");
    return emit_lds_col_xor(b, row, col, NULL, row_shift, chunk_mask, chunk_log2);
}

rocke_value_t* rocke_swizzle_lds_col(rocke_ir_builder_t* b,
                                     rocke_value_t* row,
                                     rocke_value_t* col,
                                     const int col_xor[4])
{
    if(!rocke_i_live(b))
        return NULL;
    if(col_xor == NULL)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "swizzle_lds_col: NULL col_xor");
    return emit_lds_col_xor(b, row, col, col_xor, NULL, NULL, NULL);
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

enum class ctl_geometry_error
{
    none,
    missing_loader,
    nonpositive_divisor,
    nonpositive_extent,
    partial_vector,
    index_overflow,
    uneven_distribution,
};

static ctl_geometry_error ctl_chunk_count(const rocke_coalesced_tile_loader_t* self, int& total)
{
    /* Keep refusal precedence in sync with CoalescedTileLoader.vecs_per_thread. */
    if(self == NULL)
        return ctl_geometry_error::missing_loader;
    if(self->load_vec <= 0 || self->block_size <= 0)
        return ctl_geometry_error::nonpositive_divisor;
    if(self->tile_rows <= 0 || self->tile_cols <= 0)
        return ctl_geometry_error::nonpositive_extent;
    const int axis = self->vector_axis_row ? self->tile_rows : self->tile_cols;
    const int outer = self->vector_axis_row ? self->tile_cols : self->tile_rows;
    if(axis % self->load_vec != 0)
        return ctl_geometry_error::partial_vector;
    /* Traversal materializes the linear chunk ID in i32. Widen the product
     * so malformed geometry cannot overflow before the check. */
    const int64_t total_vecs = static_cast<int64_t>(axis / self->load_vec) * outer;
    if(total_vecs > INT32_MAX)
        return ctl_geometry_error::index_overflow;
    total = static_cast<int>(total_vecs);
    return total % self->block_size == 0 ? ctl_geometry_error::none
                                         : ctl_geometry_error::uneven_distribution;
}

rocke_status_t
    rocke_coalesced_tile_loader_vecs_per_thread(const rocke_coalesced_tile_loader_t* self, int* out)
{
    int total = 0;
    if(ctl_chunk_count(self, total) != ctl_geometry_error::none)
        return ROCKE_ERR_VALUE;
    if(out != NULL)
        *out = total / self->block_size;
    return ROCKE_OK;
}

static bool
    ctl_require_geometry(rocke_ir_builder_t* b, const rocke_coalesced_tile_loader_t* self, int* out)
{
    int total = 0;
    switch(ctl_chunk_count(self, total))
    {
    case ctl_geometry_error::none:
        if(out != NULL)
            *out = total / self->block_size;
        return true;
    case ctl_geometry_error::missing_loader:
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader: missing loader");
        break;
    case ctl_geometry_error::nonpositive_divisor:
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: load_vec and block_size must be positive");
        break;
    case ctl_geometry_error::nonpositive_extent:
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: tile dimensions must be positive");
        break;
    case ctl_geometry_error::partial_vector:
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: load_vec must divide the vector axis");
        break;
    case ctl_geometry_error::index_overflow:
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader: chunk geometry exceeds i32");
        break;
    case ctl_geometry_error::uneven_distribution:
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "tile %dx%d / %d = %d not divisible by block_size %d",
                        self->tile_rows,
                        self->tile_cols,
                        self->load_vec,
                        total,
                        self->block_size);
        break;
    }
    return false;
}

int rocke_coalesced_tile_loader_cols_per_vec(const rocke_coalesced_tile_loader_t* self)
{
    /* Python: return tile_cols // load_vec for valid geometry. */
    if(self == NULL || self->load_vec <= 0)
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
    rocke_value_t* chunk = idx1;
    if(self->load_vec > 1)
    {
        rocke_value_t* width = c_load_vec ? c_load_vec : rocke_b_const_i32(b, self->load_vec);
        chunk = rocke_b_mul(b, idx1, width);
    }
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

void rocke_coalesced_tile_loader_visit_chunks(rocke_ir_builder_t* b,
                                              const rocke_coalesced_tile_loader_t* self,
                                              rocke_value_t* tid,
                                              const rocke_coalesced_chunk_grid_t* grid,
                                              rocke_coalesced_chunk_fn emit,
                                              void* user)
{
    if(!rocke_i_live(b))
        return;
    int count;
    if(self == NULL || tid == NULL || emit == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader: invalid chunk traversal");
        return;
    }
    if(!ctl_require_geometry(b, self, &count))
        return;
    rocke_coalesced_chunk_grid_t local{};
    if(grid == NULL)
    {
        local.threads = rocke_b_const_i32(b, self->block_size);
        local.width = rocke_b_const_i32(b, self->load_vec);
        local.span = rocke_b_const_i32(b, ctl_span(self));
        grid = &local;
    }
    for(int e = 0; e < count; ++e)
    {
        rocke_value_t* row;
        rocke_value_t* col;
        rocke_value_t* vec_idx
            = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), grid->threads), tid);
        if(grid->before_decode != nullptr)
            grid->before_decode(b, vec_idx, user);
        ctl_decode_row_col(b, self, vec_idx, grid->span, grid->width, &row, &col);
        emit(b, row, col, vec_idx, user);
    }
}

struct ctl_load_call
{
    const rocke_coalesced_tile_loader_t* self;
    rocke_value_t* smem_dst;
    rocke_loads_descriptor_fn descriptor;
    void* descriptor_user;
    rocke_value_t* rsrc;
    rocke_value_t* ptr;
    rocke_value_t* c_elem_bytes;
    rocke_value_t* c0;
    rocke_value_t* c_oob;
};

static void ctl_emit_load_chunk(
    rocke_ir_builder_t* b, rocke_value_t* row, rocke_value_t* col, rocke_value_t*, void* user)
{
    const auto& call = *static_cast<ctl_load_call*>(user);
    rocke_value_t* valid = NULL;
    rocke_value_t* off = call.descriptor(b, row, col, &valid, call.descriptor_user);
    rocke_value_t* value;
    if(call.self->use_buffer_rsrc)
    {
        rocke_value_t* bytes = rocke_b_mul(b, off, call.c_elem_bytes);
        rocke_value_t* safe = valid ? rocke_b_select(b, valid, bytes, call.c_oob) : bytes;
        value = ctl_buffer_read(b, call.self, call.rsrc, safe, call.c0);
    }
    else
    {
        value = ctl_global_read(b, call.self, call.ptr, off);
    }
    ctl_store_tile(b, call.self, call.smem_dst, row, col, value);
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
    if(!rocke_i_live(b))
        return;
    if(self == NULL || descriptor == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader.load: NULL loader or descriptor");
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
    if(!ctl_require_geometry(b, self, NULL))
        return;
    const rocke_coalesced_chunk_grid_t grid{rocke_b_const_i32(b, self->block_size),
                                            rocke_b_const_i32(b, self->load_vec),
                                            rocke_b_const_i32(b, ctl_span(self)),
                                            NULL};
    ctl_load_call call{self,
                       smem_dst,
                       descriptor,
                       descriptor_user,
                       rsrc,
                       ptr,
                       rocke_b_const_i32(b, elem_bytes),
                       rocke_b_const_i32(b, 0),
                       rocke_b_const_i32(b, self->oob_sentinel)};
    rocke_coalesced_tile_loader_visit_chunks(b, self, tid, &grid, ctl_emit_load_chunk, &call);
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
    if(!ctl_require_geometry(b, self, &vecs_per_thread))
        return;
    c_threads = rocke_b_const_i32(b, self->block_size);
    c_load_vec = rocke_b_const_i32(b, self->load_vec);
    c_span = rocke_b_const_i32(b, ctl_span(self));
    c_half_bytes = rocke_b_const_i32(b, elem_bytes);
    c0 = rocke_b_const_i32(b, 0);
    c_oob = rocke_b_const_i32(b, self->oob_sentinel);
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

    int vecs_per_thread = 0;
    if(!ctl_require_geometry(b, self, &vecs_per_thread))
        return;
    /* Python: c_span is rows_per_vec in row mode, cols_per_vec otherwise. */
    rocke_value_t* c_threads = rocke_b_const_i32(b, self->block_size);
    rocke_value_t* c_load_vec = rocke_b_const_i32(b, self->load_vec);
    rocke_value_t* c_span = rocke_b_const_i32(b, ctl_span(self));
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

struct atl_pass_coordinates
{
    rocke_value_t* base;
    rocke_value_t* chunk;
    rocke_value_t* row;
    rocke_value_t* col;
};

static atl_pass_coordinates atl_get_pass_coordinates(rocke_ir_builder_t* b,
                                                     const rocke_async_tile_loader_slot_t* slot,
                                                     rocke_value_t* tid,
                                                     rocke_value_t* chunks_per_row,
                                                     rocke_value_t* cols_per_chunk,
                                                     int slot_bytes,
                                                     int p)
{
    const auto& loader = slot->loader;
    const int byte_off = slot_bytes + p * rocke_async_tile_loader_pass_bytes(&loader);
    rocke_value_t* base = slot->per_wave_lds_base;
    if(byte_off)
        base = rocke_b_smem_ptr_add(
            b, base, rocke_b_zext(b, rocke_b_const_i32(b, byte_off), rocke_i64()));
    rocke_value_t* chunk = rocke_b_add(b, tid, rocke_b_const_i32(b, p * loader.block_size));
    rocke_value_t* row = rocke_b_div(b, chunk, chunks_per_row);
    rocke_value_t* col_in_chunks = rocke_b_mod(b, chunk, chunks_per_row);
    rocke_value_t* step = cols_per_chunk;
    if(step == NULL)
        step = rocke_b_const_i32(b, rocke_async_tile_loader_halves_per_chunk(&loader));
    rocke_value_t* col = rocke_b_mul(b, col_in_chunks, step);
    return {base, chunk, row, col};
}

void rocke_async_tile_loader_slot_issue_bytes(rocke_ir_builder_t* b,
                                              const rocke_async_tile_loader_slot_t* self,
                                              rocke_value_t* tid,
                                              rocke_value_t* rsrc,
                                              rocke_loads_byte_descriptor_fn descriptor,
                                              void* descriptor_user,
                                              rocke_value_t* chunks_per_row,
                                              rocke_value_t* cols_per_chunk,
                                              rocke_value_t* zero_soff,
                                              int slot_bytes,
                                              int coherency)
{
    if(!rocke_i_live(b))
        return;
    if(self == NULL || descriptor == NULL || chunks_per_row == NULL || cols_per_chunk == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "AsyncTileLoaderSlot.issue_bytes: unbound issue");
        return;
    }
    for(int p = 0; p < self->loader.passes; ++p)
    {
        const auto pass
            = atl_get_pass_coordinates(b, self, tid, chunks_per_row, cols_per_chunk, slot_bytes, p);
        rocke_value_t* offset = descriptor(b, pass.row, pass.col, descriptor_user);
        rocke_b_async_buffer_load_lds_addr(
            b, rsrc, pass.base, offset, zero_soff, self->loader.dwords, coherency);
    }
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
        const auto pass = atl_get_pass_coordinates(b, self, tid, c_chunks_per_row, NULL, 0, p);
        rocke_value_t* off_elems;
        rocke_value_t* valid;
        rocke_value_t* off_bytes;
        rocke_value_t* in_pass;
        rocke_value_t* valid_final;
        rocke_value_t* safe;

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
        off_elems = descriptor(b, pass.row, pass.col, &valid, descriptor_user);
        off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
        in_pass = rocke_b_cmp_lt(b, pass.chunk, rocke_b_const_i32(b, L->chunks_total));
        if(valid != NULL)
        {
            valid_final = rocke_b_land(b, valid, in_pass);
        }
        else
        {
            valid_final = in_pass;
        }
        safe = rocke_b_select(b, valid_final, off_bytes, c_oob);
        rocke_b_async_buffer_load_lds_addr(b, rsrc, pass.base, safe, c0, L->dwords, coherency);
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
