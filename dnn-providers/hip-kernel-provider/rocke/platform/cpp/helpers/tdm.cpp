// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helpers.tdm.c -- C99 port of rocke.helpers.tdm.
 *
 * Builds the five gfx1250 TDM `D#` descriptor groups. Byte-faithful
 * translation: every builder call happens in the same order as the Python, so
 * the emitted IR and its SSA numbering are identical. Two places where that
 * ordering is load-bearing and easy to "tidy" into a divergence:
 *
 *   - `pack_word` emits the constant accumulator *before* walking the dynamic
 *     fields, and within each dynamic field emits mask -> shift -> or. Folding
 *     the mask constant upward renumbers every later value.
 *   - group 0's `global_lo` / `global_hi` are computed before the group-0 word
 *     list is built, not lazily inside it.
 *
 * The encoding asymmetry that motivates the split helper: `tensor_strides[0]`
 * splits at bit 32 (lo32 + hi16), `tensor_strides[1]` splits at bit 16
 * (lo16 + hi32). See helper_helpers.tdm.h.
 */

#include "rocke/helper_helpers.tdm.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err, rocke_i_live */

#include <stdio.h>
#include <string.h>

/* group1.data_size is a 2-bit log2(element bytes) code. */
static int data_size_code(int elem_bytes)
{
    switch(elem_bytes)
    {
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 8: return 3;
    default: return -1;
    }
}

int rocke_h_tdm_data_size_code(int elem_bytes)
{
    return data_size_code(elem_bytes);
}

/* 0 = NULL tensor, 1 = valid tensor. */
#define TDM_GROUP0_COUNT_VALID 1
/* "must be 2" */
#define TDM_GROUP0_TYPE 2

static rocke_tdm_operand_t op_k(int64_t k)
{
    rocke_tdm_operand_t o;
    o.is_value = false;
    o.k = k;
    o.v = NULL;
    return o;
}

static rocke_tdm_operand_t op_v(rocke_value_t* v)
{
    rocke_tdm_operand_t o;
    o.is_value = true;
    o.k = 0;
    o.v = v;
    return o;
}

typedef struct tdm_field
{
    const char* name;
    rocke_tdm_operand_t val;
    int offset;
    int width;
} tdm_field_t;

/* Reinterpret an unsigned 32-bit pattern as the signed value LLVM wants. */
static int64_t as_signed_i32(uint64_t value)
{
    value &= 0xFFFFFFFFu;
    return value >= 0x80000000u ? (int64_t)value - ((int64_t)1 << 32) : (int64_t)value;
}

/* Pack (name, value, bit_offset, width) fields into one i32. Compile-time ints
 * fold into a single constant; SSA values are masked, shifted, and or-ed in.
 * Out-of-range constants are rejected rather than silently truncated -- a
 * descriptor that is quietly wrong drives a DMA engine. */
static rocke_value_t* pack_word(rocke_ir_builder_t* b, const tdm_field_t* fields, int n)
{
    uint64_t const_bits = 0;
    const tdm_field_t* dynamic[16];
    int ndyn = 0;

    for(int i = 0; i < n; ++i)
    {
        const tdm_field_t* f = &fields[i];
        const uint64_t mask = (f->width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << f->width) - 1);
        if(!f->val.is_value)
        {
            if(f->val.k < 0 || (uint64_t)f->val.k > mask)
            {
                char msg[160];
                snprintf(msg,
                         sizeof(msg),
                         "TDM field '%s' does not fit %d bits (0..%llu), got %lld",
                         f->name,
                         f->width,
                         (unsigned long long)mask,
                         (long long)f->val.k);
                return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
            }
            const_bits |= (uint64_t)f->val.k << f->offset;
        }
        else
        {
            dynamic[ndyn++] = f;
        }
    }

    if(ndyn == 0)
        return rocke_b_const_i32(b, as_signed_i32(const_bits));

    rocke_value_t* acc = NULL;
    if(const_bits)
        acc = rocke_b_const_i32(b, as_signed_i32(const_bits));

    for(int i = 0; i < ndyn; ++i)
    {
        const tdm_field_t* f = dynamic[i];
        rocke_value_t* part = f->val.v;
        if(f->width < 32)
            part = rocke_b_land(
                b, part, rocke_b_const_i32(b, as_signed_i32(((uint64_t)1 << f->width) - 1)));
        if(f->offset)
            part = rocke_b_shl(b, part, rocke_b_const_i32(b, f->offset));
        acc = (acc == NULL) ? part : rocke_b_lor(b, acc, part);
    }
    return acc;
}

/* Split `value` into (low, high) about `shift`.
 *
 * Constants split exactly. SSA operands are i32, so a split at or above bit 32
 * has a statically-zero high half -- and it must be returned as the literal 0,
 * NOT as `lshr i32 x, 32`: a shift by the full width is poison in LLVM, and
 * poison in a stride field points the DMA engine at a wild address. That is a
 * runtime memory fault, not a compile error, so no static check catches it. */
static void lo_hi(rocke_ir_builder_t* b,
                  rocke_tdm_operand_t value,
                  int shift,
                  rocke_tdm_operand_t* lo,
                  rocke_tdm_operand_t* hi)
{
    if(!value.is_value)
    {
        const uint64_t uv = (uint64_t)value.k;
        *lo = op_k((int64_t)(uv & (((uint64_t)1 << shift) - 1)));
        *hi = op_k((int64_t)(uv >> shift));
        return;
    }
    if(shift >= 32)
    {
        *lo = value;
        *hi = op_k(0);
        return;
    }
    *lo = value;
    *hi = op_v(rocke_b_lshr(b, value.v, rocke_b_const_i32(b, shift)));
}

static rocke_value_t*
    make_vector(rocke_ir_builder_t* b, rocke_value_t* const* words, int nwords, int lanes)
{
    rocke_value_t* vec = rocke_b_zero_vec(b, rocke_i32(), lanes);
    for(int i = 0; i < nwords; ++i)
        vec = rocke_b_vec_insert(b, vec, words[i], i);
    return vec;
}

/* interval_dwords is a power of two in [2, 256]; the field is bit_length - 2. */
static int bit_length(int v)
{
    int n = 0;
    while(v)
    {
        ++n;
        v >>= 1;
    }
    return n;
}

bool rocke_h_tdm_descriptor_groups(rocke_ir_builder_t* b,
                                   const rocke_tdm_desc_args_t* a,
                                   rocke_value_t* out[5])
{
    if(!rocke_i_live(b))
        return false;
    if(!a || !out)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm_descriptor_groups: NULL args");
        return false;
    }
    const int rank = a->rank;
    if(rank < 1 || rank > ROCKE_TDM_MAX_RANK)
    {
        char msg[192];
        snprintf(msg,
                 sizeof(msg),
                 "tdm_descriptor_groups supports rank 1..%d, got %d. Higher ranks need "
                 "groups 2/3, whose mode union (dims 2-4 vs gather indices vs iterate "
                 "config) is not modelled here.",
                 ROCKE_TDM_MAX_RANK,
                 rank);
        rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
        return false;
    }
    if(!a->global_addr || a->global_addr->type != rocke_i64())
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm_descriptor_groups global_addr must be i64");
        return false;
    }
    const int ds = data_size_code(a->elem_bytes);
    if(ds < 0)
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "TDM elem_bytes must be 1, 2, 4, or 8, got %d", a->elem_bytes);
        rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
        return false;
    }

    rocke_value_t* lds_addr = a->lds_addr;
    if(!lds_addr)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm_descriptor_groups: NULL lds_addr");
        return false;
    }
    if(lds_addr->type == rocke_i64())
        lds_addr = rocke_b_trunc(b, lds_addr, rocke_i32());
    else if(lds_addr->type != rocke_i32())
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "tdm_descriptor_groups lds_addr must be i32 or i64");
        return false;
    }

    for(int i = 0; i < rank; ++i)
    {
        if(a->tile_dims[i] < 0 || a->tile_dims[i] > 0xFFFF)
        {
            char msg[112];
            snprintf(
                msg, sizeof(msg), "tile_dims[%d] must fit 16 bits, got %d", i, a->tile_dims[i]);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
            return false;
        }
        if(!a->tensor_strides[i].is_value
           && (a->tensor_strides[i].k < 0 || a->tensor_strides[i].k >= ((int64_t)1 << 48)))
        {
            char msg[144];
            snprintf(msg,
                     sizeof(msg),
                     "tensor_strides[%d] must fit the 48-bit field, got %lld",
                     i,
                     (long long)a->tensor_strides[i].k);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
            return false;
        }
    }

    /* Pad rank-1 out to the two-dimension descriptor slots with zeros. */
    rocke_tdm_operand_t dims[ROCKE_TDM_MAX_RANK];
    rocke_tdm_operand_t strides[ROCKE_TDM_MAX_RANK];
    int tiles[ROCKE_TDM_MAX_RANK];
    for(int i = 0; i < ROCKE_TDM_MAX_RANK; ++i)
    {
        dims[i] = (i < rank) ? a->tensor_dims[i] : op_k(0);
        strides[i] = (i < rank) ? a->tensor_strides[i] : op_k(0);
        tiles[i] = (i < rank) ? a->tile_dims[i] : 0;
    }

    int pad_interval = 0, pad_amount = 0;
    if(a->padding)
    {
        const int iv = a->padding->interval_dwords;
        const int amt = a->padding->amount_dwords;
        if(iv < 2 || iv > 256 || (iv & (iv - 1)) != 0)
        {
            char msg[128];
            snprintf(msg,
                     sizeof(msg),
                     "TdmPadding.interval_dwords must be a power of two in [2, 256], got %d",
                     iv);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
            return false;
        }
        if(amt < 1 || amt > 128)
        {
            char msg[112];
            snprintf(
                msg, sizeof(msg), "TdmPadding.amount_dwords must be in [1, 128], got %d", amt);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, msg);
            return false;
        }
        pad_interval = bit_length(iv) - 2;
        pad_amount = amt - 1;
    }

    /* ---- group 0: addresses and op mode ---------------------------------
     * is_store stays 0: direction is chosen by which intrinsic is called.
     * scope/th stay 0: cache policy rides the intrinsic's own immediate. */
    rocke_value_t* global_lo = rocke_b_trunc(b, a->global_addr, rocke_i32());
    rocke_value_t* global_hi = rocke_b_trunc(
        b, rocke_b_lshr(b, a->global_addr, rocke_b_const_i64(b, 32)), rocke_i32());

    const tdm_field_t g0w0[] = {
        {"count", op_k(TDM_GROUP0_COUNT_VALID), 0, 2},
        {"is_restore", op_k(a->is_restore ? 1 : 0), 2, 1},
        {"is_store", op_k(0), 3, 1},
        {"nv", op_k(0), 4, 1},
        {"scope", op_k(0), 5, 2},
        {"th", op_k(0), 7, 3},
        {"gather_index_size", op_k(0), 30, 1},
        {"gather_mode", op_k(0), 31, 1},
    };
    const tdm_field_t g0w3[] = {
        {"global_addr_hi", op_v(global_hi), 0, 25},
        {"type", op_k(TDM_GROUP0_TYPE), 30, 2},
    };

    rocke_value_t* group0_words[4];
    group0_words[0] = pack_word(b, g0w0, (int)(sizeof(g0w0) / sizeof(g0w0[0])));
    group0_words[1] = lds_addr;
    group0_words[2] = global_lo;
    group0_words[3] = pack_word(b, g0w3, (int)(sizeof(g0w3) / sizeof(g0w3[0])));

    /* ---- group 1: shape, stride, modifiers ------------------------------ */
    rocke_tdm_operand_t dim0_lo, dim0_hi, dim1_lo, dim1_hi;
    rocke_tdm_operand_t stride0_lo, stride0_hi, stride1_lo, stride1_hi;
    lo_hi(b, dims[0], 16, &dim0_lo, &dim0_hi);
    lo_hi(b, dims[1], 16, &dim1_lo, &dim1_hi);
    /* The two strides are NOT encoded alike: stride[0] is lo32+hi16 (split at
     * bit 32), stride[1] is lo16+hi32 (split at bit 16). Splitting the second
     * at 32 drops bits [16:32]; that is masked only while stride[1] < 2**16. */
    lo_hi(b, strides[0], 32, &stride0_lo, &stride0_hi);
    lo_hi(b, strides[1], 16, &stride1_lo, &stride1_hi);

    const tdm_field_t g1w0[] = {
        {"workgroup_mask", op_k(a->workgroup_mask), 0, 16},
        {"data_size", op_k(ds), 16, 2},
        {"atomic_barrier_enable", op_k(0), 18, 1},
        {"iterate_enable", op_k(0), 19, 1},
        {"pad_enable", op_k(a->padding ? 1 : 0), 20, 1},
        {"early_timeout", op_k(a->early_timeout ? 1 : 0), 21, 1},
        {"pad_interval", op_k(pad_interval), 22, 3},
        {"pad_amount", op_k(pad_amount), 25, 7},
    };
    const tdm_field_t g1w1[] = {
        {"atomic_barrier_address", op_k(0), 0, 16},
        {"tensor_dim0_lo", dim0_lo, 16, 16},
    };
    const tdm_field_t g1w2[] = {
        {"tensor_dim0_hi", dim0_hi, 0, 16},
        {"tensor_dim1_lo", dim1_lo, 16, 16},
    };
    const tdm_field_t g1w3[] = {
        {"tensor_dim1_hi", dim1_hi, 0, 16},
        {"tile_dim0", op_k(tiles[0]), 16, 16},
    };
    const tdm_field_t g1w4[] = {
        {"tile_dim1", op_k(tiles[1]), 0, 16},
        {"tile_dim2", op_k(0), 16, 16},
    };
    const tdm_field_t g1w5[] = {
        {"tensor_dim0_stride_lo", stride0_lo, 0, 32},
    };
    const tdm_field_t g1w6[] = {
        {"tensor_dim0_stride_hi", stride0_hi, 0, 16},
        {"tensor_dim1_stride_lo", stride1_lo, 16, 16},
    };
    const tdm_field_t g1w7[] = {
        {"tensor_dim1_stride_hi", stride1_hi, 0, 32},
    };

    rocke_value_t* group1_words[8];
    group1_words[0] = pack_word(b, g1w0, 8);
    group1_words[1] = pack_word(b, g1w1, 2);
    group1_words[2] = pack_word(b, g1w2, 2);
    group1_words[3] = pack_word(b, g1w3, 2);
    group1_words[4] = pack_word(b, g1w4, 2);
    group1_words[5] = pack_word(b, g1w5, 1);
    group1_words[6] = pack_word(b, g1w6, 2);
    group1_words[7] = pack_word(b, g1w7, 1);

    if(!rocke_i_live(b))
        return false;

    if(a->scalarize)
    {
        for(int i = 0; i < 4; ++i)
            group0_words[i] = rocke_b_to_sgpr_u32(b, group0_words[i]);
        for(int i = 0; i < 8; ++i)
            group1_words[i] = rocke_b_to_sgpr_u32(b, group1_words[i]);
    }

    out[0] = make_vector(b, group0_words, 4, 4);
    out[1] = make_vector(b, group1_words, 8, 8);
    out[2] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out[3] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out[4] = rocke_b_zero_vec(b, rocke_i32(), 8);
    return rocke_i_live(b);
}

bool rocke_h_tdm_row_major_2d(rocke_ir_builder_t* b,
                              const rocke_tdm_desc_args_t* base,
                              rocke_tdm_operand_t rows,
                              rocke_tdm_operand_t cols,
                              rocke_tdm_operand_t row_pitch,
                              int tile_rows,
                              int tile_cols,
                              rocke_value_t* out[5])
{
    if(!base)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "tdm_row_major_2d: NULL base args");
        return false;
    }
    /* Natural (rows, cols) order in, fastest-first out, so the dimension order
     * cannot be got backwards at the call site. */
    rocke_tdm_desc_args_t a = *base;
    a.rank = 2;
    a.tensor_dims[0] = cols;
    a.tensor_dims[1] = rows;
    a.tensor_strides[0] = row_pitch;
    a.tensor_strides[1] = op_k(0);
    a.tile_dims[0] = tile_cols;
    a.tile_dims[1] = tile_rows;
    return rocke_h_tdm_descriptor_groups(b, &a, out);
}

bool rocke_h_tdm_load_to_lds(rocke_ir_builder_t* b,
                             const rocke_tdm_desc_args_t* args,
                             int cachepolicy,
                             bool wait)
{
    rocke_value_t* d[5];
    if(!rocke_h_tdm_descriptor_groups(b, args, d))
        return false;
    rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], cachepolicy);
    if(wait)
        rocke_b_s_wait_tensorcnt(b, 0);
    return rocke_i_live(b);
}
