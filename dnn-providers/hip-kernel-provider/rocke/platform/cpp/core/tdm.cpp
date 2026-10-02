// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * core/tdm.cpp -- C99 port of rocke.core.tdm (gfx1250 TDM descriptor
 * construction). See rocke/tdm.h for the Python <-> C99 correspondence.
 *
 * The bit layout is transcribed from the Python field tables, which mirror
 * TDM_GROUP0 / TDM_GROUP1 in CK's amd_tdm_descriptor.hpp. A wrong bit gives
 * silent garbage or a hang, never a compile error; tests/core/test_tdm.cpp pins
 * the packing field by field.
 *
 * rocke_tdm_build_descriptor_2d must issue builder calls in exactly the order
 * build_tdm_descriptor_2d does (argument evaluation order included): every call
 * allocates an SSA name, so a reordering changes the emitted IR bytes.
 */
#include "rocke/tdm.h"

#include <stdarg.h>
#include <stdio.h>

#include "rocke/ir_internal.h"

namespace
{

typedef struct tdm_field
{
    const char* name;
    int word;
    int lsb;
    int width;
} tdm_field_t;

/* (word, lsb, width) per field, the same entries as _GROUP0_FIELDS /
 * _GROUP1_FIELDS. */
const tdm_field_t k_g0_count = {"count", 0, 0, 2};
const tdm_field_t k_g0_lds_addr = {"lds_addr", 1, 0, 32};
const tdm_field_t k_g0_global_addr_lo = {"global_addr_lo", 2, 0, 32};
const tdm_field_t k_g0_global_addr_hi = {"global_addr_hi", 3, 0, 25};
const tdm_field_t k_g0_type = {"type", 3, 30, 2};
const tdm_field_t k_g0_gather_index_size = {"gather_index_size", 0, 30, 1};
const tdm_field_t k_g0_gather_mode = {"gather_mode", 0, 31, 1};

const tdm_field_t k_g1_workgroup_mask = {"workgroup_mask", 0, 0, 16};
const tdm_field_t k_g1_data_size = {"data_size", 0, 16, 2};
const tdm_field_t k_g1_atomic_barrier_enable = {"atomic_barrier_enable", 0, 18, 1};
const tdm_field_t k_g1_iterate_enable = {"iterate_enable", 0, 19, 1};
const tdm_field_t k_g1_pad_enable = {"pad_enable", 0, 20, 1};
const tdm_field_t k_g1_early_timeout = {"early_timeout", 0, 21, 1};
const tdm_field_t k_g1_pad_interval = {"pad_interval", 0, 22, 3};
const tdm_field_t k_g1_pad_amount = {"pad_amount", 0, 25, 7};
const tdm_field_t k_g1_atomic_barrier_address = {"atomic_barrier_address", 1, 0, 16};
const tdm_field_t k_g1_tensor_dim0_lo = {"tensor_dim0_lo", 1, 16, 16};
const tdm_field_t k_g1_tensor_dim0_hi = {"tensor_dim0_hi", 2, 0, 16};
const tdm_field_t k_g1_tensor_dim1_lo = {"tensor_dim1_lo", 2, 16, 16};
const tdm_field_t k_g1_tensor_dim1_hi = {"tensor_dim1_hi", 3, 0, 16};
const tdm_field_t k_g1_tile_dim0 = {"tile_dim0", 3, 16, 16};
const tdm_field_t k_g1_tile_dim1 = {"tile_dim1", 4, 0, 16};
const tdm_field_t k_g1_tile_dim2 = {"tile_dim2", 4, 16, 16};
const tdm_field_t k_g1_tensor_dim0_stride_lo = {"tensor_dim0_stride_lo", 5, 0, 32};
const tdm_field_t k_g1_tensor_dim0_stride_hi = {"tensor_dim0_stride_hi", 6, 0, 16};
const tdm_field_t k_g1_tensor_dim1_stride_lo = {"tensor_dim1_stride_lo", 6, 16, 16};
const tdm_field_t k_g1_tensor_dim1_stride_hi = {"tensor_dim1_stride_hi", 7, 0, 32};

/* CK's TDM_GROUP0 constructor: count = 1 and type = 2 ("set to 2 for spg"). */
const int64_t k_group0_count = 1;
const int64_t k_group0_type = 2;
const int64_t k_bytes_per_dword = 4;

rocke_status_t tdm_fail(char* reason, size_t cap, const char* fmt, ...)
{
    va_list ap;
    if(reason && cap)
    {
        va_start(ap, fmt);
        vsnprintf(reason, cap, fmt, ap);
        va_end(ap);
    }
    return ROCKE_ERR_VALUE;
}

/* One iteration of Python _pack: range-check `value` against the field width and
 * OR it into place. */
rocke_status_t
    tdm_put(uint32_t* words, const tdm_field_t* f, int64_t value, char* reason, size_t cap)
{
    if(value < 0 || (value >> f->width) != 0)
    {
        if(value < 0)
            return tdm_fail(reason,
                            cap,
                            "TDM field '%s' is %d bits, cannot hold %lld (0x-%llx)",
                            f->name,
                            f->width,
                            (long long)value,
                            (unsigned long long)(-value));
        return tdm_fail(reason,
                        cap,
                        "TDM field '%s' is %d bits, cannot hold %lld (0x%llx)",
                        f->name,
                        f->width,
                        (long long)value,
                        (unsigned long long)value);
    }
    words[f->word] |= (uint32_t)((uint64_t)value << f->lsb);
    return ROCKE_OK;
}

/* Python int.bit_length() for a positive value. */
int bit_length(int64_t v)
{
    int n = 0;
    while(v > 0)
    {
        ++n;
        v >>= 1;
    }
    return n;
}

/* Python _insert_words: zero_vec, then for each index either a readfirstlane'd
 * runtime word or a non-zero constant word. */
rocke_value_t* insert_words(rocke_ir_builder_t* b,
                            int words,
                            const uint32_t* statics,
                            const bool* has_static,
                            rocke_value_t* const* dynamic)
{
    rocke_value_t* vec = rocke_b_zero_vec(b, rocke_i32(), words);
    int index;
    for(index = 0; index < words; ++index)
    {
        if(dynamic[index] != NULL)
        {
            vec = rocke_b_vec_insert(b, vec, rocke_b_readfirstlane(b, dynamic[index]), index);
        }
        else if(has_static[index] && statics[index] != 0)
        {
            vec = rocke_b_vec_insert(b, vec, rocke_b_const_i32(b, (int64_t)statics[index]), index);
        }
    }
    return vec;
}

} // namespace

extern "C" {

rocke_status_t
    rocke_tdm_data_size_code(int64_t elem_bytes, int* out_code, char* reason, size_t reason_cap)
{
    int code;
    switch(elem_bytes)
    {
    case 8: code = 3; break;
    case 4: code = 2; break;
    case 2: code = 1; break;
    case 1: code = 0; break;
    default:
        return tdm_fail(reason,
                        reason_cap,
                        "TDM data_size has no encoding for %lld-byte elements",
                        (long long)elem_bytes);
    }
    if(out_code)
        *out_code = code;
    return ROCKE_OK;
}

rocke_status_t rocke_tdm_encode_padding(int64_t interval_bytes,
                                        int64_t pad_bytes,
                                        int* out_pad_interval,
                                        int* out_pad_amount,
                                        char* reason,
                                        size_t reason_cap)
{
    int64_t interval_dwords;
    int64_t pad_interval;
    int64_t pad_amount;
    if(pad_bytes <= 0)
        return tdm_fail(reason,
                        reason_cap,
                        "encode_tdm_padding needs a positive pad; pad_enable=0 otherwise");
    if(interval_bytes % k_bytes_per_dword || pad_bytes % k_bytes_per_dword)
        return tdm_fail(reason,
                        reason_cap,
                        "TDM padding is dword-granular, got interval=%lldB pad=%lldB",
                        (long long)interval_bytes,
                        (long long)pad_bytes);
    interval_dwords = interval_bytes / k_bytes_per_dword;
    if(interval_dwords < 2 || (interval_dwords & (interval_dwords - 1)))
        return tdm_fail(reason,
                        reason_cap,
                        "TDM pad interval must be a power-of-two dword count >= 2, got %lld",
                        (long long)interval_dwords);
    pad_interval = bit_length(interval_dwords) - 2;
    pad_amount = pad_bytes / k_bytes_per_dword - 1;
    if(pad_interval > ROCKE_TDM_PAD_INTERVAL_MAX)
        return tdm_fail(reason,
                        reason_cap,
                        "TDM pad_interval %lld exceeds %d (interval %lldB too large)",
                        (long long)pad_interval,
                        ROCKE_TDM_PAD_INTERVAL_MAX,
                        (long long)interval_bytes);
    if(pad_amount > ROCKE_TDM_PAD_AMOUNT_MAX)
        return tdm_fail(reason,
                        reason_cap,
                        "TDM pad_amount %lld exceeds %d (pad %lldB too large)",
                        (long long)pad_amount,
                        ROCKE_TDM_PAD_AMOUNT_MAX,
                        (long long)pad_bytes);
    if(out_pad_interval)
        *out_pad_interval = (int)pad_interval;
    if(out_pad_amount)
        *out_pad_amount = (int)pad_amount;
    return ROCKE_OK;
}

rocke_status_t rocke_tdm_padding_for_tile(int64_t elem_bytes,
                                          int64_t row_elems,
                                          int64_t pad_elems,
                                          int* out_pad_enable,
                                          int* out_pad_interval,
                                          int* out_pad_amount,
                                          char* reason,
                                          size_t reason_cap)
{
    int interval = 0;
    int amount = 0;
    int enable = 0;
    if(pad_elems != 0)
    {
        rocke_status_t st = rocke_tdm_encode_padding(
            elem_bytes * row_elems, elem_bytes * pad_elems, &interval, &amount, reason, reason_cap);
        if(st != ROCKE_OK)
            return st;
        enable = 1;
    }
    if(out_pad_enable)
        *out_pad_enable = enable;
    if(out_pad_interval)
        *out_pad_interval = interval;
    if(out_pad_amount)
        *out_pad_amount = amount;
    return ROCKE_OK;
}

rocke_status_t rocke_tdm_pack_group0(int64_t lds_addr,
                                     int64_t global_addr,
                                     int64_t gather_index_size,
                                     int64_t gather_mode,
                                     uint32_t out_words[4],
                                     char* reason,
                                     size_t reason_cap)
{
    uint32_t w[4] = {0, 0, 0, 0};
    rocke_status_t st;
    int i;
    /* Same field order as the Python values dict, so the first bad field wins. */
    if((st = tdm_put(w, &k_g0_count, k_group0_count, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g0_lds_addr, lds_addr & 0xFFFFFFFF, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g0_global_addr_lo, global_addr & 0xFFFFFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g0_global_addr_hi, (global_addr >> 32) & 0x1FFFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w, &k_g0_type, k_group0_type, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g0_gather_index_size, gather_index_size, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w, &k_g0_gather_mode, gather_mode, reason, reason_cap)) != ROCKE_OK)
        return st;
    for(i = 0; i < 4; ++i)
        out_words[i] = w[i];
    return ROCKE_OK;
}

rocke_status_t rocke_tdm_pack_group1_2d(const rocke_tdm_group1_2d_args_t* a,
                                        uint32_t out_words[8],
                                        char* reason,
                                        size_t reason_cap)
{
    uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    rocke_status_t st;
    int data_size;
    int i;
    if(a->dim1_stride > ROCKE_TDM_MAX_DIM1_STRIDE)
        return tdm_fail(reason,
                        reason_cap,
                        "TDM dim1 stride %lld exceeds the %d-element field; the descriptor "
                        "truncates it to 16 bits",
                        (long long)a->dim1_stride,
                        ROCKE_TDM_MAX_DIM1_STRIDE);
    st = rocke_tdm_data_size_code(a->elem_bytes, &data_size, reason, reason_cap);
    if(st != ROCKE_OK)
        return st;
    /* Same field order as the Python values dict, so the first bad field wins.
     * CK truncates the dim-1 stride to 16 bits; see ROCKE_TDM_MAX_DIM1_STRIDE. */
    if((st = tdm_put(w, &k_g1_workgroup_mask, a->workgroup_mask, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_data_size, data_size, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_atomic_barrier_enable, a->atomic_barrier_enable, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_atomic_barrier_address, a->atomic_barrier_address, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w, &k_g1_iterate_enable, 0, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_pad_enable, a->pad_enable, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_early_timeout, 0, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_pad_interval, a->pad_interval, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_pad_amount, a->pad_amount, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_tensor_dim0_lo, a->tensor_dim0 & 0xFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_tensor_dim0_hi, (a->tensor_dim0 >> 16) & 0xFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_tensor_dim1_lo, a->tensor_dim1 & 0xFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_tensor_dim1_hi, (a->tensor_dim1 >> 16) & 0xFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w, &k_g1_tile_dim0, a->tile_dim0, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_tile_dim1, a->tile_dim1, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w, &k_g1_tile_dim2, 0, reason, reason_cap)) != ROCKE_OK
       || (st = tdm_put(w,
                        &k_g1_tensor_dim0_stride_lo,
                        a->dim0_stride & 0xFFFFFFFF,
                        reason,
                        reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w,
                        &k_g1_tensor_dim0_stride_hi,
                        (a->dim0_stride >> 32) & 0xFFFF,
                        reason,
                        reason_cap))
              != ROCKE_OK
       || (st = tdm_put(
               w, &k_g1_tensor_dim1_stride_lo, a->dim1_stride & 0xFFFF, reason, reason_cap))
              != ROCKE_OK
       || (st = tdm_put(w,
                        &k_g1_tensor_dim1_stride_hi,
                        (a->dim1_stride >> 32) & 0xFFFFFFFF,
                        reason,
                        reason_cap))
              != ROCKE_OK)
        return st;
    for(i = 0; i < 8; ++i)
        out_words[i] = w[i];
    return ROCKE_OK;
}

rocke_status_t rocke_tdm_build_descriptor_2d(rocke_ir_builder_t* b,
                                             const rocke_tdm_descriptor_2d_args_t* a,
                                             rocke_value_t* out_groups[5])
{
    char reason[ROCKE_ERR_MSG_CAP];
    uint32_t word0[4] = {0, 0, 0, 0};
    uint32_t static1[8];
    bool has_static0[4] = {true, false, false, false};
    bool has_static1[8] = {true, false, false, false, true, true, true, true};
    rocke_value_t* dyn0[4] = {NULL, NULL, NULL, NULL};
    rocke_value_t* dyn1[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    rocke_value_t *lds32, *g_lo, *g_hi, *d0, *d1, *zeros4, *zeros8;
    rocke_value_t *mask16, *sixteen, *dim0_lo, *dim0_hi, *dim1_lo, *dim1_hi, *mid;
    rocke_tdm_group1_2d_args_t g1 = {};
    bool stride0_is_value;
    int i;

    for(i = 0; i < 5; ++i)
        out_groups[i] = NULL;
    if(!rocke_i_live(b))
        return b->status;
    if(!a || !a->global_addr || !a->lds_addr || !a->tensor_dim0 || !a->tensor_dim1)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "build_tdm_descriptor_2d: NULL operand");
        return b->status;
    }

    /* Group 0: the two addresses are runtime, the rest is a constant word. */
    lds32 = rocke_b_trunc(b, a->lds_addr, rocke_i32());
    g_lo = rocke_b_trunc(b, a->global_addr, rocke_i32());
    g_hi = rocke_b_trunc(b, rocke_b_lshr(b, a->global_addr, rocke_b_const_i64(b, 32)), rocke_i32());
    g_hi = rocke_b_land(b, g_hi, rocke_b_const_i32(b, 0x1FFFFFF));
    word0[0] = (uint32_t)k_group0_count;
    word0[3] = (uint32_t)k_group0_type << 30;
    /* `type` shares word 3 with global_addr_hi, so that word is merged at
     * runtime rather than taken from the static pattern. */
    dyn0[1] = lds32;
    dyn0[2] = g_lo;
    dyn0[3] = rocke_b_lor(b, g_hi, rocke_b_const_i32(b, (int64_t)word0[3]));
    d0 = insert_words(b, 4, word0, has_static0, dyn0);

    /* Group 1: the tensor extents are runtime, and the row pitch may be too. */
    stride0_is_value = a->dim0_stride_value != NULL;
    g1.elem_bytes = a->elem_bytes;
    g1.tile_dim0 = a->tile_dim0;
    g1.tile_dim1 = a->tile_dim1;
    g1.dim0_stride = stride0_is_value ? 0 : a->dim0_stride;
    g1.dim1_stride = a->dim1_stride;
    g1.pad_enable = a->pad_enable;
    g1.pad_interval = a->pad_interval;
    g1.pad_amount = a->pad_amount;
    if(rocke_tdm_pack_group1_2d(&g1, static1, reason, sizeof(reason)) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", reason);
        return b->status;
    }
    mask16 = rocke_b_const_i32(b, 0xFFFF);
    sixteen = rocke_b_const_i32(b, 16);
    dim0_lo = rocke_b_shl(b, rocke_b_land(b, a->tensor_dim0, mask16), sixteen);
    dim0_hi = rocke_b_land(b, rocke_b_lshr(b, a->tensor_dim0, sixteen), mask16);
    dim1_lo = rocke_b_shl(b, rocke_b_land(b, a->tensor_dim1, mask16), sixteen);
    dim1_hi = rocke_b_land(b, rocke_b_lshr(b, a->tensor_dim1, sixteen), mask16);
    dyn1[1] = rocke_b_lor(b, dim0_lo, rocke_b_const_i32(b, (int64_t)static1[1]));
    mid = rocke_b_lor(b, dim0_hi, dim1_lo);
    dyn1[2] = rocke_b_lor(b, mid, rocke_b_const_i32(b, (int64_t)static1[2]));
    dyn1[3] = rocke_b_lor(b, dim1_hi, rocke_b_const_i32(b, (int64_t)static1[3]));
    if(stride0_is_value)
    {
        /* Slot 0 is 48 bits (word 5 plus word 6's low half); a row pitch that
         * fits an i32 leaves the high half at its static value. */
        dyn1[5] = a->dim0_stride_value;
    }
    d1 = insert_words(b, 8, static1, has_static1, dyn1);

    zeros4 = rocke_b_zero_vec(b, rocke_i32(), 4);
    zeros8 = rocke_b_zero_vec(b, rocke_i32(), 8);
    if(!rocke_i_live(b))
        return b->status;
    out_groups[0] = d0;
    out_groups[1] = d1;
    out_groups[2] = zeros4;
    out_groups[3] = zeros4;
    out_groups[4] = zeros8;
    return ROCKE_OK;
}

} /* extern "C" */
