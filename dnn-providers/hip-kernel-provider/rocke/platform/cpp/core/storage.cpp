// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/storage.py: little-endian patterns, checked sizes and coordinates.
 * Matrix and scale operands share these mechanics without sharing semantics.
 */
#include "rocke/storage.h"

#include <string.h>

static bool valid(const rocke_bit_packing_t* p)
{
    return p && p->element_bits >= 1 && p->element_bits <= p->slot_bits && p->slot_bits <= 64;
}

static bool add(uint64_t a, uint64_t b, uint64_t* out)
{
    if(!out || a > UINT64_MAX - b)
        return false;
    *out = a + b;
    return true;
}

static bool mul(uint64_t a, uint64_t b, uint64_t* out)
{
    if(!out || (b && a > UINT64_MAX / b))
        return false;
    *out = a * b;
    return true;
}

bool rocke_bit_packing_init(rocke_bit_packing_t* out, int element_bits, int slot_bits)
{
    rocke_bit_packing_t p = {element_bits, slot_bits ? slot_bits : element_bits};
    if(!out || !valid(&p))
        return false;
    *out = p;
    return true;
}

bool rocke_bit_packing_group(const rocke_bit_packing_t* p,
                             int carrier_bits,
                             uint64_t* values,
                             uint64_t* carriers)
{
    if(!valid(p) || !values || !carriers
       || (carrier_bits != 8 && carrier_bits != 16 && carrier_bits != 32 && carrier_bits != 64))
        return false;
    int a = p->slot_bits, b = carrier_bits;
    while(b)
    {
        int next = a % b;
        a = b;
        b = next;
    }
    *values = carrier_bits / a;
    *carriers = p->slot_bits / a;
    return true;
}

bool rocke_bit_packing_offset(const rocke_bit_packing_t* p, uint64_t index, uint64_t* bits)
{
    return valid(p) && mul(index, p->slot_bits, bits);
}

bool rocke_bit_packing_bytes(const rocke_bit_packing_t* p,
                             uint64_t count,
                             uint64_t bit_offset,
                             uint64_t* bytes)
{
    uint64_t bits;
    if(!bytes || !rocke_bit_packing_offset(p, count, &bits))
        return false;
    if(!count)
    {
        *bytes = 0;
        return true;
    }
    if(!add(bits, bit_offset, &bits))
        return false;
    *bytes = bits / 8 + (bits % 8 != 0);
    return true;
}

bool rocke_bit_pack(const rocke_bit_packing_t* p,
                    const uint64_t* patterns,
                    size_t count,
                    uint64_t bit_offset,
                    uint8_t* dst,
                    size_t dst_size)
{
    uint64_t bytes;
    if(!rocke_bit_packing_bytes(p, count, bit_offset, &bytes) || bytes > dst_size
       || (count && (!patterns || !dst)))
        return false;
    for(size_t i = 0; i < count; ++i)
        if(p->element_bits < 64 && patterns[i] >= (uint64_t(1) << p->element_bits))
            return false;
    if(bytes)
        memset(dst, 0, (size_t)bytes);
    for(size_t i = 0; i < count; ++i)
    {
        const uint64_t pos = bit_offset + i * uint64_t(p->slot_bits);
        for(int bit = 0; bit < p->element_bits; ++bit)
        {
            const uint64_t at = pos + bit;
            dst[at / 8] |= uint8_t(((patterns[i] >> bit) & 1) << (at % 8));
        }
    }
    return true;
}

bool rocke_bit_unpack(const rocke_bit_packing_t* p,
                      const uint8_t* src,
                      size_t src_size,
                      size_t count,
                      uint64_t bit_offset,
                      uint64_t* patterns)
{
    uint64_t bytes;
    if(!rocke_bit_packing_bytes(p, count, bit_offset, &bytes) || bytes > src_size
       || (count && (!patterns || !src)))
        return false;
    for(size_t i = 0; i < count; ++i)
    {
        patterns[i] = 0;
        const uint64_t pos = bit_offset + i * uint64_t(p->slot_bits);
        for(int bit = 0; bit < p->element_bits; ++bit)
        {
            const uint64_t at = pos + bit;
            patterns[i] |= uint64_t((src[at / 8] >> (at % 8)) & 1) << bit;
        }
    }
    return true;
}

bool rocke_fragment_packing_init(rocke_fragment_packing_t* out,
                                 const rocke_bit_packing_t* p,
                                 uint64_t count,
                                 int carrier_bits,
                                 uint64_t carrier_count)
{
    uint64_t values, carriers, bits, capacity;
    if(!out || !rocke_bit_packing_group(p, carrier_bits, &values, &carriers)
       || !rocke_bit_packing_offset(p, count, &bits) || !mul(carrier_count, carrier_bits, &capacity)
       || bits > capacity)
        return false;
    *out = {*p, count, carrier_bits, carrier_count};
    return true;
}

bool rocke_fragment_pack(const rocke_fragment_packing_t* p,
                         const uint64_t* patterns,
                         size_t count,
                         uint64_t* carriers,
                         size_t carrier_count)
{
    rocke_fragment_packing_t checked;
    if(!p
       || !rocke_fragment_packing_init(
           &checked, &p->packing, p->count, p->carrier_bits, p->carrier_count)
       || count != p->count || carrier_count != p->carrier_count || (count && !patterns)
       || (carrier_count && !carriers))
        return false;
    for(size_t i = 0; i < count; ++i)
        if(p->packing.element_bits < 64 && patterns[i] >= (uint64_t(1) << p->packing.element_bits))
            return false;
    for(size_t i = 0; i < carrier_count; ++i)
        carriers[i] = 0;
    for(size_t i = 0; i < count; ++i)
        for(int bit = 0; bit < p->packing.element_bits; ++bit)
        {
            uint64_t at = i * uint64_t(p->packing.slot_bits) + bit;
            carriers[at / p->carrier_bits] |= ((patterns[i] >> bit) & 1) << (at % p->carrier_bits);
        }
    return true;
}

bool rocke_tensor_storage_init(rocke_tensor_storage_t* out,
                               const char* dtype,
                               uint64_t rows,
                               uint64_t cols,
                               uint64_t row_stride_bytes,
                               int slot_bits,
                               uint64_t base_bit_offset,
                               uint64_t alignment_bytes)
{
    rocke_tensor_storage_t p = {};
    p.dtype = rocke_dtype_info(dtype);
    uint64_t row_bytes, bytes;
    if(!out || !p.dtype || !alignment_bytes || (alignment_bytes & (alignment_bytes - 1))
       || !rocke_bit_packing_init(&p.packing, p.dtype->encoded_bits, slot_bits)
       || !rocke_bit_packing_bytes(&p.packing, cols, base_bit_offset, &row_bytes))
        return false;
    p.rows = rows;
    p.cols = cols;
    p.row_stride_bytes = row_stride_bytes == UINT64_MAX ? row_bytes : row_stride_bytes;
    p.base_bit_offset = base_bit_offset;
    p.alignment_bytes = alignment_bytes;
    if(p.row_stride_bytes < row_bytes || !rocke_tensor_storage_bytes(&p, &bytes))
        return false;
    *out = p;
    return true;
}

bool rocke_tensor_storage_bytes(const rocke_tensor_storage_t* p, uint64_t* bytes)
{
    if(!p || !bytes || !valid(&p->packing))
        return false;
    if(!p->rows || !p->cols)
    {
        *bytes = 0;
        return true;
    }
    uint64_t row_bytes, prefix;
    return rocke_bit_packing_bytes(&p->packing, p->cols, p->base_bit_offset, &row_bytes)
           && mul(p->rows - 1, p->row_stride_bytes, &prefix) && add(prefix, row_bytes, bytes);
}

bool rocke_tensor_storage_address(
    const rocke_tensor_storage_t* p, uint64_t row, uint64_t col, uint64_t* byte, int* shift)
{
    if(!p || !byte || !shift || row >= p->rows || col >= p->cols)
        return false;
    uint64_t bits, prefix;
    if(!rocke_bit_packing_offset(&p->packing, col, &bits) || !add(bits, p->base_bit_offset, &bits)
       || !mul(row, p->row_stride_bytes, &prefix) || !add(prefix, bits / 8, byte))
        return false;
    *shift = bits % 8;
    return true;
}

bool rocke_matrix_fragment_layout_init(rocke_matrix_fragment_layout_t* out,
                                       const rocke_fragment_packing_t* fragment,
                                       int chunk_elements,
                                       int lane_groups,
                                       int lanes_per_group)
{
    rocke_fragment_packing_t checked;
    uint64_t bits;
    if(!out || !fragment || chunk_elements <= 0 || lane_groups <= 0 || lanes_per_group <= 0
       || fragment->count % chunk_elements
       || !rocke_fragment_packing_init(&checked,
                                       &fragment->packing,
                                       fragment->count,
                                       fragment->carrier_bits,
                                       fragment->carrier_count)
       || !rocke_bit_packing_offset(&fragment->packing, chunk_elements, &bits) || bits % 8)
        return false;
    *out = {*fragment, chunk_elements, lane_groups, lanes_per_group};
    return true;
}

bool rocke_matrix_fragment_coord(
    const rocke_matrix_fragment_layout_t* p, int lane, uint64_t slot, uint64_t* row, uint64_t* k)
{
    // Public C descriptors can be constructed without going through init().
    rocke_matrix_fragment_layout_t checked;
    if(!p || !row || !k || lane < 0
       || !rocke_matrix_fragment_layout_init(
           &checked, &p->fragment, p->chunk_elements, p->lane_groups, p->lanes_per_group)
       || uint64_t(lane) >= uint64_t(p->lane_groups) * p->lanes_per_group
       || slot >= p->fragment.count)
        return false;
    uint64_t chunk;
    *row = lane % p->lanes_per_group;
    return mul(slot / p->chunk_elements, p->lane_groups, &chunk)
           && add(chunk, lane / p->lanes_per_group, &chunk) && mul(chunk, p->chunk_elements, &chunk)
           && add(chunk, slot % p->chunk_elements, k);
}
