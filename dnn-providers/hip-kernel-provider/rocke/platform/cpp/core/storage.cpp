// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors core/storage.py: little-endian patterns, checked sizes and coordinates.
 * Matrix and scale operands share these mechanics without sharing semantics.
 */
#include "rocke/storage.h"

#include <exception>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "rocke/dtypes_internal.h"
#include "rocke/error.hpp"

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

/* ======================= sub-byte values in registers =======================
 *
 * Mirrors core/storage.py register_packing and packed_word_range. The bodies
 * throw a ckc::Error where Python raises, with Python's text; the extern "C"
 * entry points turn it into `false` plus the message, so no exception crosses
 * the C API.
 */

/* Width of the registers packed operands occupy (Python _REGISTER_BITS). */
static const int k_register_bits = 32;

/* Python _checked's refusal of a size or offset past the uint64 range. */
[[noreturn]] static void raise_storage_overflow()
{
    throw ckc::ValueError("storage size or offset exceeds uint64 range");
}

/* Python _require_int_at_least, less the integer-type check C makes at compile
 * time. */
static void require_at_least(const char* name, int64_t value, int64_t minimum)
{
    if(value < minimum)
        throw ckc::ValueError(ckc::format_error(
            "%s must be at least %" PRId64 ", got %" PRId64, name, minimum, value));
}

/* Run `body` and report its outcome through the C API: true, or false with the
 * exception's message copied into `err`. */
template <class Body>
static bool report_through_c_api(char* err, size_t err_cap, Body&& body)
{
    try
    {
        body();
    }
    catch(const std::exception& error)
    {
        if(err && err_cap)
            snprintf(err, err_cap, "%s", error.what());
        return false;
    }
    if(err && err_cap)
        err[0] = '\0';
    return true;
}

bool rocke_register_packing(const char* dtype,
                            int64_t count,
                            bool* packed,
                            rocke_fragment_packing_t* out,
                            char* err,
                            size_t err_cap)
{
    return report_through_c_api(err, err_cap, [&] {
        if(!packed || !out)
            throw ckc::ValueError("rocke_register_packing: packed and out must not be NULL");
        const rocke_dtype_info_t* info = rocke_require_dtype_info(dtype);
        require_at_least("element count", count, 1);
        if(info->category == ROCKE_DTYPE_PREDICATE)
            throw ckc::ValueError(
                ckc::format_error("%s is a predicate, not register data", info->name));
        if(info->encoded_bits >= 8)
        {
            *packed = false;
            return;
        }
        uint64_t bits;
        if(!mul(uint64_t(count), uint64_t(info->encoded_bits), &bits))
            raise_storage_overflow();
        if(bits % k_register_bits)
            throw ckc::ValueError(ckc::format_error(
                "%" PRId64 " %s elements occupy %" PRIu64 " bits, not a whole number of %d-bit "
                "registers -- packed operands must fill whole registers",
                count,
                info->name,
                bits,
                k_register_bits));
        rocke_bit_packing_t element;
        if(!rocke_bit_packing_init(&element, info->encoded_bits, 0)
           || !rocke_fragment_packing_init(
               out, &element, uint64_t(count), k_register_bits, bits / k_register_bits))
            raise_storage_overflow();
        *packed = true;
    });
}

bool rocke_packed_word_range(const rocke_fragment_packing_t* packing,
                             int64_t start,
                             int64_t length,
                             uint64_t* first_word,
                             uint64_t* end_word,
                             char* err,
                             size_t err_cap)
{
    return report_through_c_api(err, err_cap, [&] {
        // Python cannot hold an invalid FragmentPacking; a C descriptor can be
        // filled in by hand, so check it the way rocke_fragment_pack does.
        rocke_fragment_packing_t checked;
        if(!packing || !first_word || !end_word
           || !rocke_fragment_packing_init(&checked,
                                           &packing->packing,
                                           packing->count,
                                           packing->carrier_bits,
                                           packing->carrier_count))
            throw ckc::ValueError("rocke_packed_word_range: needs a valid fragment packing "
                                  "(rocke_fragment_packing_init) and two output words");
        require_at_least("start", start, 0);
        require_at_least("length", length, 1);
        // Both are below 2^63, so the sum fits.
        const uint64_t first = uint64_t(start);
        const uint64_t end = first + uint64_t(length);
        if(end > packing->count)
            throw ckc::ValueError(ckc::format_error("elements [%" PRIu64 ", %" PRIu64
                                                    ") run past the fragment's "
                                                    "%" PRIu64 " elements",
                                                    first,
                                                    end,
                                                    packing->count));
        uint64_t first_bit, end_bit;
        if(!rocke_bit_packing_offset(&packing->packing, first, &first_bit)
           || !rocke_bit_packing_offset(&packing->packing, end, &end_bit))
            raise_storage_overflow();
        const uint64_t word = uint64_t(packing->carrier_bits);
        if(first_bit % word || end_bit % word)
        {
            uint64_t step, carriers;
            rocke_bit_packing_group(&packing->packing, packing->carrier_bits, &step, &carriers);
            throw ckc::ValueError(ckc::format_error(
                "elements [%" PRIu64 ", %" PRIu64 ") of a fragment packed at %d bits per element "
                "do not start and end on a register boundary (%d bits) -- start and length "
                "must be multiples of %" PRIu64 " elements",
                first,
                end,
                packing->packing.slot_bits,
                packing->carrier_bits,
                step));
        }
        *first_word = first_bit / word;
        *end_word = end_bit / word;
    });
}
