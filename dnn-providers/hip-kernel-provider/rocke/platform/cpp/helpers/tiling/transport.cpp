// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Internal global storage-load policy; mirrors helpers/tiling/transport.py. */
#include "rocke/tiling_transport_internal.h"

#include <limits.h>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"

static uint64_t gcd(uint64_t a, uint64_t b)
{
    while(b)
    {
        uint64_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

rocke_status_t rocke_h_load_storage_chunks(rocke_ir_builder_t* b,
                                           rocke_value_t* ptr,
                                           rocke_value_t* offset,
                                           const rocke_type_t* element,
                                           int count,
                                           int alignment_bytes,
                                           bool byte_addressed,
                                           bool word_vectors,
                                           rocke_value_t** pieces,
                                           size_t capacity,
                                           int* piece_count)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(count <= 0)
            ckc::raise_status(ROCKE_ERR_VALUE, "storage load count must be positive");
        if(alignment_bytes <= 0 || (alignment_bytes & (alignment_bytes - 1)))
            ckc::raise_status(ROCKE_ERR_VALUE, "alignment_bytes must be a positive power of two");
        if(byte_addressed && !rocke_type_eq(element, rocke_i32()))
            ckc::raise_status(ROCKE_ERR_VALUE, "byte-addressed storage loads require i32 elements");
        const auto* info = element ? rocke_dtype_info(element->name) : NULL;
        if(!info || info->encoded_bits % 8 || !ptr || !offset || !pieces || !piece_count)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid storage load arguments");
        const int element_bytes = info->encoded_bits / 8;
        if(byte_addressed && uint64_t(count - 1) * element_bytes > INT_MAX)
            ckc::raise_status(ROCKE_ERR_VALUE, "storage load displacement exceeds i32 range");
        const int max_width = byte_addressed ? 4 : element_bytes == 4 ? 8 : 16;
        int remaining = count, consumed = 0, written = 0;
        while(remaining)
        {
            int width = 1;
            while(width < max_width && width * 2 <= remaining)
                width *= 2;
            if(byte_addressed || word_vectors)
                width = remaining < 4 ? remaining : 4;
            if(size_t(written) >= capacity)
                ckc::raise_status(ROCKE_ERR_VALUE, "storage load output array is too small");
            const int displacement = byte_addressed ? consumed * element_bytes : consumed;
            auto* at
                = consumed ? rocke_b_add(b, offset, rocke_b_const_i32(b, displacement)) : offset;
            const int align = gcd(alignment_bytes, uint64_t(consumed) * element_bytes);
            auto* load_ptr = byte_addressed
                                 ? rocke_b_global_ptr_add(b, ptr, rocke_b_sext(b, at, rocke_i64()))
                                 : ptr;
            auto* load_at = byte_addressed ? rocke_b_const_i32(b, 0) : at;
            auto* value = width == 1
                              ? rocke_b_vector_splat(
                                    b, rocke_b_global_load(b, load_ptr, load_at, element, align), 1)
                              : rocke_b_global_load_vN(b, load_ptr, load_at, element, width, align);
            pieces[written++] = value;
            consumed += width;
            remaining -= width;
        }
        *piece_count = written;
        return ROCKE_OK;
    });
}
