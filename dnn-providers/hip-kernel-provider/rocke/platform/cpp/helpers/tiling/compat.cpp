// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Compatibility raw-chunk adapter. Mirrors helpers/tiling/compat.py. */
#include "rocke/tiling_compat.h"

#include <limits.h>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"
#include "rocke/ir_storage.h"
#include "rocke/tiling_transport_internal.h"

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

rocke_value_t* rocke_h_load_matrix_fragment(rocke_ir_builder_t* b,
                                            rocke_value_t* ptr,
                                            rocke_value_t* row_base,
                                            rocke_value_t* lane_group,
                                            uint64_t k0,
                                            const char* dtype,
                                            const rocke_matrix_fragment_layout_t* layout,
                                            const rocke_type_t* carrier_type,
                                            int alignment_bytes)
{
    return ckc::guard_builder(b, [&]() -> rocke_value_t* {
        if(!rocke_i_live(b))
            return NULL;
        if(!dtype || !layout || !carrier_type || !ptr || !row_base || !lane_group)
            ckc::raise_status(ROCKE_ERR_VALUE, "null matrix fragment argument");
        if(alignment_bytes <= 0 || (alignment_bytes & (alignment_bytes - 1)))
            ckc::raise_status(ROCKE_ERR_VALUE, "alignment_bytes must be a positive power of two");
        const auto& packing = layout->fragment;
        rocke_matrix_fragment_layout_t checked;
        if(!rocke_matrix_fragment_layout_init(&checked,
                                              &packing,
                                              layout->chunk_elements,
                                              layout->lane_groups,
                                              layout->lanes_per_group)
           || !packing.count || packing.count > INT_MAX || packing.carrier_count > INT_MAX)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid matrix fragment chunk layout");
        const rocke_type_t* unit_type = rocke_storage_ir_type(dtype);
        if(!unit_type || ptr->type->kind != ROCKE_TYPE_PTR
           || !rocke_type_eq(ptr->type->pointee, unit_type))
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix pointer storage type mismatch");
        const uint64_t unit_bytes = rocke_dtype_info(unit_type->name)->encoded_bits / 8;
        if(rocke_dtype_info(dtype)->encoded_bits != packing.packing.element_bits)
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix dtype and packing width mismatch");
        uint64_t origin_bits, chunk_bytes;
        if(!rocke_bit_packing_offset(&packing.packing, k0, &origin_bits)
           || !rocke_bit_packing_bytes(&packing.packing, layout->chunk_elements, 0, &chunk_bytes)
           || origin_bits % (8 * unit_bytes) || chunk_bytes % unit_bytes)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "matrix fragment is not aligned to pointer storage units");
        const rocke_dtype_info_t* carrier_info = rocke_dtype_info(carrier_type->name);
        if(!carrier_info || carrier_info->encoded_bits != packing.carrier_bits)
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix carrier type width mismatch");
        const uint64_t payload_bits = packing.count * packing.packing.slot_bits;
        if(payload_bits % packing.carrier_bits)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "matrix fragment payload must occupy whole carriers");
        const int live_carriers = payload_bits / packing.carrier_bits;
        const int padding = packing.carrier_count - live_carriers;
        if(padding && !rocke_type_eq(carrier_type, rocke_i32()))
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "padded matrix fragments currently require i32 carriers");
        const uint64_t chunk_units = chunk_bytes / unit_bytes;
        const uint64_t origin_bytes = origin_bits / 8;
        // Bound the last loaded unit without overflowing intermediate products.
        if(origin_bytes / unit_bytes > INT_MAX
           || chunk_units > uint64_t(INT_MAX) / layout->lane_groups
           || packing.count / layout->chunk_elements
                  > (uint64_t(INT_MAX) + 1 - origin_bytes / unit_bytes) / chunk_units
                        / layout->lane_groups)
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix fragment offset exceeds i32 range");
        int alignment = gcd(alignment_bytes, gcd(chunk_bytes, origin_bytes));
        // Use 2--4-word vectors; retain storage-element loads for a one-word tail.
        const bool word_loads = unit_bytes == 1 && rocke_type_eq(carrier_type, rocke_i32())
                                && chunk_bytes % 4 == 0 && chunk_bytes % 16 != 4;
        const auto* load_type = word_loads ? rocke_i32() : unit_type;
        const int load_step = word_loads ? 4 : 1;
        auto* lane_chunk = rocke_b_mul(b, lane_group, rocke_b_const_i32(b, chunk_units));
        auto* step_base = rocke_b_add(b, row_base, rocke_b_const_i32(b, origin_bytes / unit_bytes));
        /* Collect loads before concatenation to preserve Python SSA numbering.
         * Each load consumes at least one storage unit, including padded slots. */
        rocke_value_t** chunks = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, size_t(payload_bits / (8 * unit_bytes)) * sizeof(rocke_value_t*));
        if(!chunks)
            ckc::raise_status(ROCKE_ERR_OOM, "matrix fragment allocation failed");
        int num_chunks = 0;
        for(uint64_t j = 0; j < packing.count / layout->chunk_elements; ++j)
        {
            auto* offset = rocke_b_add(
                b,
                rocke_b_add(
                    b, step_base, rocke_b_const_i32(b, j * layout->lane_groups * chunk_units)),
                lane_chunk);
            int piece_count;
            if(rocke_h_load_storage_chunks(b,
                                           ptr,
                                           offset,
                                           load_type,
                                           int(chunk_units / load_step),
                                           alignment,
                                           word_loads,
                                           false,
                                           chunks + num_chunks,
                                           size_t(payload_bits / (8 * unit_bytes)) - num_chunks,
                                           &piece_count)
               != ROCKE_OK)
                return NULL;
            num_chunks += piece_count;
        }
        auto* payload = chunks[0];
        for(int i = 1; i < num_chunks; ++i)
            payload = rocke_b_vec_concat(b, payload, chunks[i]);
        const auto* payload_type = rocke_vector_type(b, carrier_type, live_carriers);
        if(!rocke_type_eq(payload->type, payload_type))
            payload = rocke_b_bitcast(b, payload, payload_type);
        if(padding)
            payload = rocke_b_vec_concat(
                b, payload, rocke_b_vector_splat(b, rocke_b_const_i32(b, 0), padding));
        return payload;
    });
}
