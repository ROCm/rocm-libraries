// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Descriptor-driven transport. Mirrors helpers/mma_io.py in emission order. */
#include "rocke/helper_rocke.helpers.mma_io.h"

#include <limits.h>
#include <string.h>

#include "rocke/error_boundary.hpp"

const rocke_type_t* rocke_storage_ir_type(const char* dtype)
{
    const rocke_dtype_info_t* info = rocke_dtype_info(dtype);
    if(!info)
        return NULL;
    if(info->encoded_bits % 8 || strcmp(info->name, "e8m0") == 0 || strcmp(info->name, "e4m3") == 0
       || strcmp(info->name, "e5m3") == 0)
        return rocke_i8();
    return rocke_dtype_to_ir_type(info->name);
}

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
                                            const rocke_tensor_storage_t* storage,
                                            const rocke_matrix_fragment_layout_t* layout,
                                            const rocke_type_t* carrier_type)
{
    return ckc::guard_builder(b, [&]() -> rocke_value_t* {
        if(!rocke_i_live(b))
            return NULL;
        if(!storage || !storage->dtype || !layout || !carrier_type || !ptr || !row_base
           || !lane_group)
            ckc::raise_status(ROCKE_ERR_VALUE, "null matrix fragment argument");
        const auto& packing = layout->fragment;
        rocke_matrix_fragment_layout_t checked;
        if(!rocke_matrix_fragment_layout_init(&checked,
                                              &packing,
                                              layout->chunk_elements,
                                              layout->lane_groups,
                                              layout->lanes_per_group)
           || !packing.count || packing.count > INT_MAX || packing.carrier_count > INT_MAX)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid matrix fragment chunk layout");
        const rocke_type_t* unit_type = rocke_storage_ir_type(storage->dtype->name);
        if(!unit_type || ptr->type->kind != ROCKE_TYPE_PTR
           || !rocke_type_eq(ptr->type->pointee, unit_type))
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix pointer storage type mismatch");
        const uint64_t unit_bytes = rocke_dtype_info(unit_type->name)->encoded_bits / 8;
        if(storage->packing.element_bits != packing.packing.element_bits
           || storage->packing.slot_bits != packing.packing.slot_bits || storage->base_bit_offset)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "matrix fragment requires matching packing and a zero bit origin");
        const uint64_t span = packing.count * layout->lane_groups;
        if(k0 > storage->cols || span > storage->cols - k0)
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix fragment exceeds the packed row");
        uint64_t origin_bits, chunk_bytes;
        if(!rocke_bit_packing_offset(&storage->packing, k0, &origin_bits)
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
        if(origin_bytes / unit_bytes > INT_MAX || chunk_units * layout->lane_groups > INT_MAX)
            ckc::raise_status(ROCKE_ERR_VALUE, "matrix fragment offset exceeds i32 range");
        int alignment = gcd(gcd(storage->alignment_bytes, storage->row_stride_bytes),
                            gcd(chunk_bytes, origin_bytes));
        auto* lane_chunk = rocke_b_mul(b, lane_group, rocke_b_const_i32(b, chunk_units));
        auto* step_base = rocke_b_add(b, row_base, rocke_b_const_i32(b, origin_bytes / unit_bytes));
        /* Collect loads before concatenation to preserve Python SSA numbering. */
        rocke_value_t** chunks = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, size_t(packing.count) * sizeof(rocke_value_t*));
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
            uint64_t remaining = chunk_units, consumed = 0;
            const int max_width = unit_bytes == 4 ? 8 : 16;
            while(remaining)
            {
                int width = 1;
                while(width < max_width && uint64_t(width * 2) <= remaining)
                    width *= 2;
                auto* at
                    = consumed ? rocke_b_add(b, offset, rocke_b_const_i32(b, consumed)) : offset;
                int load_align = gcd(alignment, consumed * unit_bytes);
                auto* value
                    = width == 1 ? rocke_b_vector_splat(
                                       b, rocke_b_global_load(b, ptr, at, unit_type, load_align), 1)
                                 : rocke_b_global_load_vN(b, ptr, at, unit_type, width, load_align);
                chunks[num_chunks++] = value;
                consumed += width;
                remaining -= width;
            }
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

rocke_status_t rocke_h_pack_fragment_bits(rocke_ir_builder_t* b,
                                          rocke_load_bits_fn load_bits,
                                          void* ctx,
                                          const rocke_fragment_packing_t* fragment,
                                          rocke_value_t** words,
                                          size_t word_count)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        rocke_fragment_packing_t checked;
        if(!fragment || !load_bits || !words
           || !rocke_fragment_packing_init(&checked,
                                           &fragment->packing,
                                           fragment->count,
                                           fragment->carrier_bits,
                                           fragment->carrier_count)
           || word_count != fragment->carrier_count || fragment->count > INT_MAX)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid pattern packing arguments");
        const auto& f = *fragment;
        if(f.carrier_bits != 32 && f.carrier_bits != 64)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "IR pattern packing currently requires i32 or i64 carriers");
        const auto* word_type = f.carrier_bits == 64 ? rocke_i64() : rocke_i32();
        auto constant = [&](uint64_t value) {
            return f.carrier_bits == 64 ? rocke_b_const_i64(b, int64_t(value))
                                        : rocke_b_const_i32(b, int32_t(value));
        };
        for(size_t j = 0; j < word_count; ++j)
            words[j] = constant(0);
        for(uint64_t j = 0; j < f.count; ++j)
        {
            auto* pattern = load_bits(b, (int)j, ctx);
            if(!pattern
               || (!rocke_type_eq(pattern->type, rocke_i8())
                   && !rocke_type_eq(pattern->type, rocke_i16())
                   && !rocke_type_eq(pattern->type, rocke_i32())
                   && !rocke_type_eq(pattern->type, rocke_i64())))
                ckc::raise_status(ROCKE_ERR_VALUE,
                                  "pattern packing requires unsigned patterns in integer carriers");
            const int bits = rocke_dtype_info(pattern->type->name)->encoded_bits;
            if(bits < f.packing.element_bits)
                ckc::raise_status(ROCKE_ERR_VALUE, "pattern carrier is smaller than encoded width");
            if(bits < f.carrier_bits)
                pattern = rocke_b_zext(b, pattern, word_type);
            else if(bits > f.carrier_bits)
                ckc::raise_status(ROCKE_ERR_VALUE, "pattern carrier is wider than output carrier");
            uint64_t start = j * f.packing.slot_bits;
            int remaining = f.packing.element_bits, consumed = 0;
            while(remaining)
            {
                const uint64_t word = start / f.carrier_bits;
                const int shift = start % f.carrier_bits;
                const int take
                    = remaining < f.carrier_bits - shift ? remaining : f.carrier_bits - shift;
                auto* part = consumed ? rocke_b_lshr(b, pattern, constant(consumed)) : pattern;
                if(take < remaining)
                    part = rocke_b_land(b, part, constant((uint64_t(1) << take) - 1));
                words[word] = rocke_b_lor(b, words[word], rocke_b_shl(b, part, constant(shift)));
                start += take;
                consumed += take;
                remaining -= take;
            }
        }
        return ROCKE_OK;
    });
}
