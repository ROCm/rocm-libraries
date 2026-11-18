// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Canonical encoded-pattern placement; mirrors helpers/packing.py. */
#include "rocke/helper_rocke.helpers.packing.h"

#include <limits.h>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"

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
        if(f.packing.element_bits > 32)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "IR pattern packing supports encoded fields of at most 32 bits");
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
