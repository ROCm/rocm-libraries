// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Host packing fixtures and native authoring emission for test_mma_io.py. */
#include "rocke/storage.h"

#include <initializer_list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.mma_io.h"
#include "rocke/helper_rocke.helpers.quant.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/wmma_scale_internal.h"

#define CHECK(x)                                                       \
    do                                                                 \
    {                                                                  \
        if(!(x))                                                       \
        {                                                              \
            fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); \
            return 1;                                                  \
        }                                                              \
    } while(0)

static rocke_value_t* load_bits(rocke_ir_builder_t* b, int j, void* ctx)
{
    return rocke_b_global_load(b, (rocke_value_t*)ctx, rocke_b_const_i32(b, j), rocke_i8(), 1);
}

static int emit(const char* dtype, bool hip)
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "transport") == ROCKE_OK);
    const bool padded = strstr(dtype, "_padded") != NULL;
    if(padded)
        dtype = strcmp(dtype, "f16_padded") == 0    ? "f16"
                : strcmp(dtype, "bf16_padded") == 0 ? "bf16"
                                                    : "fp6";
    bool patterns = strncmp(dtype, "pack_", 5) == 0;
    const bool load96 = strncmp(dtype, "load96_", 7) == 0;
    const auto* unit = load96     ? (strcmp(dtype, "load96_i8") == 0    ? rocke_i8()
                                     : strcmp(dtype, "load96_f16") == 0 ? rocke_f16()
                                                                        : rocke_i32())
                       : patterns ? rocke_i8()
                                  : rocke_storage_ir_type(dtype);
    CHECK(unit);
    const bool typed = strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0;
    const auto* carrier = (typed || load96) ? unit : rocke_i32();
    if(strcmp(dtype, "pack_scale_bytes_i64") == 0)
        carrier = rocke_i64();
    auto* a = rocke_b_param(&b, "A", rocke_ptr_type(&b, unit, "global"), NULL);
    auto* o = rocke_b_param(&b, "O", rocke_ptr_type(&b, carrier, "global"), NULL);
    if(load96)
    {
        const int n = unit == rocke_i8() ? 12 : unit == rocke_f16() ? 6 : 3;
        auto* one = rocke_b_const_i32(&b, 1);
        auto* value = rocke_b_global_load_vN(&b, a, one, unit, n, 0);
        int shape[] = {n + 1};
        auto* smem = rocke_b_smem_alloc(&b, unit, shape, 1, "payload");
        for(int j = 0; j < n; ++j)
        {
            rocke_value_t* indices[] = {rocke_b_const_i32(&b, j + 1)};
            auto* element = rocke_b_vec_extract(&b, value, j);
            rocke_b_smem_store_vN(&b, smem, indices, 1, element, 1);
        }
        rocke_b_s_barrier_bare(&b);
        value = rocke_b_smem_load_vN(&b, smem, &one, 1, unit, n);
        auto* aligned = rocke_b_global_load_vN(&b, a, rocke_b_const_i32(&b, 0), unit, n, 16);
        rocke_value_t* vectors[] = {value, aligned};
        for(int k = 0; k < 2; ++k)
            for(int j = 0; j < n; ++j)
            {
                auto* index = rocke_b_const_i32(&b, k * n + j);
                auto* element = rocke_b_vec_extract(&b, vectors[k], j);
                rocke_b_global_store(&b, o, index, element, 12 / n);
            }
    }
    else if(patterns)
    {
        int bits = strcmp(dtype, "pack_fp6_cross_word") == 0 ? 6 : 8;
        int count = bits == 6 ? 16 : (strcmp(dtype, "pack_scale_bytes_i64") == 0 ? 8 : 4);
        rocke_bit_packing_t packing;
        rocke_fragment_packing_t fragment;
        CHECK(rocke_bit_packing_init(&packing, bits, 0));
        int words = bits == 6 ? 3 : 1;
        CHECK(rocke_fragment_packing_init(
            &fragment, &packing, count, carrier == rocke_i64() ? 64 : 32, words));
        rocke_value_t* values[3];
        CHECK(rocke_h_pack_fragment_bits(&b, load_bits, a, &fragment, values, words) == ROCKE_OK);
        for(int j = 0; j < words; ++j)
            rocke_b_global_store(&b, o, rocke_b_const_i32(&b, j), values[j], 4);
    }
    else
    {
        rocke_matrix_fragment_layout_t layout;
        if(typed)
        {
            rocke_bit_packing_t packing;
            rocke_fragment_packing_t fragment;
            CHECK(rocke_bit_packing_init(&packing, 16, 0));
            CHECK(rocke_fragment_packing_init(&fragment, &packing, 32, 16, 32));
            CHECK(rocke_matrix_fragment_layout_init(&layout, &fragment, 16, 2, 16));
        }
        else
            layout = rocke_scaled_matrix_layout(dtype, 16);
        auto* base = rocke_b_const_i32(&b, padded ? (typed ? 129 : 97) : 0);
        auto* thread = rocke_b_thread_id_x(&b);
        auto* lane = rocke_b_mod(&b, thread, rocke_b_const_i32(&b, 32));
        auto* group = rocke_b_div(&b, lane, rocke_b_const_i32(&b, 16));
        auto* value = rocke_h_load_matrix_fragment(
            &b, a, base, group, 0, dtype, &layout, carrier, padded ? (typed ? 2 : 1) : 16);
        CHECK(value && rocke_ir_builder_ok(&b));
        for(int j = 0; j < value->type->count; ++j)
        {
            auto* index = rocke_b_const_i32(&b, j);
            auto* element = rocke_b_vec_extract(&b, value, j);
            rocke_b_global_store(&b, o, index, element, typed ? 2 : 4);
        }
    }
    CHECK(rocke_ir_builder_ok(&b));
    if(hip)
    {
        rocke_strbuf_t text;
        CHECK(rocke_strbuf_init(&text, 256) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx1250";
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &text) == ROCKE_OK);
        fputs(rocke_strbuf_cstr(&text), stdout);
        rocke_strbuf_free(&text);
    }
    else
    {
        char* text = NULL;
        CHECK(rocke_ir_serialize(b.kernel, &text) == ROCKE_OK);
        fputs(text, stdout);
        free(text);
    }
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_fragment_inputs()
{
    const struct
    {
        const char* dtype;
        const char* layout_dtype;
        uint64_t k0;
        int alignment;
        const char* error;
    } cases[] = {{"fp6", "fp6", 0, 3, "positive power of two"},
                 {"fp4", "fp6", 0, 1, "packing width mismatch"},
                 {"fp4", "fp4", 1, 1, "aligned to pointer storage units"},
                 {"fp8", "fp8", 0, 1, "pointer storage type mismatch"},
                 {"bf8", "bf8", 0, 1, "pointer storage type mismatch"}};
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_fragment") == ROCKE_OK);
        auto* ptr = rocke_b_param(&b, "A", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        auto* zero = rocke_b_const_i32(&b, 0);
        const auto layout = rocke_scaled_matrix_layout(c.layout_dtype, 16);
        const int before = b.kernel->body->num_ops;
        CHECK(!rocke_h_load_matrix_fragment(
            &b, ptr, zero, zero, c.k0, c.dtype, &layout, rocke_i32(), c.alignment));
        CHECK(rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE);
        CHECK(strstr(rocke_ir_builder_error(&b), c.error));
        CHECK(b.kernel->body->num_ops == before);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_fragment_offset_bounds()
{
    const struct
    {
        int count, groups;
        uint64_t k0;
        bool valid;
    } cases[] = {{64, 100000000, 0, false},
                 {64, 1 << 25, 0, true},
                 {64, 1 << 25, 1, false},
                 {64, 2, (uint64_t(1) << 31) - 128, true},
                 {64, 2, (uint64_t(1) << 31) - 127, false},
                 {16, 100000000, 1500000000, false},
                 {64, INT32_MAX, 0, false}};
    for(const char* dtype : {"e8m0", "fp6", "f16"})
    {
        const int slot_bits = strcmp(dtype, "f16") == 0 ? 16 : 8;
        for(const auto& c : cases)
        {
            rocke_bit_packing_t packing;
            CHECK(rocke_bit_packing_init(
                &packing, strcmp(dtype, "fp6") == 0 ? 6 : slot_bits, slot_bits));
            rocke_fragment_packing_t fragment;
            CHECK(rocke_fragment_packing_init(
                &fragment, &packing, c.count, 32, c.count * slot_bits / 32));
            rocke_matrix_fragment_layout_t layout;
            CHECK(rocke_matrix_fragment_layout_init(&layout, &fragment, 16, c.groups, 1));
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "offset_bounds") == ROCKE_OK);
            auto* ptr = rocke_b_param(
                &b, "A", rocke_ptr_type(&b, rocke_storage_ir_type(dtype), "global"), NULL);
            auto* zero = rocke_b_const_i32(&b, 0);
            const int before = b.kernel->body->num_ops;
            auto* value = rocke_h_load_matrix_fragment(
                &b, ptr, zero, zero, c.k0, dtype, &layout, rocke_i32(), 1);
            if(c.valid)
                CHECK(value && rocke_ir_builder_ok(&b));
            else
            {
                CHECK(!value && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE);
                CHECK(strstr(rocke_ir_builder_error(&b), "offset exceeds i32 range"));
                CHECK(b.kernel->body->num_ops == before);
            }
            rocke_ir_builder_free(&b);
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--emit") == 0)
        return emit(argv[2], false);
    if(argc == 3 && strcmp(argv[1], "--hip") == 0)
        return emit(argv[2], true);
    if(argc == 3 && strcmp(argv[1], "--parse") == 0)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "parse") == ROCKE_OK);
        rocke_kernel_def_t* kernel = NULL;
        CHECK(rocke_ir_parse(argv[2], &b, &kernel) == ROCKE_OK);
        char* text = NULL;
        CHECK(rocke_ir_serialize(kernel, &text) == ROCKE_OK);
        fputs(text, stdout);
        free(text);
        rocke_ir_builder_free(&b);
        return 0;
    }
    if(argc == 3 && strcmp(argv[1], "--quant-error") == 0)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "quant_error") == ROCKE_OK);
        try
        {
            rocke_b_quant_ir_type(&b, argv[2]);
            CHECK(false);
        }
        catch(const ckc::Error& error)
        {
            CHECK(error.code() == ROCKE_ERR_VALUE);
            fputs(error.what(), stdout);
        }
        rocke_ir_builder_free(&b);
        return 0;
    }
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "unsupported_packing") == ROCKE_OK);
        rocke_bit_packing_t packing;
        rocke_fragment_packing_t fragment;
        rocke_value_t* words[2];
        CHECK(rocke_bit_packing_init(&packing, 40, 0));
        CHECK(rocke_fragment_packing_init(&fragment, &packing, 2, 64, 2));
        CHECK(rocke_h_pack_fragment_bits(&b, load_bits, NULL, &fragment, words, 2)
              == ROCKE_ERR_VALUE);
        CHECK(strstr(rocke_ir_builder_error(&b), "encoded fields of at most 32 bits"));
        CHECK(b.kernel->body->num_ops == 0);
        rocke_ir_builder_free(&b);
    }
    CHECK(test_fragment_inputs() == 0);
    CHECK(test_fragment_offset_bounds() == 0);
    CHECK(rocke_dtype_info("e4m3") == rocke_dtype_info("fp8e4m3"));
    CHECK(rocke_dtype_to_ir_type("e4m3") == rocke_fp8e4m3());
    CHECK(rocke_storage_ir_type("e4m3") == rocke_fp8e4m3());
    CHECK(rocke_quant_ir_type("e4m3") == rocke_fp8e4m3());
    CHECK(rocke_scalar_by_name("e4m3") == rocke_fp8e4m3());
    CHECK(rocke_dtype_info("e5m3") != rocke_dtype_info("bf8e5m2"));
    for(const char* elem_type : {"unknown", "fp4e2m1"})
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_element") == ROCKE_OK);
        auto* ptr = rocke_b_param(&b, "A", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        auto* value = rocke_b_global_load_vN(&b, ptr, rocke_b_const_i32(&b, 0), rocke_i8(), 16, 16);
        CHECK(value);
        rocke_attr_set_str(&b, &value->op->attrs, "elem_type", elem_type);
        rocke_strbuf_t text;
        CHECK(rocke_strbuf_init(&text, 256) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx1250";
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &text) == ROCKE_ERR_KEY);
        rocke_strbuf_free(&text);
        rocke_ir_builder_free(&b);
    }
    for(int alignment : {0, -1, -16, 3, 24})
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_alignment") == ROCKE_OK);
        auto* ptr = rocke_b_param(&b, "A", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        auto* value = rocke_b_global_load_vN(&b, ptr, rocke_b_const_i32(&b, 0), rocke_i8(), 16, 16);
        CHECK(value);
        // Exercise raw IR, including alignments normalized by the builder.
        rocke_attr_set_int(&b, &value->op->attrs, "align", alignment);
        rocke_strbuf_t text;
        CHECK(rocke_strbuf_init(&text, 256) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx1250";
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &text) == ROCKE_ERR_VALUE);
        rocke_strbuf_free(&text);
        rocke_ir_builder_free(&b);
    }
    const auto layout = rocke_scaled_matrix_layout("fp6", 16);
    uint64_t row, k;
    CHECK(rocke_matrix_fragment_coord(&layout, 31, 63, &row, &k));
    CHECK(row == 15 && k == 127);
    // C callers can construct or mutate public descriptors without init().
    for(int defect = 0; defect < 6; ++defect)
    {
        auto invalid = layout;
        switch(defect)
        {
        case 0:
            invalid.fragment.packing.element_bits = 0;
            break;
        case 1:
            invalid.fragment.packing.slot_bits = 4;
            break;
        case 2:
            invalid.fragment.carrier_bits = 7;
            break;
        case 3:
            invalid.fragment.carrier_count = 1;
            break;
        case 4:
            invalid.chunk_elements = 3;
            break;
        case 5:
            invalid.chunk_elements = 2;
            break; // 12 bits: not byte aligned.
        }
        CHECK(!rocke_matrix_fragment_coord(&invalid, 0, 0, &row, &k));
    }
    for(int bits : {4, 6, 8, 16, 32, 64})
    {
        rocke_bit_packing_t packing;
        CHECK(rocke_bit_packing_init(&packing, bits, 0));
        uint64_t patterns[256], decoded[256];
        size_t count = bits <= 8 ? size_t(1) << bits : 4;
        for(size_t i = 0; i < count; ++i)
            patterns[i] = bits <= 8 ? i : (i == 3 ? UINT64_MAX >> (64 - bits) : i);
        for(uint64_t offset : {uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(9)})
        {
            uint64_t size;
            CHECK(rocke_bit_packing_bytes(&packing, count, offset, &size));
            uint8_t data[2049];
            memset(data, 0xA5, sizeof(data));
            CHECK(rocke_bit_pack(&packing, patterns, count, offset, data, size));
            CHECK(data[size] == 0xA5); // No tail overrun.
            CHECK(!rocke_bit_unpack(&packing, data, size - 1, count, offset, decoded));
            CHECK(rocke_bit_unpack(&packing, data, size, count, offset, decoded));
            CHECK(memcmp(patterns, decoded, count * sizeof(uint64_t)) == 0);
        }
    }
    rocke_bit_packing_t six;
    CHECK(rocke_bit_packing_init(&six, 6, 0));
    uint64_t values, carriers;
    CHECK(rocke_bit_packing_group(&six, 32, &values, &carriers));
    CHECK(values == 16 && carriers == 3);
    uint64_t patterns[16] = {}, words[3];
    patterns[5] = patterns[10] = 63;
    rocke_fragment_packing_t fragment;
    CHECK(rocke_fragment_packing_init(&fragment, &six, 16, 32, 3));
    CHECK(rocke_fragment_pack(&fragment, patterns, 16, words, 3));
    CHECK(words[0] == 0xC0000000 && words[1] == 0xF000000F && words[2] == 3);
    CHECK(!rocke_bit_packing_bytes(&six, UINT64_MAX, 0, &values));
    CHECK(!rocke_fragment_packing_init(&fragment, &six, 16, 32, 2));
    for(const char* dtype : {"fp4", "fp6", "bf6", "e8m0", "e4m3", "e5m3"})
    {
        const auto* logical = rocke_dtype_to_ir_type(dtype);
        CHECK(logical && logical != rocke_i8());
        CHECK(rocke_scalar_by_name(logical->name) == logical);
        if(dtype[0] != 'e')
            CHECK(rocke_quant_ir_type(dtype) == logical);
        rocke_ir_builder_t b, parsed;
        CHECK(rocke_ir_builder_init(&b, "types") == ROCKE_OK);
        CHECK(rocke_ir_builder_init(&parsed, "parsed") == ROCKE_OK);
        rocke_b_param(&b, "pattern", logical, NULL);
        char* text = NULL;
        CHECK(rocke_ir_serialize(b.kernel, &text) == ROCKE_OK);
        rocke_kernel_def_t* kernel = NULL;
        CHECK(rocke_ir_parse(text, &parsed, &kernel) == ROCKE_OK);
        char* again = NULL;
        CHECK(rocke_ir_serialize(kernel, &again) == ROCKE_OK);
        CHECK(strcmp(text, again) == 0);
        free(text);
        free(again);
        rocke_ir_builder_free(&b);
        rocke_ir_builder_free(&parsed);
    }
    return 0;
}
