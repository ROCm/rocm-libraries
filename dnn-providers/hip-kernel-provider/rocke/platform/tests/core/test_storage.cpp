// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Host packing fixtures and native authoring emission for test_mma_io.py. */
#include "rocke/storage.h"

#include <initializer_list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    const auto* unit = patterns ? rocke_i8() : rocke_storage_ir_type(dtype);
    CHECK(unit);
    const bool typed = strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0;
    const auto* carrier = typed ? unit : rocke_i32();
    if(strcmp(dtype, "pack_scale_bytes_i64") == 0)
        carrier = rocke_i64();
    auto* a = rocke_b_param(&b, "A", rocke_ptr_type(&b, unit, "global"), NULL);
    auto* o = rocke_b_param(&b, "O", rocke_ptr_type(&b, carrier, "global"), NULL);
    if(patterns)
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
        rocke_tensor_storage_t storage;
        CHECK(rocke_tensor_storage_init(
            &storage, dtype, 16, 128, padded ? (typed ? 258 : 97) : UINT64_MAX, 0, 0, 16));
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
        auto* value
            = rocke_h_load_matrix_fragment(&b, a, base, group, 0, &storage, &layout, carrier);
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

static int test_storage_validation()
{
    rocke_tensor_storage_t storage;
    CHECK(rocke_tensor_storage_init(&storage, "fp6", 16, 128, 97, 0, 0, 16));
    const auto layout = rocke_scaled_matrix_layout("fp6", 16);
    for(int defect = 0; defect < 10; ++defect)
    {
        auto invalid = storage;
        switch(defect)
        {
        case 0:
            invalid.dtype = NULL;
            break;
        case 1:
            invalid.dtype = rocke_dtype_info("fp8");
            break;
        case 2:
            invalid.packing.element_bits = 0;
            break;
        case 3:
            invalid.packing.slot_bits = 0;
            break;
        case 4:
            invalid.packing.slot_bits = 4;
            break;
        case 5:
            invalid.row_stride_bytes = 95;
            break;
        case 6:
            invalid.alignment_bytes = 0;
            break;
        case 7:
            invalid.alignment_bytes = 3;
            break;
        case 8:
            invalid.rows = UINT64_MAX;
            break;
        case 9:
            invalid.base_bit_offset = UINT64_MAX;
            break;
        }
        uint64_t bytes;
        int shift;
        CHECK(!rocke_tensor_storage_bytes(&invalid, &bytes));
        CHECK(!rocke_tensor_storage_address(&invalid, 0, 0, &bytes, &shift));
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_storage") == ROCKE_OK);
        auto* ptr = rocke_b_param(&b, "A", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        auto* zero = rocke_b_const_i32(&b, 0);
        CHECK(
            !rocke_h_load_matrix_fragment(&b, ptr, zero, zero, 0, &invalid, &layout, rocke_i32()));
        CHECK(!rocke_ir_builder_ok(&b));
        rocke_ir_builder_free(&b);
        if(defect < 8)
        {
            invalid.rows = 0;
            CHECK(!rocke_tensor_storage_bytes(&invalid, &bytes));
        }
    }
    for(const char* dtype : {"f16", "bf16", "fp8", "bf8"})
    {
        const auto* unit = rocke_storage_ir_type(dtype);
        CHECK(unit && !rocke_type_eq(unit, rocke_i8()));
        const bool typed = strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0;
        rocke_tensor_storage_t direct = {rocke_dtype_info(dtype),
                                         2,
                                         128,
                                         typed ? 257u : 129u,
                                         0,
                                         1,
                                         {typed ? 16 : 8, typed ? 16 : 8}};
        uint64_t bytes;
        int shift;
        CHECK(rocke_tensor_storage_bytes(&direct, &bytes));
        CHECK(rocke_tensor_storage_address(&direct, 1, 0, &bytes, &shift));
        CHECK(bytes == direct.row_stride_bytes && shift == 0);
        rocke_matrix_fragment_layout_t fragment;
        if(typed)
        {
            rocke_fragment_packing_t packing;
            CHECK(rocke_fragment_packing_init(&packing, &direct.packing, 32, 16, 32));
            CHECK(rocke_matrix_fragment_layout_init(&fragment, &packing, 16, 2, 16));
        }
        else
            fragment = rocke_scaled_matrix_layout(dtype, 16);
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "invalid_pointer_units") == ROCKE_OK);
        auto* ptr
            = rocke_b_param(&b, "A", rocke_ptr_type(&b, typed ? unit : rocke_i8(), "global"), NULL);
        auto* zero = rocke_b_const_i32(&b, 0);
        CHECK(!rocke_h_load_matrix_fragment(
            &b, ptr, zero, zero, 0, &direct, &fragment, typed ? unit : rocke_i32()));
        CHECK(!rocke_ir_builder_ok(&b));
        rocke_ir_builder_free(&b);
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--emit") == 0)
        return emit(argv[2], false);
    if(argc == 3 && strcmp(argv[1], "--hip") == 0)
        return emit(argv[2], true);
    CHECK(test_storage_validation() == 0);
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
    rocke_tensor_storage_t storage;
    CHECK(rocke_tensor_storage_init(&storage, "bf6", 3, 5, 8, 0, 3, 1));
    CHECK(rocke_tensor_storage_bytes(&storage, &values) && values == 21);
    int shift;
    CHECK(rocke_tensor_storage_address(&storage, 2, 4, &values, &shift));
    CHECK(values == 19 && shift == 3);
    CHECK(!rocke_tensor_storage_address(&storage, 3, 0, &values, &shift));
    CHECK(!rocke_tensor_storage_init(&storage, "fp6", 3, 5, 3, 0, 0, 1));
    CHECK(!rocke_tensor_storage_init(&storage, "fp8", UINT64_MAX, 3, 3, 0, 0, 1));
    CHECK(rocke_tensor_storage_init(&storage, "fp4", 3, 0, 0, 0, 7, 1));
    CHECK(rocke_tensor_storage_bytes(&storage, &values) && values == 0);
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
