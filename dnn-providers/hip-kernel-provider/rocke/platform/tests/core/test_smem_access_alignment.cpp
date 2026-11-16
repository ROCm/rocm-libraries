// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_smem_access_alignment.cpp -- the C++ engine's LDS alignment claims,
 * through the C API.
 *
 * Every smem allocation lives in one 16-byte aligned pool global at a byte
 * offset the pool layout chooses, and each access through a segment prints
 * `align N`. The claim must hold: N may not exceed what the segment's offset
 * guarantees. This is the C++ half of tests/core/test_smem_access_alignment.py
 * (which compares the two engines byte for byte only where the pybind engine
 * is built): the kernels are built with the C builders and lowered by the C++
 * engine, and each case pins the segment offset and the claim the Python test
 * pins for the same kernel. A vector store also claims a width: the builder
 * and the verifier refuse a stored value without that many lanes. The f32
 * epilogue load and store refuse a tile or value that does not hold f32, and
 * smem_store_f16 a value that does not hold f16.
 */

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

namespace
{

constexpr const char* kArch = "gfx942";

/* The outcome of lowering one kernel: the status, and the IR text or the
 * refusal message. */
struct Lowering
{
    rocke_status_t status;
    std::string text;
    std::string error;
};

/* Lower the kernel `builder` holds for gfx942 at llvm20, then free the builder. */
Lowering lower_and_free(rocke_ir_builder_t* builder)
{
    Lowering lowering{ROCKE_ERR_VALUE, "", ""};
    if(!rocke_ir_builder_ok(builder))
    {
        lowering.error = "the builder refused the kernel";
        rocke_ir_builder_free(builder);
        return lowering;
    }
    char* llvm_text = NULL;
    char err[ROCKE_ERR_MSG_CAP] = {0};
    lowering.status = rocke_lower_kernel_to_llvm_ex(
        builder->kernel, ROCKE_LLVM_FLAVOR_LLVM20, kArch, &llvm_text, err, sizeof(err));
    if(llvm_text != NULL)
    {
        lowering.text = llvm_text;
        std::free(llvm_text);
    }
    lowering.error = err;
    rocke_ir_builder_free(builder);
    return lowering;
}

/* The IR lines that contain `needle`. */
std::vector<std::string> lines_with(const std::string& text, const std::string& needle)
{
    std::vector<std::string> found;
    size_t start = 0;
    while(start < text.size())
    {
        size_t end = text.find('\n', start);
        if(end == std::string::npos)
            end = text.size();
        const std::string line = text.substr(start, end - start);
        if(line.find(needle) != std::string::npos)
            found.push_back(line);
        start = end + 1;
    }
    return found;
}

/* True when there is at least one LDS (addrspace(3)) access line containing
 * `access`, and every such line ends with `, align <claim>`. */
bool every_lds_access_claims(const std::string& text, const std::string& access, int claim)
{
    const std::string suffix = ", align " + std::to_string(claim);
    int lds_accesses = 0;
    for(const std::string& line : lines_with(text, access))
    {
        if(line.find("ptr addrspace(3)") == std::string::npos)
            continue;
        lds_accesses++;
        if(line.size() < suffix.size()
           || line.compare(line.size() - suffix.size(), suffix.size(), suffix) != 0)
            return false;
    }
    return lds_accesses > 0;
}

/* The byte offsets of the pool segments placed after byte 0, in order. */
std::vector<int> segment_offsets(const std::string& text)
{
    std::vector<int> offsets;
    for(const std::string& line :
        lines_with(text, "= getelementptr inbounds i8, ptr addrspace(3) @smem_pool."))
    {
        offsets.push_back(std::atoi(line.substr(line.rfind(" i32 ") + 5).c_str()));
    }
    return offsets;
}

/* Rewrites the `align` attribute of the kernel's tile.smem_store_vN, as
 * serialized or hand-written IR may carry it. */
using AlignOverwrite = void (*)(rocke_ir_builder_t*, rocke_attr_map_t*);

/* A one-element f16 buffer, then an f32 buffer written and read as
 * <`lanes` x float>. The f16 store comes before and after the vector traffic,
 * so both buffers are live together and the f32 one cannot share offset 0.
 * Mirrors _f16_then_f32x4 and _f16_then_f32x8 in test_smem_access_alignment.py. */
Lowering lower_f16_then_f32_vector(int lanes, AlignOverwrite overwrite_align)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, lanes == 4 ? "lds_align_f16_f32x4" : "lds_align_f16_f32x8")
       != ROCKE_OK)
        return Lowering{ROCKE_ERR_OOM, "", "builder init failed"};
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int float_shape[] = {2 * lanes};
    rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
    rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
    rocke_value_t* half_value = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_value_t* loaded = rocke_b_global_load_vN(&b, out, zero, rocke_f32(), lanes, 4 * lanes);
    rocke_value_t* const at_lanes[] = {rocke_b_const_i32(&b, lanes)};
    rocke_b_smem_store_vN(&b, float_buf, at_lanes, 1, loaded, lanes);
    rocke_value_t* reloaded = rocke_b_smem_load_vN(&b, float_buf, at_zero, 1, rocke_f32(), lanes);
    rocke_b_global_store_vN(&b, out, zero, reloaded, lanes, 4 * lanes);
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_b_ret(&b);
    if(overwrite_align != nullptr)
    {
        rocke_region_t* body = b.kernel->body;
        for(int i = 0; i < body->num_ops; i++)
        {
            if(body->ops[i]->opcode == ROCKE_OP_TILE_SMEM_STORE_VN)
                overwrite_align(&b, &body->ops[i]->attrs);
        }
    }
    return lower_and_free(&b);
}

/* The f32 segment after a 2-byte f16 buffer lands at byte 16, not 4, so the
 * <4 x float> accesses' `align 16` holds. */
int test_vector_access_moves_its_segment_to_the_vector_alignment()
{
    const Lowering lowering = lower_f16_then_f32_vector(4, nullptr);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(segment_offsets(lowering.text) == std::vector<int>({16}));
    CHECK(every_lds_access_claims(lowering.text, "store <4 x float> %", 16));
    CHECK(every_lds_access_claims(lowering.text, "= load <4 x float>, ptr addrspace(3)", 16));
    return 0;
}

/* A 32-byte <8 x float> access claims the 16-byte pool alignment: no segment
 * can promise more than the pool it lives in. */
int test_access_wider_than_the_pool_claims_the_pool_alignment()
{
    const Lowering lowering = lower_f16_then_f32_vector(8, nullptr);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(segment_offsets(lowering.text) == std::vector<int>({16}));
    CHECK(every_lds_access_claims(lowering.text, "store <8 x float> %", 16));
    CHECK(every_lds_access_claims(lowering.text, "= load <8 x float>, ptr addrspace(3)", 16));
    return 0;
}

/* Any power of two above the pool alignment is capped to 16, including values
 * too wide for a 32-bit integer: the cap applies before the value is
 * narrowed, or 2**32 would wrap to 0 and print `align 0`. */
int test_a_valid_store_alignment_wider_than_the_pool_is_still_capped()
{
    const AlignOverwrite overwrites[] = {
        [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
            rocke_attr_set_int(b, attrs, "align", int64_t(1) << 32);
        },
        [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
            rocke_attr_set_int(b, attrs, "align", int64_t(1) << 62);
        },
    };
    for(const AlignOverwrite overwrite : overwrites)
    {
        const Lowering lowering = lower_f16_then_f32_vector(8, overwrite);
        CHECK(lowering.status == ROCKE_OK);
        CHECK(every_lds_access_claims(lowering.text, "store <8 x float> %", 16));
    }
    return 0;
}

/* A store `align` that is not a positive power of two is refused with
 * ROCKE_ERR_VALUE, not capped to the 16-byte pool alignment (24 would
 * otherwise print as a valid 16). */
int test_store_alignment_not_a_power_of_two_is_refused()
{
    const AlignOverwrite align_24 = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_int(b, attrs, "align", 24);
    };
    const Lowering lowering = lower_f16_then_f32_vector(4, align_24);
    CHECK(lowering.status == ROCKE_ERR_VALUE);
    CHECK(lowering.text.empty());
    CHECK(lowering.error
          == "tile.smem_store_vN failed - align=24, expected a positive power of two");
    return 0;
}

/* A store `align` that is not an integer is the wrong kind of value: it is
 * refused with ROCKE_ERR_TYPE, never read as a missing attribute (which would
 * fall back to the vector width). */
int test_store_alignment_not_an_integer_is_refused()
{
    const AlignOverwrite align_true = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_bool(b, attrs, "align", true);
    };
    const Lowering bool_lowering = lower_f16_then_f32_vector(4, align_true);
    CHECK(bool_lowering.status == ROCKE_ERR_TYPE);
    CHECK(bool_lowering.text.empty());
    CHECK(bool_lowering.error == "tile.smem_store_vN failed - align=bool, expected an integer");
    const AlignOverwrite align_float = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_float(b, attrs, "align", 24.0);
    };
    const Lowering float_lowering = lower_f16_then_f32_vector(4, align_float);
    CHECK(float_lowering.status == ROCKE_ERR_TYPE);
    CHECK(float_lowering.text.empty());
    CHECK(float_lowering.error == "tile.smem_store_vN failed - align=float, expected an integer");
    return 0;
}

/* A one-element f16 buffer, then an f32 buffer written at element 1 by one
 * whole-value tile.smem_store of a <lanes x float> value. smem_store_vN
 * refuses a vector whose lanes are not its n, so the kernel is serialized IR,
 * as hand-written IR may still carry it. Mirrors _f16_then_one_f32_vector_store
 * in the Python test. */
std::string f16_then_one_f32_vector_store(int lanes)
{
    const std::string lane_count = std::to_string(lanes);
    const std::string vector = "vec<f32x" + lane_count + ">";
    return "ckdsl.ir v1\n"
           "kernel @lds_align_f16_one_f32x"
           + lane_count
           + " {\n"
             "  params {\n"
             "    %out : ptr<f32,global>\n"
             "  }\n"
             "  region @entry {\n"
             "    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
             "    %h2 : smem<f16, [1]> = tile.smem_alloc (  )\n"
             "    %f3 : smem<f32, [12]> = tile.smem_alloc (  )\n"
             "    %c4 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }\n"
             "    %t5 : f16 = arith.trunc_f32_to_f16 ( %c4 )\n"
             "    tile.smem_store ( %h2, %c1, %t5 ) { elem_type = s:\"f16\", rank = i:1 }\n"
             "    %v6 : "
           + vector
           + " = memref.global_load_vN ( %out, %c1 ) { align = i:4, elem_type = s:\"f32\", "
             "vec = i:"
           + lane_count
           + " }\n"
             "    %c7 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n"
             "    tile.smem_store ( %f3, %c7, %v6 ) { elem_type = s:\""
           + vector
           + "\", rank = i:1 }\n"
             "    tile.smem_store ( %h2, %c1, %t5 ) { elem_type = s:\"f16\", rank = i:1 }\n"
             "    cf.return (  )\n"
             "  }\n"
             "}\n";
}

/* The scalar store records no align, so a vector stored whole claims only its
 * 4-byte element, whatever its width, and its segment stays at byte 4.
 * Mirrors test_a_whole_value_store_claims_element_alignment. */
int test_a_whole_value_store_claims_element_alignment()
{
    for(const int lanes : {3, 4})
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "parsed") == ROCKE_OK);
        rocke_kernel_def_t* kernel = NULL;
        CHECK(rocke_ir_parse(f16_then_one_f32_vector_store(lanes).c_str(), &b, &kernel)
              == ROCKE_OK);
        const Lowering lowering = lower_and_free(&b);
        CHECK(lowering.status == ROCKE_OK);
        CHECK(segment_offsets(lowering.text) == std::vector<int>({4}));
        CHECK(every_lds_access_claims(
            lowering.text, "store <" + std::to_string(lanes) + " x float> %", 4));
    }
    return 0;
}

/* Serialized IR loading a vector whose byte width is not a power of two from
 * an LDS buffer: a shape hand-written IR may carry even where the builder
 * admits no such width. Mirrors _load_then_store in the Python test. */
std::string load_then_store(const std::string& elem, const std::string& vector, int lanes)
{
    const std::string lane_count = std::to_string(lanes);
    return "ckdsl.ir v1\n"
           "kernel @lds_odd_width {\n"
           "  params {\n"
           "    %out : ptr<"
           + elem
           + ",global>\n"
             "  }\n"
             "  region @entry {\n"
             "    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
             "    %buf : smem<"
           + elem
           + ", [16]> = tile.smem_alloc (  )\n"
             "    %v : vec<"
           + vector + "> = tile.smem_load_vN ( %buf, %c1 ) { elem_type = s:\"" + elem
           + "\", rank = i:1, vec = i:" + lane_count
           + " }\n"
             "    memref.global_store_vN ( %out, %c1, %v ) { align = i:1, elem_type = s:\""
           + elem + "\", vec = i:" + lane_count
           + " }\n"
             "    cf.return (  )\n"
             "  }\n"
             "}\n";
}

/* A load whose byte width is not a power of two guarantees only element
 * alignment, whatever its element type. */
int test_a_load_whose_width_is_not_a_power_of_two_claims_its_element_width()
{
    const struct
    {
        const char* elem;
        const char* vector;
        int lanes;
        const char* llvm_vector;
        int claim;
    } cases[] = {
        {"f32", "f32x3", 3, "<3 x float>", 4},
        {"f16", "f16x6", 6, "<6 x half>", 2},
        {"fp8e4m3", "fp8e4m3x12", 12, "<12 x i8>", 1},
        {"f16", "f16x3", 3, "<3 x half>", 2},
        {"i8", "i8x3", 3, "<3 x i8>", 1},
        {"f32", "f32x6", 6, "<6 x float>", 4},
    };
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "parsed") == ROCKE_OK);
        rocke_kernel_def_t* kernel = NULL;
        CHECK(rocke_ir_parse(
                  load_then_store(test_case.elem, test_case.vector, test_case.lanes).c_str(),
                  &b,
                  &kernel)
              == ROCKE_OK);
        const Lowering lowering = lower_and_free(&b);
        CHECK(lowering.status == ROCKE_OK);
        CHECK(every_lds_access_claims(lowering.text,
                                      std::string("= load ") + test_case.llvm_vector
                                          + ", ptr addrspace(3)",
                                      test_case.claim));
    }
    return 0;
}

/* smem_store_distributed writes one element at a time, so each store claims
 * the stored element's width (4 bytes for f32), not a fixed 2. */
int test_a_distributed_store_claims_its_element_width()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_align_distributed_f32") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int float_shape[] = {4};
    rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
    rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
    rocke_value_t* half_value = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_value_t* values = rocke_b_global_load_vN(&b, out, zero, rocke_f32(), 4, 16);
    const rocke_attr_map_t no_layout = {NULL, 0, 0};
    rocke_b_smem_store_distributed(&b, float_buf, &no_layout, values);
    rocke_value_t* reloaded = rocke_b_smem_load_vN(&b, float_buf, at_zero, 1, rocke_f32(), 4);
    rocke_b_global_store_vN(&b, out, zero, reloaded, 4, 16);
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_b_ret(&b);
    const Lowering lowering = lower_and_free(&b);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(lines_with(lowering.text, "store float %").size() == 4); /* all four in LDS */
    CHECK(every_lds_access_claims(lowering.text, "store float %", 4));
    return 0;
}

/* A segment of 8-byte elements is 8-byte aligned even when no typed access
 * claims it (here it is only address-taken), for i64 as for f64. */
int test_an_eight_byte_element_segment_is_aligned_to_its_width()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_align_i64_address_taken") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_i32(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int wide_shape[] = {2};
    rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
    rocke_value_t* wide_buf = rocke_b_smem_alloc(&b, rocke_i64(), wide_shape, 1, "w");
    rocke_value_t* half_value = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_value_t* address = rocke_b_trunc(&b, rocke_b_smem_addr_of(&b, wide_buf), rocke_i32());
    rocke_b_global_store_vN(&b, out, zero, address, 1, 0);
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_b_ret(&b);
    const Lowering lowering = lower_and_free(&b);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(segment_offsets(lowering.text) == std::vector<int>({8}));
    return 0;
}

/* An 8 x fp8 LDS load is 8 bytes wide, so it claims 8-byte alignment (a
 * width table that missed the fp8 types once made it claim 16). */
int test_eight_wide_fp8_load_claims_eight_bytes()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_align_fp8x8") == ROCKE_OK);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int byte_shape[] = {24};
    rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
    rocke_value_t* byte_buf = rocke_b_smem_alloc(&b, rocke_fp8e4m3(), byte_shape, 1, "bv");
    rocke_value_t* half_value = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_value_t* const at_eight[] = {rocke_b_const_i32(&b, 8)};
    rocke_value_t* bytes = rocke_b_smem_load_vN(&b, byte_buf, at_eight, 1, rocke_fp8e4m3(), 8);
    rocke_value_t* const at_sixteen[] = {rocke_b_const_i32(&b, 16)};
    rocke_b_smem_store_vN(&b, byte_buf, at_sixteen, 1, bytes, 8);
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
    rocke_b_ret(&b);
    const Lowering lowering = lower_and_free(&b);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(every_lds_access_claims(lowering.text, "= load <8 x i8>, ptr addrspace(3)", 8));
    CHECK(every_lds_access_claims(lowering.text, "store <8 x i8> %", 8));
    return 0;
}

/* An f16 tile whose rows are 12 elements, read and written 8 wide: row 1
 * starts 24 bytes in, so the builder records, and both accesses claim, only
 * the 8 bytes every row start guarantees (gcd(8, 12) = 4 elements). Mirrors
 * test_a_padded_tile_bounds_the_vector_claim_by_its_row_starts. */
int test_a_padded_tile_bounds_the_vector_claim_by_its_row_starts()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_align_padded_rows") == ROCKE_OK);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int tile_shape[] = {2, 12};
    rocke_value_t* tile = rocke_b_smem_alloc(&b, rocke_f16(), tile_shape, 2, "t");
    rocke_value_t* const row_zero[] = {zero, zero};
    rocke_value_t* row = rocke_b_smem_load_vN(&b, tile, row_zero, 2, rocke_f16(), 8);
    rocke_value_t* const row_one[] = {rocke_b_const_i32(&b, 1), zero};
    rocke_b_smem_store_vN(&b, tile, row_one, 2, row, 8);
    rocke_b_ret(&b);
    int64_t recorded = 0;
    rocke_region_t* body = b.kernel->body;
    for(int i = 0; i < body->num_ops; i++)
    {
        if(body->ops[i]->opcode == ROCKE_OP_TILE_SMEM_STORE_VN)
            CHECK(rocke_attr_get_int(&body->ops[i]->attrs, "align", &recorded));
    }
    CHECK(recorded == 8);
    const Lowering lowering = lower_and_free(&b);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(every_lds_access_claims(lowering.text, "= load <8 x half>, ptr addrspace(3)", 8));
    CHECK(every_lds_access_claims(lowering.text, "store <8 x half> %", 8));
    return 0;
}

/* An f32 tile whose rows are 6 elements, read and written 4 wide by the f32
 * epilogue ops: row 1 starts 24 bytes in, so both accesses claim only the 8
 * bytes every row start guarantees (gcd(4, 6) = 2 elements). Mirrors the
 * Python test of the same name in test_smem_access_alignment.py. */
int test_a_padded_tile_bounds_the_f32_epilogue_claims_by_its_row_starts()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_align_padded_f32_rows") == ROCKE_OK);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int tile_shape[] = {2, 6};
    rocke_value_t* tile = rocke_b_smem_alloc(&b, rocke_f32(), tile_shape, 2, "t");
    rocke_value_t* const row_zero[] = {zero, zero};
    rocke_value_t* row = rocke_b_smem_load_vN_f32(&b, tile, row_zero, 2, 4);
    rocke_value_t* const row_one[] = {rocke_b_const_i32(&b, 1), zero};
    rocke_b_smem_store_vN_f32(&b, tile, row_one, 2, row, 4);
    rocke_b_ret(&b);
    const Lowering lowering = lower_and_free(&b);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(every_lds_access_claims(lowering.text, "= load <4 x float>, ptr addrspace(3)", 8));
    CHECK(every_lds_access_claims(lowering.text, "store <4 x float> %", 8));
    return 0;
}

/* True when `build`, a call to one C builder on `builder`, is refused with
 * ROCKE_ERR_VALUE and exactly `message`, and adds no op to the kernel. */
template <typename Build>
bool builder_refuses(rocke_ir_builder_t* builder, Build build, const char* message)
{
    const int ops_before = builder->kernel->body->num_ops;
    bool refused = false;
    try
    {
        build();
    }
    catch(const ckc::Error& error)
    {
        refused = error.code() == ROCKE_ERR_VALUE && std::strcmp(error.what(), message) == 0;
    }
    return refused && builder->kernel->body->num_ops == ops_before;
}

/* True when the serialized kernel `text` parses, the verifier reports exactly
 * `message` for it, and the LLVM lowering refuses it with ROCKE_ERR_VALUE and
 * the same text, emitting nothing. */
bool serialized_kernel_is_refused(const std::string& text, const char* message)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "parsed") != ROCKE_OK)
        return false;
    rocke_kernel_def_t* kernel = NULL;
    if(rocke_ir_parse(text.c_str(), &b, &kernel) != ROCKE_OK)
    {
        rocke_ir_builder_free(&b);
        return false;
    }
    rocke_diag_t* diags = NULL;
    size_t num_diags = 0;
    const bool verified = rocke_verify(kernel, &diags, &num_diags) == ROCKE_OK;
    const bool reported = verified && num_diags == 1 && std::strcmp(diags[0].message, message) == 0;
    rocke_diags_free(diags, num_diags);
    const Lowering lowering = lower_and_free(&b);
    return reported && lowering.status == ROCKE_ERR_VALUE && lowering.text.empty()
           && lowering.error == message;
}

/* The lowerings store the value as an n-element vector, so smem_store_vN
 * refuses a value of any other lane count (a scalar has one) with
 * ROCKE_ERR_VALUE and adds no op; for n = 1 that includes a wider vector, which
 * is no longer stored whole. Mirrors
 * test_a_store_whose_value_lanes_differ_from_n_is_refused. */
int test_a_store_whose_value_lanes_differ_from_n_is_refused()
{
    const struct
    {
        int lanes;
        int n;
        const char* message;
    } cases[] = {
        {4, 2, "smem_store_vN failed - lanes=4, expected n=2"},
        {4, 8, "smem_store_vN failed - lanes=4, expected n=8"},
        {2, 4, "smem_store_vN failed - lanes=2, expected n=4"},
        {4, 1, "smem_store_vN failed - lanes=4, expected n=1"},
        {3, 1, "smem_store_vN failed - lanes=3, expected n=1"},
        {1, 4, "smem_store_vN failed - lanes=1, expected n=4"},
    };
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "lds_store_lanes") == ROCKE_OK);
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        const int float_shape[] = {16};
        rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
        rocke_value_t* value
            = test_case.lanes == 1
                  ? rocke_b_global_load(&b, out, zero, rocke_f32(), 4)
                  : rocke_b_global_load_vN(&b, out, zero, rocke_f32(), test_case.lanes, 4);
        CHECK(rocke_ir_builder_ok(&b));
        rocke_value_t* const at_zero[] = {zero};
        const bool refused = builder_refuses(
            &b,
            [&] { rocke_b_smem_store_vN(&b, float_buf, at_zero, 1, value, test_case.n); },
            test_case.message);
        rocke_ir_builder_free(&b);
        CHECK(refused);
    }
    return 0;
}

/* n = 1 takes a one-lane vector as it takes a scalar: one element, stored as
 * <1 x float>, which the verifier accepts. Mirrors
 * test_a_one_lane_vector_is_a_one_element_store. */
int test_a_one_lane_vector_is_a_one_element_store()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_store_f32x1") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int float_shape[] = {16};
    rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
    rocke_value_t* one_lane
        = rocke_b_vector_splat(&b, rocke_b_global_load(&b, out, zero, rocke_f32(), 4), 1);
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_vN(&b, float_buf, at_zero, 1, one_lane, 1);
    rocke_b_ret(&b);
    CHECK(rocke_ir_builder_ok(&b));
    rocke_diag_t* diags = NULL;
    size_t num_diags = 0;
    CHECK(rocke_verify(b.kernel, &diags, &num_diags) == ROCKE_OK);
    rocke_diags_free(diags, num_diags);
    const Lowering lowering = lower_and_free(&b);
    CHECK(num_diags == 0);
    CHECK(lowering.status == ROCKE_OK);
    CHECK(every_lds_access_claims(lowering.text, "store <1 x float> %", 4));
    return 0;
}

/* Serialized IR reaches the verifier and the lowerings without passing the
 * builder: the verifier reports, and the LLVM lowering refuses with
 * ROCKE_ERR_VALUE, a tile.smem_store_vN whose value does not have `vec` lanes
 * (a scalar has one). Mirrors
 * test_a_serialized_store_whose_value_lanes_differ_from_vec_is_refused. */
int test_a_serialized_store_whose_value_lanes_differ_from_vec_is_refused()
{
    const struct
    {
        const char* value_line;
        int vec;
        const char* message;
    } cases[] = {
        {"%v3 : vec<f32x4> = memref.global_load_vN ( %out, %c1 ) "
         "{ align = i:4, elem_type = s:\"f32\", vec = i:4 }",
         2,
         "tile.smem_store_vN failed - lanes=4, expected vec=2"},
        {"%v3 : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:1.0 }",
         4,
         "tile.smem_store_vN failed - lanes=1, expected vec=4"},
    };
    for(const auto& test_case : cases)
    {
        const std::string text
            = std::string("ckdsl.ir v1\n"
                          "kernel @lds_store_lanes {\n"
                          "  params {\n"
                          "    %out : ptr<f32,global>\n"
                          "  }\n"
                          "  region @entry {\n"
                          "    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
                          "    %buf2 : smem<f32, [16]> = tile.smem_alloc (  )\n"
                          "    ")
              + test_case.value_line
              + "\n"
                "    tile.smem_store_vN ( %buf2, %c1, %v3 ) { align = i:4, elem_type = "
                "s:\"f32\", rank = i:1, vec = i:"
              + std::to_string(test_case.vec)
              + " }\n"
                "    cf.return (  )\n"
                "  }\n"
                "}\n";
        CHECK(serialized_kernel_is_refused(text, test_case.message));
    }
    return 0;
}

/* smem_store_vN_f32 stores its value as an n-element f32 vector (a scalar for
 * n = 1), so it refuses a value of another lane count as smem_store_vN does.
 * Mirrors test_an_f32_epilogue_store_whose_value_lanes_differ_from_n_is_refused. */
int test_an_f32_epilogue_store_whose_value_lanes_differ_from_n_is_refused()
{
    const struct
    {
        int lanes;
        int n;
        const char* message;
    } cases[] = {
        {4, 2, "smem_store_vN_f32 failed - lanes=4, expected n=2"},
        {2, 4, "smem_store_vN_f32 failed - lanes=2, expected n=4"},
        {4, 1, "smem_store_vN_f32 failed - lanes=4, expected n=1"},
        {1, 4, "smem_store_vN_f32 failed - lanes=1, expected n=4"},
    };
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "lds_store_f32_epilogue_lanes") == ROCKE_OK);
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        const int tile_shape[] = {16};
        rocke_value_t* tile = rocke_b_smem_alloc(&b, rocke_f32(), tile_shape, 1, "f");
        rocke_value_t* value
            = test_case.lanes == 1
                  ? rocke_b_global_load(&b, out, zero, rocke_f32(), 4)
                  : rocke_b_global_load_vN(&b, out, zero, rocke_f32(), test_case.lanes, 4);
        CHECK(rocke_ir_builder_ok(&b));
        rocke_value_t* const at_zero[] = {zero};
        const bool refused = builder_refuses(
            &b,
            [&] { rocke_b_smem_store_vN_f32(&b, tile, at_zero, 1, value, test_case.n); },
            test_case.message);
        rocke_ir_builder_free(&b);
        CHECK(refused);
    }
    return 0;
}

/* The f32 epilogue ops emit an f32 access and size its alignment claim for f32
 * elements, so they refuse a tile of any other element type. Mirrors
 * test_an_f32_epilogue_access_to_a_tile_not_holding_f32_is_refused. */
int test_an_f32_epilogue_access_to_a_tile_not_holding_f32_is_refused()
{
    const struct
    {
        const rocke_type_t* tile_elem;
        bool is_store;
        const char* message;
    } cases[] = {
        {rocke_f16(), false, "smem_load_vN_f32 failed - tile_elem=f16, expected f32"},
        {rocke_bf16(), false, "smem_load_vN_f32 failed - tile_elem=bf16, expected f32"},
        {rocke_f16(), true, "smem_store_vN_f32 failed - tile_elem=f16, expected f32"},
        {rocke_bf16(), true, "smem_store_vN_f32 failed - tile_elem=bf16, expected f32"},
    };
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "lds_f32_epilogue_tile") == ROCKE_OK);
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        const int tile_shape[] = {16};
        rocke_value_t* tile = rocke_b_smem_alloc(&b, test_case.tile_elem, tile_shape, 1, "t");
        rocke_value_t* value = rocke_b_global_load_vN(&b, out, zero, rocke_f32(), 4, 4);
        CHECK(rocke_ir_builder_ok(&b));
        rocke_value_t* const at_zero[] = {zero};
        const bool refused = builder_refuses(
            &b,
            [&] {
                if(test_case.is_store)
                    rocke_b_smem_store_vN_f32(&b, tile, at_zero, 1, value, 4);
                else
                    rocke_b_smem_load_vN_f32(&b, tile, at_zero, 1, 4);
            },
            test_case.message);
        rocke_ir_builder_free(&b);
        CHECK(refused);
    }
    return 0;
}

/* smem_store_vN_f32 emits an f32 store, so it refuses a value of any other
 * element type. Mirrors
 * test_an_f32_epilogue_store_of_a_value_not_holding_f32_is_refused. */
int test_an_f32_epilogue_store_of_a_value_not_holding_f32_is_refused()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "lds_store_f32_epilogue_f16_value") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f16(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int tile_shape[] = {16};
    rocke_value_t* tile = rocke_b_smem_alloc(&b, rocke_f32(), tile_shape, 1, "f");
    rocke_value_t* value = rocke_b_global_load_vN(&b, out, zero, rocke_f16(), 4, 2);
    CHECK(rocke_ir_builder_ok(&b));
    rocke_value_t* const at_zero[] = {zero};
    const bool refused = builder_refuses(
        &b,
        [&] { rocke_b_smem_store_vN_f32(&b, tile, at_zero, 1, value, 4); },
        "smem_store_vN_f32 failed - value_elem=f16, expected f32");
    rocke_ir_builder_free(&b);
    CHECK(refused);
    return 0;
}

/* smem_store_f16 records an f16 store, so it refuses a value of any other
 * element type. Mirrors test_an_f16_store_of_a_value_not_holding_f16_is_refused. */
int test_an_f16_store_of_a_value_not_holding_f16_is_refused()
{
    const struct
    {
        const rocke_type_t* value_elem;
        const char* message;
    } cases[] = {
        {rocke_f32(), "smem_store_f16 failed - value_elem=f32, expected f16"},
        {rocke_bf16(), "smem_store_f16 failed - value_elem=bf16, expected f16"},
    };
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "lds_store_f16_value") == ROCKE_OK);
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, test_case.value_elem, "global"), NULL);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        const int tile_shape[] = {16};
        rocke_value_t* tile = rocke_b_smem_alloc(&b, rocke_f16(), tile_shape, 1, "h");
        rocke_value_t* value = rocke_b_global_load(&b, out, zero, test_case.value_elem, 2);
        CHECK(rocke_ir_builder_ok(&b));
        rocke_value_t* const at_zero[] = {zero};
        const bool refused = builder_refuses(
            &b, [&] { rocke_b_smem_store_f16(&b, tile, at_zero, 1, value); }, test_case.message);
        rocke_ir_builder_free(&b);
        CHECK(refused);
    }
    return 0;
}

/* Serialized IR reaches the verifier and the lowerings without passing the
 * builder: the verifier reports, and the LLVM lowering refuses with
 * ROCKE_ERR_VALUE, an f32 epilogue access to a tile that does not hold f32,
 * and a store whose value does not hold f32 or does not have `vec` lanes.
 * Mirrors test_a_serialized_f32_epilogue_access_breaking_its_contract_is_refused. */
int test_a_serialized_f32_epilogue_access_breaking_its_contract_is_refused()
{
    const struct
    {
        const char* tile_elem;
        const char* access_lines;
        const char* message;
    } cases[] = {
        {"f16",
         "%av4 : vec<f32x4> = tile.smem_load_vN_f32 ( %buf2, %c1 ) "
         "{ elem_type = s:\"f32\", rank = i:1, vec = i:4 }",
         "tile.smem_load_vN_f32 failed - tile_elem=f16, expected f32"},
        {"f16",
         "%v3 : vec<f32x4> = memref.global_load_vN ( %out, %c1 ) "
         "{ align = i:4, elem_type = s:\"f32\", vec = i:4 }\n"
         "    tile.smem_store_vN_f32 ( %buf2, %c1, %v3 ) "
         "{ elem_type = s:\"f32\", rank = i:1, vec = i:4 }",
         "tile.smem_store_vN_f32 failed - tile_elem=f16, expected f32"},
        {"f32",
         "%v3 : vec<f16x4> = memref.global_load_vN ( %outh, %c1 ) "
         "{ align = i:2, elem_type = s:\"f16\", vec = i:4 }\n"
         "    tile.smem_store_vN_f32 ( %buf2, %c1, %v3 ) "
         "{ elem_type = s:\"f32\", rank = i:1, vec = i:4 }",
         "tile.smem_store_vN_f32 failed - value_elem=f16, expected f32"},
        {"f32",
         "%v3 : vec<f32x4> = memref.global_load_vN ( %out, %c1 ) "
         "{ align = i:4, elem_type = s:\"f32\", vec = i:4 }\n"
         "    tile.smem_store_vN_f32 ( %buf2, %c1, %v3 ) "
         "{ elem_type = s:\"f32\", rank = i:1, vec = i:2 }",
         "tile.smem_store_vN_f32 failed - lanes=4, expected vec=2"},
    };
    for(const auto& test_case : cases)
    {
        const std::string text = std::string("ckdsl.ir v1\n"
                                             "kernel @lds_f32_epilogue_contract {\n"
                                             "  params {\n"
                                             "    %out : ptr<f32,global>\n"
                                             "    %outh : ptr<f16,global>\n"
                                             "  }\n"
                                             "  region @entry {\n"
                                             "    %c1 : i32 = arith.constant (  ) "
                                             "{ ity = s:\"i32\", value = i:0 }\n"
                                             "    %buf2 : smem<")
                                 + test_case.tile_elem + ", [16]> = tile.smem_alloc (  )\n    "
                                 + test_case.access_lines
                                 + "\n"
                                   "    cf.return (  )\n"
                                   "  }\n"
                                   "}\n";
        CHECK(serialized_kernel_is_refused(text, test_case.message));
    }
    return 0;
}

} // namespace

int main()
{
    if(test_vector_access_moves_its_segment_to_the_vector_alignment()
       || test_access_wider_than_the_pool_claims_the_pool_alignment()
       || test_a_valid_store_alignment_wider_than_the_pool_is_still_capped()
       || test_store_alignment_not_a_power_of_two_is_refused()
       || test_store_alignment_not_an_integer_is_refused()
       || test_a_whole_value_store_claims_element_alignment()
       || test_a_load_whose_width_is_not_a_power_of_two_claims_its_element_width()
       || test_a_distributed_store_claims_its_element_width()
       || test_an_eight_byte_element_segment_is_aligned_to_its_width()
       || test_eight_wide_fp8_load_claims_eight_bytes()
       || test_a_padded_tile_bounds_the_vector_claim_by_its_row_starts()
       || test_a_padded_tile_bounds_the_f32_epilogue_claims_by_its_row_starts()
       || test_a_store_whose_value_lanes_differ_from_n_is_refused()
       || test_a_one_lane_vector_is_a_one_element_store()
       || test_a_serialized_store_whose_value_lanes_differ_from_vec_is_refused()
       || test_an_f32_epilogue_store_whose_value_lanes_differ_from_n_is_refused()
       || test_an_f32_epilogue_access_to_a_tile_not_holding_f32_is_refused()
       || test_an_f32_epilogue_store_of_a_value_not_holding_f32_is_refused()
       || test_an_f16_store_of_a_value_not_holding_f16_is_refused()
       || test_a_serialized_f32_epilogue_access_breaking_its_contract_is_refused())
    {
        std::fprintf(stderr, "LDS alignment contract failed\n");
        return 1;
    }
    std::printf("rocke_smem_access_alignment: ok\n");
    return 0;
}
