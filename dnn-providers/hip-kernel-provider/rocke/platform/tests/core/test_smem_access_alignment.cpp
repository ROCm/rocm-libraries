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
 * pins for the same kernel.
 */

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
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

/* A one-element f16 buffer, then an f32 buffer written by one
 * smem_store_vN(..., n=1) of a <lanes x float> value, which goes through the
 * whole-value tile.smem_store. A <4 x float> claims its 16-byte width and moves
 * its segment to byte 16; a 12-byte <3 x float> claims only its 4-byte element
 * (`align 12` is not valid LLVM) and its segment stays at byte 4. Mirrors
 * test_a_whole_value_store_claims_the_vector_access_alignment. */
int test_a_whole_value_store_claims_the_vector_access_alignment()
{
    const struct
    {
        int lanes;
        int claim;
    } cases[] = {{3, 4}, {4, 16}};
    for(const auto& test_case : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "lds_align_f16_one_f32_vector") == ROCKE_OK);
        rocke_value_t* out
            = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        const int half_shape[] = {1};
        const int float_shape[] = {12};
        rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
        rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
        rocke_value_t* half_value = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
        rocke_value_t* const at_zero[] = {zero};
        rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
        rocke_value_t* vector
            = rocke_b_global_load_vN(&b, out, zero, rocke_f32(), test_case.lanes, 4);
        rocke_b_smem_store_vN(&b, float_buf, at_zero, 1, vector, 1);
        rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, half_value);
        rocke_b_ret(&b);
        const Lowering lowering = lower_and_free(&b);
        CHECK(lowering.status == ROCKE_OK);
        CHECK(segment_offsets(lowering.text) == std::vector<int>({test_case.claim}));
        CHECK(every_lds_access_claims(lowering.text,
                                      "store <" + std::to_string(test_case.lanes) + " x float> %",
                                      test_case.claim));
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

} // namespace

int main()
{
    if(test_vector_access_moves_its_segment_to_the_vector_alignment()
       || test_access_wider_than_the_pool_claims_the_pool_alignment()
       || test_a_valid_store_alignment_wider_than_the_pool_is_still_capped()
       || test_store_alignment_not_a_power_of_two_is_refused()
       || test_store_alignment_not_an_integer_is_refused()
       || test_a_whole_value_store_claims_the_vector_access_alignment()
       || test_a_load_whose_width_is_not_a_power_of_two_claims_its_element_width()
       || test_a_distributed_store_claims_its_element_width()
       || test_an_eight_byte_element_segment_is_aligned_to_its_width()
       || test_eight_wide_fp8_load_claims_eight_bytes())
    {
        std::fprintf(stderr, "LDS alignment contract failed\n");
        return 1;
    }
    std::printf("rocke_smem_access_alignment: ok\n");
    return 0;
}
