// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_global_access_alignment.cpp -- the alignment a global vector load or
 * store claims in the C++ engine's LLVM and HIP lowerings, through the C API.
 *
 * memref.global_load_vN and memref.global_store_vN claim their `align`
 * attribute (rocke_i_align_attr), or, when they carry none, the natural
 * alignment of the access (rocke_i_vector_access_align): its byte width when
 * that is a power of two, else the element width. The builders always set
 * `align`, so these cases parse serialized IR. They mirror
 * tests/core/test_global_access_alignment.py and the HIP refusals of
 * tests/core/test_mma_io.py and tests/core/test_tf32.py, whose C++ halves
 * otherwise run only where the pybind engine is built.
 */

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

/* The outcome of lowering one kernel: the status, and the text or the refusal
 * message. */
struct Lowering
{
    rocke_status_t status;
    std::string text;
    std::string error;
};

/* Serialized IR that loads `lanes` `elem` values from global memory and stores
 * them back, neither op carrying an `align` attribute. */
std::string copy_without_align(const std::string& elem, int lanes)
{
    const std::string vector = elem + "x" + std::to_string(lanes);
    const std::string attrs
        = "{ elem_type = s:\"" + elem + "\", vec = i:" + std::to_string(lanes) + " }";
    return "ckdsl.ir v1\n"
           "kernel @global_copy_without_align {\n"
           "  params {\n"
           "    %p : ptr<"
           + elem
           + ",global>\n"
             "  }\n"
             "  region @entry {\n"
             "    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
             "    %v : vec<"
           + vector + "> = memref.global_load_vN ( %p, %c0 ) " + attrs
           + "\n"
             "    memref.global_store_vN ( %p, %c0, %v ) "
           + attrs
           + "\n"
             "    cf.return (  )\n"
             "  }\n"
             "}\n";
}

/* Serialized IR storing a four-lane `elem` vector parameter, with no `align`. */
std::string store_of_a_parameter(const std::string& elem)
{
    return "ckdsl.ir v1\n"
           "kernel @global_store_without_align {\n"
           "  params {\n"
           "    %p : ptr<"
           + elem
           + ",global>\n"
             "    %v : vec<"
           + elem
           + "x4>\n"
             "  }\n"
             "  region @entry {\n"
             "    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
             "    memref.global_store_vN ( %p, %c0, %v ) { elem_type = s:\""
           + elem
           + "\", vec = i:4 }\n"
             "    cf.return (  )\n"
             "  }\n"
             "}\n";
}

/* Rewrites the `align` attribute of the kernel's op `opcode`, as serialized or
 * hand-written IR may carry it. */
using AlignOverwrite = void (*)(rocke_ir_builder_t*, rocke_attr_map_t*);

/* Parse `ir_text` into `builder`, applying `overwrite_align` to every op of
 * `opcode` when it is given. */
bool parse_into(rocke_ir_builder_t* builder,
                const std::string& ir_text,
                rocke_opcode_t opcode,
                AlignOverwrite overwrite_align)
{
    rocke_kernel_def_t* kernel = NULL;
    if(rocke_ir_builder_init(builder, "parsed") != ROCKE_OK)
        return false;
    if(rocke_ir_parse(ir_text.c_str(), builder, &kernel) != ROCKE_OK)
        return false;
    if(overwrite_align != nullptr)
    {
        rocke_region_t* body = kernel->body;
        for(int i = 0; i < body->num_ops; i++)
        {
            if(body->ops[i]->opcode == opcode)
                overwrite_align(builder, &body->ops[i]->attrs);
        }
    }
    return true;
}

/* Lower `ir_text` to LLVM IR for gfx942 at llvm20. */
Lowering lower_llvm(const std::string& ir_text,
                    rocke_opcode_t opcode = ROCKE_OP_MEMREF_GLOBAL_LOAD_VN,
                    AlignOverwrite overwrite_align = nullptr)
{
    Lowering lowering{ROCKE_ERR_VALUE, "", "the IR did not parse"};
    rocke_ir_builder_t builder;
    if(parse_into(&builder, ir_text, opcode, overwrite_align))
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP] = {0};
        lowering.status = rocke_lower_kernel_to_llvm_ex(
            builder.kernel, ROCKE_LLVM_FLAVOR_LLVM20, kArch, &llvm_text, err, sizeof(err));
        if(llvm_text != NULL)
        {
            lowering.text = llvm_text;
            std::free(llvm_text);
        }
        lowering.error = err;
    }
    rocke_ir_builder_free(&builder);
    return lowering;
}

/* Lower `ir_text` to HIP source for gfx942. */
Lowering lower_hip(const std::string& ir_text,
                   rocke_opcode_t opcode = ROCKE_OP_MEMREF_GLOBAL_LOAD_VN,
                   AlignOverwrite overwrite_align = nullptr)
{
    Lowering lowering{ROCKE_ERR_VALUE, "", "the IR did not parse"};
    rocke_ir_builder_t builder;
    if(parse_into(&builder, ir_text, opcode, overwrite_align))
    {
        rocke_strbuf_t out;
        rocke_strbuf_init(&out, 0);
        rocke_lower_hip_opts_t opts{};
        opts.arch = kArch;
        lowering.status = rocke_lower_kernel_to_hip(&builder, builder.kernel, &opts, &out);
        if(lowering.status == ROCKE_OK)
            lowering.text = rocke_strbuf_cstr(&out);
        else
            lowering.error = rocke_ir_builder_error(&builder);
        rocke_strbuf_free(&out);
    }
    rocke_ir_builder_free(&builder);
    return lowering;
}

bool contains(const std::string& text, const std::string& needle)
{
    return text.find(needle) != std::string::npos;
}

/* Sixteen bytes claim 16; a 12-byte <3 x float> claims its 4-byte element,
 * where the load used to print an invalid `align 6` and the store was refused. */
int test_an_llvm_vector_access_without_align_claims_its_natural_alignment()
{
    const Lowering four = lower_llvm(copy_without_align("f32", 4));
    CHECK(four.status == ROCKE_OK);
    CHECK(contains(four.text, " = load <4 x float>, ptr addrspace(1) %gep.1, align 16\n"));
    CHECK(contains(four.text, "  store <4 x float> %v, ptr addrspace(1) %gep.2, align 16\n"));
    const Lowering three = lower_llvm(copy_without_align("f32", 3));
    CHECK(three.status == ROCKE_OK);
    CHECK(contains(three.text, " = load <3 x float>, ptr addrspace(1) %gep.1, align 4\n"));
    CHECK(contains(three.text, "  store <3 x float> %v, ptr addrspace(1) %gep.2, align 4\n"));
    return 0;
}

/* A <4 x float> load and store claim 16 bytes and use the direct vector access;
 * a <3 x float> pair claims 4 and copies only its 12-byte payload. */
int test_a_hip_vector_access_without_align_claims_its_natural_alignment()
{
    const Lowering four = lower_hip(copy_without_align("f32", 4));
    CHECK(four.status == ROCKE_OK);
    CHECK(!contains(four.text, "__builtin_memcpy"));
    CHECK(contains(four.text, "f32x4 v = *reinterpret_cast<const f32x4*>(p + c0);"));
    CHECK(contains(four.text, "*reinterpret_cast<f32x4*>(p + c0) = v;"));
    const Lowering three = lower_hip(copy_without_align("f32", 3));
    CHECK(three.status == ROCKE_OK);
    CHECK(contains(three.text, "__builtin_memcpy(&v, __builtin_assume_aligned(p + c0, 4), 12);"));
    CHECK(contains(three.text, "__builtin_memcpy(__builtin_assume_aligned(p + c0, 4), &v, 12);"));
    return 0;
}

/* Four i1 lanes have no byte width to claim, so the LLVM lowering refuses the
 * store rather than give it two bytes per lane. */
int test_a_sub_byte_vector_store_without_align_is_refused()
{
    const Lowering lowering = lower_llvm(store_of_a_parameter("i1"));
    CHECK(lowering.status == ROCKE_ERR_VALUE);
    CHECK(lowering.text.empty());
    CHECK(lowering.error
          == "type_bytes failed - type=i1 (1 bits), expected a whole number of bytes");
    return 0;
}

/* The LLVM vector load checks its `align` as the store does, instead of
 * printing an invalid claim such as `align 6`. */
int test_an_llvm_vector_load_refuses_an_invalid_align()
{
    const AlignOverwrite align_6 = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_int(b, attrs, "align", 6);
    };
    const Lowering six
        = lower_llvm(copy_without_align("f32", 3), ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, align_6);
    CHECK(six.status == ROCKE_ERR_VALUE);
    CHECK(six.error == "memref.global_load_vN failed - align=6, expected a positive power of two");
    const AlignOverwrite align_true = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_bool(b, attrs, "align", true);
    };
    const Lowering boolean
        = lower_llvm(copy_without_align("f32", 3), ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, align_true);
    CHECK(boolean.status == ROCKE_ERR_TYPE);
    CHECK(boolean.error == "memref.global_load_vN failed - align=bool, expected an integer");
    return 0;
}

/* The HIP lowering refuses the same `align` values with the same texts as the
 * LLVM lowering, and never reads a bool or a float as an integer alignment. */
int test_the_hip_lowering_refuses_an_invalid_align()
{
    const AlignOverwrite align_true = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_bool(b, attrs, "align", true);
    };
    const Lowering load
        = lower_hip(copy_without_align("f32", 4), ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, align_true);
    CHECK(load.status == ROCKE_ERR_TYPE);
    CHECK(load.error == "memref.global_load_vN failed - align=bool, expected an integer");
    const AlignOverwrite align_float = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_float(b, attrs, "align", 16.0);
    };
    const Lowering float_load
        = lower_hip(copy_without_align("f32", 4), ROCKE_OP_MEMREF_GLOBAL_LOAD_VN, align_float);
    CHECK(float_load.status == ROCKE_ERR_TYPE);
    CHECK(float_load.error == "memref.global_load_vN failed - align=float, expected an integer");
    const AlignOverwrite align_24 = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_int(b, attrs, "align", 24);
    };
    const Lowering store
        = lower_hip(copy_without_align("f32", 4), ROCKE_OP_MEMREF_GLOBAL_STORE_VN, align_24);
    CHECK(store.status == ROCKE_ERR_VALUE);
    CHECK(store.error
          == "memref.global_store_vN failed - align=24, expected a positive power of two");
    return 0;
}

} // namespace

int main()
{
    if(test_an_llvm_vector_access_without_align_claims_its_natural_alignment()
       || test_a_hip_vector_access_without_align_claims_its_natural_alignment()
       || test_a_sub_byte_vector_store_without_align_is_refused()
       || test_an_llvm_vector_load_refuses_an_invalid_align()
       || test_the_hip_lowering_refuses_an_invalid_align())
    {
        std::fprintf(stderr, "global access alignment contract failed\n");
        return 1;
    }
    std::printf("rocke_global_access_alignment: ok\n");
    return 0;
}
