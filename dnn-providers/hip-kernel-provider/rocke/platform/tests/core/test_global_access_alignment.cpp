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
 * `align`, to the caller's value or that same natural alignment, so the
 * lowering cases parse serialized IR. They mirror
 * tests/core/test_global_access_alignment.py and the HIP refusals of
 * tests/core/test_mma_io.py and tests/core/test_tf32.py, whose C++ halves
 * otherwise run only where the pybind engine is built. The scalar global loads
 * and stores check a given `align` with the same rule when they are built and
 * read it with that rule when they are lowered; with `align` 0 they record
 * their own default, as the Python builders do when it is omitted.
 */

#include "rocke/error.hpp"
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

/* Serialized IR that loads an f16 and an f32 from global memory and stores them
 * back through the four scalar global ops, none of them carrying an `align`
 * attribute. */
std::string scalar_copy_without_align()
{
    return "ckdsl.ir v1\n"
           "kernel @global_scalar_copy_without_align {\n"
           "  params {\n"
           "    %h : ptr<f16,global>\n"
           "    %f : ptr<f32,global>\n"
           "  }\n"
           "  region @entry {\n"
           "    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
           "    %x : f16 = memref.global_load ( %h, %c0 )\n"
           "    %y : f32 = memref.global_load_typed ( %f, %c0 ) { elem_type = s:\"f32\" }\n"
           "    memref.global_store ( %h, %c0, %x ) { elem_type = s:\"f16\" }\n"
           "    memref.global_store_typed ( %f, %c0, %y )\n"
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

/* Without `align` the f16 load and store claim 2 bytes and the typed load and
 * store claim 1, as each did before it read the shared rule. */
int test_an_llvm_scalar_global_access_without_align_keeps_its_claim()
{
    const Lowering lowering = lower_llvm(scalar_copy_without_align());
    CHECK(lowering.status == ROCKE_OK);
    CHECK(contains(lowering.text, " = load half, ptr addrspace(1) %gep.1, align 2\n"));
    CHECK(contains(lowering.text, " = load float, ptr addrspace(1) %gep.2, align 1\n"));
    CHECK(contains(lowering.text, "  store half %x, ptr addrspace(1) %gep.3, align 2\n"));
    CHECK(contains(lowering.text, "  store float %y, ptr addrspace(1) %gep.4, align 1\n"));
    return 0;
}

/* A scalar global load or store refuses an `align` that is not a positive power
 * of two with the vector ops' text and status, instead of printing it or
 * reading a bool or a float as missing. */
int test_an_llvm_scalar_global_access_refuses_an_invalid_align()
{
    struct InvalidAlign
    {
        AlignOverwrite overwrite;
        rocke_status_t status;
        const char* refusal;
    };
    const InvalidAlign invalid_aligns[] = {
        {[](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
             rocke_attr_set_bool(b, attrs, "align", true);
         },
         ROCKE_ERR_TYPE,
         "align=bool, expected an integer"},
        {[](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
             rocke_attr_set_float(b, attrs, "align", 8.0);
         },
         ROCKE_ERR_TYPE,
         "align=float, expected an integer"},
        {[](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
             rocke_attr_set_int(b, attrs, "align", 24);
         },
         ROCKE_ERR_VALUE,
         "align=24, expected a positive power of two"},
    };
    const rocke_opcode_t scalar_accesses[] = {ROCKE_OP_MEMREF_GLOBAL_LOAD,
                                              ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
                                              ROCKE_OP_MEMREF_GLOBAL_STORE,
                                              ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED};
    for(const rocke_opcode_t opcode : scalar_accesses)
    {
        for(const InvalidAlign& invalid : invalid_aligns)
        {
            const Lowering lowering
                = lower_llvm(scalar_copy_without_align(), opcode, invalid.overwrite);
            CHECK(lowering.status == invalid.status);
            CHECK(lowering.text.empty());
            CHECK(lowering.error
                  == std::string(rocke_opcode_name(opcode)) + " failed - " + invalid.refusal);
        }
    }
    return 0;
}

/* The int `align` attribute the builder recorded on `op`, or -1 when absent. */
int64_t recorded_align(const rocke_op_t* op)
{
    int64_t align = -1;
    return rocke_attr_get_int(&op->attrs, "align", &align) ? align : -1;
}

/* The access the global vector builders record and what they claim with no
 * `align` (0): its byte width when that is a power of two, else its element
 * width. */
struct BuilderCase
{
    const rocke_type_t* elem;
    int lanes;
    int64_t claim;
};

/* With `align` 0 a global vector load or store records the natural alignment
 * of the access (rocke_i_vector_access_align), not a width of its own; a given
 * power of two is recorded as given. */
int test_the_global_vector_builders_default_to_the_natural_alignment()
{
    const BuilderCase loads[] = {{rocke_f32(), 4, 16},
                                 {rocke_f32(), 3, 4},
                                 {rocke_f16(), 6, 2},
                                 {rocke_f16(), 16, 32},
                                 {rocke_i8(), 12, 1}};
    for(const BuilderCase& load : loads)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "global_load_default_align") == ROCKE_OK);
        rocke_value_t* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, load.elem, "global"), NULL);
        rocke_value_t* c0 = rocke_b_const_i32(&b, 0);
        rocke_value_t* v = rocke_b_global_load_vN(&b, p, c0, load.elem, load.lanes, 0);
        CHECK(v != NULL && recorded_align(v->op) == load.claim);
        rocke_value_t* given = rocke_b_global_load_vN(&b, p, c0, load.elem, load.lanes, 8);
        CHECK(given != NULL && recorded_align(given->op) == 8);
        rocke_ir_builder_free(&b);
    }
    const BuilderCase stores[]
        = {{rocke_f32(), 4, 16}, {rocke_f32(), 1, 4}, {rocke_f64(), 2, 16}, {rocke_i8(), 16, 16}};
    for(const BuilderCase& store : stores)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "global_store_default_align") == ROCKE_OK);
        const rocke_type_t* value_type
            = store.lanes == 1 ? store.elem : rocke_vector_type(&b, store.elem, store.lanes);
        rocke_value_t* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, store.elem, "global"), NULL);
        rocke_value_t* v = rocke_b_param(&b, "v", value_type, NULL);
        rocke_b_global_store_vN(&b, p, rocke_b_const_i32(&b, 0), v, store.lanes, 0);
        CHECK(rocke_ir_builder_status(&b) == ROCKE_OK);
        const rocke_region_t* body = b.kernel->body;
        CHECK(recorded_align(body->ops[body->num_ops - 1]) == store.claim);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

/* A negative `align`, or one that is not a power of two, is refused when the
 * op is built, with the lowering's text and status, instead of being read as
 * the default or recorded. */
int test_the_global_vector_builders_refuse_an_invalid_align()
{
    for(const int align : {-4, 6, 24})
    {
        const std::string claim
            = " failed - align=" + std::to_string(align) + ", expected a positive power of two";
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "global_vector_bad_align") == ROCKE_OK);
        rocke_value_t* p = rocke_b_param(&b, "p", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
        rocke_value_t* v = rocke_b_param(&b, "v", rocke_vector_type(&b, rocke_f32(), 4), NULL);
        rocke_value_t* c0 = rocke_b_const_i32(&b, 0);
        const int ops_before = b.kernel->body->num_ops;
        bool load_refused = false;
        try
        {
            (void)rocke_b_global_load_vN(&b, p, c0, rocke_f32(), 3, align);
        }
        catch(const ckc::Error& error)
        {
            load_refused = error.code() == ROCKE_ERR_VALUE
                           && error.message() == "memref.global_load_vN" + claim;
        }
        CHECK(load_refused);
        bool store_refused = false;
        try
        {
            rocke_b_global_store_vN(&b, p, c0, v, 4, align);
        }
        catch(const ckc::Error& error)
        {
            store_refused = error.code() == ROCKE_ERR_VALUE
                            && error.message() == "memref.global_store_vN" + claim;
        }
        CHECK(store_refused);
        CHECK(b.kernel->body->num_ops == ops_before);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

/* The last scalar global load or store recorded in `b`'s kernel body. */
const rocke_op_t* last_scalar_access(const rocke_ir_builder_t* b)
{
    const rocke_region_t* body = b->kernel->body;
    for(int i = body->num_ops - 1; i >= 0; --i)
    {
        switch(body->ops[i]->opcode)
        {
        case ROCKE_OP_MEMREF_GLOBAL_LOAD:
        case ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED:
        case ROCKE_OP_MEMREF_GLOBAL_STORE:
        case ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED:
            return body->ops[i];
        default:
            break;
        }
    }
    return nullptr;
}

/* Builds one access of `elem` values through a scalar global builder with
 * `align`, on a ptr<elem> parameter `ptr` at index `idx`. */
using ScalarAccess = void (*)(rocke_ir_builder_t*,
                              rocke_value_t* ptr,
                              rocke_value_t* idx,
                              const rocke_type_t* elem,
                              int align);

/* A scalar global builder, the op it records, the element type it moves, and
 * the `align` it records for `align` 0 (the Python builder's default when
 * `align` is omitted). */
struct ScalarBuilderCase
{
    const char* name;
    ScalarAccess build;
    rocke_opcode_t opcode;
    const rocke_type_t* elem;
    int64_t unannotated;
};

/* Every scalar global builder of the C API. */
const ScalarBuilderCase* scalar_builder_cases(size_t* count)
{
    static const ScalarBuilderCase cases[] = {
        {"rocke_b_global_load",
         [](rocke_ir_builder_t* b,
            rocke_value_t* p,
            rocke_value_t* i,
            const rocke_type_t* e,
            int a) { (void)rocke_b_global_load(b, p, i, e, a); },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_f32(),
         1},
        {"rocke_b_global_load_f16",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_f16(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD,
         rocke_f16(),
         2},
        {"rocke_b_global_load_f32",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_f32(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_f32(),
         4},
        {"rocke_b_global_load_i32",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_i32(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_i32(),
         4},
        {"rocke_b_global_load_i64",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_i64(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_i64(),
         8},
        {"rocke_b_global_load_bf16",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_bf16(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_bf16(),
         2},
        {"rocke_b_global_load_f64",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_f64(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_f64(),
         8},
        {"rocke_b_global_load_i8",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_i8(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_i8(),
         1},
        {"rocke_b_global_load_i16",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_i16(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_i16(),
         2},
        {"rocke_b_global_load_bf8e5m2",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_bf8e5m2(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_bf8e5m2(),
         1},
        {"rocke_b_global_load_fp8e4m3",
         [](rocke_ir_builder_t* b, rocke_value_t* p, rocke_value_t* i, const rocke_type_t*, int a) {
             (void)rocke_b_global_load_fp8e4m3(b, p, i, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_fp8e4m3(),
         1},
        {"rocke_b_masked_global_load",
         [](rocke_ir_builder_t* b,
            rocke_value_t* p,
            rocke_value_t* i,
            const rocke_type_t* e,
            int a) {
             rocke_value_t* mask = rocke_b_param(b, "mask", rocke_i1(), NULL);
             rocke_value_t* other = rocke_b_param(b, "other", e, NULL);
             (void)rocke_b_masked_global_load(b, p, i, mask, other, e, a);
         },
         ROCKE_OP_MEMREF_GLOBAL_LOAD_TYPED,
         rocke_f32(),
         1},
        {"rocke_b_global_store",
         [](rocke_ir_builder_t* b,
            rocke_value_t* p,
            rocke_value_t* i,
            const rocke_type_t* e,
            int a) { rocke_b_global_store(b, p, i, rocke_b_param(b, "v", e, NULL), a); },
         ROCKE_OP_MEMREF_GLOBAL_STORE_TYPED,
         rocke_f32(),
         1},
    };
    *count = sizeof(cases) / sizeof(cases[0]);
    return cases;
}

/* With `align` 0 each scalar builder records its own default, the value the
 * Python builder records when `align` is omitted; a given power of two is
 * recorded as given. */
int test_a_scalar_global_builder_records_its_default_or_the_given_align()
{
    size_t count = 0;
    const ScalarBuilderCase* cases = scalar_builder_cases(&count);
    for(size_t c = 0; c < count; ++c)
    {
        const ScalarBuilderCase& builder = cases[c];
        for(const int align : {0, 8})
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "scalar_global_align") == ROCKE_OK);
            rocke_value_t* p
                = rocke_b_param(&b, "p", rocke_ptr_type(&b, builder.elem, "global"), NULL);
            builder.build(&b, p, rocke_b_const_i32(&b, 0), builder.elem, align);
            CHECK(rocke_ir_builder_status(&b) == ROCKE_OK);
            const rocke_op_t* access = last_scalar_access(&b);
            CHECK(access != nullptr && access->opcode == builder.opcode);
            CHECK(recorded_align(access) == (align == 0 ? builder.unannotated : align));
            rocke_ir_builder_free(&b);
        }
    }
    return 0;
}

/* A negative `align`, or one that is not a power of two, is refused when the
 * op is built with ROCKE_ERR_VALUE and the Python builder's text, instead of
 * being read as the default or recorded. */
int test_a_scalar_global_builder_refuses_an_invalid_align()
{
    size_t count = 0;
    const ScalarBuilderCase* cases = scalar_builder_cases(&count);
    for(size_t c = 0; c < count; ++c)
    {
        const ScalarBuilderCase& builder = cases[c];
        for(const int align : {-1, 3})
        {
            const std::string refusal = std::string(rocke_opcode_name(builder.opcode))
                                        + " failed - align=" + std::to_string(align)
                                        + ", expected a positive power of two";
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "scalar_global_bad_align") == ROCKE_OK);
            rocke_value_t* p
                = rocke_b_param(&b, "p", rocke_ptr_type(&b, builder.elem, "global"), NULL);
            bool refused = false;
            try
            {
                builder.build(&b, p, rocke_b_const_i32(&b, 0), builder.elem, align);
            }
            catch(const ckc::Error& error)
            {
                refused = error.code() == ROCKE_ERR_VALUE && error.message() == refusal;
            }
            if(!refused)
                std::fprintf(stderr, "%s align=%d was not refused\n", builder.name, align);
            CHECK(refused);
            CHECK(last_scalar_access(&b) == nullptr);
            rocke_ir_builder_free(&b);
        }
    }
    return 0;
}

} // namespace

int main()
{
    if(test_an_llvm_vector_access_without_align_claims_its_natural_alignment()
       || test_a_hip_vector_access_without_align_claims_its_natural_alignment()
       || test_a_sub_byte_vector_store_without_align_is_refused()
       || test_an_llvm_vector_load_refuses_an_invalid_align()
       || test_the_hip_lowering_refuses_an_invalid_align()
       || test_an_llvm_scalar_global_access_without_align_keeps_its_claim()
       || test_an_llvm_scalar_global_access_refuses_an_invalid_align()
       || test_the_global_vector_builders_default_to_the_natural_alignment()
       || test_the_global_vector_builders_refuse_an_invalid_align()
       || test_a_scalar_global_builder_records_its_default_or_the_given_align()
       || test_a_scalar_global_builder_refuses_an_invalid_align())
    {
        std::fprintf(stderr, "global access alignment contract failed\n");
        return 1;
    }
    std::printf("rocke_global_access_alignment: ok\n");
    return 0;
}
