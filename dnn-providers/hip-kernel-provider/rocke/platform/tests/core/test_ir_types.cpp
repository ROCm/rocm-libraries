// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* The IR type rules the C++ builders apply, checked against the Python builders.
 *
 * The serialized-IR parity seam (rocke.tests.engines) lowers a kernel the
 * PYTHON builder built, so it never runs a C++ rocke_b_* check. This test does.
 *
 * Without arguments it self-checks with the literals core/ir.py uses. With
 * arguments it answers one question for tests/core/test_u8_memory.py, which
 * compares the answer with the Python builder (found through
 * ROCKE_IR_TYPES_TEST):
 *
 *   --unsigned <case>
 *       build the u8 op named <case> (a row of k_unsigned_cases, mirrored in
 *       tests/core/test_u8_arith.py) on u8 or i32 operands; print "ok", or
 *       "error: <message>" when the builder refuses it.
 *
 *   --packed-only <builder> <elem>
 *       make one IR value, or LDS tile, of the packed-only elem with the
 *       builder named by a row of k_packed_only_builders (the rows of
 *       tests/core/test_packed_only.py); print "ok", or "error: <message>"
 *       when the builder refuses it.
 *
 *   --smem-vector <load|store> <elem> <n>
 *       build one LDS vector access of n elem values; print "ok", or
 *       "error: <message>" when the builder refuses it.
 *
 *   --scalar-bits
 *       print "<name> <bits>" for every IR scalar type, the width the engine
 *       gives it (test_dtypes.py compares the map with _IR_SCALAR_BITS).
 *
 *   --bitcast <case>
 *       build the bitcast named <case> (a row of k_bitcast_cases, mirrored in
 *       tests/core/test_bitcast_width.py); print "ok", or
 *       "error: <Python exception class>: <message>" when the builder refuses
 *       it, since a refusal is a TypeError or a ValueError.
 *
 *   --mma <op_id>
 *       build one target-neutral MMA of op_id on i32 zero operands; print its
 *       result type, or "error: <message>" when the builder refuses the call
 *       (tests/core/test_mma_frag_tables.py runs it for every op_id).
 */
#include "rocke/ir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "rocke/dtypes_internal.h"
#include "rocke/error.hpp"
#include "rocke/ir_internal.h"

#define CHECK(x)                                                       \
    do                                                                 \
    {                                                                  \
        if(!(x))                                                       \
        {                                                              \
            fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); \
            return 1;                                                  \
        }                                                              \
    } while(0)

/* The outcome of one builder call: accepted, or refused with a message. */
struct build_outcome
{
    bool accepted;
    std::string error;
    /* Ops the measured call appended to the kernel body, counted after a
     * refusal too, so a refused call that left an op behind is caught; -1 when
     * the setup was refused and the measured call never ran. */
    int ops_added;
};

/* Build an LDS allocation of `elem` and one vector access of `n` elements:
 * smem_load_vN, or smem_store_vN of vector_splat(<loaded elem>, n). Only the
 * access is measured; the setup ops are counted out. */
static build_outcome smem_vector_access(bool store, const char* elem_name, int n)
{
    build_outcome outcome = {false, "", 0};
    const rocke_type_t* elem = rocke_scalar_by_name(elem_name);
    if(!elem)
    {
        outcome.error = std::string("no scalar type named '") + elem_name + "'";
        return outcome;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "smem_vector_access") != ROCKE_OK)
    {
        outcome.error = "builder init failed";
        return outcome;
    }
    int ops_before = -1; /* set once the setup is built */
    try
    {
        const int shape[] = {64};
        rocke_value_t* smem = rocke_b_smem_alloc(&b, elem, shape, 1, "tile");
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_value_t* value = NULL;
        if(store)
        {
            rocke_value_t* src = rocke_b_param(&b, "SRC", rocke_ptr_type(&b, elem, "global"), NULL);
            value = rocke_b_vector_splat(&b, rocke_b_global_load(&b, src, zero, elem, 1), n);
        }
        ops_before = b.kernel->body->num_ops;
        if(store)
            rocke_b_smem_store_vN(&b, smem, &zero, 1, value, n);
        else
            rocke_b_smem_load_vN(&b, smem, &zero, 1, elem, n);
        outcome.accepted = true;
    }
    catch(const ckc::Error& error)
    {
        outcome.error = error.what();
    }
    outcome.ops_added = ops_before < 0 ? -1 : b.kernel->body->num_ops - ops_before;
    rocke_ir_builder_free(&b);
    return outcome;
}

static int print_smem_vector_access(const char* direction, const char* elem, const char* count)
{
    const bool store = strcmp(direction, "store") == 0;
    CHECK(store || strcmp(direction, "load") == 0);
    const build_outcome outcome = smem_vector_access(store, elem, atoi(count));
    if(outcome.accepted)
        printf("ok\n");
    else
        printf("error: %s\n", outcome.error.c_str());
    return 0;
}

/* A builder that would make an IR value or LDS tile of its element type, and
 * the IR op its refusal names. */
struct packed_only_builder
{
    const char* name;
    const char* op;
};

static const packed_only_builder k_packed_only_builders[] = {
    {"param", "param"},
    {"param_vector", "param"},
    {"zero_vec", "zero_vec"},
    {"undef_vec", "vector.undef"},
    {"trunc", "arith.trunc"},
    {"vec_bitcast", "vector.bitcast"},
    {"global_load", "memref.global_load_typed"},
    {"smem_alloc", "tile.smem_alloc"},
};

/* Build the operands, then make one `elem_name` value or LDS tile with
 * `builder`; only that call is measured. */
static build_outcome
    packed_only_value(const char* builder, const char* elem_name, rocke_status_t* code)
{
    build_outcome outcome = {false, "", 0};
    *code = ROCKE_OK;
    const rocke_type_t* elem = rocke_scalar_by_name(elem_name);
    if(!elem)
    {
        outcome.error = std::string("no scalar type named '") + elem_name + "'";
        return outcome;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "packed_only_value") != ROCKE_OK)
    {
        outcome.error = "builder init failed";
        return outcome;
    }
    int ops_before = -1; /* set once the operands are built */
    try
    {
        rocke_value_t* zero = rocke_b_const_i32(&b, 0);
        rocke_value_t* word = rocke_b_zero_vec(&b, rocke_i32(), 1);
        rocke_value_t* ptr = rocke_b_param(&b, "P", rocke_ptr_type(&b, elem, "global"), NULL);
        const rocke_type_t* vector = rocke_vector_type(&b, elem, 8);
        const int shape[] = {64};
        ops_before = b.kernel->body->num_ops;
        if(strcmp(builder, "param") == 0)
            rocke_b_param(&b, "X", elem, NULL);
        else if(strcmp(builder, "param_vector") == 0)
            rocke_b_param(&b, "X", vector, NULL);
        else if(strcmp(builder, "zero_vec") == 0)
            rocke_b_zero_vec(&b, elem, 8);
        else if(strcmp(builder, "undef_vec") == 0)
            rocke_b_undef_vec(&b, elem, 8);
        else if(strcmp(builder, "trunc") == 0)
            rocke_b_trunc(&b, zero, elem);
        else if(strcmp(builder, "vec_bitcast") == 0)
            rocke_b_vec_bitcast(&b, word, vector);
        else if(strcmp(builder, "global_load") == 0)
            rocke_b_global_load(&b, ptr, zero, elem, 1);
        else if(strcmp(builder, "smem_alloc") == 0)
            rocke_b_smem_alloc(&b, elem, shape, 1, "tile");
        else
            throw ckc::ValueError(std::string("no packed-only builder named '") + builder + "'");
        outcome.accepted = true;
    }
    catch(const ckc::Error& error)
    {
        outcome.error = error.what();
        *code = error.code();
    }
    outcome.ops_added = ops_before < 0 ? -1 : b.kernel->body->num_ops - ops_before;
    rocke_ir_builder_free(&b);
    return outcome;
}

static int print_packed_only_value(const char* builder, const char* elem)
{
    rocke_status_t code;
    const build_outcome outcome = packed_only_value(builder, elem, &code);
    if(outcome.accepted)
        printf("ok\n");
    else
        printf("error: %s\n", outcome.error.c_str());
    return 0;
}

/* One builder call on u8 operands (or i32 ones, to show what the unsigned ops
 * refuse). The rows match _UNSIGNED_CASES in tests/core/test_u8_arith.py. */
struct unsigned_case
{
    const char* name;
    rocke_value_t* (*build)(rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y);
    /* The builder's refusal status and text; error is NULL when the call builds
     * one op. */
    rocke_status_t code;
    const char* error;
};

static const unsigned_case k_unsigned_cases[] = {
    {"umax",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_umax(b, x, y);
     },
     ROCKE_OK,
     NULL},
    {"umin",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_umin(b, x, y);
     },
     ROCKE_OK,
     NULL},
    {"uitofp_f32",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t*) {
         return rocke_b_uitofp_f32(b, x);
     },
     ROCKE_OK,
     NULL},
    {"zext",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t*) {
         return rocke_b_zext(b, x, rocke_i32());
     },
     ROCKE_OK,
     NULL},
    {"smax",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_smax(b, x, y);
     },
     ROCKE_ERR_TYPE,
     "smax treats its operands as signed, but u8 is unsigned -- use umax"},
    {"smin",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_smin(b, x, y);
     },
     ROCKE_ERR_TYPE,
     "smin treats its operands as signed, but u8 is unsigned -- use umin"},
    {"sext",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t*) {
         return rocke_b_sext(b, x, rocke_i32());
     },
     ROCKE_ERR_TYPE,
     "sext treats its operands as signed, but u8 is unsigned -- use zext"},
    {"vector_smax",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_vector_smax(
             b, rocke_b_vector_splat(b, x, 4), rocke_b_vector_splat(b, y, 4));
     },
     ROCKE_ERR_TYPE,
     "vector_smax treats its operands as signed, but u8 is unsigned"},
    {"vector_smin",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t* y) {
         return rocke_b_vector_smin(
             b, rocke_b_vector_splat(b, x, 4), rocke_b_vector_splat(b, y, 4));
     },
     ROCKE_ERR_TYPE,
     "vector_smin treats its operands as signed, but u8 is unsigned"},
    {"vector_sext",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t*) {
         return rocke_b_vector_sext(b, rocke_b_vector_splat(b, x, 4), rocke_i32());
     },
     ROCKE_ERR_TYPE,
     "vector_sext treats its operands as signed, but u8 is unsigned"},
    {"umax_of_i32",
     [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
         rocke_value_t* i = rocke_b_const_i32(b, 1);
         return rocke_b_umax(b, i, i);
     },
     ROCKE_ERR_VALUE,
     "umax expects two operands of one unsigned integer type, got i32 and i32"},
    {"uitofp_f32_of_i32",
     [](rocke_ir_builder_t* b, rocke_value_t*, rocke_value_t*) {
         return rocke_b_uitofp_f32(b, rocke_b_const_i32(b, 1));
     },
     ROCKE_ERR_VALUE,
     "uitofp_f32 expects an unsigned integer input, got i32"},
    {"cvt_f32_to_u8_sat_of_u8",
     [](rocke_ir_builder_t* b, rocke_value_t* x, rocke_value_t*) {
         return rocke_b_cvt_f32_to_u8_sat(b, x);
     },
     ROCKE_ERR_VALUE,
     "cvt_f32_to_u8_sat expects f32 input, got u8"},
};

static const unsigned_case* find_unsigned_case(const char* name)
{
    for(const unsigned_case& c : k_unsigned_cases)
        if(strcmp(c.name, name) == 0)
            return &c;
    return NULL;
}

/* Declare two u8 parameters X and Y, then run `c`; ops_added counts every op
 * the case adds, the operands it builds itself included. */
static build_outcome build_unsigned_case(const unsigned_case& c, rocke_status_t* code)
{
    build_outcome outcome = {false, "", 0};
    *code = ROCKE_OK;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "unsigned_case") != ROCKE_OK)
    {
        outcome.error = "builder init failed";
        return outcome;
    }
    int ops_before = -1; /* set once the parameters are declared */
    try
    {
        rocke_value_t* x = rocke_b_param(&b, "X", rocke_u8(), NULL);
        rocke_value_t* y = rocke_b_param(&b, "Y", rocke_u8(), NULL);
        ops_before = b.kernel->body->num_ops;
        c.build(&b, x, y);
        outcome.accepted = true;
    }
    catch(const ckc::Error& error)
    {
        outcome.error = error.what();
        *code = error.code();
    }
    outcome.ops_added = ops_before < 0 ? -1 : b.kernel->body->num_ops - ops_before;
    rocke_ir_builder_free(&b);
    return outcome;
}

static int print_unsigned_case(const char* name)
{
    const unsigned_case* c = find_unsigned_case(name);
    CHECK(c != NULL);
    rocke_status_t code;
    const build_outcome outcome = build_unsigned_case(*c, &code);
    if(outcome.accepted)
        printf("ok\n");
    else
        printf("error: %s\n", outcome.error.c_str());
    return 0;
}

/* The unsigned ops build one op on u8; the signed-by-definition ops refuse u8,
 * and the unsigned ops refuse a signed operand, with core/ir.py's text. */
static int test_unsigned_ops()
{
    for(const unsigned_case& c : k_unsigned_cases)
    {
        rocke_status_t code;
        const build_outcome outcome = build_unsigned_case(c, &code);
        if(c.error == NULL)
        {
            CHECK(outcome.accepted);
            CHECK(outcome.ops_added == 1);
            continue;
        }
        CHECK(!outcome.accepted);
        CHECK(code == c.code);
        CHECK(outcome.error == c.error);
    }
    return 0;
}

/* Every builder refuses an i4 or u4 value or LDS tile with core/ir.py's
 * TypeError text, before any op is built. */
static int test_packed_only_values()
{
    for(const char* elem : {"i4", "u4"})
        for(const packed_only_builder& builder : k_packed_only_builders)
        {
            rocke_status_t code;
            const build_outcome outcome = packed_only_value(builder.name, elem, &code);
            CHECK(!outcome.accepted);
            CHECK(code == ROCKE_ERR_TYPE);
            CHECK(outcome.error
                  == std::string(builder.op) + ": " + elem
                         + " values travel packed in 32-bit words and have no scalar or vector "
                           "form -- keep them in i32 storage and unpack explicitly");
            CHECK(outcome.ops_added == 0);
        }
    return 0;
}

/* The LDS vector pair admits one element list (core/ir.py
 * _SMEM_VECTOR_ELEMENTS), so u8 is refused in both directions with the same
 * text, before any op is built; admitted elements build exactly one op. */
static int test_smem_vector_elements()
{
    static const char* const admitted
        = "f16 / bf16 / i16 / f32 / i32 / tf32 / f64 / fp8e4m3 / bf8e5m2 / i8";
    for(bool store : {false, true})
    {
        const build_outcome refused = smem_vector_access(store, "u8", 4);
        CHECK(!refused.accepted);
        CHECK(refused.error
              == std::string(store ? "smem_store_vN" : "smem_load_vN") + " supports " + admitted
                     + ", got u8");
        CHECK(refused.ops_added == 0);
        for(const char* elem : {"i8", "i16", "f64"})
        {
            const build_outcome built = smem_vector_access(store, elem, 2);
            CHECK(built.accepted);
            CHECK(built.ops_added == 1);
        }
        CHECK(smem_vector_access(store, "i8", 16).accepted);
    }
    return 0;
}

/* The Python exception class a refusal raises, for the --bitcast report. */
static const char* python_exception_class(rocke_status_t code)
{
    switch(code)
    {
    case ROCKE_ERR_TYPE:
        return "TypeError";
    case ROCKE_ERR_VALUE:
        return "ValueError";
    default:
        return "unexpected status";
    }
}

/* One bitcast: the builder (`vec_bitcast` or `bitcast`), the source value and
 * the target type. The rows match _BITCAST_CASES in test_bitcast_width.py. */
struct bitcast_case
{
    const char* name;
    bool vector_builder;
    rocke_value_t* (*source)(rocke_ir_builder_t* b);
    const rocke_type_t* (*target)(rocke_ir_builder_t* b);
};

static rocke_value_t* param_of(rocke_ir_builder_t* b, const rocke_type_t* type)
{
    return rocke_b_param(b, "X", type, NULL);
}

static const bitcast_case k_bitcast_cases[] = {
    {"f32_as_i32",
     false,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_f32()); },
     [](rocke_ir_builder_t*) { return rocke_i32(); }},
    {"f64_as_i32x2",
     false,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_f64()); },
     [](rocke_ir_builder_t* b) { return rocke_vector_type(b, rocke_i32(), 2); }},
    {"i32x4_as_f16x8",
     true,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_vector_type(b, rocke_i32(), 4)); },
     [](rocke_ir_builder_t* b) { return rocke_vector_type(b, rocke_f16(), 8); }},
    {"f32_as_i64",
     false,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_f32()); },
     [](rocke_ir_builder_t*) { return rocke_i64(); }},
    {"f16x8_as_i32x8",
     true,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_vector_type(b, rocke_f16(), 8)); },
     [](rocke_ir_builder_t* b) { return rocke_vector_type(b, rocke_i32(), 8); }},
    {"pointer_as_i64",
     false,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_ptr_type(b, rocke_f32(), "global")); },
     [](rocke_ir_builder_t*) { return rocke_i64(); }},
    {"i64_as_pointer",
     false,
     [](rocke_ir_builder_t* b) { return param_of(b, rocke_i64()); },
     [](rocke_ir_builder_t* b) { return rocke_ptr_type(b, rocke_f32(), "global"); }},
    {"lds_tile_as_i32",
     false,
     [](rocke_ir_builder_t* b) {
         const int shape[] = {16, 16};
         return rocke_b_smem_alloc(b, rocke_f32(), shape, 2, "tile");
     },
     [](rocke_ir_builder_t*) { return rocke_i32(); }},
};

static const bitcast_case* find_bitcast_case(const char* name)
{
    for(const bitcast_case& c : k_bitcast_cases)
        if(strcmp(c.name, name) == 0)
            return &c;
    return NULL;
}

/* Build the source and target of `c`, then the bitcast itself; only the cast
 * is measured. */
static build_outcome build_bitcast(const bitcast_case& c, rocke_status_t* code)
{
    build_outcome outcome = {false, "", 0};
    *code = ROCKE_OK;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "bitcast_case") != ROCKE_OK)
    {
        outcome.error = "builder init failed";
        return outcome;
    }
    int ops_before = -1; /* set once the source and target are built */
    try
    {
        rocke_value_t* source = c.source(&b);
        const rocke_type_t* target = c.target(&b);
        ops_before = b.kernel->body->num_ops;
        if(c.vector_builder)
            rocke_b_vec_bitcast(&b, source, target);
        else
            rocke_b_bitcast(&b, source, target);
        outcome.accepted = true;
    }
    catch(const ckc::Error& error)
    {
        outcome.error = error.what();
        *code = error.code();
    }
    outcome.ops_added = ops_before < 0 ? -1 : b.kernel->body->num_ops - ops_before;
    rocke_ir_builder_free(&b);
    return outcome;
}

static int print_bitcast(const char* name)
{
    const bitcast_case* c = find_bitcast_case(name);
    CHECK(c != NULL);
    rocke_status_t code;
    const build_outcome outcome = build_bitcast(*c, &code);
    if(outcome.accepted)
        printf("ok\n");
    else
        printf("error: %s: %s\n", python_exception_class(code), outcome.error.c_str());
    return 0;
}

/* Equal-width casts build one op; a width change, a pointer or an LDS
 * allocation is refused with core/ir.py's text before any op is built. */
static int test_bitcast_width()
{
    for(const char* name : {"f32_as_i32", "f64_as_i32x2", "i32x4_as_f16x8"})
    {
        rocke_status_t code;
        const build_outcome built = build_bitcast(*find_bitcast_case(name), &code);
        CHECK(built.accepted);
        CHECK(built.ops_added == 1);
    }
    const struct
    {
        const char* name;
        rocke_status_t code;
        const char* error;
    } refused[] = {
        {"f32_as_i64",
         ROCKE_ERR_VALUE,
         "bitcast of f32 (32 bits) to i64 (64 bits) changes the bit width -- a bitcast "
         "keeps every bit; convert the value to change its width"},
        {"f16x8_as_i32x8",
         ROCKE_ERR_VALUE,
         "vec_bitcast of vec<f16x8> (128 bits) to vec<i32x8> (256 bits) changes the bit "
         "width -- a bitcast keeps every bit; slice or concatenate the vector first"},
        {"pointer_as_i64",
         ROCKE_ERR_TYPE,
         "bitcast of ptr<f32,global> to i64: a bitcast does not convert pointers"},
        {"lds_tile_as_i32",
         ROCKE_ERR_TYPE,
         "smem<f32, [16x16]> is an LDS allocation, not a value with a bit width"},
    };
    for(const auto& r : refused)
    {
        rocke_status_t code;
        const build_outcome outcome = build_bitcast(*find_bitcast_case(r.name), &code);
        CHECK(!outcome.accepted);
        CHECK(code == r.code);
        CHECK(outcome.error == r.error);
        CHECK(outcome.ops_added == 0);
    }
    return 0;
}

/* A scalar's width is its format's width; an unknown format is refused with
 * core/dtypes.py's text, naming the spelling that was passed. */
static int test_scalar_widths()
{
    CHECK(rocke_i_type_bits(rocke_f64()) == 64);
    CHECK(rocke_i_type_bits(rocke_fp6e2m3()) == 6);
    CHECK(rocke_require_dtype_info("half") == rocke_dtype_info("f16"));
    try
    {
        rocke_require_dtype_info("custom_format");
        CHECK(false);
    }
    catch(const ckc::ValueError& error)
    {
        CHECK(strcmp(error.what(), "unknown dtype 'custom_format'") == 0);
    }
    return 0;
}

/* A dtype's logical IR type, as core/ir.py::dtype_to_ir_type gives it: any
 * accepted spelling resolves, and an unknown name is refused with
 * core/dtypes.py's text. */
static int test_ir_type_of_dtype()
{
    CHECK(rocke_require_ir_type("half") == rocke_f16());
    CHECK(rocke_require_ir_type("fp8") == rocke_fp8e4m3());
    try
    {
        rocke_require_ir_type("custom_format");
        CHECK(false);
    }
    catch(const ckc::ValueError& error)
    {
        CHECK(strcmp(error.what(), "unknown dtype 'custom_format'") == 0);
    }
    return 0;
}

/* One target-neutral MMA of `op_id` on i32 zero operands (the operands of
 * test_mma_frag_tables.py's twin). The builder types the result from the op_id
 * alone, so ordinary op_ids build; XF32 and scaled MFMA op_ids refuse these
 * operands, and an unknown op_id is refused. */
static build_outcome build_mma(const char* op_id, std::string* result_type)
{
    build_outcome outcome = {false, "", 0};
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "mma_case") != ROCKE_OK)
    {
        outcome.error = "builder init failed";
        return outcome;
    }
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int ops_before = b.kernel->body->num_ops;
    try
    {
        const rocke_value_t* result = rocke_b_mma(&b, op_id, zero, zero, zero, NULL, 0);
        *result_type = result->type->name;
        outcome.accepted = true;
    }
    catch(const ckc::Error& error)
    {
        outcome.error = error.what();
    }
    /* Counted after a refusal too: a refused call must leave no op behind. */
    outcome.ops_added = b.kernel->body->num_ops - ops_before;
    rocke_ir_builder_free(&b);
    return outcome;
}

static int print_mma(const char* op_id)
{
    std::string result_type;
    const build_outcome outcome = build_mma(op_id, &result_type);
    if(outcome.accepted)
        printf("%s\n", result_type.c_str());
    else
        printf("error: %s\n", outcome.error.c_str());
    return 0;
}

/* The operands are checked before the op_id, as in IRBuilder.mma: an XF32 op_id
 * on the wrong operands names the operands, and a refusal builds no op. */
static int test_mma_refusals()
{
    std::string result_type;
    const build_outcome built = build_mma("wmma_i32_16x16x16_iu8", &result_type);
    CHECK(built.accepted && built.ops_added == 1 && result_type == "vec<i32x8>");
    const build_outcome xf32 = build_mma("mfma_f32_16x16x8_xf32", &result_type);
    CHECK(!xf32.accepted && xf32.ops_added == 0);
    CHECK(xf32.error == "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator");
    const build_outcome unknown = build_mma("not_an_op_id", &result_type);
    CHECK(!unknown.accepted && unknown.ops_added == 0);
    CHECK(unknown.error.rfind("unknown MMA op_id 'not_an_op_id'; ", 0) == 0);
    return 0;
}

/* Every IR scalar type by name: one per rocke_scalar_kind_t, which the
 * static_assert and test_scalar_kinds_are_covered hold to. Widths are not
 * listed: they come from the engine. */
static const char* const k_scalar_names[]
    = {"i1",  "i4",   "u4",  "i8",      "u8",      "i16",     "i32",     "i64",     "f16",  "bf16",
       "f32", "tf32", "f64", "fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1", "e8m0", "e5m3"};
static_assert(sizeof k_scalar_names / sizeof k_scalar_names[0] == ROCKE_SCALAR__COUNT,
              "add the new scalar type's name to k_scalar_names");

/* Each name resolves to the singleton of that name, and no two names share a
 * scalar kind, so the list holds every kind exactly once. */
static int test_scalar_kinds_are_covered()
{
    bool seen[ROCKE_SCALAR__COUNT] = {};
    for(const char* name : k_scalar_names)
    {
        const rocke_type_t* t = rocke_scalar_by_name(name);
        CHECK(t != NULL && strcmp(t->name, name) == 0);
        CHECK(!seen[t->scalar]);
        seen[t->scalar] = true;
    }
    return 0;
}

static int print_scalar_bits()
{
    for(const char* name : k_scalar_names)
        printf("%s %d\n", name, rocke_i_type_bits(rocke_scalar_by_name(name)));
    return 0;
}

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--unsigned") == 0)
        return print_unsigned_case(argv[2]);
    if(argc == 4 && strcmp(argv[1], "--packed-only") == 0)
        return print_packed_only_value(argv[2], argv[3]);
    if(argc == 5 && strcmp(argv[1], "--smem-vector") == 0)
        return print_smem_vector_access(argv[2], argv[3], argv[4]);
    if(argc == 2 && strcmp(argv[1], "--scalar-bits") == 0)
        return print_scalar_bits();
    if(argc == 3 && strcmp(argv[1], "--bitcast") == 0)
        return print_bitcast(argv[2]);
    if(argc == 3 && strcmp(argv[1], "--mma") == 0)
        return print_mma(argv[2]);
    CHECK(argc == 1);
    CHECK(test_unsigned_ops() == 0);
    CHECK(test_packed_only_values() == 0);
    CHECK(test_smem_vector_elements() == 0);
    CHECK(test_bitcast_width() == 0);
    CHECK(test_scalar_widths() == 0);
    CHECK(test_ir_type_of_dtype() == 0);
    CHECK(test_mma_refusals() == 0);
    CHECK(test_scalar_kinds_are_covered() == 0);
    printf("rocke_ir_types: ok\n");
    return 0;
}
