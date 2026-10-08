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
 */
#include "rocke/ir.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "rocke/error.hpp"

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

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--unsigned") == 0)
        return print_unsigned_case(argv[2]);
    if(argc == 4 && strcmp(argv[1], "--packed-only") == 0)
        return print_packed_only_value(argv[2], argv[3]);
    if(argc == 5 && strcmp(argv[1], "--smem-vector") == 0)
        return print_smem_vector_access(argv[2], argv[3], argv[4]);
    CHECK(argc == 1);
    CHECK(test_unsigned_ops() == 0);
    CHECK(test_packed_only_values() == 0);
    CHECK(test_smem_vector_elements() == 0);
    printf("rocke_ir_types: ok\n");
    return 0;
}
