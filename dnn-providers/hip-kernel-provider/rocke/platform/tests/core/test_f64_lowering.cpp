// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_f64_lowering.cpp -- C++ engine coverage for the f64 op boundary.
 * Mirrors TestHipF64Rejection / TestF64VectorArith in
 * tests/core/test_amdgpu_lowering_ops.py, and the C++ builder side of
 * tests/core/test_f64_conversions.py (the explicit f32 <-> f64 conversions)
 * and tests/core/test_zero_vec_and_lds_element_types.py (zero_vec of f64 and
 * i16, LDS loads of i16).
 */

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

enum f64_op
{
    OP_EXP2,
    OP_LOG2,
    OP_SQRT,
    OP_RSQRT,
    OP_RCP_FAST,
    OP_FABS,
    OP_FMA,
    OP_FMAX3,
    OP_FMIN3,
    OP_VECTOR_FMA,
    OP_RCP,
    OP_VECTOR_ADD,
    OP_VECTOR_SUB,
    OP_VECTOR_MUL,
};

/* f64 `out[0] = op(out[0], ...)`; vector ops use a 2-wide load/store. */
static rocke_value_t* build_f64_kernel(rocke_ir_builder_t* b, f64_op op)
{
    rocke_value_t* out = rocke_b_param(b, "out", rocke_ptr_type(b, rocke_f64(), "global"), NULL);
    rocke_value_t* idx = rocke_b_const_i32(b, 0);
    rocke_value_t* r = NULL;
    if(op >= OP_VECTOR_FMA && op != OP_RCP)
    {
        rocke_value_t* v = rocke_b_global_load_vN(b, out, idx, rocke_f64(), 2, 16);
        r = op == OP_VECTOR_FMA   ? rocke_b_vector_fma(b, v, v, v)
            : op == OP_VECTOR_ADD ? rocke_b_vector_add(b, v, v)
            : op == OP_VECTOR_SUB ? rocke_b_vector_sub(b, v, v)
                                  : rocke_b_vector_mul(b, v, v);
        rocke_b_global_store_vN(b, out, idx, r, 2, 16);
    }
    else
    {
        rocke_value_t* x = rocke_b_global_load(b, out, idx, rocke_f64(), 8);
        switch(op)
        {
        case OP_EXP2:
            r = rocke_b_exp2(b, x);
            break;
        case OP_LOG2:
            r = rocke_b_log2(b, x);
            break;
        case OP_SQRT:
            r = rocke_b_sqrt(b, x);
            break;
        case OP_RSQRT:
            r = rocke_b_rsqrt(b, x);
            break;
        case OP_RCP_FAST:
            r = rocke_b_rcp_fast(b, x);
            break;
        case OP_FABS:
            r = rocke_b_fabs(b, x);
            break;
        case OP_FMA:
            r = rocke_b_fma(b, x, x, x);
            break;
        case OP_FMAX3:
            r = rocke_b_fmax3(b, x, x, x);
            break;
        case OP_FMIN3:
            r = rocke_b_fmin3(b, x, x, x);
            break;
        default:
            r = rocke_b_rcp(b, x);
            break;
        }
        rocke_b_global_store(b, out, idx, r, 8);
    }
    rocke_b_ret(b);
    return r;
}

static int test_unsupported_ops_rejected()
{
    for(int op = OP_EXP2; op <= OP_VECTOR_FMA; ++op)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "f64_reject") == ROCKE_OK);
        build_f64_kernel(&b, static_cast<f64_op>(op));
        CHECK(rocke_ir_builder_ok(&b));

        rocke_strbuf_t hip;
        CHECK(rocke_strbuf_init(&hip, 0) == 0);
        rocke_lower_hip_opts_t opts = {};
        opts.arch = "gfx942";
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_ERR_NOTIMPL);
        rocke_strbuf_free(&hip);

        char* llvm_text = NULL;
        CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text)
              == ROCKE_ERR_NOTIMPL);
        std::free(llvm_text);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_rcp_divides_in_double()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "f64_rcp") == ROCKE_OK);
    build_f64_kernel(&b, OP_RCP);

    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx942";
    CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
    CHECK(std::strstr(hip.data, "= 1.0 / ") != NULL);
    CHECK(std::strstr(hip.data, "__builtin_amdgcn_rcpf") == NULL);
    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_vector_binops_use_float_ops()
{
    const struct
    {
        f64_op op;
        const char* llvm_op;
    } cases[] = {{OP_VECTOR_ADD, "fadd"}, {OP_VECTOR_SUB, "fsub"}, {OP_VECTOR_MUL, "fmul"}};
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "f64_vector") == ROCKE_OK);
        rocke_value_t* r = build_f64_kernel(&b, c.op);

        char* llvm_text = NULL;
        CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text)
              == ROCKE_OK);
        char expected[64];
        std::snprintf(expected, sizeof(expected), "%s = %s <2 x double>", r->name, c.llvm_op);
        CHECK(std::strstr(llvm_text, expected) != NULL);
        std::free(llvm_text);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

/* out[0] = f64(f32(out[0])): one conversion each way. */
static int test_conversions_lower_to_one_instruction_each()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "f64_conversions") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f64(), "global"), NULL);
    rocke_value_t* idx = rocke_b_const_i32(&b, 0);
    rocke_value_t* narrowed
        = rocke_b_cvt_f64_to_f32(&b, rocke_b_global_load(&b, out, idx, rocke_f64(), 8));
    rocke_value_t* widened = rocke_b_cvt_f32_to_f64(&b, narrowed);
    rocke_b_global_store(&b, out, idx, widened, 8);
    rocke_b_ret(&b);
    CHECK(rocke_ir_builder_ok(&b));

    char* llvm_text = NULL;
    CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text)
          == ROCKE_OK);
    char expected[96];
    std::snprintf(expected, sizeof(expected), "%s = fptrunc double ", narrowed->name);
    CHECK(std::strstr(llvm_text, expected) != NULL);
    std::snprintf(
        expected, sizeof(expected), "%s = fpext float %s to double", widened->name, narrowed->name);
    CHECK(std::strstr(llvm_text, expected) != NULL);
    std::free(llvm_text);

    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx942";
    CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
    CHECK(std::strstr(hip.data, " = (float)") != NULL);
    CHECK(std::strstr(hip.data, " = (double)") != NULL);
    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&b);
    return 0;
}

/* Each conversion takes exactly its source type, with the Python builder's message,
 * and emits no op when it refuses. */
static int test_conversions_reject_the_wrong_source_type()
{
    const struct
    {
        bool from_f32;
        const char* message;
    } cases[] = {{true, "cvt_f32_to_f64 expects f32 input, got f64"},
                 {false, "cvt_f64_to_f32 expects f64 input, got f32"}};
    for(const auto& c : cases)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "f64_conversion_reject") == ROCKE_OK);
        rocke_value_t* wrong = rocke_b_param(&b, "x", c.from_f32 ? rocke_f64() : rocke_f32(), NULL);
        int before = b.kernel->body->num_ops;
        bool rejected = false;
        try
        {
            c.from_f32 ? rocke_b_cvt_f32_to_f64(&b, wrong) : rocke_b_cvt_f64_to_f32(&b, wrong);
        }
        catch(const ckc::Error& error)
        {
            rejected = error.code() == ROCKE_ERR_VALUE && std::strcmp(error.what(), c.message) == 0;
        }
        CHECK(rejected);
        CHECK(b.kernel->body->num_ops == before);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_zero_vec_and_lds_load_admit_f64_and_i16()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "zero_vec_and_lds") == ROCKE_OK);
    CHECK(rocke_b_zero_vec(&b, rocke_f64(), 2) != NULL);
    CHECK(rocke_b_zero_vec(&b, rocke_i16(), 4) != NULL);
    const int shape[] = {256};
    rocke_value_t* smem = rocke_b_smem_alloc(&b, rocke_i16(), shape, 1, "halves");
    rocke_value_t* index = rocke_b_thread_id_x(&b);
    CHECK(rocke_b_smem_load_vN(&b, smem, &index, 1, rocke_i16(), 4) != NULL);
    rocke_b_ret(&b);
    CHECK(rocke_ir_builder_ok(&b));
    rocke_ir_builder_free(&b);
    return 0;
}

int main()
{
    if(test_unsupported_ops_rejected() || test_rcp_divides_in_double()
       || test_vector_binops_use_float_ops() || test_conversions_lower_to_one_instruction_each()
       || test_conversions_reject_the_wrong_source_type()
       || test_zero_vec_and_lds_load_admit_f64_and_i16())
    {
        std::fprintf(stderr, "f64 lowering contract failed\n");
        return 1;
    }
    return 0;
}
