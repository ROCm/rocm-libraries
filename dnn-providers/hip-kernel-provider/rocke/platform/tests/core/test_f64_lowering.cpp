// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_f64_lowering.cpp -- C++ engine coverage for the f64 op boundary.
 * Mirrors TestHipF64Rejection / TestF64VectorArith in
 * tests/core/test_amdgpu_lowering_ops.py.
 */

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

int main()
{
    if(test_unsupported_ops_rejected() || test_rcp_divides_in_double()
       || test_vector_binops_use_float_ops())
    {
        std::fprintf(stderr, "f64 lowering contract failed\n");
        return 1;
    }
    return 0;
}
