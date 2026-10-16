// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_f64_lowering.cpp -- C++ engine coverage for the f64 op boundary.
 * Mirrors TestHipF64Rejection / TestF64VectorArith in
 * tests/core/test_amdgpu_lowering_ops.py.
 */

#include "rocke/instance_elementwise.h"
#include "rocke/instance_reduce.h"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
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

enum f64_op
{
    OP_EXP2,
    OP_LOG2,
    OP_SQRT,
    OP_RSQRT,
    OP_RCP_FAST,
    OP_FMA,
    OP_FMAX3,
    OP_FMIN3,
    OP_VECTOR_FMA,
    OP_RCP,
    OP_FABS,
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
    if(op == OP_VECTOR_FMA || op >= OP_VECTOR_ADD)
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
        case OP_FMA:
            r = rocke_b_fma(b, x, x, x);
            break;
        case OP_FMAX3:
            r = rocke_b_fmax3(b, x, x, x);
            break;
        case OP_FMIN3:
            r = rocke_b_fmin3(b, x, x, x);
            break;
        case OP_FABS:
            r = rocke_b_fabs(b, x);
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

/* fabs stays in double in both lowerings (the f32 helper would round). */
static int test_fabs_stays_in_double()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "f64_fabs") == ROCKE_OK);
    build_f64_kernel(&b, OP_FABS);

    char* llvm_text = NULL;
    CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text)
          == ROCKE_OK);
    CHECK(std::strstr(llvm_text, "declare double @llvm.fabs.f64(double)") != NULL);
    CHECK(std::strstr(llvm_text, "= call double @llvm.fabs.f64(double %") != NULL);
    std::free(llvm_text);

    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx942";
    CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
    CHECK(std::strstr(hip.data, " = fabs(") != NULL);
    CHECK(std::strstr(hip.data, "fabsf") == NULL);
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

/* Number of scalar f64 global loads and stores in `ll`, or -1 if any of them
 * claims less than the 8-byte element alignment. */
static int count_aligned_scalar_f64_global_accesses(const char* ll)
{
    const std::string text(ll);
    const std::string aligned = ", align 8";
    int count = 0;
    size_t start = 0;
    while(start < text.size())
    {
        size_t end = text.find('\n', start);
        if(end == std::string::npos)
            end = text.size();
        const std::string line = text.substr(start, end - start);
        const bool load = line.find("load double, ptr addrspace(1)") != std::string::npos;
        const bool store = line.find("store double ") != std::string::npos
                           && line.find("ptr addrspace(1)") != std::string::npos;
        if(load || store)
        {
            if(line.size() < aligned.size()
               || line.compare(line.size() - aligned.size(), aligned.size(), aligned) != 0)
                return -1;
            ++count;
        }
        start = end + 1;
    }
    return count;
}

/* The elementwise tail (two loads and one store per lane, vec=2) and the
 * reduce output store claim 8-byte alignment, so they do not depend on the
 * target's unaligned-access mode. */
static int test_scalar_global_accesses_are_8_byte_aligned()
{
    char err[256];
    char* llvm_text = NULL;

    rocke_elementwise_spec_t ew = rocke_elementwise_spec_default();
    ew.op = "add";
    ew.dtype = "f64";
    ew.vec = 2;
    CHECK(rocke_elementwise_lower_to_llvm(
              &ew, "gfx942", ROCKE_LLVM_FLAVOR_LLVM20, &llvm_text, err, sizeof(err))
          == ROCKE_OK);
    CHECK(count_aligned_scalar_f64_global_accesses(llvm_text) == 6);
    std::free(llvm_text);

    rocke_reduce2d_spec_t rd = rocke_reduce2d_spec_default();
    rd.n_per_block = 4096;
    rd.dtype = "f64";
    rd.vec = 2;
    llvm_text = NULL;
    CHECK(rocke_reduce2d_lower_to_llvm(
              &rd, "gfx942", ROCKE_LLVM_FLAVOR_LLVM20, &llvm_text, err, sizeof(err))
          == ROCKE_OK);
    CHECK(count_aligned_scalar_f64_global_accesses(llvm_text) == 1);
    std::free(llvm_text);
    return 0;
}

int main()
{
    if(test_unsupported_ops_rejected() || test_rcp_divides_in_double()
       || test_fabs_stays_in_double() || test_vector_binops_use_float_ops()
       || test_scalar_global_accesses_are_8_byte_aligned())
    {
        std::fprintf(stderr, "f64 lowering contract failed\n");
        return 1;
    }
    return 0;
}
