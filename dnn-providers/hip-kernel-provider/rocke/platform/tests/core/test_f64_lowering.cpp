// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_f64_lowering.cpp -- C++ engine coverage for the f64 op boundary.
 * Mirrors TestHipF64Rejection / TestF64VectorArith in
 * tests/core/test_amdgpu_lowering_ops.py, and the C++ builder side of
 * tests/core/test_f64_conversions.py (the explicit f32 <-> f64 conversions)
 * and tests/core/test_zero_vec_and_lds_element_types.py (zero_vec of f64 and
 * i16, LDS loads of i16). The LDS pool layout cases at the end also cover
 * the segment alignment of tests/core/test_smem_access_alignment.py through
 * the C API: a vector access moves its segment to the vector alignment, and
 * a store alignment that is not an integer power of two is refused.
 */

#include "rocke/error.hpp"
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

/* The f64 mean is the sum divided by N (one rounding), not the sum times
 * 1/N, which rounds twice. N=3072 has no exact reciprocal; its double bit
 * pattern is 0x40A8000000000000. */
static int test_mean_divides_by_n()
{
    char err[256];
    char* llvm_text = NULL;

    rocke_reduce2d_spec_t rd = rocke_reduce2d_spec_default();
    rd.n_per_block = 3072;
    rd.op = "mean";
    rd.dtype = "f64";
    rd.vec = 2;
    CHECK(rocke_reduce2d_lower_to_llvm(
              &rd, "gfx942", ROCKE_LLVM_FLAVOR_LLVM20, &llvm_text, err, sizeof(err))
          == ROCKE_OK);
    const char* div = std::strstr(llvm_text, " = fdiv ");
    CHECK(div != NULL);
    CHECK(std::strstr(div + 1, " = fdiv ") == NULL);
    const char* div_end = std::strchr(div, '\n');
    CHECK(div_end != NULL);
    const std::string div_line(div, div_end);
    CHECK(div_line.find(" = fdiv double %") == 0);
    CHECK(div_line.find(", 0x40A8000000000000") == div_line.size() - 20);
    CHECK(std::strstr(llvm_text, "fmul double") == NULL);
    std::free(llvm_text);
    return 0;
}

/* An f64 LDS allocation is aligned to its 8-byte element. A live 2-byte f16
 * allocation ends at byte 2; the 4-byte default would place the f64 buffer
 * at byte 4, where each double straddles two 8-byte words. Mirrors
 * test_f64_alloc_after_two_byte_alloc_is_8b_aligned in
 * tests/core/test_smem_liveness_packing_extra.py. */
static int test_lds_allocation_is_8_byte_aligned()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "f64_lds") == ROCKE_OK);
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f64(), "global"), NULL);
    rocke_value_t* half_in
        = rocke_b_param(&b, "half_in", rocke_ptr_type(&b, rocke_f16(), "global"), NULL);
    rocke_value_t* idx = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int f64_shape[] = {2};
    rocke_value_t* half_lds = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "half_lds");
    rocke_value_t* f64_lds = rocke_b_smem_alloc(&b, rocke_f64(), f64_shape, 1, "f64_lds");
    rocke_value_t* h = rocke_b_global_load(&b, half_in, idx, rocke_f16(), 2);
    /* One double: the buffer's 8-byte alignment is all a store of it claims. */
    rocke_value_t* v = rocke_b_global_load_f64(&b, out, idx, 8);
    rocke_value_t* const lds_idx[] = {idx};
    /* The f16 buffer is written before and after the f64 one, so both are
     * live together and the f64 buffer cannot reuse byte 0. */
    rocke_b_smem_store_f16(&b, half_lds, lds_idx, 1, h);
    rocke_b_smem_store_vN(&b, f64_lds, lds_idx, 1, v, 1);
    rocke_b_smem_store_f16(&b, half_lds, lds_idx, 1, h);
    rocke_b_ret(&b);
    CHECK(rocke_ir_builder_ok(&b));

    char* llvm_text = NULL;
    CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", &llvm_text)
          == ROCKE_OK);
    CHECK(std::strstr(llvm_text,
                      "= getelementptr inbounds i8, ptr addrspace(3) @smem_pool.f64_lds, i32 8\n")
          != NULL);
    /* The double is stored alone, claiming the 8 bytes its offset gives. */
    const char* store = std::strstr(llvm_text, "store double %");
    CHECK(store != NULL);
    const std::string store_line(store, std::strchr(store, '\n'));
    CHECK(store_line.find("ptr addrspace(3)") != std::string::npos);
    CHECK(store_line.size() > 9 && store_line.compare(store_line.size() - 9, 9, ", align 8") == 0);
    std::free(llvm_text);
    rocke_ir_builder_free(&b);
    return 0;
}

/* A one-element f16 buffer, then an f32 buffer written and read as
 * <4 x float>. Both are live together (the f16 store comes before and after
 * the vector traffic), so the f32 buffer cannot share offset 0. When
 * `overwrite_align` is given it rewrites the vector store's `align`
 * attribute, as serialized or hand-written IR may carry it. Mirrors
 * _f16_then_f32x4 in tests/core/test_smem_access_alignment.py. */
using align_overwrite_fn = void (*)(rocke_ir_builder_t*, rocke_attr_map_t*);

static rocke_status_t
    lower_f16_then_f32x4(align_overwrite_fn overwrite_align, char** llvm_text, char* err)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "lds_align_f16_f32x4") != ROCKE_OK)
        return ROCKE_ERR_OOM;
    rocke_value_t* out = rocke_b_param(&b, "out", rocke_ptr_type(&b, rocke_f32(), "global"), NULL);
    rocke_value_t* zero = rocke_b_const_i32(&b, 0);
    const int half_shape[] = {1};
    const int float_shape[] = {8};
    rocke_value_t* half_buf = rocke_b_smem_alloc(&b, rocke_f16(), half_shape, 1, "h");
    rocke_value_t* float_buf = rocke_b_smem_alloc(&b, rocke_f32(), float_shape, 1, "f");
    rocke_value_t* h = rocke_b_trunc_f32_to_f16(&b, rocke_b_const_f32(&b, 1.0));
    rocke_value_t* const at_zero[] = {zero};
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, h);
    rocke_value_t* v = rocke_b_global_load_vN(&b, out, zero, rocke_f32(), 4, 16);
    rocke_value_t* const at_four[] = {rocke_b_const_i32(&b, 4)};
    rocke_b_smem_store_vN(&b, float_buf, at_four, 1, v, 4);
    rocke_value_t* w = rocke_b_smem_load_vN(&b, float_buf, at_zero, 1, rocke_f32(), 4);
    rocke_b_global_store_vN(&b, out, zero, w, 4, 16);
    rocke_b_smem_store_f16(&b, half_buf, at_zero, 1, h);
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
    rocke_status_t st
        = rocke_ir_builder_ok(&b)
              ? rocke_lower_kernel_to_llvm_ex(
                    b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx942", llvm_text, err, ROCKE_ERR_MSG_CAP)
              : ROCKE_ERR_VALUE;
    rocke_ir_builder_free(&b);
    return st;
}

/* The f32 segment after a 2-byte f16 buffer lands at byte 16, not 4, so the
 * <4 x float> store's `align 16` holds. */
static int test_vector_access_moves_its_segment_to_the_vector_alignment()
{
    char err[ROCKE_ERR_MSG_CAP];
    char* llvm_text = NULL;
    CHECK(lower_f16_then_f32x4(nullptr, &llvm_text, err) == ROCKE_OK);
    CHECK(std::strstr(llvm_text,
                      "= getelementptr inbounds i8, ptr addrspace(3) "
                      "@smem_pool.lds_align_f16_f32x4, i32 16\n")
          != NULL);
    const char* store = std::strstr(llvm_text, "store <4 x float> %");
    CHECK(store != NULL);
    const std::string store_line(store, std::strchr(store, '\n'));
    CHECK(store_line.find("ptr addrspace(3)") != std::string::npos);
    CHECK(store_line.size() > 10
          && store_line.compare(store_line.size() - 10, 10, ", align 16") == 0);
    std::free(llvm_text);
    return 0;
}

/* A store `align` that is not a positive power of two is refused, not capped
 * to the 16-byte pool alignment (24 would otherwise print as a valid 16). */
static int test_store_alignment_not_a_power_of_two_is_refused()
{
    char err[ROCKE_ERR_MSG_CAP];
    char* llvm_text = NULL;
    const align_overwrite_fn align_24 = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_int(b, attrs, "align", 24);
    };
    CHECK(lower_f16_then_f32x4(align_24, &llvm_text, err) == ROCKE_ERR_VALUE);
    CHECK(llvm_text == NULL);
    CHECK(std::strstr(err, "tile.smem_store_vN failed - align=24, expected a positive power of two")
          != NULL);
    return 0;
}

/* A store `align` that is not an integer is the wrong kind of value: it is
 * refused with ROCKE_ERR_TYPE, never read as a missing attribute (which would
 * fall back to the vector width). */
static int test_store_alignment_not_an_integer_is_refused()
{
    char err[ROCKE_ERR_MSG_CAP];
    char* llvm_text = NULL;
    const align_overwrite_fn align_true = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_bool(b, attrs, "align", true);
    };
    CHECK(lower_f16_then_f32x4(align_true, &llvm_text, err) == ROCKE_ERR_TYPE);
    CHECK(llvm_text == NULL);
    CHECK(std::strstr(err, "tile.smem_store_vN failed - align=bool, expected an integer") != NULL);
    const align_overwrite_fn align_float = [](rocke_ir_builder_t* b, rocke_attr_map_t* attrs) {
        rocke_attr_set_float(b, attrs, "align", 24.0);
    };
    CHECK(lower_f16_then_f32x4(align_float, &llvm_text, err) == ROCKE_ERR_TYPE);
    CHECK(llvm_text == NULL);
    CHECK(std::strstr(err, "tile.smem_store_vN failed - align=float, expected an integer") != NULL);
    return 0;
}

int main()
{
    if(test_unsupported_ops_rejected() || test_rcp_divides_in_double()
       || test_fabs_stays_in_double() || test_vector_binops_use_float_ops()
       || test_conversions_lower_to_one_instruction_each()
       || test_conversions_reject_the_wrong_source_type()
       || test_zero_vec_and_lds_load_admit_f64_and_i16()
       || test_scalar_global_accesses_are_8_byte_aligned() || test_mean_divides_by_n()
       || test_lds_allocation_is_8_byte_aligned()
       || test_vector_access_moves_its_segment_to_the_vector_alignment()
       || test_store_alignment_not_a_power_of_two_is_refused()
       || test_store_alignment_not_an_integer_is_refused())
    {
        std::fprintf(stderr, "f64 lowering contract failed\n");
        return 1;
    }
    return 0;
}
