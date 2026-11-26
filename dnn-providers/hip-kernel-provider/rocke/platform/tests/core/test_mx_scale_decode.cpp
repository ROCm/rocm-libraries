// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_mx_scale_decode.cpp -- C++ engine coverage for the E8M0 scale decoder,
 * the scaled-dot helper and the safe scale-code bound.
 *
 * Builds the same one-thread-per-code kernel as tests/core/test_mx_scale_decode.py
 * (load one scale byte, decode it, store the f32 multiplier) through the C++
 * decoder, and checks the decoder's lines in both C++ lowerers. The LLVM lines
 * are the exact text the Python test pins, which keeps the two engines
 * byte-identical for the decoder. The HIP check covers the byte read and the
 * NaN select; it leaves out the 127.0 constant line because the C++ HIP lowerer
 * spells integral float constants differently from the Python one (a known
 * divergence in rocke_h_f32_literal, not specific to this decoder).
 *
 * The decoder must read the byte as unsigned: zext, never sext, so codes
 * 128..255 keep their value (the following sitofp sees 0..255) and code 255
 * reaches the NaN select.
 */

#include "rocke/helper_rocke.helpers.mx_scale.h"
#include "rocke/instance_mx_gemm.h"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

/* The decoder's LLVM lines, as the Python test pins them. */
static const char* const k_expected_llvm
    = "  %zx3 = zext i8 %gl2 to i32\n"
      "  %eq5 = icmp eq i32 %zx3, 255\n"
      "  %sitof6 = sitofp i32 %zx3 to float\n"
      "  %fsub8 = fsub float %sitof6, 0x405FC00000000000\n"
      "  %exp29 = call float @llvm.exp2.f32(float %fsub8)\n"
      "  %sel11 = select i1 %eq5, float 0x7FF8000000000000, float %exp29\n";

static const char* const k_expected_hip_byte_read = " int zx3 = (int)(uint8_t)gl2;\n"
                                                    " int c4 = 255;\n"
                                                    " bool eq5 = zx3 == c4;\n"
                                                    " float sitof6 = (float)zx3;\n";

static const char* const k_expected_hip_nan_select = " float exp29 = exp2f(fsub8);\n"
                                                     " float c10 = ((float)NAN);\n"
                                                     " float sel11 = eq5 ? c10 : exp29;\n";

/* Out[tid] = decode_mx_scale_e8m0(Codes[tid]) */
static void build_decode_kernel(rocke_ir_builder_t* b)
{
    rocke_value_t* codes = rocke_b_param(b, "Codes", rocke_ptr_type(b, rocke_i8(), "global"), NULL);
    rocke_value_t* out = rocke_b_param(b, "Out", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* code = rocke_b_global_load_i8(b, codes, tid, 0);
    rocke_b_global_store(b, out, tid, rocke_decode_mx_scale_e8m0(b, code), 4);
    rocke_b_ret(b);
}

static int test_llvm_reads_the_byte_unsigned()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "mx_scale_decode") == ROCKE_OK);
    build_decode_kernel(&b);
    CHECK(rocke_ir_builder_ok(&b));

    char* llvm_text = NULL;
    CHECK(rocke_lower_kernel_to_llvm(b.kernel, ROCKE_LLVM_FLAVOR_LLVM20, "gfx90a", &llvm_text)
          == ROCKE_OK);
    if(std::strstr(llvm_text, k_expected_llvm) == NULL)
        std::fprintf(stderr, "LLVM text:\n%s\n", llvm_text);
    CHECK(std::strstr(llvm_text, k_expected_llvm) != NULL);
    CHECK(std::strstr(llvm_text, "sext") == NULL);
    std::free(llvm_text);
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_hip_reads_the_byte_unsigned()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "mx_scale_decode") == ROCKE_OK);
    build_decode_kernel(&b);
    CHECK(rocke_ir_builder_ok(&b));

    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx90a";
    opts.include_prologue = false;
    opts.include_prologue_set = true;
    CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip) == ROCKE_OK);
    if(std::strstr(hip.data, k_expected_hip_byte_read) == NULL
       || std::strstr(hip.data, k_expected_hip_nan_select) == NULL)
        std::fprintf(stderr, "HIP text:\n%s\n", hip.data);
    CHECK(std::strstr(hip.data, k_expected_hip_byte_read) != NULL);
    CHECK(std::strstr(hip.data, k_expected_hip_nan_select) != NULL);
    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&b);
    return 0;
}

/* A non-byte input is refused with the Python engine's message. The builder
 * either raises (ckc::ValueError) or latches the error, depending on the build. */
static int test_non_byte_input_is_rejected()
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "mx_scale_decode_i32") == ROCKE_OK);
    std::string err;
    try
    {
        CHECK(rocke_decode_mx_scale_e8m0(&b, rocke_b_const_i32(&b, 0)) == NULL);
        const char* msg = rocke_ir_builder_error(&b);
        err = msg ? msg : "";
    }
    catch(const std::exception& e)
    {
        err = e.what();
    }
    CHECK(err.find("decode_mx_scale_e8m0 expects i8 input, got i32") != std::string::npos);
    rocke_ir_builder_free(&b);
    return 0;
}

/* add_mx_scaled_dot refuses a non-vector dot, a mismatched accumulator, a
 * wrong code count and a non-byte scale code with the Python engine's messages
 * (tests/core/test_mx_scale_decode.py::test_scaled_dot_refuses_wrong_types). */
static int scaled_dot_refusal(int which_case, const char* expected)
{
    rocke_ir_builder_t b;
    CHECK(rocke_ir_builder_init(&b, "mx_scaled_dot_refusal") == ROCKE_OK);
    std::string err;
    try
    {
        rocke_value_t* dot = rocke_b_vector_splat(&b, rocke_b_const_f32(&b, 1.0), 4);
        rocke_value_t* acc = dot;
        rocke_value_t* codes
            = rocke_b_param(&b, "Codes", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        rocke_value_t* code = rocke_b_global_load_i8(&b, codes, rocke_b_const_i32(&b, 0), 0);
        rocke_value_t* a_codes[4] = {code, code, code, code};
        rocke_value_t* b_codes[4] = {code, code, code, code};
        size_t n_a = 4;
        if(which_case == 0)
            dot = rocke_b_const_f32(&b, 1.0);
        else if(which_case == 1)
            acc = rocke_b_vector_splat(&b, rocke_b_const_f32(&b, 1.0), 2);
        else if(which_case == 2)
            n_a = 3;
        else if(which_case == 3)
            b_codes[0] = rocke_b_const_i32(&b, 0);
        CHECK(rocke_add_mx_scaled_dot(&b, acc, dot, a_codes, n_a, b_codes, 4) == NULL);
        const char* msg = rocke_ir_builder_error(&b);
        err = msg ? msg : "";
    }
    catch(const std::exception& e)
    {
        err = e.what();
    }
    if(err.find(expected) == std::string::npos)
        std::fprintf(stderr, "got: %s\nwant: %s\n", err.c_str(), expected);
    CHECK(err.find(expected) != std::string::npos);
    rocke_ir_builder_free(&b);
    return 0;
}

static int test_scaled_dot_refuses_wrong_types()
{
    return scaled_dot_refusal(0, "add_mx_scaled_dot expects an f32 vector dot, got f32")
           || scaled_dot_refusal(1,
                                 "add_mx_scaled_dot expects the accumulator type to match the "
                                 "dot type vec<f32x4>, got vec<f32x2>")
           || scaled_dot_refusal(2,
                                 "add_mx_scaled_dot expects 4 A and 4 B scale bytes, got 3 "
                                 "and 4")
           || scaled_dot_refusal(3, "add_mx_scaled_dot expects i8 input, got i32");
}

/* largest_safe_mx_scale_code gives the Python engine's codes for the four fp8 /
 * bf8 encodings at 64 products and refuses a count that is not positive with
 * the same message (tests/core/test_mx_scale_decode.py::
 * test_safe_code_for_the_mfma_formats and
 * test_safe_code_refuses_a_count_that_is_not_positive). */
static int safe_code_is(const rocke_mx_element_format_t* format, int want)
{
    int code = -1;
    CHECK(rocke_largest_safe_mx_scale_code(format, format, 64, &code, NULL, 0) == ROCKE_OK);
    if(code != want)
        std::fprintf(stderr, "%s: got %d, want %d\n", format->name, code, want);
    CHECK(code == want);
    return 0;
}

static int test_safe_code()
{
    char err[256];
    int code = -1;
    if(safe_code_is(&ROCKE_FP8_E4M3_OCP, 178) || safe_code_is(&ROCKE_BF8_E5M2_OCP, 171)
       || safe_code_is(&ROCKE_FP8_E4M3_FNUZ, 179) || safe_code_is(&ROCKE_BF8_E5M2_FNUZ, 171))
        return 1;
    CHECK(rocke_largest_safe_mx_scale_code(
              &ROCKE_FP8_E4M3_OCP, &ROCKE_FP8_E4M3_OCP, 0, &code, err, sizeof(err))
          == ROCKE_ERR_VALUE);
    CHECK(std::strcmp(err, "largest_safe_mx_scale_code: products must be positive, got 0") == 0);
    return 0;
}

/* mx_gemm_element_format reads the FNUZ encodings on gfx942 and the OCP ones
 * on gfx950, and refuses other targets with the Python engine's message
 * (tests/instances/test_mx_gemm_scale_range.py::test_mx_gemm_element_format and
 * test_mx_gemm_element_format_refuses_other_targets). */
static int test_mx_gemm_element_formats()
{
    char err[256];
    CHECK(rocke_mx_gemm_element_format("fp8e4m3", "gfx942", NULL, 0) == &ROCKE_FP8_E4M3_FNUZ);
    CHECK(rocke_mx_gemm_element_format("bf8e5m2", "gfx942", NULL, 0) == &ROCKE_BF8_E5M2_FNUZ);
    CHECK(rocke_mx_gemm_element_format("fp8e4m3", "gfx950", NULL, 0) == &ROCKE_FP8_E4M3_OCP);
    CHECK(rocke_mx_gemm_element_format("bf8e5m2", "gfx950", NULL, 0) == &ROCKE_BF8_E5M2_OCP);
    CHECK(rocke_mx_gemm_element_format("fp8e4m3", "gfx90a", err, sizeof(err)) == NULL);
    CHECK(std::strcmp(err,
                      "mx_gemm: no fp8e4m3 element format known for gfx90a; the MX GEMM "
                      "reads fp8e4m3 / bf8e5m2 on gfx942 and gfx950")
          == 0);
    return 0;
}

int main()
{
    if(test_llvm_reads_the_byte_unsigned() || test_hip_reads_the_byte_unsigned()
       || test_non_byte_input_is_rejected() || test_scaled_dot_refuses_wrong_types()
       || test_safe_code() || test_mx_gemm_element_formats())
    {
        std::fprintf(stderr, "E8M0 scale helper contract failed\n");
        return 1;
    }
    return 0;
}
