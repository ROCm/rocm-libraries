// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_mx_scale_decode.cpp -- C++ engine coverage for the E8M0 scale decoder
 * and the scale-pair fma helper.
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

/* fma_with_mx_scale_pair refuses a missing scale split, a non-vector value, a
 * mismatched addend and a non-byte scale code with the Python engine's
 * messages (tests/core/test_mx_scale_decode.py::
 * test_scale_pair_fma_refuses_wrong_types). */
static int scale_pair_refusal(int which_case, const char* expected)
{
    rocke_ir_builder_t b;
    rocke_mx_scale_split_t split;
    CHECK(rocke_exact_mx_scale_split(&ROCKE_BF8_E5M2_OCP, &ROCKE_BF8_E5M2_OCP, 32, &split, NULL, 0)
          == ROCKE_OK);
    CHECK(rocke_ir_builder_init(&b, "mx_scale_pair_refusal") == ROCKE_OK);
    std::string err;
    try
    {
        rocke_value_t* value = rocke_b_vector_splat(&b, rocke_b_const_f32(&b, 1.0), 4);
        rocke_value_t* addend = value;
        rocke_value_t* codes
            = rocke_b_param(&b, "Codes", rocke_ptr_type(&b, rocke_i8(), "global"), NULL);
        rocke_value_t* a_code = rocke_b_global_load_i8(&b, codes, rocke_b_const_i32(&b, 0), 0);
        rocke_value_t* b_code = a_code;
        if(which_case == 0)
            value = rocke_b_const_f32(&b, 1.0);
        else if(which_case == 1)
            addend = rocke_b_vector_splat(&b, rocke_b_const_f32(&b, 1.0), 2);
        else if(which_case == 2)
            b_code = rocke_b_const_i32(&b, 0);
        CHECK(rocke_fma_with_mx_scale_pair(
                  &b, value, a_code, b_code, addend, which_case == 3 ? NULL : &split)
              == NULL);
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

static int test_scale_pair_fma_refuses_wrong_types()
{
    return scale_pair_refusal(0, "fma_with_mx_scale_pair expects an f32 vector value, got f32")
           || scale_pair_refusal(1,
                                 "fma_with_mx_scale_pair expects the addend type to match the "
                                 "value type vec<f32x4>, got vec<f32x2>")
           || scale_pair_refusal(2, "fma_with_mx_scale_pair expects i8 input, got i32")
           || scale_pair_refusal(3, "fma_with_mx_scale_pair expects an MxScaleSplit, got None");
}

/* exact_mx_scale_split derives the same split as the Python engine for the four
 * fp8 / bf8 encodings, and refuses formats whose bounds admit no exact split
 * with the same messages
 * (tests/core/test_mx_scale_decode.py::test_exact_split_for_the_mfma_formats and
 * test_exact_split_refuses_formats_without_one). */
static int split_cap_is(const rocke_mx_element_format_t* format, int cap)
{
    rocke_mx_scale_split_t split;
    CHECK(rocke_exact_mx_scale_split(format, format, 32, &split, NULL, 0) == ROCKE_OK);
    CHECK(split.exponent_cap == cap);
    CHECK(split.fused_min == -149);
    CHECK(split.fused_max == 127);
    CHECK(rocke_mx_scale_split_pre_max(&split) == cap - 127);
    return 0;
}

static int split_refusal(const rocke_mx_element_format_t* format, const char* expected)
{
    rocke_mx_scale_split_t split;
    char err[512];
    CHECK(rocke_exact_mx_scale_split(format, format, 32, &split, err, sizeof(err))
          == ROCKE_ERR_VALUE);
    if(std::strcmp(err, expected) != 0)
        std::fprintf(stderr, "got: %s\nwant: %s\n", err, expected);
    CHECK(std::strcmp(err, expected) == 0);
    return 0;
}

static int test_exact_split()
{
    static const rocke_mx_element_format_t too_large = {"wide", -9, 1.0e30};
    static const rocke_mx_element_format_t too_small = {"fine", -60, 1.0};
    static const rocke_mx_element_format_t too_wide = {"spread", -22, 1099511627776.0};
    return split_cap_is(&ROCKE_FP8_E4M3_OCP, 232) || split_cap_is(&ROCKE_BF8_E5M2_OCP, 218)
           || split_cap_is(&ROCKE_FP8_E4M3_FNUZ, 234) || split_cap_is(&ROCKE_BF8_E5M2_FNUZ, 218)
           || split_refusal(&too_large,
                            "exact_mx_scale_split: the dot product of 32 products of wide and "
                            "wide values can reach 2^204, beyond the f32 range")
           || split_refusal(&too_small,
                            "exact_mx_scale_split: the dot product of 32 products of fine and "
                            "fine values can be as small as 2^-120; at the smallest scales "
                            "(ea + eb = -254) it needs a factor of 2^-105 before the fma, "
                            "which would round it")
           || split_refusal(&too_wide,
                            "exact_mx_scale_split: the dot product of 32 products of spread "
                            "and spread values spans 2^-44 to 2^86, too wide to saturate "
                            "ea + eb at 169 without changing a result");
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
       || test_non_byte_input_is_rejected() || test_scale_pair_fma_refuses_wrong_types()
       || test_exact_split() || test_mx_gemm_element_formats())
    {
        std::fprintf(stderr, "E8M0 scale helper contract failed\n");
        return 1;
    }
    return 0;
}
