// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_mx_scale_decode.cpp -- C++ engine coverage for the E8M0 scale decoder.
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

int main()
{
    if(test_llvm_reads_the_byte_unsigned() || test_hip_reads_the_byte_unsigned()
       || test_non_byte_input_is_rejected())
    {
        std::fprintf(stderr, "E8M0 scale decoder contract failed\n");
        return 1;
    }
    return 0;
}
