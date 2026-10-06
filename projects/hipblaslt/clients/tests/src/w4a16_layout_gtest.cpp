// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Verify 256-byte zero-point alignment; tensile_host.cpp must use the same layout.

#include "w4a16_datagen.hpp"

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <stdexcept>
#include <tuple>
#include <string>

#include <cstdint>
#include <cstring>

namespace
{
    /// As documented on HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_ZP_EXT.
    constexpr size_t c_documentedAlignment = 256;

    struct Shape
    {
        int64_t m;
        int64_t k;
        int     groupSize;
    };

    int64_t kGroupsOf(const Shape& s)
    {
        return (s.k + s.groupSize - 1) / s.groupSize;
    }

    size_t scaleRegionBytes(const Shape& s)
    {
        return static_cast<size_t>(s.m) * static_cast<size_t>(kGroupsOf(s)) * 2;
    }

    // The gtest sizes, plus ones landing the scale region short of, exactly on,
    // and just past a boundary.
    const Shape shapes[] = {
        {64, 256, 32},
        {128, 512, 32},
        {192, 256, 64},
        {65, 256, 64},      // odd M: zero-point rows pad to a multiple of eight
        {128, 320, 128},
        {128, 384, 128},    // odd number of K groups
        {2560, 32896, 128}, // a realistic weight matrix
        {1, 32, 32},        // scales far short of one boundary
        {128, 32, 32},      // scales exactly 256 bytes
        {129, 32, 32},      // one element past a boundary
        {2049, 8192, 128},
    };
}

TEST(w4a16_layout, zero_points_start_on_a_256_byte_boundary)
{
    for(const auto& s : shapes)
    {
        const size_t off = w4a16::zeroPointOffset(s.m, kGroupsOf(s));
        EXPECT_EQ(off % c_documentedAlignment, 0u)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize
            << ": zero-point offset " << off << " is not " << c_documentedAlignment
            << "-byte aligned";
    }
}

TEST(w4a16_layout, zero_points_start_past_the_scales)
{
    // Rounding up must not overlap the scales.
    for(const auto& s : shapes)
    {
        const size_t scales = scaleRegionBytes(s);
        const size_t off    = w4a16::zeroPointOffset(s.m, kGroupsOf(s));
        EXPECT_GE(off, scales) << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
        // ...nor waste a whole boundary.
        EXPECT_LT(off - scales, c_documentedAlignment)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
    }
}

TEST(w4a16_layout, allocation_covers_both_regions)
{
    for(const auto& s : shapes)
    {
        const int64_t kGroups = kGroupsOf(s);
        const size_t  zpBytes = static_cast<size_t>((s.m + 7) / 8) * static_cast<size_t>(kGroups) * 4;
        EXPECT_EQ(w4a16::scaleBytes(s.m, kGroups, true),
                  w4a16::zeroPointOffset(s.m, kGroups) + zpBytes)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
        // No zero-points, no padding.
        EXPECT_EQ(w4a16::scaleBytes(s.m, kGroups, false), scaleRegionBytes(s))
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
    }
}

TEST(w4a16_encoding, cpp_api_accepts_only_supported_encodings)
{
    hipblaslt_ext::GemmProblemType problem;
    for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                         HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
    {
        problem.setInt4EncodingA(encoding);
        EXPECT_EQ(problem.getInt4EncodingA(), encoding);
    }
    for(int value : {-1, 2, 3, 99})
    {
        EXPECT_THROW(problem.setInt4EncodingA(static_cast<hipblasLtInt4Encoding_t>(value)),
                     std::invalid_argument);
        EXPECT_EQ(problem.getInt4EncodingA(), HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT);
    }
}

TEST(w4a16_layout, eight_row_words_match_dequantized_reference)
{
    for(int64_t m : {1, 7, 8, 9, 65})
        for(int groupSize : {32, 64, 128})
            for(auto dtype : {HIP_R_16F, HIP_R_16BF})
                for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                                     HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
                {
                    constexpr int64_t k = 384;
                    const int64_t groups = k / groupSize;
                    const size_t offset = (m * groups * 2 + 255) & ~size_t(255);
                    const size_t bytes = offset + ((m + 7) / 8) * groups * 4;
                    std::vector<uint8_t> packed(m * k / 2, 0);
                    std::vector<uint32_t> allocation((bytes + 4) / 4, 0);
                    allocation.back() = 0xdeadbeef;
                    const auto reference = generateW4A16Input(packed.data(), allocation.data(),
                        dtype, m, k, k, groupSize, true, encoding);
                    const auto* scales = reinterpret_cast<const uint8_t*>(allocation.data());
                    for(int64_t row = 0; row < m; ++row)
                        for(int64_t col = 0; col < k; ++col)
                        {
                            uint32_t word;
                            std::memcpy(&word, scales + offset
                                + 4 * ((row / 8) * groups + col / groupSize), 4);
                            int z = (word >> (4 * (row % 8))) & 15;
                            const size_t element = row * k + col;
                            int q = (packed[element / 2] >> (4 * (element % 2))) & 15;
                            if(encoding == HIPBLASLT_INT4_ENCODING_SIGNED_EXT)
                            {
                                q = (q ^ 8) - 8;
                                z = (z ^ 8) - 8;
                            }
                            const float scale = w4a16::loadAs(scales,
                                row * groups + col / groupSize, dtype);
                            const float value = (q - z) * scale;
                            const float rounded = dtype == HIP_R_16F
                                ? float(hipblasLtHalf(value)) : float(hip_bfloat16(value));
                            ASSERT_EQ(reference[element], rounded)
                                << "m=" << m << " row=" << row << " col=" << col
                                << " group=" << groupSize << " dtype=" << dtype
                                << " encoding=" << encoding;
                        }
                    EXPECT_EQ(allocation.back(), 0xdeadbeef);
                }
}

namespace
{
    using BScaleCase = std::tuple<hipDataType, hipblasLtInt4Encoding_t,
                                 hipblasLtMatmulMatrixScale_t, int64_t>;

    class w4a16_b_scale : public ::testing::TestWithParam<BScaleCase>
    {
    protected:
        hipblasLtHandle_t handle = nullptr;
        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
        void* storage = nullptr;
        static constexpr int64_t m = 128, k = 256;

        void SetUp() override
        {
            int device;
            hipDeviceProp_t properties;
            ASSERT_EQ(hipGetDevice(&device), hipSuccess);
            ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
            if(std::string(properties.gcnArchName).find("gfx1151") != 0)
                GTEST_SKIP() << "W4A16 custom kernels require gfx1151";
            const auto [dtype, encoding, mode, n] = GetParam();
            ASSERT_EQ(hipMalloc(&storage, m * k * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&a, HIP_R_4I, k, m, k), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&b, dtype, k, n, k), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&c, dtype, m, n, m), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&d, dtype, m, n, m), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            const size_t workspace = 32 * 1024 * 1024;
            ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
                HIPBLAS_STATUS_SUCCESS);
            const hipblasOperation_t trans = HIPBLAS_OP_T;
            const auto scalar = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT, &encoding, sizeof(encoding)),
                HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scalar, sizeof(scalar)), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &storage, sizeof(storage)), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(pref) hipblasLtMatmulPreferenceDestroy(pref);
            if(desc) hipblasLtMatmulDescDestroy(desc);
            if(a) hipblasLtMatrixLayoutDestroy(a);
            if(b) hipblasLtMatrixLayoutDestroy(b);
            if(c) hipblasLtMatrixLayoutDestroy(c);
            if(d) hipblasLtMatrixLayoutDestroy(d);
            if(handle) hipblasLtDestroy(handle);
            if(storage) EXPECT_EQ(hipFree(storage), hipSuccess);
        }
    };

    std::vector<BScaleCase> bScaleCases()
    {
        std::vector<BScaleCase> result;
        for(auto dtype : {HIP_R_16F, HIP_R_16BF})
            for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                                 HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
                for(bool zp : {false, true})
                {
                    for(int group = 0; group < 3; ++group)
                        for(int64_t n : {1, 16})
                            result.emplace_back(dtype, encoding,
                                static_cast<hipblasLtMatmulMatrixScale_t>(1006 + group + (zp ? 3 : 0)), n);
                }
        return result;
    }
}

TEST_P(w4a16_b_scale, c_api_scalar_b_has_no_matching_kernel)
{
    // No launch: selection must depend on pointer presence, not the scale value.
    for(void* scaleB : {static_cast<void*>(nullptr), storage, static_cast<void*>(nullptr)})
    {
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
            desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)), HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t result;
        int count = -1;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 1, &result, &count),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(count, scaleB ? 0 : 1);
    }
}

TEST_P(w4a16_b_scale, cpp_api_scalar_b_has_no_matching_kernel)
{
    const auto [dtype, encoding, mode, n] = GetParam();
    hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                            HIP_R_4I, dtype, dtype, dtype, HIPBLAS_COMPUTE_32F);
    hipblaslt_ext::GemmProblemType problem;
    problem.setOpA(HIPBLAS_OP_T);
    problem.setOpB(HIPBLAS_OP_N);
    problem.setTypeA(HIP_R_4I);
    problem.setTypeB(dtype);
    problem.setTypeC(dtype);
    problem.setTypeD(dtype);
    problem.setTypeCompute(HIPBLAS_COMPUTE_32F);
    problem.setInt4EncodingA(encoding);
    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setScalingAType(mode);
    epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
    hipblaslt_ext::GemmInputs inputs;
    const float alpha = 1, beta = 0;
    inputs.setA(storage);
    inputs.setB(storage);
    inputs.setC(storage);
    inputs.setD(storage);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setScaleA(storage);
    hipblaslt_ext::GemmPreference preference;
    preference.setMaxWorkspaceBytes(32 * 1024 * 1024);
    for(void* scaleB : {static_cast<void*>(nullptr), storage, static_cast<void*>(nullptr)})
    {
        inputs.setScaleB(scaleB);
        ASSERT_EQ(gemm.setProblem(m, n, k, 1, k, k, m, m, 0, 0, 0, 0, epilogue, inputs, problem),
                  HIPBLAS_STATUS_SUCCESS);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        ASSERT_EQ(gemm.algoGetHeuristic(1, preference, results), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(results.size(), scaleB ? 0u : 1u);
    }

    problem.setTypeA(dtype);
    epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
    inputs.setScaleA(nullptr);
    ASSERT_EQ(gemm.setProblem(m, n, k, 1, k, k, m, m, 0, 0, 0, 0, epilogue, inputs, problem),
              HIPBLAS_STATUS_SUCCESS);
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    ASSERT_EQ(gemm.algoGetHeuristic(1, preference, results), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(results.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(shipped_modes, w4a16_b_scale, ::testing::ValuesIn(bScaleCases()));
