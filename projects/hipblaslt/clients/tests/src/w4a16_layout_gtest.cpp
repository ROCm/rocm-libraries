// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// W4A16: int4 encoding validation, the 8-row zero-point packing of the test
/// data, and kernel selection by scale mode, B scale and lda.

#include "w4a16_datagen.hpp"

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>
#include <cstring>

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
    // The layout does not depend on the scale type.
    constexpr auto    dtype = HIP_R_16BF;
    constexpr int64_t k     = 384;
    for(int64_t m : {7, 8, 9})
        for(int groupSize : {32, 128})
            for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                                 HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
            {
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
                        ASSERT_EQ(reference[element], float(hip_bfloat16((q - z) * scale)))
                            << "m=" << m << " row=" << row << " col=" << col
                            << " group=" << groupSize << " encoding=" << encoding;
                    }
                EXPECT_EQ(allocation.back(), 0xdeadbeef);
            }
}

namespace
{
    // group: 0, 1, 2 for G32, G64, G128. The modes run [zero-point][group].
    hipblasLtMatmulMatrixScale_t w4a16Mode(hipDataType dtype, int group, bool zp)
    {
        const int base = dtype == HIP_R_16BF ? HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT
                                             : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT;
        return static_cast<hipblasLtMatmulMatrixScale_t>(base + group + (zp ? 3 : 0));
    }

    class w4a16_b_scale : public ::testing::Test
    {
    protected:
        static constexpr int64_t m = 128, k = 256, n = 16;
        static constexpr float alpha = 1, beta = 0;
        hipblasLtHandle_t handle = nullptr;
        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
        void* storage = nullptr;

        void SetUp() override
        {
            int device;
            hipDeviceProp_t properties;
            ASSERT_EQ(hipGetDevice(&device), hipSuccess);
            ASSERT_EQ(hipGetDeviceProperties(&properties, device), hipSuccess);
            if(std::string(properties.gcnArchName).find("gfx1151") != 0)
                GTEST_SKIP() << "W4A16 custom kernels require gfx1151";
            ASSERT_EQ(hipMalloc(&storage, m * k * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            const size_t workspace = 32 * 1024 * 1024;
            ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
                HIPBLAS_STATUS_SUCCESS);
        }

        void release()
        {
            if(desc) hipblasLtMatmulDescDestroy(desc);
            for(auto* layout : {&a, &b, &c, &d})
                if(*layout) hipblasLtMatrixLayoutDestroy(*layout);
            desc = nullptr;
            a = b = c = d = nullptr;
        }

        void TearDown() override
        {
            release();
            if(pref) hipblasLtMatmulPreferenceDestroy(pref);
            if(handle) hipblasLtDestroy(handle);
            if(storage) EXPECT_EQ(hipFree(storage), hipSuccess);
        }

        // C API problem with a scalar B scale mode.
        void configure(hipDataType dtype, hipblasLtInt4Encoding_t encoding,
                       hipblasLtMatmulMatrixScale_t mode, int64_t cols, int64_t lda = k)
        {
            release();
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&a, HIP_R_4I, k, m, lda), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&b, dtype, k, cols, k), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&c, dtype, m, cols, m), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&d, dtype, m, cols, m), HIPBLAS_STATUS_SUCCESS);
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

        int heuristicCount(void* scaleB = nullptr)
        {
            EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)), HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulHeuristicResult_t result;
            int count = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 1, &result, &count),
                      HIPBLAS_STATUS_SUCCESS);
            return count;
        }

        // The same problem through the ext API.
        struct Ext
        {
            hipblaslt_ext::Gemm gemm;
            hipblaslt_ext::GemmProblemType problem;
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs inputs;
            hipblaslt_ext::GemmPreference preference;

            Ext(hipblasLtHandle_t handle, hipDataType dtype, hipblasLtInt4Encoding_t encoding,
                hipblasLtMatmulMatrixScale_t mode, void* storage)
                : gemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                       HIP_R_4I, dtype, dtype, dtype, HIPBLAS_COMPUTE_32F)
            {
                problem.setOpA(HIPBLAS_OP_T);
                problem.setOpB(HIPBLAS_OP_N);
                problem.setTypeA(HIP_R_4I);
                problem.setTypeB(dtype);
                problem.setTypeC(dtype);
                problem.setTypeD(dtype);
                problem.setTypeCompute(HIPBLAS_COMPUTE_32F);
                problem.setInt4EncodingA(encoding);
                epilogue.setScalingAType(mode);
                epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
                inputs.setA(storage);
                inputs.setB(storage);
                inputs.setC(storage);
                inputs.setD(storage);
                inputs.setAlpha(&alpha);
                inputs.setBeta(&beta);
                inputs.setScaleA(storage);
                preference.setMaxWorkspaceBytes(32 * 1024 * 1024);
            }

            hipblasStatus_t setProblem(int64_t lda = k)
            {
                return gemm.setProblem(m, n, k, 1, lda, k, m, m, 0, 0, 0, 0, epilogue, inputs, problem);
            }

            size_t count()
            {
                std::vector<hipblasLtMatmulHeuristicResult_t> results;
                EXPECT_EQ(gemm.algoGetHeuristic(1, preference, results), HIPBLAS_STATUS_SUCCESS);
                return results.size();
            }
        };
    };
}

TEST_F(w4a16_b_scale, every_mode_has_a_kernel_without_b_scale)
{
    for(auto dtype : {HIP_R_16F, HIP_R_16BF})
        for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                             HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
            for(bool zp : {false, true})
                for(int group = 0; group < 3; ++group)
                    for(int64_t cols : {int64_t(1), n})
                    {
                        SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " encoding="
                                     << encoding << " zp=" << zp << " group=" << group
                                     << " n=" << cols);
                        ASSERT_NO_FATAL_FAILURE(
                            configure(dtype, encoding, w4a16Mode(dtype, group, zp), cols));
                        EXPECT_EQ(heuristicCount(), 1);
                    }
}

TEST_F(w4a16_b_scale, scalar_b_has_no_matching_kernel)
{
    // No launch: selection must depend on pointer presence, not the scale value.
    const auto dtype = HIP_R_16F;
    const auto encoding = HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT;
    const auto mode = w4a16Mode(dtype, 2, true);
    ASSERT_NO_FATAL_FAILURE(configure(dtype, encoding, mode, n));
    EXPECT_EQ(heuristicCount(storage), 0);
    EXPECT_EQ(heuristicCount(), 1);

    Ext ext(handle, dtype, encoding, mode, storage);
    for(void* scaleB : {storage, static_cast<void*>(nullptr)})
    {
        ext.inputs.setScaleB(scaleB);
        ASSERT_EQ(ext.setProblem(), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(ext.count(), scaleB ? 0u : 1u);
    }

    // The same Gemm still serves a plain fp16 problem afterwards.
    ext.problem.setTypeA(dtype);
    ext.epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
    ext.inputs.setScaleA(nullptr);
    ASSERT_EQ(ext.setProblem(), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(ext.count(), 1u);
}

TEST_F(w4a16_b_scale, lda_must_cover_k_in_whole_dwords)
{
    // Kernels step A rows by lda >> 3 dwords.
    const auto encoding = HIPBLASLT_INT4_ENCODING_SIGNED_EXT;
    for(auto dtype : {HIP_R_16F, HIP_R_16BF})
    {
        const auto mode = w4a16Mode(dtype, 0, false);
        Ext ext(handle, dtype, encoding, mode, storage);
        for(int64_t lda : {k, k + 8, k - 8, k + 4})
        {
            const bool valid = lda >= k && lda % 8 == 0;
            SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " lda=" << lda);
            ASSERT_NO_FATAL_FAILURE(configure(dtype, encoding, mode, n, lda));
            if(valid)
                EXPECT_EQ(heuristicCount(), 1);
            else // Rejected during argument validation, before any algo or launch.
                EXPECT_EQ(hipblasLtMatmul(handle, desc, &alpha, storage, a, storage, b, &beta,
                                          storage, c, storage, d, nullptr, nullptr, 0, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

            ASSERT_EQ(ext.setProblem(lda),
                      valid ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_INVALID_VALUE);
            if(valid)
                EXPECT_EQ(ext.count(), 1u);
        }
    }
}
