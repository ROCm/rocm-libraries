/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Tests for the handle-based composable fused-epilogue API (RMSNorm focus).
//
// The lightweight tests cover descriptor lifecycle, chain ordering, attribute validation, and
// CPU reference math. They also verify attach-time completeness for residual-add, full RMSNorm,
// decomposed partial-stats producer, decomposed scale-apply consumer, and requant stages.
//
// The gfx950 E2E tests drive real kernels:
//  - Full RMSNorm: K1 PartialRMS + row_div reduce-and-apply.
//  - Residual-add + full RMSNorm: K1 PartialRMS residual path + row_div.
//  - Decomposed consumer: GEMM2 + K3 RstdScale using a test-populated handoff rstd.
//  - Decomposed two-call flow: producer K1 + row_rstd fills the handoff, then consumer K3
//    applies the per-row rstd.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <random>
#include <string>
#include <vector>

namespace
{
    void cpuRmsNorm(float*       out,
                    const float* in,
                    const float* gamma,
                    std::size_t  rows,
                    std::size_t  cols,
                    float        eps)
    {
        for(std::size_t row = 0; row < rows; ++row)
        {
            const auto offset = row * cols;
            float      sum_sq = 0.0f;
            for(std::size_t col = 0; col < cols; ++col)
                sum_sq += in[offset + col] * in[offset + col];

            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
            for(std::size_t col = 0; col < cols; ++col)
                out[offset + col] = in[offset + col] * inv_rms * gamma[col];
        }
    }

    class FusedEpilogueTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(fused)
                hipblasLtFusedEpilogueDestroy(fused);
            if(stats)
                hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
        }

        // Set gamma/eps so an RMSNorm handle passes attach-time validation.
        void completeRmsnorm()
        {
            int   dummy_gamma_storage = 0;
            void* gamma               = &dummy_gamma_storage;
            const float eps           = 1e-5f;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Set the residual input pointer so a residual-add handle passes attach-time
        // validation. Without a residual output pointer, the API uses this pointer as the
        // in-place destination for the updated residual stream.
        void completeResidual()
        {
            int   dummy_residual_storage = 0;
            void* residual               = &dummy_residual_storage;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Create and set the opaque RMSNorm handoff descriptor so a decomposed producer or
        // consumer handle passes attach-time validation.
        void completeStats()
        {
            ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t attach()
        {
            // The attribute value is the handle (a pointer); pass its pointer-sized storage.
            return hipblasLtMatmulDescSetAttribute(desc,
                                                   HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                   &fused,
                                                   sizeof(hipblasLtFusedEpilogueDescriptor_t));
        }

        hipblasLtMatmulDesc_t                     desc  = nullptr;
        hipblasLtFusedEpilogueDescriptor_t        fused = nullptr;
        hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    };
}

// ---- RMSNorm math reference ----

TEST(FusedEpilogueMath, rmsnormReferenceNormalizesRows)
{
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> gamma(cols, 1.0f);
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    for(std::size_t row = 0; row < rows; ++row)
    {
        float mean_sq = 0.0f;
        for(std::size_t col = 0; col < cols; ++col)
        {
            const auto v = output[row * cols + col];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(cols);
        EXPECT_NEAR(mean_sq, 1.0f, 1e-6f);
    }
}

TEST(FusedEpilogueMath, rmsnormReferenceAppliesGamma)
{
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> gamma = {1.0f, 0.5f, 2.0f, -1.0f};
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    const float inv_rms = 1.0f / std::sqrt(7.5f);
    EXPECT_NEAR(output[0], 1.0f * inv_rms, 1e-6f);
    EXPECT_NEAR(output[1], 2.0f * inv_rms * 0.5f, 1e-6f);
    EXPECT_NEAR(output[2], 3.0f * inv_rms * 2.0f, 1e-6f);
    EXPECT_NEAR(output[3], 4.0f * inv_rms * -1.0f, 1e-6f);
}

// ---- Lifecycle ----

TEST(FusedEpilogueLifecycle, createAddDestroy)
{
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(fused, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, createNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateDestroy)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(stats, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: ordering legality ----

TEST_F(FusedEpilogueTest, legalOrderAccepted)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, illegalOrderRejected)
{
    // Requant then RMSNorm violates the supported RMSNorm order (requant must come last).
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, amaxAfterRequantRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, swigluRejectedByRmsnormChainValidator)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, rmsnormBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, duplicateStageRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, unknownEpilogueRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, static_cast<hipblasLtFuseableEpilogue_t>(999)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: decomposed-flow ordering and family mixing ----

TEST_F(FusedEpilogueTest, decomposedProducerOrderAccepted)
{
    // Producer chain: residual add -> partial RMSNorm stats.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, decomposedProducerRequantOrderAccepted)
{
    // Dynamic-quantized producer chain: residual add -> partial RMSNorm stats -> requant.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, decomposedConsumerAccepted)
{
    // Consumer chain: RMSNorm scale-apply only.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, decomposedConsumerRequantRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, partialStatsBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, fullRmsnormThenPartialStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, partialStatsThenFullRmsnormRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, producerAndConsumerStagesInOneChainRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- SetAttribute validation ----

TEST_F(FusedEpilogueTest, setUnknownAttributeRejected)
{
    const float eps = 1e-5f;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, static_cast<hipblasLtFusedEpilogueAttribute_t>(999), &eps, sizeof(eps)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualPointerRejected)
{
    void* residual = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormGammaRejected)
{
    void* gamma = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormStatsRejected)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t null_stats = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &null_stats, sizeof(null_stats)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualOutputAcceptedAsInPlace)
{
    void* residual_output = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, setInvalidRequantComputeModeRejected)
{
    auto mode = static_cast<hipblasLtRequantScaleComputeMode_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setInvalidRequantGranularityRejected)
{
    auto granularity = static_cast<hipblasLtRequantScaleGranularity_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                                 &granularity,
                                                 sizeof(granularity)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Attach-time completeness validation ----

TEST_F(FusedEpilogueTest, attachResidualMissingPointerRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    // residual pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachResidualInPlaceWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    // No residual-output pointer is required; unset means update the residual input in place.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualSeparateWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualOutputCanBeClearedToInPlace)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    residual_output = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingGammaRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    const float eps = 1e-5f;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    // gamma never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy = 0;
    void* gamma = &dummy;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_SUCCESS);
    // eps never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

// ---- Attach-time completeness validation: decomposed flow ----

TEST_F(FusedEpilogueTest, attachPartialStatsMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachPartialStatsMissingGammaEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // gamma/eps never set -> attach must reject (the producer computes the partial stats).
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteProducerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    completeStats();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachPartialStatsRequantStaticPolicyRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    completeStats();

    int   dummy_scale_storage = 0;
    void* scale               = &dummy_scale_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &scale, sizeof(scale)),
              HIPBLAS_STATUS_SUCCESS);

    // The CODA producer requant path requires dynamic per-row scale.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteDynamicQuantizedProducerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    completeStats();

    int   dummy_scale_storage = 0;
    void* scale               = &dummy_scale_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &scale, sizeof(scale)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtRequantScaleComputeMode_t mode = HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t granularity = HIPBLASLT_REQUANT_SCALE_PER_ROW;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                                 &granularity,
                                                 sizeof(granularity)),
              HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachScaleApplyMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteConsumerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // scale-apply only needs the handoff descriptor; gamma/eps live on the producer.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachNullFusedEpilogueDetaches)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t null_fused = nullptr;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                              HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                              &null_fused,
                                              sizeof(null_fused)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRequantMissingScaleRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    // scale pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRequantAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy_scale_storage = 0;
    void* scale               = &dummy_scale_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &scale, sizeof(scale)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

// ---- End-to-end numeric test: full RMSNorm flow on device ----
//
// Drives a real bf16 TN matmul through hipblasLtMatmul with a full-RMSNorm fused epilogue
// attached, then compares D against a CPU reference RMSNorm(alpha * op(A)*op(B), gamma, eps).
// This exercises the wired path end to end: solution selection (UsePartialRMS predicate),
// K1 (GEMM + partial-stats producer), and Kernel 2 (row_div reduce-and-apply). gfx950-only,
// since the PartialRMS solution + row_div code object ship for gfx950.

namespace
{
    inline uint16_t f32_to_bf16(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round to nearest even.
        const uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return static_cast<uint16_t>(bits >> 16);
    }

    inline float bf16_to_f32(uint16_t h)
    {
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        float          f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    bool deviceIsGfx950()
    {
        int dev = 0;
        if(hipGetDevice(&dev) != hipSuccess)
            return false;
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).rfind("gfx950", 0) == 0;
    }

    void fillRandomBf16(std::vector<uint16_t>& values,
                        std::mt19937&         rng,
                        std::uniform_real_distribution<float>& dist)
    {
        for(auto& x : values)
            x = f32_to_bf16(dist(rng));
    }

    hipblasStatus_t runBf16TnFusedMatmul(hipblasLtHandle_t                   handle,
                                         int64_t                            m,
                                         int64_t                            n,
                                         int64_t                            k,
                                         void*                              dA,
                                         int64_t                            lda,
                                         void*                              dB,
                                         void*                              dC,
                                         void*                              dD,
                                         hipblasLtFusedEpilogueDescriptor_t fused,
                                         void*                              dWorkspace,
                                         size_t                             workspaceSize,
                                         int&                               algoCount)
    {
        // TN bf16 GEMM with col-major A/B/C/D descriptors. The lda override lets the
        // decomposed consumer feed a row-major [M, N_hidden] producer output as op(A)^T.
        algoCount = 0;

        hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
        hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, k, m, lda);
        hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, k, n, k);
        hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, m, n, m);
        hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, m, n, m);

        hipblasLtMatmulDesc_t mm = nullptr;
        hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(
            mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));

        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

        hipblasLtMatmulHeuristicResult_t heur[1];
        hipblasLtMatmulAlgoGetHeuristic(
            handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount);

        const float     alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        if(algoCount > 0)
        {
            status = hipblasLtMatmul(handle,
                                     mm,
                                     &alpha,
                                     dA,
                                     layA,
                                     dB,
                                     layB,
                                     &beta,
                                     dC,
                                     layC,
                                     dD,
                                     layD,
                                     &heur[0].algo,
                                     dWorkspace,
                                     workspaceSize,
                                     nullptr);
        }

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(mm);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layC);
        hipblasLtMatrixLayoutDestroy(layD);
        return status;
    }

    hipblasStatus_t runBf16TnFusedMatmulFp8D(hipblasLtHandle_t                   handle,
                                             int64_t                            m,
                                             int64_t                            n,
                                             int64_t                            k,
                                             void*                              dA,
                                             int64_t                            lda,
                                             void*                              dB,
                                             void*                              dC,
                                             void*                              dD,
                                             hipblasLtFusedEpilogueDescriptor_t fused,
                                             void*                              dWorkspace,
                                             size_t                             workspaceSize,
                                             int&                               algoCount)
    {
        algoCount = 0;

        hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
        hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, k, m, lda);
        hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, k, n, k);
        hipblasLtMatrixLayoutCreate(&layC, HIP_R_8F_E4M3, m, n, m);
        hipblasLtMatrixLayoutCreate(&layD, HIP_R_8F_E4M3, m, n, m);

        hipblasLtMatmulDesc_t mm = nullptr;
        hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(
            mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));

        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

        hipblasLtMatmulHeuristicResult_t heur[1];
        hipblasLtMatmulAlgoGetHeuristic(
            handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount);

        const float     alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        if(algoCount > 0)
        {
            status = hipblasLtMatmul(handle,
                                     mm,
                                     &alpha,
                                     dA,
                                     layA,
                                     dB,
                                     layB,
                                     &beta,
                                     dC,
                                     layC,
                                     dD,
                                     layD,
                                     &heur[0].algo,
                                     dWorkspace,
                                     workspaceSize,
                                     nullptr);
        }

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(mm);
        hipblasLtMatrixLayoutDestroy(layA);
        hipblasLtMatrixLayoutDestroy(layB);
        hipblasLtMatrixLayoutDestroy(layC);
        hipblasLtMatrixLayoutDestroy(layD);
        return status;
    }

    std::vector<float> referenceRmsNormRows(const std::vector<uint16_t>& hA,
                                            const std::vector<uint16_t>& hB,
                                            const std::vector<uint16_t>& hGamma,
                                            const std::vector<uint16_t>* hResidual,
                                            int64_t                      m,
                                            int64_t                      n,
                                            int64_t                      k,
                                            float                        eps)
    {
        std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
        std::vector<float> gamma(n);
        for(int64_t col = 0; col < n; ++col)
            gamma[col] = bf16_to_f32(hGamma[col]);

        for(int64_t row = 0; row < m; ++row)
        {
            std::vector<float> h1(n);
            float              sumSq = 0.0f;
            for(int64_t col = 0; col < n; ++col)
            {
                float acc = 0.0f;
                for(int64_t kk = 0; kk < k; ++kk)
                    acc += bf16_to_f32(hA[kk + row * k]) * bf16_to_f32(hB[kk + col * k]);
                if(hResidual)
                    acc += bf16_to_f32((*hResidual)[row * n + col]);
                h1[col] = acc;
                sumSq += acc * acc;
            }

            const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(n) + eps);
            for(int64_t col = 0; col < n; ++col)
                out[row * n + col] = h1[col] * invRms * gamma[col];
        }
        return out;
    }

    void expectBf16Near(const std::vector<uint16_t>& actual,
                        const std::vector<float>&    expected,
                        float                        absTol = 5e-5f,
                        float                        relTol = 5e-2f)
    {
        ASSERT_EQ(actual.size(), expected.size());
        size_t mismatches = 0;
        double maxAbsErr  = 0.0;
        double maxRelErr  = 0.0;
        for(size_t i = 0; i < actual.size(); ++i)
        {
            const float  got   = bf16_to_f32(actual[i]);
            const float  ref   = expected[i];
            const float  abs   = std::abs(got - ref);
            const float  denom = std::max(std::abs(ref), 1e-3f);
            const double rel   = abs / denom;
            maxAbsErr          = std::max(maxAbsErr, static_cast<double>(abs));
            maxRelErr          = std::max(maxRelErr, rel);
            if(abs > std::max(absTol, relTol * std::abs(ref)))
                ++mismatches;
        }
        EXPECT_EQ(mismatches, 0u) << "max abs error " << maxAbsErr << ", max relative error "
                                  << maxRelErr;
    }

    void runFullRmsNormE2E(bool residualAdd)
    {
        // TN, bf16, col-major. Shape matches a gfx950 PartialRMS logic entry.
        const int64_t M = 1024, N = 1024, K = 4096;
        const float   eps = 1e-5f;

        // Host inputs. op(A)=T => A stored K x M col-major; op(B)=N => B stored K x N col-major.
        std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
        std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
        std::vector<uint16_t> hGamma(N);
        std::vector<uint16_t> hResidual(residualAdd ? static_cast<size_t>(M) * N : 0);
        std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);

        std::mt19937                          rng(123);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
        fillRandomBf16(hA, rng, dist);
        fillRandomBf16(hB, rng, dist);
        fillRandomBf16(hGamma, rng, gdist);
        fillRandomBf16(hResidual, rng, dist);

        void*        dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr,
              *dGamma = nullptr, *dResidual = nullptr, *dWs = nullptr;
        const size_t wsSize = size_t(256) * 1024 * 1024;
        ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
        if(residualAdd)
            ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
        dC = dD; // beta = 0, C unused numerically but must be a valid pointer.

        ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t),
                            hipMemcpyHostToDevice),
                  hipSuccess);
        if(residualAdd)
            ASSERT_EQ(hipMemcpy(dResidual,
                                hResidual.data(),
                                hResidual.size() * sizeof(uint16_t),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
        ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
        if(residualAdd)
        {
            ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused,
                          HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                          &dResidual,
                          sizeof(dResidual)),
                      HIPBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
                  HIPBLAS_STATUS_SUCCESS);

        int algoCount = 0;
        ASSERT_EQ(runBf16TnFusedMatmul(
                      handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(algoCount, 0)
            << (residualAdd ? "no residualAdd PartialRMS solution selected"
                            : "no PartialRMS solution selected")
            << " for the fused RMSNorm problem";
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
                  hipSuccess);
        expectBf16Near(
            hD, referenceRmsNormRows(hA, hB, hGamma, residualAdd ? &hResidual : nullptr, M, N, K, eps));

        hipblasLtFusedEpilogueDestroy(fused);
        hipblasLtDestroy(handle);
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dD));
        static_cast<void>(hipFree(dGamma));
        if(dResidual)
            static_cast<void>(hipFree(dResidual));
        static_cast<void>(hipFree(dWs));
    }
}

TEST(FusedEpilogueE2E, fullRmsNormMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (PartialRMS) is wired for gfx950 only";

    runFullRmsNormE2E(/*residualAdd=*/false);
}

// ---- End-to-end numeric test: full RMSNorm flow with a fused residual-add stage ----
//
// Same full flow as above, but with RESIDUAL_ADD before RMSNorm. This exercises
// the residualAdd=True PartialRMS solution and cache-key path.

TEST(FusedEpilogueE2E, fullRmsNormResidualAddMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (PartialRMS) is wired for gfx950 only";

    runFullRmsNormE2E(/*residualAdd=*/true);
}

TEST(FusedEpilogueE2E, fullRmsNormResidualAddRequantMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm requant is wired for gfx950 only";

    using fp8 = hipblaslt_f8;
    auto f8ToF = [](fp8 v) { return static_cast<float>(static_cast<_Float16>(v)); };

    // TN, BF16 inputs with FP8 D. Shape matches the gfx950 PartialRMS residual-add logic.
    const int64_t M = 1024, N = 1024, K = 4096;
    const float   eps = 1e-5f;
    const float   dequantScale = 0.125f;

    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    std::vector<uint16_t> hResidual(static_cast<size_t>(M) * N);
    std::vector<fp8>      hC(static_cast<size_t>(M) * N, fp8(0.0f));
    std::vector<fp8>      hD(static_cast<size_t>(M) * N);

    std::mt19937                          rng(2026);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hResidual, rng, dist);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dGamma = nullptr,
         *dResidual = nullptr, *dScale = nullptr, *dWs = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dC, hC.size() * sizeof(fp8)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(fp8)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dScale, sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dC, hC.data(), hC.size() * sizeof(fp8), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dScale, &dequantScale, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(fp8)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &dScale, sizeof(dScale)),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleComputeMode_t mode = HIPBLASLT_REQUANT_SCALE_STATIC;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t granularity = HIPBLASLT_REQUANT_SCALE_PER_TENSOR;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                                 &granularity,
                                                 sizeof(granularity)),
              HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    const hipblasStatus_t status = runBf16TnFusedMatmulFp8D(
        handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount);
    ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no PartialRMS requant solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(fp8), hipMemcpyDeviceToHost),
              hipSuccess);

    const auto reference = referenceRmsNormRows(hA, hB, hGamma, &hResidual, M, N, K, eps);
    size_t     mismatches = 0;
    double     maxAbsErr  = 0.0;
    for(size_t i = 0; i < hD.size(); ++i)
    {
        const float got = f8ToF(hD[i]) * dequantScale;
        const float ref = reference[i];
        const float abs = std::abs(got - ref);
        maxAbsErr       = std::max(maxAbsErr, static_cast<double>(abs));
        if(abs > std::max(0.08f, 0.12f * std::abs(ref)))
            ++mismatches;
    }
    EXPECT_EQ(mismatches, 0u) << "max abs error " << maxAbsErr;

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dC));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dScale));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end numeric test: decomposed RMSNorm consumer (Kernel 3 RstdScale) ----
//
// Exercises the decomposed flow's consumer stage in isolation: a GEMM2 with the
// HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY epilogue multiplies each output row by a
// pre-computed per-row rstd carried in the handoff descriptor (K3 RstdScale, normal
// orientation, no reduction). This test injects a host-computed rstd into the opaque handoff via
// a test-only hook so the consumer can be exercised independently of the producer. Verifies
// D[m,n] = (alpha * op(A)*op(B))[m,n] * rstd[m]. gfx950-only.

// Test-only hook (defined in amd_detail/hipblaslt.cpp) to populate the opaque RMSNorm handoff
// descriptor with a caller-provided device rstd buffer.
extern "C++" bool rocblaslt_rmsnorm_handoff_set_scale_for_testing(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc, void* per_row_scale);

namespace
{
    void createScaleApplyDescriptor(hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                    hipblasLtFusedEpilogueDescriptor_t*       fused)
    {
        ASSERT_EQ(hipblasLtFusedEpilogueCreate(fused), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueAdd(*fused,
                                            HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      *fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
                  HIPBLAS_STATUS_SUCCESS);
    }

    void createPartialStatsDescriptor(hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                      void*                                     gamma,
                                      float                                     eps,
                                      hipblasLtFusedEpilogueDescriptor_t*       fused)
    {
        ASSERT_EQ(hipblasLtFusedEpilogueCreate(fused), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueAdd(*fused,
                                            HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      *fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      *fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                      *fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
                  HIPBLAS_STATUS_SUCCESS);
    }
}

TEST(FusedEpilogueE2E, decomposedScaleApplyMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (RstdScale) is wired for gfx950 only";

    // TN, bf16, col-major. K3 RstdScale library tiles are N_out=64 wide; K = N_hidden.
    const int64_t M = 256, N = 64, K = 64;
    const float   alpha = 1.0f;

    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);
    std::vector<float>    hRstd(static_cast<size_t>(M));

    std::mt19937                          rng(321);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> rdist(0.25f, 1.75f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    for(auto& r : hRstd)
        r = rdist(rng); // arbitrary per-row scale standing in for the producer's rstd

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dRstd = nullptr,
         *dWs                = nullptr;
    const size_t wsSize      = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, hRstd.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;
    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dRstd, hRstd.data(), hRstd.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Decomposed handoff, populated with the host rstd via the test-only hook.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(rocblaslt_rmsnorm_handoff_set_scale_for_testing(stats, dRstd));

    // Consumer chain: RMSNorm scale-apply reads the deferred per-row scale from the handoff.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_NO_FATAL_FAILURE(createScaleApplyDescriptor(stats, &cons));

    int algoCount = 0;
    ASSERT_EQ(runBf16TnFusedMatmul(handle, M, N, K, dA, K, dB, dC, dD, cons, dWs, wsSize,
                                   algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no RstdScale (K3) solution selected for the scale-apply problem";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // Reference: D[m,n] = (alpha * sum_k A[k,m]*B[k,n]) * rstd[m]. K3 keeps the normal
    // orientation (per-M-row scale, no reduction), so D is col-major [M, N].
    std::vector<float> expected(static_cast<size_t>(M) * N);
    for(int64_t m = 0; m < M; ++m)
        for(int64_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + m * K]) * bf16_to_f32(hB[kk + n * K]);
            expected[n * M + m] = acc * alpha * hRstd[m]; // D col-major [M, N]
        }
    expectBf16Near(hD, expected);

    hipblasLtFusedEpilogueDestroy(cons);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dRstd));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end: decomposed producer(GEMM1) -> consumer(GEMM2) two-call flow ----
//
// The full decomposed RMSNorm flow across two matmul calls linked by the library-populated
// RMSNorm handoff descriptor (no test hook):
//   GEMM1 (producer, PARTIAL_RMSNORM_STATS): h2 = (x @ W0) * gamma  [M, N_hidden]; the library
//     runs K1 (PartialRMS) + row_rstd, stashing rstd = rsqrt(mean(h1^2)+eps) in the handoff.
//   GEMM2 (consumer, RMSNORM_SCALE_APPLY):    y  = rstd * (h2 @ W1)  [M, N_out] via Kernel 3.
// The combined result equals RMSNorm(x @ W0) @ W1. gamma=1 keeps the reference simple. TN bf16;
// h2 is produced row-major [M, N_hidden] and fed to GEMM2 as its TN A operand (lda=N_hidden).
// Needs a merged K1(PartialRMS)+K3(RstdScale) gfx950 library; gfx950-only.
TEST(FusedEpilogueE2E, decomposedProducerConsumerMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed RMSNorm flow is wired for gfx950 only";

    const int64_t M = 1024, Nhidden = 1024, K0 = 64, Nout = 64;
    const float   eps = 1e-5f;

    std::vector<uint16_t> hX(static_cast<size_t>(K0) * M);
    std::vector<uint16_t> hW0(static_cast<size_t>(K0) * Nhidden);
    std::vector<uint16_t> hW1(static_cast<size_t>(Nhidden) * Nout);
    std::vector<uint16_t> hGamma(static_cast<size_t>(Nhidden), f32_to_bf16(1.0f)); // gamma = 1

    std::mt19937                          rng(4242);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    fillRandomBf16(hX, rng, dist);
    fillRandomBf16(hW0, rng, dist);
    fillRandomBf16(hW1, rng, dist);

    void *dX = nullptr, *dW0 = nullptr, *dGamma = nullptr, *dH2 = nullptr, *dW1 = nullptr,
         *dD2                = nullptr, *dWs = nullptr;
    const size_t wsSize      = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dX, hX.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW0, hW0.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dH2, size_t(M) * Nhidden * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, size_t(M) * Nout * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dW0, hW0.data(), hW0.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * 2, hipMemcpyHostToDevice), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Library-populated handoff shared by both calls (no test hook).
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);

    // Producer chain: partial RMSNorm stats + gamma + eps + handoff.
    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_NO_FATAL_FAILURE(createPartialStatsDescriptor(stats, dGamma, eps, &prod));

    // GEMM1 producer: h2 [M, N_hidden] (row-major) + rstd stashed in the handoff.
    int algoCount = 0;
    ASSERT_EQ(runBf16TnFusedMatmul(
                  handle, M, Nhidden, K0, dX, K0, dW0, dH2, dH2, prod, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no PartialRMS (K1) producer solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Consumer chain: scale-apply + the same handoff.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_NO_FATAL_FAILURE(createScaleApplyDescriptor(stats, &cons));

    // GEMM2 consumer: h2 (row-major [M, N_hidden]) is the TN A operand [N_hidden, M] (lda=N_hidden).
    ASSERT_EQ(runBf16TnFusedMatmul(handle,
                                   M,
                                   Nout,
                                   Nhidden,
                                   dH2,
                                   Nhidden,
                                   dW1,
                                   dD2,
                                   dD2,
                                   cons,
                                   dWs,
                                   wsSize,
                                   algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no RstdScale (K3) consumer solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint16_t> hD2(static_cast<size_t>(M) * Nout);
    ASSERT_EQ(hipMemcpy(hD2.data(), dD2, hD2.size() * 2, hipMemcpyDeviceToHost), hipSuccess);

    // Reference: gemm1 = x@W0 (TN); rstd = rsqrt(mean(gemm1^2)+eps); y = rstd * (gemm1 @ W1).
    std::vector<float> gemm1(static_cast<size_t>(M) * Nhidden);
    std::vector<float> rstd(static_cast<size_t>(M));
    for(int64_t m = 0; m < M; ++m)
    {
        float ss = 0.0f;
        for(int64_t j = 0; j < Nhidden; ++j)
        {
            float acc = 0.0f;
            for(int64_t k = 0; k < K0; ++k)
                acc += bf16_to_f32(hX[k + m * K0]) * bf16_to_f32(hW0[k + j * K0]);
            gemm1[m * Nhidden + j] = acc;
            ss += acc * acc;
        }
        rstd[m] = 1.0f / std::sqrt(ss / static_cast<float>(Nhidden) + eps);
    }
    // Consumer reference. Mirror the device's bf16 storage: h2 = bf16(gemm1) (gamma=1) is stored
    // by the producer, GEMM2 reads it, then y = bf16(rstd * (h2 @ W1)). Compare with a combined
    // absolute+relative tolerance so near-zero cancellation elements (tiny ref) do not blow up a
    // pure relative metric.
    std::vector<float> expected(static_cast<size_t>(M) * Nout);
    for(int64_t m = 0; m < M; ++m)
        for(int64_t n = 0; n < Nout; ++n)
        {
            float acc = 0.0f;
            for(int64_t j = 0; j < Nhidden; ++j)
            {
                const float h2bf = bf16_to_f32(f32_to_bf16(gemm1[m * Nhidden + j])); // gamma=1
                acc += h2bf * bf16_to_f32(hW1[j + n * Nhidden]);
            }
            expected[n * M + m] = bf16_to_f32(f32_to_bf16(acc * rstd[m])); // D2 col-major
        }
    expectBf16Near(hD2, expected, 3e-2f);

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueDestroy(cons);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dX));
    static_cast<void>(hipFree(dW0));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dH2));
    static_cast<void>(hipFree(dW1));
    static_cast<void>(hipFree(dD2));
    static_cast<void>(hipFree(dWs));
}
