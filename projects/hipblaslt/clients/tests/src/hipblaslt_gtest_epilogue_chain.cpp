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
#include <hipblaslt/hipblaslt_float8.h>
#include <random>
#include <string>
#include <vector>

static void cpuRmsNorm(
    float* out, const float* in, const float* gamma, std::size_t rows, std::size_t cols, float eps)
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

namespace
{
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
            int         dummy_gamma_storage = 0;
            void*       gamma               = &dummy_gamma_storage;
            const float eps                 = 1e-5f;
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
            ASSERT_EQ(
                hipblasLtFusedEpilogueSetAttribute(
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
    constexpr std::size_t    rows  = 2;
    constexpr std::size_t    cols  = 4;
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
    constexpr std::size_t    rows  = 1;
    constexpr std::size_t    cols  = 4;
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

TEST(FusedEpilogueLifecycle, rmsnormStatsBufferValidation)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);

    float storage = 0.0f;
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(nullptr, &storage, sizeof(storage)),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, nullptr, sizeof(storage)),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, &storage, 0),
              HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, &storage, sizeof(storage)),
              HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats), HIPBLAS_STATUS_SUCCESS);
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
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                  desc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &null_fused, sizeof(null_fused)),
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

static inline uint16_t f32_to_bf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even.
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

static inline float bf16_to_f32(uint16_t h)
{
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float          f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint8_t packF8(float f)
{
    hipblaslt_f8 v(f);
    uint8_t      b;
    std::memcpy(&b, &v, 1);
    return b;
}

static inline float unpackF8(uint8_t b)
{
    hipblaslt_f8 v;
    std::memcpy(&v, &b, 1);
    return static_cast<float>(v);
}

static inline uint8_t packBf8(float f)
{
    hipblaslt_bf8 v(f);
    uint8_t       b;
    std::memcpy(&b, &v, 1);
    return b;
}

static inline float unpackBf8(uint8_t b)
{
    hipblaslt_bf8 v;
    std::memcpy(&v, &b, 1);
    return static_cast<float>(v);
}

static inline uint16_t packF16(float f)
{
    _Float16 v = static_cast<_Float16>(f);
    uint16_t b;
    std::memcpy(&b, &v, 2);
    return b;
}

static inline float unpackF16(uint16_t b)
{
    _Float16 v;
    std::memcpy(&v, &b, 2);
    return static_cast<float>(v);
}

static bool deviceIsGfx950()
{
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess)
        return false;
    hipDeviceProp_t prop{};
    if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
        return false;
    return std::string(prop.gcnArchName).rfind("gfx950", 0) == 0;
}

static void fillRandomBf16(std::vector<uint16_t>&                 values,
                           std::mt19937&                          rng,
                           std::uniform_real_distribution<float>& dist)
{
    for(auto& x : values)
        x = f32_to_bf16(dist(rng));
}

static void fillRandomF16(std::vector<uint16_t>&                 values,
                          std::mt19937&                          rng,
                          std::uniform_real_distribution<float>& dist)
{
    for(uint16_t& x : values)
        x = packF16(dist(rng));
}

static void fillRandomF8(std::vector<uint8_t>&                  values,
                         std::mt19937&                          rng,
                         std::uniform_real_distribution<float>& dist)
{
    for(uint8_t& x : values)
        x = packF8(dist(rng));
}

static void fillRandomBf8(std::vector<uint8_t>&                  values,
                          std::mt19937&                          rng,
                          std::uniform_real_distribution<float>& dist)
{
    for(uint8_t& x : values)
        x = packBf8(dist(rng));
}

// Write a float into a raw byte buffer, rounded through the given element type.
static void writeTyped(std::vector<uint8_t>& out, size_t elemIdx, hipDataType type, float f)
{
    if(type == HIP_R_16F)
    {
        uint16_t b = packF16(f);
        std::memcpy(&out[elemIdx * 2], &b, 2);
        return;
    }
    if(type == HIP_R_8F_E4M3)
    {
        out[elemIdx] = packF8(f);
        return;
    }
    out[elemIdx] = packBf8(f);
}

// Read a float back from a raw byte buffer of the given element type.
static float readTyped(const std::vector<uint8_t>& in, size_t elemIdx, hipDataType type)
{
    if(type == HIP_R_16F)
    {
        uint16_t b;
        std::memcpy(&b, &in[elemIdx * 2], 2);
        return unpackF16(b);
    }
    if(type == HIP_R_8F_E4M3)
        return unpackF8(in[elemIdx]);
    return unpackBf8(in[elemIdx]);
}

struct FusedMatmulLayout
{
    hipDataType aType;
    hipDataType bType;
    hipDataType cdType;
};

// Set an MX block-32 UE8M0 scale on the matmul desc when a scale pointer is provided.
static void setMxBlockScale(hipblasLtMatmulDesc_t           mm,
                            hipblasLtMatmulDescAttributes_t modeAttr,
                            hipblasLtMatmulDescAttributes_t ptrAttr,
                            void*                           scale)
{
    if(scale == nullptr)
        return;
    hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
    hipblasLtMatmulDescSetAttribute(mm, modeAttr, &mode, sizeof(mode));
    hipblasLtMatmulDescSetAttribute(mm, ptrAttr, &scale, sizeof(scale));
}

// Core TN fused matmul: op(A)=T, op(B)=N, alpha=1, beta=0, one heuristic result.
// Optional MX block scales on A and/or B (pass nullptr to skip).
static hipblasStatus_t runTnFusedMatmul(hipblasLtHandle_t                  handle,
                                        FusedMatmulLayout                  layout,
                                        int64_t                            m,
                                        int64_t                            n,
                                        int64_t                            k,
                                        void*                              dA,
                                        int64_t                            lda,
                                        void*                              dScaleA,
                                        void*                              dB,
                                        void*                              dScaleB,
                                        void*                              dC,
                                        void*                              dD,
                                        hipblasLtFusedEpilogueDescriptor_t fused,
                                        void*                              dWorkspace,
                                        size_t                             workspaceSize,
                                        int&                               algoCount)
{
    algoCount = 0;

    hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    hipblasLtMatrixLayoutCreate(&layA, layout.aType, k, m, lda);
    hipblasLtMatrixLayoutCreate(&layB, layout.bType, k, n, k);
    hipblasLtMatrixLayoutCreate(&layC, layout.cdType, m, n, m);
    hipblasLtMatrixLayoutCreate(&layD, layout.cdType, m, n, m);

    hipblasLtMatmulDesc_t mm = nullptr;
    hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    setMxBlockScale(
        mm, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, dScaleA);
    setMxBlockScale(
        mm, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, dScaleB);
    hipblasLtMatmulDescSetAttribute(
        mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));

    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    hipblasLtMatmulHeuristicResult_t heur[1];
    hipblasLtMatmulAlgoGetHeuristic(handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount);

    const float     alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    if(algoCount > 0)
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

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(mm);
    hipblasLtMatrixLayoutDestroy(layA);
    hipblasLtMatrixLayoutDestroy(layB);
    hipblasLtMatrixLayoutDestroy(layC);
    hipblasLtMatrixLayoutDestroy(layD);
    return status;
}

static hipblasStatus_t runBf16TnFusedMatmul(hipblasLtHandle_t                  handle,
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
    return runTnFusedMatmul(handle,
                            {HIP_R_16BF, HIP_R_16BF, HIP_R_16BF},
                            m,
                            n,
                            k,
                            dA,
                            lda,
                            nullptr,
                            dB,
                            nullptr,
                            dC,
                            dD,
                            fused,
                            dWorkspace,
                            workspaceSize,
                            algoCount);
}

static hipblasStatus_t runBf16TnFusedMatmulFp8D(hipblasLtHandle_t                  handle,
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
    return runTnFusedMatmul(handle,
                            {HIP_R_16BF, HIP_R_16BF, HIP_R_8F_E4M3},
                            m,
                            n,
                            k,
                            dA,
                            lda,
                            nullptr,
                            dB,
                            nullptr,
                            dC,
                            dD,
                            fused,
                            dWorkspace,
                            workspaceSize,
                            algoCount);
}

// TN fp8-e4m3 A × fp8-e4m3 B → bf16 D with pre-swizzled MX block-32 UE8M0 scales on both
// A and B, and a fused epilogue. A is (k × m) col-major fp8 with scale dMxScaleA, B is
// (k × n) col-major fp8 with scale dMxScaleB, D is (m × n) bf16.
static hipblasStatus_t runFp8Fp8TnFusedMatmulBf16D(hipblasLtHandle_t                  handle,
                                                   int64_t                            m,
                                                   int64_t                            n,
                                                   int64_t                            k,
                                                   void*                              dA,
                                                   int64_t                            lda,
                                                   void*                              dMxScaleA,
                                                   void*                              dB,
                                                   void*                              dMxScaleB,
                                                   void*                              dC,
                                                   void*                              dD,
                                                   hipblasLtFusedEpilogueDescriptor_t fused,
                                                   void*                              dWorkspace,
                                                   size_t                             workspaceSize,
                                                   int&                               algoCount)
{
    return runTnFusedMatmul(handle,
                            {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16BF},
                            m,
                            n,
                            k,
                            dA,
                            lda,
                            dMxScaleA,
                            dB,
                            dMxScaleB,
                            dC,
                            dD,
                            fused,
                            dWorkspace,
                            workspaceSize,
                            algoCount);
}

static std::vector<float> referenceRmsNormRows(const std::vector<uint16_t>& hA,
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

static void expectBf16Near(const std::vector<uint16_t>& actual,
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

static void runFullRmsNormE2E(bool residualAdd)
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

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dGamma = nullptr,
         *dResidual = nullptr, *dWs = nullptr;
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
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
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
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
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
    ASSERT_EQ(
        runBf16TnFusedMatmul(handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << (residualAdd ? "no residualAdd PartialRMS solution selected"
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

// ---- End-to-end numeric test: full RMSNorm with bf16 residual-output buffer (no quant) ----
//
// Exercises the DQuantType=None / PartialRMSStoreBf16D=true solution path. The kernel writes
// bf16(H+residual) to the residual-out buffer and the RMS-normalised result to D. Shape
// [640,640,1,512] matches the exact entry in the deployed gfx950 LibraryLogic YAML.
static void runFullRmsNormResidualOutE2E()
{
    const int64_t M   = 640, N = 640, K = 512;
    const float   eps = 1e-5f;

    // op(A)=T => A stored K x M col-major; op(B)=N => B stored K x N col-major.
    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    std::vector<uint16_t> hResidual(static_cast<size_t>(M) * N);
    std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);
    std::vector<uint16_t> hResidualOut(static_cast<size_t>(M) * N, 0);

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hResidual, rng, dist);

    void* dA           = nullptr;
    void* dB           = nullptr;
    void* dC           = nullptr;
    void* dD           = nullptr;
    void* dGamma       = nullptr;
    void* dResidual    = nullptr;
    void* dResidualOut = nullptr;
    void* dWs          = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidualOut, hResidualOut.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD; // beta = 0, C unused numerically but must be a valid pointer.

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMemset(dResidualOut, 0, hResidualOut.size() * sizeof(uint16_t)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &dResidualOut,
                                                 sizeof(dResidualOut)),
              HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(
        runBf16TnFusedMatmul(handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no bf16 residual-out PartialRMS solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);
    // Validate the normalised D output.
    expectBf16Near(hD,
                   referenceRmsNormRows(hA, hB, hGamma, &hResidual, M, N, K, eps));

    ASSERT_EQ(hipMemcpy(hResidualOut.data(),
                        dResidualOut,
                        hResidualOut.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    // Reference: residualOut[row, col] = bf16(H[row,col] + residual[row,col]), where H is the
    // GEMM result. Layout mirrors referenceRmsNormRows exactly (hA[kk + row*K], hB[kk + col*K]).
    std::vector<float> refResidualOut(static_cast<size_t>(M) * N);
    for(int64_t row = 0; row < M; ++row)
        for(int64_t col = 0; col < N; ++col)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + row * K]) * bf16_to_f32(hB[kk + col * K]);
            acc += bf16_to_f32(hResidual[row * N + col]);
            // The kernel stores the pre-normalisation sum as bf16, so round-trip through bf16.
            refResidualOut[row * N + col] = bf16_to_f32(f32_to_bf16(acc));
        }
    expectBf16Near(hResidualOut, refResidualOut, 1e-2f, 0.10f);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dResidualOut));
    static_cast<void>(hipFree(dWs));
}

TEST(FusedEpilogueE2E, fullRmsNormBf16ResidualOutMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (PartialRMS) is wired for gfx950 only";

    runFullRmsNormResidualOutE2E();
}

TEST(FusedEpilogueE2E, fullRmsNormResidualAddRequantMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm requant is wired for gfx950 only";

    using fp8  = hipblaslt_f8;
    auto f8ToF = [](fp8 v) { return static_cast<float>(static_cast<_Float16>(v)); };

    // TN, BF16 inputs with FP8 D. Shape matches the gfx950 PartialRMS residual-add logic.
    const int64_t M = 1024, N = 1024, K = 4096;
    const float   eps          = 1e-5f;
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
    ASSERT_EQ(hipMemcpy(dC, hC.data(), hC.size() * sizeof(fp8), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
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

    int                   algoCount = 0;
    const hipblasStatus_t status    = runBf16TnFusedMatmulFp8D(
        handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount);
    ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no PartialRMS requant solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(fp8), hipMemcpyDeviceToHost), hipSuccess);

    const auto reference  = referenceRmsNormRows(hA, hB, hGamma, &hResidual, M, N, K, eps);
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
// orientation, no reduction). This test puts a host-computed rstd in the caller-owned handoff
// buffer so the consumer can be exercised independently of the producer. Verifies
// D[m,n] = (alpha * op(A)*op(B))[m,n] * rstd[m]. gfx950-only.

static void createScaleApplyDescriptor(hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                       hipblasLtFusedEpilogueDescriptor_t*       fused)
{
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(*fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  *fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);
}

static void createPartialStatsDescriptor(hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                         void*                                     gamma,
                                         float                                     eps,
                                         hipblasLtFusedEpilogueDescriptor_t*       fused)
{
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(*fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
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
         *dWs           = nullptr;
    const size_t wsSize = size_t(64) * 1024 * 1024;
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

    // Decomposed handoff, populated with the host rstd in caller-owned device memory.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, hRstd.size() * sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);

    // Consumer chain: RMSNorm scale-apply reads the deferred per-row scale from the handoff.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_NO_FATAL_FAILURE(createScaleApplyDescriptor(stats, &cons));

    int algoCount = 0;
    ASSERT_EQ(
        runBf16TnFusedMatmul(handle, M, N, K, dA, K, dB, dC, dD, cons, dWs, wsSize, algoCount),
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
// The full decomposed RMSNorm flow across two matmul calls linked by a caller-buffered RMSNorm
// handoff descriptor:
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
         *dD2 = nullptr, *dRstd = nullptr, *dWs = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dX, hX.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW0, hW0.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dH2, size_t(M) * Nhidden * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, size_t(M) * Nout * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, size_t(M) * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dW0, hW0.data(), hW0.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * 2, hipMemcpyHostToDevice), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Caller-owned handoff shared by both calls.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, size_t(M) * sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);

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
    ASSERT_EQ(
        runBf16TnFusedMatmul(
            handle, M, Nout, Nhidden, dH2, Nhidden, dW1, dD2, dD2, cons, dWs, wsSize, algoCount),
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
    static_cast<void>(hipFree(dRstd));
    static_cast<void>(hipFree(dWs));
}

// ---- Validation: the decomposed flow requires a caller-owned handoff buffer ----
//
// The library does not allocate the per-row rstd storage, so both decomposed stages reject a
// handoff descriptor with no buffer, or one too small for that call's D row count. This covers
// the matmul-level enforcement; the argument checks on
// hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer itself are in
// FusedEpilogueLifecycle.rmsnormStatsBufferValidation. gfx950-only, because reaching the
// enforcement requires a real PartialRMS (K1) / RstdScale (K3) solution to be selected.
TEST(FusedEpilogueE2E, decomposedHandoffBufferIsValidated)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed RMSNorm flow is wired for gfx950 only";

    const int64_t M = 1024, Nhidden = 1024, K0 = 64, Nout = 64;
    const float   eps = 1e-5f;

    std::vector<uint16_t> hX(static_cast<size_t>(K0) * M);
    std::vector<uint16_t> hW0(static_cast<size_t>(K0) * Nhidden);
    std::vector<uint16_t> hW1(static_cast<size_t>(Nhidden) * Nout);
    std::vector<uint16_t> hGamma(static_cast<size_t>(Nhidden), f32_to_bf16(1.0f));

    std::mt19937                          rng(99);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    fillRandomBf16(hX, rng, dist);
    fillRandomBf16(hW0, rng, dist);
    fillRandomBf16(hW1, rng, dist);

    void *dX = nullptr, *dW0 = nullptr, *dGamma = nullptr, *dH2 = nullptr, *dW1 = nullptr,
         *dD2 = nullptr, *dRstd = nullptr, *dWs = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dX, hX.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW0, hW0.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dH2, size_t(M) * Nhidden * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, size_t(M) * Nout * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, size_t(M) * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dW0, hW0.data(), hW0.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * 2, hipMemcpyHostToDevice), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Handoff descriptor deliberately left without a buffer for the first case.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_NO_FATAL_FAILURE(createPartialStatsDescriptor(stats, dGamma, eps, &prod));

    const size_t requiredBytes = size_t(M) * sizeof(float);
    int          algoCount     = 0;

    // No buffer: the producer rejects instead of allocating one internally.
    EXPECT_EQ(runBf16TnFusedMatmul(
                  handle, M, Nhidden, K0, dX, K0, dW0, dH2, dH2, prod, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_INVALID_VALUE);
    ASSERT_GT(algoCount, 0) << "no PartialRMS (K1) producer solution selected";

    // One row short of M * batchCount * sizeof(float).
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, requiredBytes - sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(runBf16TnFusedMatmul(
                  handle, M, Nhidden, K0, dX, K0, dW0, dH2, dH2, prod, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_INVALID_VALUE);

    // Correctly sized: the same producer call now runs, confirming the rejections above are
    // caused by the buffer and not by the problem setup.
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, dRstd, requiredBytes),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(runBf16TnFusedMatmul(
                  handle, M, Nhidden, K0, dX, K0, dW0, dH2, dH2, prod, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // The consumer validates the buffer against its own D row count as well.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_NO_FATAL_FAILURE(createScaleApplyDescriptor(stats, &cons));

    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, requiredBytes - sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        runBf16TnFusedMatmul(
            handle, M, Nout, Nhidden, dH2, Nhidden, dW1, dD2, dD2, cons, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_INVALID_VALUE);
    ASSERT_GT(algoCount, 0) << "no RstdScale (K3) consumer solution selected";

    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, dRstd, requiredBytes),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        runBf16TnFusedMatmul(
            handle, M, Nout, Nhidden, dH2, Nhidden, dW1, dD2, dD2, cons, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

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
    static_cast<void>(hipFree(dRstd));
    static_cast<void>(hipFree(dWs));
}

// ---- CPU reference helpers for MX fp8 quant ----

// Compute e8m0 scale byte and quantization multiplier for a single block amax.
// Returns the float multiplier (0 when amax is 0) and writes the UE8M0 byte to outSb.
static float e8m0QuantMult(float amax, uint8_t& outSb)
{
    constexpr float fp8Max = 448.0f;
    if(amax == 0.0f)
    {
        outSb = 0;
        return 0.0f;
    }
    const float scaleF = amax / fp8Max;
    uint32_t    bits;
    std::memcpy(&bits, &scaleF, sizeof(bits));
    const uint32_t expByte = (bits >> 23) & 0xFFu;
    const uint32_t mant    = bits & 0x7FFFFFu;
    const uint32_t ceilAdj = (mant != 0) ? 1u : 0u;
    const uint32_t sb      = std::min(expByte + ceilAdj, 254u);
    const uint32_t qExp
        = static_cast<uint32_t>(std::max(1, std::min(254, 254 - static_cast<int>(sb))));
    const uint32_t qBits = qExp << 23;
    float          mult;
    std::memcpy(&mult, &qBits, sizeof(mult));
    outSb = static_cast<uint8_t>(sb);
    return mult;
}

// Apply GFX950 pre-swizzle: write scalePlain[ti, tj] → scaleSwizzled[swzOff].
// scalePlain must be (paddedRows × paddedCols) in row-major order.
static std::vector<uint8_t>
    swizzleGfx950(const std::vector<uint8_t>& scalePlain, int64_t paddedRows, int64_t paddedCols)
{
    const int64_t        colBlocks = paddedCols / 8;
    std::vector<uint8_t> out(static_cast<size_t>(paddedRows) * paddedCols, 0);
    for(int64_t ti = 0; ti < paddedRows; ++ti)
        for(int64_t tj = 0; tj < paddedCols; ++tj)
        {
            const int64_t d0 = ti >> 5;
            const int64_t d1 = (ti >> 4) & 1;
            const int64_t d2 = ti & 0xF;
            const int64_t d3 = tj >> 3;
            const int64_t d4 = (tj >> 2) & 1;
            const int64_t d5 = tj & 3;
            const int64_t swzOff
                = d0 * (colBlocks * 256) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1;
            out[swzOff] = scalePlain[ti * paddedCols + tj];
        }
    return out;
}

namespace
{
    struct MxFp8Ref
    {
        std::vector<uint8_t> mxScale; // GFX950 pre-swizzled UE8M0 bytes
        std::vector<uint8_t> dFp8; // OCP e4m3 bytes, col-major (m × n)
        int64_t              paddedRows; // padded rows of scale tensor
        int64_t              paddedCols; // padded cols of scale tensor
    };
}

// Standard MX-fp8 block quant of a col-major [m, n] f32 matrix: free0=m (q0=1),
// free1=n (q1=blockSize). Returns swizzled UE8M0 scale bytes and fp8 e4m3 D bytes.
// Tensile always lays the scale grid out with rows = free1 tiles and cols = free0
// tiles, whichever axis carries the block; see mxScaleSwizzleOffset(tj, ti) and the
// padding of nTiles/mTiles in the tensilelite client's Reference.cpp. Here that is
// rows = n/blockSize and cols = m.
static MxFp8Ref
    quantizeMxfp8Standard(const std::vector<float>& dF32, int64_t m, int64_t n, int32_t blockSize)
{
    const int64_t nTiles     = (n + blockSize - 1) / blockSize;
    const int64_t paddedRows = ((nTiles + 31) / 32) * 32;
    const int64_t paddedCols = ((m + 7) / 8) * 8;

    std::vector<uint8_t> scalePlain(static_cast<size_t>(paddedRows) * paddedCols, 0);
    std::vector<float>   dQuantF32(static_cast<size_t>(m) * n, 0.0f);
    for(int64_t ti = 0; ti < m; ++ti)
        for(int64_t tj = 0; tj < nTiles; ++tj)
        {
            float amax = 0.0f;
            for(int64_t dj = 0; dj < blockSize; ++dj)
            {
                const int64_t col = tj * blockSize + dj;
                if(col >= n)
                    break;
                amax = std::max(amax, std::abs(dF32[ti + col * m]));
            }
            uint8_t     sb;
            const float mult                 = e8m0QuantMult(amax, sb);
            scalePlain[tj * paddedCols + ti] = sb;
            for(int64_t dj = 0; dj < blockSize; ++dj)
            {
                const int64_t col = tj * blockSize + dj;
                if(col >= n)
                    break;
                dQuantF32[ti + col * m] = dF32[ti + col * m] * mult;
            }
        }

    std::vector<uint8_t> dFp8(static_cast<size_t>(m) * n);
    for(size_t idx = 0; idx < dFp8.size(); ++idx)
        dFp8[idx] = packF8(dQuantF32[idx]);

    MxFp8Ref ref;
    ref.mxScale    = swizzleGfx950(scalePlain, paddedRows, paddedCols);
    ref.dFp8       = dFp8;
    ref.paddedRows = paddedRows;
    ref.paddedCols = paddedCols;
    return ref;
}

// CPU MX fp8 quant reference. aF32 is (k × m) col-major float, bF32 is (k × n) col-major
// float. The caller converts host input values through the actual GPU type (f16/fp8/bf8) before
// calling this function so the reference matches what the GPU sees.
static MxFp8Ref referenceMxfp8QuantF32(const std::vector<float>& aF32,
                                       const std::vector<float>& bF32,
                                       int64_t                   m,
                                       int64_t                   n,
                                       int64_t                   k,
                                       int32_t                   blockSize)
{
    std::vector<float> dF32(static_cast<size_t>(m) * n, 0.0f);
    for(int64_t i = 0; i < m; ++i)
        for(int64_t j = 0; j < n; ++j)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < k; ++kk)
                acc += aF32[kk + i * k] * bF32[kk + j * k];
            dF32[i + j * m] = acc;
        }
    return quantizeMxfp8Standard(dF32, m, n, blockSize);
}

// CPU MX fp8 quant reference for bf16 A/B inputs; delegates to the f32 path.
static MxFp8Ref referenceMxfp8Quant(const std::vector<uint16_t>& hA,
                                    const std::vector<uint16_t>& hB,
                                    int64_t                      m,
                                    int64_t                      n,
                                    int64_t                      k,
                                    int32_t                      blockSize)
{
    std::vector<float> aF32(hA.size()), bF32(hB.size());
    for(size_t i = 0; i < hA.size(); ++i)
        aF32[i] = bf16_to_f32(hA[i]);
    for(size_t i = 0; i < hB.size(); ++i)
        bF32[i] = bf16_to_f32(hB[i]);
    return referenceMxfp8QuantF32(aF32, bF32, m, n, k, blockSize);
}

// Count fp8-e4m3 byte positions whose decoded value differs from the reference.
static size_t countFp8Mismatches(const std::vector<uint8_t>& got, const std::vector<uint8_t>& ref)
{
    size_t mismatches = 0;
    for(size_t i = 0; i < got.size(); ++i)
        if(unpackF8(got[i]) != unpackF8(ref[i]))
            ++mismatches;
    return mismatches;
}

// Assert an MX UE8M0 scale buffer matches the reference byte-for-byte.
static void expectMxScaleEqual(const std::vector<uint8_t>& got, const std::vector<uint8_t>& ref)
{
    ASSERT_EQ(got.size(), ref.size());
    EXPECT_EQ(got, ref) << "MX UE8M0 scale buffer mismatch";
}

// Compare MX quant GPU output against a CPU reference.
static void expectMxfp8Near(const std::vector<uint8_t>& hD,
                            const std::vector<uint8_t>& hMxScale,
                            const MxFp8Ref&             ref)
{
    expectMxScaleEqual(hMxScale, ref.mxScale);
    ASSERT_EQ(hD.size(), ref.dFp8.size());
    const size_t mismatches = countFp8Mismatches(hD, ref.dFp8);
    EXPECT_EQ(mismatches, 0u) << "D e4m3 output has " << mismatches << " mismatches";
}

// CPU reference for the producer's transposed MX-fp8 quant: dOutT[nh, mt] = gamma[nh]*h1[mt, nh],
// block along the N_hidden (free0) axis with q1=1 over M_tokens. Scale grid is
// [M_tokens (rows) x N_hidden/blockSize (cols)]. Returns swizzled scale + fp8 D bytes.
static MxFp8Ref referenceProducerMxfp8(const std::vector<float>&    h1,
                                       const std::vector<uint16_t>& hGamma,
                                       int64_t                      mTok,
                                       int64_t                      nHid,
                                       int32_t                      blockSize,
                                       int64_t                      paddedRows,
                                       int64_t                      paddedCols)
{
    const int64_t mTiles = mTok;                               // rows = free1 (M_tokens).
    const int64_t nTiles = (nHid + blockSize - 1) / blockSize; // cols = kblock (N_hidden/blockSize).

    std::vector<uint8_t> scalePlain(static_cast<size_t>(paddedRows) * paddedCols, 0);
    std::vector<float>   dQuantF32(static_cast<size_t>(mTok) * nHid, 0.0f);
    for(int64_t ti = 0; ti < mTiles; ++ti)         // ti = M_token (free1).
        for(int64_t tj = 0; tj < nTiles; ++tj)     // tj = N_hidden block (free0/blockSize).
        {
            float amax = 0.0f;
            for(int64_t dj = 0; dj < blockSize; ++dj)
            {
                const int64_t nh = tj * blockSize + dj;
                if(nh >= nHid)
                    break;
                amax = std::max(amax, std::abs(h1[ti * nHid + nh] * bf16_to_f32(hGamma[nh])));
            }
            uint8_t     sb;
            const float mult                 = e8m0QuantMult(amax, sb);
            scalePlain[ti * paddedCols + tj] = sb;
            for(int64_t dj = 0; dj < blockSize; ++dj)
            {
                const int64_t nh = tj * blockSize + dj;
                if(nh >= nHid)
                    break;
                dQuantF32[nh + ti * nHid] = h1[ti * nHid + nh] * bf16_to_f32(hGamma[nh]) * mult;
            }
        }

    std::vector<uint8_t> refFp8(static_cast<size_t>(mTok) * nHid);
    for(size_t idx = 0; idx < refFp8.size(); ++idx)
        refFp8[idx] = packF8(dQuantF32[idx]);

    MxFp8Ref ref;
    ref.mxScale    = swizzleGfx950(scalePlain, paddedRows, paddedCols);
    ref.dFp8       = refFp8;
    ref.paddedRows = paddedRows;
    ref.paddedCols = paddedCols;
    return ref;
}

// TN GEMM with typed A/B → fp8 e4m3 D with MX scale output via a fused epilogue.
// abType selects the A/B element type (HIP_R_16F, HIP_R_8F_E4M3, or HIP_R_8F_E5M2).
// A is (k × m) col-major, B is (k × n) col-major. No A/B MX input scales.
static hipblasStatus_t runTypedTnFusedMatmulFp8D(hipblasLtHandle_t                  handle,
                                                 int64_t                            m,
                                                 int64_t                            n,
                                                 int64_t                            k,
                                                 void*                              dA,
                                                 int64_t                            lda,
                                                 void*                              dB,
                                                 void*                              dC,
                                                 void*                              dD,
                                                 hipDataType                        abType,
                                                 hipblasLtFusedEpilogueDescriptor_t fused,
                                                 void*                              dWorkspace,
                                                 size_t                             workspaceSize,
                                                 int&                               algoCount)
{
    return runTnFusedMatmul(handle,
                            {abType, abType, HIP_R_8F_E4M3},
                            m,
                            n,
                            k,
                            dA,
                            lda,
                            nullptr,
                            dB,
                            nullptr,
                            dC,
                            dD,
                            fused,
                            dWorkspace,
                            workspaceSize,
                            algoCount);
}

// ---- End-to-end test: standalone MX fp8 quant ----
//
// TN bf16 GEMM → e4m3 D with per-1×32-block UE8M0 scale output. No RMSNorm.
// m = N_hidden (free0), n = M_tokens (free1), block along free1 axis.

TEST(FusedEpilogueE2E, mxfp8QuantMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "MX fp8 quant epilogue is wired for gfx950 only";

    const int64_t M = 128, N = 512, K = 64;
    const int32_t blockSize  = 32;
    const int64_t mTiles     = M; // q0=1
    const int64_t nTiles     = (N + blockSize - 1) / blockSize;
    // Tensile scale grid: rows = free1 tiles (the blocked axis), cols = free0 tiles.
    const int64_t paddedRows = ((nTiles + 31) / 32) * 32;
    const int64_t paddedCols = ((mTiles + 7) / 8) * 8;
    const size_t  scaleBufSz = static_cast<size_t>(paddedRows) * paddedCols;

    std::vector<uint16_t>                 hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t>                 hB(static_cast<size_t>(K) * N);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dMxScale = nullptr,
         *dWs           = nullptr;
    const size_t wsSize = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(
        runBf16TnFusedMatmulFp8D(handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no standalone MX fp8 quant solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    const MxFp8Ref ref = referenceMxfp8Quant(hA, hB, M, N, K, blockSize);
    expectMxfp8Near(hD, hMxScale, ref);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end test: standalone MX fp8 quant, f16 inputs ----
//
// TN f16 GEMM → e4m3 D with per-1×32-block UE8M0 scale output. Uses the HF8S logic.
TEST(FusedEpilogueE2E, mxfp8QuantF16MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "MX fp8 quant epilogue is wired for gfx950 only";

    const int64_t M = 128, N = 512, K = 64;
    const int32_t blockSize  = 32;
    const int64_t mTiles     = M;
    const int64_t nTiles     = (N + blockSize - 1) / blockSize;
    // Tensile scale grid: rows = free1 tiles (the blocked axis), cols = free0 tiles.
    const int64_t paddedRows = ((nTiles + 31) / 32) * 32;
    const int64_t paddedCols = ((mTiles + 7) / 8) * 8;
    const size_t  scaleBufSz = static_cast<size_t>(paddedRows) * paddedCols;
    const size_t  szA        = static_cast<size_t>(K) * M;
    const size_t  szB        = static_cast<size_t>(K) * N;

    std::mt19937                          rng(44);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Quantise host values through f16 so the reference matches what the GPU sees.
    std::vector<uint16_t> hA(szA), hB(szB);
    fillRandomF16(hA, rng, dist);
    fillRandomF16(hB, rng, dist);
    std::vector<float> aF32(szA), bF32(szB);
    for(size_t i = 0; i < szA; ++i)
        aF32[i] = unpackF16(hA[i]);
    for(size_t i = 0; i < szB; ++i)
        bF32[i] = unpackF16(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dMxScale = nullptr,
         *dWs           = nullptr;
    const size_t wsSize = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, szA * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, szB * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), szA * sizeof(uint16_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), szB * sizeof(uint16_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(runTypedTnFusedMatmulFp8D(
                  handle, M, N, K, dA, K, dB, dC, dD, HIP_R_16F, fused, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no standalone MX fp8 quant (f16 input) solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    const MxFp8Ref ref = referenceMxfp8QuantF32(aF32, bF32, M, N, K, blockSize);
    expectMxfp8Near(hD, hMxScale, ref);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end test: standalone MX fp8 quant, fp8 e4m3 inputs ----
//
// TN fp8-e4m3 GEMM → e4m3 D with per-1×32-block UE8M0 scale output. Uses the F8F8S logic.
TEST(FusedEpilogueE2E, mxfp8QuantFp8MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "MX fp8 quant epilogue is wired for gfx950 only";

    const int64_t M = 128, N = 512, K = 128;
    const int32_t blockSize  = 32;
    const int64_t mTiles     = M;
    const int64_t nTiles     = (N + blockSize - 1) / blockSize;
    // Tensile scale grid: rows = free1 tiles (the blocked axis), cols = free0 tiles.
    const int64_t paddedRows = ((nTiles + 31) / 32) * 32;
    const int64_t paddedCols = ((mTiles + 7) / 8) * 8;
    const size_t  scaleBufSz = static_cast<size_t>(paddedRows) * paddedCols;
    const size_t  szA        = static_cast<size_t>(K) * M;
    const size_t  szB        = static_cast<size_t>(K) * N;

    std::mt19937                          rng(45);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Quantise through fp8 e4m3 so the reference matches what the GPU accumulates.
    std::vector<uint8_t> hA(szA), hB(szB);
    fillRandomF8(hA, rng, dist);
    fillRandomF8(hB, rng, dist);
    std::vector<float> aF32(szA), bF32(szB);
    for(size_t i = 0; i < szA; ++i)
        aF32[i] = unpackF8(hA[i]);
    for(size_t i = 0; i < szB; ++i)
        bF32[i] = unpackF8(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dMxScale = nullptr,
         *dWs           = nullptr;
    const size_t wsSize = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, szA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, szB), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), szA, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), szB, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(runTypedTnFusedMatmulFp8D(
                  handle, M, N, K, dA, K, dB, dC, dD, HIP_R_8F_E4M3, fused, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no standalone MX fp8 quant (fp8 e4m3 input) solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    const MxFp8Ref ref = referenceMxfp8QuantF32(aF32, bF32, M, N, K, blockSize);
    expectMxfp8Near(hD, hMxScale, ref);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end test: standalone MX fp8 quant, bf8 e5m2 inputs ----
//
// TN bf8-e5m2 GEMM → e4m3 D with per-1×32-block UE8M0 scale output. Uses the B8F8S logic.
TEST(FusedEpilogueE2E, mxfp8QuantBf8MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "MX fp8 quant epilogue is wired for gfx950 only";

    const int64_t M = 128, N = 512, K = 128;
    const int32_t blockSize  = 32;
    const int64_t mTiles     = M;
    const int64_t nTiles     = (N + blockSize - 1) / blockSize;
    // Tensile scale grid: rows = free1 tiles (the blocked axis), cols = free0 tiles.
    const int64_t paddedRows = ((nTiles + 31) / 32) * 32;
    const int64_t paddedCols = ((mTiles + 7) / 8) * 8;
    const size_t  scaleBufSz = static_cast<size_t>(paddedRows) * paddedCols;
    const size_t  szA        = static_cast<size_t>(K) * M;
    const size_t  szB        = static_cast<size_t>(K) * N;

    std::mt19937                          rng(46);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Quantise through bf8 e5m2 so the reference matches what the GPU accumulates.
    std::vector<uint8_t> hA(szA), hB(szB);
    fillRandomBf8(hA, rng, dist);
    fillRandomBf8(hB, rng, dist);
    std::vector<float> aF32(szA), bF32(szB);
    for(size_t i = 0; i < szA; ++i)
        aF32[i] = unpackBf8(hA[i]);
    for(size_t i = 0; i < szB; ++i)
        bF32[i] = unpackBf8(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dMxScale = nullptr,
         *dWs           = nullptr;
    const size_t wsSize = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, szA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, szB), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), szA, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), szB, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(runTypedTnFusedMatmulFp8D(
                  handle, M, N, K, dA, K, dB, dC, dD, HIP_R_8F_E5M2, fused, dWs, wsSize, algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no standalone MX fp8 quant (bf8 e5m2 input) solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    const MxFp8Ref ref = referenceMxfp8QuantF32(aF32, bF32, M, N, K, blockSize);
    expectMxScaleEqual(hMxScale, ref.mxScale);
    // For E5M2 inputs the gfx950 MFMA accumulates in a different order than the sequential
    // CPU reference, so a handful of D fp8 output values near a quantization boundary can
    // differ by 1 fp8 ULP. Allow up to 50 such mismatches (<<0.1% of 65536 elements).
    ASSERT_EQ(hD.size(), ref.dFp8.size());
    const size_t mismatches = countFp8Mismatches(hD, ref.dFp8);
    EXPECT_LE(mismatches, 50u) << "D e4m3 output has " << mismatches << " mismatches";

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dWs));
}

// ---- Typed helper: chained MXfp8 RMSNorm producer/consumer for f16/fp8/bf8 inputs ----
//
// Exercises the same decomposed flow as decomposedMxfp8ProducerConsumerMatchesReference
// but with GEMM1 input types HIP_R_16F, HIP_R_8F_E4M3, or HIP_R_8F_E5M2.
// GEMM1 output h2 is always fp8-e4m3 regardless of input type; the consumer (GEMM2)
// is fp8+fp8 symmetric and is therefore identical for all three input types.
// Gamma is random non-trivial bf16, applied per-column in the reference. Validation
// tolerances match the C1 standalone tests: byte-exact for f16/fp8, bounded-mismatch for bf8.

namespace
{
    // Fixed and derived dimensions for the typed chained-MXfp8 test.
    struct TypedTestDims
    {
        int64_t mTok      = 256;
        int64_t nHid      = 2048;
        int64_t k0        = 128;
        int64_t nOut      = 64;
        int32_t blockSize = 32;
        float   eps       = 1e-5f;
        // Producer scale geometry (free0=nHid q0=1, free1=mTok q1=blockSize).
        int64_t mTiles;
        int64_t nTiles;
        int64_t paddedRows;
        int64_t paddedCols;
        size_t  scaleBufSz;
        int64_t rstdRows;
        // Input byte sizes.
        size_t szABytes;
        size_t szBBytes;
        size_t elemSz;
        bool   isBf8;
        // Consumer scale geometry.
        int64_t consAPaddedRows;
        int64_t consAPaddedCols;
        size_t  consAScaleSz;
        int64_t consBPaddedRows;
        int64_t consBPaddedCols;
        size_t  consBScaleSz;
    };

    // Host input data for the typed chained-MXfp8 test.
    struct MxfpTypedInputData
    {
        std::vector<uint8_t>  rawA;
        std::vector<uint8_t>  rawB;
        std::vector<float>    aF32;
        std::vector<float>    bF32;
        std::vector<uint16_t> hGamma;
        std::vector<uint16_t> hW1;
    };

    // Data read back from the producer GEMM1 kernel.
    struct ProducerResults
    {
        std::vector<uint8_t>  hD1;
        std::vector<uint8_t>  hMxScale;
        std::vector<float>    hRstd;
        std::vector<uint16_t> hResidualOut; // bf16; non-empty when dResidualOut was non-null.
        int                   algoCount = 0;
    };

    // Consumer input tensors after quantizing the producer output and W1.
    struct ConsumerQuantData
    {
        std::vector<uint8_t> consAFp8;
        std::vector<uint8_t> consAScale;
        std::vector<uint8_t> consBFp8;
        std::vector<uint8_t> consBScale;
        std::vector<float>   consADequant;
        std::vector<float>   consBDequant;
    };
}

// Compute all test dimensions from the GEMM1 input element type.
static TypedTestDims makeTypedTestDims(hipDataType gemm1InType)
{
    TypedTestDims d;
    d.elemSz = (gemm1InType == HIP_R_16F || gemm1InType == HIP_R_16BF) ? 2u : 1u;
    d.isBf8  = (gemm1InType == HIP_R_8F_E5M2);

    // Producer scale (new orientation): rows = M_tokens (free1, pad x32),
    // cols = N_hidden/blockSize (kblock, pad x8) with the AITER GFX950 swizzle.
    d.mTiles     = d.mTok;
    d.nTiles     = (d.nHid + d.blockSize - 1) / d.blockSize;
    d.paddedRows = ((d.mTiles + 31) / 32) * 32;
    d.paddedCols = ((d.nTiles + 7) / 8) * 8;
    d.scaleBufSz = static_cast<size_t>(d.paddedRows) * d.paddedCols;
    d.rstdRows   = d.mTok;

    d.szABytes = static_cast<size_t>(d.k0) * d.mTok * d.elemSz;
    d.szBBytes = static_cast<size_t>(d.k0) * d.nHid * d.elemSz;

    d.consAPaddedRows = d.paddedRows;
    d.consAPaddedCols = d.paddedCols;
    d.consAScaleSz    = d.scaleBufSz;

    d.consBPaddedRows = ((d.nOut + 31) / 32) * 32;
    d.consBPaddedCols = ((d.nHid / d.blockSize + 7) / 8) * 8;
    d.consBScaleSz    = static_cast<size_t>(d.consBPaddedRows) * d.consBPaddedCols;

    return d;
}

// Generate raw A/B byte buffers (rounded through the GPU type) and their f32 round-trips.
// W1 is also generated using the same rng stream so the seed ordering matches exactly.
static MxfpTypedInputData generateTypedInputData(hipDataType gemm1InType, const TypedTestDims& d)
{
    MxfpTypedInputData data;
    data.rawA.resize(d.szABytes);
    data.rawB.resize(d.szBBytes);
    data.hW1.resize(static_cast<size_t>(d.nHid) * d.nOut);

    // Distinct seeds per type for independent test data.
    const uint32_t                        seed = (gemm1InType == HIP_R_16F)       ? 50000u
                                                 : (gemm1InType == HIP_R_8F_E4M3) ? 50001u
                                                                                  : 50002u;
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    // Independent generator for gamma so the rawA/rawB/hW1 rng stream is unchanged.
    std::mt19937                          grng(seed ^ 0x9e3779b9u);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    data.hGamma.resize(static_cast<size_t>(d.nHid));
    fillRandomBf16(data.hGamma, grng, gdist);

    // Generate input data through the actual GPU type so the reference is exact.
    const size_t nA = static_cast<size_t>(d.k0) * d.mTok;
    const size_t nB = static_cast<size_t>(d.k0) * d.nHid;
    for(size_t i = 0; i < nA; ++i)
        writeTyped(data.rawA, i, gemm1InType, dist(rng));
    for(size_t i = 0; i < nB; ++i)
        writeTyped(data.rawB, i, gemm1InType, dist(rng));
    fillRandomBf16(data.hW1, rng, dist);

    // Build f32 round-trip values of A and B for the CPU reference.
    data.aF32.resize(nA);
    data.bF32.resize(nB);
    for(size_t i = 0; i < nA; ++i)
        data.aF32[i] = readTyped(data.rawA, i, gemm1InType);
    for(size_t i = 0; i < nB; ++i)
        data.bF32[i] = readTyped(data.rawB, i, gemm1InType);
    return data;
}

// Build the PARTIAL_RMSNORM_STATS + REQUANT(MX) producer fused epilogue descriptor.
// Pass a non-null dResidual to prepend a RESIDUAL_ADD stage.
// Pass a non-null dResidualOut to also write the pre-quant bf16 H into a separate buffer.
static void buildProducerDescriptor(void*                                     dGamma,
                                    float                                     eps,
                                    hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                    void*                                     dMxScale,
                                    int32_t                                   blockSize,
                                    hipblasLtFusedEpilogueDescriptor_t*       prodOut,
                                    void*                                     dResidual    = nullptr,
                                    void*                                     dResidualOut = nullptr)
{
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(prodOut), HIPBLAS_STATUS_SUCCESS);
    if(dResidual != nullptr)
    {
        ASSERT_EQ(hipblasLtFusedEpilogueAdd(*prodOut, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(
                *prodOut, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
            HIPBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(
        hipblasLtFusedEpilogueAdd(*prodOut, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(*prodOut, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  *prodOut, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  *prodOut, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  *prodOut, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            *prodOut, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(*prodOut,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER,
                                                 &dMxScale,
                                                 sizeof(dMxScale)),
              HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  *prodOut, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            *prodOut, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);
    if(dResidualOut != nullptr)
        ASSERT_EQ(
            hipblasLtFusedEpilogueSetAttribute(*prodOut,
                                               HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER,
                                               &dResidualOut,
                                               sizeof(dResidualOut)),
            HIPBLAS_STATUS_SUCCESS);
}

// Launch the GEMM1 producer kernel and read back D1, MX scale, and rstd handoff.
// Pass non-null dMxScaleA/dMxScaleB to use the MX-scaled input path (runTnFusedMatmul).
// Pass non-null dResidualOut to also read back the bf16 pre-quant residual output.
static void launchProducerAndReadback(hipblasLtHandle_t                         handle,
                                      const TypedTestDims&                      d,
                                      hipDataType                               gemm1InType,
                                      void*                                     dA,
                                      void*                                     dB,
                                      void*                                     dD1,
                                      void*                                     dMxScale,
                                      hipblasLtFusedEpilogueDescriptor_t        prod,
                                      void*                                     dRstd,
                                      void*                                     dWs,
                                      size_t                                    wsSize,
                                      ProducerResults&                          out,
                                      void*                                     dMxScaleA    = nullptr,
                                      void*                                     dMxScaleB    = nullptr,
                                      void*                                     dResidualOut = nullptr)
{
    int algoCount = 0;
    if(dMxScaleA != nullptr && dMxScaleB != nullptr)
    {
        ASSERT_EQ(runTnFusedMatmul(handle,
                                   {gemm1InType, gemm1InType, HIP_R_8F_E4M3},
                                   d.mTok,
                                   d.nHid,
                                   d.k0,
                                   dA,
                                   d.k0,
                                   dMxScaleA,
                                   dB,
                                   dMxScaleB,
                                   dD1,
                                   dD1,
                                   prod,
                                   dWs,
                                   wsSize,
                                   algoCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(algoCount, 0) << "no MX-scaled PartialRMS+MXfp8 (K1) producer solution selected";
    }
    else
    {
        ASSERT_EQ(runTypedTnFusedMatmulFp8D(handle,
                                            d.mTok,
                                            d.nHid,
                                            d.k0,
                                            dA,
                                            d.k0,
                                            dB,
                                            dD1,
                                            dD1,
                                            gemm1InType,
                                            prod,
                                            dWs,
                                            wsSize,
                                            algoCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GT(algoCount, 0) << "no typed PartialRMS+MXfp8 (K1) producer solution selected";
    }
    out.algoCount = algoCount;
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    out.hD1.resize(static_cast<size_t>(d.mTok) * d.nHid);
    out.hMxScale.resize(d.scaleBufSz);
    ASSERT_EQ(hipMemcpy(out.hD1.data(), dD1, out.hD1.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(out.hMxScale.data(), dMxScale, d.scaleBufSz, hipMemcpyDeviceToHost),
              hipSuccess);

    out.hRstd.resize(static_cast<size_t>(d.rstdRows));
    ASSERT_EQ(
        hipMemcpy(out.hRstd.data(), dRstd, d.rstdRows * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);

    if(dResidualOut != nullptr)
    {
        const size_t residualOutSz = static_cast<size_t>(d.mTok) * d.nHid * sizeof(uint16_t);
        out.hResidualOut.resize(static_cast<size_t>(d.mTok) * d.nHid);
        ASSERT_EQ(
            hipMemcpy(out.hResidualOut.data(), dResidualOut, residualOutSz, hipMemcpyDeviceToHost),
            hipSuccess);
    }
}

// Validate producer output: rstd handoff, MX scale bytes, and fp8 D bytes.
// Writes the CPU GEMM1 reference into h1Out for use by the consumer validation.
static void validateProducer(const TypedTestDims&         d,
                             const std::vector<float>&    aF32,
                             const std::vector<float>&    bF32,
                             const std::vector<uint16_t>& hGamma,
                             const ProducerResults&       pr,
                             hipDataType                  gemm1InType,
                             std::vector<float>&          h1Out,
                             const std::vector<uint16_t>* hResidual = nullptr)
{
    // CPU reference: h1[mt, nh] = sum_k aF32[k + mt*k0] * bF32[k + nh*k0].
    h1Out.assign(static_cast<size_t>(d.mTok) * d.nHid, 0.0f);
    for(int64_t mt = 0; mt < d.mTok; ++mt)
        for(int64_t nh = 0; nh < d.nHid; ++nh)
        {
            float acc = 0.0f;
            for(int64_t k = 0; k < d.k0; ++k)
                acc += aF32[k + mt * d.k0] * bF32[k + nh * d.k0];
            if(hResidual != nullptr)
                acc += bf16_to_f32((*hResidual)[mt * d.nHid + nh]);
            h1Out[mt * d.nHid + nh] = acc;
        }

    // Validate rstd handoff: rstd[mt] = 1/sqrt(mean(h1[mt,:]^2) + eps).
    {
        std::vector<float> refRstd(static_cast<size_t>(d.mTok));
        for(int64_t mt = 0; mt < d.mTok; ++mt)
        {
            float ss = 0.0f;
            for(int64_t nh = 0; nh < d.nHid; ++nh)
                ss += h1Out[mt * d.nHid + nh] * h1Out[mt * d.nHid + nh];
            refRstd[mt] = 1.0f / std::sqrt(ss / static_cast<float>(d.nHid) + d.eps);
        }
        int64_t rstdMismatches = 0;
        float   maxRstdErr     = 0.0f;
        for(int64_t mt = 0; mt < d.mTok; ++mt)
        {
            const float err = std::abs(pr.hRstd[mt] - refRstd[mt]);
            maxRstdErr      = std::max(maxRstdErr, err);
            if(err > 1e-3f)
                ++rstdMismatches;
        }
        EXPECT_EQ(rstdMismatches, 0)
            << "rstd handoff mismatch (max abs error " << maxRstdErr << ")";
    }

    // Validate producer fp8 D + MX scales using shared reference helpers.
    {
        const MxFp8Ref ref = referenceProducerMxfp8(
            h1Out, hGamma, d.mTok, d.nHid, d.blockSize, d.paddedRows, d.paddedCols);
        expectMxScaleEqual(pr.hMxScale, ref.mxScale);
        ASSERT_EQ(pr.hD1.size(), ref.dFp8.size());
        const size_t dMismatches = countFp8Mismatches(pr.hD1, ref.dFp8);
        if(d.isBf8)
            EXPECT_LE(dMismatches, 200u) << "D e4m3 output has " << dMismatches << " mismatches";
        else if(gemm1InType == HIP_R_16F)
            EXPECT_LE(dMismatches, 20u) << "D e4m3 output has " << dMismatches << " mismatches";
        else if(gemm1InType == HIP_R_16BF)
            EXPECT_LE(dMismatches, 8u) << "D e4m3 output has " << dMismatches << " mismatches";
        else
            EXPECT_EQ(dMismatches, 0u) << "D e4m3 output has " << dMismatches << " mismatches";
    }

    // Validate bf16 residual output if the producer was asked to write it.
    if(!pr.hResidualOut.empty())
        expectBf16Near(pr.hResidualOut, h1Out, 1e-2f, 0.10f);
}

// Build consumer A+B MX buffers for GEMM2.
// The producer's fp8 D and pre-swizzled scale are passed through directly as consumer A.
static ConsumerQuantData buildConsumerQuantData(const TypedTestDims&         d,
                                                const std::vector<uint16_t>& hW1,
                                                const std::vector<uint8_t>&  hD1,
                                                const std::vector<uint8_t>&  hMxScale)
{
    ConsumerQuantData cq;

    // Pass the producer's fp8 D and pre-swizzled scale directly to the consumer.
    cq.consAFp8   = hD1;
    cq.consAScale = hMxScale;

    // Dequant the producer's fp8 D for the CPU reference computation.
    cq.consADequant.assign(static_cast<size_t>(d.nHid) * d.mTok, 0.0f);
    for(int64_t nh = 0; nh < d.nHid; ++nh)
        for(int64_t mt = 0; mt < d.mTok; ++mt)
        {
            const int64_t kj        = nh / d.blockSize; // N_hidden block (col).
            const int64_t d0        = mt >> 5;           // row = M_token (free1).
            const int64_t d1        = (mt >> 4) & 1;
            const int64_t d2        = mt & 0xF;
            const int64_t d3        = kj >> 3;           // col = kblock.
            const int64_t d4        = (kj >> 2) & 1;
            const int64_t d5        = kj & 3;
            const int64_t colBlocks = d.paddedCols / 8;
            const int64_t swzOff    = d0 * (colBlocks * 256) + d3 * 256 + d5 * 64 + d2 * 4 + d4 * 2 + d1;
            const uint8_t sb        = hMxScale[static_cast<size_t>(swzOff)];
            float         dqMult    = 0.0f;
            if(sb != 0)
            {
                const uint32_t bits = static_cast<uint32_t>(sb) << 23;
                std::memcpy(&dqMult, &bits, sizeof(dqMult));
            }
            cq.consADequant[nh + mt * d.nHid] = unpackF8(hD1[nh + mt * d.nHid]) * dqMult;
        }

    // Quantize hW1 (bf16) to fp8 B with consumer B MX scale (blocks of K=nh at N=no).
    cq.consBFp8.resize(static_cast<size_t>(d.nHid) * d.nOut);
    cq.consBDequant.assign(static_cast<size_t>(d.nHid) * d.nOut, 0.0f);
    std::vector<uint8_t> consBScalePlain(d.consBScaleSz, 0);
    for(int64_t no = 0; no < d.nOut; ++no)
        for(int64_t nhBlock = 0; nhBlock < d.consBPaddedCols; ++nhBlock)
        {
            float amax = 0.0f;
            for(int64_t j = 0; j < d.blockSize; ++j)
            {
                const int64_t nh = nhBlock * d.blockSize + j;
                if(nh >= d.nHid)
                    break;
                amax = std::max(amax, std::abs(bf16_to_f32(hW1[nh + no * d.nHid])));
            }
            uint8_t     sb;
            const float qmult  = e8m0QuantMult(amax, sb);
            float       dqMult = 0.0f;
            if(sb != 0)
            {
                const uint32_t bits = static_cast<uint32_t>(sb) << 23;
                std::memcpy(&dqMult, &bits, sizeof(dqMult));
            }
            consBScalePlain[no * d.consBPaddedCols + nhBlock] = sb;
            for(int64_t j = 0; j < d.blockSize; ++j)
            {
                const int64_t nh = nhBlock * d.blockSize + j;
                if(nh >= d.nHid)
                    break;
                cq.consBFp8[nh + no * d.nHid] = packF8(bf16_to_f32(hW1[nh + no * d.nHid]) * qmult);
                cq.consBDequant[nh + no * d.nHid]
                    = unpackF8(cq.consBFp8[nh + no * d.nHid]) * dqMult;
            }
        }
    cq.consBScale = swizzleGfx950(consBScalePlain, d.consBPaddedRows, d.consBPaddedCols);

    return cq;
}

// Run the consumer GEMM2 (fp8-symmetric + RMSNORM_SCALE_APPLY) and validate against
// the CPU reference. Allocates and frees its own device buffers.
static void runConsumerAndValidate(hipblasLtHandle_t                         handle,
                                   const TypedTestDims&                      d,
                                   const std::vector<float>&                 h1,
                                   const ConsumerQuantData&                  cq,
                                   hipblasLtFusedEpilogueRMSNormDescriptor_t stats,
                                   void*                                     dD2,
                                   void*                                     dWs,
                                   size_t                                    wsSize)
{
    void *dConsA = nullptr, *dConsScaleA = nullptr;
    void *dConsB = nullptr, *dConsScaleB = nullptr;
    ASSERT_EQ(hipMalloc(&dConsA, cq.consAFp8.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&dConsScaleA, cq.consAScale.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&dConsB, cq.consBFp8.size()), hipSuccess);
    ASSERT_EQ(hipMalloc(&dConsScaleB, cq.consBScale.size()), hipSuccess);
    ASSERT_EQ(hipMemcpy(dConsA, cq.consAFp8.data(), cq.consAFp8.size(), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dConsScaleA, cq.consAScale.data(), cq.consAScale.size(), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dConsB, cq.consBFp8.data(), cq.consBFp8.size(), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dConsScaleB, cq.consBScale.data(), cq.consBScale.size(), hipMemcpyHostToDevice),
        hipSuccess);

    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_NO_FATAL_FAILURE(createScaleApplyDescriptor(stats, &cons));

    int                   consAlgoCount = 0;
    const hipblasStatus_t consStatus    = runFp8Fp8TnFusedMatmulBf16D(handle,
                                                                      d.mTok,
                                                                      d.nOut,
                                                                      d.nHid,
                                                                      dConsA,
                                                                      d.nHid,
                                                                      dConsScaleA,
                                                                      dConsB,
                                                                      dConsScaleB,
                                                                      dD2,
                                                                      dD2,
                                                                      cons,
                                                                      dWs,
                                                                      wsSize,
                                                                      consAlgoCount);
    ASSERT_GT(consAlgoCount, 0)
        << "no fp8+fp8 MX-input ScaleAlphaVec (K3) consumer solution selected";
    ASSERT_EQ(consStatus, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint16_t> hD2(static_cast<size_t>(d.mTok) * d.nOut);
    ASSERT_EQ(hipMemcpy(hD2.data(), dD2, hD2.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // Reference: y[mt,no] = rstd[mt] * sum_nh(consADequant[nh,mt]*consBDequant[nh,no]).
    std::vector<float> refRstd2(static_cast<size_t>(d.mTok));
    for(int64_t mt = 0; mt < d.mTok; ++mt)
    {
        float ss = 0.0f;
        for(int64_t nh = 0; nh < d.nHid; ++nh)
            ss += h1[mt * d.nHid + nh] * h1[mt * d.nHid + nh];
        refRstd2[mt] = 1.0f / std::sqrt(ss / static_cast<float>(d.nHid) + d.eps);
    }
    std::vector<float> refY(static_cast<size_t>(d.mTok) * d.nOut, 0.0f);
    for(int64_t mt = 0; mt < d.mTok; ++mt)
        for(int64_t no = 0; no < d.nOut; ++no)
        {
            float acc = 0.0f;
            for(int64_t nh = 0; nh < d.nHid; ++nh)
                acc += cq.consADequant[nh + mt * d.nHid] * cq.consBDequant[nh + no * d.nHid];
            refY[mt * d.nOut + no] = refRstd2[mt] * acc;
        }

    // fp8 double-quantization introduces error; use generous tolerance (abs=0.08, rel=0.12).
    int64_t consMismatches = 0;
    float   maxAbsErr2     = 0.0f;
    float   maxRelErr2     = 0.0f;
    for(int64_t mt = 0; mt < d.mTok; ++mt)
        for(int64_t no = 0; no < d.nOut; ++no)
        {
            const float got   = bf16_to_f32(hD2[mt + no * d.mTok]);
            const float ref   = refY[mt * d.nOut + no];
            const float abse  = std::abs(got - ref);
            const float denom = std::max(std::abs(ref), 1e-3f);
            maxAbsErr2        = std::max(maxAbsErr2, abse);
            maxRelErr2        = std::max(maxRelErr2, abse / denom);
            if(abse > std::max(0.08f, 0.12f * std::abs(ref)))
                ++consMismatches;
        }
    EXPECT_EQ(consMismatches, 0) << "consumer GEMM2 output mismatch (max abs=" << maxAbsErr2
                                 << ", max rel=" << maxRelErr2 << ")";

    hipblasLtFusedEpilogueDestroy(cons);
    static_cast<void>(hipFree(dConsA));
    static_cast<void>(hipFree(dConsScaleA));
    static_cast<void>(hipFree(dConsB));
    static_cast<void>(hipFree(dConsScaleB));
}

// Orchestrate the decomposed MXfp8 RMSNorm producer/consumer chain for a typed GEMM1 input.
static void runDecomposedMxfp8ProducerConsumerTyped(hipDataType gemm1InType)
{
    const TypedTestDims      d    = makeTypedTestDims(gemm1InType);
    const MxfpTypedInputData data = generateTypedInputData(gemm1InType, d);

    void *       dA = nullptr, *dB = nullptr, *dGamma = nullptr;
    void *       dD1 = nullptr, *dMxScale = nullptr, *dWs = nullptr;
    void *       dW1 = nullptr, *dD2 = nullptr, *dRstd = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;

    ASSERT_EQ(hipMalloc(&dA, d.szABytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, d.szBBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, data.hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD1, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, data.hW1.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);

    ASSERT_EQ(hipMemcpy(dA, data.rawA.data(), d.szABytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, data.rawB.data(), d.szBBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma,
                        data.hGamma.data(),
                        data.hGamma.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dW1, data.hW1.data(), data.hW1.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(dD1, 0, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMemset(dD2, 0, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)),
              hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Caller-owned rstd handoff shared by both calls.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        buildProducerDescriptor(dGamma, d.eps, stats, dMxScale, d.blockSize, &prod));

    ProducerResults pr;
    ASSERT_NO_FATAL_FAILURE(launchProducerAndReadback(
        handle, d, gemm1InType, dA, dB, dD1, dMxScale, prod, dRstd, dWs, wsSize, pr));

    std::vector<float> h1;
    ASSERT_NO_FATAL_FAILURE(
        validateProducer(d, data.aF32, data.bF32, data.hGamma, pr, gemm1InType, h1));

    const ConsumerQuantData cq = buildConsumerQuantData(d, data.hW1, pr.hD1, pr.hMxScale);
    ASSERT_NO_FATAL_FAILURE(runConsumerAndValidate(handle, d, h1, cq, stats, dD2, dWs, wsSize));

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dD1));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dW1));
    static_cast<void>(hipFree(dD2));
    static_cast<void>(hipFree(dRstd));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end test: decomposed two-call MXfp8 RMSNorm producer/consumer ----
//
// GEMM1 producer: PARTIAL_RMSNORM_STATS + REQUANT(MX). K1 kernel writes fp8-e4m3 D
// (MX-quantized gamma*h1, NOT normalized) in the transposed [N_hidden, M_tokens] layout,
// the UE8M0 MX block scales, and per-MT0-tile partial sum-of-squares to partialBuf.
// row_rstd (Kernel 2) reduces partialBuf and writes per-row 1/sqrt(mean(h1²)+eps) to the
// caller-owned rstd handoff buffer.
//
// GEMM2 consumer: RMSNORM_SCALE_APPLY reads the same handoff via ScaleAlphaVec.
// For the consumer: fp8-e4m3 A (producer output) × bf16 B (W1) → bf16 D.
// The producer's pre-swizzled MX scale is passed as the A block scale.
//
// Problem sizes from gemm_partial_rms_mxfp8_quant_k1.yaml:
//   GEMM1: [M_tokens=256, N_hidden=2048, batch=1, K=128] (exact listed size).
//   GEMM2: fp8-input A [N_hidden=2048, M_tokens=256] × bf16 W1 [N_hidden=2048, Nout=64].
//
// gfx950-only.

static void runDecomposedMxfp8ProducerConsumer(bool residualAdd)
{
    const TypedTestDims d      = makeTypedTestDims(HIP_R_16BF);
    const size_t        wsSize = size_t(256) * 1024 * 1024;

    std::vector<uint16_t> hA(static_cast<size_t>(d.k0) * d.mTok);
    std::vector<uint16_t> hB(static_cast<size_t>(d.k0) * d.nHid);
    std::vector<uint16_t> hGamma(static_cast<size_t>(d.nHid));
    std::vector<uint16_t> hW1(static_cast<size_t>(d.nHid) * d.nOut);
    std::vector<uint16_t> hResidual(residualAdd ? static_cast<size_t>(d.mTok) * d.nHid : 0);

    std::mt19937                          rng(31415);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hW1, rng, dist);
    if(residualAdd)
        fillRandomBf16(hResidual, rng, dist);

    std::vector<float> aF32(hA.size()), bF32(hB.size());
    for(size_t i = 0; i < hA.size(); ++i)
        aF32[i] = bf16_to_f32(hA[i]);
    for(size_t i = 0; i < hB.size(); ++i)
        bF32[i] = bf16_to_f32(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dGamma = nullptr;
    void *dD1 = nullptr, *dMxScale = nullptr, *dWs = nullptr;
    void *dW1 = nullptr, *dD2 = nullptr, *dResidual = nullptr, *dRstd = nullptr;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD1, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    if(residualAdd)
        ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD1, 0, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMemset(dD2, 0, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)),
              hipSuccess);
    if(residualAdd)
        ASSERT_EQ(hipMemcpy(dResidual,
                            hResidual.data(),
                            hResidual.size() * sizeof(uint16_t),
                            hipMemcpyHostToDevice),
                  hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Caller-owned rstd handoff shared by both calls.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);

    // Producer: [RESIDUAL_ADD ->] PARTIAL_RMSNORM_STATS -> REQUANT(MX).
    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_NO_FATAL_FAILURE(buildProducerDescriptor(
        dGamma, d.eps, stats, dMxScale, d.blockSize, &prod, residualAdd ? dResidual : nullptr));

    ProducerResults pr;
    ASSERT_NO_FATAL_FAILURE(launchProducerAndReadback(
        handle, d, HIP_R_16BF, dA, dB, dD1, dMxScale, prod, dRstd, dWs, wsSize, pr));

    std::vector<float> h1;
    ASSERT_NO_FATAL_FAILURE(validateProducer(
        d, aF32, bF32, hGamma, pr, HIP_R_16BF, h1, residualAdd ? &hResidual : nullptr));

    const ConsumerQuantData cq = buildConsumerQuantData(d, hW1, pr.hD1, pr.hMxScale);
    ASSERT_NO_FATAL_FAILURE(runConsumerAndValidate(handle, d, h1, cq, stats, dD2, dWs, wsSize));

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dD1));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dW1));
    static_cast<void>(hipFree(dD2));
    static_cast<void>(hipFree(dRstd));
    if(dResidual)
        static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dWs));
}

TEST(FusedEpilogueE2E, decomposedMxfp8ProducerConsumerMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed MXfp8 RMSNorm flow is wired for gfx950 only";
    runDecomposedMxfp8ProducerConsumer(/*residualAdd=*/false);
}

// Residual-add MXfp8 producer/consumer is bf16-input only: the kernel's residual load
// width follows the GEMM input dtype, so non-bf16 inputs would require a separate path.
TEST(FusedEpilogueE2E, decomposedMxfp8ResidualAddProducerConsumerMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed MXfp8 RMSNorm residual-add flow is wired for gfx950 only";
    runDecomposedMxfp8ProducerConsumer(/*residualAdd=*/true);
}

// ---- End-to-end: chained MXfp8 RMSNorm producer/consumer, f16 inputs ----
//
// Same two-call flow as decomposedMxfp8ProducerConsumerMatchesReference but with
// f16 (HIP_R_16F) GEMM1 inputs. Uses the partialrms_mxfp8_quant_k1 HF8S logic.
TEST(FusedEpilogueE2E, decomposedMxfp8ProducerConsumerF16MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed MXfp8 RMSNorm flow is wired for gfx950 only";
    runDecomposedMxfp8ProducerConsumerTyped(HIP_R_16F);
}

// ---- End-to-end: chained MXfp8 RMSNorm producer/consumer, fp8 e4m3 inputs ----
//
// Same flow with fp8/e4m3 (HIP_R_8F_E4M3) GEMM1 inputs. Uses the F8F8S logic.
TEST(FusedEpilogueE2E, decomposedMxfp8ProducerConsumerFp8MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed MXfp8 RMSNorm flow is wired for gfx950 only";
    runDecomposedMxfp8ProducerConsumerTyped(HIP_R_8F_E4M3);
}

// ---- End-to-end: chained MXfp8 RMSNorm producer/consumer, bf8 e5m2 inputs ----
//
// Same flow with bf8/e5m2 (HIP_R_8F_E5M2) GEMM1 inputs. Uses the B8F8S logic.
// Producer fp8-D validation uses bounded-mismatch tolerance (up to 200) because
// the gfx950 MFMA accumulates e5m2 in a different order than the sequential CPU
// reference, widened by the non-trivial per-column gamma; scale bytes are still byte-exact.
TEST(FusedEpilogueE2E, decomposedMxfp8ProducerConsumerBf8MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed MXfp8 RMSNorm flow is wired for gfx950 only";
    runDecomposedMxfp8ProducerConsumerTyped(HIP_R_8F_E5M2);
}

// ---- End-to-end: PartialRMS MXFP8 with dual bf16 residual output ----
//
// Exercises the HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER attribute.
// The K1 kernel runs the RESIDUAL_ADD -> RMSNORM -> REQUANT chain (PartialRMSStoreBf16D=true),
// writing the MXFP8-quantised fp8 D + UE8M0 block scale AND additionally storing the
// pre-quantisation value H+residual as bf16 in a separate buffer. The test verifies that
// the bf16 residualOut matches the CPU reference: dot(A,B) + residual (no gamma, no invRms).

TEST(FusedEpilogueE2E, partialRmsMxfp8ResidualOutBf16MatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "partialRMSStoreBf16D MXFP8 epilogue is wired for gfx950 only";

    // M_tokens=256 (multiple of MacroTile1=128), N_hidden=512 (multiple of MacroTile0=64).
    const int64_t M         = 256;
    const int64_t N         = 512;
    const int64_t K         = 64;
    const int32_t blockSize = 32;
    const float   eps       = 1e-5f;

    // MX scale tensor dimensions after the PartialRMS transpose (free0=N_hidden, free1=M_tokens).
    const int64_t kBlockTiles = (N + blockSize - 1) / blockSize;
    const int64_t freeTiles   = M;
    const int64_t paddedRows  = ((freeTiles + 31) / 32) * 32;
    const int64_t paddedCols  = ((kBlockTiles + 7) / 8) * 8;
    const size_t  scaleBufSz  = static_cast<size_t>(paddedRows) * paddedCols;

    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    std::vector<uint16_t> hResidual(static_cast<size_t>(M) * N);

    std::mt19937                          rng(2027);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hResidual, rng, dist);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dGamma = nullptr,
         *dResidual = nullptr, *dMxScale = nullptr, *dResidualOut = nullptr, *dWs = nullptr;
    const size_t wsSize      = size_t(256) * 1024 * 1024;
    const size_t residualOutSz = static_cast<size_t>(M) * N * sizeof(uint16_t);
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidualOut, residualOutSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD; // beta = 0, C unused numerically but must be a valid pointer.

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMemset(dResidualOut, 0, residualOutSz), hipSuccess);

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
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);
    // Request the bf16 pre-quantisation dual-store output (PartialRMSStoreBf16D path).
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER,
                                                 &dResidualOut,
                                                 sizeof(dResidualOut)),
              HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(
        runBf16TnFusedMatmulFp8D(handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no PartialRMSStoreBf16D MXFP8 solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Copy back the bf16 residual-out buffer produced by the kernel.
    std::vector<uint16_t> hResidualOut(static_cast<size_t>(M) * N);
    ASSERT_EQ(hipMemcpy(hResidualOut.data(), dResidualOut, residualOutSz, hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference: residualOut[row][col] = dot(A_row, B_col) + residual[row][col].
    // This is the pre-gamma, pre-invRms value H that the kernel stores as bf16.
    std::vector<float> refResidualOut(static_cast<size_t>(M) * N, 0.0f);
    for(int64_t row = 0; row < M; ++row)
        for(int64_t col = 0; col < N; ++col)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + row * K]) * bf16_to_f32(hB[kk + col * K]);
            acc += bf16_to_f32(hResidual[row * N + col]);
            refResidualOut[row * N + col] = acc;
        }

    // Use relaxed tolerances: bf16 rounding introduces up to 0.5 ULP, and parallel GPU
    // accumulation can differ from sequential CPU by a few fp32 ULPs.
    expectBf16Near(hResidualOut, refResidualOut, 1e-2f, 0.10f);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dResidualOut));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end: PartialRMS MXFP8, fp8-e4m3 A/B inputs with MX input scales ----
//
// Exercises the RESIDUAL_ADD -> RMSNORM -> REQUANT fused chain with fp8-e4m3 A and B
// inputs scaled by input MX block-32 UE8M0 scales (DataTypeMXSA/B path). Uniform-127
// input scales (scale=1.0) exercise the MXSA/B path without altering the numeric reference.
// The output D is MXFP8-quantised fp8-e4m3. The selected kernel also writes a bf16
// residual-out buffer (pre-quantisation H = GEMM+residual), which is verified against
// the CPU reference H below.

TEST(FusedEpilogueE2E, partialRmsMxfp8InputMxfp8QuantMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "partialRMS MXFP8-input MXFP8-quant epilogue is wired for gfx950 only";

    const int64_t M         = 256;
    const int64_t N         = 512;
    const int64_t K         = 256;
    const int32_t blockSize = 32;
    const float   eps       = 1e-5f;

    // Output MX scale geometry (post-PartialRMS transpose: free0=N_hidden, free1=M_tokens).
    const int64_t kBlockTiles = (N + blockSize - 1) / blockSize;
    const int64_t freeTiles   = M;
    const int64_t paddedRows  = ((freeTiles + 31) / 32) * 32;
    const int64_t paddedCols  = ((kBlockTiles + 7) / 8) * 8;
    const size_t  scaleBufSz  = static_cast<size_t>(paddedRows) * paddedCols;

    // Input MX scale geometry: blocks of 32 along K, one scale per (row or col, K/32 block).
    const int64_t inputScaleColsK = ((K / blockSize + 7) / 8) * 8;
    const size_t  aScaleSz        = static_cast<size_t>(((M + 31) / 32) * 32) * inputScaleColsK;
    const size_t  bScaleSz        = static_cast<size_t>(((N + 31) / 32) * 32) * inputScaleColsK;

    const size_t szA = static_cast<size_t>(K) * M;
    const size_t szB = static_cast<size_t>(K) * N;

    std::vector<uint8_t>  hA(szA), hB(szB);
    std::vector<uint16_t> hGamma(N), hResidual(static_cast<size_t>(M) * N);

    std::mt19937                          rng(2100);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomF8(hA, rng, dist);
    fillRandomF8(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hResidual, rng, dist);

    // Unpack fp8 bytes to float for the CPU reference accumulation.
    std::vector<float> aF32(szA), bF32(szB);
    for(size_t i = 0; i < szA; ++i)
        aF32[i] = unpackF8(hA[i]);
    for(size_t i = 0; i < szB; ++i)
        bF32[i] = unpackF8(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    void *dGamma = nullptr, *dResidual = nullptr, *dMxScale = nullptr;
    void *dMxScaleA = nullptr, *dMxScaleB = nullptr, *dResidualOut = nullptr, *dWs = nullptr;
    const size_t wsSize       = size_t(256) * 1024 * 1024;
    const size_t residualOutSz = static_cast<size_t>(M) * N * sizeof(uint16_t);
    ASSERT_EQ(hipMalloc(&dA, szA), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, szB), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScaleA, aScaleSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScaleB, bScaleSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidualOut, residualOutSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), szA, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), szB, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMemset(dResidualOut, 0, residualOutSz), hipSuccess);

    // UE8M0 byte 127 encodes scale 2^(127-127)=1.0; uniform-127 input scales exercise the
    // MXSA/B path without changing the values the accumulator sees.
    const std::vector<uint8_t> hScaleA(aScaleSz, 127u);
    const std::vector<uint8_t> hScaleB(bScaleSz, 127u);
    ASSERT_EQ(hipMemcpy(dMxScaleA, hScaleA.data(), aScaleSz, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dMxScaleB, hScaleB.data(), bScaleSz, hipMemcpyHostToDevice), hipSuccess);

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
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);
    // The kernel writes the pre-quantisation bf16 H = GEMM+residual into this buffer.
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER,
                                                 &dResidualOut,
                                                 sizeof(dResidualOut)),
              HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(runTnFusedMatmul(handle,
                               {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_8F_E4M3},
                               M,
                               N,
                               K,
                               dA,
                               K,
                               dMxScaleA,
                               dB,
                               dMxScaleB,
                               dC,
                               dD,
                               fused,
                               dWs,
                               wsSize,
                               algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no MXFP8-input PartialRMS MXFP8-quant solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    // CPU reference: the K1 PartialRMS kernel stores MXFP8_quant(gamma * H) as fp8 D,
    // where H = A*B + residual. The invRms is computed internally but not applied to D;
    // it is used only in the subsequent row_div step (consumer side).
    std::vector<float> h1(static_cast<size_t>(M) * N, 0.0f);
    for(int64_t mt = 0; mt < M; ++mt)
        for(int64_t nh = 0; nh < N; ++nh)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += aF32[kk + mt * K] * bF32[kk + nh * K];
            acc += bf16_to_f32(hResidual[mt * N + nh]);
            h1[mt * N + nh] = acc;
        }
    const MxFp8Ref ref
        = referenceProducerMxfp8(h1, hGamma, M, N, blockSize, paddedRows, paddedCols);
    expectMxScaleEqual(hMxScale, ref.mxScale);
    ASSERT_EQ(hD.size(), ref.dFp8.size());
    const size_t mismatches = countFp8Mismatches(hD, ref.dFp8);
    EXPECT_EQ(mismatches, 0u) << "D e4m3 output has " << mismatches << " mismatches";

    // Verify the bf16 residual-out dual-store: it holds the pre-quantisation
    // H = A*B + residual (same value as h1), stored as bf16.
    std::vector<uint16_t> hResidualOut(static_cast<size_t>(M) * N);
    ASSERT_EQ(hipMemcpy(hResidualOut.data(), dResidualOut, residualOutSz, hipMemcpyDeviceToHost),
              hipSuccess);
    // Relaxed tolerances: bf16 rounding (~0.5 ULP) plus parallel-vs-sequential fp accumulation.
    expectBf16Near(hResidualOut, h1, 1e-2f, 0.10f);

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dMxScaleA));
    static_cast<void>(hipFree(dMxScaleB));
    static_cast<void>(hipFree(dResidualOut));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end: PartialRMS MXFP8, bf16 A/B inputs, no residual-out ----
//
// Exercises the RESIDUAL_ADD -> RMSNORM -> REQUANT fused chain with bf16 A and B inputs.
// The output D is MXFP8-quantised fp8-e4m3. No dual-store residual-out buffer is requested.

TEST(FusedEpilogueE2E, partialRmsBf16InputMxfp8QuantMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "partialRMS bf16-input MXFP8-quant epilogue is wired for gfx950 only";

    // M=256 (multiple of MacroTile1=128), N=512 (multiple of MacroTile0=64), K=64.
    const int64_t M         = 256;
    const int64_t N         = 512;
    const int64_t K         = 64;
    const int32_t blockSize = 32;
    const float   eps       = 1e-5f;

    // Output MX scale geometry (post-PartialRMS transpose: free0=N_hidden, free1=M_tokens).
    const int64_t kBlockTiles = (N + blockSize - 1) / blockSize;
    const int64_t freeTiles   = M;
    const int64_t paddedRows  = ((freeTiles + 31) / 32) * 32;
    const int64_t paddedCols  = ((kBlockTiles + 7) / 8) * 8;
    const size_t  scaleBufSz  = static_cast<size_t>(paddedRows) * paddedCols;

    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    std::vector<uint16_t> hResidual(static_cast<size_t>(M) * N);

    std::mt19937                          rng(2101);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomBf16(hA, rng, dist);
    fillRandomBf16(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hResidual, rng, dist);

    std::vector<float> aF32(hA.size()), bF32(hB.size());
    for(size_t i = 0; i < hA.size(); ++i)
        aF32[i] = bf16_to_f32(hA[i]);
    for(size_t i = 0; i < hB.size(); ++i)
        bF32[i] = bf16_to_f32(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    void *dGamma = nullptr, *dResidual = nullptr, *dMxScale = nullptr, *dWs = nullptr;
    const size_t wsSize = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, static_cast<size_t>(M) * N), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, scaleBufSz), hipSuccess);

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
    hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY, &gran, sizeof(gran)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER, &dMxScale, sizeof(dMxScale)),
        HIPBLAS_STATUS_SUCCESS);
    int32_t bs = blockSize;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE, &bs, sizeof(bs)),
              HIPBLAS_STATUS_SUCCESS);
    hipDataType outType = HIP_R_8F_E4M3;
    ASSERT_EQ(
        hipblasLtFusedEpilogueSetAttribute(
            fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE, &outType, sizeof(outType)),
        HIPBLAS_STATUS_SUCCESS);

    int algoCount = 0;
    ASSERT_EQ(
        runBf16TnFusedMatmulFp8D(
            handle, M, N, K, dA, K, dB, dC, dD, fused, dWs, wsSize, algoCount),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no bf16-input PartialRMS MXFP8-quant solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint8_t> hD(static_cast<size_t>(M) * N);
    std::vector<uint8_t> hMxScale(scaleBufSz);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size(), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(hMxScale.data(), dMxScale, scaleBufSz, hipMemcpyDeviceToHost), hipSuccess);

    // CPU reference: the K1 PartialRMS kernel stores MXFP8_quant(gamma * H) as fp8 D,
    // where H = A*B + residual. The invRms is computed internally but applied in the
    // separate row_div step; the fp8 D in dD is the pre-normalization K1 output.
    // This matches the decomposed-producer reference (validateProducer) which also uses h1
    // directly without invRms.
    std::vector<float> h1(static_cast<size_t>(M) * N, 0.0f);
    for(int64_t mt = 0; mt < M; ++mt)
        for(int64_t nh = 0; nh < N; ++nh)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += aF32[kk + mt * K] * bF32[kk + nh * K];
            acc += bf16_to_f32(hResidual[mt * N + nh]);
            h1[mt * N + nh] = acc;
        }
    const MxFp8Ref ref
        = referenceProducerMxfp8(h1, hGamma, M, N, blockSize, paddedRows, paddedCols);
    expectMxScaleEqual(hMxScale, ref.mxScale);
    ASSERT_EQ(hD.size(), ref.dFp8.size());
    const size_t mismatches = countFp8Mismatches(hD, ref.dFp8);
    EXPECT_LE(mismatches, 8u) << "D e4m3 output has " << mismatches << " mismatches";

    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dD));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dWs));
}

// ---- End-to-end: chained MXfp8 producer/consumer with MX input scales + bf16 residualOut ----
//
// GEMM1 is F8F8S with MXAE8B32/MXBE8B32 input scales (uniform-127, i.e. scale=1).
// The epilogue chain is RESIDUAL_ADD -> PARTIAL_RMSNORM_STATS -> REQUANT(MX) with a bf16
// residualOut dual-store.  GEMM2 (consumer) applies the per-row rstd, producing bf16 D2.
// This exercises the partialrms_residual_mxfp8quant_residualout_scaled_mxfp8_k1 kernel.
TEST(FusedEpilogueE2E, chainedMxfp8ScaledResidualOutProducerConsumerMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "chained MXfp8 scaled residual-out flow is wired for gfx950 only";

    const TypedTestDims d      = makeTypedTestDims(HIP_R_8F_E4M3);
    const size_t        wsSize = size_t(256) * 1024 * 1024;

    // Input MX scale geometry: one UE8M0 byte per (row-block, k-block), k-block cols
    // padded to a multiple of 8 as required by the GFX950 swizzle.
    const int64_t inputScaleColsK = ((d.k0 / d.blockSize + 7) / 8) * 8;
    const size_t  aScaleSz
        = static_cast<size_t>(((d.mTok + 31) / 32) * 32) * inputScaleColsK;
    const size_t bScaleSz
        = static_cast<size_t>(((d.nHid + 31) / 32) * 32) * inputScaleColsK;
    const size_t residualOutSz = static_cast<size_t>(d.mTok) * d.nHid * sizeof(uint16_t);

    std::vector<uint8_t>  hA(d.szABytes), hB(d.szBBytes);
    std::vector<uint16_t> hGamma(static_cast<size_t>(d.nHid));
    std::vector<uint16_t> hW1(static_cast<size_t>(d.nHid) * d.nOut);
    std::vector<uint16_t> hResidual(static_cast<size_t>(d.mTok) * d.nHid);

    std::mt19937                          rng(31416);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    fillRandomF8(hA, rng, dist);
    fillRandomF8(hB, rng, dist);
    fillRandomBf16(hGamma, rng, gdist);
    fillRandomBf16(hW1, rng, dist);
    fillRandomBf16(hResidual, rng, dist);

    std::vector<float> aF32(d.szABytes), bF32(d.szBBytes);
    for(size_t i = 0; i < d.szABytes; ++i)
        aF32[i] = unpackF8(hA[i]);
    for(size_t i = 0; i < d.szBBytes; ++i)
        bF32[i] = unpackF8(hB[i]);

    void *dA = nullptr, *dB = nullptr, *dGamma = nullptr;
    void *dD1 = nullptr, *dMxScale = nullptr, *dWs = nullptr;
    void *dW1 = nullptr, *dD2 = nullptr, *dResidual = nullptr, *dRstd = nullptr;
    void *dMxScaleA = nullptr, *dMxScaleB = nullptr, *dResidualOut = nullptr;
    ASSERT_EQ(hipMalloc(&dA, d.szABytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, d.szBBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, static_cast<size_t>(d.nHid) * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD1, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScale, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(
        hipMalloc(&dW1, static_cast<size_t>(d.nHid) * d.nOut * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(
        hipMalloc(&dD2, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    ASSERT_EQ(
        hipMalloc(&dResidual, static_cast<size_t>(d.mTok) * d.nHid * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScaleA, aScaleSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dMxScaleB, bScaleSz), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidualOut, residualOutSz), hipSuccess);

    ASSERT_EQ(hipMemcpy(dA, hA.data(), d.szABytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), d.szBBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma,
                        hGamma.data(),
                        static_cast<size_t>(d.nHid) * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1,
                        hW1.data(),
                        static_cast<size_t>(d.nHid) * d.nOut * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual,
                        hResidual.data(),
                        static_cast<size_t>(d.mTok) * d.nHid * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD1, 0, static_cast<size_t>(d.mTok) * d.nHid), hipSuccess);
    ASSERT_EQ(hipMemset(dMxScale, 0, d.scaleBufSz), hipSuccess);
    ASSERT_EQ(hipMemset(dD2, 0, static_cast<size_t>(d.mTok) * d.nOut * sizeof(uint16_t)),
              hipSuccess);
    ASSERT_EQ(hipMemset(dResidualOut, 0, residualOutSz), hipSuccess);

    // UE8M0 byte 127 encodes 2^(127-127)=1.0; uniform-127 scales exercise the MXSA/B
    // code path without altering the numeric reference.
    const std::vector<uint8_t> hScaleA(aScaleSz, 127u);
    const std::vector<uint8_t> hScaleB(bScaleSz, 127u);
    ASSERT_EQ(hipMemcpy(dMxScaleA, hScaleA.data(), aScaleSz, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dMxScaleB, hScaleB.data(), bScaleSz, hipMemcpyHostToDevice), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Caller-owned rstd handoff shared by both GEMM calls.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
                  stats, dRstd, static_cast<size_t>(d.rstdRows) * sizeof(float)),
              HIPBLAS_STATUS_SUCCESS);

    // Producer: RESIDUAL_ADD -> PARTIAL_RMSNORM_STATS -> REQUANT(MX) with bf16 residualOut.
    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_NO_FATAL_FAILURE(buildProducerDescriptor(
        dGamma, d.eps, stats, dMxScale, d.blockSize, &prod, dResidual, dResidualOut));

    ProducerResults pr;
    ASSERT_NO_FATAL_FAILURE(launchProducerAndReadback(handle,
                                                      d,
                                                      HIP_R_8F_E4M3,
                                                      dA,
                                                      dB,
                                                      dD1,
                                                      dMxScale,
                                                      prod,
                                                      dRstd,
                                                      dWs,
                                                      wsSize,
                                                      pr,
                                                      dMxScaleA,
                                                      dMxScaleB,
                                                      dResidualOut));
    ASSERT_GT(pr.algoCount, 0)
        << "no partialrms_residual_mxfp8quant_residualout_scaled_mxfp8_k1 solution selected";

    std::vector<float> h1;
    ASSERT_NO_FATAL_FAILURE(
        validateProducer(d, aF32, bF32, hGamma, pr, HIP_R_8F_E4M3, h1, &hResidual));

    const ConsumerQuantData cq = buildConsumerQuantData(d, hW1, pr.hD1, pr.hMxScale);
    ASSERT_NO_FATAL_FAILURE(runConsumerAndValidate(handle, d, h1, cq, stats, dD2, dWs, wsSize));

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    static_cast<void>(hipFree(dA));
    static_cast<void>(hipFree(dB));
    static_cast<void>(hipFree(dGamma));
    static_cast<void>(hipFree(dD1));
    static_cast<void>(hipFree(dMxScale));
    static_cast<void>(hipFree(dW1));
    static_cast<void>(hipFree(dD2));
    static_cast<void>(hipFree(dResidual));
    static_cast<void>(hipFree(dMxScaleA));
    static_cast<void>(hipFree(dMxScaleB));
    static_cast<void>(hipFree(dResidualOut));
    static_cast<void>(hipFree(dRstd));
    static_cast<void>(hipFree(dWs));
}
