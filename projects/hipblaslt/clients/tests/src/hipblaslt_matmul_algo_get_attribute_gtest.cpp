#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

class HipblasLtMatmulAlgoGetAttributeTest : public ::testing::Test
{
protected:
    hipblasLtHandle_t handle_{nullptr};

    void SetUp() override
    {
        ASSERT_EQ(hipblasLtCreate(&handle_), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(handle_)
            hipblasLtDestroy(handle_);
    }

    hipblasLtMatmulAlgo_t getValidAlgo()
    {
        hipblasLtMatrixLayout_t A, B, C, D;
        hipblasLtMatmulDesc_t   matmulDesc;
        hipblasLtMatmulPreference_t pref;

        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&A, HIP_R_16F, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&B, HIP_R_16F, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&C, HIP_R_16F, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&D, HIP_R_16F, 128, 128, 128),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(hipblasLtMatmulDescCreate(
                      &matmulDesc,
                      HIPBLAS_COMPUTE_32F,
                      HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&pref),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int algoCount = 0;

        EXPECT_EQ(
            hipblasLtMatmulAlgoGetHeuristic(
                handle_,
                matmulDesc,
                A, B, C, D,
                pref,
                1,
                &heuristic,
                &algoCount),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_GT(algoCount, 0);

        hipblasLtMatmulAlgo_t algo = heuristic.algo;

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtMatrixLayoutDestroy(A);
        hipblasLtMatrixLayoutDestroy(B);
        hipblasLtMatrixLayoutDestroy(C);
        hipblasLtMatrixLayoutDestroy(D);

        return algo;
    }
};

TEST_F(HipblasLtMatmulAlgoGetAttributeTest, QueryValidAttributes)
{
    hipblasLtMatmulAlgo_t algo = getValidAlgo();
    size_t sizeWritten = 0;

    size_t ldsBytes = 0;
    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            &algo,
            HIPBLASLT_ALGO_ATTR_LDS_BYTES,
            &ldsBytes,
            sizeof(ldsBytes),
            &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(sizeWritten, sizeof(ldsBytes));
    EXPECT_GT(ldsBytes, 0u);

    int waveCount = 0;
    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            &algo,
            HIPBLASLT_ALGO_ATTR_WAVE_COUNT,
            &waveCount,
            sizeof(waveCount),
            &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_GT(waveCount, 0);

    int isTensorCore = 0;
    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            &algo,
            HIPBLASLT_ALGO_ATTR_IS_TENSOR_CORE,
            &isTensorCore,
            sizeof(isTensorCore),
            &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);

    EXPECT_TRUE(isTensorCore == 0 || isTensorCore == 1);
}

TEST_F(HipblasLtMatmulAlgoGetAttributeTest, InvalidArguments)
{
    hipblasLtMatmulAlgo_t dummy{};
    size_t value = 0;
    size_t written = 0;

    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            nullptr,
            HIPBLASLT_ALGO_ATTR_LDS_BYTES,
            &value,
            sizeof(value),
            &written),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            &dummy,
            HIPBLASLT_ALGO_ATTR_LDS_BYTES,
            nullptr,
            sizeof(value),
            &written),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_EQ(
        hipblasLtMatmulAlgoGetAttribute(
            &dummy,
            static_cast<hipblasLtMatmulAlgoAttribute_t>(999),
            &value,
            sizeof(value),
            &written),
        HIPBLAS_STATUS_NOT_SUPPORTED);
}
