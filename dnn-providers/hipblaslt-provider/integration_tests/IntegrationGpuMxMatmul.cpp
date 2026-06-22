// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>

#include "IntegrationGpuMatmulBase.hpp"
#include "utils/MxMatmulUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_test_sdk::utilities;
using namespace hipblaslt_plugin::test_utilities;
using namespace test_matmul_common;
using namespace test_mx_matmul_common;

namespace
{

template <typename InputDataType, typename OutputDataType>
class IntegrationGpuMxMatmul
    : public IntegrationGpuMatmulBase<OutputDataType, MatmulTestCase, float>
{
protected:
    /// Gate all MX tests on GPU arch. Called after the parent SetUp which
    /// gates on device presence — if that already skipped, we never reach here.
    void SetUp() override
    {
        using Base = IntegrationGpuMatmulBase<OutputDataType, MatmulTestCase, float>;
        Base::SetUp();

        if(::testing::Test::IsSkipped())
        {
            return;
        }

        const auto archName = hipdnn_test_sdk::utilities::currentDeviceArch();
        if(!isMxSupportedArch(archName))
        {
            GTEST_SKIP() << "MX block-scaled GEMM is not supported on " << archName;
        }
    }

    std::shared_ptr<graph::TensorAttributes>
        initGraph(const MatmulTestCase& testParams,
                  hipdnn_frontend::graph::Graph& graphObj) const override
    {
        const auto inType = hipdnn_frontend::getDataTypeEnumFromType<InputDataType>();

        // Logical matmul dims live in the last two axes; any leading axes are
        // batch (which MX requires to be 1). A is [..., M, K], B is [..., K, N].
        const auto& aDims = testParams.aDims;
        const auto& bDims = testParams.bDims;
        const int64_t scaleK = aDims[aDims.size() - 1] / 32;

        // A from the case (transA → col-major, opA=T).
        auto aAttr = graph::makeTensorAttributes(
            "a", inType, aDims, generateInputStrideOrder(aDims, testParams.transA));
        auto aTensor = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        // Scale_A mirrors A's shape with the K axis split into 32-wide blocks.
        std::vector<int64_t> scaleADims = aDims;
        scaleADims.back() = scaleK;
        auto scaleAAttr = graph::makeTensorAttributes("scale_a",
                                                      hipdnn_frontend::DataType::FP8_E8M0,
                                                      scaleADims,
                                                      generateInputStrideOrder(scaleADims, false));
        auto scaleATensor = std::make_shared<graph::TensorAttributes>(std::move(scaleAAttr));

        graph::BlockScaleDequantizeAttributes deqAttrA;
        deqAttrA.set_block_size(32);
        auto yATensor = graphObj.block_scale_dequantize(aTensor, scaleATensor, deqAttrA);

        // B from the case (transB → row-major, opB=N).
        auto bAttr = graph::makeTensorAttributes(
            "b", inType, bDims, generateInputStrideOrder(bDims, testParams.transB));
        auto bTensor = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        // Scale_B mirrors B's shape with the K axis split into 32-wide blocks.
        std::vector<int64_t> scaleBDims = bDims;
        scaleBDims[scaleBDims.size() - 2] = scaleK;
        auto scaleBAttr = graph::makeTensorAttributes("scale_b",
                                                      hipdnn_frontend::DataType::FP8_E8M0,
                                                      scaleBDims,
                                                      generateInputStrideOrder(scaleBDims, false));
        auto scaleBTensor = std::make_shared<graph::TensorAttributes>(std::move(scaleBAttr));

        // B is [..., K, N]: the 32-wide blocks run along K, the second-to-last
        // axis. block_size maps to trailing axes, so block K by 32 and N by 1.
        graph::BlockScaleDequantizeAttributes deqAttrB;
        deqAttrB.set_block_size(std::vector<int32_t>{32, 1});
        auto yBTensor = graphObj.block_scale_dequantize(bTensor, scaleBTensor, deqAttrB);

        graph::MatmulAttributes const matmulAttrs;
        return graphObj.matmul(yATensor, yBTensor, matmulAttrs);
    }

    std::string getGraphName() const override
    {
        return "MxMatmulTest";
    }

    unsigned int getSeed(const MatmulTestCase& testParams) const override
    {
        return testParams.seed;
    }
};

// Input (FP8 OCP) × output (FP16 / BF16 / FP32) combinations.
using IntegrationGpuMxGemmE4M3ToFp16
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e4m3, hipdnn_data_sdk::types::half>;
using IntegrationGpuMxGemmE4M3ToBf16
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e4m3, hipdnn_data_sdk::types::bfloat16>;
using IntegrationGpuMxGemmE4M3ToFp32
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e4m3, float>;
using IntegrationGpuMxGemmE5M2ToFp16
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e5m2, hipdnn_data_sdk::types::half>;
using IntegrationGpuMxGemmE5M2ToBf16
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e5m2, hipdnn_data_sdk::types::bfloat16>;
using IntegrationGpuMxGemmE5M2ToFp32
    = IntegrationGpuMxMatmul<hipdnn_data_sdk::types::fp8_e5m2, float>;

} // namespace

TEST_P(IntegrationGpuMxGemmE4M3ToFp16, Correctness)
{
    runGraphTest(matmul::getMxTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMxGemmE4M3ToBf16, Correctness)
{
    runGraphTest(matmul::getMxTolerance<hipdnn_data_sdk::types::bfloat16>());
}

TEST_P(IntegrationGpuMxGemmE4M3ToFp32, Correctness)
{
    runGraphTest(matmul::getMxTolerance<float>());
}

TEST_P(IntegrationGpuMxGemmE5M2ToFp16, Correctness)
{
    runGraphTest(matmul::getMxTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMxGemmE5M2ToBf16, Correctness)
{
    runGraphTest(matmul::getMxTolerance<hipdnn_data_sdk::types::bfloat16>());
}

TEST_P(IntegrationGpuMxGemmE5M2ToFp32, Correctness)
{
    runGraphTest(matmul::getMxTolerance<float>());
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE4M3ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE4M3ToBf16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE4M3ToFp32,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE5M2ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE5M2ToBf16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMxMatmul,
                         IntegrationGpuMxGemmE5M2ToFp32,
                         testing::ValuesIn(getMxMatmulTestCases()));
