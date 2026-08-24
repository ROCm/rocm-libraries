// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/MxMatmulCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_matmul_common;
using namespace test_mx_matmul_common;

namespace
{

// MX dequantize -> GEMM with independent A/B input element types, for mixed
// operands (e.g. FP8 OCP A + FP4 B), which hipBLASLt supports.
template <typename InputDataTypeA, typename InputDataTypeB, typename OutputDataType>
class MxMatmulMixed : public IntegrationGraphVerificationHarness<OutputDataType, MatmulTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> c;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const MatmulTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("MxMatmulTest");

        auto outType = getDataTypeEnumFromType<OutputDataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(outType);

        const auto aType = getDataTypeEnumFromType<InputDataTypeA>();
        const auto bType = getDataTypeEnumFromType<InputDataTypeB>();

        // Logical matmul dims live in the last two axes; any leading axes are
        // batch (which MX requires to be 1). A is [..., M, K], B is [..., K, N].
        const auto& aDims = tc.aDims;
        const auto& bDims = tc.bDims;
        const int64_t scaleK = aDims[aDims.size() - 1] / 32;

        // A from the case (transA → col-major, opA=T).
        auto aAttr = graph::makeTensorAttributes(
            "a", aType, aDims, generateInputStrideOrder(aDims, tc.transA));
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
            "b", bType, bDims, generateInputStrideOrder(bDims, tc.transB));
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

        const graph::MatmulAttributes matmulAttrs;
        auto cAttr = graphObj.matmul(yATensor, yBTensor, matmulAttrs);
        cAttr->set_output(true);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{cAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        // MX block-scale dequantization introduces additional quantization
        // error beyond plain matmul; use the MX-specific tolerance directly.
        this->registerValidator(outputs.c, matmul::getMxTolerance<OutputDataType>());

        this->inputFillRecipes().setGlobalSeed(testCase.seed);
        this->verifyGraph(graphObj);
    }
};

template <typename InputDataType, typename OutputDataType>
class MxMatmul : public MxMatmulMixed<InputDataType, InputDataType, OutputDataType>
{
};

// Input (FP8 OCP) × output (FP16 / BF16 / FP32) combinations.
using IntegrationGpuMxMatmulE4M3ToFp16 = MxMatmul<fp8_e4m3, half>;
using IntegrationGpuMxMatmulE4M3ToBf16 = MxMatmul<fp8_e4m3, bfloat16>;
using IntegrationGpuMxMatmulE4M3ToFp32 = MxMatmul<fp8_e4m3, float>;
using IntegrationGpuMxMatmulE5M2ToFp16 = MxMatmul<fp8_e5m2, half>;
using IntegrationGpuMxMatmulE5M2ToBf16 = MxMatmul<fp8_e5m2, bfloat16>;
using IntegrationGpuMxMatmulE5M2ToFp32 = MxMatmul<fp8_e5m2, float>;

// Input (FP4 E2M1) × output (FP16 / BF16 / FP32) combinations.
using IntegrationGpuMxMatmulE2M1ToFp16 = MxMatmul<fp4_e2m1, half>;
using IntegrationGpuMxMatmulE2M1ToBf16 = MxMatmul<fp4_e2m1, bfloat16>;
using IntegrationGpuMxMatmulE2M1ToFp32 = MxMatmul<fp4_e2m1, float>;

// Mixed operands: FP8 OCP A × FP4 B (and the reverse) → FP16.
using IntegrationGpuMxMatmulMixedE4M3E2M1ToFp16 = MxMatmulMixed<fp8_e4m3, fp4_e2m1, half>;
using IntegrationGpuMxMatmulMixedE2M1E4M3ToFp16 = MxMatmulMixed<fp4_e2m1, fp8_e4m3, half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE4M3ToFp16);
TEST_P(IntegrationGpuMxMatmulE4M3ToFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE4M3ToBf16);
TEST_P(IntegrationGpuMxMatmulE4M3ToBf16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE4M3ToFp32);
TEST_P(IntegrationGpuMxMatmulE4M3ToFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE5M2ToFp16);
TEST_P(IntegrationGpuMxMatmulE5M2ToFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE5M2ToBf16);
TEST_P(IntegrationGpuMxMatmulE5M2ToBf16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE5M2ToFp32);
TEST_P(IntegrationGpuMxMatmulE5M2ToFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE2M1ToFp16);
TEST_P(IntegrationGpuMxMatmulE2M1ToFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE2M1ToBf16);
TEST_P(IntegrationGpuMxMatmulE2M1ToBf16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulE2M1ToFp32);
TEST_P(IntegrationGpuMxMatmulE2M1ToFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulMixedE4M3E2M1ToFp16);
TEST_P(IntegrationGpuMxMatmulMixedE4M3E2M1ToFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMxMatmulMixedE2M1E4M3ToFp16);
TEST_P(IntegrationGpuMxMatmulMixedE2M1E4M3ToFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE4M3ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE4M3ToBf16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE4M3ToFp32,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE5M2ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE5M2ToBf16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE5M2ToFp32,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE2M1ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE2M1ToBf16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulE2M1ToFp32,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulMixedE4M3E2M1ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMxMatmulMixedE2M1E4M3ToFp16,
                         testing::ValuesIn(getMxMatmulTestCases()));
