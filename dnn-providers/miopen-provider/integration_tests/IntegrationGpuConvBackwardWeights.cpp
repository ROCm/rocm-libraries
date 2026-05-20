// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ConvolutionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;
using namespace miopen_plugin::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvBackwardWeights : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.sentinelFillOutputTensors();

        for(auto& tensorPair : bundle.tensors)
        {
            if(!bundle.isOutput(tensorPair.first))
            {
                bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
            }
        }
    }

    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        // Skipping until CK is working on Windows
        SKIP_IF_WINDOWS();
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionBackwardWeightTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::ConvWgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dwTensorAttr = graphObj.conv_wgrad(dyTensorAttr, xTensorAttr, convAttrs);
        dwTensorAttr->set_output(true);

        // Set these explicitly since grouped convs cannot infer tensor shape.
        // Infer behavior will assume groups == 1, but some cases have groups > 1.
        dwTensorAttr->set_dim(testCase.wDims);
        dwTensorAttr->set_stride(generateStrides(testCase.wDims, layout.strideOrder));

        this->registerValidator(dwTensorAttr, tolerance, 0.01f);
        this->verifyGraph(graphObj, testCase.seed);
    }

    float _minVal = IntegrationGraphVerificationHarness<DataType, ConvTestCase>::DEFAULT_MIN;
    float _maxVal = IntegrationGraphVerificationHarness<DataType, ConvTestCase>::DEFAULT_MAX;
};

using IntegrationGpuConvWrwDataNchwFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrwDataNcdhwFp32 = ConvBackwardWeights<float>;

using IntegrationGpuConvWrwDataNchwBfp16 = ConvBackwardWeights<bfloat16>;
using IntegrationGpuConvWrwDataNcdhwBfp16 = ConvBackwardWeights<bfloat16>;

using IntegrationGpuConvWrwDataNchwFp16 = ConvBackwardWeights<half>;
using IntegrationGpuConvWrwDataNcdhwFp16 = ConvBackwardWeights<half>;

using IntegrationGpuConvWrwDataNhwcFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrwDataNdhwcFp32 = ConvBackwardWeights<float>;

using IntegrationGpuConvWrwDataNhwcBfp16 = ConvBackwardWeights<bfloat16>;
using IntegrationGpuConvWrwDataNdhwcBfp16 = ConvBackwardWeights<bfloat16>;

using IntegrationGpuConvWrwDataNhwcFp16 = ConvBackwardWeights<half>;
using IntegrationGpuConvWrwDataNdhwcFp16 = ConvBackwardWeights<half>;

template <typename DataType>
class ConvBackwardWeightsLargeValues : public ConvBackwardWeights<DataType>
{
public:
    ConvBackwardWeightsLargeValues()
    {
        this->_minVal = -10.0f;
        this->_maxVal = 10.0f;
    }
};

using IntegrationGpuConvWrwDataLargeValuesFp32 = ConvBackwardWeightsLargeValues<float>;

} // namespace

TEST_P(IntegrationGpuConvWrwDataLargeValuesFp32, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<float, float, float>(static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataLargeValuesFp32,
                         testing::Values(getConvTestCases4D()[0]));

TEST_P(IntegrationGpuConvWrwDataNchwFp32, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<float, float, float>(static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp32, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<float, float, float>(static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwBfp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance
        = calculateConvWrwTolerance<bfloat16, bfloat16, float>(static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwBfp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance
        = calculateConvWrwTolerance<bfloat16, bfloat16, float>(static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwFp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<half, half, float>(static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<half, half, float>(static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp32, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<float, float, float>(static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp32, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<float, float, float>(static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    static_cast<double>(_minVal),
                                                                    static_cast<double>(_maxVal),
                                                                    testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcBfp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance
        = calculateConvWrwTolerance<bfloat16, bfloat16, float>(static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcBfp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance
        = calculateConvWrwTolerance<bfloat16, bfloat16, float>(static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               static_cast<double>(_minVal),
                                                               static_cast<double>(_maxVal),
                                                               testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<half, half, float>(static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp16, Correctness)
{
    const auto& testCase = GetParam();
    auto tolerance = calculateConvWrwTolerance<half, half, float>(static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  static_cast<double>(_minVal),
                                                                  static_cast<double>(_maxVal),
                                                                  testCase.yDims);
    runGraphTest(tolerance, TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
