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
class ConvBackwardData : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        assert(_minVal < _maxVal && "Invalid tensor value range");

        bundle.sentinelFillOutputTensors();

        for(auto& tensorPair : bundle.tensors)
        {
            if(!bundle.isOutput(tensorPair.first))
            {
                bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
            }
        }
    }

    // Helper to calculate convolution dgrad tolerance based on tensor ranges and dimensions.
    // Both dy and w tensors are initialized with the same range [_minVal, _maxVal] in initializeBundle().
    template <typename OutputType, typename InputType, typename ComputeType = float>
    float getConvDgradTolerance() const
    {
        const auto& testCase = this->GetParam();
        return calculateConvDgradTolerance<OutputType, InputType, ComputeType>(
            static_cast<double>(_minVal),
            static_cast<double>(_maxVal),
            static_cast<double>(_minVal),
            static_cast<double>(_maxVal),
            testCase.wDims);
    }

    // Fixed small rtol; dynamic atol already accounts for accumulation error.
    static constexpr float RELATIVE_TOLERANCE = 0.01f;

    void runGraphTest(float absoluteTolerance,
                      float relativeTolerance,
                      const TensorLayout& layout = TensorLayout::NCHW)
    {
        // Skipping until CK is working on Windows
        SKIP_IF_WINDOWS();
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();
        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionBackwardDataTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto dyAttr = makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto wAttr = makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvDgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dxTensorAttr = graphObj.conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);
        dxTensorAttr->set_output(true);

        // Set these explicitly since grouped convs cannot infer tensor shape.
        // Infer behavior will assume groups == 1, but some cases have groups > 1.
        dxTensorAttr->set_dim(testCase.xDims);
        dxTensorAttr->set_stride(generateStrides(testCase.xDims, layout.strideOrder));

        this->registerValidator(dxTensorAttr, absoluteTolerance, relativeTolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }

    float _minVal = IntegrationGraphVerificationHarness<DataType, ConvTestCase>::DEFAULT_MIN;
    float _maxVal = IntegrationGraphVerificationHarness<DataType, ConvTestCase>::DEFAULT_MAX;
};

using IntegrationGpuConvBwdDataNchwFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdDataNcdhwFp32 = ConvBackwardData<float>;

using IntegrationGpuConvBwdDataNchwBfp16 = ConvBackwardData<bfloat16>;
using IntegrationGpuConvBwdDataNcdhwBfp16 = ConvBackwardData<bfloat16>;

using IntegrationGpuConvBwdDataNchwFp16 = ConvBackwardData<half>;
using IntegrationGpuConvBwdDataNcdhwFp16 = ConvBackwardData<half>;

using IntegrationGpuConvBwdDataNhwcFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdDataNdhwcFp32 = ConvBackwardData<float>;

using IntegrationGpuConvBwdDataNhwcBfp16 = ConvBackwardData<bfloat16>;
using IntegrationGpuConvBwdDataNdhwcBfp16 = ConvBackwardData<bfloat16>;

using IntegrationGpuConvBwdDataNhwcFp16 = ConvBackwardData<half>;
using IntegrationGpuConvBwdDataNdhwcFp16 = ConvBackwardData<half>;

} // namespace

TEST_P(IntegrationGpuConvBwdDataNchwFp32, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<float, float, float>(), RELATIVE_TOLERANCE, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp32, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<float, float, float>(), RELATIVE_TOLERANCE, TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwBfp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<bfloat16, bfloat16, float>(), RELATIVE_TOLERANCE, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwBfp16, Correctness)
{
    runGraphTest(getConvDgradTolerance<bfloat16, bfloat16, float>(),
                 RELATIVE_TOLERANCE,
                 TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwFp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<half, half, float>(), RELATIVE_TOLERANCE, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<half, half, float>(), RELATIVE_TOLERANCE, TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp32, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<float, float, float>(), RELATIVE_TOLERANCE, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp32, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<float, float, float>(), RELATIVE_TOLERANCE, TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcBfp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<bfloat16, bfloat16, float>(), RELATIVE_TOLERANCE, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcBfp16, Correctness)
{
    runGraphTest(getConvDgradTolerance<bfloat16, bfloat16, float>(),
                 RELATIVE_TOLERANCE,
                 TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<half, half, float>(), RELATIVE_TOLERANCE, TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp16, Correctness)
{
    runGraphTest(
        getConvDgradTolerance<half, half, float>(), RELATIVE_TOLERANCE, TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
