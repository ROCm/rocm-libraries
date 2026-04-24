// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;
using namespace hip_kernel_provider::test_utilities;

namespace
{

struct SdpaTestCase
{
    SdpaTestCase(std::vector<int64_t> qDimsIn,
                 std::vector<int64_t> kDimsIn,
                 std::vector<int64_t> vDimsIn,
                 std::optional<float> attnScaleValueIn = std::nullopt,
                 std::vector<int64_t> attnMaskDimsIn = {},
                 bool causalMaskIn = false)
        : qDims(std::move(qDimsIn))
        , kDims(std::move(kDimsIn))
        , vDims(std::move(vDimsIn))
        , qStrides(generateStrides(qDims))
        , kStrides(generateStrides(kDims))
        , vStrides(generateStrides(vDims))
        , attnScaleValue(attnScaleValueIn)
        , attnMaskDims(std::move(attnMaskDimsIn))
        , attnMaskStrides(generateStrides(attnMaskDims))
        , causalMask(causalMaskIn)
    {
    }

    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
    std::vector<int64_t> qStrides;
    std::vector<int64_t> kStrides;
    std::vector<int64_t> vStrides;

    std::optional<float> attnScaleValue;
    std::vector<int64_t> attnMaskDims;
    std::vector<int64_t> attnMaskStrides;
    bool causalMask;
};

std::vector<SdpaTestCase> getSdpaTestCases()
{
    return {SdpaTestCase({4, 8, 256, 128}, {4, 8, 256, 128}, {4, 8, 256, 128})};
}

template <typename DataType>
class SdpaForward : public IntegrationGraphVerificationHarness<DataType, SdpaTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {

        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
        }
    }

    void runGraphTest(float tolerance)
    {

        if(hip_kernel_provider_common::getDeviceString(this->stream()) != "gfx942")
        {
            GTEST_SKIP() << "Skipped: ASM SDPA kernel only supports gfx942.";
        }

        const SdpaTestCase& testCase = this->GetParam();

        Graph graph;
        graph.set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_dim(testCase.qDims)
            .set_stride(testCase.qStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto k = std::make_shared<TensorAttributes>();
        k->set_dim(testCase.kDims)
            .set_stride(testCase.kStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto v = std::make_shared<TensorAttributes>();
        v->set_dim(testCase.vDims)
            .set_stride(testCase.vStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        SdpaAttributes attributes;
        attributes.set_name("SdpaNode");
        if(testCase.attnScaleValue.has_value())
        {
            attributes.set_attn_scale_value(testCase.attnScaleValue.value());
        }
        if(!testCase.attnMaskDims.empty())
        {
            auto attnMask = std::make_shared<TensorAttributes>();
            attnMask->set_dim(testCase.attnMaskDims)
                .set_stride(testCase.attnMaskStrides)
                .set_data_type(getDataTypeEnumFromType<DataType>());
            attributes.set_bias(attnMask);
        }
        attributes.set_causal_mask(testCase.causalMask);

        auto [o, stats] = graph.sdpa(q, k, v, attributes);

        o->set_output(true);
        o->set_data_type(getDataTypeEnumFromType<DataType>());

        auto validationResult = graph.validate();
        EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

        this->registerValidator(o, tolerance);
        this->verifyGraph(graph, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

using IntegrationGpuSdpaFwdBf16 = SdpaForward<bfloat16>;

} // namespace

TEST_P(IntegrationGpuSdpaFwdBf16, Correctness)
{
    auto tolerance = 1e-2f;

    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuSdpaFwdBf16, testing::ValuesIn(getSdpaTestCases()));
