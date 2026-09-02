// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

struct ResampleBwdTestCase
{
    std::string name;
    std::vector<int64_t> dyDims;
    TensorLayout layout;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    ResampleMode resampleMode;
    PaddingMode paddingMode = PaddingMode::ZERO_PAD;
};

std::vector<ResampleBwdTestCase> getResampleBwdTestCases()
{
    struct TensorCase
    {
        std::string name;
        std::vector<int64_t> dyDims;
        TensorLayout layout;
    };

    struct ParameterCase
    {
        std::string name;
        std::vector<int64_t> prePadding;
        std::vector<int64_t> postPadding;
        std::vector<int64_t> stride;
        std::vector<int64_t> window;
        ResampleMode resampleMode;
        PaddingMode paddingMode = PaddingMode::ZERO_PAD;
    };

    const std::vector<TensorCase> tensorCases{{"2d_nchw", {2, 3, 7, 5}, TensorLayout::NCHW},
                                              {"2d_nhwc", {2, 3, 7, 5}, TensorLayout::NHWC},
                                              {"2d_wide", {1, 2, 8, 6}, TensorLayout::NCHW},
                                              {"3d_ncdhw", {1, 2, 4, 5, 3}, TensorLayout::NCDHW},
                                              {"3d_ndhwc", {1, 2, 4, 5, 3}, TensorLayout::NDHWC}};

    const std::vector<ParameterCase> twoDimParameterCases{
        {"max_pool", {1, 1}, {1, 1}, {2, 2}, {3, 3}, ResampleMode::MAXPOOL},
        {"avg_exclude", {1, 0}, {0, 1}, {2, 1}, {3, 2}, ResampleMode::AVGPOOL_EXCLUDE_PADDING},
        {"avg_include", {0, 1}, {1, 0}, {1, 2}, {2, 3}, ResampleMode::AVGPOOL_INCLUDE_PADDING}};

    const std::vector<ParameterCase> threeDimParameterCases{
        {"max_pool", {0, 0, 0}, {0, 0, 0}, {1, 2, 1}, {2, 2, 2}, ResampleMode::MAXPOOL},
        {"avg_exclude",
         {0, 1, 0},
         {1, 0, 1},
         {1, 2, 1},
         {2, 2, 2},
         ResampleMode::AVGPOOL_EXCLUDE_PADDING},
        {"avg_include",
         {1, 0, 1},
         {0, 1, 0},
         {1, 2, 1},
         {2, 2, 2},
         ResampleMode::AVGPOOL_INCLUDE_PADDING}};

    std::vector<ResampleBwdTestCase> testCases;
    for(const auto& tensorCase : tensorCases)
    {
        const auto& parameterCases
            = tensorCase.dyDims.size() == 4 ? twoDimParameterCases : threeDimParameterCases;
        for(const auto& parameterCase : parameterCases)
        {
            testCases.push_back({tensorCase.name + "_" + parameterCase.name,
                                 tensorCase.dyDims,
                                 tensorCase.layout,
                                 parameterCase.prePadding,
                                 parameterCase.postPadding,
                                 parameterCase.stride,
                                 parameterCase.window,
                                 parameterCase.resampleMode});
        }
    }

    return testCases;
}

template <typename T>
constexpr float getTolerance()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1e-5f;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return 1e-3f;
    }
    else
    {
        static_assert(std::is_same_v<T, bfloat16>);
        return 1e-2f;
    }
}

template <typename DyDataType, typename DxDataType, typename ComputeDataType>
class ResampleBackward : public IntegrationGraphVerificationHarness<DyDataType, ResampleBwdTestCase>
{
protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ResampleBwdTest");

        auto dyDataType = getDataTypeEnumFromType<DyDataType>();
        auto dxDataType = getDataTypeEnumFromType<DxDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dyDataType);

        auto dyAttr
            = makeTensorAttributes("DY",
                                   dyDataType,
                                   testCase.dyDims,
                                   generateStrides(testCase.dyDims, testCase.layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // Maxpool requires an index tensor as input
        std::shared_ptr<graph::TensorAttributes> indexTensorAttr;
        if(testCase.resampleMode == ResampleMode::MAXPOOL)
        {
            auto indexAttr = makeTensorAttributes(
                "INDEX",
                hipdnn_frontend::DataType::INT32,
                testCase.dyDims,
                generateStrides(testCase.dyDims, testCase.layout.strideOrder));
            indexTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(indexAttr));
        }

        graph::ResampleBwdAttributes resampleAttrs;
        resampleAttrs.set_pre_padding(testCase.prePadding)
            .set_post_padding(testCase.postPadding)
            .set_stride(testCase.stride)
            .set_window(testCase.window)
            .set_resample_mode(testCase.resampleMode)
            .set_padding_mode(testCase.paddingMode);

        auto dxTensorAttr = graphObj.resample_bwd(dyTensorAttr, resampleAttrs, indexTensorAttr);
        dxTensorAttr->set_output(true);
        dxTensorAttr->set_data_type(dxDataType);
        this->registerValidator(dxTensorAttr, getTolerance<DxDataType>());

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(getSharedHandle());
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        this->inputFillRecipes().setGlobalSeed(getGlobalTestSeed());
        verifyResampleBwdGraph(graphObj,
                               dxTensorAttr,
                               indexTensorAttr,
                               testCase.prePadding,
                               testCase.stride,
                               testCase.window,
                               testCase.resampleMode);
    }

    void verifyResampleBwdGraph(hipdnn_frontend::graph::Graph& graph,
                                const std::shared_ptr<graph::TensorAttributes>& dxTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& indexTensorAttr,
                                const std::vector<int64_t>& prePadding,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& window,
                                ResampleMode resampleMode)
    {
        // This function is the same as verifyGraph() from the IntegrationGraphVerificationHarness
        // but also computes the index tensor required for the backward pass of maxpool resampling
        // using the forward CPU reference implementation
        if(TestConfig::get().hasCaptureDir())
        {
            this->captureGraphBundle(graph);
            auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            HIPDNN_SDK_LOG_INFO("Capture-only mode: skipping execution for "
                                << (info ? info->test_suite_name() : "?") << "."
                                << (info ? info->name() : "?"));
            return;
        }

        ASSERT_NO_FATAL_FAILURE(this->ensureEngineSupport(graph));
        if(testing::Test::IsSkipped())
        {
            return;
        }

        if(TestConfig::get().skipGraphValidation())
        {
            return;
        }

        ASSERT_NO_FATAL_FAILURE(this->buildExecutionPlans(graph));

        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
        hipdnn_test_sdk::utilities::GraphTensorBundle refBundle;
        this->generateBundles(graph, refBundle, gpuBundle);

        auto initResult = this->initializeBundle(graph, gpuBundle);
        if(!initResult.filled)
        {
            GTEST_SKIP() << initResult.reason;
        }
        initResult = this->initializeBundle(graph, refBundle);
        if(!initResult.filled)
        {
            GTEST_SKIP() << initResult.reason;
        }

        // Compute the index tensor using the forward CPU reference implementation
        if(resampleMode == ResampleMode::MAXPOOL && indexTensorAttr != nullptr)
        {
            const auto& dyDims = indexTensorAttr->get_dim();
            const auto& dyStrides = indexTensorAttr->get_stride();

            Tensor<float> xScratch(dxTensorAttr->get_dim(), dxTensorAttr->get_stride());
            Tensor<float> yScratch(dyDims, dyStrides);
            Tensor<int32_t> indexScratch(dyDims, dyStrides);
            xScratch.fillWithRandomValues(-1.0f, 1.0f, getGlobalTestSeed());

            CpuFpReferenceResampleFwd::forward<float, float, float, int32_t>(
                xScratch,
                yScratch,
                prePadding,
                stride,
                window,
                hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
                hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD,
                &indexScratch);

            // Copy the index tensor to both bundles
            const auto indexUid = indexTensorAttr->get_uid();
            std::memcpy(refBundle.tensors.at(indexUid)->rawHostData(),
                        indexScratch.rawHostData(),
                        indexScratch.elementSpace() * sizeof(int32_t));
            refBundle.tensors.at(indexUid)->markHostModified();
            std::memcpy(gpuBundle.tensors.at(indexUid)->rawHostData(),
                        indexScratch.rawHostData(),
                        indexScratch.elementSpace() * sizeof(int32_t));
            gpuBundle.tensors.at(indexUid)->markHostModified();
        }

        ASSERT_NO_FATAL_FAILURE(this->executeGpuGraph(getSharedHandle(), graph, gpuBundle));
        ASSERT_NO_FATAL_FAILURE(this->executeReferenceGraph(graph, refBundle));

        ASSERT_NO_FATAL_FAILURE(this->validateOutputs(gpuBundle, refBundle));
    }
};

using IntegrationGpuResampleBackwardFp32 = ResampleBackward<float, float, float>;
using IntegrationGpuResampleBackwardFp16 = ResampleBackward<half, half, float>;
using IntegrationGpuResampleBackwardBfp16 = ResampleBackward<bfloat16, bfloat16, float>;

} // namespace

TEST_P(IntegrationGpuResampleBackwardFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuResampleBackwardFp32,
                         testing::ValuesIn(getResampleBwdTestCases()));

TEST_P(IntegrationGpuResampleBackwardFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuResampleBackwardFp16,
                         testing::ValuesIn(getResampleBwdTestCases()));

TEST_P(IntegrationGpuResampleBackwardBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuResampleBackwardBfp16,
                         testing::ValuesIn(getResampleBwdTestCases()));
