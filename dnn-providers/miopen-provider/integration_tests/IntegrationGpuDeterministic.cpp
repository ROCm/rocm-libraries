// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstring>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "../tests/common/ConvolutionCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace test_conv_common;
using namespace test_bn_common;

namespace
{

// ============================================================================
// Helper Functions
// ============================================================================

/// Populates a GraphTensorBundle from a built graph by visiting all nodes
/// and creating tensors for non-virtual tensor attributes.
void populateBundleFromGraph(Graph& graph, GraphTensorBundle& bundle)
{
    graph.visit([&](const INode& node) {
        for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
        {
            int64_t tensorId = tensorAttr->get_uid();
            if(!tensorAttr->get_is_virtual()
               && bundle.tensors.find(tensorId) == bundle.tensors.end())
            {
                bundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
            }
        }
        for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
        {
            int64_t tensorId = tensorAttr->get_uid();
            if(!tensorAttr->get_is_virtual()
               && bundle.tensors.find(tensorId) == bundle.tensors.end())
            {
                bundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
            }
        }
    });
}

/// Randomizes all tensors in a bundle with the given seed.
void randomizeBundle(GraphTensorBundle& bundle,
                     unsigned int seed,
                     float min = -1.0f,
                     float max = 1.0f)
{
    for(auto& [tensorId, tensor] : bundle.tensors)
    {
        bundle.randomizeTensor(tensorId, min, max, seed);
    }
}

/// Compares output tensors between two bundles for bit-exact equality.
/// Uses raw byte comparison to handle all data types correctly.
void assertBundleOutputsMatch(GraphTensorBundle& bundle1,
                              GraphTensorBundle& bundle2,
                              int64_t outputTensorId,
                              hipStream_t stream)
{
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    auto& tensor1 = bundle1.tensors.at(outputTensorId);
    auto& tensor2 = bundle2.tensors.at(outputTensorId);

    tensor1->markDeviceModified();
    tensor2->markDeviceModified();

    auto* host1 = static_cast<const uint8_t*>(tensor1->rawHostData());
    auto* host2 = static_cast<const uint8_t*>(tensor2->rawHostData());

    size_t elementCount = tensor1->elementCount();
    size_t elementSize = tensor1->elementSize();
    size_t totalBytes = elementCount * elementSize;

    ASSERT_EQ(elementCount, tensor2->elementCount()) << "Output tensor sizes differ";

    int mismatchResult = std::memcmp(host1, host2, totalBytes);
    ASSERT_EQ(mismatchResult, 0) << "Output tensors are not bit-exact";
}

// ============================================================================
// Base Test Fixture for Deterministic Tests
// Manages hipdnn handle and HIP stream lifecycle
// ============================================================================

template <typename TestCaseType>
class DeterministicTestBase : public ::testing::TestWithParam<TestCaseType>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

// ============================================================================
// Deterministic Convolution Smoke Test Cases
// ============================================================================

inline std::vector<ConvTestCase> getDeterministicConvTestCases4D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1 - basic case
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Filter 3x3 with padding - common case
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16}, {2, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
    };
}

inline std::vector<ConvTestCase> getDeterministicConvTestCases5D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1x1 - basic 5D case
        {{1, 8, 8, 8, 8}, {1, 8, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Filter 3x3x3 with padding - common 5D case
        {{1, 8, 8, 8, 8}, {1, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
    };
}

// ============================================================================
// Deterministic Batchnorm Test Cases (for no-solver verification)
// ============================================================================

inline std::vector<BatchnormTestCase> getDeterministicBnTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        {{1, 3, 14, 14}, seed}, // Basic inference case
        {{2, 3, 14, 14}, seed}, // Basic training case
    };
}

// ============================================================================
// Fused Convolution Test Cases
// ============================================================================

using FusedConvTestCase = std::tuple<ConvTestCase, bool, test_activation_common::ActivTestCase>;

inline std::vector<FusedConvTestCase> getDeterministicFusedConvTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    std::vector<ConvTestCase> convCases = {
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
    };

    auto activCases = test_activation_common::createFwdActivationSmokeCases();

    std::vector<FusedConvTestCase> fusedCases;
    for(const auto& convCase : convCases)
    {
        for(const auto& activCase : activCases)
        {
            fusedCases.emplace_back(convCase, true, activCase); // With bias
            fusedCases.emplace_back(convCase, false, activCase); // Without bias
        }
    }
    return fusedCases;
}

// ============================================================================
// Convolution Forward Determinism Test
// ============================================================================

template <typename DataType>
class DeterministicConvForward : public DeterministicTestBase<ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();

        const ConvTestCase& testCase = DeterministicTestBase<ConvTestCase>::GetParam();

        // Build the graph
        Graph graphObj;
        graphObj.set_name("DeterministicConvForwardTest");
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder)));
        auto wTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder)));

        ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yAttr->set_output(true);

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // Create two bundles with same random values for determinism check
        GraphTensorBundle bundle1;
        GraphTensorBundle bundle2;
        populateBundleFromGraph(graphObj, bundle1);
        populateBundleFromGraph(graphObj, bundle2);
        randomizeBundle(bundle1, testCase.seed);
        randomizeBundle(bundle2, testCase.seed);

        // Execute twice
        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack1 = bundle1.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack1, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack2 = bundle2.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack2, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        assertBundleOutputsMatch(bundle1, bundle2, yAttr->get_uid(), _stream);
    }
};

using DeterministicConvFwdNchwFp32 = DeterministicConvForward<float>;
using DeterministicConvFwdNchwBfp16 = DeterministicConvForward<bfloat16>;
using DeterministicConvFwdNchwFp16 = DeterministicConvForward<half>;

using DeterministicConvFwdNhwcFp32 = DeterministicConvForward<float>;
using DeterministicConvFwdNhwcBfp16 = DeterministicConvForward<bfloat16>;
using DeterministicConvFwdNhwcFp16 = DeterministicConvForward<half>;

using DeterministicConvFwd5dNcdhwFp32 = DeterministicConvForward<float>;
using DeterministicConvFwd5dNcdhwBfp16 = DeterministicConvForward<bfloat16>;
using DeterministicConvFwd5dNcdhwFp16 = DeterministicConvForward<half>;

using DeterministicConvFwd5dNdhwcFp32 = DeterministicConvForward<float>;
using DeterministicConvFwd5dNdhwcBfp16 = DeterministicConvForward<bfloat16>;
using DeterministicConvFwd5dNdhwcFp16 = DeterministicConvForward<half>;

// ============================================================================
// Convolution Backward Data (Dgrad) Determinism Test
// ============================================================================

template <typename DataType>
class DeterministicConvDgrad : public DeterministicTestBase<ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();

        const ConvTestCase& testCase = DeterministicTestBase<ConvTestCase>::GetParam();

        Graph graphObj;
        graphObj.set_name("DeterministicConvDgradTest");
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto dyTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder)));
        auto wTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder)));

        ConvDgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dxTensorAttr = graphObj.conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);
        dxTensorAttr->set_output(true);
        dxTensorAttr->set_dim(testCase.xDims);
        dxTensorAttr->set_stride(generateStrides(testCase.xDims, layout.strideOrder));

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        GraphTensorBundle bundle1;
        GraphTensorBundle bundle2;
        populateBundleFromGraph(graphObj, bundle1);
        populateBundleFromGraph(graphObj, bundle2);
        randomizeBundle(bundle1, testCase.seed);
        randomizeBundle(bundle2, testCase.seed);

        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack1 = bundle1.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack1, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack2 = bundle2.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack2, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        assertBundleOutputsMatch(bundle1, bundle2, dxTensorAttr->get_uid(), _stream);
    }
};

using DeterministicConvDgradNchwFp32 = DeterministicConvDgrad<float>;
using DeterministicConvDgradNchwBfp16 = DeterministicConvDgrad<bfloat16>;
using DeterministicConvDgradNchwFp16 = DeterministicConvDgrad<half>;

using DeterministicConvDgradNhwcFp32 = DeterministicConvDgrad<float>;
using DeterministicConvDgradNhwcBfp16 = DeterministicConvDgrad<bfloat16>;
using DeterministicConvDgradNhwcFp16 = DeterministicConvDgrad<half>;

using DeterministicConvDgrad5dNcdhwFp32 = DeterministicConvDgrad<float>;
using DeterministicConvDgrad5dNcdhwBfp16 = DeterministicConvDgrad<bfloat16>;
using DeterministicConvDgrad5dNcdhwFp16 = DeterministicConvDgrad<half>;

using DeterministicConvDgrad5dNdhwcFp32 = DeterministicConvDgrad<float>;
using DeterministicConvDgrad5dNdhwcBfp16 = DeterministicConvDgrad<bfloat16>;
using DeterministicConvDgrad5dNdhwcFp16 = DeterministicConvDgrad<half>;

// ============================================================================
// Convolution Backward Weights (Wgrad) Determinism Test
// ============================================================================

template <typename DataType>
class DeterministicConvWgrad : public DeterministicTestBase<ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();

        const ConvTestCase& testCase = DeterministicTestBase<ConvTestCase>::GetParam();

        Graph graphObj;
        graphObj.set_name("DeterministicConvWgradTest");
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder)));
        auto dyTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder)));

        ConvWgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dwTensorAttr = graphObj.conv_wgrad(dyTensorAttr, xTensorAttr, convAttrs);
        dwTensorAttr->set_output(true);
        dwTensorAttr->set_dim(testCase.wDims);
        dwTensorAttr->set_stride(generateStrides(testCase.wDims, layout.strideOrder));

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        GraphTensorBundle bundle1;
        GraphTensorBundle bundle2;
        populateBundleFromGraph(graphObj, bundle1);
        populateBundleFromGraph(graphObj, bundle2);
        randomizeBundle(bundle1, testCase.seed);
        randomizeBundle(bundle2, testCase.seed);

        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack1 = bundle1.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack1, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack2 = bundle2.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack2, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        assertBundleOutputsMatch(bundle1, bundle2, dwTensorAttr->get_uid(), _stream);
    }
};

using DeterministicConvWgradNchwFp32 = DeterministicConvWgrad<float>;
using DeterministicConvWgradNchwBfp16 = DeterministicConvWgrad<bfloat16>;
using DeterministicConvWgradNchwFp16 = DeterministicConvWgrad<half>;

using DeterministicConvWgradNhwcFp32 = DeterministicConvWgrad<float>;
using DeterministicConvWgradNhwcBfp16 = DeterministicConvWgrad<bfloat16>;
using DeterministicConvWgradNhwcFp16 = DeterministicConvWgrad<half>;

using DeterministicConvWgrad5dNcdhwFp32 = DeterministicConvWgrad<float>;
using DeterministicConvWgrad5dNcdhwBfp16 = DeterministicConvWgrad<bfloat16>;
using DeterministicConvWgrad5dNcdhwFp16 = DeterministicConvWgrad<half>;

using DeterministicConvWgrad5dNdhwcFp32 = DeterministicConvWgrad<float>;
using DeterministicConvWgrad5dNdhwcBfp16 = DeterministicConvWgrad<bfloat16>;
using DeterministicConvWgrad5dNdhwcFp16 = DeterministicConvWgrad<half>;

// ============================================================================
// Fused Convolution Forward + Bias + Activation Determinism Test
// ============================================================================

template <typename DataType>
class DeterministicConvFwdBiasActiv : public DeterministicTestBase<FusedConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();

        const auto& [convTestCase, doBias, activTestCase]
            = DeterministicTestBase<FusedConvTestCase>::GetParam();

        Graph graphObj;
        graphObj.set_name(doBias ? "DeterministicConvFwdBiasActivTest"
                                 : "DeterministicConvFwdActivTest");
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "x", convTestCase.xDims, generateStrides(convTestCase.xDims, layout.strideOrder)));
        auto wTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "w", convTestCase.wDims, generateStrides(convTestCase.wDims, layout.strideOrder)));

        ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(convTestCase.convPrePadding);
        convAttrs.set_post_padding(convTestCase.convPostPadding);
        convAttrs.set_stride(convTestCase.convStride);
        convAttrs.set_dilation(convTestCase.convDilation);

        auto yConvTensorAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        std::shared_ptr<TensorAttributes> biasTensorAttr;
        std::shared_ptr<TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(convTestCase.yDims);
            biasTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
                "bias", biasDims, generateStrides(biasDims, layout.strideOrder)));

            PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(dataType);

            yBiasTensorAttr = graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
        }

        PointwiseAttributes activAttrs;
        activAttrs.set_mode(static_cast<PointwiseMode>(activTestCase.mode));
        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }

        auto yTensorAttr
            = graphObj.pointwise(doBias ? yBiasTensorAttr : yConvTensorAttr, activAttrs);
        yTensorAttr->set_output(true);

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        GraphTensorBundle bundle1;
        GraphTensorBundle bundle2;
        populateBundleFromGraph(graphObj, bundle1);
        populateBundleFromGraph(graphObj, bundle2);
        randomizeBundle(bundle1, convTestCase.seed);
        randomizeBundle(bundle2, convTestCase.seed);

        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack1 = bundle1.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack1, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack2 = bundle2.toDeviceVariantPack();
        result = graphObj.execute(_handle, variantPack2, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        assertBundleOutputsMatch(bundle1, bundle2, yTensorAttr->get_uid(), _stream);
    }
};

using DeterministicConvFwdBiasActivNchwFp32 = DeterministicConvFwdBiasActiv<float>;
using DeterministicConvFwdBiasActivNchwBfp16 = DeterministicConvFwdBiasActiv<bfloat16>;
using DeterministicConvFwdBiasActivNchwFp16 = DeterministicConvFwdBiasActiv<half>;

// ============================================================================
// Batchnorm No-Solver Test
// Verifies that deterministic engine does not support batchnorm operations
// ============================================================================

class DeterministicBnNoSolver : public DeterministicTestBase<BatchnormTestCase>
{
protected:
    void runNoSolverTest()
    {
        const BatchnormTestCase& testCase = GetParam();
        auto derivedDims = getDerivedShape(testCase.dims);

        Graph graphObj;
        graphObj.set_name("DeterministicBnNoSolverTest");
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(dataType)
            .set_io_data_type(dataType);

        auto xTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("X", testCase.dims, generateStrides(testCase.dims)));
        auto meanTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("mean", dataType, derivedDims, generateStrides(derivedDims)));
        auto invVarianceTensorAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "inv_variance", dataType, derivedDims, generateStrides(derivedDims)));
        auto scaleTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("scale", dataType, derivedDims, generateStrides(derivedDims)));
        auto biasTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("bias", dataType, derivedDims, generateStrides(derivedDims)));

        BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);
        yTensorAttr->set_output(true);

        auto result = graphObj.validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graphObj.get_ranked_engine_ids(rankedEngineIds);

        bool deterministicEngineFound = false;
        for(auto engineId : rankedEngineIds)
        {
            if(engineId == MIOPEN_ENGINE_DETERMINISTIC_ID)
            {
                deterministicEngineFound = true;
                break;
            }
        }

        EXPECT_FALSE(deterministicEngineFound)
            << "Deterministic engine should not support batchnorm operations";
    }
};

} // namespace

// ============================================================================
// Convolution Forward Determinism Tests
// ============================================================================

TEST_P(DeterministicConvFwdNchwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvFwdNchwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvFwdNchwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNchwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNchwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// NHWC Layout Tests
TEST_P(DeterministicConvFwdNhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvFwdNhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvFwdNhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdNhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// 5D NCDHW Layout Tests
TEST_P(DeterministicConvFwd5dNcdhwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvFwd5dNcdhwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvFwd5dNcdhwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNcdhwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNcdhwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNcdhwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// 5D NDHWC Layout Tests
TEST_P(DeterministicConvFwd5dNdhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvFwd5dNdhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvFwd5dNdhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNdhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNdhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwd5dNdhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// ============================================================================
// Convolution Backward Data (Dgrad) Determinism Tests
// ============================================================================

TEST_P(DeterministicConvDgradNchwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvDgradNchwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvDgradNchwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNchwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNchwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// NHWC Layout Tests
TEST_P(DeterministicConvDgradNhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvDgradNhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvDgradNhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgradNhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// 5D NCDHW Layout Tests
TEST_P(DeterministicConvDgrad5dNcdhwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvDgrad5dNcdhwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvDgrad5dNcdhwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNcdhwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNcdhwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNcdhwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// 5D NDHWC Layout Tests
TEST_P(DeterministicConvDgrad5dNdhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvDgrad5dNdhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvDgrad5dNdhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNdhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNdhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvDgrad5dNdhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// ============================================================================
// Convolution Backward Weights (Wgrad) Determinism Tests
// ============================================================================

TEST_P(DeterministicConvWgradNchwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvWgradNchwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvWgradNchwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNchwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNchwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// NHWC Layout Tests
TEST_P(DeterministicConvWgradNhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvWgradNhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

TEST_P(DeterministicConvWgradNhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgradNhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

// 5D NCDHW Layout Tests
TEST_P(DeterministicConvWgrad5dNcdhwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvWgrad5dNcdhwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

TEST_P(DeterministicConvWgrad5dNcdhwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNcdhwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNcdhwBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNcdhwFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// 5D NDHWC Layout Tests
TEST_P(DeterministicConvWgrad5dNdhwcFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvWgrad5dNdhwcBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

TEST_P(DeterministicConvWgrad5dNdhwcFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNdhwcFp32,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNdhwcBfp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvWgrad5dNdhwcFp16,
                         testing::ValuesIn(getDeterministicConvTestCases5D()));

// ============================================================================
// Fused Convolution Forward + Bias + Activation Determinism Tests
// ============================================================================

TEST_P(DeterministicConvFwdBiasActivNchwFp32, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvFwdBiasActivNchwBfp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

TEST_P(DeterministicConvFwdBiasActivNchwFp16, Determinism)
{
    runDeterminismTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdBiasActivNchwFp32,
                         testing::ValuesIn(getDeterministicFusedConvTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdBiasActivNchwBfp16,
                         testing::ValuesIn(getDeterministicFusedConvTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicConvFwdBiasActivNchwFp16,
                         testing::ValuesIn(getDeterministicFusedConvTestCases()));

// ============================================================================
// Batchnorm No-Solver Tests
// ============================================================================

TEST_P(DeterministicBnNoSolver, NoSolverAvailable)
{
    runNoSolverTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         DeterministicBnNoSolver,
                         testing::ValuesIn(getDeterministicBnTestCases()));
