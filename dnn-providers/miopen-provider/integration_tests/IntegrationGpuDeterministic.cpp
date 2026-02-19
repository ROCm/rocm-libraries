// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "../tests/common/ConvolutionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_plugin::test_utilities;
using namespace test_conv_common;
using namespace test_bn_common;

namespace
{

// ============================================================================
// Deterministic Convolution Smoke Test Cases
// Uses a subset of the standard convolution test cases for smoke testing
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
// Convolution Forward Determinism Test
// Verifies that running the same convolution twice produces identical results
// ============================================================================

template <typename DataType>
class DeterministicConvForward : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();
        SKIP_IF_NO_DEVICES();

        const ConvTestCase& testCase = this->GetParam();

        // First execution
        auto result1 = executeConvolution(testCase, layout);

        // Second execution with same inputs
        auto result2 = executeConvolution(testCase, layout);

        // Compare results - should be bit-exact for deterministic execution
        ASSERT_EQ(result1.size(), result2.size()) << "Output sizes differ";

        for(size_t i = 0; i < result1.size(); ++i)
        {
            ASSERT_EQ(result1[i], result2[i])
                << "Mismatch at index " << i << ": " << result1[i] << " != " << result2[i];
        }
    }

private:
    std::vector<float> executeConvolution(const ConvTestCase& testCase, const TensorLayout& layout)
    {
        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("DeterministicConvForwardTest");

        // Set preferred engine to deterministic
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yAttr->set_output(true);

        // Build the graph
        hipdnnHandle_t handle;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        hipStream_t stream;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

        auto result = graphObj.build(handle);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        // Allocate and initialize tensors
        auto xSize = getTensorSize(testCase.xDims);
        auto wSize = getTensorSize(testCase.wDims);
        auto ySize = getTensorSize(yAttr->get_dim());

        std::vector<DataType> xHost(xSize);
        std::vector<DataType> wHost(wSize);
        std::vector<float> yHost(ySize);

        // Initialize with deterministic values based on seed
        std::mt19937 gen(testCase.seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(auto& val : xHost)
        {
            val = static_cast<DataType>(dist(gen));
        }
        for(auto& val : wHost)
        {
            val = static_cast<DataType>(dist(gen));
        }

        // Allocate device memory
        DataType* xDev = nullptr;
        DataType* wDev = nullptr;
        DataType* yDev = nullptr;

        EXPECT_EQ(hipMalloc(&xDev, xSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&wDev, wSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&yDev, ySize * sizeof(DataType)), hipSuccess);

        EXPECT_EQ(hipMemcpy(xDev, xHost.data(), xSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);
        EXPECT_EQ(hipMemcpy(wDev, wHost.data(), wSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);

        // Create variant pack
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr->get_uid()] = xDev;
        variantPack[wTensorAttr->get_uid()] = wDev;
        variantPack[yAttr->get_uid()] = yDev;

        // Get workspace
        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        // Execute
        result = graphObj.execute(handle, variantPack, workspace.get());
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        // Copy results back
        std::vector<DataType> yDevHost(ySize);
        EXPECT_EQ(hipMemcpy(yDevHost.data(), yDev, ySize * sizeof(DataType), hipMemcpyDeviceToHost),
                  hipSuccess);

        // Convert to float for comparison
        for(size_t i = 0; i < ySize; ++i)
        {
            yHost[i] = static_cast<float>(yDevHost[i]);
        }

        // Cleanup
        EXPECT_EQ(hipFree(xDev), hipSuccess);
        EXPECT_EQ(hipFree(wDev), hipSuccess);
        EXPECT_EQ(hipFree(yDev), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

        return yHost;
    }

    static size_t getTensorSize(const std::vector<int64_t>& dims)
    {
        size_t size = 1;
        for(auto dim : dims)
        {
            size *= static_cast<size_t>(dim);
        }
        return size;
    }
};

using DeterministicConvFwdNchwFp32 = DeterministicConvForward<float>;
using DeterministicConvFwdNchwBfp16 = DeterministicConvForward<bfloat16>;
using DeterministicConvFwdNchwFp16 = DeterministicConvForward<half>;

// ============================================================================
// Convolution Backward Data (Dgrad) Determinism Test
// Verifies that running the same conv dgrad twice produces identical results
// ============================================================================

template <typename DataType>
class DeterministicConvDgrad : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();
        SKIP_IF_NO_DEVICES();

        const ConvTestCase& testCase = this->GetParam();

        // First execution
        auto result1 = executeConvDgrad(testCase, layout);

        // Second execution with same inputs
        auto result2 = executeConvDgrad(testCase, layout);

        // Compare results - should be bit-exact for deterministic execution
        ASSERT_EQ(result1.size(), result2.size()) << "Output sizes differ";

        for(size_t i = 0; i < result1.size(); ++i)
        {
            ASSERT_EQ(result1[i], result2[i])
                << "Mismatch at index " << i << ": " << result1[i] << " != " << result2[i];
        }
    }

private:
    std::vector<float> executeConvDgrad(const ConvTestCase& testCase, const TensorLayout& layout)
    {
        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("DeterministicConvDgradTest");

        // Set preferred engine to deterministic
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

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

        // Set these explicitly since grouped convs cannot infer tensor shape
        dxTensorAttr->set_dim(testCase.xDims);
        dxTensorAttr->set_stride(generateStrides(testCase.xDims, layout.strideOrder));

        // Build the graph
        hipdnnHandle_t handle;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        hipStream_t stream;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

        auto result = graphObj.build(handle);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        // Allocate and initialize tensors
        auto dySize = getTensorSize(testCase.yDims);
        auto wSize = getTensorSize(testCase.wDims);
        auto dxSize = getTensorSize(testCase.xDims);

        std::vector<DataType> dyHost(dySize);
        std::vector<DataType> wHost(wSize);
        std::vector<float> dxHost(dxSize);

        // Initialize with deterministic values based on seed
        std::mt19937 gen(testCase.seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(auto& val : dyHost)
        {
            val = static_cast<DataType>(dist(gen));
        }
        for(auto& val : wHost)
        {
            val = static_cast<DataType>(dist(gen));
        }

        // Allocate device memory
        DataType* dyDev = nullptr;
        DataType* wDev = nullptr;
        DataType* dxDev = nullptr;

        EXPECT_EQ(hipMalloc(&dyDev, dySize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&wDev, wSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&dxDev, dxSize * sizeof(DataType)), hipSuccess);

        EXPECT_EQ(hipMemcpy(dyDev, dyHost.data(), dySize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);
        EXPECT_EQ(hipMemcpy(wDev, wHost.data(), wSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);

        // Create variant pack
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[dyTensorAttr->get_uid()] = dyDev;
        variantPack[wTensorAttr->get_uid()] = wDev;
        variantPack[dxTensorAttr->get_uid()] = dxDev;

        // Get workspace
        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        // Execute
        result = graphObj.execute(handle, variantPack, workspace.get());
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        // Copy results back
        std::vector<DataType> dxDevHost(dxSize);
        EXPECT_EQ(
            hipMemcpy(dxDevHost.data(), dxDev, dxSize * sizeof(DataType), hipMemcpyDeviceToHost),
            hipSuccess);

        // Convert to float for comparison
        for(size_t i = 0; i < dxSize; ++i)
        {
            dxHost[i] = static_cast<float>(dxDevHost[i]);
        }

        // Cleanup
        EXPECT_EQ(hipFree(dyDev), hipSuccess);
        EXPECT_EQ(hipFree(wDev), hipSuccess);
        EXPECT_EQ(hipFree(dxDev), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

        return dxHost;
    }

    static size_t getTensorSize(const std::vector<int64_t>& dims)
    {
        size_t size = 1;
        for(auto dim : dims)
        {
            size *= static_cast<size_t>(dim);
        }
        return size;
    }
};

using DeterministicConvDgradNchwFp32 = DeterministicConvDgrad<float>;
using DeterministicConvDgradNchwBfp16 = DeterministicConvDgrad<bfloat16>;
using DeterministicConvDgradNchwFp16 = DeterministicConvDgrad<half>;

// ============================================================================
// Convolution Backward Weights (Wgrad) Determinism Test
// Verifies that running the same conv wgrad twice produces identical results
// ============================================================================

template <typename DataType>
class DeterministicConvWgrad : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();
        SKIP_IF_NO_DEVICES();

        const ConvTestCase& testCase = this->GetParam();

        // First execution
        auto result1 = executeConvWgrad(testCase, layout);

        // Second execution with same inputs
        auto result2 = executeConvWgrad(testCase, layout);

        // Compare results - should be bit-exact for deterministic execution
        ASSERT_EQ(result1.size(), result2.size()) << "Output sizes differ";

        for(size_t i = 0; i < result1.size(); ++i)
        {
            ASSERT_EQ(result1[i], result2[i])
                << "Mismatch at index " << i << ": " << result1[i] << " != " << result2[i];
        }
    }

private:
    std::vector<float> executeConvWgrad(const ConvTestCase& testCase, const TensorLayout& layout)
    {
        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("DeterministicConvWgradTest");

        // Set preferred engine to deterministic
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

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

        // Set these explicitly since grouped convs cannot infer tensor shape
        dwTensorAttr->set_dim(testCase.wDims);
        dwTensorAttr->set_stride(generateStrides(testCase.wDims, layout.strideOrder));

        // Build the graph
        hipdnnHandle_t handle;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        hipStream_t stream;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

        auto result = graphObj.build(handle);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        // Allocate and initialize tensors
        auto xSize = getTensorSize(testCase.xDims);
        auto dySize = getTensorSize(testCase.yDims);
        auto dwSize = getTensorSize(testCase.wDims);

        std::vector<DataType> xHost(xSize);
        std::vector<DataType> dyHost(dySize);
        std::vector<float> dwHost(dwSize);

        // Initialize with deterministic values based on seed
        std::mt19937 gen(testCase.seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(auto& val : xHost)
        {
            val = static_cast<DataType>(dist(gen));
        }
        for(auto& val : dyHost)
        {
            val = static_cast<DataType>(dist(gen));
        }

        // Allocate device memory
        DataType* xDev = nullptr;
        DataType* dyDev = nullptr;
        DataType* dwDev = nullptr;

        EXPECT_EQ(hipMalloc(&xDev, xSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&dyDev, dySize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&dwDev, dwSize * sizeof(DataType)), hipSuccess);

        EXPECT_EQ(hipMemcpy(xDev, xHost.data(), xSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);
        EXPECT_EQ(hipMemcpy(dyDev, dyHost.data(), dySize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);

        // Create variant pack
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr->get_uid()] = xDev;
        variantPack[dyTensorAttr->get_uid()] = dyDev;
        variantPack[dwTensorAttr->get_uid()] = dwDev;

        // Get workspace
        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        // Execute
        result = graphObj.execute(handle, variantPack, workspace.get());
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        // Copy results back
        std::vector<DataType> dwDevHost(dwSize);
        EXPECT_EQ(
            hipMemcpy(dwDevHost.data(), dwDev, dwSize * sizeof(DataType), hipMemcpyDeviceToHost),
            hipSuccess);

        // Convert to float for comparison
        for(size_t i = 0; i < dwSize; ++i)
        {
            dwHost[i] = static_cast<float>(dwDevHost[i]);
        }

        // Cleanup
        EXPECT_EQ(hipFree(xDev), hipSuccess);
        EXPECT_EQ(hipFree(dyDev), hipSuccess);
        EXPECT_EQ(hipFree(dwDev), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

        return dwHost;
    }

    static size_t getTensorSize(const std::vector<int64_t>& dims)
    {
        size_t size = 1;
        for(auto dim : dims)
        {
            size *= static_cast<size_t>(dim);
        }
        return size;
    }
};

using DeterministicConvWgradNchwFp32 = DeterministicConvWgrad<float>;
using DeterministicConvWgradNchwBfp16 = DeterministicConvWgrad<bfloat16>;
using DeterministicConvWgradNchwFp16 = DeterministicConvWgrad<half>;

// ============================================================================
// Fused Convolution Forward + Bias + Activation Determinism Test
// Verifies that running the same fused conv twice produces identical results
// ============================================================================

using FusedConvTestCase = std::tuple<ConvTestCase, bool, test_activation_common::ActivTestCase>;

inline std::vector<FusedConvTestCase> getDeterministicFusedConvTestCases()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    // Use a subset of conv test cases with bias and activation
    std::vector<ConvTestCase> convCases = {
        // Filter 3x3 with padding - common case
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
    };

    // ReLU activation only (supported by MIOpen fusion)
    auto activCases = test_activation_common::createFwdActivationSmokeCases();

    std::vector<FusedConvTestCase> fusedCases;
    for(const auto& convCase : convCases)
    {
        for(const auto& activCase : activCases)
        {
            // With bias
            fusedCases.emplace_back(convCase, true, activCase);
            // Without bias
            fusedCases.emplace_back(convCase, false, activCase);
        }
    }
    return fusedCases;
}

template <typename DataType>
class DeterministicConvFwdBiasActiv
    : public IntegrationGraphVerificationHarness<DataType, FusedConvTestCase>
{
protected:
    void runDeterminismTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        SKIP_IF_WINDOWS();
        SKIP_IF_NO_DEVICES();

        const auto& [convTestCase, doBias, activTestCase] = this->GetParam();

        // First execution
        auto result1 = executeFusedConv(convTestCase, doBias, activTestCase, layout);

        // Second execution with same inputs
        auto result2 = executeFusedConv(convTestCase, doBias, activTestCase, layout);

        // Compare results - should be bit-exact for deterministic execution
        ASSERT_EQ(result1.size(), result2.size()) << "Output sizes differ";

        for(size_t i = 0; i < result1.size(); ++i)
        {
            ASSERT_EQ(result1[i], result2[i])
                << "Mismatch at index " << i << ": " << result1[i] << " != " << result2[i];
        }
    }

private:
    std::vector<float> executeFusedConv(const ConvTestCase& testCase,
                                        bool doBias,
                                        const test_activation_common::ActivTestCase& activTestCase,
                                        const TensorLayout& layout)
    {
        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name(doBias ? "DeterministicConvFwdBiasActivTest"
                                 : "DeterministicConvFwdActivTest");

        // Set preferred engine to deterministic
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yConvTensorAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        std::shared_ptr<graph::TensorAttributes> biasTensorAttr;
        std::shared_ptr<graph::TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(testCase.yDims);

            auto biasAttr = makeTensorAttributes(
                "bias", biasDims, generateStrides(biasDims, layout.strideOrder));
            biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

            graph::PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(dataType);

            yBiasTensorAttr = graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
        }

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
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

        // Build the graph
        hipdnnHandle_t handle;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        hipStream_t stream;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

        auto result = graphObj.build(handle);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        // Allocate and initialize tensors
        auto xSize = getTensorSize(testCase.xDims);
        auto wSize = getTensorSize(testCase.wDims);
        auto ySize = getTensorSize(yTensorAttr->get_dim());

        std::vector<DataType> xHost(xSize);
        std::vector<DataType> wHost(wSize);
        std::vector<float> yHost(ySize);

        // Initialize with deterministic values based on seed
        std::mt19937 gen(testCase.seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for(auto& val : xHost)
        {
            val = static_cast<DataType>(dist(gen));
        }
        for(auto& val : wHost)
        {
            val = static_cast<DataType>(dist(gen));
        }

        // Allocate device memory
        DataType* xDev = nullptr;
        DataType* wDev = nullptr;
        DataType* yDev = nullptr;
        DataType* biasDev = nullptr;

        EXPECT_EQ(hipMalloc(&xDev, xSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&wDev, wSize * sizeof(DataType)), hipSuccess);
        EXPECT_EQ(hipMalloc(&yDev, ySize * sizeof(DataType)), hipSuccess);

        EXPECT_EQ(hipMemcpy(xDev, xHost.data(), xSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);
        EXPECT_EQ(hipMemcpy(wDev, wHost.data(), wSize * sizeof(DataType), hipMemcpyHostToDevice),
                  hipSuccess);

        // Create variant pack
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr->get_uid()] = xDev;
        variantPack[wTensorAttr->get_uid()] = wDev;
        variantPack[yTensorAttr->get_uid()] = yDev;

        std::vector<DataType> biasHost;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(testCase.yDims);
            auto biasSize = getTensorSize(biasDims);
            biasHost.resize(biasSize);

            for(auto& val : biasHost)
            {
                val = static_cast<DataType>(dist(gen));
            }

            EXPECT_EQ(hipMalloc(&biasDev, biasSize * sizeof(DataType)), hipSuccess);
            EXPECT_EQ(
                hipMemcpy(
                    biasDev, biasHost.data(), biasSize * sizeof(DataType), hipMemcpyHostToDevice),
                hipSuccess);

            variantPack[biasTensorAttr->get_uid()] = biasDev;
        }

        // Get workspace
        int64_t workspaceSize;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        // Execute
        result = graphObj.execute(handle, variantPack, workspace.get());
        EXPECT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

        // Copy results back
        std::vector<DataType> yDevHost(ySize);
        EXPECT_EQ(hipMemcpy(yDevHost.data(), yDev, ySize * sizeof(DataType), hipMemcpyDeviceToHost),
                  hipSuccess);

        // Convert to float for comparison
        for(size_t i = 0; i < ySize; ++i)
        {
            yHost[i] = static_cast<float>(yDevHost[i]);
        }

        // Cleanup
        EXPECT_EQ(hipFree(xDev), hipSuccess);
        EXPECT_EQ(hipFree(wDev), hipSuccess);
        EXPECT_EQ(hipFree(yDev), hipSuccess);
        if(doBias)
        {
            EXPECT_EQ(hipFree(biasDev), hipSuccess);
        }
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

        return yHost;
    }

    static size_t getTensorSize(const std::vector<int64_t>& dims)
    {
        size_t size = 1;
        for(auto dim : dims)
        {
            size *= static_cast<size_t>(dim);
        }
        return size;
    }
};

using DeterministicConvFwdBiasActivNchwFp32 = DeterministicConvFwdBiasActiv<float>;
using DeterministicConvFwdBiasActivNchwBfp16 = DeterministicConvFwdBiasActiv<bfloat16>;
using DeterministicConvFwdBiasActivNchwFp16 = DeterministicConvFwdBiasActiv<half>;

// ============================================================================
// Batchnorm No-Solver Test
// Verifies that deterministic engine does not support batchnorm operations
// ============================================================================

class DeterministicBnNoSolver : public ::testing::TestWithParam<BatchnormTestCase>
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

    void runNoSolverTest()
    {
        const BatchnormTestCase& testCase = GetParam();
        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("DeterministicBnNoSolverTest");

        // Set preferred engine to deterministic
        graphObj.set_preferred_engine_id_ext(MIOPEN_ENGINE_DETERMINISTIC_NAME);

        auto dataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(dataType)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes("X", testCase.dims, generateStrides(testCase.dims));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto meanAttr
            = makeTensorAttributes("mean", dataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = makeTensorAttributes(
            "inv_variance", dataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr
            = makeTensorAttributes("scale", dataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = makeTensorAttributes("bias", dataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

        yTensorAttr->set_output(true);

        auto result = graphObj.validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        result = graphObj.build_operation_graph(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        // Get ranked engine IDs - should be empty or not contain deterministic engine for batchnorm
        std::vector<int64_t> rankedEngineIds;
        result = graphObj.get_ranked_engine_ids(rankedEngineIds);

        // The deterministic engine should not be available for batchnorm
        // Either no engines are available, or the build should fail
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

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
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
