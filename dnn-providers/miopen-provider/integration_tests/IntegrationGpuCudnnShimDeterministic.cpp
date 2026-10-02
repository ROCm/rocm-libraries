// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cuDNN-shim deselect_numeric_notes({NONDETERMINISTIC}) must select the deterministic
// MIOpen engine. That engine's determinism is covered by IntegrationGpuDeterministic.cpp.

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ConvolutionCommon.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace test_conv_common;

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// ============================================================================
// Base Test Fixture for cuDNN-Shim Deterministic Tests
// Manages the cuDNN-shim handle lifecycle
// ============================================================================

class CudnnShimDeterministicTestBase : public ::testing::TestWithParam<ConvTestCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
    }

    static void initGraph(fe::graph::Graph& graph)
    {
        graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
    }

    static std::shared_ptr<fe::graph::Tensor_attributes> makeTensor(
        fe::graph::Graph& graph, const std::string& name, const std::vector<int64_t>& dims)
    {
        return graph.tensor(fe::graph::Tensor_attributes()
                                .set_name(name)
                                .set_dim(dims)
                                .set_stride(generateStrides(dims))
                                .set_data_type(fe::DataType_t::FLOAT));
    }

    template <typename Attributes>
    static Attributes makeConvAttributes(const ConvTestCase& testCase)
    {
        Attributes attributes;
        attributes.set_pre_padding(testCase.convPrePadding);
        attributes.set_post_padding(testCase.convPostPadding);
        attributes.set_stride(testCase.convStride);
        attributes.set_dilation(testCase.convDilation);
        return attributes;
    }

    void expectDeterministicEngineSelected(fe::graph::Graph& graph)
    {
        graph.deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});

        auto err = graph.validate();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.build_operation_graph(_handle);
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.create_execution_plans({fe::HeurMode_t::A});
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.check_support(_handle);
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.build_plans(_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
        ASSERT_TRUE(err.is_good()) << err.get_message();

        std::string planName;
        ASSERT_TRUE(graph.get_plan_name(planName).is_good());
        EXPECT_EQ(planName, MIOPEN_ENGINE_DETERMINISTIC_NAME);
    }

    cudnnHandle_t _handle = nullptr;
};

// ============================================================================
// Convolution Forward
// ============================================================================

class CudnnShimDeterministicConvForward : public CudnnShimDeterministicTestBase
{
protected:
    void runEngineSelectionTest()
    {
        const ConvTestCase& testCase = GetParam();

        fe::graph::Graph graph;
        initGraph(graph);
        auto x = makeTensor(graph, "x", testCase.xDims);
        auto w = makeTensor(graph, "w", testCase.wDims);
        auto y = graph.conv_fprop(
            x, w, makeConvAttributes<fe::graph::Conv_fprop_attributes>(testCase));
        y->set_output(true);

        expectDeterministicEngineSelected(graph);
    }
};

// ============================================================================
// Convolution Backward Data (Dgrad)
// ============================================================================

class CudnnShimDeterministicConvDgrad : public CudnnShimDeterministicTestBase
{
protected:
    void runEngineSelectionTest()
    {
        const ConvTestCase& testCase = GetParam();

        fe::graph::Graph graph;
        initGraph(graph);
        auto dy = makeTensor(graph, "dy", testCase.yDims);
        auto w = makeTensor(graph, "w", testCase.wDims);
        auto dx = graph.conv_dgrad(
            dy, w, makeConvAttributes<fe::graph::Conv_dgrad_attributes>(testCase));
        dx->set_output(true).set_dim(testCase.xDims).set_stride(generateStrides(testCase.xDims));

        expectDeterministicEngineSelected(graph);
    }
};

// ============================================================================
// Convolution Backward Weights (Wgrad)
// ============================================================================

class CudnnShimDeterministicConvWgrad : public CudnnShimDeterministicTestBase
{
protected:
    void runEngineSelectionTest()
    {
        const ConvTestCase& testCase = GetParam();

        fe::graph::Graph graph;
        initGraph(graph);
        auto dy = makeTensor(graph, "dy", testCase.yDims);
        auto x = makeTensor(graph, "x", testCase.xDims);
        auto dw = graph.conv_wgrad(
            dy, x, makeConvAttributes<fe::graph::Conv_wgrad_attributes>(testCase));
        dw->set_output(true).set_dim(testCase.wDims).set_stride(generateStrides(testCase.wDims));

        expectDeterministicEngineSelected(graph);
    }
};

using IntegrationGpuCudnnShimDeterministicConvFwdNchwFp32 = CudnnShimDeterministicConvForward;
using IntegrationGpuCudnnShimDeterministicConvDgradNchwFp32 = CudnnShimDeterministicConvDgrad;
using IntegrationGpuCudnnShimDeterministicConvWgradNchwFp32 = CudnnShimDeterministicConvWgrad;

} // namespace

TEST_P(IntegrationGpuCudnnShimDeterministicConvFwdNchwFp32, SelectsDeterministicEngine)
{
    runEngineSelectionTest();
}

TEST_P(IntegrationGpuCudnnShimDeterministicConvDgradNchwFp32, SelectsDeterministicEngine)
{
    runEngineSelectionTest();
}

TEST_P(IntegrationGpuCudnnShimDeterministicConvWgradNchwFp32, SelectsDeterministicEngine)
{
    runEngineSelectionTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuCudnnShimDeterministicConvFwdNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuCudnnShimDeterministicConvDgradNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuCudnnShimDeterministicConvWgradNchwFp32,
                         testing::ValuesIn(getDeterministicConvTestCases4D()));
