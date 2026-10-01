// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cuDNN-shim deselect_numeric_notes({NONDETERMINISTIC}) must land on the
// deterministic MIOpen engine and reproduce bit-identical results.

#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_test_sdk::utilities::createTensorFromAttribute;
using hipdnn_test_sdk::utilities::GraphTensorBundle;

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;
constexpr int64_t DY_UID = 4;
constexpr int64_t DW_UID = 5;

// NCHW, 3x3 filter, padding 1, stride 1: x, y and dy share a shape.
struct ConvShape
{
    std::vector<int64_t> x{2, 16, 16, 16};
    std::vector<int64_t> w{16, 16, 3, 3};
};

struct ConvGraphTensors
{
    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> inputs;
    std::shared_ptr<fe::graph::Tensor_attributes> output;
};

fe::graph::Tensor_attributes& describeTensor(fe::graph::Tensor_attributes& tensor,
                                             const std::string& name,
                                             const std::vector<int64_t>& dims,
                                             int64_t uid)
{
    return tensor.set_name(name)
        .set_dim(dims)
        .set_stride(generateStrides(dims))
        .set_uid(uid)
        .set_data_type(fe::DataType_t::FLOAT);
}

std::shared_ptr<fe::graph::Tensor_attributes> makeInput(fe::graph::Graph& graph,
                                                        const std::string& name,
                                                        const std::vector<int64_t>& dims,
                                                        int64_t uid)
{
    fe::graph::Tensor_attributes attributes;
    return graph.tensor(describeTensor(attributes, name, dims, uid));
}

ConvGraphTensors buildFprop(fe::graph::Graph& graph)
{
    const ConvShape shape;
    auto x = makeInput(graph, "x", shape.x, X_UID);
    auto w = makeInput(graph, "w", shape.w, W_UID);
    auto y = graph.conv_fprop(
        x,
        w,
        fe::graph::Conv_fprop_attributes{}.set_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    describeTensor(y->set_output(true), "y", shape.x, Y_UID);
    return {{x, w}, y};
}

ConvGraphTensors buildWgrad(fe::graph::Graph& graph)
{
    const ConvShape shape;
    auto dy = makeInput(graph, "dy", shape.x, DY_UID);
    auto x = makeInput(graph, "x", shape.x, X_UID);
    auto dw = graph.conv_wgrad(
        dy,
        x,
        fe::graph::Conv_wgrad_attributes{}.set_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    describeTensor(dw->set_output(true), "dw", shape.w, DW_UID);
    return {{dy, x}, dw};
}

struct ConvCase
{
    const char* name;
    ConvGraphTensors (*build)(fe::graph::Graph&);
};

constexpr std::array<ConvCase, 2> CONV_CASES{{
    {"Fprop", &buildFprop},
    {"Wgrad", &buildWgrad},
}};

// Every tensor, output included, is seeded in a fixed order so two bundles built
// from the same tensors start from identical device contents.
GraphTensorBundle makeBundle(const ConvGraphTensors& tensors)
{
    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> ordered = tensors.inputs;
    ordered.push_back(tensors.output);

    GraphTensorBundle bundle;
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();
    for(const auto& tensor : ordered)
    {
        bundle.addTensor(*tensor, createTensorFromAttribute(*tensor));
        bundle.randomizeTensor(tensor->get_uid(), -1.0f, 1.0f, seed++);
    }
    return bundle;
}

class IntegrationGpuCudnnShimDeterministic : public ::testing::TestWithParam<ConvCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        SKIP_IF_WINDOWS();
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(cudnnSetStream(_handle, _stream), CUDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    cudnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
};

TEST_P(IntegrationGpuCudnnShimDeterministic, DeselectNondeterministicRunsBitExact)
{
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const ConvGraphTensors tensors = GetParam().build(graph);
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
    EXPECT_EQ(planName, hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_NAME);

    int64_t workspaceSize = 0;
    ASSERT_TRUE(graph.get_workspace_size(workspaceSize).is_good());
    const Workspace workspace(static_cast<size_t>(workspaceSize));

    std::array<GraphTensorBundle, 2> bundles{makeBundle(tensors), makeBundle(tensors)};
    for(auto& bundle : bundles)
    {
        auto variantPack = bundle.toDeviceVariantPack();
        err = graph.execute(_handle, variantPack, workspace.get());
        ASSERT_TRUE(err.is_good()) << err.get_message();
    }
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

    const int64_t outputUid = tensors.output->get_uid();
    auto& first = bundles[0].tensors.at(outputUid);
    auto& second = bundles[1].tensors.at(outputUid);
    first->markDeviceModified();
    second->markDeviceModified();
    ASSERT_EQ(first->elementCount(), second->elementCount());
    EXPECT_EQ(std::memcmp(first->rawHostData(),
                          second->rawHostData(),
                          first->elementCount() * first->elementSize()),
              0)
        << "Outputs are not bit-exact";
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuCudnnShimDeterministic,
                         ::testing::ValuesIn(CONV_CASES),
                         [](const ::testing::TestParamInfo<ConvCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace
