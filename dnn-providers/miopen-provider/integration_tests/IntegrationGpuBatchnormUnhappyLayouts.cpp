// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

TensorLayout makeLayout(std::initializer_list<int64_t> order)
{
    TensorLayout t = TensorLayout::NCHW;
    t.strideOrder = order;
    return t;
}

struct UnhappyBnLayoutCase
{
    std::vector<int64_t> dims;
    TensorLayout layoutX;
    TensorLayout layoutParam;
    const char* name;
};

std::vector<int64_t> make3d()
{
    return {2, 4, 8};
}
std::vector<int64_t> make4d()
{
    return {2, 4, 8, 8};
}

Graph makeGraph(const UnhappyBnLayoutCase& tc)
{
    Graph g;
    g.set_name("IntegrationGpuBatchnormUnhappyLayouts");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    const auto& dims = tc.dims;
    const auto cDims = getDerivedShape(dims);

    auto xAttr = makeTensorAttributes(
        "X", DataType::FLOAT, dims, generateStrides(dims, tc.layoutX.strideOrder));
    auto x = std::make_shared<TensorAttributes>(std::move(xAttr));

    auto meanAttr = makeTensorAttributes(
        "mean", DataType::FLOAT, cDims, generateStrides(cDims, tc.layoutParam.strideOrder));
    auto invVarAttr = makeTensorAttributes(
        "inv_variance", DataType::FLOAT, cDims, generateStrides(cDims, tc.layoutParam.strideOrder));
    auto scaleAttr = makeTensorAttributes(
        "scale", DataType::FLOAT, cDims, generateStrides(cDims, tc.layoutParam.strideOrder));
    auto biasAttr = makeTensorAttributes(
        "bias", DataType::FLOAT, cDims, generateStrides(cDims, tc.layoutParam.strideOrder));

    auto mean = std::make_shared<TensorAttributes>(std::move(meanAttr));
    auto invVar = std::make_shared<TensorAttributes>(std::move(invVarAttr));
    auto scale = std::make_shared<TensorAttributes>(std::move(scaleAttr));
    auto bias = std::make_shared<TensorAttributes>(std::move(biasAttr));

    const BatchnormInferenceAttributes bn;
    auto y = g.batchnorm_inference(x, mean, invVar, scale, bias, bn);
    y->set_output(true);

    return g;
}

class IntegrationGpuBatchnormUnhappyLayouts : public ::testing::TestWithParam<UnhappyBnLayoutCase>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
};
} // namespace

TEST_P(IntegrationGpuBatchnormUnhappyLayouts, RejectsUnsupportedLayouts)
{
    const auto& tc = GetParam();
    auto g = makeGraph(tc);

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED) << "err_msg: " << result.err_msg;

    EXPECT_FALSE(result.err_msg.empty()) << "err_msg is empty";
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormUnhappyLayouts,
    ::testing::Values(
        // 3D batchnorm is supported, but only with a standard NCH
        // layout (strideOrder {2, 1, 0}). This case uses a non-standard
        // strideOrder {0, 1, 2}, which no engine accepts, so the graph
        // is rejected.
        UnhappyBnLayoutCase{
            make3d(), makeLayout({0, 1, 2}), makeLayout({0, 1, 2}), "ThreeDimInput"},
        UnhappyBnLayoutCase{
            make4d(), makeLayout({1, 0, 2, 3}), makeLayout({0, 2, 3, 1}), "MixedLayouts"},
        UnhappyBnLayoutCase{make4d(),
                            makeLayout({0, 1, 2, 3}),
                            makeLayout({1, 0, 2, 3}),
                            "NonStandardStrideOrder"}),
    [](const ::testing::TestParamInfo<UnhappyBnLayoutCase>& info) {
        return std::string(info.param.name);
    });
