// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
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

struct UnsupportedBnDtypeCase
{
    DataType io;
    DataType scale;
    DataType bias;
    DataType stats;
    const char* name;
};

std::vector<int64_t> makeDims()
{
    return {1, 4, 8, 8};
}

Graph makeGraph(const UnsupportedBnDtypeCase& tc, const TensorLayout& layout = TensorLayout::NCHW)
{
    Graph g;
    g.set_name("IntegrationGpuBatchnormUnsupportedDataTypes");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(tc.io);

    const auto dims = makeDims();
    const auto cDims = getDerivedShape(dims);

    auto xAttr = makeTensorAttributes("X", tc.io, dims, generateStrides(dims, layout.strideOrder));
    auto x = std::make_shared<TensorAttributes>(std::move(xAttr));

    auto meanAttr = makeTensorAttributes("mean", tc.stats, cDims, generateStrides(cDims));
    auto invVarAttr = makeTensorAttributes("inv_variance", tc.stats, cDims, generateStrides(cDims));
    auto mean = std::make_shared<TensorAttributes>(std::move(meanAttr));
    auto invVar = std::make_shared<TensorAttributes>(std::move(invVarAttr));

    auto scaleAttr = makeTensorAttributes("scale", tc.scale, cDims, generateStrides(cDims));
    auto biasAttr = makeTensorAttributes("bias", tc.bias, cDims, generateStrides(cDims));
    auto scale = std::make_shared<TensorAttributes>(std::move(scaleAttr));
    auto bias = std::make_shared<TensorAttributes>(std::move(biasAttr));

    const BatchnormInferenceAttributes bn;
    auto y = g.batchnorm_inference(x, mean, invVar, scale, bias, bn);
    y->set_output(true);

    return g;
}

class IntegrationGpuBatchnormUnsupportedDataTypes
    : public ::testing::TestWithParam<UnsupportedBnDtypeCase>
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

TEST_P(IntegrationGpuBatchnormUnsupportedDataTypes, RejectsUnsupportedDataTypes)
{
    const auto& tc = GetParam();
    auto g = makeGraph(tc, TensorLayout::NCHW);

    auto result = g.build(_handle);

    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED) << "err_msg: " << result.err_msg;

    EXPECT_FALSE(result.err_msg.empty()) << "err_msg is empty";
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormUnsupportedDataTypes,
    ::testing::Values(
        UnsupportedBnDtypeCase{
            DataType::UINT8, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, "Uint8IO"},
        UnsupportedBnDtypeCase{
            DataType::FLOAT, DataType::HALF, DataType::HALF, DataType::FLOAT, "HalfScaleBias"},
        UnsupportedBnDtypeCase{DataType::FLOAT,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               "MismatchedScaleHalfBiasFloat"}),
    [](const ::testing::TestParamInfo<UnsupportedBnDtypeCase>& info) {
        return std::string(info.param.name);
    });
