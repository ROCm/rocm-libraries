// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

constexpr double EPSILON = BATCHNORM_DEFAULT_EPSILON;

// Test fixture
class IntegrationGpuBatchnormSavedStatsNotRequested : public ::testing::Test
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

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

} // anonymous namespace

// With only y marked as an output, the frontend leaves the saved mean and inv_variance virtual:
// the plan does not write them, and y must still match the CPU reference.
TEST_F(IntegrationGpuBatchnormSavedStatsNotRequested, TrainingWithOnlyYAsOutputMatchesReference)
{
    const std::vector<int64_t> dims = {4, 3, 8, 8};
    const auto cDims = getDerivedShape(dims);

    Graph g;
    g.set_name("IntegrationGpuBatchnormSavedStatsNotRequested");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", DataType::FLOAT, dims, generateStrides(dims)));
    auto scale = std::make_shared<TensorAttributes>(
        makeTensorAttributes("scale", DataType::FLOAT, cDims, generateStrides(cDims)));
    auto bias = std::make_shared<TensorAttributes>(
        makeTensorAttributes("bias", DataType::FLOAT, cDims, generateStrides(cDims)));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1}).set_stride({1}).set_data_type(DataType::DOUBLE);
    epsilon->set_compile_time_constant(EPSILON);

    BatchnormAttributes bn;
    bn.set_epsilon(epsilon);

    const auto bnOutputs = g.batchnorm(x, scale, bias, bn);
    const auto& y = bnOutputs[0];
    y->set_output(true);

    const auto buildResult = g.build(_handle);
    ASSERT_TRUE(buildResult.is_good()) << "err_msg: " << buildResult.err_msg;

    Tensor<float> xTensor(dims);
    Tensor<float> scaleTensor(cDims);
    Tensor<float> biasTensor(cDims);
    Tensor<float> yTensor(dims);
    Tensor<float> yRefTensor(dims);

    xTensor.fillWithRandomValues(-1.0f, 1.0f);
    scaleTensor.fillWithRandomValues(-2.0f, 2.0f);
    biasTensor.fillWithRandomValues(-2.0f, 2.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

    const auto executeResult = g.execute(_handle, variantPack, nullptr);
    ASSERT_TRUE(executeResult.is_good()) << "err_msg: " << executeResult.err_msg;
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
    yTensor.memory().markDeviceModified();

    // Momentum and the saved and running statistics are not part of this graph.
    hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm::fwdTraining<float, float, float, float>(
        xTensor,
        scaleTensor,
        biasTensor,
        yRefTensor,
        EPSILON,
        0.0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr);

    const auto tolerance = hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<float>();
    const auto yValidator
        = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float>(tolerance, tolerance);

    EXPECT_TRUE(hipdnn_test_sdk::utilities::validateAndReport<float>(
        std::cout, "y", yValidator, yRefTensor, yTensor, tolerance, tolerance));
}
