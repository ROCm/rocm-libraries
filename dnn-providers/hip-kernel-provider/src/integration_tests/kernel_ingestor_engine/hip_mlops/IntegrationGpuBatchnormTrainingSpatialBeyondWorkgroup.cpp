// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace hip_kernel_provider::batchnorm_training_spatial::test
{

namespace
{

using Fp16 = hipdnn_data_sdk::types::half;

// The engine serving HIP MLops batchnorm training: the kernel ingestor's per-operation engine.
constexpr const char* SERVING_ENGINE_NAME = "hipkernel:BatchnormTraining";
const int64_t SERVING_ENGINE_ID = engineNameToId(SERVING_ENGINE_NAME);

// Batchnorm forward training of NCHW graphs whose spatial size H * W exceeds the 1024
// threads of the workgroup the launch heuristics give variants 0 and 3. Those variants give
// each thread one spatial position and do not loop over the spatial dimension, so positions
// past the workgroup are never read or written. Spatial multiple is not preferred for an
// fp16 x upcast to an fp32 y with N <= 768, nor for fp32 with N <= 2, and the forward
// heuristics used to give variant 1 to a spatial size above 1024 only while a channel held
// fewer than N * H * W = 2^25 values, falling back to variant 0 beyond. These are the
// smallest graphs that reached that fallback with two channels (C = 1 gives tied N and C
// strides); the heuristics now give them variant 1, which loops over the spatial dimension.
//
// The saved mean and inverse variance and y are compared against a double-precision
// reference computed from the same (rounded) x values.
class IntegrationGpuBatchnormTrainingSpatialBeyondWorkgroup : public ::testing::Test
{
protected:
    static constexpr int64_t C = 2;
    static constexpr float EPSILON = 1e-5f;
    static constexpr unsigned int SEED = 3066;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);

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
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    // Deterministic, library-independent uniform value in [lo, hi).
    static float uniform(std::mt19937& rng, float lo, float hi)
    {
        const double unit = static_cast<double>(rng() >> 8) / 16777216.0;
        return static_cast<float>(static_cast<double>(lo) + static_cast<double>(hi - lo) * unit);
    }

    // Trains on an NCHW x of XType (fp16 upcast to the fp32 y, or fp32) of N x C x H x W
    // with saved statistics and no running statistics, on the serving engine, and checks
    // the saved mean, the saved inverse variance and y against the reference.
    template <typename XType>
    void runCase(int64_t n, int64_t h, int64_t w)
    {
        static_assert(std::is_same_v<XType, Fp16> || std::is_same_v<XType, float>);
        const auto xDataType = std::is_same_v<XType, Fp16> ? hipdnn_frontend::DataType::HALF
                                                           : hipdnn_frontend::DataType::FLOAT;
        const int64_t spatial = h * w;
        const std::vector<int64_t> dims = {n, C, h, w};
        const std::vector<int64_t> strides = {C * spatial, spatial, w, 1};
        const std::vector<int64_t> channelDims = {1, C, 1, 1};
        const std::vector<int64_t> channelStrides = {C, 1, 1, 1};

        Tensor<XType> xTensor(dims, strides);
        Tensor<float> scaleTensor(channelDims, channelStrides);
        Tensor<float> biasTensor(channelDims, channelStrides);
        Tensor<float> yTensor(dims, strides);
        Tensor<float> meanTensor(channelDims, channelStrides);
        Tensor<float> invVarianceTensor(channelDims, channelStrides);

        // x is packed NCHW, so a flat index walks it in memory order. Each channel is offset
        // from zero so that a mean over part of it, or over the wrong count, shows.
        xTensor.memory().markHostModified();
        scaleTensor.memory().markHostModified();
        biasTensor.memory().markHostModified();
        XType* x = xTensor.memory().hostData();
        float* scale = scaleTensor.memory().hostData();
        float* bias = biasTensor.memory().hostData();

        std::mt19937 rng(SEED);
        for(int64_t b = 0; b < n; ++b)
        {
            for(int64_t c = 0; c < C; ++c)
            {
                const float offset = 0.5f + static_cast<float>(c);
                XType* slice = x + (b * C + c) * spatial;
                for(int64_t i = 0; i < spatial; ++i)
                {
                    slice[i] = static_cast<XType>(offset + uniform(rng, -1.0f, 1.0f));
                }
            }
        }
        for(int64_t c = 0; c < C; ++c)
        {
            scale[c] = uniform(rng, 0.5f, 1.5f);
            bias[c] = uniform(rng, -0.5f, 0.5f);
        }

        Graph graphObj;
        graphObj.set_name("BatchnormTrainingSpatialBeyondWorkgroup")
            .set_io_data_type(xDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_preferred_engine_id_ext(SERVING_ENGINE_ID);

        auto xAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("x", xDataType, dims, strides));
        auto scaleAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "scale", hipdnn_frontend::DataType::FLOAT, channelDims, channelStrides));
        auto biasAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "bias", hipdnn_frontend::DataType::FLOAT, channelDims, channelStrides));
        auto epsilonAttr = std::make_shared<TensorAttributes>();
        epsilonAttr->set_name("epsilon")
            .set_dim({1})
            .set_stride({1})
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compile_time_constant(EPSILON);

        BatchnormAttributes bnAttrs;
        bnAttrs.set_epsilon(epsilonAttr);

        auto outputs = graphObj.batchnorm(xAttr, scaleAttr, biasAttr, bnAttrs);
        const auto& yAttr = outputs[0];
        const auto& meanAttr = outputs[1];
        const auto& invVarianceAttr = outputs[2];
        yAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(dims)
            .set_stride(strides);
        meanAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(channelDims)
            .set_stride(channelStrides);
        invVarianceAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(channelDims)
            .set_stride(channelStrides);

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The engine preference is soft; the test is about this engine's heuristics and
        // kernel, so a plan served by any other engine must fail it rather than pass
        // vacuously.
        int64_t servingEngineId = 0;
        result = graphObj.get_execution_plan_engine_id(servingEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngineId, SERVING_ENGINE_ID)
            << "batchnorm training was not served by " << SERVING_ENGINE_NAME;

        int64_t workspaceSize = 0;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace workspace(static_cast<size_t>(workspaceSize));

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[scaleAttr->get_uid()] = scaleTensor.memory().deviceData();
        variantPack[biasAttr->get_uid()] = biasTensor.memory().deviceData();
        variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();
        variantPack[meanAttr->get_uid()] = meanTensor.memory().deviceData();
        variantPack[invVarianceAttr->get_uid()] = invVarianceTensor.memory().deviceData();

        result = graphObj.execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        yTensor.memory().markDeviceModified();
        meanTensor.memory().markDeviceModified();
        invVarianceTensor.memory().markDeviceModified();
        const float* yGpu = yTensor.memory().hostData();
        const float* meanGpu = meanTensor.memory().hostData();
        const float* invVarianceGpu = invVarianceTensor.memory().hostData();

        // Reference: per-channel population statistics in double over the x values the
        // kernel reads, then y = scale * (x - mean) * invVariance + bias.
        const auto value = [&](int64_t b, int64_t c, int64_t i) {
            return static_cast<double>(static_cast<float>(x[(b * C + c) * spatial + i]));
        };
        const double count = static_cast<double>(n * spatial);
        const double tolerance = static_cast<double>(
            hipdnn_test_sdk::utilities::batchnorm::getToleranceTraining<float>());
        for(int64_t c = 0; c < C; ++c)
        {
            double sum = 0.0;
            for(int64_t b = 0; b < n; ++b)
            {
                for(int64_t i = 0; i < spatial; ++i)
                {
                    sum += value(b, c, i);
                }
            }
            const double mean = sum / count;
            double sumSq = 0.0;
            for(int64_t b = 0; b < n; ++b)
            {
                for(int64_t i = 0; i < spatial; ++i)
                {
                    const double d = value(b, c, i) - mean;
                    sumSq += d * d;
                }
            }
            const double invVariance
                = 1.0 / std::sqrt(sumSq / count + static_cast<double>(EPSILON));

            EXPECT_NEAR(static_cast<double>(meanGpu[c]), mean, tolerance)
                << "saved mean of channel " << c;
            EXPECT_NEAR(static_cast<double>(invVarianceGpu[c]), invVariance, tolerance)
                << "saved inverse variance of channel " << c;

            // One expectation per channel: report how many of its y values mismatch and the
            // first of them, which tells which spatial positions the kernel missed.
            int64_t mismatches = 0;
            int64_t firstBatch = -1;
            int64_t firstPosition = -1;
            double firstGpu = 0.0;
            double firstReference = 0.0;
            for(int64_t b = 0; b < n; ++b)
            {
                for(int64_t i = 0; i < spatial; ++i)
                {
                    const double reference
                        = static_cast<double>(scale[c]) * (value(b, c, i) - mean) * invVariance
                          + static_cast<double>(bias[c]);
                    const auto gpu = static_cast<double>(yGpu[(b * C + c) * spatial + i]);
                    // A NaN fails the comparison and so counts as a mismatch.
                    if(!(std::abs(gpu - reference) <= tolerance))
                    {
                        if(mismatches == 0)
                        {
                            firstBatch = b;
                            firstPosition = i;
                            firstGpu = gpu;
                            firstReference = reference;
                        }
                        ++mismatches;
                    }
                }
            }
            EXPECT_EQ(mismatches, 0)
                << "y of channel " << c << ": " << mismatches << " of "
                << static_cast<int64_t>(count) << " values off by more than " << tolerance
                << ", first at batch " << firstBatch << ", spatial position " << firstPosition
                << " (gpu " << firstGpu << ", reference " << firstReference << ")";
        }
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
};

} // namespace

// fp16 x upcast to an fp32 y: 32 x 2 x 1024 x 1024, 2^25 values per channel
TEST_F(IntegrationGpuBatchnormTrainingSpatialBeyondWorkgroup, Fp16UpcastN32Spatial1024x1024)
{
    runCase<Fp16>(32, 1024, 1024);
}

// fp32 with two batches: 2 x 2 x 4096 x 4096, 2^25 values per channel
TEST_F(IntegrationGpuBatchnormTrainingSpatialBeyondWorkgroup, Fp32N2Spatial4096x4096)
{
    runCase<float>(2, 4096, 4096);
}

} // namespace hip_kernel_provider::batchnorm_training_spatial::test
