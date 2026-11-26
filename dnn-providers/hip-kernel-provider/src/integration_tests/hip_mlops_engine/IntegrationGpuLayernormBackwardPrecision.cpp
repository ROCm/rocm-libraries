// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace hip_kernel_provider::layernorm_bwd_precision::test
{

namespace
{

// The HIP MLops layernorm backward kernel must stay accurate
// in fp32 when the activations sit far from zero relative to their spread (|mean| >> std).
// An uncentered dx = rstd*dy*scale - a*x - b subtracts terms of size |mean|*rstd that cancel
// down to the spread, so the kernel centers x before forming them. The fp32 dx is compared
// against CpuFpReferenceLayernorm::bprop evaluated in double, fed the same fp32 inputs and
// the same saved statistics the kernel receives.
class IntegrationGpuLayernormBackwardPrecision : public ::testing::Test
{
protected:
    static constexpr int64_t N = 4;
    static constexpr int64_t C = 6;
    static constexpr int64_t H = 8;
    static constexpr int64_t W = 16;
    static constexpr int64_t ROW_SIZE = C * H * W;
    static constexpr float EPSILON = 1e-5f;
    static constexpr float SPREAD = 0.01f;
    static constexpr unsigned int SEED = 3339;

    // Acceptance bound on dx; a centered fp32 kernel lands near 3e-5.
    static constexpr double DX_TOLERANCE = 0.004;
    // dscale and dbias accumulate over only N = 4 rows from already-centered terms: values
    // reach a few units, fp32 rounding is ~1e-6 relative, so an absolute-plus-relative bound
    // of 1e-3 is generous without being vacuous against a wrong sum.
    static constexpr double WEIGHT_TOLERANCE = 1e-3;

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

    void runCase(float meanOffset)
    {
        const std::vector<int64_t> dims = {N, C, H, W};
        const std::vector<int64_t> dataStrides = {ROW_SIZE, H * W, W, 1};
        const std::vector<int64_t> scaleDims = {1, C, H, W};
        const std::vector<int64_t> scaleStrides = {ROW_SIZE, H * W, W, 1};
        const std::vector<int64_t> statDims = {N, 1, 1, 1};
        const std::vector<int64_t> statStrides = {1, 1, 1, 1};

        Tensor<float> dyTensor(dims, dataStrides);
        Tensor<float> xTensor(dims, dataStrides);
        Tensor<float> scaleTensor(scaleDims, scaleStrides);
        Tensor<float> meanTensor(statDims, statStrides);
        Tensor<float> invVarianceTensor(statDims, statStrides);
        Tensor<float> dxTensor(dims, dataStrides);
        Tensor<float> dscaleTensor(scaleDims, scaleStrides);
        Tensor<float> dbiasTensor(scaleDims, scaleStrides);

        // The statistics enter the reference as double copies of the stored fp32 values.
        Tensor<double> meanRef(statDims, statStrides);
        Tensor<double> invVarianceRef(statDims, statStrides);

        // All of dy, x, scale and dx are packed, so a flat index walks them in memory order.
        dyTensor.memory().markHostModified();
        xTensor.memory().markHostModified();
        scaleTensor.memory().markHostModified();
        meanTensor.memory().markHostModified();
        invVarianceTensor.memory().markHostModified();
        meanRef.memory().markHostModified();
        invVarianceRef.memory().markHostModified();
        float* dy = dyTensor.memory().hostData();
        float* x = xTensor.memory().hostData();
        float* scale = scaleTensor.memory().hostData();
        float* mean = meanTensor.memory().hostData();
        float* invVariance = invVarianceTensor.memory().hostData();
        double* meanD = meanRef.memory().hostData();
        double* invVarianceD = invVarianceRef.memory().hostData();

        std::mt19937 rng(SEED);
        for(int64_t i = 0; i < N * ROW_SIZE; ++i)
        {
            x[i] = meanOffset + SPREAD * uniform(rng, -1.0f, 1.0f);
        }
        for(int64_t i = 0; i < N * ROW_SIZE; ++i)
        {
            dy[i] = uniform(rng, -1.0f, 1.0f);
        }
        for(int64_t i = 0; i < ROW_SIZE; ++i)
        {
            scale[i] = uniform(rng, 0.5f, 1.5f);
        }

        // Saved statistics, the way the forward pass would have produced them: double
        // accumulation over the fp32-rounded x, stored as fp32.
        for(int64_t n = 0; n < N; ++n)
        {
            const float* row = x + n * ROW_SIZE;
            double sum = 0.0;
            for(int64_t i = 0; i < ROW_SIZE; ++i)
            {
                sum += static_cast<double>(row[i]);
            }
            const double rowMean = sum / static_cast<double>(ROW_SIZE);
            double sumSq = 0.0;
            for(int64_t i = 0; i < ROW_SIZE; ++i)
            {
                const double d = static_cast<double>(row[i]) - rowMean;
                sumSq += d * d;
            }
            const double variance = sumSq / static_cast<double>(ROW_SIZE);
            mean[n] = static_cast<float>(rowMean);
            invVariance[n]
                = static_cast<float>(1.0 / std::sqrt(variance + static_cast<double>(EPSILON)));
            meanD[n] = static_cast<double>(mean[n]);
            invVarianceD[n] = static_cast<double>(invVariance[n]);
        }

        Graph graphObj;
        graphObj.set_name("LayernormBackwardLargeMean")
            .set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_preferred_engine_id_ext(HIP_MLOPS_ENGINE_ID);

        auto dyAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("dy", hipdnn_frontend::DataType::FLOAT, dims, dataStrides));
        auto xAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("x", hipdnn_frontend::DataType::FLOAT, dims, dataStrides));
        auto scaleAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "scale", hipdnn_frontend::DataType::FLOAT, scaleDims, scaleStrides));
        auto meanAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("mean", hipdnn_frontend::DataType::FLOAT, statDims, statStrides));
        auto invVarianceAttr = std::make_shared<TensorAttributes>(makeTensorAttributes(
            "inv_variance", hipdnn_frontend::DataType::FLOAT, statDims, statStrides));
        auto epsilonAttr = std::make_shared<TensorAttributes>();
        epsilonAttr->set_name("epsilon")
            .set_dim({1})
            .set_stride({1})
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compile_time_constant(EPSILON);

        LayernormBackwardAttributes lnAttrs;
        lnAttrs.set_epsilon(epsilonAttr);
        lnAttrs.set_saved_mean_and_inv_variance(meanAttr, invVarianceAttr);

        auto outputs = graphObj.layernorm_backward(dyAttr, xAttr, scaleAttr, lnAttrs);
        const auto& dxAttr = outputs[0];
        const auto& dscaleAttr = outputs[1];
        const auto& dbiasAttr = outputs[2];
        dxAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(dims)
            .set_stride(dataStrides);
        dscaleAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(scaleDims)
            .set_stride(scaleStrides);
        dbiasAttr->set_output(true)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(scaleDims)
            .set_stride(scaleStrides);

        auto result = graphObj.build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The engine preference is soft; the test is about this engine's kernel, so a plan
        // served by any other engine must fail it rather than pass vacuously.
        int64_t servingEngineId = 0;
        result = graphObj.get_execution_plan_engine_id(servingEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngineId, HIP_MLOPS_ENGINE_ID)
            << "layernorm backward was not served by HIP_MLOPS_ENGINE";

        int64_t workspaceSize = 0;
        result = graphObj.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace workspace(static_cast<size_t>(workspaceSize));

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[dyAttr->get_uid()] = dyTensor.memory().deviceData();
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[scaleAttr->get_uid()] = scaleTensor.memory().deviceData();
        variantPack[meanAttr->get_uid()] = meanTensor.memory().deviceData();
        variantPack[invVarianceAttr->get_uid()] = invVarianceTensor.memory().deviceData();
        variantPack[dxAttr->get_uid()] = dxTensor.memory().deviceData();
        variantPack[dscaleAttr->get_uid()] = dscaleTensor.memory().deviceData();
        variantPack[dbiasAttr->get_uid()] = dbiasTensor.memory().deviceData();

        result = graphObj.execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        // Reference: double compute and double saved statistics over the same fp32 inputs.
        Tensor<float> dxRef(dims, dataStrides);
        Tensor<float> dscaleRef(scaleDims, scaleStrides);
        Tensor<float> dbiasRef(scaleDims, scaleStrides);
        CpuFpReferenceLayernorm::bprop<float, float, float, double, double>(
            dyTensor,
            xTensor,
            scaleTensor,
            dxRef,
            dscaleRef,
            dbiasRef,
            static_cast<double>(EPSILON),
            &meanRef,
            &invVarianceRef,
            3);

        dxTensor.memory().markDeviceModified();
        dscaleTensor.memory().markDeviceModified();
        dbiasTensor.memory().markDeviceModified();

        const float* dxGpu = dxTensor.memory().hostData();
        const float* dscaleGpu = dscaleTensor.memory().hostData();
        const float* dbiasGpu = dbiasTensor.memory().hostData();
        const float* dxCpu = dxRef.memory().hostData();
        const float* dscaleCpu = dscaleRef.memory().hostData();
        const float* dbiasCpu = dbiasRef.memory().hostData();

        double maxDxError = 0.0;
        int64_t worstDx = 0;
        for(int64_t i = 0; i < N * ROW_SIZE; ++i)
        {
            const double error
                = std::abs(static_cast<double>(dxGpu[i]) - static_cast<double>(dxCpu[i]));
            // A NaN is sticky: it stays the recorded worst element, so the check below fails
            // on it even when later elements have finite errors.
            if(std::isnan(maxDxError))
            {
                continue;
            }
            if(std::isnan(error) || error > maxDxError)
            {
                maxDxError = error;
                worstDx = i;
            }
        }
        EXPECT_LT(maxDxError, DX_TOLERANCE)
            << "dx max |gpu - reference| = " << maxDxError << " at element " << worstDx << " (gpu "
            << dxGpu[worstDx] << ", reference " << dxCpu[worstDx] << ") for x = " << meanOffset
            << " + " << SPREAD << "*U(-1,1)";

        double maxDscaleError = 0.0;
        double maxDbiasError = 0.0;
        for(int64_t i = 0; i < ROW_SIZE; ++i)
        {
            const double dscaleError
                = std::abs(static_cast<double>(dscaleGpu[i]) - static_cast<double>(dscaleCpu[i]));
            const double dbiasError
                = std::abs(static_cast<double>(dbiasGpu[i]) - static_cast<double>(dbiasCpu[i]));
            EXPECT_LE(dscaleError,
                      WEIGHT_TOLERANCE * (1.0 + std::abs(static_cast<double>(dscaleCpu[i]))))
                << "dscale element " << i << " gpu " << dscaleGpu[i] << " reference "
                << dscaleCpu[i];
            EXPECT_LE(dbiasError,
                      WEIGHT_TOLERANCE * (1.0 + std::abs(static_cast<double>(dbiasCpu[i]))))
                << "dbias element " << i << " gpu " << dbiasGpu[i] << " reference " << dbiasCpu[i];
            maxDscaleError = std::max(maxDscaleError, dscaleError);
            maxDbiasError = std::max(maxDbiasError, dbiasError);
        }
        const auto scientific = [](double value) {
            std::ostringstream stream;
            stream << std::scientific << std::setprecision(3) << value;
            return stream.str();
        };
        RecordProperty("max_dx_error", scientific(maxDxError));
        RecordProperty("max_dscale_error", scientific(maxDscaleError));
        RecordProperty("max_dbias_error", scientific(maxDbiasError));
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
};

} // namespace

TEST_F(IntegrationGpuLayernormBackwardPrecision, LargeMeanOffset100)
{
    runCase(100.0f);
}

TEST_F(IntegrationGpuLayernormBackwardPrecision, LargeMeanOffset1000)
{
    runCase(1000.0f);
}

} // namespace hip_kernel_provider::layernorm_bwd_precision::test
