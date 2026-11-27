// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

namespace hip_kernel_provider::batchnorm::test
{
namespace
{
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_test_sdk::utilities::CpuFpReferenceBatchnorm;

using FloatTensor = Tensor<float>;
using TensorAttr = std::shared_ptr<TensorAttributes>;

FloatTensor like(const FloatTensor& tensor)
{
    return {tensor.dims(), tensor.strides()};
}

struct BatchnormInputs
{
    BatchnormInputs(const std::vector<int64_t>& dims, const TensorLayout& layout, bool shifted)
        : x(dims, layout)
        , scale({1, dims[1], 1, 1}, layout)
        , bias(like(scale))
    {
        x.fillWithRandomValues(shifted ? 99.99f : -1.0f, shifted ? 100.01f : 1.0f, 42);
        scale.fillWithRandomValues(0.75f, 1.25f, 43);
        bias.fillWithRandomValues(-0.125f, 0.125f, 44);
    }

    FloatTensor x;
    FloatTensor scale;
    FloatTensor bias;
};

struct BatchnormStats
{
    explicit BatchnormStats(const BatchnormInputs& inputs)
        : mean(like(inputs.scale))
        , invVariance(like(inputs.scale))
    {
    }

    FloatTensor mean;
    FloatTensor invVariance;
};

struct BatchnormGradients
{
    explicit BatchnormGradients(const BatchnormInputs& inputs)
        : dx(like(inputs.x))
        , dscale(like(inputs.scale))
        , dbias(like(inputs.scale))
    {
    }

    FloatTensor dx;
    FloatTensor dscale;
    FloatTensor dbias;
};

struct BoundGraph
{
    BoundGraph()
    {
        graph.set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);
    }

    TensorAttr input(const char* name, FloatTensor& tensor)
    {
        auto attr = std::make_shared<TensorAttributes>();
        attr->set_name(name)
            .set_dim(tensor.dims())
            .set_stride(tensor.strides())
            .set_data_type(DataType::FLOAT);
        inputs.emplace_back(attr, &tensor);
        return attr;
    }

    void output(const TensorAttr& attr, FloatTensor& tensor)
    {
        attr->set_output(true)
            .set_dim(tensor.dims())
            .set_stride(tensor.strides())
            .set_data_type(DataType::FLOAT);
        outputs.emplace_back(attr, &tensor);
    }

    Graph graph;
    std::vector<std::pair<TensorAttr, FloatTensor*>> inputs;
    std::vector<std::pair<TensorAttr, FloatTensor*>> outputs;
};

void expectClose(
    const char* name, FloatTensor& expected, FloatTensor& actual, float atol, float rtol = 0.0f)
{
    ASSERT_EQ(expected.dims(), actual.dims());
    ASSERT_EQ(expected.strides(), actual.strides());
    // Non-const host access copies device results back before the comparison.
    const auto* ref = expected.memory().hostData();
    const auto* got = actual.memory().hostData();
    if(atol == 0.0f && rtol == 0.0f)
    {
        EXPECT_EQ(std::memcmp(got, ref, expected.elementCount() * sizeof(float)), 0) << name;
        return;
    }
    for(size_t i = 0; i < expected.elementCount(); ++i)
    {
        ASSERT_NEAR(got[i], ref[i], atol + rtol * std::abs(ref[i])) << name << " at offset " << i;
    }
}

void applyRelu(FloatTensor& tensor)
{
    auto* values = tensor.memory().hostData();
    std::transform(values, values + tensor.elementCount(), values, [](float value) {
        return std::max(0.0f, value);
    });
    tensor.markHostModified();
}

void referenceTraining(const BatchnormInputs& inputs, FloatTensor& y, BatchnormStats& stats)
{
    // The SDK reference uses two passes. Explicitly select double computation, not its float default.
    CpuFpReferenceBatchnorm::fwdTraining<float, float, float, float, double>(
        inputs.x,
        inputs.scale,
        inputs.bias,
        y,
        BATCHNORM_DEFAULT_EPSILON,
        0.0,
        &stats.mean,
        &stats.invVariance);
}

void referenceBackward(const BatchnormInputs& inputs,
                       const FloatTensor& dy,
                       BatchnormGradients& gradients,
                       const BatchnormStats* saved = nullptr)
{
    CpuFpReferenceBatchnorm::backward<float, float, float, float, float, double>(
        dy,
        inputs.x,
        inputs.scale,
        gradients.dx,
        gradients.dscale,
        gradients.dbias,
        saved != nullptr ? &saved->mean : nullptr,
        saved != nullptr ? &saved->invVariance : nullptr,
        BATCHNORM_DEFAULT_EPSILON);
}

class IntegrationGpuBatchnorm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        const auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH)
                  .string();
        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    void run(BoundGraph& bound, bool supported = true)
    {
        auto& graph = bound.graph;
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> engines;
        result = graph.get_ranked_engine_ids(engines);
        const auto found = std::find(engines.begin(), engines.end(), HIP_MLOPS_ENGINE_ID);
        if(!supported)
        {
            ASSERT_TRUE(result.code == ErrorCode::OK
                        || result.code == ErrorCode::GRAPH_NOT_SUPPORTED)
                << result.err_msg;
            EXPECT_EQ(found, engines.end());
            return;
        }
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(found, engines.end());
        result = graph.create_execution_plan_ext(HIP_MLOPS_ENGINE_ID, {});
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t workspaceSize = 0;
        result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace workspace(static_cast<size_t>(workspaceSize));
        std::unordered_map<int64_t, void*> pack;
        for(const auto& [attr, tensor] : bound.inputs)
        {
            pack.emplace(attr->get_uid(), tensor->rawDeviceData());
        }
        for(const auto& [attr, tensor] : bound.outputs)
        {
            tensor->fillWithSentinelValue();
            pack.emplace(attr->get_uid(), tensor->rawDeviceData());
        }
        result = graph.execute(_handle, pack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        for(const auto& [attr, tensor] : bound.outputs)
        {
            tensor->markDeviceModified();
        }
    }

    void train(BatchnormInputs& inputs,
               FloatTensor& y,
               BatchnormStats& stats,
               const PointwiseAttributes* activation = nullptr,
               bool supported = true)
    {
        BoundGraph bound;
        auto x = bound.input("X", inputs.x);
        auto scale = bound.input("Scale", inputs.scale);
        auto bias = bound.input("Bias", inputs.bias);
        auto epsilon = std::make_shared<TensorAttributes>();
        epsilon->set_name("Epsilon")
            .set_dim({1})
            .set_stride({1})
            .set_data_type(DataType::FLOAT)
            .set_compile_time_constant(static_cast<float>(BATCHNORM_DEFAULT_EPSILON));
        auto [norm, mean, invVariance, nextMean, nextVar]
            = bound.graph.batchnorm(x, scale, bias, BatchnormAttributes().set_epsilon(epsilon));
        auto output = activation != nullptr ? bound.graph.pointwise(norm, *activation) : norm;
        bound.output(output, y);
        bound.output(mean, stats.mean);
        bound.output(invVariance, stats.invVariance);
        run(bound, supported);
    }

    static void addGradients(BoundGraph& bound,
                             const TensorAttr& dy,
                             const TensorAttr& x,
                             const TensorAttr& scale,
                             const BatchnormBackwardAttributes& attrs,
                             BatchnormGradients& gradients)
    {
        auto [dx, dscale, dbias] = bound.graph.batchnorm_backward(dy, x, scale, attrs);
        bound.output(dx, gradients.dx);
        bound.output(dscale, gradients.dscale);
        bound.output(dbias, gradients.dbias);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

struct VarianceCase
{
    const char* name;
    std::vector<int64_t> dims;
    const TensorLayout& layout;
    bool shifted = true;
};

class IntegrationGpuBatchnormVariance : public IntegrationGpuBatchnorm,
                                        public ::testing::WithParamInterface<VarianceCase>
{
};

TEST_P(IntegrationGpuBatchnormVariance, Training)
{
    const auto& testCase = GetParam();
    BatchnormInputs inputs(testCase.dims, testCase.layout, testCase.shifted);
    auto actual = like(inputs.x);
    auto expected = like(inputs.x);
    BatchnormStats actualStats(inputs);
    BatchnormStats expectedStats(inputs);
    ASSERT_NO_FATAL_FAILURE(train(inputs, actual, actualStats));
    referenceTraining(inputs, expected, expectedStats);

    // At mean 100, one float ULP becomes ~0.001 after multiplying by invStd (~150).
    // Allow rounding in mean/variance reductions, but not the O(1) cancellation in output.
    expectClose("Y", expected, actual, testCase.shifted ? 8e-3f : 2e-5f);
    expectClose("saved mean", expectedStats.mean, actualStats.mean, 5e-5f);
    expectClose("saved invVariance",
                expectedStats.invVariance,
                actualStats.invVariance,
                0.0f,
                testCase.shifted ? 5e-3f : 1e-4f);
}

TEST_P(IntegrationGpuBatchnormVariance, BackwardWithoutSavedStats)
{
    const auto& testCase = GetParam();
    BatchnormInputs inputs(testCase.dims, testCase.layout, testCase.shifted);
    auto dy = like(inputs.x);
    dy.fillWithRandomValues(0.1f, 0.3f, 45);
    const auto* xValues = inputs.x.memory().hostData();
    auto* dyValues = dy.memory().hostData();
    for(size_t i = 0; i < dy.elementCount(); ++i)
    {
        // A nonzero dot product makes dscale sensitive to a wrong inverse variance.
        dyValues[i] += testCase.shifted ? 50.0f * (xValues[i] - 100.0f) : 0.5f * xValues[i];
    }
    dy.markHostModified();

    BatchnormGradients actual(inputs);
    BatchnormGradients expected(inputs);
    BoundGraph bound;
    auto xAttr = bound.input("X", inputs.x);
    auto dyAttr = bound.input("DY", dy);
    auto scaleAttr = bound.input("Scale", inputs.scale);
    addGradients(bound, dyAttr, xAttr, scaleAttr, {}, actual);
    ASSERT_NO_FATAL_FAILURE(run(bound));
    referenceBackward(inputs, dy, expected);

    // dX amplifies the statistics error by another invStd. dBias does not depend on stats.
    expectClose("DX", expected.dx, actual.dx, testCase.shifted ? 2.5e-1f : 1e-3f, 5e-3f);
    expectClose("DScale",
                expected.dscale,
                actual.dscale,
                testCase.shifted ? 3e-2f : 2e-5f,
                testCase.shifted ? 1e-2f : 1e-4f);
    expectClose("DBias", expected.dbias, actual.dbias, 2e-4f, 2e-5f);
}

INSTANTIATE_TEST_SUITE_P(
    Standard,
    IntegrationGpuBatchnormVariance,
    // BatchnormCommon.hpp selects these variants independently for FP32 training and backward:
    // 0: NCHW, small HW; 1: NCHW, HW > 1024 and N*HW < 32M.
    // 2: NHWC, C divisible by 4, several HW tiles, and N >= 4 for backward scratch storage.
    // 3: NCHW, 512 < HW <= 1024, N < 3 for training and N < 32 for backward.
    ::testing::Values(VarianceCase{"Variant0", {4, 8, 5, 5}, TensorLayout::NCHW},
                      VarianceCase{"Variant1", {4, 8, 33, 33}, TensorLayout::NCHW},
                      VarianceCase{"Variant2", {4, 8, 17, 17}, TensorLayout::NHWC},
                      VarianceCase{"Variant3", {2, 8, 24, 24}, TensorLayout::NCHW},
                      VarianceCase{"CenteredVariant0", {1, 2048, 1, 2}, TensorLayout::NCHW, false}),
    [](const ::testing::TestParamInfo<VarianceCase>& info) { return info.param.name; });

PointwiseAttributes relu(PointwiseMode mode, std::optional<float> slope)
{
    PointwiseAttributes attrs;
    attrs.set_mode(mode);
    if(slope.has_value())
    {
        attrs.set_relu_lower_clip_slope(*slope);
    }
    return attrs;
}

struct ReluCase
{
    const char* name;
    float slope;
    bool supported;
};

class IntegrationGpuBatchnormRelu : public IntegrationGpuBatchnorm,
                                    public ::testing::WithParamInterface<ReluCase>
{
};

TEST_P(IntegrationGpuBatchnormRelu, Training)
{
    const auto& [name, slope, supported] = GetParam();
    BatchnormInputs inputs({4, 8, 5, 5}, TensorLayout::NCHW, false);
    auto actual = like(inputs.x);
    BatchnormStats actualStats(inputs);
    const auto explicitSlope = relu(PointwiseMode::RELU_FWD, slope);
    ASSERT_NO_FATAL_FAILURE(train(inputs, actual, actualStats, &explicitSlope, supported));
    if(!supported)
    {
        return;
    }

    auto unset = like(inputs.x);
    auto expected = like(inputs.x);
    BatchnormStats unsetStats(inputs);
    BatchnormStats expectedStats(inputs);
    const auto unsetSlope = relu(PointwiseMode::RELU_FWD, std::nullopt);
    ASSERT_NO_FATAL_FAILURE(train(inputs, unset, unsetStats, &unsetSlope));
    referenceTraining(inputs, expected, expectedStats);
    applyRelu(expected);
    expectClose("Y", unset, actual, 0.0f);
    expectClose("Y reference", expected, actual, 2e-5f);
    expectClose("saved mean", unsetStats.mean, actualStats.mean, 0.0f);
    expectClose("saved mean reference", expectedStats.mean, actualStats.mean, 2e-6f);
    expectClose("saved invVariance", unsetStats.invVariance, actualStats.invVariance, 0.0f);
    expectClose(
        "saved invVariance reference", expectedStats.invVariance, actualStats.invVariance, 2e-5f);
}

TEST_P(IntegrationGpuBatchnormRelu, Inference)
{
    const auto& [name, slope, supported] = GetParam();
    BatchnormInputs inputs({4, 8, 5, 5}, TensorLayout::NCHW, false);
    BatchnormStats saved(inputs);
    auto expected = like(inputs.x);
    // Use the same saved statistics as the independent training reference, then apply ReLU.
    referenceTraining(inputs, expected, saved);
    auto runInference = [&](FloatTensor& output, std::optional<float> reluSlope, bool isSupported) {
        BoundGraph bound;
        auto norm = bound.graph.batchnorm_inference(bound.input("X", inputs.x),
                                                    bound.input("Mean", saved.mean),
                                                    bound.input("InvVariance", saved.invVariance),
                                                    bound.input("Scale", inputs.scale),
                                                    bound.input("Bias", inputs.bias),
                                                    {});
        bound.output(bound.graph.pointwise(norm, relu(PointwiseMode::RELU_FWD, reluSlope)), output);
        run(bound, isSupported);
    };
    auto actual = like(inputs.x);
    ASSERT_NO_FATAL_FAILURE(runInference(actual, slope, supported));
    if(!supported)
    {
        return;
    }

    auto unset = like(inputs.x);
    ASSERT_NO_FATAL_FAILURE(runInference(unset, std::nullopt, true));
    applyRelu(expected);
    expectClose("Y", unset, actual, 0.0f);
    expectClose("Y reference", expected, actual, 2e-5f);
}

TEST_P(IntegrationGpuBatchnormRelu, Backward)
{
    const auto& [name, slope, supported] = GetParam();
    BatchnormInputs inputs({4, 8, 5, 5}, TensorLayout::NCHW, false);
    BatchnormStats saved(inputs);
    auto preActivation = like(inputs.x);
    referenceTraining(inputs, preActivation, saved);
    auto dy = like(inputs.x);
    dy.fillWithRandomValues(-1.0f, 1.0f, 45);

    auto runBackward
        = [&](BatchnormGradients& gradients, std::optional<float> reluSlope, bool isSupported) {
              BoundGraph bound;
              auto x = bound.input("X", inputs.x);
              auto scale = bound.input("Scale", inputs.scale);
              auto mean = bound.input("Mean", saved.mean);
              auto invVariance = bound.input("InvVariance", saved.invVariance);
              auto norm = bound.graph.batchnorm_inference(
                  x, mean, invVariance, scale, bound.input("Bias", inputs.bias), {});
              auto maskedDy = bound.graph.pointwise(
                  bound.input("DY", dy), norm, relu(PointwiseMode::RELU_BWD, reluSlope));
              BatchnormBackwardAttributes attrs;
              attrs.set_saved_mean_and_inv_variance(mean, invVariance);
              addGradients(bound, maskedDy, x, scale, attrs, gradients);
              run(bound, isSupported);
          };

    BatchnormGradients actual(inputs);
    ASSERT_NO_FATAL_FAILURE(runBackward(actual, slope, supported));
    if(!supported)
    {
        return;
    }

    BatchnormGradients unset(inputs);
    ASSERT_NO_FATAL_FAILURE(runBackward(unset, std::nullopt, true));
    auto maskedDy = like(dy);
    const auto* norm = preActivation.memory().hostData();
    const auto* upstream = dy.memory().hostData();
    auto* masked = maskedDy.memory().hostData();
    for(size_t i = 0; i < dy.elementCount(); ++i)
    {
        masked[i] = norm[i] > 0.0f ? upstream[i] : 0.0f;
    }
    maskedDy.markHostModified();
    BatchnormGradients expected(inputs);
    referenceBackward(inputs, maskedDy, expected, &saved);
    expectClose("DX", unset.dx, actual.dx, 0.0f);
    expectClose("DScale", unset.dscale, actual.dscale, 0.0f);
    expectClose("DBias", unset.dbias, actual.dbias, 0.0f);
    expectClose("DX reference", expected.dx, actual.dx, 2e-5f);
    expectClose("DScale reference", expected.dscale, actual.dscale, 2e-5f);
    expectClose("DBias reference", expected.dbias, actual.dbias, 2e-5f);
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         IntegrationGpuBatchnormRelu,
                         ::testing::Values(ReluCase{"PositiveZero", 0.0f, true},
                                           ReluCase{"NegativeZero", -0.0f, true},
                                           ReluCase{"PositiveNonzero", 0.125f, false},
                                           ReluCase{"NegativeNonzero", -0.125f, false}),
                         [](const ::testing::TestParamInfo<ReluCase>& info) {
                             return info.param.name;
                         });

} // namespace
} // namespace hip_kernel_provider::batchnorm::test
