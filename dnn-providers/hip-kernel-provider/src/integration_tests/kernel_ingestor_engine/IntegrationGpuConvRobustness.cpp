// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"
#include "engines/kernel_ingestor_engine/ConvEngineGroupGate.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

/**
 * @file IntegrationGpuConvRobustness.cpp
 * @brief The conv engines' failure paths through the public API:
 *        - a HIPRTC compile failure costs only its engine: the graph is served by the next one;
 *        - with HIPRTC disabled, a program the disk cache holds still serves and any other plan
 *          declines at plan build.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

enum class Direction
{
    FWD,
    BWD,
    WRW,
};

/// One convolution, 2-D, packed.
struct ConvSpec
{
    Direction direction = Direction::FWD;
    int64_t n = 2;
    int64_t c = 8;
    int64_t k = 8;
    int64_t groups = 1;
    int64_t h = 8;
    int64_t w = 8;
    int64_t r = 3;
    int64_t s = 3;
    int64_t pad = 1;
    int64_t stride = 1;
    bool channelsLast = false;
    DataType dataType = DataType::FLOAT;

    int64_t outH() const
    {
        return (h + 2 * pad - r) / stride + 1;
    }
    int64_t outW() const
    {
        return (w + 2 * pad - s) / stride + 1;
    }
};

std::vector<int64_t> stridesFor(const std::vector<int64_t>& dims, bool channelsLast)
{
    if(channelsLast)
    {
        return {dims[1] * dims[2] * dims[3], 1, dims[3] * dims[1], dims[1]};
    }
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

std::shared_ptr<TensorAttributes> tensor(int64_t uid,
                                         const std::string& name,
                                         const std::vector<int64_t>& dims,
                                         const ConvSpec& spec)
{
    auto attributes = std::make_shared<TensorAttributes>();
    attributes->set_uid(uid)
        .set_name(name)
        .set_dim(dims)
        .set_stride(stridesFor(dims, spec.channelsLast))
        .set_data_type(spec.dataType);
    return attributes;
}

std::shared_ptr<Graph> buildGraph(const ConvSpec& spec)
{
    const std::vector<int64_t> x = {spec.n, spec.c, spec.h, spec.w};
    const std::vector<int64_t> w = {spec.k, spec.c / spec.groups, spec.r, spec.s};
    const std::vector<int64_t> y = {spec.n, spec.k, spec.outH(), spec.outW()};
    const std::vector<int64_t> pad = {spec.pad, spec.pad};
    const std::vector<int64_t> stride = {spec.stride, spec.stride};
    const std::vector<int64_t> dilation = {1, 1};

    auto graph = std::make_shared<Graph>();
    graph->set_name("conv_robustness")
        .set_io_data_type(spec.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    std::shared_ptr<TensorAttributes> output;
    std::vector<int64_t> outputDims;
    switch(spec.direction)
    {
    case Direction::FWD:
    {
        ConvFpropAttributes attributes;
        attributes.set_pre_padding(pad).set_post_padding(pad).set_stride(stride).set_dilation(
            dilation);
        output = graph->conv_fprop(tensor(1, "X", x, spec), tensor(2, "W", w, spec), attributes);
        output->set_uid(3).set_name("Y");
        outputDims = y;
        break;
    }
    case Direction::BWD:
    {
        ConvDgradAttributes attributes;
        attributes.set_pre_padding(pad).set_post_padding(pad).set_stride(stride).set_dilation(
            dilation);
        output = graph->conv_dgrad(tensor(3, "DY", y, spec), tensor(2, "W", w, spec), attributes);
        output->set_uid(1).set_name("DX");
        outputDims = x;
        break;
    }
    case Direction::WRW:
    default:
    {
        ConvWgradAttributes attributes;
        attributes.set_pre_padding(pad).set_post_padding(pad).set_stride(stride).set_dilation(
            dilation);
        output = graph->conv_wgrad(tensor(3, "DY", y, spec), tensor(1, "X", x, spec), attributes);
        output->set_uid(2).set_name("DW");
        outputDims = w;
        break;
    }
    }
    output->set_dim(outputDims)
        .set_stride(stridesFor(outputDims, spec.channelsLast))
        .set_output(true)
        .set_data_type(spec.dataType);
    return graph;
}

/// Error relative to fp32 epsilon an output may carry: the reduction length, widened for a
/// 16-bit type (its rounding of the output, past fp32's).
float toleranceEpsilons(const ConvSpec& spec)
{
    const auto reduction = spec.direction == Direction::WRW
                               ? spec.n * spec.outH() * spec.outW()
                               : (spec.c / spec.groups) * spec.r * spec.s * spec.k / spec.groups;
    const float base = 4.0F * static_cast<float>(reduction);
    return spec.dataType == DataType::FLOAT ? base : base + 16384.0F;
}

int64_t engineId(std::string_view name)
{
    return hipdnn_data_sdk::utilities::engineNameToId(std::string(name));
}

/// One case: an engine, and a graph for it.
struct ConvCase
{
    std::string name;
    const char* engine;
    ConvSpec spec;
};

ConvSpec spec(Direction direction,
              int64_t c,
              int64_t k,
              int64_t groups,
              int64_t size,
              int64_t filter,
              int64_t pad,
              int64_t stride,
              bool channelsLast,
              DataType dataType = DataType::FLOAT,
              int64_t n = 2)
{
    ConvSpec result;
    result.direction = direction;
    result.n = n;
    result.c = c;
    result.k = k;
    result.groups = groups;
    result.h = size;
    result.w = size;
    result.r = filter;
    result.s = filter;
    result.pad = pad;
    result.stride = stride;
    result.channelsLast = channelsLast;
    result.dataType = dataType;
    return result;
}

/// gtest prints a case by its name rather than its bytes.
void PrintTo(const ConvCase& value, std::ostream* out)
{
    *out << value.name;
}

} // namespace

class IntegrationGpuConvRobustness
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      ConvCase>
{
protected:
    void SetUp() override
    {
        if(!convEngineGroupEnabled())
        {
            GTEST_SKIP() << "the conv engine group is not enabled (set " << CONV_ENABLE_ENV
                         << "=1)";
        }
        IntegrationGraphVerificationHarness::SetUp();
    }

    /// Builds @p graph with @p engine preferred, under @p workspaceLimit when it is not negative.
    Error buildPreferring(Graph& graph, int64_t engine, int64_t workspaceLimit = -1)
    {
        auto result = graph.build_operation_graph(_handle);
        if(result.is_bad())
        {
            return result;
        }
        graph.set_preferred_engine_id_ext(engine);
        result = graph.create_execution_plans();
        if(result.is_bad())
        {
            return result;
        }
        // After create_execution_plans(), which clears any limit set before it.
        if(workspaceLimit >= 0)
        {
            graph.deselect_workspace_greater_than(workspaceLimit);
        }
        result = graph.check_support();
        if(result.is_bad())
        {
            return result;
        }
        return graph.build_plans();
    }

    int64_t servingEngine(Graph& graph)
    {
        int64_t serving = 0;
        const auto result = graph.get_execution_plan_engine_id(serving);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        return serving;
    }

    void verify(Graph& graph, const ConvSpec& spec)
    {
        hip_kernel_provider::test_utilities::GraphVerificationContext context(graph);
        registerValidatorsForOutputs(context, toleranceEpsilons(spec));
        verifyBuiltGraph(context, 0);
    }
};

// A compile that fails costs its engine only: with the naive engine preferred and its program
// failing to compile, the graph is served by another engine, and one warning names the program.
TEST_F(IntegrationGpuConvRobustness, ACompileFailureFallsToTheNextEngine)
{
    ASSERT_EQ(detail::resetProviderHiprtcModules(), "");
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "conv-compile-failure", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter inject(
        "HIP_KERNEL_PROVIDER_TEST_FAIL_COMPILE", "kernels/ConvNaive.cpp");
    const ScopedPluginLogCapture capture(this, HIPDNN_SEV_WARN);

    const auto conv = spec(Direction::FWD, 8, 8, 1, 8, 3, 1, 1, false);
    auto graph = buildGraph(conv);
    const auto naive = engineId("hipkernel:CONV_NAIVE_FWD");
    const auto result = buildPreferring(*graph, naive);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const auto serving = servingEngine(*graph);
    EXPECT_NE(serving, naive);

    EXPECT_TRUE(capture.recorder().hasLogContaining(HIPDNN_SEV_WARN, "kernels/ConvNaive.cpp"))
        << capture.recorder().getRecordedLogsAsString();
    verify(*graph, conv);
    ASSERT_EQ(detail::resetProviderHiprtcModules(), "");
}

// With HIPRTC disabled, a plan whose program is not cached declines at plan build, never at
// execute; once the disk cache holds it, the plan builds and runs without HIPRTC.
TEST_F(IntegrationGpuConvRobustness, WithHiprtcDisabledCachedProgramsServeAndMissesDecline)
{
    ASSERT_EQ(detail::resetProviderHiprtcModules(), "");
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir(
        "conv-no-hiprtc", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);
    const auto conv = spec(Direction::FWD, 8, 8, 1, 8, 3, 1, 1, false);
    const auto naive = engineId("hipkernel:CONV_NAIVE_FWD");
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
            "HIP_KERNEL_PROVIDER_DISABLE_HIPRTC", "1");
        auto graph = buildGraph(conv);
        const auto result = buildPreferring(*graph, naive);
        // Every conv engine this binary loads compiles from source, so none can serve.
        EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED) << result.err_msg;
    }

    // Compile once, with HIPRTC available, into the disk cache.
    {
        auto graph = buildGraph(conv);
        const auto result = buildPreferring(*graph, naive);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngine(*graph), naive);
    }
    ASSERT_EQ(detail::resetProviderHiprtcModules(), "");

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter disabled(
        "HIP_KERNEL_PROVIDER_DISABLE_HIPRTC", "1");
    auto graph = buildGraph(conv);
    const auto result = buildPreferring(*graph, naive);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(servingEngine(*graph), naive);
    verify(*graph, conv);
    ASSERT_EQ(detail::resetProviderHiprtcModules(), "");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
