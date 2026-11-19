// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/EngineQueries.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
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
 *          declines at plan build;
 *        - a workspace limit below an engine's need declines that engine at plan build, through
 *          the frontend's deselect_workspace_greater_than() and through the backend knob
 *          global.workspace_size_limit as the plugin receives it; and the workspace a plan
 *          reports is all it uses.
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

class InspectableConvGraph : public Graph
{
public:
    using Graph::get_raw_graph_descriptor;
};

std::shared_ptr<InspectableConvGraph> buildGraph(const ConvSpec& spec)
{
    const std::vector<int64_t> x = {spec.n, spec.c, spec.h, spec.w};
    const std::vector<int64_t> w = {spec.k, spec.c / spec.groups, spec.r, spec.s};
    const std::vector<int64_t> y = {spec.n, spec.k, spec.outH(), spec.outW()};
    const std::vector<int64_t> pad = {spec.pad, spec.pad};
    const std::vector<int64_t> stride = {spec.stride, spec.stride};
    const std::vector<int64_t> dilation = {1, 1};

    auto graph = std::make_shared<InspectableConvGraph>();
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

std::string engineName(hipdnnHandle_t handle, int64_t id)
{
    size_t length = 0;
    if(hipdnnGetEngineNameById_ext(handle, id, nullptr, &length) != HIPDNN_STATUS_SUCCESS)
    {
        return std::to_string(id);
    }
    std::vector<char> name(length);
    if(hipdnnGetEngineNameById_ext(handle, id, name.data(), &length) != HIPDNN_STATUS_SUCCESS)
    {
        return std::to_string(id);
    }
    return name.data();
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

/// gtest prints a case by its name rather than its bytes; gtest looks the function up by this name.
// NOLINTNEXTLINE(readability-identifier-naming)
void PrintTo(const ConvCase& value, std::ostream* out)
{
    *out << value.name;
}

/// The families whose plans can take workspace, each with a small graph that makes its engine
/// stage, split or transform through it. The depthwise VALU, rows and matrix-core forward and
/// backward-data engines and the naive engines take none.
std::vector<ConvCase> workspaceCases()
{
    const auto fwd = Direction::FWD;
    const auto bwd = Direction::BWD;
    const auto wrw = Direction::WRW;
    const auto half = DataType::HALF;
    return {
        {"WinogradFwd", "hipkernel:CONV_WINOGRAD_FWD", spec(fwd, 16, 16, 1, 16, 3, 1, 1, false)},
        {"WinogradWrw", "hipkernel:CONV_WINOGRAD_WRW", spec(wrw, 16, 16, 1, 16, 3, 1, 1, false)},
        // Above the FFT engines' arithmetic floor: a large filter on a 64x64 image.
        {"FftFwd",
         "hipkernel:CONV_FFT_FWD",
         spec(fwd, 32, 32, 1, 64, 11, 5, 1, false, DataType::FLOAT, 1)},
        {"FftWrw",
         "hipkernel:CONV_FFT_WRW",
         spec(wrw, 32, 32, 1, 64, 11, 5, 1, false, DataType::FLOAT, 1)},
        {"Wrw", "hipkernel:CONV_WRW", spec(wrw, 32, 32, 1, 14, 3, 1, 1, true, half, 4)},
        {"IgemmFwd", "hipkernel:CONV_IGEMM_FWD", spec(fwd, 32, 32, 1, 14, 3, 1, 1, false, half)},
        {"IgemmBwd", "hipkernel:CONV_IGEMM_BWD", spec(bwd, 32, 32, 1, 14, 3, 1, 2, true, half)},
        // A long reduction over few rows, which the native-NCHW engine splits.
        {"IgemmNchwBwd",
         "hipkernel:CONV_IGEMM_NCHW_BWD",
         spec(bwd, 32, 256, 1, 4, 3, 1, 1, false, half, 1)},
        {"GroupFwd", "hipkernel:CONV_GROUP_FWD", spec(fwd, 32, 32, 4, 14, 3, 1, 1, false, half)},
        {"GroupWrw", "hipkernel:CONV_GROUP_WRW", spec(wrw, 32, 32, 4, 14, 3, 1, 1, true, half, 4)},
        {"GroupValuFwd", "hipkernel:CONV_GROUP_VALU_FWD", spec(fwd, 32, 32, 8, 14, 3, 1, 1, false)},
        {"GroupValuWrw",
         "hipkernel:CONV_GROUP_VALU_WRW",
         spec(wrw, 32, 32, 8, 28, 3, 1, 1, false, DataType::FLOAT, 8)},
        {"DepthwiseValuWrw",
         "hipkernel:CONV_DEPTHWISE_VALU_WRW",
         spec(wrw, 16, 16, 16, 16, 3, 1, 1, false, DataType::FLOAT, 4)},
        {"DirectFwd", "hipkernel:CONV_DIRECT_FWD", spec(fwd, 16, 16, 1, 14, 3, 1, 1, false)},
        {"SmallcFwd", "hipkernel:CONV_SMALLC_FWD", spec(fwd, 3, 16, 1, 32, 3, 1, 1, false, half)},
        {"StemBwd", "hipkernel:CONV_STEM_BWD", spec(bwd, 3, 16, 1, 32, 7, 3, 2, false, half)},
    };
}

/// The provider's plugin ABI, resolved from the copy the harness loaded.
struct PluginAbi
{
    void* library = nullptr;
    decltype(&hipdnnEnginePluginCreate) create = nullptr;
    decltype(&hipdnnEnginePluginDestroy) destroy = nullptr;
    decltype(&hipdnnEnginePluginCreateExecutionContext) createContext = nullptr;
    decltype(&hipdnnEnginePluginDestroyExecutionContext) destroyContext = nullptr;
    decltype(&hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext) workspaceSize = nullptr;

    PluginAbi()
    {
        library = hipdnn_data_sdk::utilities::openLoadedLibrary(detail::loadedProviderPluginPath());
        if(library == nullptr)
        {
            return;
        }
        // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
        create = reinterpret_cast<decltype(create)>(
            hipdnn_data_sdk::utilities::getSymbol(library, "hipdnnEnginePluginCreate"));
        destroy = reinterpret_cast<decltype(destroy)>(
            hipdnn_data_sdk::utilities::getSymbol(library, "hipdnnEnginePluginDestroy"));
        createContext
            = reinterpret_cast<decltype(createContext)>(hipdnn_data_sdk::utilities::getSymbol(
                library, "hipdnnEnginePluginCreateExecutionContext"));
        destroyContext
            = reinterpret_cast<decltype(destroyContext)>(hipdnn_data_sdk::utilities::getSymbol(
                library, "hipdnnEnginePluginDestroyExecutionContext"));
        workspaceSize
            = reinterpret_cast<decltype(workspaceSize)>(hipdnn_data_sdk::utilities::getSymbol(
                library, "hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext"));
        // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    }

    ~PluginAbi()
    {
        if(library != nullptr)
        {
            hipdnn_data_sdk::utilities::closeLibrary(library);
        }
    }

    PluginAbi(const PluginAbi&) = delete;
    PluginAbi& operator=(const PluginAbi&) = delete;
    PluginAbi(PluginAbi&&) = delete;
    PluginAbi& operator=(PluginAbi&&) = delete;

    bool complete() const
    {
        return create != nullptr && destroy != nullptr && createContext != nullptr
               && destroyContext != nullptr && workspaceSize != nullptr;
    }
};

/// An engine config for @p engine carrying global.workspace_size_limit = @p limit.
std::vector<uint8_t> engineConfigWithWorkspaceLimit(int64_t engine, int64_t limit)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::KnobSetting>> knobs;
    knobs.push_back(data_objects::CreateKnobSettingDirect(
        builder,
        hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME,
        data_objects::KnobValue::IntValue,
        data_objects::CreateIntValue(builder, limit).Union()));
    builder.Finish(data_objects::CreateEngineConfig(builder, engine, builder.CreateVector(knobs)));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
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

    static int64_t servingEngine(Graph& graph)
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

TEST_F(IntegrationGpuConvRobustness, SingletonSpatialLayoutsHaveDistinctCandidates)
{
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        GTEST_SKIP() << "no production descriptor catalogs are installed";
    }
    const auto properties = hip_kernel_provider_common::getDeviceProperties(_stream);
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    for(const auto* name : {"hipkernel:CONV_GROUP_VALU_BWD", "hipkernel:CONV_GROUP_VALU_WRW"})
    {
        std::set<std::string> availableLayouts;
        detail::forEachServedKernel({name, arch.c_str()}, [&](const auto& kernel) {
            const auto layout = kernel.metadata.find("layout");
            if(layout != kernel.metadata.end())
            {
                if(const auto* value = std::get_if<std::string>(&layout->second))
                {
                    availableLayouts.insert(*value);
                }
            }
        });
        if(availableLayouts != std::set<std::string>{"nchw", "nhwc"})
        {
            GTEST_SKIP() << name << " does not ship both layouts for " << arch;
        }
    }
    for(const auto direction : {Direction::BWD, Direction::WRW})
    {
        ConvSpec value;
        value.direction = direction;
        value.n = 4;
        value.c = direction == Direction::BWD ? 256 : 8;
        value.k = 8;
        value.h = value.w = 1;
        auto graph = buildGraph(value);
        ASSERT_FALSE(graph->build_operation_graph(_handle).is_bad());
        const auto name = direction == Direction::BWD ? "hipkernel:CONV_GROUP_VALU_BWD"
                                                      : "hipkernel:CONV_GROUP_VALU_WRW";
        EngineCandidatePage page;
        const auto result = hipdnn_frontend::detail::getEngineCandidates(
            graph->get_raw_graph_descriptor(), engineId(name), page);
        ASSERT_FALSE(result.is_bad()) << result.err_msg;
        std::set<std::string> layouts;
        for(const auto& candidate : page.candidates)
        {
            layouts.insert(candidate.kernelFeatures.at("kernel.layout").get<std::string>());
        }
        EXPECT_EQ(layouts, (std::set<std::string>{"nchw", "nhwc"}));
    }
}

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

// One case per family: the plan's reported workspace is all it uses, and a limit one byte
// below it declines the engine at plan build -- through the frontend and through the backend
// knob -- never at execute.
TEST_P(IntegrationGpuConvRobustness, WorkspaceLimitBelowTheNeedDeclinesAtPlanBuild)
{
    const auto& param = GetParam();
    const auto engine = engineId(param.engine);

    auto graph = buildGraph(param.spec);
    auto result = buildPreferring(*graph, engine);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    if(servingEngine(*graph) != engine)
    {
        GTEST_SKIP() << param.engine << " does not serve this graph on this device";
    }
    int64_t need = 0;
    ASSERT_EQ(graph->get_workspace_size(need).code, ErrorCode::OK);
    if(need == 0)
    {
        GTEST_SKIP() << param.engine << " needs no workspace for this graph on this device";
    }

    // The reported workspace is all the plan touches: a guard band past it stays intact.
    {
        hip_kernel_provider::test_utilities::GraphVerificationContext context(*graph);
        registerValidatorsForOutputs(context, toleranceEpsilons(param.spec));
        hip_kernel_provider::test_utilities::VerificationRun run;
        ASSERT_NO_FATAL_FAILURE(prepareVerification(context, 0, run));

        constexpr size_t GUARD = size_t{64} * 1024;
        constexpr unsigned char PATTERN = 0xA5;
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, static_cast<size_t>(need) + GUARD), hipSuccess);
        auto* guard = static_cast<unsigned char*>(workspace) + need;
        ASSERT_EQ(hipMemset(guard, PATTERN, GUARD), hipSuccess);
        auto variantPack = run.gpuBundle.toDeviceVariantPack();
        result = graph->execute(_handle, variantPack, workspace);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(stream()), hipSuccess);
        std::vector<unsigned char> band(GUARD);
        ASSERT_EQ(hipMemcpy(band.data(), guard, GUARD, hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        EXPECT_TRUE(std::all_of(
            band.begin(), band.end(), [](unsigned char byte) { return byte == PATTERN; }))
            << param.engine << " wrote past the " << need << " workspace bytes it reported";

        // And the result is right.
        ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *graph, _handle));
    }

    // The frontend's limit: the engine's plan is never the one that runs. Either the graph is
    // served within the limit, or the build fails at plan time.
    {
        auto limited = buildGraph(param.spec);
        result = buildPreferring(*limited, engine, need - 1);
        if(result.code == ErrorCode::OK)
        {
            int64_t used = 0;
            ASSERT_EQ(limited->get_workspace_size(used).code, ErrorCode::OK);
            EXPECT_LE(used, need - 1)
                << "served by " << engineName(_handle, servingEngine(*limited));
            verify(*limited, param.spec);
        }
        else
        {
            EXPECT_TRUE(result.code == ErrorCode::INVALID_VALUE
                        || result.code == ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED)
                << result.err_msg;
        }
    }

    // The backend knob, as the plugin receives it: the engine builds within the limit or
    // declines at plan build.
    {
        const PluginAbi abi;
        ASSERT_TRUE(abi.complete()) << "the provider's plugin ABI is not resolvable";
        auto [serialized, error] = graph->to_binary();
        ASSERT_TRUE(error.is_good()) << error.get_message();
        hipdnnEnginePluginHandle_t plugin = nullptr;
        ASSERT_EQ(abi.create(&plugin), HIPDNN_PLUGIN_STATUS_SUCCESS);
        for(const int64_t limit : {need - 1, int64_t{0}})
        {
            const auto config = engineConfigWithWorkspaceLimit(engine, limit);
            const hipdnnPluginConstData_t configData{config.data(), config.size()};
            const hipdnnPluginConstData_t graphData{serialized.data(), serialized.size()};
            hipdnnEnginePluginExecutionContext_t context = nullptr;
            if(abi.createContext(plugin, &configData, &graphData, &context)
               != HIPDNN_PLUGIN_STATUS_SUCCESS)
            {
                continue; // Declined at plan build.
            }
            size_t used = 0;
            EXPECT_EQ(abi.workspaceSize(plugin, context, &used), HIPDNN_PLUGIN_STATUS_SUCCESS);
            EXPECT_LE(used, static_cast<size_t>(limit))
                << param.engine << " built a plan over global.workspace_size_limit";
            EXPECT_EQ(abi.destroyContext(plugin, context), HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
        EXPECT_EQ(abi.destroy(plugin), HIPDNN_PLUGIN_STATUS_SUCCESS);
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvRobustness,
                         ::testing::ValuesIn(workspaceCases()),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
