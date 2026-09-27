// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx950ConvFwd";
constexpr const char* TILE_K_KNOB = "tile_k";

struct ConvProblem
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
    int64_t hi = 14;
    int64_t wi = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padH = 1;
    int64_t padW = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
};

struct Variant
{
    std::string name;
    DataType dtype;
    int64_t tileK;
    ConvProblem problem{};
};

std::vector<Variant> correctnessVariants()
{
    // Same five spatial cases, two dtypes and two tile sizes as the initial
    // verification requests in gfx950_conv_fwd.requests.json.
    const std::vector<std::pair<std::string, ConvProblem>> problems{
        {"Smoke", {}},
        {"Pointwise", {1, 32, 64, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1}},
        {"Strided", {2, 32, 64, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1}},
        {"Dilated", {1, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2}},
        {"Nonsquare", {2, 32, 64, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1}}};
    std::vector<Variant> variants;
    for(const auto& [name, problem] : problems)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            for(const auto tileK : {int64_t{64}, int64_t{128}})
            {
                const auto dtypeName = dtype == DataType::HALF ? "Fp16" : "Bf16";
                variants.push_back(
                    {name + dtypeName + "TileK" + std::to_string(tileK), dtype, tileK, problem});
            }
        }
    }
    return variants;
}

std::shared_ptr<Graph>
    buildConvGraph(DataType dtype, const ConvProblem& problem = {}, bool channelsLast = true)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("gfx950_conv_fwd")
        .set_io_data_type(dtype)
        .set_intermediate_data_type(dtype)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({problem.n, problem.c, problem.hi, problem.wi})
        .set_stride(channelsLast ? std::vector<int64_t>{problem.c * problem.hi * problem.wi,
                                                        1,
                                                        problem.wi * problem.c,
                                                        problem.c}
                                 : std::vector<int64_t>{problem.c * problem.hi * problem.wi,
                                                        problem.hi * problem.wi,
                                                        problem.wi,
                                                        1})
        .set_data_type(dtype);
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({problem.k, problem.c, problem.y, problem.x})
        .set_stride({problem.c * problem.y * problem.x, 1, problem.x * problem.c, problem.c})
        .set_data_type(dtype);

    ConvFpropAttributes attributes;
    attributes.set_name("gfx950_conv_fwd")
        .set_padding({problem.padH, problem.padW})
        .set_stride({problem.strideH, problem.strideW})
        .set_dilation({problem.dilationH, problem.dilationW})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    auto y = graph->conv_fprop(x, w, attributes);
    // The frontend infers output dims/strides from the channels-last input.
    y->set_uid(3).set_name("Y").set_output(true).set_data_type(dtype);
    return graph;
}

KnobSetting enableBenchmarkingKnob()
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1});
}

size_t selectionCount(const LogRecorderBase& recorder)
{
    const auto logs = recorder.getRecordedLogs();
    return static_cast<size_t>(std::count_if(logs.begin(), logs.end(), [](const auto& log) {
        return log.message.find("benchmarking selected kernel") != std::string::npos;
    }));
}

class ScopedPluginLogCapture
{
public:
    explicit ScopedPluginLogCapture(void* userHandle)
        : _userHandle(userHandle)
    {
        const auto read = getGlobalLogLevel(_previousLevel);
        EXPECT_EQ(read.code, ErrorCode::OK) << read.err_msg;
        const auto registered = setCallback(HIPDNN_SEV_INFO);
        EXPECT_EQ(registered.code, ErrorCode::OK) << registered.err_msg;
        _registered = registered.code == ErrorCode::OK;
        const auto level = setGlobalLogLevel(HIPDNN_SEV_INFO);
        EXPECT_EQ(level.code, ErrorCode::OK) << level.err_msg;
    }

    ~ScopedPluginLogCapture()
    {
        if(_registered)
        {
            static_cast<void>(setCallback(HIPDNN_SEV_OFF));
        }
        static_cast<void>(setGlobalLogLevel(_previousLevel));
    }

    ScopedPluginLogCapture(const ScopedPluginLogCapture&) = delete;
    ScopedPluginLogCapture& operator=(const ScopedPluginLogCapture&) = delete;

    IsolatedLogRecorder& recorder() const
    {
        return _recorder;
    }

private:
    Error setCallback(hipdnnSeverity_t level) const
    {
        return setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                  level,
                                  LogCallbackMode::SYNC,
                                  _userHandle);
    }

    hipdnnSeverity_t _previousLevel = HIPDNN_SEV_OFF;
    void* _userHandle;
    bool _registered = false;
    mutable IsolatedLogRecorder _recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
};

} // namespace

class IntegrationGpuGfx950ConvFwd
    : public test_utilities::IntegrationGraphVerificationHarness<float, Variant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, Variant>;

    void SetUp() override
    {
        Base::SetUp();
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, _deviceId), hipSuccess);
        _arch = properties.gcnArchName;
        if(!hipdnn_plugin_sdk::archMatches(
               _arch, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        {
            GTEST_SKIP() << "The packaged convolution family requires gfx950";
        }
    }

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    void buildAndCompile(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> engineIds;
        result = graph.get_ranked_engine_ids(engineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(engineIds.begin(), engineIds.end(), engineId()), engineIds.end())
            << "The installed plugin did not offer " << ENGINE_NAME;

        // Hard selection: a preferred-engine hint can fall back silently to another
        // provider. This path must execute this engine even when other engines match.
        result = graph.create_execution_plan_ext(engineId(), knobs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t selected = 0;
        result = graph.get_execution_plan_engine_id(selected);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(selected, engineId());
        int64_t workspaceSize = -1;
        result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(workspaceSize, 0);
    }

    void executeAndVerifyStorage(Graph& graph, DataType dtype, unsigned int seed)
    {
        auto [serialized, serializationError] = graph.to_binary();
        ASSERT_TRUE(serializationError.is_good()) << serializationError.get_message();
        const auto wrapper
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                serialized.data(), serialized.size());
        GraphTensorBundle gpu(wrapper.getTensorMap());
        GraphTensorBundle cpu(wrapper.getTensorMap());
        for(const auto uid : {int64_t{1}, int64_t{2}})
        {
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            gpu.randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
            cpu.randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
        }
        gpu.getTensor(3).fillWithSentinelValue();
        cpu.getTensor(3).fillWithSentinelValue();
        auto buffers = gpu.toDeviceVariantPack();
        const auto result = graph.execute(_handle, buffers, nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        CpuReferenceGraphExecutor().execute(
            serialized.data(), serialized.size(), cpu.toHostVariantPack());
        gpu.getTensor(3).markDeviceModified();

        // FP32 accumulation followed by a low-precision store. Absolute slack covers
        // cancellation/order differences in accumulation; relative slack covers
        // two output ulps. Inputs are bounded to [-1,1] and the reference uses FP32.
        const float relativeTolerance = dtype == DataType::HALF ? 0.002F : 0.016F;
        const auto validator
            = createAllCloseValidator(frontendToSdkDataType(dtype), 0.001F, relativeTolerance);
        EXPECT_TRUE(validator->allClose(cpu.getTensor(3), gpu.getTensor(3)));
    }

    std::optional<ingestor::WinnerRecord> persistedRanking() const
    {
        const auto path = ingestor::winnerCacheShardPath(ENGINE_NAME, _arch);
        std::ifstream stream(path);
        std::string line;
        std::optional<ingestor::WinnerRecord> ranking;
        while(std::getline(stream, line))
        {
            if(const auto entry = ingestor::decodeWinnerRecordLine(line))
            {
                // Every parameterized test owns a fresh cache and benchmarks one
                // graph only. A second record here indicates leaked test state.
                EXPECT_FALSE(ranking.has_value());
                ranking = entry->second;
            }
        }
        return ranking;
    }

    // The inherited harness opens the plugin relative to the running binary. No
    // descriptor override is set: installed tests must load installed kpack archives.
    ScopedTestCacheDir _cacheDir{"gfx950-conv", ScopedTestCacheDir::Scope::TEST};
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    std::string _arch;
};

TEST_P(IntegrationGpuGfx950ConvFwd, ForcedVariantMatchesCpuReference)
{
    const auto& variant = GetParam();
    auto graph = buildConvGraph(variant.dtype, variant.problem);
    const std::vector<KnobSetting> knobs{
        KnobSetting(TILE_K_KNOB, variant.tileK),
        KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{0})};
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, knobs));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 3));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 11));
}

TEST_F(IntegrationGpuGfx950ConvFwd, CatalogExposesBothTileKVariantsAndDefault)
{
    auto graph = buildConvGraph(DataType::HALF);
    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<Knob> knobs;
    result = graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const auto tileK = std::find_if(
        knobs.begin(), knobs.end(), [](const auto& knob) { return knob.knobId() == TILE_K_KNOB; });
    ASSERT_NE(tileK, knobs.end());
    const auto* constraint = dynamic_cast<const IntConstraint*>(tileK->constraint());
    ASSERT_NE(constraint, nullptr);
    EXPECT_EQ(constraint->getValidValues(), (std::unordered_set<int64_t>{64, 128}));
    const auto* defaultValue = std::get_if<int64_t>(&tileK->defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 64);
}

TEST_F(IntegrationGpuGfx950ConvFwd, DeclinesChannelsFirstStorage)
{
    auto graph = buildConvGraph(DataType::HALF, {}, false);
    const auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<int64_t> engineIds;
    const auto ranked = graph->get_ranked_engine_ids(engineIds);
    EXPECT_TRUE(ranked.code == ErrorCode::OK || ranked.code == ErrorCode::GRAPH_NOT_SUPPORTED)
        << ranked.err_msg;
    EXPECT_EQ(std::find(engineIds.begin(), engineIds.end(), engineId()), engineIds.end());
}

class IntegrationGpuGfx950ConvFwdBenchmark : public IntegrationGpuGfx950ConvFwd
{
};

TEST_P(IntegrationGpuGfx950ConvFwdBenchmark, MeasuresTwoCandidatesAndReusesFastestRecordedWinner)
{
    const auto dtype = GetParam().dtype;
    // Exercise both the public knob and the process override requested by the runbook.
    _forceBenchmarking.setValue("1");
    ASSERT_FALSE(_cacheDir.path().empty());
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    auto graph = buildConvGraph(dtype);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark 2 candidate(s)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, dtype, 19));
    ASSERT_EQ(selectionCount(recorder), 1U) << recorder.getRecordedLogsAsString();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), 2U);
    EXPECT_NE((*ranking)[0].kernelId, (*ranking)[1].kernelId);
    for(const auto& candidate : *ranking)
    {
        EXPECT_TRUE(std::isfinite(candidate.timeMs));
        EXPECT_GT(candidate.timeMs, 0.0);
    }
    EXPECT_LE((*ranking)[0].timeMs, (*ranking)[1].timeMs);
    const std::string winner = ingestor::toString(ranking->front().kernelId);
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel " + winner))
        << recorder.getRecordedLogsAsString();

    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, dtype, 23));
    EXPECT_EQ(selectionCount(recorder), 1U);

    graph.reset();
    recorder.clearLogs();
    auto recreated = buildConvGraph(dtype);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("served kernel " + winner + " at rank 0"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(
        recorder.hasLogContaining("from a benchmarked record of 2 entry(s) for 2 candidate(s)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*recreated, dtype, 29));
    EXPECT_EQ(selectionCount(recorder), 0U);
}

INSTANTIATE_TEST_SUITE_P(Correctness,
                         IntegrationGpuGfx950ConvFwd,
                         ::testing::ValuesIn(correctnessVariants()),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Autotuning,
                         IntegrationGpuGfx950ConvFwdBenchmark,
                         ::testing::Values(Variant{"Fp16", DataType::HALF, 64},
                                           Variant{"Bf16", DataType::BFLOAT16, 64}),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
