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
#include <tuple>
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
constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
constexpr int64_t DIRECT_DEPTHWISE_FAMILY = 1;
// Both tile_k arms of every catalog request, against which a direct-capable
// graph's candidate counts are stated.
constexpr size_t IMPLICIT_GEMM_ARMS = 2;

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
    // hipDNN carries no group attribute: groups = X channels / W dims[1].
    int64_t groups = 1;
};

struct Variant
{
    std::string name;
    DataType dtype;
    int64_t tileK;
    ConvProblem problem{};
    // Direct depthwise kernels the catalog packs for this graph; 0 when the
    // graph is outside the direct family's guard.
    size_t directArms = 0;
};

std::string dtypeSuffix(DataType dtype)
{
    return dtype == DataType::HALF ? "Fp16" : "Bf16";
}

std::vector<Variant> correctnessVariants()
{
    // The same synthetic spatial, grouped and depthwise cases, two dtypes and two
    // tile sizes as the verification requests in gfx950_conv_fwd.requests.json.
    // Grouped pointwise (1x1, stride 1, no padding) is outside the catalog.
    const std::vector<std::pair<std::string, ConvProblem>> problems{
        {"Smoke", {}},
        {"Pointwise", {1, 32, 64, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1}},
        {"Strided", {2, 32, 64, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1}},
        {"Dilated", {1, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2}},
        {"Nonsquare", {2, 32, 64, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1}},
        {"GroupedG2", {2, 16, 16, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 2}},
        {"GroupedG4Cpg3", {2, 12, 24, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}},
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}},
        {"DepthwiseStrided", {2, 32, 32, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 32}},
        // K / groups is odd, so the kernel uses the default (non-cshuffle) epilogue.
        {"DepthwiseOdd7x7", {2, 5, 5, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 5}},
        {"DepthwiseMultiplier2", {2, 16, 32, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}},
        {"GroupedG2Dilated", {2, 16, 16, 15, 15, 5, 5, 1, 1, 4, 4, 2, 2, 2}},
        // Depthwise graphs the direct family declines: implicit GEMM serves them.
        {"DepthwisePad0", {2, 24, 24, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 24}},
        {"DepthwisePad2", {2, 24, 24, 12, 12, 3, 3, 1, 1, 2, 2, 1, 1, 24}},
        {"DepthwiseStride2x1", {2, 24, 24, 13, 13, 3, 3, 2, 1, 1, 1, 1, 1, 24}},
        {"DepthwiseDilated", {2, 24, 24, 13, 13, 3, 3, 1, 1, 2, 2, 2, 2, 24}}};
    std::vector<Variant> variants;
    for(const auto& [name, problem] : problems)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            for(const auto tileK : {int64_t{64}, int64_t{128}})
            {
                variants.push_back({name + dtypeSuffix(dtype) + "TileK" + std::to_string(tileK),
                                    dtype,
                                    tileK,
                                    problem});
            }
        }
    }
    return variants;
}

std::vector<Variant>
    withBothDtypes(const std::vector<std::tuple<std::string, ConvProblem, size_t>>& problems)
{
    std::vector<Variant> variants;
    for(const auto& [name, problem, directArms] : problems)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back({name + dtypeSuffix(dtype), dtype, 0, problem, directArms});
        }
    }
    return variants;
}

std::vector<Variant> directVariants()
{
    // Every depthwise request in gfx950_conv_fwd.requests.json that carries
    // _catalog_direct arms, with its arm count. N is 2 throughout, so a write
    // that spills past the last output row lands in the next image and fails.
    return withBothDtypes({
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 2},
        // Stride 2 on an odd input height.
        {"DepthwiseStrided", {2, 32, 32, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 32}, 2},
        {"DepthwiseOdd7x7", {2, 5, 5, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 5}, 2},
        // 96 channels leave the last 64-lane channel tile partly empty.
        {"DepthwiseC96", {2, 96, 96, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 96}, 2},
        {"DepthwiseC96StridedOddH", {2, 96, 96, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 1},
        {"DepthwiseC96StridedEvenH", {2, 96, 96, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 1},
        // The block_w=32 arm of each 70x40 graph takes rocKE's runtime H loop.
        {"DepthwiseC64RuntimeLoop", {2, 64, 64, 70, 40, 3, 3, 1, 1, 1, 1, 1, 1, 64}, 2},
        {"DepthwiseC64StridedRuntimeLoop", {2, 64, 64, 70, 40, 3, 3, 2, 2, 1, 1, 1, 1, 64}, 2},
        // Spatial variant on the runtime H loop.
        {"DepthwiseC5Spatial17x17RuntimeLoop", {2, 5, 5, 56, 24, 17, 17, 1, 1, 8, 8, 1, 1, 5}, 1},
    });
}

std::vector<Variant> declinedDirectVariants()
{
    // Catalog depthwise graphs outside the direct family's guard: rocKE's direct
    // kernels would skip or misplace output rows for the first two (row coverage),
    // and do not implement the other three.
    return withBothDtypes({
        {"DepthwisePad0", {2, 24, 24, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 24}, 0},
        {"DepthwisePad2", {2, 24, 24, 12, 12, 3, 3, 1, 1, 2, 2, 1, 1, 24}, 0},
        {"DepthwiseStride2x1", {2, 24, 24, 13, 13, 3, 3, 2, 1, 1, 1, 1, 1, 24}, 0},
        {"DepthwiseDilated", {2, 24, 24, 13, 13, 3, 3, 1, 1, 2, 2, 2, 2, 24}, 0},
        {"DepthwiseMultiplier2", {2, 16, 32, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 0},
    });
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
    // Filter [K, C/groups, Y, X] stored channels-last (KYXC over the group's channels).
    const auto cpg = problem.c / problem.groups;
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({problem.k, cpg, problem.y, problem.x})
        .set_stride({cpg * problem.y * problem.x, 1, problem.x * cpg, cpg})
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

KnobSetting disableBenchmarkingKnob()
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{0});
}

KnobSetting kernelFamilyKnob(int64_t family)
{
    return KnobSetting(KERNEL_FAMILY_KNOB, family);
}

std::string candidateCount(size_t candidates)
{
    return std::to_string(candidates) + " candidate(s)";
}

// The kernel the last non-benchmarked plan build selected, or empty.
std::string selectedKernel(const LogRecorderBase& recorder)
{
    const std::string marker = "' selected kernel ";
    std::string selected;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at == std::string::npos)
        {
            continue;
        }
        const auto begin = at + marker.size();
        selected = log.message.substr(begin, log.message.find(' ', begin) - begin);
    }
    return selected;
}

const Knob* findKnob(const std::vector<Knob>& knobs, const std::string& name)
{
    const auto found = std::find_if(
        knobs.begin(), knobs.end(), [&](const auto& knob) { return knob.knobId() == name; });
    return found == knobs.end() ? nullptr : &*found;
}

void expectIntKnob(const std::vector<Knob>& knobs,
                   const std::string& name,
                   const std::unordered_set<int64_t>& values,
                   int64_t defaultValue)
{
    const auto* knob = findKnob(knobs, name);
    ASSERT_NE(knob, nullptr) << name;
    const auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
    ASSERT_NE(constraint, nullptr) << name;
    EXPECT_EQ(constraint->getValidValues(), values) << name;
    const auto* value = std::get_if<int64_t>(&knob->defaultValue());
    ASSERT_NE(value, nullptr) << name;
    EXPECT_EQ(*value, defaultValue) << name;
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

    // The last record line in this test's shard, key included. Later lines
    // supersede earlier ones for the same key.
    std::optional<std::pair<ingestor::WinnerKey, ingestor::WinnerRecord>>
        lastPersistedRecord() const
    {
        std::ifstream stream(ingestor::winnerCacheShardPath(ENGINE_NAME, _arch));
        std::string line;
        std::optional<std::pair<ingestor::WinnerKey, ingestor::WinnerRecord>> last;
        while(std::getline(stream, line))
        {
            if(auto entry = ingestor::decodeWinnerRecordLine(line))
            {
                last = std::move(entry);
            }
        }
        return last;
    }

    // Appends @p record for @p key to this test's shard through the shard's own
    // lock, as the engine's write-back does.
    void appendPersistedRecord(const ingestor::WinnerKey& key,
                               const ingestor::WinnerRecord& record) const
    {
        auto [shard, opened] = ingestor::openWinnerCacheShard(ENGINE_NAME, _arch);
        ASSERT_EQ(opened, hipdnn_data_sdk::utilities::LineStoreStatus::OK);
        ASSERT_TRUE(shard.has_value());
        ASSERT_EQ(hipdnn_data_sdk::utilities::lockLineStore(*shard),
                  hipdnn_data_sdk::utilities::LineStoreStatus::OK);
        const auto appended = hipdnn_data_sdk::utilities::appendLine(
            *shard, ingestor::encodeWinnerRecordLine(key, record));
        hipdnn_data_sdk::utilities::unlockLineStore(*shard);
        ASSERT_EQ(appended, hipdnn_data_sdk::utilities::LineStoreStatus::OK);
    }

    // Destroying the only handle releases the plugin's engines, so the next
    // handle's engine reads this test's shard afresh instead of the ranking the
    // previous engine holds in memory.
    void recreateHandle()
    {
        ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        _handle = nullptr;
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
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
    ASSERT_NO_FATAL_FAILURE(expectIntKnob(knobs, TILE_K_KNOB, {64, 128}, 64));
    // A dense graph is outside the direct depthwise family.
    ASSERT_NO_FATAL_FAILURE(
        expectIntKnob(knobs, KERNEL_FAMILY_KNOB, {IMPLICIT_GEMM_FAMILY}, IMPLICIT_GEMM_FAMILY));
}

TEST_F(IntegrationGpuGfx950ConvFwd, DepthwiseCatalogExposesBothFamiliesAndDirectDefault)
{
    auto graph = buildConvGraph(DataType::HALF, {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32});
    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<Knob> knobs;
    result = graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    // Direct kernels rank first unbenchmarked. Known quirk, pinned so a change is seen:
    // their tile_k placeholder 0 is advertised too, because knob choices come from
    // metadata; kernel_family is the supported way to choose a family.
    ASSERT_NO_FATAL_FAILURE(expectIntKnob(knobs, TILE_K_KNOB, {0, 64, 128}, 0));
    ASSERT_NO_FATAL_FAILURE(expectIntKnob(knobs,
                                          KERNEL_FAMILY_KNOB,
                                          {IMPLICIT_GEMM_FAMILY, DIRECT_DEPTHWISE_FAMILY},
                                          DIRECT_DEPTHWISE_FAMILY));
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

class IntegrationGpuGfx950ConvFwdDirect : public IntegrationGpuGfx950ConvFwd
{
};

TEST_P(IntegrationGpuGfx950ConvFwdDirect, UnforcedPlanServesTheDirectFamily)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + IMPLICIT_GEMM_ARMS;

    auto forced = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *forced, {kernelFamilyKnob(DIRECT_DEPTHWISE_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.directArms)
                                          + " (" + std::to_string(catalog)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto direct = selectedKernel(recorder);
    ASSERT_FALSE(direct.empty()) << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*forced, variant.dtype, 3));
    forced.reset();

    recorder.clearLogs();
    auto implicitGemm = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *implicitGemm, {kernelFamilyKnob(IMPLICIT_GEMM_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(IMPLICIT_GEMM_ARMS)))
        << recorder.getRecordedLogsAsString();
    EXPECT_NE(selectedKernel(recorder), direct);
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*implicitGemm, variant.dtype, 5));
    implicitGemm.reset();

    recorder.clearLogs();
    auto unforced = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*unforced, {disableBenchmarkingKnob()}));
    EXPECT_EQ(selectedKernel(recorder), direct) << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*unforced, variant.dtype, 7));
}

TEST_P(IntegrationGpuGfx950ConvFwdDirect, EveryDirectArmMatchesCpuReference)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const std::vector<KnobSetting> knobs{kernelFamilyKnob(DIRECT_DEPTHWISE_FAMILY),
                                         enableBenchmarkingKnob()};

    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, knobs));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(variant.directArms)))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 13));
    graph.reset();

    const auto measured = lastPersistedRecord();
    ASSERT_TRUE(measured) << "Benchmarking did not persist its measured direct ranking";
    const auto& [key, ranking] = *measured;
    ASSERT_EQ(ranking.size(), variant.directArms);

    // Benchmarking executes only its winner. Each arm is served in turn from a
    // record that ranks it first, so every packed direct kernel, not just the
    // fastest, is checked against the reference.
    for(size_t arm = 0; arm < ranking.size(); ++arm)
    {
        auto pinned = ranking;
        std::rotate(pinned.begin(),
                    pinned.begin() + static_cast<std::ptrdiff_t>(arm),
                    pinned.begin() + static_cast<std::ptrdiff_t>(arm) + 1);
        ASSERT_NO_FATAL_FAILURE(appendPersistedRecord(key, pinned));
        ASSERT_NO_FATAL_FAILURE(recreateHandle());
        recorder.clearLogs();

        auto served = buildConvGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*served, knobs));
        const auto kernel = ingestor::toString(pinned.front().kernelId);
        EXPECT_TRUE(recorder.hasLogContaining("served kernel " + kernel + " at rank 0"))
            << "arm " << arm << ": " << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
        ASSERT_NO_FATAL_FAILURE(
            executeAndVerifyStorage(*served, variant.dtype, 17 + static_cast<unsigned int>(arm)));
        EXPECT_EQ(selectionCount(recorder), 0U);
    }
}

TEST_P(IntegrationGpuGfx950ConvFwdDirect, BenchmarkingComparesBothFamiliesAndReusesTheWinner)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + IMPLICIT_GEMM_ARMS;

    // One kernel from each family, as the unbenchmarked heuristic orders them.
    std::unordered_set<std::string> families;
    for(const auto family : {IMPLICIT_GEMM_FAMILY, DIRECT_DEPTHWISE_FAMILY})
    {
        recorder.clearLogs();
        auto forced = buildConvGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(
            buildAndCompile(*forced, {kernelFamilyKnob(family), disableBenchmarkingKnob()}));
        families.insert(selectedKernel(recorder));
    }
    ASSERT_EQ(families.size(), 2U);

    recorder.clearLogs();
    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(catalog)))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 19));
    ASSERT_EQ(selectionCount(recorder), 1U) << recorder.getRecordedLogsAsString();
    graph.reset();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), catalog);
    std::unordered_set<std::string> measured;
    for(const auto& candidate : *ranking)
    {
        EXPECT_TRUE(std::isfinite(candidate.timeMs));
        EXPECT_GT(candidate.timeMs, 0.0);
        measured.insert(ingestor::toString(candidate.kernelId));
    }
    EXPECT_EQ(measured.size(), catalog);
    for(const auto& kernel : families)
    {
        EXPECT_EQ(measured.count(kernel), 1U) << kernel;
    }
    const std::string winner = ingestor::toString(ranking->front().kernelId);
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel " + winner))
        << recorder.getRecordedLogsAsString();

    recorder.clearLogs();
    auto recreated = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("served kernel " + winner + " at rank 0"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*recreated, variant.dtype, 23));
    EXPECT_EQ(selectionCount(recorder), 0U);
}

class IntegrationGpuGfx950ConvFwdDirectDeclined : public IntegrationGpuGfx950ConvFwd
{
};

TEST_P(IntegrationGpuGfx950ConvFwdDirectDeclined, ImplicitGemmServesTheGraphAlone)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();

    auto probe = buildConvGraph(variant.dtype, variant.problem);
    auto result = probe->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<Knob> knobs;
    result = probe->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_NO_FATAL_FAILURE(expectIntKnob(knobs, TILE_K_KNOB, {64, 128}, 64));
    ASSERT_NO_FATAL_FAILURE(
        expectIntKnob(knobs, KERNEL_FAMILY_KNOB, {IMPLICIT_GEMM_FAMILY}, IMPLICIT_GEMM_FAMILY));

    // Forcing the direct family leaves no candidate.
    result = probe->create_execution_plan_ext(
        engineId(), {kernelFamilyKnob(DIRECT_DEPTHWISE_FAMILY), disableBenchmarkingKnob()});
    if(result.code == ErrorCode::OK)
    {
        result = probe->check_support();
    }
    if(result.code == ErrorCode::OK)
    {
        result = probe->build_plans();
    }
    EXPECT_NE(result.code, ErrorCode::OK);
    probe.reset();

    recorder.clearLogs();
    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(IMPLICIT_GEMM_ARMS)
                                          + " (" + std::to_string(IMPLICIT_GEMM_ARMS)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 3));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 11));
}

INSTANTIATE_TEST_SUITE_P(Correctness,
                         IntegrationGpuGfx950ConvFwd,
                         ::testing::ValuesIn(correctnessVariants()),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(DirectDepthwise,
                         IntegrationGpuGfx950ConvFwdDirect,
                         ::testing::ValuesIn(directVariants()),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(DirectDepthwiseDeclined,
                         IntegrationGpuGfx950ConvFwdDirectDeclined,
                         ::testing::ValuesIn(declinedDirectVariants()),
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
