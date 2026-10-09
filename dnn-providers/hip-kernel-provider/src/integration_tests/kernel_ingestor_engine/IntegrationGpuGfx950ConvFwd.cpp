// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
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
#include <nlohmann/json.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

namespace
{

// Each engine's suites compile only where that engine ships: HKP_TEST_GFX950_CONV_FWD for
// the exact-shape engine, HKP_TEST_GFX950_CONV_FWD_AOT for the AOT-instance engine and
// HKP_TEST_GFX950_CONV_FWD_AOT_SEED for its test-only seed (src/integration_tests/CMakeLists.txt).
constexpr const char* ENGINE_NAME = "hipkernel:Gfx950ConvFwd";
constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
constexpr int64_t DIRECT_DEPTHWISE_FAMILY = 1;
#ifdef HKP_TEST_GFX950_CONV_FWD
constexpr const char* TILE_K_KNOB = "tile_k";
// rocKE's AOT kernels take the problem as kernel arguments, so a graph is served by every
// packaged kernel whose build-time capabilities admit it, not only the one compiled for
// its shape. Every count below is the number of shipped kernels that
// gfx950_conv_fwd_serves in convolution_forward.py accepts for the graph, so the native
// matcher is checked against the Python one.
constexpr size_t SMOKE_CANDIDATES = 42;
#endif

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
    // Direct depthwise kernels that serve this graph; 0 when the graph is outside the
    // direct family's guard.
    size_t directArms = 0;
    // Implicit-GEMM kernels that serve this graph, both tile_k arms.
    size_t implicitArms = 0;
};

std::string dtypeSuffix(DataType dtype)
{
    return dtype == DataType::HALF ? "Fp16" : "Bf16";
}

#ifdef HKP_TEST_GFX950_CONV_FWD
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

std::vector<Variant> withBothDtypes(
    const std::vector<std::tuple<std::string, ConvProblem, size_t, size_t>>& problems)
{
    std::vector<Variant> variants;
    for(const auto& [name, problem, directArms, implicitArms] : problems)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back(
                {name + dtypeSuffix(dtype), dtype, 0, problem, directArms, implicitArms});
        }
    }
    return variants;
}

std::vector<Variant> directVariants()
{
    // Every depthwise request in gfx950_conv_fwd.requests.json that carries
    // _catalog_direct arms. A direct kernel serves every graph with its compiled filter,
    // stride and padding, so the arm counts
    // include kernels built for other images and channel counts. N is 2 throughout, so a
    // write that spills past the last output row lands in the next image and fails.
    return withBothDtypes({
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 6, 26},
        // Stride 2 on an odd input height.
        {"DepthwiseStrided", {2, 32, 32, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 32}, 6, 26},
        {"DepthwiseOdd7x7", {2, 5, 5, 9, 9, 7, 7, 1, 1, 3, 3, 1, 1, 5}, 2, 26},
        // 96 channels leave the last 64-lane channel tile partly empty.
        {"DepthwiseC96", {2, 96, 96, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 96}, 5, 26},
        {"DepthwiseC96StridedOddH", {2, 96, 96, 17, 17, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 5, 26},
        {"DepthwiseC96StridedEvenH", {2, 96, 96, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 5, 26},
        // The block_w=32 arm of each 70x40 graph takes rocKE's runtime H loop.
        {"DepthwiseC64RuntimeLoop", {2, 64, 64, 70, 40, 3, 3, 1, 1, 1, 1, 1, 1, 64}, 5, 26},
        {"DepthwiseC64StridedRuntimeLoop", {2, 64, 64, 70, 40, 3, 3, 2, 2, 1, 1, 1, 1, 64}, 5, 26},
        // Spatial variant on the runtime H loop.
        {"DepthwiseC5Spatial17x17RuntimeLoop",
         {2, 5, 5, 56, 24, 17, 17, 1, 1, 8, 8, 1, 1, 5},
         1,
         26},
    });
}

std::vector<Variant> declinedDirectVariants()
{
    // Catalog depthwise graphs outside the direct family's guard: rocKE's direct
    // kernels would skip or misplace output rows for the first two (row coverage),
    // and do not implement the other three.
    return withBothDtypes({
        {"DepthwisePad0", {2, 24, 24, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 24}, 0, 26},
        {"DepthwisePad2", {2, 24, 24, 12, 12, 3, 3, 1, 1, 2, 2, 1, 1, 24}, 0, 26},
        {"DepthwiseStride2x1", {2, 24, 24, 13, 13, 3, 3, 2, 1, 1, 1, 1, 1, 24}, 0, 26},
        {"DepthwiseDilated", {2, 24, 24, 13, 13, 3, 3, 1, 1, 2, 2, 2, 2, 24}, 0, 26},
        {"DepthwiseMultiplier2", {2, 16, 32, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 0, 30},
    });
}

std::vector<Variant> uncatalogedVariants()
{
    // Graphs no packaged kernel was compiled for: every candidate is a kernel built for
    // another shape. They cover each capability the matcher checks: a grouped binary on a
    // dense graph, a non-pointwise binary on a 1x1 graph and on a grouped 1x1 graph,
    // narrowed load and store widths, and direct kernels on new images and group counts.
    return withBothDtypes({
        {"Dense", {3, 48, 40, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 42},
        {"Strided5x5", {2, 64, 96, 23, 19, 5, 5, 2, 2, 2, 2, 1, 1}, 0, 42},
        {"Pointwise", {4, 64, 48, 9, 11, 1, 1, 1, 1, 0, 0, 1, 1}, 0, 44},
        {"GroupedPointwise", {2, 64, 64, 10, 10, 1, 1, 1, 1, 0, 0, 1, 1, 8}, 0, 34},
        {"Grouped", {2, 48, 48, 15, 13, 3, 3, 1, 1, 1, 1, 1, 1, 3}, 0, 34},
        {"Depthwise", {3, 64, 64, 28, 20, 3, 3, 1, 1, 1, 1, 1, 1, 64}, 5, 26},
        {"DepthwiseStrided", {1, 96, 96, 31, 29, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 5, 26},
        {"Depthwise7x7", {2, 5, 5, 21, 15, 7, 7, 1, 1, 3, 3, 1, 1, 5}, 2, 26},
        // Spatial kernels built for 32 and 5 groups serve 16 and 12: the variant derives
        // its columns per wave from the runtime group count.
        {"DepthwiseG16", {2, 16, 16, 20, 18, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 6, 26},
        {"Depthwise7x7G12", {2, 12, 12, 15, 13, 7, 7, 1, 1, 3, 3, 1, 1, 12}, 2, 26},
    });
}
#endif

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
    return {KERNEL_FAMILY_KNOB, family};
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

#ifdef HKP_TEST_GFX950_CONV_FWD
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
#endif

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

    /// The engine under test; the AOT-instance suite overrides it.
    virtual const char* engineName() const
    {
        return ENGINE_NAME;
    }

    int64_t engineId() const
    {
        return hipdnn_data_sdk::utilities::engineNameToId(engineName());
    }

    void buildAndCompile(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> engineIds;
        result = graph.get_ranked_engine_ids(engineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(engineIds.begin(), engineIds.end(), engineId()), engineIds.end())
            << "The installed plugin did not offer " << engineName();

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

    // The engine keys its shard by its full identity (revision, UHD id, model hash), which
    // only the loaded descriptor set knows. Each test owns a fresh cache, so the engine's
    // one shard for this arch is found on disk instead:
    // <engine>/<uhd id>/<revision>-<model hash>/<arch>/winners.jsonl.
    std::filesystem::path winnerShardPath() const
    {
        const auto engineDir = hipdnn_data_sdk::utilities::cacheRoot() / "ingestor-winners"
                               / std::string(ingestor::winnerCacheVersion())
                               / hipdnn_data_sdk::utilities::sanitizeForPath(engineName());
        const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(_arch));
        std::vector<std::filesystem::path> shards;
        std::error_code error;
        for(auto it = std::filesystem::recursive_directory_iterator(engineDir, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            const auto& path = it->path();
            if(path.filename() == "winners.jsonl" && path.parent_path().filename() == arch)
            {
                shards.push_back(path);
            }
        }
        EXPECT_LE(shards.size(), 1U) << "more than one winner shard under " << engineDir;
        return shards.empty() ? std::filesystem::path{} : shards.front();
    }

    std::optional<ingestor::WinnerRecord> persistedRanking() const
    {
        const auto path = winnerShardPath();
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
        std::ifstream stream(winnerShardPath());
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
    // lock, as the engine's write-back does. The engine's measured record created it.
    void appendPersistedRecord(const ingestor::WinnerKey& key,
                               const ingestor::WinnerRecord& record) const
    {
        const auto path = winnerShardPath();
        ASSERT_FALSE(path.empty()) << "The engine has not written its winner shard";
        auto [shard, opened]
            = hipdnn_data_sdk::utilities::openLineStore(path, ingestor::winnerCacheVersion());
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

#ifdef HKP_TEST_GFX950_CONV_FWD
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

TEST_P(IntegrationGpuGfx950ConvFwdBenchmark, MeasuresEveryServingKernelAndReusesTheFastest)
{
    const auto dtype = GetParam().dtype;
    // Exercise both the public knob and the process override requested by the runbook.
    _forceBenchmarking.setValue("1");
    ASSERT_FALSE(_cacheDir.path().empty());
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    auto graph = buildConvGraph(dtype);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(SMOKE_CANDIDATES)))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, dtype, 19));
    ASSERT_EQ(selectionCount(recorder), 1U) << recorder.getRecordedLogsAsString();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), SMOKE_CANDIDATES);
    std::unordered_set<std::string> measured;
    for(const auto& candidate : *ranking)
    {
        EXPECT_TRUE(std::isfinite(candidate.timeMs));
        EXPECT_GT(candidate.timeMs, 0.0);
        EXPECT_LE(ranking->front().timeMs, candidate.timeMs);
        measured.insert(ingestor::toString(candidate.kernelId));
    }
    EXPECT_EQ(measured.size(), SMOKE_CANDIDATES);
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
    const auto count = std::to_string(SMOKE_CANDIDATES);
    EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count + " entry(s) for "
                                          + count + " candidate(s)"))
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
    const auto catalog = variant.directArms + variant.implicitArms;

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
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.implicitArms)))
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
    const auto catalog = variant.directArms + variant.implicitArms;

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
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.implicitArms)
                                          + " (" + std::to_string(variant.implicitArms)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 3));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 11));
}

#endif // HKP_TEST_GFX950_CONV_FWD

class IntegrationGpuGfx950ConvFwdServing : public IntegrationGpuGfx950ConvFwd
{
protected:
    /// Benchmarks the parameter's graph over every kernel that serves it, then serves each
    /// measured kernel in turn from a record that ranks it first. Benchmarking executes
    /// only its winner, so this is what runs every serving kernel against the reference.
    void verifyEveryServingKernel(unsigned int seed, unsigned int armSeed)
    {
        const auto& variant = GetParam();
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        const std::vector<KnobSetting> knobs{enableBenchmarkingKnob()};
        const auto candidates = variant.directArms + variant.implicitArms;

        auto graph = buildConvGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, knobs));
        EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(candidates)))
            << recorder.getRecordedLogsAsString();
        ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, seed));
        graph.reset();

        const auto measured = lastPersistedRecord();
        ASSERT_TRUE(measured) << "Benchmarking did not persist its measured ranking";
        const auto& [key, ranking] = *measured;
        ASSERT_EQ(ranking.size(), candidates);

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
                << "kernel " << arm << ": " << recorder.getRecordedLogsAsString();
            EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
            ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(
                *served, variant.dtype, armSeed + static_cast<unsigned int>(arm)));
        }
    }
};

#ifdef HKP_TEST_GFX950_CONV_FWD
// Every kernel here was compiled for another shape.
TEST_P(IntegrationGpuGfx950ConvFwdServing, EveryServingKernelMatchesCpuReference)
{
    verifyEveryServingKernel(43, 47);
}
#endif

#if defined(HKP_TEST_GFX950_CONV_FWD_AOT) || defined(HKP_TEST_GFX950_CONV_FWD_AOT_SEED)

// ---- The AOT-instance engines' graphs ----
//
// hipkernel:Gfx950ConvFwdAot ships every AOT instance of rocKE's sweep that the packaged
// launch runs, each built once against rocKE's probe problem and matched to a graph by
// capability. hipkernel:Gfx950ConvFwdAotSeed runs the same native code over a seed of
// them, per dtype 12 implicit-GEMM instances, 4 std and spatial direct depthwise
// instances, and 22 each of the tiled, col, grouped and non-grouped direct kernels
// (test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed). Both run the graphs of
// aotGraphs(), in both dtypes; the seed also runs aotSeedDirectGraphs().

namespace
{

/// One graph and how many instances of each family each catalog selects for it, as
/// gfx950_conv_fwd_aot_runtime_candidates in convolution_forward.py counts them over
/// gfx950_conv_fwd_aot_seed.instances.json and the full catalog's per-dtype lists,
/// gfx950_conv_fwd_aot_{fp16,bf16}.instances.json. Selected is rocKE's runtime selection
/// among the instances that serve the graph (only the widest vector widths present), so
/// it is the engine's candidate list, not its serving set. The seed and the direct family
/// are the same in both dtypes; the full catalog's implicit-GEMM count is stated per
/// dtype, since each dtype's list is rocKE's sweep for that dtype. The engines' kernel
/// matcher and candidate filter are checked against those counts, and
/// test_make_conv_aot_config.py recomputes this table from the lists: keep one row per
/// line.
struct AotGraph
{
    std::string name;
    ConvProblem problem;
    size_t seedDirect;
    size_t seedImplicit;
    size_t direct;
    size_t implicitFp16;
    size_t implicitBf16;
};

// A direct instance serves only graphs with its compiled filter, stride, padding and
// dilation; batch, image and group count are kernel arguments. In the seed: std 3x3/1 and
// 5x5/2, spatial (fewer than 64 groups) 3x3/1 and 7x7/2, and the tiled, col, grouped and
// non-grouped instances aotSeedDirectGraphs() is for, which also serve some graphs here
// (non-grouped the dense 3x3 ones whose C, K and Wo they tile).
// clang-format off
const std::vector<AotGraph>& aotGraphs()
{
    static const std::vector<AotGraph> s_graphs{
        {"Smoke", {}, 11, 4, 146, 3260, 3260},
        {"Pointwise", {1, 32, 64, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1}, 0, 4, 0, 3260, 3260},
        // An instance is built against a 3x3 probe, so it is never the pointwise GEMM.
        {"GroupedPointwiseG4", {2, 32, 32, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 4}, 0, 4, 0, 3260, 3260},
        {"Strided", {2, 32, 64, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1}, 1, 4, 110, 3260, 3260},
        {"Dilated", {1, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2}, 0, 4, 128, 3260, 3260},
        {"Nonsquare", {2, 32, 64, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1}, 0, 4, 0, 3260, 3260},
        // Three channels per group: only instances 1 wide along C/groups and K/groups.
        {"GroupedG4Cpg3", {2, 12, 24, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}, 0, 1, 0, 3720, 3720},
        // Eight channels per group: the grouped direct instances serve it too.
        {"GroupedG8Cpg8", {2, 64, 64, 13, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 3, 4, 16, 3260, 3260},
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 8, 1, 116, 7440, 7440},
        {"DepthwiseG5N3", {3, 5, 5, 9, 21, 3, 3, 1, 1, 1, 1, 1, 1, 5}, 8, 1, 116, 7440, 7440},
        // 64 groups and more: no spatial instance.
        {"DepthwiseG96", {3, 96, 96, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 96}, 7, 1, 113, 7440, 7440},
        {"DepthwiseG256N1", {1, 256, 256, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 256}, 7, 1, 113, 7440, 7440},
        {"DepthwiseG64Strided5x5", {2, 64, 64, 17, 17, 5, 5, 2, 2, 2, 2, 1, 1, 64}, 1, 1, 105, 7440, 7440},
        {"DepthwiseG48Strided5x5", {1, 48, 48, 23, 19, 5, 5, 2, 2, 2, 2, 1, 1, 48}, 1, 1, 108, 7440, 7440},
        {"DepthwiseG12Strided7x7", {2, 12, 12, 21, 15, 7, 7, 2, 2, 3, 3, 1, 1, 12}, 7, 1, 108, 7440, 7440},
        {"DepthwiseG40Strided7x7", {3, 40, 40, 16, 30, 7, 7, 2, 2, 3, 3, 1, 1, 40}, 7, 1, 108, 7440, 7440},
        // "Valid" padding: of the direct instances, only col serves it.
        {"DepthwiseG64Pad0", {2, 64, 64, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 64}, 1, 1, 27, 7440, 7440},
        // C = 12 admits load widths up to 4 and K = 20 store widths up to 4.
        {"C12K20", {2, 12, 20, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 3364, 3364},
        {"C24K40", {2, 24, 40, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 4, 0, 3260, 3260},
        {"Wide", {4, 64, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1}, 8, 4, 202, 3260, 3260},
        // Each selects a seed width pair that no row above selects:
        // 2/2/2, 4/4/2, 8/8/4, 1/1/4, and the default epilogue's 4/4/1 with cshuffle 2/2/1.
        {"C6K10", {2, 6, 10, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 3676, 3676},
        {"C12K6", {2, 12, 6, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 3364, 3364},
        {"C16K12", {2, 16, 12, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 3260, 3260},
        {"C3K8", {2, 3, 8, 15, 15, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 3720, 3720},
        {"C12K5", {2, 12, 5, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 2, 0, 6728, 6728},
    };
    return s_graphs;
}
// clang-format on

/// One graph for the seed's tiled, col, grouped and non-grouped direct instances, and how
/// many instances of each family the seed selects for it (the same in both dtypes),
/// counted as for aotGraphs(). Each such instance serves at least two of these graphs: images,
/// batches and group counts with tails, "valid" and "same" padding for col, every grouped
/// channel count, and every dilation the non-grouped sweep builds. Only the seed suite
/// runs them: the shipped catalog's suite samples its thousands of candidates per graph.
struct AotSeedGraph
{
    std::string name;
    ConvProblem problem;
    size_t seedDirect;
    size_t seedImplicit;
};

// clang-format off
const std::vector<AotSeedGraph>& aotSeedDirectGraphs()
{
    static const std::vector<AotSeedGraph> s_graphs{
        // Tiled depthwise: 3x3, 7x7, 15x15 and 31x31 filters, strides 1 and 2.
        {"TiledG32", {2, 32, 32, 13, 17, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 8, 1},
        {"TiledG200", {1, 200, 200, 9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 200}, 7, 1},
        {"TiledG5Strided", {3, 5, 5, 12, 12, 3, 3, 2, 2, 1, 1, 1, 1, 5}, 6, 1},
        {"TiledG96Strided", {1, 96, 96, 17, 14, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 6, 1},
        {"TiledG24Strided7x7", {2, 24, 24, 15, 15, 7, 7, 2, 2, 3, 3, 1, 1, 24}, 7, 1},
        {"TiledG130Strided7x7", {1, 130, 130, 21, 9, 7, 7, 2, 2, 3, 3, 1, 1, 130}, 6, 1},
        {"TiledG16Filter15", {1, 16, 16, 20, 23, 15, 15, 1, 1, 7, 7, 1, 1, 16}, 1, 1},
        {"TiledG70Filter15", {2, 70, 70, 16, 16, 15, 15, 1, 1, 7, 7, 1, 1, 70}, 1, 1},
        {"TiledG8Filter31", {1, 8, 8, 40, 37, 31, 31, 2, 2, 15, 15, 1, 1, 8}, 2, 1},
        {"TiledG65Filter31", {1, 65, 65, 33, 31, 31, 31, 2, 2, 15, 15, 1, 1, 65}, 2, 1},
        // Column-streamed depthwise: "valid" (padding 0) and "same" padding.
        {"ColG40Valid", {2, 40, 40, 11, 19, 3, 3, 1, 1, 0, 0, 1, 1, 40}, 1, 1},
        {"ColG7Valid", {3, 7, 7, 6, 9, 3, 3, 1, 1, 0, 0, 1, 1, 7}, 1, 1},
        {"ColG24Strided5x5Valid", {2, 24, 24, 13, 17, 5, 5, 2, 2, 0, 0, 1, 1, 24}, 5, 1},
        {"ColG100Strided5x5Valid", {1, 100, 100, 18, 21, 5, 5, 2, 2, 0, 0, 1, 1, 100}, 5, 1},
        {"ColG24Strided5x5", {2, 24, 24, 13, 17, 5, 5, 2, 2, 2, 2, 1, 1, 24}, 1, 1},
        {"ColG72Filter9", {1, 72, 72, 20, 16, 9, 9, 1, 1, 4, 4, 1, 1, 72}, 10, 1},
        {"ColG9Filter9", {2, 9, 9, 13, 11, 9, 9, 1, 1, 4, 4, 1, 1, 9}, 10, 1},
        {"ColG9Strided11x11Valid", {2, 9, 9, 25, 27, 11, 11, 2, 2, 0, 0, 1, 1, 9}, 2, 1},
        {"ColG64Strided11x11Valid", {1, 64, 64, 31, 23, 11, 11, 2, 2, 0, 0, 1, 1, 64}, 2, 1},
        {"ColG12Filter7Valid", {2, 12, 12, 16, 18, 7, 7, 1, 1, 0, 0, 1, 1, 12}, 1, 1},
        {"ColG80Filter7Valid", {1, 80, 80, 9, 20, 7, 7, 1, 1, 0, 0, 1, 1, 80}, 1, 1},
        {"ColG20Strided9x9", {1, 20, 20, 23, 19, 9, 9, 2, 2, 4, 4, 1, 1, 20}, 4, 1},
        {"ColG66Strided9x9", {2, 66, 66, 17, 25, 9, 9, 2, 2, 4, 4, 1, 1, 66}, 4, 1},
        // Grouped, 4, 8, 16 and 32 channels per group.
        {"GroupedG16Cpg4", {2, 64, 64, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 4, 1},
        {"GroupedG32Cpg4", {1, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 4, 1},
        {"GroupedG16Cpg8", {2, 128, 128, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 4, 4},
        {"GroupedG48Cpg8", {1, 384, 384, 7, 33, 3, 3, 1, 1, 1, 1, 1, 1, 48}, 4, 4},
        {"GroupedG16Cpg16", {1, 256, 256, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 8, 4},
        {"GroupedG32Cpg16", {2, 512, 512, 8, 17, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 8, 4},
        {"GroupedG16Cpg32", {1, 512, 512, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 6, 4},
        {"GroupedG8Cpg32", {2, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 6, 4},
        // Non-grouped: output widths every swept tile width divides, row and channel tails,
        // stride 2 and the swept dilations.
        {"NongroupedC64K96", {1, 64, 96, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 9, 4},
        {"NongroupedC32K64", {2, 32, 64, 7, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 7, 4},
        {"NongroupedC128K32", {1, 128, 32, 5, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 8, 4},
        {"NongroupedWo192", {1, 32, 32, 3, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 14, 4},
        {"NongroupedWo160", {2, 64, 64, 2, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 10, 4},
        {"NongroupedWo128", {1, 64, 128, 4, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 10, 4},
        {"NongroupedWo384", {1, 16, 32, 2, 384, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 11, 4},
        {"NongroupedWo80", {2, 32, 64, 3, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 8, 4},
        {"NongroupedWo320", {1, 16, 32, 2, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 9, 4},
        {"NongroupedC64K64Strided", {1, 64, 64, 13, 191, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 1, 4},
        {"NongroupedC48K96Strided", {2, 48, 96, 9, 95, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 1, 4},
        {"NongroupedDilated2x2", {1, 64, 64, 12, 96, 3, 3, 1, 1, 2, 2, 2, 2, 1}, 1, 4},
        {"NongroupedDilated2x2C32", {2, 32, 32, 7, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1}, 1, 4},
        {"NongroupedDilated3x3", {1, 64, 64, 8, 96, 3, 3, 1, 1, 3, 3, 3, 3, 1}, 1, 4},
        {"NongroupedDilated3x3C32", {2, 32, 32, 9, 32, 3, 3, 1, 1, 3, 3, 3, 3, 1}, 1, 4},
        {"NongroupedDilated4x4", {1, 64, 64, 10, 96, 3, 3, 1, 1, 4, 4, 4, 4, 1}, 1, 4},
        {"NongroupedDilated4x4C32", {2, 32, 32, 9, 32, 3, 3, 1, 1, 4, 4, 4, 4, 1}, 1, 4},
        {"NongroupedDilated3x1", {1, 64, 64, 9, 96, 3, 3, 1, 1, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1C32", {2, 32, 32, 11, 32, 3, 3, 1, 1, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1Strided", {1, 64, 96, 20, 191, 3, 3, 2, 2, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1StridedC32", {2, 32, 32, 13, 63, 3, 3, 2, 2, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated1x2", {1, 64, 64, 5, 96, 3, 3, 1, 1, 2, 2, 1, 2, 1}, 1, 4},
        {"NongroupedDilated1x2C32", {2, 32, 32, 6, 32, 3, 3, 1, 1, 2, 2, 1, 2, 1}, 1, 4},
        {"NongroupedDilated1x4", {1, 32, 32, 9, 96, 3, 3, 1, 1, 4, 4, 1, 4, 1}, 1, 4},
        {"NongroupedDilated1x4C64", {1, 64, 96, 6, 48, 3, 3, 1, 1, 4, 4, 1, 4, 1}, 1, 4},
    };
    return s_graphs;
}
// clang-format on

/// aotGraphs() in both dtypes, with the seed's selected counts or the full catalog's; the
/// seed's also run aotSeedDirectGraphs().
std::vector<Variant> aotVariants(bool fullCatalog)
{
    std::vector<Variant> variants;
    for(const auto& graph : aotGraphs())
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            const auto implicit = dtype == DataType::HALF ? graph.implicitFp16 : graph.implicitBf16;
            variants.push_back({graph.name + dtypeSuffix(dtype),
                                dtype,
                                0,
                                graph.problem,
                                fullCatalog ? graph.direct : graph.seedDirect,
                                fullCatalog ? implicit : graph.seedImplicit});
        }
    }
    if(!fullCatalog)
    {
        for(const auto& graph : aotSeedDirectGraphs())
        {
            for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
            {
                variants.push_back({graph.name + dtypeSuffix(dtype),
                                    dtype,
                                    0,
                                    graph.problem,
                                    graph.seedDirect,
                                    graph.seedImplicit});
            }
        }
    }
    return variants;
}

} // namespace

#endif

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT_SEED

// ---- hipkernel:Gfx950ConvFwdAotSeed: every selected AOT instance, pinned ----
//
// The seed is small enough that every instance selected for a graph is pinned and checked
// against the reference, which the full catalog's thousands per graph do not afford.
// Every seed instance is selected for at least one graph of aotGraphs().

namespace
{

constexpr const char* AOT_SEED_ENGINE_NAME = "hipkernel:Gfx950ConvFwdAotSeed";

/// Whether @p problem is pure depthwise (groups == C == K): the graphs the std, spatial,
/// tiled and col kernels serve, where an unbenchmarked plan prefers a direct instance.
bool isDepthwise(const ConvProblem& problem)
{
    return problem.groups == problem.c && problem.c == problem.k;
}

} // namespace

class IntegrationGpuGfx950ConvFwdAotSeed : public IntegrationGpuGfx950ConvFwdServing
{
protected:
    const char* engineName() const override
    {
        return AOT_SEED_ENGINE_NAME;
    }
};

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, EverySelectedInstanceMatchesCpuReference)
{
    verifyEveryServingKernel(31, 37);
}

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, UnbenchmarkedPlanServesACapableInstance)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();

    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {disableBenchmarkingKnob()}));
    EXPECT_FALSE(selectedKernel(recorder).empty()) << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 41));
}

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, KernelFamilyKnobSelectsEachFamilysInstances)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + variant.implicitArms;

    auto implicitGemm = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *implicitGemm, {kernelFamilyKnob(IMPLICIT_GEMM_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.implicitArms)
                                          + " (" + std::to_string(catalog)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto implicitKernel = selectedKernel(recorder);
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*implicitGemm, variant.dtype, 43));
    implicitGemm.reset();
    if(variant.directArms == 0)
    {
        return;
    }

    recorder.clearLogs();
    auto direct = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *direct, {kernelFamilyKnob(DIRECT_DEPTHWISE_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.directArms)
                                          + " (" + std::to_string(catalog)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto directKernel = selectedKernel(recorder);
    EXPECT_NE(directKernel, implicitKernel);
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*direct, variant.dtype, 47));
    direct.reset();

    // Unbenchmarked, a depthwise direct instance ranks above every implicit-GEMM one; the
    // grouped and non-grouped direct kernels rank below them (aotScore).
    recorder.clearLogs();
    auto unforced = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*unforced, {disableBenchmarkingKnob()}));
    EXPECT_EQ(selectedKernel(recorder),
              isDepthwise(variant.problem) || variant.implicitArms == 0 ? directKernel
                                                                        : implicitKernel)
        << recorder.getRecordedLogsAsString();
}

#endif // HKP_TEST_GFX950_CONV_FWD_AOT_SEED

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT

// ---- hipkernel:Gfx950ConvFwdAot: the shipped catalog ----
//
// At full size thousands of instances are selected for each graph, so pinning each one is
// out of reach; the seed suite above does that. Each graph here runs (a) an unbenchmarked
// plan, which also checks the selected count of each family, (b) a fixed sample of
// selected instances, each pinned: the best few by score, then more drawn by a seeded
// shuffle of the dtype's whole catalog, and (c) a benchmarked plan under
// HIPDNN_INGESTOR_CANDIDATE_LIMIT.

namespace
{

constexpr const char* AOT_ENGINE_NAME = "hipkernel:Gfx950ConvFwdAot";
constexpr size_t TOP_BY_SCORE = 4;
constexpr size_t SHUFFLED_SAMPLES = 8;
constexpr size_t BENCHMARK_LIMIT = 64;

/// The kernel the last plan build served from a benchmarked record, or empty.
std::string servedKernel(const LogRecorderBase& recorder)
{
    const std::string marker = "' served kernel ";
    std::string served;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at == std::string::npos)
        {
            continue;
        }
        const auto begin = at + marker.size();
        served = log.message.substr(begin, log.message.find(' ', begin) - begin);
    }
    return served;
}

/// Every kernel of the shipped gfx950 shards for @p dtype, as winner-record entries, read
/// from the product tree main() loads (HIPDNN_DESCRIPTOR_RUNTIME_DIR). Empty when they are
/// not there. Read once per dtype: the shards gfx950_conv_fwd_aot_<dtype>_*.kdp.json
/// together hold every instance of the dtype, each entry under its own shard's pack.
const std::vector<ingestor::RankedEntry>& shippedCatalog(DataType dtype)
{
    static std::map<DataType, std::vector<ingestor::RankedEntry>> s_catalogs;
    auto& entries = s_catalogs[dtype];
    if(!entries.empty())
    {
        return entries;
    }
    const std::filesystem::path root
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    const std::string prefix
        = std::string("gfx950_conv_fwd_aot_") + (dtype == DataType::HALF ? "fp16" : "bf16") + "_";
    const std::string suffix = ".kdp.json";
    std::error_code error;
    if(root.empty() || !std::filesystem::is_directory(root, error))
    {
        return entries;
    }
    std::set<std::string> read;
    for(const auto& file : std::filesystem::recursive_directory_iterator(root, error))
    {
        const auto name = file.path().filename().string();
        if(name.size() <= prefix.size() + suffix.size() || name.rfind(prefix, 0) != 0
           || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0
           || !read.insert(name).second)
        {
            continue;
        }
        std::ifstream stream(file.path());
        const auto document = nlohmann::json::parse(stream);
        const auto packId
            = hipdnn_flatbuffers_sdk::utilities::parseUuid(document.at("id").get<std::string>());
        const auto dispatchId = hipdnn_flatbuffers_sdk::utilities::parseUuid(
            document.at("dispatch").get<std::string>());
        for(const auto& kernel : document.at("kernelDescriptors"))
        {
            entries.push_back(
                {hipdnn_flatbuffers_sdk::utilities::parseUuid(kernel.at("id").get<std::string>()),
                 packId,
                 dispatchId,
                 1.0});
        }
    }
    // Directory order is unspecified; the seeded shuffle below needs a fixed start.
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.kernelId < rhs.kernelId;
    });
    return entries;
}

/// FNV-1a, so each graph draws its own fixed sample on every platform.
uint32_t sampleSeed(const std::string& name)
{
    uint32_t hash = 2166136261U;
    for(const char character : name)
    {
        hash = (hash ^ static_cast<uint8_t>(character)) * 16777619U;
    }
    return hash;
}

} // namespace

class IntegrationGpuGfx950ConvFwdAot : public IntegrationGpuGfx950ConvFwd
{
protected:
    const char* engineName() const override
    {
        return AOT_ENGINE_NAME;
    }

    /// Builds the parameter's graph with @p knobs on a fresh handle, checks that the last
    /// persisted record of @p recordSize entries served @p kernel (any kernel when empty)
    /// out of @p candidates, and checks that kernel against the reference. Returns it.
    std::string serveFromRecord(const std::vector<KnobSetting>& knobs,
                                const std::string& kernel,
                                size_t recordSize,
                                size_t candidates,
                                unsigned int seed)
    {
        const auto& variant = GetParam();
        recreateHandle();
        if(HasFatalFailure())
        {
            return {};
        }
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        auto graph = buildConvGraph(variant.dtype, variant.problem);
        buildAndCompile(*graph, knobs);
        if(HasFatalFailure())
        {
            return {};
        }
        const auto served = servedKernel(recorder);
        if(!kernel.empty())
        {
            EXPECT_EQ(served, kernel) << recorder.getRecordedLogsAsString();
        }
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from a benchmarked record of "
                                              + std::to_string(recordSize) + " entry(s) for "
                                              + candidateCount(candidates)))
            << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
        executeAndVerifyStorage(*graph, variant.dtype, seed);
        return served;
    }

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _candidateLimit{
        hipdnn_plugin_sdk::INGESTOR_CANDIDATE_LIMIT_ENV_NAME, ""};
};

// (a), and the selected count of each family at full size.
TEST_P(IntegrationGpuGfx950ConvFwdAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + variant.implicitArms;

    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(catalog) + " ("
                                          + std::to_string(catalog) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(selectedKernel(recorder).empty()) << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 53));
    graph.reset();

    for(const auto& [family, count] : {std::pair{IMPLICIT_GEMM_FAMILY, variant.implicitArms},
                                       std::pair{DIRECT_DEPTHWISE_FAMILY, variant.directArms}})
    {
        if(count == 0)
        {
            continue;
        }
        recorder.clearLogs();
        auto forced = buildConvGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(
            buildAndCompile(*forced, {kernelFamilyKnob(family), disableBenchmarkingKnob()}));
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(count) + " ("
                                              + std::to_string(catalog)
                                              + " before knob filtering)"))
            << "family " << family << ": " << recorder.getRecordedLogsAsString();
    }
}

// (b): the best TOP_BY_SCORE instances by score, then SHUFFLED_SAMPLES drawn from every
// selected instance, each served alone from a record and checked against the reference.
TEST_P(IntegrationGpuGfx950ConvFwdAot, SampledSelectedInstancesMatchCpuReference)
{
    const auto& variant = GetParam();
    const auto catalog = variant.directArms + variant.implicitArms;
    const std::vector<KnobSetting> knobs{enableBenchmarkingKnob()};
    std::vector<std::string> pinned;

    // Benchmarking under a limit of TOP_BY_SCORE measures exactly the best-ranked ones.
    _candidateLimit.setValue(std::to_string(TOP_BY_SCORE));
    {
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        auto graph = buildConvGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, knobs));
        EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(TOP_BY_SCORE)
                                              + " (" + std::to_string(catalog)
                                              + " before knob filtering)"))
            << recorder.getRecordedLogsAsString();
        ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 59));
    }
    const auto measured = lastPersistedRecord();
    ASSERT_TRUE(measured) << "Benchmarking did not persist its measured ranking";
    const auto& [key, top] = *measured;
    ASSERT_EQ(top.size(), TOP_BY_SCORE);
    for(size_t arm = 0; arm < top.size(); ++arm)
    {
        auto record = top;
        std::rotate(record.begin(),
                    record.begin() + static_cast<std::ptrdiff_t>(arm),
                    record.begin() + static_cast<std::ptrdiff_t>(arm) + 1);
        ASSERT_NO_FATAL_FAILURE(appendPersistedRecord(key, record));
        const auto kernel = ingestor::toString(record.front().kernelId);
        ASSERT_NO_FATAL_FAILURE(serveFromRecord(
            knobs, kernel, TOP_BY_SCORE, TOP_BY_SCORE, 61 + static_cast<unsigned int>(arm)));
        pinned.push_back(kernel);
    }

    // A record ordering every kernel of the dtype's shards covers every candidate, so it
    // serves the first instance in its order that is selected for the graph: in a seeded
    // shuffle, a random selected instance. Instances already served move to the back.
    _candidateLimit.setValue("");
    auto shard = shippedCatalog(variant.dtype);
    ASSERT_GE(shard.size(), catalog) << "the shipped gfx950 shards for this dtype were not found";
    for(const auto& entry : top)
    {
        // The measured record and the shards agree on each kernel's pack and dispatch.
        ASSERT_NE(std::find_if(shard.begin(),
                               shard.end(),
                               [&](const auto& listed) {
                                   return listed.kernelId == entry.kernelId
                                          && listed.packId == entry.packId
                                          && listed.dispatchId == entry.dispatchId;
                               }),
                  shard.end())
            << ingestor::toString(entry.kernelId);
    }
    std::mt19937 rng(sampleSeed(variant.name));
    for(size_t i = shard.size() - 1; i > 0; --i)
    {
        std::swap(shard[i], shard[rng() % (i + 1)]);
    }
    const auto samples = std::min(SHUFFLED_SAMPLES, catalog - std::min(catalog, TOP_BY_SCORE));
    for(size_t sample = 0; sample < samples; ++sample)
    {
        std::stable_partition(shard.begin(), shard.end(), [&](const auto& entry) {
            return std::find(pinned.begin(), pinned.end(), ingestor::toString(entry.kernelId))
                   == pinned.end();
        });
        ASSERT_NO_FATAL_FAILURE(appendPersistedRecord(key, shard));
        std::string served;
        ASSERT_NO_FATAL_FAILURE(
            served = serveFromRecord(
                knobs, "", shard.size(), catalog, 67 + static_cast<unsigned int>(sample)));
        ASSERT_FALSE(served.empty());
        EXPECT_EQ(std::find(pinned.begin(), pinned.end(), served), pinned.end()) << served;
        pinned.push_back(served);
    }
}

// (c): benchmarking under HIPDNN_INGESTOR_CANDIDATE_LIMIT measures only the best-ranked
// candidates, and its record serves the next build of the graph.
TEST_P(IntegrationGpuGfx950ConvFwdAot, CappedBenchmarkMatchesCpuReference)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + variant.implicitArms;
    const auto capped = std::min(BENCHMARK_LIMIT, catalog);
    _candidateLimit.setValue(std::to_string(BENCHMARK_LIMIT));

    auto graph = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {enableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(capped) + " ("
                                          + std::to_string(catalog) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 71));
    graph.reset();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), capped);
    const std::string winner = ingestor::toString(ranking->front().kernelId);
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel " + winner))
        << recorder.getRecordedLogsAsString();

    recorder.clearLogs();
    auto recreated = buildConvGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {enableBenchmarkingKnob()}));
    const auto count = std::to_string(capped);
    EXPECT_TRUE(recorder.hasLogContaining("served kernel " + winner + " at rank 0"))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count + " entry(s) for "
                                          + count + " candidate(s)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*recreated, variant.dtype, 73));
}

#endif // HKP_TEST_GFX950_CONV_FWD_AOT

#ifdef HKP_TEST_GFX950_CONV_FWD
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

INSTANTIATE_TEST_SUITE_P(Uncataloged,
                         IntegrationGpuGfx950ConvFwdServing,
                         ::testing::ValuesIn(uncatalogedVariants()),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(Autotuning,
                         IntegrationGpuGfx950ConvFwdBenchmark,
                         ::testing::Values(Variant{"Fp16", DataType::HALF, 64},
                                           Variant{"Bf16", DataType::BFLOAT16, 64}),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                             return info.param.name;
                         });
#endif // HKP_TEST_GFX950_CONV_FWD

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT
INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx950ConvFwdAot,
                         ::testing::ValuesIn(aotVariants(true)),
                         [](const auto& info) { return info.param.name; });
#endif

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT_SEED
INSTANTIATE_TEST_SUITE_P(AotSeedInstances,
                         IntegrationGpuGfx950ConvFwdAotSeed,
                         ::testing::ValuesIn(aotVariants(false)),
                         [](const auto& info) { return info.param.name; });
#endif

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
