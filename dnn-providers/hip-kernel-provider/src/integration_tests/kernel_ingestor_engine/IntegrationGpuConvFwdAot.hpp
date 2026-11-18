// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
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
#include "IngestorIntegrationSupport.hpp"

/// @file IntegrationGpuConvFwdAot.hpp
/// The GPU suites of the forward AOT-instance engines (hipkernel:Gfx950ConvFwdAot and
/// its seed, hipkernel:Gfx942ConvFwdAot), shared: a forward graph, the CPU reference, the
/// winner-record helpers, the shipped catalog read from the product tree main() loads, and
/// the fixtures whose members are the suites' test bodies. Each engine's file supplies its
/// FwdAotTarget (engine, architecture, cache tag, descriptor prefix) and its graphs, each
/// with how many catalog instances of each family the engine selects for it: those tables
/// stay in the engine's file, where test_make_conv_aot_config.py reads them.
namespace hip_kernel_provider::kernel_ingestor_engine::integration::conv_fwd_aot_test
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
constexpr int64_t DIRECT_DEPTHWISE_FAMILY = 1;

/// One forward AOT engine under test.
struct FwdAotTarget
{
    const char* engineName;
    /// The architecture its device must be (wave64).
    const char* arch;
    /// Its descriptor files' prefix in the product tree, e.g. "gfx950_conv_fwd_aot".
    const char* descriptorPrefix;
    /// The per-test winner cache directory's tag and the graphs' name.
    const char* cacheTag;
    const char* graphName;
};

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

inline std::shared_ptr<Graph> buildConvGraph(DataType dtype,
                                             const ConvProblem& problem = {},
                                             bool channelsLast = true,
                                             const char* name = "conv_fwd_aot")
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(name)
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
    attributes.set_name(name)
        .set_padding({problem.padH, problem.padW})
        .set_stride({problem.strideH, problem.strideW})
        .set_dilation({problem.dilationH, problem.dilationW})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    auto y = graph->conv_fprop(x, w, attributes);
    // The frontend infers output dims/strides from the channels-last input.
    y->set_uid(3).set_name("Y").set_output(true).set_data_type(dtype);
    return graph;
}

inline KnobSetting enableBenchmarkingKnob()
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1});
}

inline KnobSetting disableBenchmarkingKnob()
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{0});
}

inline KnobSetting kernelFamilyKnob(int64_t family)
{
    return {KERNEL_FAMILY_KNOB, family};
}

// The kernel the last non-benchmarked plan build selected, or empty.
inline std::string selectedKernel(const LogRecorderBase& recorder)
{
    const std::string marker = detail::SELECTED_KERNEL_LOG;
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

/// The kernel the last plan build served from a benchmarked record, or empty.
inline std::string servedKernel(const LogRecorderBase& recorder)
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

/// Every kernel of the shipped <descriptorPrefix>_<dtype>_*.kdp.json shards, as
/// winner-record entries, read from the product tree main() loads
/// (HIPDNN_DESCRIPTOR_RUNTIME_DIR). Empty when they are not there. Read once per prefix and
/// dtype: the shards together hold every instance of the dtype, each entry under its own
/// shard's pack.
inline const std::vector<ingestor::RankedEntry>& shippedCatalog(const std::string& descriptorPrefix,
                                                                DataType dtype)
{
    static std::map<std::pair<std::string, DataType>, std::vector<ingestor::RankedEntry>>
        s_catalogs;
    auto& entries = s_catalogs[{descriptorPrefix, dtype}];
    if(!entries.empty())
    {
        return entries;
    }
    const std::filesystem::path root
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
    const std::string prefix
        = descriptorPrefix + "_" + (dtype == DataType::HALF ? "fp16" : "bf16") + "_";
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
inline uint32_t sampleSeed(const std::string& name)
{
    uint32_t hash = 2166136261U;
    for(const char character : name)
    {
        hash = (hash ^ static_cast<uint8_t>(character)) * 16777619U;
    }
    return hash;
}

class IntegrationGpuConvFwdAotBase
    : public test_utilities::IntegrationGraphVerificationHarness<float, Variant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, Variant>;

    explicit IntegrationGpuConvFwdAotBase(const FwdAotTarget& target)
        : _target(target)
        , _cacheDir{target.cacheTag, ScopedTestCacheDir::Scope::TEST}
    {
    }

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
               _arch, _target.arch, hipdnn_plugin_sdk::ArchMatchMode::PREFIX)
           || properties.warpSize != 64)
        {
            GTEST_SKIP() << "The packaged convolution family requires a wave64 " << _target.arch;
        }
    }

    /// The engine under test: the target's, unless a suite names another (the seed).
    virtual const char* engineName() const
    {
        return _target.engineName;
    }

    int64_t engineId() const
    {
        return hipdnn_data_sdk::utilities::engineNameToId(engineName());
    }

    std::shared_ptr<Graph> buildGraph(DataType dtype, const ConvProblem& problem) const
    {
        return buildConvGraph(dtype, problem, true, _target.graphName);
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
        // Only a split-C AOT instance (its fp32 partial sums) needs scratch.
        ASSERT_GE(workspaceSize, 0);
        _workspaceSize = workspaceSize;
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
        void* workspace = nullptr;
        if(_workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, static_cast<size_t>(_workspaceSize)), hipSuccess);
        }
        const auto result = graph.execute(_handle, buffers, workspace);
        const auto synchronized = hipStreamSynchronize(_stream);
        if(workspace != nullptr)
        {
            EXPECT_EQ(hipFree(workspace), hipSuccess);
        }
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(synchronized, hipSuccess);
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

    std::filesystem::path winnerShardPath() const
    {
        return integration::winnerShardPath(engineName(), _arch);
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

    FwdAotTarget _target;
    // The inherited harness opens the plugin relative to the running binary. No
    // descriptor override is set: installed tests must load installed kpack archives.
    ScopedTestCacheDir _cacheDir;
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    std::string _arch;
    /// The workspace the last buildAndCompile's plan asked for.
    int64_t _workspaceSize = 0;
};

class IntegrationGpuConvFwdAotServing : public IntegrationGpuConvFwdAotBase
{
protected:
    using IntegrationGpuConvFwdAotBase::IntegrationGpuConvFwdAotBase;

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

        auto graph = buildGraph(variant.dtype, variant.problem);
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

            auto served = buildGraph(variant.dtype, variant.problem);
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

// ---- The shipped catalog ----
//
// At full size hundreds of instances are selected for each graph, so pinning each one per
// graph is out of reach. Each graph runs (a) an unbenchmarked plan, which also checks the
// selected count of each family, (b) a fixed sample of selected instances, each pinned: the
// best few by score, then more drawn by a seeded shuffle of the dtype's whole catalog, and
// (c) a benchmarked plan under HIPDNN_INGESTOR_SHORTLIST_LIMIT. everyInstanceMatchesCpuReference
// walks a whole dtype instead, pinning each instance on the first graph it is selected for.

constexpr size_t TOP_BY_SCORE = 4;
constexpr size_t SHUFFLED_SAMPLES = 8;
constexpr size_t BENCHMARK_LIMIT = 64;

class IntegrationGpuConvFwdAotCatalog : public IntegrationGpuConvFwdAotBase
{
protected:
    using IntegrationGpuConvFwdAotBase::IntegrationGpuConvFwdAotBase;

    const std::vector<ingestor::RankedEntry>& shipped(DataType dtype) const
    {
        return shippedCatalog(_target.descriptorPrefix, dtype);
    }

    /// Builds @p problem in @p dtype with @p knobs on a fresh handle, checks that the last
    /// persisted record of @p recordSize entries served @p kernel (any kernel when empty)
    /// out of @p candidates, and checks that kernel against the reference. Returns it.
    std::string serveFromRecord(DataType dtype,
                                const ConvProblem& problem,
                                const std::vector<KnobSetting>& knobs,
                                const std::string& kernel,
                                size_t recordSize,
                                size_t candidates,
                                unsigned int seed)
    {
        recreateHandle();
        if(HasFatalFailure())
        {
            return {};
        }
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        auto graph = buildGraph(dtype, problem);
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
        executeAndVerifyStorage(*graph, dtype, seed);
        return served;
    }

    /// serveFromRecord on the parameter's graph.
    std::string serveFromRecord(const std::vector<KnobSetting>& knobs,
                                const std::string& kernel,
                                size_t recordSize,
                                size_t candidates,
                                unsigned int seed)
    {
        const auto& variant = GetParam();
        return serveFromRecord(
            variant.dtype, variant.problem, knobs, kernel, recordSize, candidates, seed);
    }

    // (a), and the selected count of each family at full size.
    void unbenchmarkedPlanServesFromTheWholeCatalog()
    {
        const auto& variant = GetParam();
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        const auto catalog = variant.directArms + variant.implicitArms;

        auto graph = buildGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {disableBenchmarkingKnob()}));
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(catalog) + " ("
                                              + std::to_string(catalog)
                                              + " before knob filtering)"))
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
            auto forced = buildGraph(variant.dtype, variant.problem);
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
    void sampledSelectedInstancesMatchCpuReference()
    {
        const auto& variant = GetParam();
        const auto catalog = variant.directArms + variant.implicitArms;
        const std::vector<KnobSetting> knobs{enableBenchmarkingKnob()};
        std::vector<std::string> pinned;
        // A graph with fewer selected instances than TOP_BY_SCORE measures all of them.
        const auto topCount = std::min(TOP_BY_SCORE, catalog);

        // Benchmarking under a limit of TOP_BY_SCORE measures exactly the best-ranked ones.
        _candidateLimit.setValue(std::to_string(TOP_BY_SCORE));
        {
            const ScopedPluginLogCapture capture(this);
            auto& recorder = capture.recorder();
            auto graph = buildGraph(variant.dtype, variant.problem);
            ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, knobs));
            EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(topCount)
                                                  + " (" + std::to_string(catalog)
                                                  + " before knob filtering)"))
                << recorder.getRecordedLogsAsString();
            ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 59));
        }
        const auto measured = lastPersistedRecord();
        ASSERT_TRUE(measured) << "Benchmarking did not persist its measured ranking";
        const auto& [key, top] = *measured;
        ASSERT_EQ(top.size(), topCount);
        for(size_t arm = 0; arm < top.size(); ++arm)
        {
            auto record = top;
            std::rotate(record.begin(),
                        record.begin() + static_cast<std::ptrdiff_t>(arm),
                        record.begin() + static_cast<std::ptrdiff_t>(arm) + 1);
            ASSERT_NO_FATAL_FAILURE(appendPersistedRecord(key, record));
            const auto kernel = ingestor::toString(record.front().kernelId);
            ASSERT_NO_FATAL_FAILURE(serveFromRecord(
                knobs, kernel, topCount, topCount, 61 + static_cast<unsigned int>(arm)));
            pinned.push_back(kernel);
        }

        // A record ordering every kernel of the dtype's shards covers every candidate, so it
        // serves the first instance in its order that is selected for the graph: in a seeded
        // shuffle, a random selected instance. Instances already served move to the back.
        _candidateLimit.setValue("");
        auto shard = shipped(variant.dtype);
        ASSERT_GE(shard.size(), catalog)
            << "the shipped " << _target.arch << " shards for this dtype were not found";
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

    // (c): benchmarking under HIPDNN_INGESTOR_SHORTLIST_LIMIT measures only the best-ranked
    // candidates, and its record serves the next build of the graph.
    void cappedBenchmarkMatchesCpuReference()
    {
        const auto& variant = GetParam();
        const ScopedPluginLogCapture capture(this);
        auto& recorder = capture.recorder();
        const auto catalog = variant.directArms + variant.implicitArms;
        const auto capped = std::min(BENCHMARK_LIMIT, catalog);
        _candidateLimit.setValue(std::to_string(BENCHMARK_LIMIT));

        auto graph = buildGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {enableBenchmarkingKnob()}));
        EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(capped) + " ("
                                              + std::to_string(catalog)
                                              + " before knob filtering)"))
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
        auto recreated = buildGraph(variant.dtype, variant.problem);
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {enableBenchmarkingKnob()}));
        const auto count = std::to_string(capped);
        EXPECT_TRUE(recorder.hasLogContaining("served kernel " + winner + " at rank 0"))
            << recorder.getRecordedLogsAsString();
        EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count
                                              + " entry(s) for " + count + " candidate(s)"))
            << recorder.getRecordedLogsAsString();
        EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
        ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*recreated, variant.dtype, 73));
    }

    /// Every instance of the parameter's dtype, run once: for each of @p graphs (the
    /// parameter's dtype, with how many instances the engine selects for each), a record
    /// ordering the dtype's whole shard, the instances not yet run first, serves the first
    /// of them selected for the graph; it is checked against the reference, and the next
    /// record serves the next, until the served kernel is one already run. A graph runs at
    /// most its selected count, and together the graphs must run every shipped instance.
    void everyInstanceMatchesCpuReference(const std::vector<Variant>& graphs)
    {
        const auto& variant = GetParam();
        auto shard = shipped(variant.dtype);
        ASSERT_FALSE(shard.empty())
            << "the shipped " << _target.arch << " shards for this dtype were not found";
        const std::vector<KnobSetting> knobs{enableBenchmarkingKnob()};
        std::set<std::string> pinned;
        unsigned int seed = 101;
        for(const auto& graph : graphs)
        {
            SCOPED_TRACE(graph.name);
            ASSERT_EQ(graph.dtype, variant.dtype);
            const auto selected = graph.directArms + graph.implicitArms;
            // Benchmarking one candidate writes the graph's record key.
            _candidateLimit.setValue("1");
            {
                auto measuredGraph = buildGraph(graph.dtype, graph.problem);
                ASSERT_NO_FATAL_FAILURE(buildAndCompile(*measuredGraph, knobs));
                ASSERT_NO_FATAL_FAILURE(
                    executeAndVerifyStorage(*measuredGraph, graph.dtype, seed++));
            }
            _candidateLimit.setValue("");
            const auto measured = lastPersistedRecord();
            ASSERT_TRUE(measured) << "Benchmarking did not persist its measured ranking";
            const auto key = measured->first;
            size_t ran = 0;
            for(size_t step = 0; step <= selected; ++step)
            {
                std::stable_partition(shard.begin(), shard.end(), [&](const auto& entry) {
                    return pinned.count(ingestor::toString(entry.kernelId)) == 0;
                });
                ASSERT_NO_FATAL_FAILURE(appendPersistedRecord(key, shard));
                std::string served;
                ASSERT_NO_FATAL_FAILURE(
                    served = serveFromRecord(
                        graph.dtype, graph.problem, knobs, "", shard.size(), selected, seed++));
                ASSERT_FALSE(served.empty());
                if(!pinned.insert(served).second)
                {
                    break;
                }
                ++ran;
            }
            // A graph whose instances all ran before runs none here.
            EXPECT_LE(ran, selected);
        }
        EXPECT_EQ(pinned.size(), shard.size())
            << shard.size() - std::min(shard.size(), pinned.size())
            << " shipped instance(s) never ran";
        std::cout << "[ conv fwd ] " << variant.name << ": " << pinned.size() << " of "
                  << shard.size() << " instances ran against the CPU reference\n";
    }

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter _candidateLimit{
        hipdnn_plugin_sdk::INGESTOR_SHORTLIST_LIMIT_ENV_NAME, ""};
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::conv_fwd_aot_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
