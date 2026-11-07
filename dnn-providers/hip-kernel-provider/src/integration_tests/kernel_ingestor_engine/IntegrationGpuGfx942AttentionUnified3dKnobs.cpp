// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx942AttentionUnified3dKnobs.cpp
 * @brief hipkernel:Gfx942AttentionUnified3d end to end through the frontend on a gfx942
 *        device: forcing block_size and num_segments selects exactly the expected shipped
 *        kernel, its segment and reduce launches compute what the CPU reference computes,
 *        and a second execute of the same plan does too.
 *
 * The catalog ships one kernel per GQA ratio (KMD num_queries_per_kv) and the kernels read
 * the head counts at runtime, so forcing (block_size, num_segments) leaves exactly one
 * kernel for a graph's (dtype, head_size, Hq / Hkv) cell. One case uses a head pair no
 * catalog cell lists (Hq 24, Hkv 6) to prove the runtime heads end to end. Every kernel
 * loads K/V with the "stream" cache policy (pinned in the catalog). The expected kernel id
 * and the candidate count are read from the production descriptors, never written here.
 *
 * The cases cover BHSD, BSHD, padded-row and mixed K/V layouts. Poisoned cases fill the
 * workspace with 0xFF before every execute: the workspace is not zeroed, so a segment that
 * leaves a split-KV slot or the seqused_k / cu_seqlens_q tail unwritten shows up as NaN or
 * a wrong row. The second execute also starts from a sentinel-filled O, which only the
 * reduce writes.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx942AttentionUnified3d";

constexpr const char* SERVED_ARCH = "gfx942";

constexpr const char* BLOCK_SIZE_KNOB = "block_size";
constexpr const char* NUM_SEGMENTS_KNOB = "num_segments";

// The selection line spells the kernel id in lowercase with hyphens, in 36 characters.
constexpr size_t KERNEL_ID_LENGTH = 36;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

enum class KvLayout
{
    BHSD,
    BSHD,
    BHSD_PADDED_ROW, // row stride D + 8
};

struct DecodeCase
{
    const char* name;
    DataType dataType;
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqKv;
    int64_t headSize;
    KvLayout kLayout;
    KvLayout vLayout;
    int64_t blockSize;
    int64_t numSegments;
    bool poisonWorkspace;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const DecodeCase& decodeCase, std::ostream* os)
{
    *os << decodeCase.name;
}

std::vector<DecodeCase> decodeCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    return {
        // 8 MiB of K/V.
        {"Fp16D128Gqa4Bhsd",
         FP16,
         4,
         32,
         8,
         512,
         128,
         KvLayout::BHSD,
         KvLayout::BHSD,
         16,
         32,
         false},
        // 7.8 MiB; K token-major, V head-major.
        {"Bf16D128MixedKBshdVBhsdPoisonedWorkspace",
         BF16,
         2,
         32,
         8,
         1000,
         128,
         KvLayout::BSHD,
         KvLayout::BHSD,
         16,
         38,
         true},
        // 288 MiB of K/V, past the 256 MiB Infinity Cache: a large K/V read from HBM.
        {"Bf16D128MixedKBshdVBhsdPoisonedWorkspaceLargeKv",
         BF16,
         9,
         32,
         8,
         8192,
         128,
         KvLayout::BSHD,
         KvLayout::BHSD,
         16,
         38,
         true},
        // Hq 24 / Hkv 6: a head pair no catalog cell lists. The ratio-4 kernel reads both
        // head counts at runtime, so it serves this pair too.
        {"Bf16D128Hq24Hkv6RuntimeHeadsBhsd",
         BF16,
         3,
         24,
         6,
         777,
         128,
         KvLayout::BHSD,
         KvLayout::BHSD,
         16,
         38,
         true},
        // MQA, K/V rows padded to D + 8.
        {"Fp16D64MqaPaddedRowPoisonedWorkspace",
         FP16,
         3,
         8,
         1,
         33,
         64,
         KvLayout::BHSD_PADDED_ROW,
         KvLayout::BHSD_PADDED_ROW,
         16,
         7,
         true},
        // D256 at block 32, whose KV tile rocKE resolves to 16.
        {"Fp16D256Block32Bshd",
         FP16,
         1,
         8,
         8,
         4096,
         256,
         KvLayout::BSHD,
         KvLayout::BSHD,
         32,
         16,
         false},
    };
}

/// Contiguous [B, H, 1, D] (Q and O).
std::vector<int64_t> decodeQoStrides(int64_t heads, int64_t headSize)
{
    return {heads * headSize, headSize, heads * headSize, 1};
}

std::vector<int64_t> kvStrides(KvLayout layout, int64_t heads, int64_t seq, int64_t headSize)
{
    switch(layout)
    {
    case KvLayout::BSHD:
        return {seq * heads * headSize, headSize, heads * headSize, 1};
    case KvLayout::BHSD_PADDED_ROW:
    {
        const int64_t row = headSize + 8;
        return {heads * seq * row, seq * row, row, 1};
    }
    case KvLayout::BHSD:
    default:
        return {heads * seq * headSize, seq * headSize, headSize, 1};
    }
}

std::shared_ptr<TensorAttributes> makeTensor(int64_t uid,
                                             const std::string& name,
                                             DataType dataType,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid).set_name(name).set_dim(dims).set_stride(strides).set_data_type(dataType);
    return tensor;
}

struct SdpaGraph
{
    std::shared_ptr<Graph> graph;
    std::shared_ptr<TensorAttributes> output;
};

/// One single-token SDPA-forward node: Q/O contiguous [B, Hq, 1, D], K/V in the case's
/// layouts, scale 1/sqrt(D) and a bottom-right causal band (which at Sq = 1 hides no key).
SdpaGraph buildDecodeGraph(const DecodeCase& c)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(c.name)
        .set_io_data_type(c.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    const std::vector<int64_t> kvDims{c.batch, c.kvHeads, c.seqKv, c.headSize};
    auto q = makeTensor(Q_UID,
                        "Q",
                        c.dataType,
                        {c.batch, c.queryHeads, 1, c.headSize},
                        decodeQoStrides(c.queryHeads, c.headSize));
    auto k = makeTensor(
        K_UID, "K", c.dataType, kvDims, kvStrides(c.kLayout, c.kvHeads, c.seqKv, c.headSize));
    auto v = makeTensor(
        V_UID, "V", c.dataType, kvDims, kvStrides(c.vLayout, c.kvHeads, c.seqKv, c.headSize));

    SdpaAttributes attributes;
    attributes.set_name(c.name);
    attributes.set_attn_scale(1.0f / std::sqrt(static_cast<float>(c.headSize)));
    attributes.set_diagonal_band_right_bound(0);
    attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(c.dataType)
        .set_dim({c.batch, c.queryHeads, 1, c.headSize})
        .set_stride(decodeQoStrides(c.queryHeads, c.headSize));

    return {graph, o};
}

/// The forward SDPA tolerance every SdpaFwd bundle of this engine is graded at
/// (config/hipkernel_Gfx942AttentionUnified3d.toml names no override).
float sdpaForwardTolerance(DataType dataType)
{
    return dataType == DataType::HALF
               ? hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>()
               : hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<
                     hipdnn_data_sdk::types::bfloat16>();
}

std::vector<KnobSetting> knobSettingsFor(const DecodeCase& c)
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_SIZE_KNOB, c.blockSize);
    settings.emplace_back(NUM_SEGMENTS_KNOB, c.numSegments);
    return settings;
}

/// The production descriptors HIPDNN_DESCRIPTOR_RUNTIME_DIR names, loaded once: every case
/// reads the same root, which main() sets before any case runs.
const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& runtimeCatalog()
{
    static const auto s_catalog = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(
        std::filesystem::path(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")));
    return s_catalog;
}

/// Calls @p visit for every kernel runtimeCatalog() serves this engine with on SERVED_ARCH.
template <typename Visit>
void forEachServedKernel(Visit&& visit)
{
    const auto& catalog = runtimeCatalog();
    for(const auto& entry : catalog.packs)
    {
        const auto& pack = entry.second;
        const auto& arch = pack.descriptor.arch;
        const auto engine = catalog.engines.find(pack.descriptor.engineId);
        if(pack.conflicted || engine == catalog.engines.end()
           || engine->second.descriptor.name != ENGINE_NAME
           || std::find(arch.begin(), arch.end(), SERVED_ARCH) == arch.end())
        {
            continue;
        }
        for(const auto& kernel : pack.descriptor.kernels)
        {
            visit(kernel);
        }
    }
}

bool carries(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
             const std::string& field,
             const hipdnn_plugin_sdk::ingestor::MetadataValue& want)
{
    const auto value = kernel.metadata.find(field);
    return value != kernel.metadata.end() && value->second == want;
}

/// The kernels a forced (block_size, num_segments) leaves for @p c's graph cell
/// (dtype, head_size, num_queries_per_kv = Hq / Hkv), and the id of the one it must be.
struct ExpectedSelection
{
    std::string kernelId;
    size_t candidates;
};

/// Records a failure and returns nullopt unless exactly one candidate is left.
std::optional<ExpectedSelection> expectedSelection(const DecodeCase& c)
{
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        ADD_FAILURE() << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production "
                         "descriptors to read the expected kernel id from";
        return std::nullopt;
    }

    const std::string dtype = c.dataType == DataType::HALF ? "FP16" : "BF16";
    size_t candidates = 0;
    std::vector<std::string> ids;
    forEachServedKernel([&](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        if(!carries(kernel, "dtype", dtype) || !carries(kernel, "head_size", c.headSize)
           || !carries(kernel, "num_queries_per_kv", c.queryHeads / c.kvHeads)
           || !carries(kernel, "sliding_window", int64_t{0})
           || !carries(kernel, "block_size", c.blockSize)
           || !carries(kernel, "num_segments", c.numSegments))
        {
            return;
        }
        ++candidates;
        ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
    });

    if(ids.size() != 1)
    {
        ADD_FAILURE() << ids.size() << " " << ENGINE_NAME << " kernel(s) for " << SERVED_ARCH
                      << " carry " << dtype << " D" << c.headSize << " num_queries_per_kv "
                      << c.queryHeads / c.kvHeads << " (Hq" << c.queryHeads << " Hkv" << c.kvHeads
                      << ") block_size " << c.blockSize << " num_segments " << c.numSegments
                      << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                      << "'; exactly one must";
        return std::nullopt;
    }
    return ExpectedSelection{ids.front(), candidates};
}

/// The kernel id the plugin's selection line names, or nullopt when no plan was built.
std::optional<std::string>
    selectedKernelId(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string marker = std::string("engine '") + ENGINE_NAME + detail::SELECTED_KERNEL_LOG;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at != std::string::npos)
        {
            return log.message.substr(at + marker.size(), KERNEL_ID_LENGTH);
        }
    }
    return std::nullopt;
}

} // namespace

/// Runs only on the architecture the engine ships for. There, the engine, its packs and its
/// kernels are all expected present, and any of them missing fails the case.
class IntegrationGpuGfx942AttentionUnified3dKnobs
    : public IntegrationGraphVerificationHarness<float, DecodeCase>
{
protected:
    using Harness = IntegrationGraphVerificationHarness<float, DecodeCase>;

    void SetUp() override
    {
        Harness::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }

        const auto arch = hip_kernel_provider_common::getDeviceString(_stream);
        if(arch != SERVED_ARCH)
        {
            GTEST_SKIP() << ENGINE_NAME << " ships for " << SERVED_ARCH << " only; this device is "
                         << arch;
        }
    }

    /// Offsets the seed by UID so Q, K and V carry different data: K and V share a shape,
    /// and an operand swap between them is invisible when their contents agree.
    void initializeBundle(const Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   Harness::DEFAULT_MIN,
                                   Harness::DEFAULT_MAX,
                                   seed + static_cast<unsigned int>(tensorPair.first));
        }
    }

    /// 0xFF bytes are NaN in the fp32 workspace regions, so an unwritten slot poisons the row.
    void prepareWorkspace(void* workspace, size_t bytes) override
    {
        if(GetParam().poisonWorkspace && bytes > 0)
        {
            ASSERT_EQ(hipMemset(workspace, 0xFF, bytes), hipSuccess);
        }
    }

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    /// On this architecture the engine's absence from the ranked list is a missing engine
    /// or pack, not a skip.
    void buildOperationGraphTheEngineOffersToServe(Graph& graph)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        const bool offered
            = result.code == ErrorCode::OK
              && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId())
                     != rankedEngineIds.end();
        ASSERT_TRUE(offered) << ENGINE_NAME << " did not offer to serve this graph on "
                             << SERVED_ARCH << " (" << result.err_msg
                             << "). Its packed descriptors load through "
                             << "HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                             << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                             << "'; this configuration must pack the production descriptors for "
                             << SERVED_ARCH << ".";
    }

    /// A cache root private to ONE test case. A winner record another case measured would
    /// replace the heuristic order, and the selected kernel with it.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx942-unified-3d-knobs-case",
        hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter, or a wrong ratio match, reads as
/// another kernel; the output is then checked against the CPU reference twice through the
/// same plan.
TEST_P(IntegrationGpuGfx942AttentionUnified3dKnobs,
       SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();

    auto sdpa = buildDecodeGraph(testCase);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    auto result = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase));
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = selectedKernelId(recorder);
    std::cout << "UNIFIED3D_SELECTED " << testCase.name << " " << selected.value_or("none") << '\n';
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE_NAME << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const auto expected = expectedSelection(testCase);
    ASSERT_TRUE(expected.has_value());
    EXPECT_EQ(*selected, expected->kernelId) << "Captured logs:\n"
                                             << recorder.getRecordedLogsAsString();

    // Rank 0 of the forced pair's builds: the scorer's own pick, not a fallback past a
    // kernel that failed to load.
    const auto selectionLine = std::string(detail::SELECTED_KERNEL_LOG) + expected->kernelId
                               + " at rank 0 from " + std::to_string(expected->candidates)
                               + " candidate(s)";
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(testCase.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    VerificationRun run;
    ASSERT_NO_FATAL_FAILURE(prepareVerification(context, /*seed=*/0, run));
    // Twice through the same plan: the second execute starts from a sentinel-filled O and,
    // in poisoned cases, a re-poisoned workspace.
    ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *sdpa.graph, _handle));
    ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *sdpa.graph, _handle));
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified3dKnobs,
                         ::testing::ValuesIn(decodeCases()),
                         [](const ::testing::TestParamInfo<DecodeCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
