// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
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
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx942AttentionUnified2DKnobs.cpp
 * @brief hipkernel:Gfx942AttentionUnified2D end to end through the frontend on a gfx942
 *        device: each catalog kernel class is selected by its knobs, launched with its own
 *        geometry and checked against the CPU reference; the shapes the engine must leave
 *        alone are not offered.
 *
 * The knobs are the geometry (tile_size, num_warps, block_m_per_warp and the MFMA codepath
 * flags). Each expected kernel id is resolved at run time from the production descriptors
 * (kernelIdsFor), so a reduced catalog is checked against the kernels it actually ships; the
 * coverage case holds the tables to exactly the catalog's kernel classes. The kernels read
 * the head counts at run time and bake only their ratio, so the forced cases run head pairs
 * the manifest's cells never list. Every case runs with a workspace filled with NaN bytes
 * first: hipDNN does not zero it, and the kernel must not read anything it did not write.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE = "hipkernel:Gfx942AttentionUnified2D";

constexpr const char* SERVED_ARCH = "gfx942";

/// The heuristic path's selection line (GenericPlanBuilder::buildPlan), which names the
/// kernel id as the kdp spells it: lowercase, hyphenated.
constexpr const char* SELECTED_KERNEL_MARKER = "' selected kernel ";
constexpr size_t KERNEL_ID_LENGTH = 36;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

/// The graph masks the cases use. The catalog carries only the bottom-right causal kernel
/// (mask_type 2); the other two appear only in declines.
enum class Mask
{
    NO_MASK,
    TOP_LEFT,
    BOTTOM_RIGHT
};

constexpr int64_t BOTTOM_RIGHT_MASK_TYPE = 2;

struct GraphShape
{
    DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    Mask mask;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
};

/// One forced knob value, as a (KMD field = knob id, value) pair.
using Knobs = std::vector<std::pair<std::string, int64_t>>;

/// One graph, the knobs forced on it (empty: the engine's own ranking picks) and the
/// metadata the selected kernel must carry beyond the graph's own fields.
struct KnobCase
{
    std::string name;
    GraphShape shape;
    Knobs knobs;
    Knobs expected;
};

/// One graph the engine must not offer to serve, and why.
struct DeclineCase
{
    std::string name;
    GraphShape shape;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const DeclineCase& declineCase, std::ostream* os)
{
    *os << declineCase.name;
}

/// A geometry: (tile_size at block_size 16, num_warps, block_m_per_warp, use_mfma_32x32x8,
/// use_transposed_qk_32x32), named for the case.
struct Geometry
{
    const char* name;
    int64_t tileSize;
    int64_t numWarps;
    int64_t blockMPerWarp;
    int64_t useMfma32x32x8;
    int64_t useTransposedQk32x32;
};

/// The catalog's geometries per head size; the first of each is every cell's default.
constexpr std::array<Geometry, 5> D64_GEOMETRIES{{
    {"Transposed4x32T64", 64, 4, 32, 1, 1},
    {"Transposed8x32T64", 64, 8, 32, 1, 1},
    {"Narrow4x16T64", 64, 4, 16, 0, 0},
    {"Narrow4x32T64", 64, 4, 32, 0, 0},
    {"Narrow2x16T64", 64, 2, 16, 0, 0},
}};
constexpr std::array<Geometry, 5> D128_GEOMETRIES{{
    {"Transposed2x32T32", 32, 2, 32, 1, 1},
    {"Transposed4x32T32", 32, 4, 32, 1, 1},
    {"Narrow4x16T32", 32, 4, 16, 0, 0},
    {"Narrow2x32T32", 32, 2, 32, 0, 0},
    {"Narrow2x16T16", 16, 2, 16, 0, 0},
}};

/// Head pairs the forced cases cycle through: every GQA ratio the catalog ships, several of
/// them pairs no manifest cell lists, since one kernel serves every pair of its ratio.
struct Heads
{
    int64_t queryHeads;
    int64_t kvHeads;
};

constexpr std::array<Heads, 7> HEADS{
    {{8, 8}, {24, 6}, {16, 4}, {48, 3}, {64, 8}, {20, 10}, {8, 2}}};

Knobs knobsOf(const Geometry& g)
{
    return {{"tile_size", g.tileSize},
            {"num_warps", g.numWarps},
            {"block_m_per_warp", g.blockMPerWarp},
            {"use_mfma_32x32x8", g.useMfma32x32x8},
            {"use_transposed_qk_32x32", g.useTransposedQk32x32}};
}

std::string dtypeName(DataType dataType)
{
    return dataType == DataType::HALF ? "Fp16" : "Bf16";
}

/// Every kernel class (head size x geometry x dtype, all bottom-right causal) forced once, on
/// a prefill shape with B > 1 and Sq != Skv, cycling through the head pairs; then cold cases
/// the engine's own ranking serves with each head size's default geometry, one at B 512.
std::vector<KnobCase> knobCases()
{
    std::vector<KnobCase> cases;
    std::size_t at = 0;
    const auto addAll = [&](int64_t headSize, const auto& geometries) {
        for(const auto& g : geometries)
        {
            for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
            {
                const auto& h = HEADS[at++ % HEADS.size()];
                cases.push_back(
                    {"D" + std::to_string(headSize) + "_" + g.name + "_" + dtypeName(dataType)
                         + "_Hq" + std::to_string(h.queryHeads) + "Hkv" + std::to_string(h.kvHeads),
                     {dataType, headSize, h.queryHeads, h.kvHeads, Mask::BOTTOM_RIGHT, 2, 96, 256},
                     knobsOf(g),
                     knobsOf(g)});
            }
        }
    };
    addAll(64, D64_GEOMETRIES);
    addAll(128, D128_GEOMETRIES);

    cases.push_back({"D64_Cold_SelectsTheDefaultGeometry",
                     {DataType::BFLOAT16, 64, 32, 8, Mask::BOTTOM_RIGHT, 3, 512, 512},
                     {},
                     knobsOf(D64_GEOMETRIES[0])});
    cases.push_back({"D128_Cold_SelectsTheDefaultGeometry",
                     {DataType::HALF, 128, 64, 8, Mask::BOTTOM_RIGHT, 1, 512, 2048},
                     {},
                     knobsOf(D128_GEOMETRIES[0])});
    // The kernel derives its cu_q search trip count from the num_seqs kernarg.
    cases.push_back({"D128_Cold_Batch512",
                     {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 512, 16, 64},
                     {},
                     knobsOf(D128_GEOMETRIES[0])});
    return cases;
}

/// Shapes the engine must refuse at graph or kernel match, so it is not offered at all.
std::vector<DeclineCase> declineCases()
{
    return {
        // The identity cache is cut into whole 16-token blocks.
        {"SeqKvNotAWholeNumberOfBlocks",
         {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 1, 64, 264}},
        // Decode at a ratio the split-KV gfx942 engine ships is that engine's.
        {"LeavesDecodeToTheSplitKvEngine",
         {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 2, 1, 512}},
        // The kernel computes the bottom-right causal band only.
        {"NoMask", {DataType::BFLOAT16, 128, 8, 2, Mask::NO_MASK, 2, 96, 256}},
        {"TopLeftCausal", {DataType::BFLOAT16, 128, 8, 2, Mask::TOP_LEFT, 2, 96, 256}},
        {"TopLeftCausalAtEqualLengths",
         {DataType::BFLOAT16, 128, 8, 2, Mask::TOP_LEFT, 2, 256, 256}},
        // A GQA ratio the catalog does not ship: graph_match admits it, no kernel matches.
        {"RatioThree", {DataType::BFLOAT16, 128, 24, 8, Mask::BOTTOM_RIGHT, 2, 96, 256}},
    };
}

/// BSHD: token-major, head varying fastest. The engine bakes this layout.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t seq, int64_t headSize)
{
    return {seq * heads * headSize, headSize, heads * headSize, 1};
}

std::shared_ptr<TensorAttributes> makeBshdTensor(int64_t uid,
                                                 const std::string& name,
                                                 DataType dataType,
                                                 int64_t batch,
                                                 int64_t heads,
                                                 int64_t seq,
                                                 int64_t headSize)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({batch, heads, seq, headSize})
        .set_stride(bshdStrides(heads, seq, headSize))
        .set_data_type(dataType);
    return tensor;
}

struct SdpaGraph
{
    std::shared_ptr<Graph> graph;
    std::shared_ptr<TensorAttributes> output;
};

/// A single SDPA-forward node with BSHD Q/K/V/O and the softmax scale 1/sqrt(D). O is laid
/// out explicitly: the frontend's default for an unset output is packed BHSD.
SdpaGraph buildSdpaGraph(const std::string& name, const GraphShape& shape)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(name)
        .set_io_data_type(shape.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = makeBshdTensor(
        Q_UID, "Q", shape.dataType, shape.batch, shape.queryHeads, shape.seqQ, shape.headSize);
    auto k = makeBshdTensor(
        K_UID, "K", shape.dataType, shape.batch, shape.kvHeads, shape.seqKv, shape.headSize);
    auto v = makeBshdTensor(
        V_UID, "V", shape.dataType, shape.batch, shape.kvHeads, shape.seqKv, shape.headSize);

    SdpaAttributes attributes;
    attributes.set_name(name);
    attributes.set_attn_scale(1.0f / std::sqrt(static_cast<float>(shape.headSize)));
    if(shape.mask != Mask::NO_MASK)
    {
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(shape.mask == Mask::TOP_LEFT
                                              ? DiagonalAlignment::TOP_LEFT
                                              : DiagonalAlignment::BOTTOM_RIGHT);
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(shape.dataType)
        .set_dim({shape.batch, shape.queryHeads, shape.seqQ, shape.headSize})
        .set_stride(bshdStrides(shape.queryHeads, shape.seqQ, shape.headSize));

    return {graph, o};
}

/// The forward SDPA tolerance the engine's SdpaFwd bundles are graded at (its TOML names no
/// override).
float sdpaForwardTolerance(DataType dataType)
{
    return dataType == DataType::HALF
               ? hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>()
               : hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<
                     hipdnn_data_sdk::types::bfloat16>();
}

std::vector<KnobSetting> knobSettingsFor(const Knobs& knobs)
{
    std::vector<KnobSetting> settings;
    for(const auto& [knob, value] : knobs)
    {
        settings.emplace_back(knob, value);
    }
    return settings;
}

using MetadataKey = std::vector<std::pair<std::string, hipdnn_plugin_sdk::ingestor::MetadataValue>>;

std::string describeKey(const MetadataKey& key)
{
    std::string text;
    for(const auto& [field, value] : key)
    {
        text += (text.empty() ? "" : " ") + field + "=";
        std::visit(
            [&text](const auto& alternative) {
                using Alternative = std::decay_t<decltype(alternative)>;
                if constexpr(std::is_same_v<Alternative, std::string>)
                {
                    text += alternative;
                }
                else if constexpr(std::is_arithmetic_v<Alternative>)
                {
                    text += std::to_string(alternative);
                }
            },
            value);
    }
    return text;
}

/// The production descriptors HIPDNN_DESCRIPTOR_RUNTIME_DIR names, loaded once.
const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& runtimeCatalog()
{
    static const auto s_catalog = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(
        std::filesystem::path(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")));
    return s_catalog;
}

/// Calls @p visit for every kernel runtimeCatalog() serves ENGINE with on SERVED_ARCH.
template <typename Visit>
void forEachServedKernel(Visit&& visit)
{
    const auto& catalog = runtimeCatalog();
    for(const auto& entry : catalog.packs)
    {
        const auto& pack = entry.second;
        const auto& arch = pack.descriptor.arch;
        const auto found = catalog.engines.find(pack.descriptor.engineId);
        if(pack.conflicted || found == catalog.engines.end()
           || found->second.descriptor.name != ENGINE
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

std::string catalogDtypeName(DataType dataType)
{
    return dataType == DataType::HALF ? "FP16" : "BF16";
}

/// The ids of the kernels the engine carries for @p knobCase's graph fields and expected
/// metadata, as the selection line spells them.
std::vector<std::string> kernelIdsFor(const KnobCase& knobCase)
{
    MetadataKey key{
        {"dtype", catalogDtypeName(knobCase.shape.dataType)},
        {"head_size", knobCase.shape.headSize},
        {"num_queries_per_kv", knobCase.shape.queryHeads / knobCase.shape.kvHeads},
        {"mask_type", BOTTOM_RIGHT_MASK_TYPE},
        {"sliding_window", int64_t{0}},
    };
    for(const auto& [field, value] : knobCase.expected)
    {
        key.emplace_back(field, value);
    }

    std::vector<std::string> ids;
    forEachServedKernel([&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        const bool carriesKey = std::all_of(key.begin(), key.end(), [&kernel](const auto& field) {
            const auto value = kernel.metadata.find(field.first);
            return value != kernel.metadata.end() && value->second == field.second;
        });
        if(carriesKey)
        {
            ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
        }
    });
    if(ids.empty())
    {
        ADD_FAILURE() << "no " << ENGINE << " kernel for " << SERVED_ARCH << " carries "
                      << describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
    }
    return ids;
}

/// The kernel id the plugin's selection line for ENGINE names, or nullopt.
std::optional<std::string>
    selectedKernelId(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string marker = std::string("engine '") + ENGINE + SELECTED_KERNEL_MARKER;
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

/// Prints `KNOB_SELECTED <case> <kernelId>` when the case ends, however it ends.
class SelectedKernelReport
{
public:
    SelectedKernelReport(std::string caseName, const ScopedPluginLogCapture& capture)
        : _caseName(std::move(caseName))
        , _capture(capture)
    {
    }

    ~SelectedKernelReport()
    {
        std::cout << "KNOB_SELECTED " << _caseName << " "
                  << selectedKernelId(_capture.recorder()).value_or("none") << '\n';
    }

    SelectedKernelReport(const SelectedKernelReport&) = delete;
    SelectedKernelReport& operator=(const SelectedKernelReport&) = delete;
    SelectedKernelReport(SelectedKernelReport&&) = delete;
    SelectedKernelReport& operator=(SelectedKernelReport&&) = delete;

private:
    std::string _caseName;
    const ScopedPluginLogCapture& _capture;
};

} // namespace

/// Runs only on the architecture the engine ships for.
template <typename TestCaseType>
class IntegrationGpuGfx942AttentionUnified2DBase
    : public IntegrationGraphVerificationHarness<float, TestCaseType>
{
protected:
    using Harness = IntegrationGraphVerificationHarness<float, TestCaseType>;

    void SetUp() override
    {
        Harness::SetUp();
        if(this->IsSkipped() || this->HasFatalFailure())
        {
            return;
        }

        const auto arch = hip_kernel_provider_common::getDeviceString(this->_stream);
        if(arch != SERVED_ARCH)
        {
            GTEST_SKIP() << ENGINE << " ships for " << SERVED_ARCH << " only; this device is "
                         << arch;
        }
    }

    /// Offsets the seed by UID so Q, K and V carry different data.
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

    /// NaN bytes in every workspace byte: the engine may not read what it did not write.
    void prepareWorkspace(void* workspace, size_t bytes) override
    {
        ASSERT_EQ(hipMemsetAsync(workspace, 0xFF, bytes, this->stream()), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(this->stream()), hipSuccess);
    }

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE);
    }

    /// Whether the engine is in @p graph's ranked engine list.
    bool offers(Graph& graph)
    {
        auto result = graph.build_operation_graph(this->_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        return result.code == ErrorCode::OK
               && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId())
                      != rankedEngineIds.end();
    }

    /// A cache root private to ONE test case, so no measured winner replaces the ranking.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx942-unified-2d-knobs-case",
        hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

using IntegrationGpuGfx942AttentionUnified2DKnobs
    = IntegrationGpuGfx942AttentionUnified2DBase<KnobCase>;
using IntegrationGpuGfx942AttentionUnified2DDeclines
    = IntegrationGpuGfx942AttentionUnified2DBase<DeclineCase>;
using IntegrationGpuGfx942AttentionUnified2DKnobCoverage
    = IntegrationGpuGfx942AttentionUnified2DBase<KnobCase>;

/// The knob tables register before main() names the descriptor root; this holds them to the
/// catalog. Every kernel class the catalog ships (head size, dtype, mask, geometry) must be
/// one a forced case runs, so a class reduction keeps cannot go unexecuted; a class the
/// catalog dropped is reported by the case that forces it.
TEST_F(IntegrationGpuGfx942AttentionUnified2DKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set";

    using KernelClass = std::tuple<int64_t, std::string, int64_t, Knobs>;
    const std::vector<std::string> classKnobs{"tile_size",
                                              "num_warps",
                                              "block_m_per_warp",
                                              "use_mfma_32x32x8",
                                              "use_transposed_qk_32x32"};

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        if(!knobCase.knobs.empty())
        {
            tableClasses.emplace(knobCase.shape.headSize,
                                 catalogDtypeName(knobCase.shape.dataType),
                                 BOTTOM_RIGHT_MASK_TYPE,
                                 knobCase.knobs);
        }
    }

    std::size_t served = 0;
    forEachServedKernel([&](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        ++served;
        const auto intField = [&kernel](const std::string& field) -> const int64_t* {
            const auto value = kernel.metadata.find(field);
            return value == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&value->second);
        };
        const auto dtype = kernel.metadata.find("dtype");
        const auto* dtypeName
            = dtype == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&dtype->second);
        const auto* mask = intField("mask_type");
        const auto* headSize = intField("head_size");
        Knobs knobs;
        for(const auto& knob : classKnobs)
        {
            const auto* value = intField(knob);
            if(value == nullptr)
            {
                ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                              << " carries no integer " << knob;
                return;
            }
            knobs.emplace_back(knob, *value);
        }
        ASSERT_TRUE(dtypeName != nullptr && mask != nullptr && headSize != nullptr)
            << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
            << " carries no string dtype and integer mask_type and head_size";
        EXPECT_EQ(tableClasses.count({*headSize, *dtypeName, *mask, knobs}), 1U)
            << ENGINE << " ships kernel class (D" << *headSize << ", " << *dtypeName
            << ", mask_type " << *mask << ", "
            << describeKey(MetadataKey(knobs.begin(), knobs.end())) << ") that no forced case runs";
    });
    EXPECT_GT(served, 0U) << ENGINE << " serves no kernel on " << SERVED_ARCH
                          << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                          << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                          << "'";
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the output
/// is then checked against the CPU reference, which catches a kernel launched with any
/// geometry but its own, or with head counts other than the graph's.
TEST_P(IntegrationGpuGfx942AttentionUnified2DKnobs,
       SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const SelectedKernelReport report(testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_TRUE(offers(*sdpa.graph))
        << ENGINE << " did not offer to serve this graph on " << SERVED_ARCH
        << "; its packed descriptors load through HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";

    auto result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.knobs));
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = selectedKernelId(recorder);
    ASSERT_TRUE(selected.has_value()) << "no selection line from " << ENGINE << ". Captured logs:\n"
                                      << recorder.getRecordedLogsAsString();

    // Resolved after the plan is built, so a lookup failure still leaves the selected
    // kernel in the KNOB_SELECTED line. A forced case names exactly one kernel; a cold case
    // names the kernels of its expected class and must select one.
    const auto expectedIds = kernelIdsFor(testCase);
    ASSERT_FALSE(expectedIds.empty());
    if(!testCase.knobs.empty())
    {
        ASSERT_EQ(expectedIds.size(), 1U)
            << "the forced knobs must name exactly one kernel of this graph's configuration";
        EXPECT_TRUE(recorder.hasLogContaining(std::string(SELECTED_KERNEL_MARKER)
                                              + expectedIds.front() + " at rank 0 from 1 "
                                              + "candidate(s)"))
            << "Captured logs:\n"
            << recorder.getRecordedLogsAsString();
    }
    EXPECT_NE(std::find(expectedIds.begin(), expectedIds.end(), *selected), expectedIds.end())
        << "selected " << *selected << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// A shape the engine must leave alone is not in its ranked list: no plan, so no silent
/// wrong answer from a mask, block or grid the kernel cannot honour.
TEST_P(IntegrationGpuGfx942AttentionUnified2DDeclines, IsNotOffered)
{
    const auto& testCase = GetParam();
    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    EXPECT_FALSE(offers(*sdpa.graph)) << ENGINE << " offered to serve " << testCase.name;
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified2DKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified2DDeclines,
                         ::testing::ValuesIn(declineCases()),
                         [](const ::testing::TestParamInfo<DeclineCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
