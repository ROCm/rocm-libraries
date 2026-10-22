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
#include <stdexcept>
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
 * @file IntegrationGpuGfx950AttentionUnifiedKnobs.cpp
 * @brief hipkernel:Gfx950AttentionUnified2D and hipkernel:Gfx950AttentionUnified3D end to end
 *        through the frontend on a gfx950 device: each catalog kernel class is selected by
 *        its knobs, launched with its own geometry (and, for 3D, both of its launches) and
 *        checked against the CPU reference; the shapes each engine must leave alone are not
 *        offered.
 *
 * Unified2D's knobs are its geometry (tile_size, num_warps, block_m_per_warp and the MFMA
 * codepath flags); Unified3D's is num_segments. Each expected kernel id is resolved at run
 * time from the production descriptors (expectedKernelId), so a reduced catalog is checked
 * against the kernels it actually ships; the coverage cases hold the tables to exactly the
 * catalog's kernel classes. Every 3D case runs with a workspace filled with NaN bytes first:
 * hipDNN does not zero it, and the split-KV pair must not read anything it did not write.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_2D = "hipkernel:Gfx950AttentionUnified2D";
constexpr const char* ENGINE_3D = "hipkernel:Gfx950AttentionUnified3D";

constexpr const char* SERVED_ARCH = "gfx950";

/// The heuristic path's selection line (GenericPlanBuilder::buildPlan), which names the
/// kernel id as the kdp spells it: lowercase, hyphenated.
constexpr const char* SELECTED_KERNEL_MARKER = "' selected kernel ";
constexpr size_t KERNEL_ID_LENGTH = 36;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

/// The masks both engines serve, spelled as diagonal bounds. mask_type numbering is
/// AttentionMaskType's: 0 none, 1 top-left causal, 2 bottom-right causal.
enum class Mask
{
    NO_MASK,
    TOP_LEFT,
    BOTTOM_RIGHT
};

int64_t maskType(Mask mask)
{
    switch(mask)
    {
    case Mask::NO_MASK:
        return 0;
    case Mask::TOP_LEFT:
        return 1;
    case Mask::BOTTOM_RIGHT:
        return 2;
    default:
        throw std::invalid_argument("maskType: unhandled Mask value");
    }
}

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
    const char* engine;
    GraphShape shape;
    Knobs knobs;
    Knobs expected;
};

/// One graph the engine must not offer to serve, and why.
struct DeclineCase
{
    std::string name;
    const char* engine;
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

/// The Unified2D geometries the manifest pins: (tile_size at block_size 16, num_warps,
/// block_m_per_warp, use_mfma_32x32, use_transposed_qk_32x32), named for the case.
struct Geometry2D
{
    const char* name;
    int64_t tileSize;
    int64_t numWarps;
    int64_t blockMPerWarp;
    int64_t useMfma32x32;
    int64_t useTransposedQk32x32;
};

constexpr std::array<Geometry2D, 4> GEOMETRIES_2D{{
    {"Narrow4x16T64", 64, 4, 16, 0, 0},
    {"Narrow2x16T32", 32, 2, 16, 0, 0},
    {"Wide32x4T64", 64, 4, 32, 1, 0},
    {"Transposed32x4T64", 64, 4, 32, 1, 1},
}};

constexpr std::array<int64_t, 3> SEGMENTS_3D{16, 32, 64};

/// Head configurations the cases cycle through: MHA, GQA and MQA at both head sizes.
struct Heads
{
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
};

constexpr std::array<Heads, 6> HEADS_2D{
    {{128, 8, 2}, {64, 16, 2}, {128, 32, 8}, {64, 8, 8}, {128, 8, 1}, {64, 32, 4}}};
constexpr std::array<Heads, 4> HEADS_3D{{{128, 8, 2}, {64, 16, 2}, {128, 32, 8}, {64, 8, 1}}};

std::string dtypeName(DataType dataType)
{
    return dataType == DataType::HALF ? "Fp16" : "Bf16";
}

std::string maskName(Mask mask)
{
    switch(mask)
    {
    case Mask::NO_MASK:
        return "NoMask";
    case Mask::TOP_LEFT:
        return "TopLeft";
    case Mask::BOTTOM_RIGHT:
        return "BottomRight";
    default:
        throw std::invalid_argument("maskName: unhandled Mask value");
    }
}

/// Every Unified2D kernel class -- geometry x dtype x mask -- forced once, on a prefill shape
/// with B > 1 and Sq != Skv that cycles through the head configurations; then one cold case,
/// which the engine's ranking serves with the baseline geometry.
std::vector<KnobCase> knobCases2D()
{
    std::vector<KnobCase> cases;
    std::size_t at = 0;
    for(const auto& g : GEOMETRIES_2D)
    {
        for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
        {
            for(const auto mask : {Mask::NO_MASK, Mask::TOP_LEFT, Mask::BOTTOM_RIGHT})
            {
                const auto& h = HEADS_2D[at++ % std::size(HEADS_2D)];
                const Knobs knobs{{"tile_size", g.tileSize},
                                  {"num_warps", g.numWarps},
                                  {"block_m_per_warp", g.blockMPerWarp},
                                  {"use_mfma_32x32", g.useMfma32x32},
                                  {"use_transposed_qk_32x32", g.useTransposedQk32x32}};
                cases.push_back({std::string("Unified2D_") + g.name + "_" + dtypeName(dataType)
                                     + "_" + maskName(mask),
                                 ENGINE_2D,
                                 {dataType, h.headSize, h.queryHeads, h.kvHeads, mask, 2, 96, 256},
                                 knobs,
                                 knobs});
            }
        }
    }
    const auto& baseline = GEOMETRIES_2D[0];
    cases.push_back({"Unified2D_Cold_SelectsTheBaselineGeometry",
                     ENGINE_2D,
                     {DataType::BFLOAT16, 128, 8, 2, Mask::TOP_LEFT, 3, 512, 512},
                     {},
                     {{"tile_size", baseline.tileSize},
                      {"num_warps", baseline.numWarps},
                      {"block_m_per_warp", baseline.blockMPerWarp},
                      {"use_mfma_32x32", baseline.useMfma32x32},
                      {"use_transposed_qk_32x32", baseline.useTransposedQk32x32}}});
    return cases;
}

/// Every Unified3D kernel class -- num_segments x dtype x causal mask -- forced once on a
/// decode shape (Sq 1); a NO_MASK decode graph (served by the bottom-right kernel); and a
/// cold B = 1024 decode case, past any 2D batch capacity: the decode grid has none.
std::vector<KnobCase> knobCases3D()
{
    std::vector<KnobCase> cases;
    std::size_t at = 0;
    for(const auto segments : SEGMENTS_3D)
    {
        for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
        {
            for(const auto mask : {Mask::TOP_LEFT, Mask::BOTTOM_RIGHT})
            {
                const auto& h = HEADS_3D[at++ % std::size(HEADS_3D)];
                const Knobs knobs{{"num_segments", segments}};
                cases.push_back({"Unified3D_Seg" + std::to_string(segments) + "_"
                                     + dtypeName(dataType) + "_" + maskName(mask),
                                 ENGINE_3D,
                                 {dataType, h.headSize, h.queryHeads, h.kvHeads, mask, 3, 1, 1024},
                                 knobs,
                                 knobs});
            }
        }
    }
    cases.push_back({"Unified3D_NoMaskDecodeIsTheBottomRightKernel",
                     ENGINE_3D,
                     {DataType::BFLOAT16, 128, 8, 2, Mask::NO_MASK, 2, 1, 512},
                     {{"num_segments", SEGMENTS_3D[0]}},
                     {{"num_segments", SEGMENTS_3D[0]}}});
    cases.push_back({"Unified3D_Cold_Batch1024",
                     ENGINE_3D,
                     {DataType::HALF, 64, 8, 1, Mask::BOTTOM_RIGHT, 1024, 1, 256},
                     {},
                     {}});
    return cases;
}

std::vector<KnobCase> knobCases()
{
    auto cases = knobCases2D();
    auto decode = knobCases3D();
    cases.insert(cases.end(), decode.begin(), decode.end());
    return cases;
}

/// Shapes each engine must refuse at graph or kernel match, so it is not offered at all.
std::vector<DeclineCase> declineCases()
{
    return {
        // 2D bakes a batch capacity of 511 into its binary search.
        {"Unified2D_BatchBeyondTheBakedCapacity",
         ENGINE_2D,
         {DataType::BFLOAT16, 128, 8, 2, Mask::NO_MASK, 512, 16, 64}},
        // The identity cache is cut into whole 16-token blocks.
        {"Unified2D_SeqKvNotAWholeNumberOfBlocks",
         ENGINE_2D,
         {DataType::BFLOAT16, 128, 8, 2, Mask::NO_MASK, 1, 64, 264}},
        // Decode with a GQA group dividing 16 is Unified3D's.
        {"Unified2D_LeavesDecodeToUnified3D",
         ENGINE_2D,
         {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 2, 1, 512}},
        // The decode grid leaves a longer query's rows unwritten.
        {"Unified3D_Prefill",
         ENGINE_3D,
         {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 2, 64, 512}},
        {"Unified3D_SeqKvNotAWholeNumberOfBlocks",
         ENGINE_3D,
         {DataType::BFLOAT16, 128, 8, 2, Mask::BOTTOM_RIGHT, 2, 1, 264}},
    };
}

/// BSHD: token-major, head varying fastest. Both engines bake this layout.
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

/// The forward SDPA tolerance the engines' SdpaFwd bundles are graded at (neither TOML
/// names an override).
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

/// Calls @p visit for every kernel runtimeCatalog() serves @p engine with on SERVED_ARCH.
template <typename Visit>
void forEachServedKernel(const char* engine, Visit&& visit)
{
    const auto& catalog = runtimeCatalog();
    for(const auto& entry : catalog.packs)
    {
        const auto& pack = entry.second;
        const auto& arch = pack.descriptor.arch;
        const auto found = catalog.engines.find(pack.descriptor.engineId);
        if(pack.conflicted || found == catalog.engines.end()
           || found->second.descriptor.name != engine
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

/// The mask_type a case's kernel carries: decode has no NO_MASK kernel, so Unified3D serves
/// a NO_MASK decode graph with its bottom-right causal one.
int64_t catalogMaskType(const KnobCase& knobCase)
{
    if(std::string_view(knobCase.engine) == ENGINE_3D && knobCase.shape.mask == Mask::NO_MASK)
    {
        return maskType(Mask::BOTTOM_RIGHT);
    }
    return maskType(knobCase.shape.mask);
}

/// The ids of the kernels @p knobCase's engine carries for its graph fields and expected
/// metadata, as the selection line spells them.
std::vector<std::string> kernelIdsFor(const KnobCase& knobCase)
{
    MetadataKey key{
        {"dtype", catalogDtypeName(knobCase.shape.dataType)},
        {"head_size", knobCase.shape.headSize},
        {"num_query_heads", knobCase.shape.queryHeads},
        {"num_kv_heads", knobCase.shape.kvHeads},
        {"mask_type", catalogMaskType(knobCase)},
        {"sliding_window", int64_t{0}},
    };
    for(const auto& [field, value] : knobCase.expected)
    {
        key.emplace_back(field, value);
    }

    std::vector<std::string> ids;
    forEachServedKernel(
        knobCase.engine, [&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            const bool carriesKey
                = std::all_of(key.begin(), key.end(), [&kernel](const auto& field) {
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
        ADD_FAILURE() << "no " << knobCase.engine << " kernel for " << SERVED_ARCH << " carries "
                      << describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
    }
    return ids;
}

/// The kernel id the plugin's selection line for @p engine names, or nullopt.
std::optional<std::string>
    selectedKernelId(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder,
                     const char* engine)
{
    const std::string marker = std::string("engine '") + engine + SELECTED_KERNEL_MARKER;
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
    SelectedKernelReport(std::string caseName,
                         const char* engine,
                         const ScopedPluginLogCapture& capture)
        : _caseName(std::move(caseName))
        , _engine(engine)
        , _capture(capture)
    {
    }

    ~SelectedKernelReport()
    {
        std::cout << "KNOB_SELECTED " << _caseName << " "
                  << selectedKernelId(_capture.recorder(), _engine).value_or("none") << '\n';
    }

    SelectedKernelReport(const SelectedKernelReport&) = delete;
    SelectedKernelReport& operator=(const SelectedKernelReport&) = delete;
    SelectedKernelReport(SelectedKernelReport&&) = delete;
    SelectedKernelReport& operator=(SelectedKernelReport&&) = delete;

private:
    std::string _caseName;
    const char* _engine;
    const ScopedPluginLogCapture& _capture;
};

} // namespace

/// Runs only on the architecture both engines ship for.
template <typename TestCaseType>
class IntegrationGpuGfx950AttentionUnifiedBase
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
            GTEST_SKIP() << ENGINE_2D << " and " << ENGINE_3D << " ship for " << SERVED_ARCH
                         << " only; this device is " << arch;
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

    /// NaN bytes in every workspace byte: neither engine may read what it did not write.
    void prepareWorkspace(void* workspace, size_t bytes) override
    {
        ASSERT_EQ(hipMemsetAsync(workspace, 0xFF, bytes, this->stream()), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(this->stream()), hipSuccess);
    }

    static int64_t engineId(const char* engine)
    {
        return hipdnn_data_sdk::utilities::engineNameToId(engine);
    }

    /// Whether @p engine is in @p graph's ranked engine list.
    bool offers(Graph& graph, const char* engine)
    {
        auto result = graph.build_operation_graph(this->_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        return result.code == ErrorCode::OK
               && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId(engine))
                      != rankedEngineIds.end();
    }

    /// A cache root private to ONE test case, so no measured winner replaces the ranking.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx950-unified-knobs-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

using IntegrationGpuGfx950AttentionUnifiedKnobs
    = IntegrationGpuGfx950AttentionUnifiedBase<KnobCase>;
using IntegrationGpuGfx950AttentionUnifiedDeclines
    = IntegrationGpuGfx950AttentionUnifiedBase<DeclineCase>;
using IntegrationGpuGfx950AttentionUnifiedKnobCoverage
    = IntegrationGpuGfx950AttentionUnifiedBase<KnobCase>;
using IntegrationGpuGfx950AttentionUnifiedPrecedence
    = IntegrationGpuGfx950AttentionUnifiedBase<KnobCase>;

/// The knob tables register before main() names the descriptor root; this holds them to the
/// catalog. Every kernel class the catalog ships -- 2D (dtype, mask, geometry), 3D (dtype,
/// mask, num_segments) -- must be one a forced case runs, so a class reduction keeps cannot
/// go unexecuted; a class the catalog dropped is reported by the case that forces it.
TEST_F(IntegrationGpuGfx950AttentionUnifiedKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set";

    using KernelClass = std::tuple<std::string, std::string, int64_t, Knobs>;
    const auto classKnobs = [](const char* engine) -> std::vector<std::string> {
        if(std::string_view(engine) == ENGINE_3D)
        {
            return {"num_segments"};
        }
        return {"tile_size",
                "num_warps",
                "block_m_per_warp",
                "use_mfma_32x32",
                "use_transposed_qk_32x32"};
    };

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        if(!knobCase.knobs.empty())
        {
            tableClasses.emplace(knobCase.engine,
                                 catalogDtypeName(knobCase.shape.dataType),
                                 catalogMaskType(knobCase),
                                 knobCase.knobs);
        }
    }

    for(const char* engine : {ENGINE_2D, ENGINE_3D})
    {
        std::size_t served = 0;
        forEachServedKernel(
            engine, [&](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
                ++served;
                const auto intField = [&kernel](const std::string& field) -> const int64_t* {
                    const auto value = kernel.metadata.find(field);
                    return value == kernel.metadata.end() ? nullptr
                                                          : std::get_if<int64_t>(&value->second);
                };
                const auto dtype = kernel.metadata.find("dtype");
                const auto* dtypeName = dtype == kernel.metadata.end()
                                            ? nullptr
                                            : std::get_if<std::string>(&dtype->second);
                const auto* mask = intField("mask_type");
                Knobs knobs;
                for(const auto& knob : classKnobs(engine))
                {
                    const auto* value = intField(knob);
                    if(value == nullptr)
                    {
                        ADD_FAILURE()
                            << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                            << " carries no integer " << knob;
                        return;
                    }
                    knobs.emplace_back(knob, *value);
                }
                ASSERT_TRUE(dtypeName != nullptr && mask != nullptr)
                    << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                    << " carries no string dtype and integer mask_type";
                EXPECT_EQ(tableClasses.count({engine, *dtypeName, *mask, knobs}), 1U)
                    << engine << " ships kernel class (" << *dtypeName << ", mask_type " << *mask
                    << ", " << describeKey(MetadataKey(knobs.begin(), knobs.end()))
                    << ") that no forced case runs";
            });
        EXPECT_GT(served, 0U) << engine << " serves no kernel on " << SERVED_ARCH
                              << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                              << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                              << "'";
    }
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the output
/// is then checked against the CPU reference, which catches a kernel launched with any
/// geometry but its own (and, for 3D, a reduce that read an unwritten workspace byte).
TEST_P(IntegrationGpuGfx950AttentionUnifiedKnobs, SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const SelectedKernelReport report(testCase.name, testCase.engine, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_TRUE(offers(*sdpa.graph, testCase.engine))
        << testCase.engine << " did not offer to serve this graph on " << SERVED_ARCH
        << "; its packed descriptors load through HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";

    auto result = sdpa.graph->create_execution_plan_ext(engineId(testCase.engine),
                                                        knobSettingsFor(testCase.knobs));
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = selectedKernelId(recorder, testCase.engine);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << testCase.engine << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // Resolved after the plan is built, so a lookup failure still leaves the selected
    // kernel in the KNOB_SELECTED line. A forced case names exactly one kernel; a cold case
    // names the kernels of its expected class (all of them, when it expects nothing more
    // than the graph's fields) and must select one.
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

/// Adding the unified engines does not change what an unpinned graph gets: on a prefill shape
/// both catalogs serve, hipkernel:Gfx950AttentionDense still ranks ahead of Unified2D.
TEST_F(IntegrationGpuGfx950AttentionUnifiedPrecedence, DenseStillRanksAheadWhenUnpinned)
{
    auto sdpa = buildSdpaGraph("DenseStillRanksAheadWhenUnpinned",
                               {DataType::BFLOAT16, 128, 8, 2, Mask::NO_MASK, 1, 512, 512});
    auto result = sdpa.graph->build_operation_graph(this->_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<int64_t> rankedEngineIds;
    result = sdpa.graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto dense = std::find(
        rankedEngineIds.begin(), rankedEngineIds.end(), engineId("hipkernel:Gfx950AttentionDense"));
    const auto unified
        = std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId(ENGINE_2D));
    ASSERT_NE(dense, rankedEngineIds.end()) << "hipkernel:Gfx950AttentionDense did not offer";
    ASSERT_NE(unified, rankedEngineIds.end()) << ENGINE_2D << " did not offer";
    EXPECT_LT(dense, unified);
}

/// A shape the engine must leave alone is not in its ranked list: no plan, so no silent
/// wrong answer from a capacity, block or grid the kernel cannot honour.
TEST_P(IntegrationGpuGfx950AttentionUnifiedDeclines, IsNotOffered)
{
    const auto& testCase = GetParam();
    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    EXPECT_FALSE(offers(*sdpa.graph, testCase.engine))
        << testCase.engine << " offered to serve " << testCase.name;
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionUnifiedKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionUnifiedDeclines,
                         ::testing::ValuesIn(declineCases()),
                         [](const ::testing::TestParamInfo<DeclineCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
