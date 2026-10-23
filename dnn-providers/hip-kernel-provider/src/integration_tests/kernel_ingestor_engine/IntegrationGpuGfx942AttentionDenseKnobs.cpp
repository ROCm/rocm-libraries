// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
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
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
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
 * @file IntegrationGpuGfx942AttentionDenseKnobs.cpp
 * @brief hipkernel:Gfx942AttentionDense end to end through the frontend on a gfx942
 *        device: a forced block_m/block_n selects exactly the policy-arm kernel of that
 *        tile, that kernel launches with its own geometry and computes what the CPU
 *        reference computes, and a knob pair no kernel has is refused before any plan.
 *
 * The case table is generated so it runs every (head size, dtype, mask, window, tile)
 * class the catalog compiles, at a shape the tile divides; a coverage case asserts the
 * table and the installed catalog agree. Head counts vary across rows because the body
 * reads them at runtime. Window rows run Sq = Skv = 2W, so the window really excludes
 * keys, and spell the graph's left_bound as W - 1 (the reference keeps left_bound + 1
 * keys). Hand-written rows add cold selection, B > 1, Sq != Skv, both alignments, the
 * deprecated causal flag and the default and a non-default scale.
 *
 * Each expected kernel id is resolved at run time from the production descriptors: the
 * one kernel of the class whose lds_row_pad and pv_priority are the policy arms, which
 * the engine's score ranks first among the arms sharing a tile.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx942AttentionDense";

constexpr const char* SERVED_ARCH = "gfx942";

constexpr const char* BLOCK_M_KNOB = "block_m";
constexpr const char* BLOCK_N_KNOB = "block_n";

/// The policy arms of the two knobs the engine does not expose (the score's first rank).
constexpr int64_t POLICY_LDS_ROW_PAD = 8;
constexpr int64_t POLICY_PV_PRIORITY = 0;

/// The heuristic path's selection line (GenericPlanBuilder::buildPlan), which names the
/// kernel id as the kdp spells it: lowercase, hyphenated.
constexpr const char* SELECTED_KERNEL_MARKER = "' selected kernel ";
constexpr size_t KERNEL_ID_LENGTH = 36;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;

/// Every spelling of the masks the engine serves.
enum class Mask
{
    NO_MASK,
    BOUNDS_TOP_LEFT,
    BOUNDS_BOTTOM_RIGHT,
    CAUSAL_MASK_FLAG,
    /// left_bound W - 1, right_bound 0: a causal window of W keys.
    WINDOW_TOP_LEFT,
    WINDOW_BOTTOM_RIGHT
};

struct GraphShape
{
    DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    Mask mask;
    /// The window in keys for the WINDOW_* masks; 0 otherwise.
    int64_t window;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
    /// Leaves attn_scale_value unset, so the engine must apply 1.0 (no scaling).
    bool omitScale = false;
    std::optional<float> scaleOverride = std::nullopt;
};

struct Tile
{
    int64_t blockM;
    int64_t blockN;
};

/// One graph and the tile the engine must serve it with. A forced case sets block_m and
/// block_n to the tile; a cold case sets no knob, and the tile is the one the engine's own
/// ranking picks.
struct KnobCase
{
    std::string name;
    GraphShape shape;
    bool forced;
    Tile tile;
};

/// One graph and a knob pair whose values the engine advertises for it one by one while
/// no kernel it admits carries both.
struct UnsatisfiableCase
{
    std::string name;
    GraphShape shape;
    Tile forcedTile;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const UnsatisfiableCase& unsatisfiableCase, std::ostream* os)
{
    *os << unsatisfiableCase.name;
}

GraphShape makeShape(DataType dataType,
                     int64_t headSize,
                     int64_t queryHeads,
                     int64_t kvHeads,
                     Mask mask,
                     int64_t window,
                     int64_t batch,
                     int64_t seqQ,
                     int64_t seqKv)
{
    return {dataType, headSize, queryHeads, kvHeads, mask, window, batch, seqQ, seqKv};
}

/// The tiles the reduced catalog keeps for a cell (configs/gfx942_attention_dense.yaml).
/// Window cells keep their floor -- the 256/64 default and the dividing tiles 64/64 and
/// 256/32 -- and the causal and unmasked cells keep the tiles the reduction sweep ranked.
std::vector<Tile> tilesFor(int64_t headSize, DataType dataType, bool causal, int64_t window)
{
    std::vector<Tile> tiles{{64, 64}, {256, 32}, {256, 64}};
    if(window != 0)
    {
        return tiles;
    }
    tiles.push_back({128, 64});
    if(headSize == 64 && dataType == DataType::BFLOAT16)
    {
        tiles.push_back({256, 128});
    }
    if(headSize == 64 && !causal)
    {
        tiles.push_back({512, 64});
    }
    return tiles;
}

/// The catalog's mask cells: causal, unmasked, and causal windows of these sizes.
const std::vector<int64_t>& catalogWindows()
{
    static const std::vector<int64_t> s_windows{128, 512, 1024, 2048};
    return s_windows;
}

std::string tileName(const Tile& tile)
{
    return "Bm" + std::to_string(tile.blockM) + "Bn" + std::to_string(tile.blockN);
}

/// Every catalog class once, forced to its tile, then the hand-written cases.
std::vector<KnobCase> knobCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr bool FORCED = true;
    constexpr bool COLD = false;

    // MHA, GQA and MQA head configurations, rotated across the class rows.
    const std::pair<int64_t, int64_t> heads[] = {{8, 8}, {16, 2}, {8, 1}, {4, 4}};

    std::vector<KnobCase> cases;
    std::size_t row = 0;
    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        for(const auto dataType : {BF16, FP16})
        {
            const std::string prefix = std::string(headSize == 64 ? "D64_" : "D128_")
                                       + (dataType == BF16 ? "Bf16_" : "Fp16_");
            for(const auto& tile : tilesFor(headSize, dataType, true, 0))
            {
                const auto& [hq, hkv] = heads[row++ % std::size(heads)];
                cases.push_back({prefix + "Causal_" + tileName(tile),
                                 makeShape(dataType, headSize, hq, hkv, Mask::BOUNDS_TOP_LEFT, 0, 1, 512, 512),
                                 FORCED,
                                 tile});
            }
            for(const auto& tile : tilesFor(headSize, dataType, false, 0))
            {
                const auto& [hq, hkv] = heads[row++ % std::size(heads)];
                cases.push_back({prefix + "NoMask_" + tileName(tile),
                                 makeShape(dataType, headSize, hkv, hkv, Mask::NO_MASK, 0, 1, 512, 512),
                                 FORCED,
                                 tile});
            }
            for(const int64_t window : catalogWindows())
            {
                const int64_t seq = std::max<int64_t>(512, 2 * window);
                for(const auto& tile : tilesFor(headSize, dataType, true, window))
                {
                    cases.push_back({prefix + "Window" + std::to_string(window) + "_" + tileName(tile),
                                     makeShape(dataType, headSize, 1, 1, Mask::WINDOW_TOP_LEFT, window, 1, seq, seq),
                                     FORCED,
                                     tile});
                }
            }
        }
    }

    // Cold: no knob, so the 256/64 policy kernel where it divides, and otherwise the
    // smallest dividing tile, which the score ranks next.
    cases.push_back({"Cold_Causal",
                     makeShape(BF16, 128, 32, 8, Mask::BOUNDS_TOP_LEFT, 0, 2, 1024, 1024),
                     COLD,
                     Tile{256, 64}});
    cases.push_back({"Cold_Seq384",
                     makeShape(FP16, 64, 8, 8, Mask::NO_MASK, 0, 1, 384, 384),
                     COLD,
                     Tile{64, 64}});
    // Shape breadth on a forced tile: B > 1, Sq != Skv, both alignments at equal lengths,
    // the deprecated causal flag, a top-left window at unequal lengths.
    cases.push_back({"Shape_Batch3CrossLength",
                     makeShape(FP16, 128, 16, 2, Mask::NO_MASK, 0, 3, 256, 768),
                     FORCED,
                     Tile{256, 64}});
    cases.push_back({"Shape_BottomRightEqualLengths",
                     makeShape(BF16, 64, 8, 8, Mask::BOUNDS_BOTTOM_RIGHT, 0, 2, 512, 512),
                     FORCED,
                     Tile{128, 64}});
    cases.push_back({"Shape_CausalFlag",
                     makeShape(FP16, 128, 8, 2, Mask::CAUSAL_MASK_FLAG, 0, 1, 512, 512),
                     FORCED,
                     Tile{256, 32}});
    cases.push_back({"Shape_WindowBottomRightEqualLengths",
                     makeShape(BF16, 128, 4, 2, Mask::WINDOW_BOTTOM_RIGHT, 128, 1, 512, 512),
                     FORCED,
                     Tile{256, 64}});
    cases.push_back({"Shape_WindowTopLeftCrossLength",
                     makeShape(FP16, 64, 4, 1, Mask::WINDOW_TOP_LEFT, 512, 2, 1024, 1536),
                     FORCED,
                     Tile{256, 64}});
    // The scale: absent means 1.0; a non-default value must reach the kernel.
    auto noScale = makeShape(BF16, 128, 8, 8, Mask::BOUNDS_TOP_LEFT, 0, 1, 256, 256);
    noScale.omitScale = true;
    cases.push_back({"Scale_Omitted", noScale, FORCED, Tile{256, 64}});
    auto otherScale = makeShape(FP16, 64, 8, 8, Mask::NO_MASK, 0, 1, 256, 256);
    otherScale.scaleOverride = 0.03F;
    cases.push_back({"Scale_NonDefault", otherScale, FORCED, Tile{64, 64}});
    return cases;
}

/// D128 at 512 admits 64/64, 128/64, 256/32 and 256/64, so block_m=128 and block_n=32
/// are each advertised while no D128 kernel is 128/32.
std::vector<UnsatisfiableCase> unsatisfiableCases()
{
    return {
        {"D128_Bm128Bn32",
         makeShape(DataType::HALF, 128, 8, 8, Mask::NO_MASK, 0, 1, 512, 512),
         Tile{128, 32}},
    };
}

/// BSHD: token-major, head varying fastest. The kernel bakes this layout.
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

/// A single SDPA-forward node with BSHD Q/K/V/O. O is laid out explicitly: the frontend's
/// default for an unset output is packed BHSD, which the engine declines.
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
    if(!shape.omitScale)
    {
        attributes.set_attn_scale(
            shape.scaleOverride.value_or(1.0f / std::sqrt(static_cast<float>(shape.headSize))));
    }
    switch(shape.mask)
    {
    case Mask::NO_MASK:
        break;
    case Mask::BOUNDS_TOP_LEFT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        break;
    case Mask::BOUNDS_BOTTOM_RIGHT:
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        break;
    case Mask::CAUSAL_MASK_FLAG:
        attributes.causal_mask = true;
        break;
    case Mask::WINDOW_TOP_LEFT:
        attributes.set_diagonal_band_left_bound(shape.window - 1);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        break;
    case Mask::WINDOW_BOTTOM_RIGHT:
        attributes.set_diagonal_band_left_bound(shape.window - 1);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        break;
    default:
        throw std::invalid_argument("buildSdpaGraph: unhandled Mask value");
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

/// The forward SDPA tolerance every SdpaFwd bundle of this engine is graded at
/// (config/hipkernel_Gfx942AttentionDense.toml names no override).
float sdpaForwardTolerance(DataType dataType)
{
    return dataType == DataType::HALF
               ? hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>()
               : hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<
                     hipdnn_data_sdk::types::bfloat16>();
}

std::vector<KnobSetting> knobSettingsFor(const Tile& tile)
{
    std::vector<KnobSetting> settings;
    settings.emplace_back(BLOCK_M_KNOB, tile.blockM);
    settings.emplace_back(BLOCK_N_KNOB, tile.blockN);
    return settings;
}

/// A kernel metadata field and the value a case expects it to carry.
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

std::string catalogDtypeName(DataType dataType)
{
    return dataType == DataType::HALF ? "FP16" : "BF16";
}

int64_t catalogCausal(Mask mask)
{
    return mask == Mask::NO_MASK ? 0 : 1;
}

/// The id of the one rank-zero kernel this engine's gfx942 packs carry for @p shape's
/// semantic fields and @p tile, as the selection line spells it.
std::optional<std::string> expectedKernelId(const GraphShape& shape, const Tile& tile)
{
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
    {
        ADD_FAILURE() << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production "
                         "descriptors to read the expected kernel id from";
        return std::nullopt;
    }

    const MetadataKey key{
        {"dtype", catalogDtypeName(shape.dataType)},
        {"head_size", shape.headSize},
        {"causal", catalogCausal(shape.mask)},
        {"ragged", int64_t{0}},
        {"sliding_window", shape.window},
        {"block_m", tile.blockM},
        {"block_n", tile.blockN},
        {"lds_row_pad", POLICY_LDS_ROW_PAD},
        {"pv_priority", POLICY_PV_PRIORITY},
    };

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

    if(ids.size() != 1)
    {
        ADD_FAILURE() << ids.size() << " " << ENGINE_NAME << " kernel(s) for " << SERVED_ARCH
                      << " carry " << describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                      << "'; exactly one must";
        return std::nullopt;
    }
    return ids.front();
}

/// The kernel id the plugin's selection line names, or nullopt when no plan was built.
std::optional<std::string>
    selectedKernelId(const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder)
{
    const std::string marker = std::string("engine '") + ENGINE_NAME + SELECTED_KERNEL_MARKER;
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
    SelectedKernelReport(const std::string& caseName, const ScopedPluginLogCapture& capture)
        : _caseName(caseName)
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

bool advertises(const std::vector<Knob>& knobs, const std::string& knobId, int64_t value)
{
    const auto knob = std::find_if(knobs.begin(), knobs.end(), [&](const Knob& candidate) {
        return candidate.knobId() == knobId;
    });
    if(knob == knobs.end())
    {
        return false;
    }
    const auto* constraint = dynamic_cast<const IntConstraint*>(knob->constraint());
    return constraint != nullptr && constraint->getValidValues().count(value) > 0;
}

} // namespace

/// Runs only on the architecture the engine ships for.
template <typename TestCaseType>
class IntegrationGpuGfx942AttentionDenseBase
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
            GTEST_SKIP() << ENGINE_NAME << " ships for " << SERVED_ARCH << " only; this device is "
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

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    void buildOperationGraphTheEngineOffersToServe(Graph& graph)
    {
        auto result = graph.build_operation_graph(this->_handle);
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

    /// A cache root private to ONE test case, so no other case's winner record is reused.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx942-dense-knobs-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

using IntegrationGpuGfx942AttentionDenseKnobs = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx942AttentionDenseKnobFilter
    = IntegrationGpuGfx942AttentionDenseBase<UnsatisfiableCase>;
using IntegrationGpuGfx942AttentionDenseKnobCoverage
    = IntegrationGpuGfx942AttentionDenseBase<KnobCase>;

/// Every row runs one compiled kernel, and what the compiler specializes on is the head
/// size, dtype, causal flag, window and tile. The table must run every such class the
/// catalog carries at its policy arms, and nothing the catalog lacks.
TEST_F(IntegrationGpuGfx942AttentionDenseKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set, so there are no production descriptors to "
           "read the catalog's kernel classes from";

    using KernelClass = std::tuple<int64_t, std::string, int64_t, int64_t, int64_t, int64_t>;

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        tableClasses.emplace(knobCase.shape.headSize,
                             catalogDtypeName(knobCase.shape.dataType),
                             catalogCausal(knobCase.shape.mask),
                             knobCase.shape.window,
                             knobCase.tile.blockM,
                             knobCase.tile.blockN);
    }

    std::set<KernelClass> catalogClasses;
    forEachServedKernel([&catalogClasses](
                            const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
        const auto intField = [&kernel](const char* field) -> const int64_t* {
            const auto value = kernel.metadata.find(field);
            return value == kernel.metadata.end() ? nullptr : std::get_if<int64_t>(&value->second);
        };
        const auto dtype = kernel.metadata.find("dtype");
        const auto* dtypeName
            = dtype == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&dtype->second);
        const auto* headSize = intField("head_size");
        const auto* causal = intField("causal");
        const auto* window = intField("sliding_window");
        const auto* blockM = intField("block_m");
        const auto* blockN = intField("block_n");
        if(dtypeName == nullptr || headSize == nullptr || causal == nullptr || window == nullptr
           || blockM == nullptr || blockN == nullptr)
        {
            ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                          << " carries no string dtype and integer head_size, causal, "
                             "sliding_window, block_m and block_n";
            return;
        }
        catalogClasses.emplace(*headSize, *dtypeName, *causal, *window, *blockM, *blockN);
    });

    const auto onlyIn = [](const std::set<KernelClass>& from, const std::set<KernelClass>& other) {
        std::string text;
        for(const auto& kernelClass : from)
        {
            if(other.count(kernelClass) == 0)
            {
                const auto& [headSize, dtype, causal, window, blockM, blockN] = kernelClass;
                text += " (D" + std::to_string(headSize) + ", " + dtype + ", causal "
                        + std::to_string(causal) + ", window " + std::to_string(window) + ", "
                        + std::to_string(blockM) + "/" + std::to_string(blockN) + ")";
            }
        }
        return text.empty() ? std::string(" none") : text;
    };
    EXPECT_EQ(tableClasses, catalogClasses)
        << "in the catalog but not knobCases():" << onlyIn(catalogClasses, tableClasses)
        << "\nin knobCases() but not the catalog:" << onlyIn(tableClasses, catalogClasses)
        << "\nunder HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
        << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the
/// output is then checked against the CPU reference, which catches a kernel launched with
/// any geometry, window or head count but its own.
TEST_P(IntegrationGpuGfx942AttentionDenseKnobs, SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const SelectedKernelReport report(testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    const auto settings
        = testCase.forced ? knobSettingsFor(testCase.tile) : std::vector<KnobSetting>{};
    auto result = sdpa.graph->create_execution_plan_ext(engineId(), settings);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = selectedKernelId(recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE_NAME << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const auto expectedId = expectedKernelId(testCase.shape, testCase.tile);
    ASSERT_TRUE(expectedId.has_value());
    EXPECT_EQ(*selected, *expectedId) << "Captured logs:\n" << recorder.getRecordedLogsAsString();

    // Rank 0: the policy-arm kernel heads the ranking, rather than being reached past a
    // kernel that failed to load.
    const auto selectionLine = std::string(SELECTED_KERNEL_MARKER) + *expectedId + " at rank 0";
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// A pair of advertised knob values that no kernel carries reaches the plugin's knob
/// filter, whose refusal is the only thing between it and a plan built from nothing.
TEST_P(IntegrationGpuGfx942AttentionDenseKnobFilter, RefusesAKnobPairNoKernelCarriesAndBuildsNoPlan)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    std::vector<Knob> knobs;
    auto result = sdpa.graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_TRUE(advertises(knobs, BLOCK_M_KNOB, testCase.forcedTile.blockM))
        << BLOCK_M_KNOB << "=" << testCase.forcedTile.blockM
        << " is not advertised for this graph, so the frontend refuses it before the plugin";
    ASSERT_TRUE(advertises(knobs, BLOCK_N_KNOB, testCase.forcedTile.blockN))
        << BLOCK_N_KNOB << "=" << testCase.forcedTile.blockN
        << " is not advertised for this graph, so the frontend refuses it before the plugin";

    result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.forcedTile));
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;

    const std::string refusal
        = std::string("engine '") + ENGINE_NAME + "' has no kernel satisfying";
    EXPECT_NE(result.err_msg.find(refusal), std::string::npos)
        << "expected '" << refusal << "' in: " << result.err_msg;

    result = sdpa.graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    int64_t planEngineId = 0;
    EXPECT_NE(sdpa.graph->get_execution_plan_engine_id(planEngineId).code, ErrorCode::OK)
        << "a plan backed by engine " << planEngineId << " exists";

    EXPECT_FALSE(selectedKernelId(recorder).has_value()) << "Captured logs:\n"
                                                         << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionDenseKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionDenseKnobFilter,
                         ::testing::ValuesIn(unsatisfiableCases()),
                         [](const ::testing::TestParamInfo<UnsatisfiableCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
