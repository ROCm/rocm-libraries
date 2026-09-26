// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
 * @file IntegrationGpuGfx950AttentionDenseKnobs.cpp
 * @brief The block_m/block_n knobs of hipkernel:Gfx950AttentionDense, end to end through
 *        the frontend on a gfx950 device: a forced tile selects exactly its kernel, that
 *        kernel launches with its own geometry and computes what the CPU reference
 *        computes, and a knob pair no kernel has is refused before any plan exists.
 *
 * Each expected kernel id is the kdp `id` of the catalog row in
 * descriptors/rocKE/gfx950_attention_dense/gfx950_attention_dense.kdp.json whose dtype,
 * head_size, num_query_heads, num_kv_heads, causal, block_m and block_n match the case.
 * Every forced tile except the 256/64 baseline is paired with a shape whose no-knob
 * winner is a different tile, so a knob that failed to reach the plugin would select that
 * winner and fail the id check. The baseline always wins where it fits, so its two cases
 * select the no-knob winner by design.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx950AttentionDense";

/// The one architecture the engine's packs claim.
constexpr const char* SERVED_ARCH = "gfx950";

constexpr const char* BLOCK_M_KNOB = "block_m";
constexpr const char* BLOCK_N_KNOB = "block_n";

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
    CAUSAL_MASK_BOTTOM_RIGHT_FLAG
};

struct GraphShape
{
    DataType dataType;
    int64_t headSize;
    int64_t queryHeads;
    int64_t kvHeads;
    Mask mask;
    /// NOT_SET leaves the attribute at its default.
    DataType mmaCoreMode;
    int64_t batch;
    int64_t seqQ;
    int64_t seqKv;
};

struct Tile
{
    int64_t blockM;
    int64_t blockN;
};

/// One graph and the kernel the engine must serve it with. No forced tile is a cold
/// case: no knob is set and the engine's own ranking decides.
struct KnobCase
{
    const char* name;
    GraphShape shape;
    std::optional<Tile> forcedTile;
    const char* expectedKernelId;
};

/// One graph and a knob pair whose values the engine advertises for it one by one while
/// no kernel it admits carries both.
struct UnsatisfiableCase
{
    const char* name;
    GraphShape shape;
    Tile forcedTile;
};

void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

void PrintTo(const UnsatisfiableCase& unsatisfiableCase, std::ostream* os)
{
    *os << unsatisfiableCase.name;
}

constexpr GraphShape makeShape(DataType dataType,
                               int64_t headSize,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               Mask mask,
                               DataType mmaCoreMode,
                               int64_t batch,
                               int64_t seqQ,
                               int64_t seqKv)
{
    return {dataType, headSize, queryHeads, kvHeads, mask, mmaCoreMode, batch, seqQ, seqKv};
}

/// Thirteen forced (tile, head size) pairs, then four cold cases whose id is the kdp row
/// of the tile the engine's own ranking picks for the shape. Within them: bounds on both
/// corners and both deprecated flags, fp16 and bf16, MHA, GQA and MQA, and four forced
/// cases with B > 1 and Sq != Skv.
std::vector<KnobCase> knobCases()
{
    constexpr auto FP16 = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr auto MMA_UNSET = DataType::NOT_SET;
    return {
        // D64: all seven legal tiles.
        {"D64_Bm128Bn32",
         makeShape(BF16, 64, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         Tile{128, 32},
         "f5de4107-b6bb-4ed2-b5f7-97c01ded9ca1"},
        {"D64_Bm128Bn64",
         makeShape(FP16, 64, 16, 2, Mask::BOUNDS_TOP_LEFT, MMA_UNSET, 2, 128, 256),
         Tile{128, 64},
         "9f1ed64a-1607-4bf2-9a20-bb317ec28d38"},
        {"D64_Bm128Bn128",
         makeShape(BF16, 64, 8, 1, Mask::NO_MASK, MMA_UNSET, 2, 384, 128),
         Tile{128, 128},
         "85e207d0-b2c3-4ad4-9c08-69bcf5b11e2b"},
        {"D64_Bm256Bn32",
         makeShape(FP16, 64, 8, 8, Mask::CAUSAL_MASK_FLAG, MMA_UNSET, 1, 256, 256),
         Tile{256, 32},
         "f78bab4e-82ba-4417-9a08-9e8eda571426"},
        {"D64_Bm256Bn128",
         makeShape(BF16, 64, 16, 16, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         Tile{256, 128},
         "ff210741-69b8-49f5-8282-0bb062c3e487"},
        {"D64_Bm256Bn256",
         makeShape(FP16, 64, 12, 12, Mask::NO_MASK, MMA_UNSET, 1, 512, 256),
         Tile{256, 256},
         "2d1e42af-3559-417a-812f-8d318ca65872"},
        // The baseline tile: forced and cold select the same kernel.
        {"D64_Bm256Bn64",
         makeShape(BF16, 64, 10, 10, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         Tile{256, 64},
         "b7697d0f-6a63-4ebe-91ab-839dac35128a"},
        // D128: all six legal tiles.
        {"D128_Bm128Bn32",
         makeShape(FP16, 128, 8, 2, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         Tile{128, 32},
         "3c1927b3-ce3e-44ce-a621-b5fd777c4752"},
        {"D128_Bm128Bn64",
         makeShape(BF16, 128, 8, 1, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 1, 128, 128),
         Tile{128, 64},
         "19b74331-a5c4-4981-a8c0-5a9f597fb21e"},
        {"D128_Bm128Bn128",
         makeShape(FP16, 128, 4, 4, Mask::NO_MASK, MMA_UNSET, 3, 128, 384),
         Tile{128, 128},
         "6263a9d3-aa8b-42e1-8e61-5c9ecc5b719a"},
        {"D128_Bm256Bn32",
         makeShape(BF16, 128, 16, 2, Mask::NO_MASK, MMA_UNSET, 2, 256, 480),
         Tile{256, 32},
         "545e6b54-fe0b-4c17-a04c-390b0138d95b"},
        {"D128_Bm256Bn128",
         makeShape(BF16, 128, 9, 9, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 1, 256, 256),
         Tile{256, 128},
         "fdbf347a-8cbf-41a6-bcaa-c69e7038c17c"},
        // The baseline tile: forced and cold select the same kernel.
        {"D128_Bm256Bn64",
         makeShape(FP16, 128, 8, 8, Mask::NO_MASK, MMA_UNSET, 1, 256, 256),
         Tile{256, 64},
         "d1f63599-da2a-4c43-a655-8619b6b534ff"},
        // Cold: no knob set.
        {"Cold_MmaHalf",
         makeShape(BF16, 64, 16, 2, Mask::BOUNDS_TOP_LEFT, FP16, 2, 256, 512),
         std::nullopt,
         "1a1c41ba-a165-480b-a356-55558f9151fc"},
        {"Cold_MmaBfloat16",
         makeShape(BF16, 128, 8, 1, Mask::NO_MASK, BF16, 3, 384, 256),
         std::nullopt,
         "a3fb6c5a-f503-4503-b0f8-a7083b9bd4f2"},
        {"Cold_CausalBottomRightFlag",
         makeShape(FP16, 64, 8, 8, Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG, MMA_UNSET, 2, 512, 512),
         std::nullopt,
         "ae0a3dab-bb41-4744-8075-472b4df37ea5"},
        {"Cold_D128Heads9",
         makeShape(BF16, 128, 9, 9, Mask::BOUNDS_BOTTOM_RIGHT, MMA_UNSET, 2, 256, 256),
         std::nullopt,
         "7215c5c7-14a9-4f93-8e45-c66cfb345cb5"},
    };
}

/// A D64 fp16 MHA no-mask graph at Sq = Skv = 256. It admits the D64 tiles 128/32, 128/64,
/// 128/128, 256/32, 256/64, 256/128 and 256/256, so block_m=128 and block_n=256 are each
/// advertised while no D64 kernel is 128/256.
std::vector<UnsatisfiableCase> unsatisfiableCases()
{
    return {
        {"D64_Bm128Bn256",
         makeShape(DataType::HALF, 64, 8, 8, Mask::NO_MASK, DataType::NOT_SET, 1, 256, 256),
         Tile{128, 256}},
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

/// A single SDPA-forward node with BSHD Q/K/V/O and an explicit softmax scale, the
/// shape the engine's graph_match accepts. O is laid out explicitly: the frontend's
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
    attributes.set_attn_scale(1.0f / std::sqrt(static_cast<float>(shape.headSize)));
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
        attributes.set_causal_mask(true);
        break;
    case Mask::CAUSAL_MASK_BOTTOM_RIGHT_FLAG:
        attributes.set_causal_mask_bottom_right(true);
        break;
    default:
        throw std::invalid_argument("buildSdpaGraph: unhandled Mask value");
    }
    if(shape.mmaCoreMode != DataType::NOT_SET)
    {
        attributes.set_mma_core_mode(shape.mmaCoreMode);
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
/// (config/hipkernel_Gfx950AttentionDense.toml names no override).
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

/// Prints `KNOB_SELECTED <case> <kernelId>` when the case ends, however it ends, so a
/// run's log carries the selected kernel of every case, including a failing one.
class SelectedKernelReport
{
public:
    SelectedKernelReport(const char* caseName, const ScopedPluginLogCapture& capture)
        : _caseName(caseName)
        , _capture(capture)
    {
    }

    ~SelectedKernelReport()
    {
        std::cout << "KNOB_SELECTED " << _caseName << " "
                  << selectedKernelId(_capture.recorder()).value_or("none") << std::endl;
    }

    SelectedKernelReport(const SelectedKernelReport&) = delete;
    SelectedKernelReport& operator=(const SelectedKernelReport&) = delete;
    SelectedKernelReport(SelectedKernelReport&&) = delete;
    SelectedKernelReport& operator=(SelectedKernelReport&&) = delete;

private:
    const char* _caseName;
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

/// Runs only on the architecture the engine ships for. There, the engine, its packs and
/// its kernels are all expected present, and any of them missing fails the case.
template <typename TestCaseType>
class IntegrationGpuGfx950AttentionDenseBase
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

    /// Offsets the seed by UID so Q, K and V carry different data: K and V share a shape
    /// here, and an operand swap between them is invisible when their contents agree.
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

    /// Builds the operation graph and requires the engine among those offering to serve
    /// it. On this architecture its absence is a missing engine or pack, not a skip.
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

    /// A cache root private to ONE test case. A winner record another case measured would
    /// replace the heuristic order, and the selected kernel with it.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "gfx950-dense-knobs-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

using IntegrationGpuGfx950AttentionDenseKnobs = IntegrationGpuGfx950AttentionDenseBase<KnobCase>;
using IntegrationGpuGfx950AttentionDenseKnobFilter
    = IntegrationGpuGfx950AttentionDenseBase<UnsatisfiableCase>;

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the
/// output is then checked against the CPU reference, which catches a kernel launched
/// with any geometry but its own.
TEST_P(IntegrationGpuGfx950AttentionDenseKnobs, SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const SelectedKernelReport report(testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    const auto settings = testCase.forcedTile.has_value() ? knobSettingsFor(*testCase.forcedTile)
                                                          : std::vector<KnobSetting>{};
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
    EXPECT_EQ(*selected, testCase.expectedKernelId) << "Captured logs:\n"
                                                    << recorder.getRecordedLogsAsString();

    // Rank 0: a cold case serves the heuristic's front rather than a fallback past a
    // kernel that failed to load, and a forced case filters to its one kernel.
    auto selectionLine
        = std::string(SELECTED_KERNEL_MARKER) + testCase.expectedKernelId + " at rank 0";
    if(testCase.forcedTile.has_value())
    {
        selectionLine += " from 1 candidate(s)";
    }
    EXPECT_TRUE(recorder.hasLogContaining(selectionLine))
        << "expected '" << selectionLine << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// The frontend validates each knob value on its own against the values the engine
/// advertises for this graph, never the pair. A pair of advertised values that no kernel
/// carries therefore reaches the plugin's knob filter, whose refusal is the only thing
/// that stands between it and a plan built from nothing.
TEST_P(IntegrationGpuGfx950AttentionDenseKnobFilter, RefusesAKnobPairNoKernelCarriesAndBuildsNoPlan)
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

    // No plan: nothing was compiled for build_plans() to finalize, no engine backs a plan,
    // and the plugin never selected a kernel.
    result = sdpa.graph->build_plans();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    int64_t planEngineId = 0;
    EXPECT_NE(sdpa.graph->get_execution_plan_engine_id(planEngineId).code, ErrorCode::OK)
        << "a plan backed by engine " << planEngineId << " exists";

    EXPECT_FALSE(selectedKernelId(recorder).has_value()) << "Captured logs:\n"
                                                         << recorder.getRecordedLogsAsString();
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionDenseKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return std::string(info.param.name);
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx950AttentionDenseKnobFilter,
                         ::testing::ValuesIn(unsatisfiableCases()),
                         [](const ::testing::TestParamInfo<UnsatisfiableCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)
