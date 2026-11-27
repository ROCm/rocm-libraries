// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../kernel_ingestor_engine/FlydslSdpaFrontendGraphs.hpp"
#include "SdpaProvenEngines.hpp"

/**
 * @file IntegrationGpuSdpaNativeNoteFilter.cpp
 * @brief The native frontend's numerical-note filter on SDPA graphs, through this binary's real
 *        plugin: with `deselect_numeric_notes({NONDETERMINISTIC})` set, only an engine whose
 *        notes (Graph::get_numerical_notes_for_engine, read at run time) lack NONDETERMINISTIC
 *        may be selected, and a graph every ranked engine reports NONDETERMINISTIC for is
 *        refused with GRAPH_EXECUTION_PLAN_CREATION_FAILED. No claim is listed here: a claim
 *        lives in the engine's descriptor.
 *
 * Cases are registered as
 * `Comprehensive/IntegrationGpuSdpaNativeNoteFilter.<Test>/<Case>_<arch>`; a row runs only on a
 * device of its architecture and skips elsewhere. The cuDNN v9 shim has its own filter and its own
 * suite (IntegrationGpuCudnnShimSdpaDeterministic.cpp).
 */
namespace
{

using namespace hip_kernel_provider::sdpa_selection;
namespace ingestor = hip_kernel_provider::kernel_ingestor_engine::integration;
using hipdnn_data_sdk::utilities::engineNameToId;
using hipdnn_frontend::DataType;
using hipdnn_frontend::ErrorCode;

/// The reason a note-filtered plan carries in the messages of check_support and build_plans
/// (Graph.hpp, evaluatePlanFilters).
constexpr const char* NOTE_FILTER_REASON = "numerical note filter";

enum class Check
{
    HEURISTICS_SELECT_ONLY_PROVEN,
    PINNED_UNPROVEN_IS_REFUSED,
    BACKWARD_WITH_ONLY_ASM_IS_REFUSED,
};

std::shared_ptr<hipdnn_frontend::graph::Graph> buildForwardGraph()
{
    ingestor::FrontendSdpaSpec spec;
    spec.name = "sdpa_selection_fwd";
    spec.dataType = DataType::BFLOAT16;
    spec.batch = 2;
    spec.queryHeads = 8;
    spec.kvHeads = 8;
    spec.seqQ = 128;
    spec.seqKv = 128;
    spec.headSize = 64;
    spec.leftBound.reset();
    spec.rightBound = 0;
    spec.withStats = true;
    return ingestor::buildFrontendSdpaGraph(spec).graph;
}

/// An SDPA backward graph, bf16, head size 128, sequence 256, no mask. Packed [B, H, S, D] for
/// BACKWARD. For BACKWARD_ASM_ONLY (B = H = 2): Q, K, V, dO, dK and dV stored [B, S, H, D] and
/// O and dQ with rows padded to D + 1, the layout the ASM backward integration suite runs and
/// the FlyDSL backward matcher declines (LY-3: strides must be multiples of 8 elements).
std::shared_ptr<hipdnn_frontend::graph::Graph> buildBackwardGraph(GraphKind kind)
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    const bool asmOnly = kind == GraphKind::BACKWARD_ASM_ONLY;
    const int64_t batch = asmOnly ? 2 : 1;
    const int64_t heads = asmOnly ? 2 : 1;
    const int64_t seq = 256;
    const int64_t headSize = 128;
    const DataType dataType = DataType::BFLOAT16;

    auto graph = std::make_shared<Graph>();
    graph->set_name(asmOnly ? "sdpa_selection_bwd_asm_only" : "sdpa_selection_bwd")
        .set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {batch, heads, seq, headSize};
    const std::vector<int64_t> packed = {heads * seq * headSize, seq * headSize, headSize, 1};
    const std::vector<int64_t> bshd = {seq * heads * headSize, headSize, heads * headSize, 1};
    const std::vector<int64_t> padded
        = {heads * seq * (headSize + 1), seq * (headSize + 1), headSize + 1, 1};
    const auto& io = asmOnly ? bshd : packed;
    const auto& oStrides = asmOnly ? padded : packed;
    const std::vector<int64_t> statsDims = {batch, heads, seq, 1};
    const std::vector<int64_t> statsStrides = {heads * seq, seq, 1, 1};

    const auto makeTensor = [&](int64_t uid,
                                const std::string& name,
                                DataType type,
                                const std::vector<int64_t>& tensorDims,
                                const std::vector<int64_t>& strides) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid)
            .set_name(name)
            .set_dim(tensorDims)
            .set_stride(strides)
            .set_data_type(type);
        return tensor;
    };
    auto q = makeTensor(1, "Q", dataType, dims, io);
    auto k = makeTensor(2, "K", dataType, dims, io);
    auto v = makeTensor(3, "V", dataType, dims, io);
    auto o = makeTensor(4, "O", dataType, dims, oStrides);
    auto dO = makeTensor(5, "dO", dataType, dims, io);
    auto stats = makeTensor(6, "LSE", DataType::FLOAT, statsDims, statsStrides);

    SdpaBackwardAttributes attributes;
    attributes.set_name(graph->get_name());
    attributes.set_attn_scale(1.0F / std::sqrt(static_cast<float>(headSize)));

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(attributes));
    dq->set_uid(7).set_name("dQ").set_output(true).set_data_type(dataType).set_dim(dims).set_stride(
        oStrides);
    dk->set_uid(8).set_name("dK").set_output(true).set_data_type(dataType).set_dim(dims).set_stride(
        io);
    dv->set_uid(9).set_name("dV").set_output(true).set_data_type(dataType).set_dim(dims).set_stride(
        io);
    return graph;
}

class NativeNoteFilterCase : public ::testing::Test
{
public:
    NativeNoteFilterCase(Check check, SelectionRow row)
        : _check(check)
        , _row(std::move(row))
    {
    }

    ~NativeNoteFilterCase() override
    {
        if(_handle != nullptr)
        {
            static_cast<void>(hipdnnDestroy(_handle));
        }
        if(_stream != nullptr)
        {
            static_cast<void>(hipStreamDestroy(_stream));
        }
    }

    NativeNoteFilterCase(const NativeNoteFilterCase&) = delete;
    NativeNoteFilterCase& operator=(const NativeNoteFilterCase&) = delete;

    void TestBody() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_NO_FATAL_FAILURE(openDevice());
        if(_arch != _row.arch)
        {
            GTEST_SKIP() << _row.name << " is listed for " << _row.arch << "; this device is "
                         << _arch;
        }

        auto graph
            = _row.kind == GraphKind::FORWARD ? buildForwardGraph() : buildBackwardGraph(_row.kind);
        graph->deselect_numeric_notes({hipdnn_frontend::NumericalNote::NONDETERMINISTIC});
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The ranking is the unfiltered one: the filters act on plans, not on the ranked list.
        std::vector<int64_t> ranked;
        result = graph->get_ranked_engine_ids(ranked);
        ASSERT_EQ(result.code, ErrorCode::OK)
            << "no engine is ranked for " << _row.name << " (" << result.err_msg << ")";
        ASSERT_FALSE(ranked.empty()) << "no engine is ranked for " << _row.name;

        // Each ranked engine's notes, read as the filter reads them: the survivors lack
        // NONDETERMINISTIC.
        for(const auto id : ranked)
        {
            std::vector<hipdnn_frontend::NumericalNote> notes;
            result = graph->get_numerical_notes_for_engine(id, notes);
            ASSERT_EQ(result.code, ErrorCode::OK)
                << "notes of engine " << engineNamesText({id}) << ": " << result.err_msg;
            const auto nondeterministicNote = hipdnn_frontend::NumericalNote::NONDETERMINISTIC;
            const bool nondeterministic
                = std::find(notes.begin(), notes.end(), nondeterministicNote) != notes.end();
            (nondeterministic ? _nondeterministic : _survivors).push_back(id);
        }
        RecordProperty("sdpa_ranked", engineNamesText(ranked));
        RecordProperty("sdpa_survivors", engineNamesText(_survivors));

        std::string expectedUnclaimed;
        for(const auto& name : _row.unclaimedRanked)
        {
            expectedUnclaimed += (expectedUnclaimed.empty() ? "" : ",") + name;
            const int64_t id = engineNameToId(name);
            ASSERT_NE(std::find(ranked.begin(), ranked.end(), id), ranked.end())
                << name << " must rank the graph of " << _row.name
                << " on this device; ranked: " << engineNamesText(ranked);
            ASSERT_NE(std::find(_nondeterministic.begin(), _nondeterministic.end(), id),
                      _nondeterministic.end())
                << name << " must report NONDETERMINISTIC for " << _row.name
                << "; survivors: " << engineNamesText(_survivors);
        }
        RecordProperty("sdpa_expected_unclaimed",
                       expectedUnclaimed.empty() ? "(none)" : expectedUnclaimed);

        switch(_check)
        {
        case Check::HEURISTICS_SELECT_ONLY_PROVEN:
            heuristicsSelectOnlySurvivors(*graph, ranked);
            break;
        case Check::PINNED_UNPROVEN_IS_REFUSED:
            pinnedNondeterministicIsRefused(*graph);
            break;
        case Check::BACKWARD_WITH_ONLY_ASM_IS_REFUSED:
            backwardWithOnlyAsmIsRefused(*graph, ranked);
            break;
        default:
            throw std::invalid_argument("NativeNoteFilterCase: unhandled Check value");
        }
    }

private:
    void openDevice()
    {
        ASSERT_EQ(hipInit(0), hipSuccess);
        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathText = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathText.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
        _arch = hip_kernel_provider_common::getDeviceString(_stream);
    }

    /// Both calls that decide a plan refuse with GRAPH_EXECUTION_PLAN_CREATION_FAILED, and the
    /// message names the note filter as the reason.
    static void expectRefused(hipdnn_frontend::graph::Graph& graph, const std::string& what)
    {
        auto result = graph.check_support();
        EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED)
            << what << ": check_support: " << result.err_msg;
        EXPECT_NE(result.err_msg.find(NOTE_FILTER_REASON), std::string::npos)
            << what << ": check_support message: " << result.err_msg;
        result = graph.build_plans(hipdnn_frontend::BuildPlanPolicy::HEURISTICS_CHOICE);
        EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED)
            << what << ": build_plans: " << result.err_msg;
        EXPECT_NE(result.err_msg.find(NOTE_FILTER_REASON), std::string::npos)
            << what << ": build_plans message: " << result.err_msg;
    }

    /// The compiled plan index of @p engine, or -1.
    static int64_t planIndexOf(const hipdnn_frontend::graph::Graph& graph, int64_t engine)
    {
        const int64_t count = graph.get_execution_plan_count();
        for(int64_t index = 0; index < count; ++index)
        {
            std::string name;
            if(graph.get_plan_name_at_index(index, name).code == ErrorCode::OK
               && hipdnn_data_sdk::utilities::engineNameOrIdToId(name) == engine)
            {
                return index;
            }
        }
        return -1;
    }

    void heuristicsSelectOnlySurvivors(hipdnn_frontend::graph::Graph& graph,
                                       const std::vector<int64_t>& ranked)
    {
        auto result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        if(_survivors.empty())
        {
            ASSERT_FALSE(_row.expectsSurvivor)
                << _row.name << " expects a claimed engine to survive, but every ranked engine ("
                << engineNamesText(ranked) << ") reports NONDETERMINISTIC";
            // Every ranked engine reports NONDETERMINISTIC: nothing may be selected.
            expectRefused(graph, _row.name + " (ranked: " + engineNamesText(ranked) + ")");
            return;
        }

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans(hipdnn_frontend::BuildPlanPolicy::HEURISTICS_CHOICE);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t selected = 0;
        result = graph.get_execution_plan_engine_id(selected);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_NE(std::find(_survivors.begin(), _survivors.end(), selected), _survivors.end())
            << "engine " << engineNamesText({selected}) << " was selected for " << _row.name
            << ", but it reports NONDETERMINISTIC (survivors: " << engineNamesText(_survivors)
            << ")";

        // Something was filtered out: the plan of an engine that reports NONDETERMINISTIC is
        // refused by the note filter.
        if(!_nondeterministic.empty())
        {
            const int64_t index = planIndexOf(graph, _nondeterministic.front());
            ASSERT_GE(index, 0) << "no plan of " << engineNamesText({_nondeterministic.front()});
            result = graph.build_plan_at_index(index);
            EXPECT_EQ(result.code, ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED)
                << result.err_msg;
            EXPECT_NE(result.err_msg.find(NOTE_FILTER_REASON), std::string::npos) << result.err_msg;
        }
    }

    /// Pins the unclaimed engine the row names: the frontend's filter refuses a pinned plan of
    /// an engine that reports NONDETERMINISTIC (registered only on rows that name one).
    void pinnedNondeterministicIsRefused(hipdnn_frontend::graph::Graph& graph)
    {
        ASSERT_FALSE(_row.unclaimedRanked.empty()) << _row.name << " names no engine to pin";
        const int64_t pinned = engineNameToId(_row.unclaimedRanked.front());
        const auto result = graph.create_execution_plan_ext(pinned, {});
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        expectRefused(graph, _row.name + " pinned to " + engineNamesText({pinned}));
    }

    void backwardWithOnlyAsmIsRefused(hipdnn_frontend::graph::Graph& graph,
                                      const std::vector<int64_t>& ranked) const
    {
        const int64_t asmEngine = engineNameToId("ASM_SDPA_ENGINE");
        ASSERT_TRUE(ranked.size() == 1 && ranked.front() == asmEngine)
            << "the graph of " << _row.name
            << " must be served by ASM_SDPA_ENGINE alone; ranked: " << engineNamesText(ranked);

        // ASM_SDPA_ENGINE reports NONDETERMINISTIC (asserted in TestBody), so the only ranked
        // engine is filtered.
        const auto result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        expectRefused(graph, _row.name);
    }

    Check _check;
    SelectionRow _row;
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    std::string _arch;
    /// The ranked engines that report NONDETERMINISTIC, and those that do not, in ranked order.
    std::vector<int64_t> _nondeterministic;
    std::vector<int64_t> _survivors;
};

bool registerNativeRows()
{
    const std::string suite = "Comprehensive/IntegrationGpuSdpaNativeNoteFilter";
    for(const auto& row : selectionRows())
    {
        std::vector<std::pair<Check, const char*>> checks;
        if(row.kind == GraphKind::BACKWARD_ASM_ONLY)
        {
            checks.emplace_back(Check::BACKWARD_WITH_ONLY_ASM_IS_REFUSED,
                                "BackwardWithOnlyAsmApplicableIsRefused");
        }
        else
        {
            checks.emplace_back(Check::HEURISTICS_SELECT_ONLY_PROVEN,
                                "HeuristicsSelectOnlyProvenEngines");
            if(!row.unclaimedRanked.empty())
            {
                checks.emplace_back(Check::PINNED_UNPROVEN_IS_REFUSED,
                                    "PinnedUnprovenEngineIsRefused");
            }
        }
        for(const auto& [check, testName] : checks)
        {
            static_cast<void>(
                ::testing::RegisterTest(suite.c_str(),
                                        (std::string(testName) + "/" + row.name).c_str(),
                                        nullptr,
                                        nullptr,
                                        __FILE__,
                                        __LINE__,
                                        [checkKind = check, row]() -> ::testing::Test* {
                                            return new NativeNoteFilterCase(checkKind, row);
                                        }));
        }
    }
    return true;
}

[[maybe_unused]] const bool REGISTERED = registerNativeRows();

} // namespace

#endif
