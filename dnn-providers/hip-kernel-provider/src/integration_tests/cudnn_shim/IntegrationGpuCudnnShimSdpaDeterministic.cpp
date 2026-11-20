// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HKP_CUDNN_SHIM_SDPA_TESTS) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../sdpa_selection/SdpaProvenEngines.hpp"
#include "IntegrationCudnnShimHkpFixture.hpp"

/**
 * @file IntegrationGpuCudnnShimSdpaDeterministic.cpp
 * @brief The cuDNN v9 shim's `deselect_numeric_notes({NONDETERMINISTIC})` on SDPA graphs, through
 *        this binary's real plugin: only an engine that survives the shim's filter may be
 *        selected, and a graph no ranked engine survives is refused. No claim is listed here: the
 *        shim has no per-engine note query, so each ranked engine's verdict is read through its
 *        own public API, create_execution_plan(index) on a graph with the filter set, which
 *        fails for an engine the filter bars. A claim lives in the engine's descriptor.
 *
 * The shim keeps its own note lists and bars engines itself, so a graph no engine survives fails
 * with GRAPH_NOT_SUPPORTED at create_execution_plans (graph_wrapper.h,
 * findTopSurvivingPlanIndex), not with the native filter's GRAPH_EXECUTION_PLAN_CREATION_FAILED
 * (IntegrationGpuSdpaNativeNoteFilter.cpp).
 *
 * Cases are registered as
 * `Comprehensive/IntegrationGpuCudnnShimSdpaDeterministic.SelectsOnlyProvenEngines/<Case>_<arch>`
 * over the rows of selectionRows(); a row runs only on a device of its architecture.
 */
namespace
{

using namespace hip_kernel_provider::sdpa_selection;
using hipdnn_data_sdk::utilities::engineNameOrIdToId;

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

/// What the shim says when its filters leave no created plan (graph_wrapper.h).
constexpr const char* NO_SURVIVOR_MESSAGE = "survives the plan filters";

std::shared_ptr<fe::graph::Tensor_attributes> makeTensor(fe::graph::Graph& graph,
                                                         int64_t uid,
                                                         const std::string& name,
                                                         fe::DataType_t type,
                                                         const std::vector<int64_t>& dims,
                                                         const std::vector<int64_t>& strides)
{
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_uid(uid)
                            .set_name(name)
                            .set_dim(dims)
                            .set_stride(strides)
                            .set_data_type(type));
}

/// The graphs of IntegrationGpuSdpaNativeNoteFilter.cpp, through the v9 API: a bf16 forward graph
/// (batch 2, 8 heads, sequence 128, head size 64, causal) and the two bf16 backward graphs (head
/// size 128, sequence 256; BACKWARD_ASM_ONLY is the one only ASM_SDPA_ENGINE ranks).
void buildGraph(fe::graph::Graph& graph, GraphKind kind)
{
    const fe::DataType_t type = fe::DataType_t::BFLOAT16;
    graph.set_io_data_type(type)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT);

    if(kind == GraphKind::FORWARD)
    {
        const std::vector<int64_t> dims = {2, 8, 128, 64};
        const std::vector<int64_t> strides = {int64_t{8} * 128 * 64, int64_t{128} * 64, 64, 1};
        auto q = makeTensor(graph, 1, "Q", type, dims, strides);
        auto k = makeTensor(graph, 2, "K", type, dims, strides);
        auto v = makeTensor(graph, 3, "V", type, dims, strides);
        auto [o, stats] = graph.sdpa(q,
                                     k,
                                     v,
                                     fe::graph::SDPA_attributes{}
                                         .set_name("sdpa_selection_fwd")
                                         .set_attn_scale(1.0F / std::sqrt(64.0F))
                                         .set_generate_stats(true)
                                         .set_causal_mask(true));
        o->set_output(true).set_uid(4);
        stats->set_output(true)
            .set_uid(5)
            .set_name("Stats")
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({2, 8, 128, 1})
            .set_stride({int64_t{8} * 128, 128, 1, 1});
        return;
    }

    const bool asmOnly = kind == GraphKind::BACKWARD_ASM_ONLY;
    const int64_t batch = asmOnly ? 2 : 1;
    const int64_t heads = asmOnly ? 2 : 1;
    const int64_t seq = 256;
    const int64_t headSize = 128;
    const std::vector<int64_t> dims = {batch, heads, seq, headSize};
    const std::vector<int64_t> packed = {heads * seq * headSize, seq * headSize, headSize, 1};
    const std::vector<int64_t> bshd = {seq * heads * headSize, headSize, heads * headSize, 1};
    const std::vector<int64_t> padded
        = {heads * seq * (headSize + 1), seq * (headSize + 1), headSize + 1, 1};
    const auto& io = asmOnly ? bshd : packed;
    const auto& oStrides = asmOnly ? padded : packed;

    auto q = makeTensor(graph, 1, "Q", type, dims, io);
    auto k = makeTensor(graph, 2, "K", type, dims, io);
    auto v = makeTensor(graph, 3, "V", type, dims, io);
    auto o = makeTensor(graph, 4, "O", type, dims, oStrides);
    auto dO = makeTensor(graph, 5, "dO", type, dims, io);
    auto stats = makeTensor(
        graph, 6, "LSE", fe::DataType_t::FLOAT, {batch, heads, seq, 1}, {heads * seq, seq, 1, 1});
    auto [dq, dk, dv]
        = graph.sdpa_backward(q,
                              k,
                              v,
                              o,
                              dO,
                              stats,
                              fe::graph::SDPA_backward_attributes{}
                                  .set_name("sdpa_selection_bwd")
                                  .set_attn_scale(1.0F / std::sqrt(static_cast<float>(headSize))));
    dq->set_output(true).set_uid(7).set_data_type(type).set_dim(dims).set_stride(oStrides);
    dk->set_output(true).set_uid(8).set_data_type(type).set_dim(dims).set_stride(io);
    dv->set_output(true).set_uid(9).set_data_type(type).set_dim(dims).set_stride(io);
}

class CudnnShimSdpaSelectionCase
    : public hip_kernel_provider::test_utilities::CudnnShimHkpFixture<::testing::Test>
{
public:
    explicit CudnnShimSdpaSelectionCase(SelectionRow row)
        : _row(std::move(row))
    {
    }

    void TestBody() override
    {
        if(_arch != _row.arch)
        {
            GTEST_SKIP() << _row.name << " is listed for " << _row.arch << "; this device is "
                         << _arch;
        }

        std::vector<int64_t> ranked;
        std::vector<int64_t> survivors;
        std::vector<int64_t> nondeterministic;
        ASSERT_NO_FATAL_FAILURE(classifyRankedEngines(ranked, survivors, nondeterministic));
        RecordProperty("sdpa_ranked", engineNamesText(ranked));
        RecordProperty("sdpa_survivors", engineNamesText(survivors));
        std::string expectedUnclaimed;
        for(const auto& name : _row.unclaimedRanked)
        {
            expectedUnclaimed += (expectedUnclaimed.empty() ? "" : ",") + name;
            const int64_t id = engineNameOrIdToId(name);
            ASSERT_NE(std::find(ranked.begin(), ranked.end(), id), ranked.end())
                << name << " must rank the graph of " << _row.name
                << " on this device; ranked: " << engineNamesText(ranked);
            ASSERT_NE(std::find(nondeterministic.begin(), nondeterministic.end(), id),
                      nondeterministic.end())
                << name << " must report NONDETERMINISTIC for " << _row.name
                << "; survivors: " << engineNamesText(survivors);
        }
        RecordProperty("sdpa_expected_unclaimed",
                       expectedUnclaimed.empty() ? "(none)" : expectedUnclaimed);

        fe::graph::Graph graph;
        buildGraph(graph, _row.kind);
        graph.deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});

        auto err = graph.validate();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.build_operation_graph(_handle);
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.create_execution_plans();

        if(survivors.empty())
        {
            // Every ranked engine reports NONDETERMINISTIC. The first non-good status of the
            // staged build is create_execution_plans, where the shim applies its filters and
            // finds no surviving plan.
            ASSERT_TRUE(err.is_bad())
                << _row.name << " built a plan though every ranked engine is barred (ranked: "
                << engineNamesText(ranked) << ")";
            EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED) << err.get_message();
            EXPECT_NE(err.get_message().find(NO_SURVIVOR_MESSAGE), std::string::npos)
                << err.get_message();
            return;
        }

        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.check_support(_handle);
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.build_plans(_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
        ASSERT_TRUE(err.is_good()) << err.get_message();

        std::string planName;
        ASSERT_TRUE(graph.get_plan_name(planName).is_good());
        const int64_t selected = engineNameOrIdToId(planName);
        EXPECT_NE(std::find(survivors.begin(), survivors.end(), selected), survivors.end())
            << "plan " << planName << " (engine " << engineNamesText({selected})
            << ") was selected for " << _row.name
            << ", though the shim's filter bars it (survivors: " << engineNamesText(survivors)
            << ")";
    }

private:
    /// A fresh graph of the row, built to its operation graph, with the filter set when asked.
    void buildOperationGraph(fe::graph::Graph& graph, bool withFilter)
    {
        buildGraph(graph, _row.kind);
        if(withFilter)
        {
            graph.deselect_numeric_notes({fe::NumericalNote_t::NONDETERMINISTIC});
        }
        auto err = graph.validate();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        err = graph.build_operation_graph(_handle);
        ASSERT_TRUE(err.is_good()) << err.get_message();
    }

    /// The engines the backend ranks for the row's graph, in ranking order (the shim's engine
    /// index order), each named through a plan of an unfiltered graph; and, with the filter set,
    /// whether the shim lets that engine's plan through. A barred engine fails
    /// create_execution_plan(index) with GRAPH_NOT_SUPPORTED, which is how the shim reports that
    /// the engine reports NONDETERMINISTIC.
    void classifyRankedEngines(std::vector<int64_t>& ranked,
                               std::vector<int64_t>& survivors,
                               std::vector<int64_t>& nondeterministic)
    {
        fe::graph::Graph plain;
        ASSERT_NO_FATAL_FAILURE(buildOperationGraph(plain, false));
        int64_t count = 0;
        auto err = plain.get_engine_count(count);
        ASSERT_TRUE(err.is_good()) << err.get_message();
        ASSERT_GT(count, 0) << "no engine is ranked for " << _row.name;

        for(int64_t index = 0; index < count; ++index)
        {
            fe::graph::Graph unfiltered;
            ASSERT_NO_FATAL_FAILURE(buildOperationGraph(unfiltered, false));
            err = unfiltered.create_execution_plan(index, {});
            ASSERT_TRUE(err.is_good()) << "engine index " << index << ": " << err.get_message();
            std::string name;
            ASSERT_TRUE(unfiltered.get_plan_name_at_index(0, name).is_good());
            const int64_t id = engineNameOrIdToId(name);
            ranked.push_back(id);

            fe::graph::Graph filtered;
            ASSERT_NO_FATAL_FAILURE(buildOperationGraph(filtered, true));
            err = filtered.create_execution_plan(index, {});
            if(err.is_good())
            {
                survivors.push_back(id);
                continue;
            }
            EXPECT_EQ(err.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED)
                << engineNamesText({id}) << ": " << err.get_message();
            EXPECT_NE(err.get_message().find(NO_SURVIVOR_MESSAGE), std::string::npos)
                << engineNamesText({id}) << ": " << err.get_message();
            nondeterministic.push_back(id);
        }

        if(_row.kind == GraphKind::BACKWARD_ASM_ONLY)
        {
            const int64_t asmEngine = engineNameOrIdToId("ASM_SDPA_ENGINE");
            ASSERT_TRUE(ranked.size() == 1 && ranked.front() == asmEngine)
                << "the graph of " << _row.name
                << " must be served by ASM_SDPA_ENGINE alone; ranked: " << engineNamesText(ranked);
        }
    }

    SelectionRow _row;
};

bool registerShimRows()
{
    const std::string suite = "Comprehensive/IntegrationGpuCudnnShimSdpaDeterministic";
    for(const auto& row : selectionRows())
    {
        static_cast<void>(::testing::RegisterTest(
            suite.c_str(),
            ("SelectsOnlyProvenEngines/" + row.name).c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [row]() -> ::testing::Test* { return new CudnnShimSdpaSelectionCase(row); }));
    }
    return true;
}

[[maybe_unused]] const bool REGISTERED = registerShimRows();

} // namespace

#endif
