// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuKernelIngestor.cpp
 * @brief The kernel ingestor pack, end to end through the hipDNN frontend API.
 *
 * Every sibling engine test drives `hipdnn_frontend::graph::Graph` through the harness;
 * this one used to hand-roll the plugin C ABI instead (dlopen, flatbuffer graph
 * builders, raw device buffers). That duplicated machinery the frontend already owns
 * and tested a door hipDNN itself never walks through in production. This file drives
 * the same door every other engine's E2E test does -- Graph::pointwise(),
 * get_ranked_engine_ids(), get_knobs_for_engine(), get_workspace_size(), execute() --
 * so it proves the descriptor set, the matchers, the heuristic, and the dispatch
 * handler compose the way a real caller reaches them.
 *
 * One exception: `hipdnnEnginePluginGetAllEngineIds` enumerates every engine a plugin
 * exports, independent of any graph. The frontend has no equivalent -- it only ever
 * ranks engines applicable to a graph that was actually built -- so that one assertion
 * still crosses the raw ABI via a minimal dlopen helper.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:PointwiseAdd";
constexpr const char* BLOCK_SIZE_KNOB = "block_size";

/// Workspace the pack's larger-block kernel declares, which the engine reports as the
/// maximum across its surviving kernels.
constexpr int64_t EXPECTED_WORKSPACE_BYTES = 1024;

std::shared_ptr<TensorAttributes> makeScalarTensor(int64_t uid, const std::string& name)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(DataType::FLOAT);
    return tensor;
}

/// A single pointwise-ADD node over 1-element FLOAT tensors: the one shape this pack
/// accepts. Each call returns a fresh graph so tests never share build/plan state.
std::shared_ptr<Graph> buildPointwiseAddGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("pointwise_add")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = makeScalarTensor(1, "A");
    auto b = makeScalarTensor(2, "B");

    PointwiseAttributes attrs;
    attrs.set_name("pointwise_add").set_mode(PointwiseMode::ADD);
    auto c = graph->pointwise(a, b, attrs);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

/// A graph this pack must decline: two nodes, so no single prebuilt kernel serves it.
std::shared_ptr<Graph> buildUnsupportedGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("two_node_pointwise")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = makeScalarTensor(1, "A");
    auto b = makeScalarTensor(2, "B");

    PointwiseAttributes attrs1;
    attrs1.set_name("add_1").set_mode(PointwiseMode::ADD);
    auto intermediate = graph->pointwise(a, b, attrs1);
    intermediate->set_uid(4).set_name("Intermediate").set_data_type(DataType::FLOAT);

    PointwiseAttributes attrs2;
    attrs2.set_name("add_2").set_mode(PointwiseMode::ADD);
    auto c = graph->pointwise(intermediate, b, attrs2);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

/// One case per execute() shape: a single call, and several reusing the same built
/// plan. Both must produce numerically correct results; only the second proves reuse.
struct ExecuteCase
{
    std::string name;
    int iterations;
};

} // namespace

class IntegrationGpuKernelIngestor
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      ExecuteCase>
{
protected:
    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    /// Builds `graph`, pins this pack via set_preferred_engine_id_ext() before plan
    /// creation (the precedent at IntegrationGraphEngineFiltering.cpp:159), and
    /// compiles a plan with default knobs. Returns once the plan is ready to execute.
    void buildAndCompile(Graph& graph)
    {
        graph.set_preferred_engine_id_ext(engineId());

        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    /// Builds fresh CPU/GPU tensor bundles for `graph`, executes it once on GPU with
    /// `workspace`, and verifies the result against CpuReferenceGraphExecutor. `seed`
    /// varies the input values so repeated calls never compare against stale buffers.
    void executeAndVerify(Graph& graph, void* workspace, unsigned int seed)
    {
        GraphTensorBundle gpuBundle;
        GraphTensorBundle cpuBundle;
        graph.visit([&](const INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                gpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                cpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(gpuBundle.tensors.find(tensorAttr->get_uid()) == gpuBundle.tensors.end())
                {
                    gpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                    cpuBundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
                }
            }
        });
        for(auto& [uid, tensor] : gpuBundle.tensors)
        {
            // Per-uid, not one seed for every tensor: randomizeTensor builds a fresh
            // mt19937 from the seed it is handed, so a shared seed makes every operand
            // byte-identical and allClose() below is then satisfied by a+a or b+b just
            // as well as by a+b -- it would no longer witness that the handler resolved
            // its arguments by tensor uid, which is the property this harness exists to
            // check. Offsetting by the uid is what PointwiseTensorBundles.hpp does
            // (seed, seed + 1) for the same reason.
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            gpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
            cpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
        }

        auto deviceVariantPack = gpuBundle.toDeviceVariantPack();
        auto result = graph.execute(_handle, deviceVariantPack, workspace);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), cpuBundle.toHostVariantPack());

        auto& gpuOut = gpuBundle.getTensor(3);
        auto& cpuOut = cpuBundle.getTensor(3);
        gpuOut.markDeviceModified();
        // The whole chain, end to end: the matchers admitted this graph, the heuristic
        // ranked the catalog, the dispatch descriptor's handler compiled and launched
        // the winner, and its arguments were resolved by tensor uid.
        EXPECT_TRUE(CpuFpReferenceValidation<float>().allClose(cpuOut, gpuOut));
    }
};

// ---------------------------------------------------------------------------
// Direct ABI: load-time self-registration
// ---------------------------------------------------------------------------

// The only assertion with no frontend equivalent: the frontend never lists "every
// engine this plugin exports", only engines ranked for a graph that was built. Proving
// the pack self-registers at load time, independent of any graph, still needs the raw
// C ABI -- kept minimal rather than reusing the deleted PluginApi wrapper.
TEST(IntegrationGpuKernelIngestorDirectAbi, SelfRegistersAllEngineIds)
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
        / pluginFile);
    auto* library = hipdnn_data_sdk::utilities::openLibrary(pluginPath);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* getAllEngineIds = reinterpret_cast<decltype(&hipdnnEnginePluginGetAllEngineIds)>(
        hipdnn_data_sdk::utilities::getSymbol(library, "hipdnnEnginePluginGetAllEngineIds"));
    ASSERT_NE(getAllEngineIds, nullptr);

    uint32_t count = 0;
    ASSERT_EQ(getAllEngineIds(nullptr, 0, &count), HIPDNN_PLUGIN_STATUS_SUCCESS);
    std::vector<int64_t> engines(count);
    ASSERT_EQ(getAllEngineIds(engines.data(), count, &count), HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_NE(std::find(engines.begin(), engines.end(), engineNameToId(ENGINE_NAME)),
              engines.end());
}

// ---------------------------------------------------------------------------
// Applicability
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, AcceptsTheGraphItsDescriptorsDescribe)
{
    auto graph = buildPointwiseAddGraph();

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), engineId()),
              rankedEngineIds.end());
}

TEST_F(IntegrationGpuKernelIngestor, DeclinesATwoNodeGraph)
{
    // Declining is free: no engine config comes back and hipDNN moves on. Getting this
    // wrong is what turns a cheap decline into a failed plan build.
    auto graph = buildUnsupportedGraph();

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
    EXPECT_TRUE(rankedEngineIds.empty());
}

// ---------------------------------------------------------------------------
// Engine details: knobs from the catalog
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ReportsAKnobWhoseValuesComeFromTheCatalog)
{
    auto graph = buildPointwiseAddGraph();

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<Knob> knobs;
    result = graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(knobs.size(), 1U);
    EXPECT_EQ(knobs[0].knobId(), BLOCK_SIZE_KNOB);

    // The pack ships three kernels, but the HALF one is pruned for this FLOAT graph, so
    // the knob offers exactly the two block sizes the surviving kernels implement --
    // never the schema's theoretical range.
    const auto* constraint = dynamic_cast<const IntConstraint*>(knobs[0].constraint());
    ASSERT_NE(constraint, nullptr);
    const auto& validValues = constraint->getValidValues();
    EXPECT_EQ(validValues, (std::unordered_set<int64_t>{64, 256}));

    // The default is whatever the heuristic ranked first, so leaving the knob alone
    // reproduces the out-of-the-box selection.
    const auto* defaultValue = std::get_if<int64_t>(&knobs[0].defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 256);
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ReportsTheMaximumWorkspaceAcrossSurvivingKernels)
{
    auto graph = buildPointwiseAddGraph();
    buildAndCompile(*graph);

    // One surviving kernel declares 0 bytes and the other 1024, so this answer proves
    // the query aggregates across the catalog rather than reporting one kernel's value.
    int64_t workspaceSize = 0;
    auto result = graph->get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(workspaceSize, EXPECTED_WORKSPACE_BYTES);
}

// ---------------------------------------------------------------------------
// Plan build and execute
// ---------------------------------------------------------------------------

// Collapses ExecutesTheSelectedKernelOnDevice and ReusesOnePlanAcrossExecutions from
// the original file: one case executes once, the other executes the same compiled
// plan repeatedly with fresh buffers each time. Every decision the plan makes happens
// at build, so execution must never depend on the previous call.
TEST_P(IntegrationGpuKernelIngestor, ExecutesTheSelectedKernelOnDevice)
{
    const auto& testCase = GetParam();

    auto graph = buildPointwiseAddGraph();
    buildAndCompile(*graph);

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    for(int iteration = 0; iteration < testCase.iterations; ++iteration)
    {
        executeAndVerify(*graph, workspace.get(), static_cast<unsigned int>(iteration));
    }
}

// hipDNN never exposes a caller-settable graph identity through the frontend:
// GraphDescriptor::finalize() always synthesizes a fresh UUID, so every graph built
// through Graph::pointwise() is a guaranteed catalog miss relative to every other one.
// This is the frontend-reachable form of the original file's uncacheable-graph
// coverage: two independently built graphs, each compiled and executed in the same
// test body, both must still match the reference executor -- proving the recompute
// path is correct rather than accidentally relying on state left by a prior graph.
TEST_F(IntegrationGpuKernelIngestor, ExecutesTwoIndependentlyBuiltGraphsCorrectly)
{
    auto graphA = buildPointwiseAddGraph();
    buildAndCompile(*graphA);
    int64_t workspaceSizeA = 0;
    ASSERT_EQ(graphA->get_workspace_size(workspaceSizeA).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspaceA(static_cast<size_t>(workspaceSizeA));
    executeAndVerify(*graphA, workspaceA.get(), 0);

    auto graphB = buildPointwiseAddGraph();
    buildAndCompile(*graphB);
    int64_t workspaceSizeB = 0;
    ASSERT_EQ(graphB->get_workspace_size(workspaceSizeB).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspaceB(static_cast<size_t>(workspaceSizeB));
    executeAndVerify(*graphB, workspaceB.get(), 1);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuKernelIngestor,
                         ::testing::Values(ExecuteCase{"SingleExecute", 1},
                                           ExecuteCase{"RepeatedExecute", 3}),
                         [](const ::testing::TestParamInfo<ExecuteCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
