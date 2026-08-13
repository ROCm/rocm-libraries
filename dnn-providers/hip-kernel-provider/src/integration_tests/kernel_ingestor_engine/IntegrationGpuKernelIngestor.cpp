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
 * Drives the same path a real caller takes -- Graph::pointwise(),
 * get_ranked_engine_ids(), get_knobs_for_engine(), get_workspace_size(), execute() --
 * so the descriptor set, the matchers, the heuristic, and the dispatch handler are
 * exercised as they compose in production.
 *
 * Exception: `hipdnnEnginePluginGetAllEngineIds` enumerates every engine a plugin
 * exports independent of any graph, and has no frontend equivalent, so that one
 * assertion crosses the raw C ABI via a minimal dlopen helper.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:PointwiseAdd";
constexpr const char* SUB_ENGINE_NAME = "hipkernel:PointwiseSub";
constexpr const char* BLOCK_SIZE_KNOB = "block_size";

/// Maximum workspace across the pack's surviving kernels for a FLOAT graph.
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

/// A single binary-pointwise node over 1-element FLOAT tensors: the one shape these
/// packs accept, differing only in operation. Each call returns a fresh graph, never
/// shared build/plan state.
std::shared_ptr<Graph> buildPointwiseGraph(PointwiseMode mode)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("pointwise")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = makeScalarTensor(1, "A");
    auto b = makeScalarTensor(2, "B");

    PointwiseAttributes attrs;
    attrs.set_name("pointwise").set_mode(mode);
    auto c = graph->pointwise(a, b, attrs);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

std::shared_ptr<Graph> buildPointwiseAddGraph()
{
    return buildPointwiseGraph(PointwiseMode::ADD);
}

std::shared_ptr<Graph> buildPointwiseSubGraph()
{
    return buildPointwiseGraph(PointwiseMode::SUB);
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

/// Execute shapes: a single call, and several reusing the same built plan.
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

    static int64_t subEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(SUB_ENGINE_NAME);
    }

    /// Builds `graph`, pins @p pinnedEngineId before plan creation, and compiles a plan
    /// with default knobs.
    void buildAndCompile(Graph& graph, int64_t pinnedEngineId)
    {
        graph.set_preferred_engine_id_ext(pinnedEngineId);

        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    /// Builds and compiles pinned to the add pack, the default for the tests below
    /// that predate the second engine.
    void buildAndCompile(Graph& graph)
    {
        buildAndCompile(graph, engineId());
    }

    /// Builds fresh CPU/GPU tensor bundles for `graph`, executes it once on GPU with
    /// `workspace`, and verifies against CpuReferenceGraphExecutor. `seed` varies the
    /// input values so repeated calls never compare against stale buffers.
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
            // Offset per uid: randomizeTensor seeds a fresh mt19937, so a shared seed
            // makes every operand byte-identical and allClose() would pass on a+a as
            // readily as a+b, no longer witnessing uid-based argument resolution.
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
        // Proves the full chain: matcher admission, heuristic ranking, handler compile
        // and launch, and argument resolution by tensor uid.
        EXPECT_TRUE(CpuFpReferenceValidation<float>().allClose(cpuOut, gpuOut));
    }
};

// ---------------------------------------------------------------------------
// Direct ABI: load-time self-registration
// ---------------------------------------------------------------------------

// The frontend never lists "every engine this plugin exports", so proving load-time
// self-registration needs the raw C ABI.
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

    // The pack ships three kernels; the HALF one is pruned for this FLOAT graph, so the
    // knob offers exactly the two block sizes the surviving kernels implement.
    const auto* constraint = dynamic_cast<const IntConstraint*>(knobs[0].constraint());
    ASSERT_NE(constraint, nullptr);
    const auto& validValues = constraint->getValidValues();
    EXPECT_EQ(validValues, (std::unordered_set<int64_t>{64, 256}));

    // The default is whatever the heuristic ranked first.
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

    // One surviving kernel declares 0 bytes and the other 1024, so this proves the
    // query aggregates across the catalog rather than reporting one kernel's value.
    int64_t workspaceSize = 0;
    auto result = graph->get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(workspaceSize, EXPECTED_WORKSPACE_BYTES);
}

// ---------------------------------------------------------------------------
// Plan build and execute
// ---------------------------------------------------------------------------

// Every decision the plan makes happens at build, so execution must never depend on
// the previous call.
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

// GraphDescriptor::finalize() always synthesizes a fresh UUID, so every graph built
// through the frontend is a guaranteed catalog miss relative to every other one. Two
// independently built graphs must both match the reference executor, proving the
// recompute path is correct rather than relying on state left by a prior graph.
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

// ---------------------------------------------------------------------------
// Two packs, one provider: the topology commit 2 exists to prove
// ---------------------------------------------------------------------------

// The claim the seam makes good on. Two engines are described entirely by data and
// native symbols resolved by name, and hipDNN routes a graph to the right one with no
// code above the packs knowing either exists.
TEST_F(IntegrationGpuKernelIngestor, ResolvesEachOperationToItsOwnEngine)
{
    auto addGraph = buildPointwiseAddGraph();
    ASSERT_EQ(addGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> addEngines;
    ASSERT_EQ(addGraph->get_ranked_engine_ids(addEngines).code, ErrorCode::OK);

    auto subGraph = buildPointwiseSubGraph();
    ASSERT_EQ(subGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> subEngines;
    ASSERT_EQ(subGraph->get_ranked_engine_ids(subEngines).code, ErrorCode::OK);

    const auto offers = [](const std::vector<int64_t>& engines, int64_t id) {
        return std::find(engines.begin(), engines.end(), id) != engines.end();
    };

    // Each engine claims its own operation...
    EXPECT_TRUE(offers(addEngines, engineId()));
    EXPECT_TRUE(offers(subEngines, subEngineId()));
    // ...and declines the other's. Without this the two packs would both match every
    // pointwise graph and selection between them would be arbitrary.
    EXPECT_FALSE(offers(addEngines, subEngineId()));
    EXPECT_FALSE(offers(subEngines, engineId()));
}

// Numeric proof, not just routing: a-b and b-a are both plausible, so only comparing
// against the CPU reference catches an operand swap in the second pack's binding.
TEST_F(IntegrationGpuKernelIngestor, ExecutesASubtractGraphThroughItsOwnPack)
{
    auto graph = buildPointwiseSubGraph();
    buildAndCompile(*graph, subEngineId());

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    executeAndVerify(*graph, workspace.get(), 0);
}

// Both packs' catalogs are cached under (graph, device) keys in per-engine state
// managers. Executing one after the other proves neither engine's catalog or bound
// token state reaches the other -- the failure mode that only exists once a provider
// serves more than one descriptor set.
TEST_F(IntegrationGpuKernelIngestor, ExecutesBothPacksInOneProcessWithoutInterference)
{
    auto addGraph = buildPointwiseAddGraph();
    buildAndCompile(*addGraph, engineId());
    int64_t addWorkspaceSize = 0;
    ASSERT_EQ(addGraph->get_workspace_size(addWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace addWorkspace(static_cast<size_t>(addWorkspaceSize));
    executeAndVerify(*addGraph, addWorkspace.get(), 0);

    auto subGraph = buildPointwiseSubGraph();
    buildAndCompile(*subGraph, subEngineId());
    int64_t subWorkspaceSize = 0;
    ASSERT_EQ(subGraph->get_workspace_size(subWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace subWorkspace(static_cast<size_t>(subWorkspaceSize));
    executeAndVerify(*subGraph, subWorkspace.get(), 1);

    // And the add graph still answers correctly after the sub graph ran.
    executeAndVerify(*addGraph, addWorkspace.get(), 2);
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
