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
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuKernelIngestor.cpp
 * @brief The kernel ingestor pack, end to end through the hipDNN frontend API: proves
 *        the descriptor set, matchers, heuristic, and dispatch handler compose in
 *        production, driving the same path a real caller takes.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Pointwise";
constexpr const char* CONV_ENGINE_NAME = "hipkernel:ConvFwd";
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

std::shared_ptr<Graph> buildPointwiseMulGraph()
{
    return buildPointwiseGraph(PointwiseMode::MUL);
}

std::shared_ptr<Graph> buildPointwiseSubGraph()
{
    return buildPointwiseGraph(PointwiseMode::SUB);
}

/// N=1, C=2, H=4, W=4, K=3, R=3, S=3, unit stride/dilation, no padding, cross-correlation,
/// NCHW/KCRS. y's dims/strides are left unset -- infer_properties_node() derives NKPQ
/// from x, w and the attributes -- and keeps uid 3 to match executeAndVerify()'s
/// hardcoded output uid.
std::shared_ptr<Graph> buildConvFwdGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("conv_fwd")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 4, 4})
        .set_stride({32, 16, 4, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({3, 2, 3, 3})
        .set_stride({18, 9, 3, 1})
        .set_data_type(DataType::FLOAT);

    ConvFpropAttributes attrs;
    attrs.set_name("conv_fwd")
        .set_padding({0, 0})
        .set_stride({1, 1})
        .set_dilation({1, 1})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, attrs);
    y->set_uid(3).set_name("Y").set_output(true).set_data_type(DataType::FLOAT);

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

/// A single call, and several reusing the same built plan.
struct ExecuteCase
{
    std::string name;
    int iterations;
};

/// How many times the composite plan has resolved a winner, counted from the plugin's
/// selection log. BenchmarkPlan emits exactly one of these per sampling sweep.
size_t countSelectionLogs(const hipdnn_test_sdk::utilities::LogRecorderBase& recorder)
{
    const auto logs = recorder.getRecordedLogs();
    return static_cast<size_t>(std::count_if(logs.begin(), logs.end(), [](const auto& log) {
        return log.message.find("benchmarking selected kernel") != std::string::npos;
    }));
}

/// Captures plugin logs for one test and restores every piece of process-global state it
/// touched. Both the global log level and the user callback registration outlive the
/// test otherwise: a raised level changes what later tests emit, and a callback keyed on
/// a destroyed fixture would stay registered. Manual teardown at the end of the body is
/// not enough, because an early ASSERT return skips it.
class ScopedPluginLogCapture
{
public:
    explicit ScopedPluginLogCapture(void* userHandle)
        : _userHandle(userHandle)
    {
        const auto levelRead = hipdnn_frontend::getGlobalLogLevel(_previousLevel);
        EXPECT_EQ(levelRead.code, ErrorCode::OK) << levelRead.err_msg;

        const auto registered = setCallback(HIPDNN_SEV_INFO);
        EXPECT_EQ(registered.code, ErrorCode::OK) << registered.err_msg;
        _registered = registered.code == ErrorCode::OK;

        const auto levelSet = hipdnn_frontend::setGlobalLogLevel(HIPDNN_SEV_INFO);
        EXPECT_EQ(levelSet.code, ErrorCode::OK) << levelSet.err_msg;
    }

    ~ScopedPluginLogCapture()
    {
        if(_registered)
        {
            static_cast<void>(setCallback(HIPDNN_SEV_OFF));
        }
        static_cast<void>(hipdnn_frontend::setGlobalLogLevel(_previousLevel));
    }

    ScopedPluginLogCapture(const ScopedPluginLogCapture&) = delete;
    ScopedPluginLogCapture& operator=(const ScopedPluginLogCapture&) = delete;
    ScopedPluginLogCapture(ScopedPluginLogCapture&&) = delete;
    ScopedPluginLogCapture& operator=(ScopedPluginLogCapture&&) = delete;

    hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder() const
    {
        return _recorder;
    }

private:
    hipdnn_frontend::Error setCallback(hipdnnSeverity_t minLevel) const
    {
        return hipdnn_frontend::setUserLogCallback(
            hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
            minLevel,
            hipdnn_frontend::LogCallbackMode::SYNC,
            _userHandle);
    }

    // Declared before the recorder so the recorder's own saved-level restore runs first.
    hipdnnSeverity_t _previousLevel = HIPDNN_SEV_OFF;
    void* _userHandle;
    bool _registered = false;
    mutable hipdnn_test_sdk::utilities::IsolatedLogRecorder _recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
};

} // namespace

class IntegrationGpuKernelIngestor
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      ExecuteCase>
{
protected:
    /// A cache root private to ONE test case, not merely to this binary.
    ///
    /// main() already scopes HIPDNN_CACHE_DIR away from the developer's ~/.cache/hipdnn,
    /// which stops one RUN inheriting another's shards. It does not stop one CASE
    /// inheriting another's: every case in this process shares that single root.
    /// ExecutesCorrectlyWithBenchmarkingEnabled records a measured ranking for a
    /// single-node FLOAT add, and ReportsAKnobWhoseValuesComeFromTheCatalog and
    /// ReportsTheMaximumWorkspaceAcrossSurvivingKernels then read it back: a benchmarked
    /// record replaces the heuristic order, so the knob default and the plan's workspace
    /// both follow the measured winner instead of the catalog's own ranking.
    ///
    /// That made those two cases fail under --gtest_shuffle, and -- because the shard
    /// outlives the process -- in default order too, on any machine where an earlier run
    /// left one behind. It is also nondeterministic rather than merely order-dependent:
    /// the two surviving FLOAT candidates differ by a few nanoseconds, so which one the
    /// sweep records is a coin flip.
    ///
    /// Scope::TEST rather than the default: a deferring instance nested inside main()'s
    /// would see HIPDNN_CACHE_DIR already set and quietly own nothing, isolating exactly
    /// nothing. Destroyed with the fixture, so the redirect is undone and the scratch
    /// tree removed on the failing path as well as the passing one.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "ingestor-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    static int64_t convEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(CONV_ENGINE_NAME);
    }

    /// Pins @p pinnedEngineId before plan creation and compiles with default knobs.
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

    void buildAndCompile(Graph& graph)
    {
        buildAndCompile(graph, engineId());
    }

    /// Like buildAndCompile(), but drives create_execution_plan_ext() with explicit
    /// knob settings instead of create_execution_plans()'s heuristic default path.
    /// That is the only way to set global.benchmarking, which add_engine_sweep() and
    /// the default heuristic path both strip.
    void buildAndCompileWithKnobs(Graph& graph,
                                  int64_t pinnedEngineId,
                                  const std::vector<KnobSetting>& knobSettings)
    {
        graph.set_preferred_engine_id_ext(pinnedEngineId);

        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_FALSE(rankedEngineIds.empty());

        result = graph.create_execution_plan_ext(rankedEngineIds.front(), knobSettings);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }
};

// Direct ABI: load-time self-registration

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

// Applicability

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

// Engine details: knobs from the catalog

TEST_F(IntegrationGpuKernelIngestor, ReportsAKnobWhoseValuesComeFromTheCatalog)
{
    auto graph = buildPointwiseAddGraph();

    auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<Knob> knobs;
    result = graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Two knobs: the engine's own block_size, plus the benchmarking knob every
    // descriptor-backed engine advertises out-of-band. Found by name rather than by
    // index, since the out-of-band knob is prepended.
    ASSERT_EQ(knobs.size(), 2U);
    const auto blockSizeKnob = std::find_if(knobs.begin(), knobs.end(), [](const Knob& knob) {
        return knob.knobId() == BLOCK_SIZE_KNOB;
    });
    ASSERT_NE(blockSizeKnob, knobs.end());
    EXPECT_NE(std::find_if(knobs.begin(),
                           knobs.end(),
                           [](const Knob& knob) {
                               return knob.knobId() == hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME;
                           }),
              knobs.end());

    // The HALF kernel is pruned for this FLOAT graph.
    const auto* constraint = dynamic_cast<const IntConstraint*>(blockSizeKnob->constraint());
    ASSERT_NE(constraint, nullptr);
    const auto& validValues = constraint->getValidValues();
    EXPECT_EQ(validValues, (std::unordered_set<int64_t>{64, 256}));

    const auto* defaultValue = std::get_if<int64_t>(&blockSizeKnob->defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, 256);
}

// Workspace

TEST_F(IntegrationGpuKernelIngestor, ReportsTheMaximumWorkspaceAcrossSurvivingKernels)
{
    auto graph = buildPointwiseAddGraph();
    buildAndCompile(*graph);

    // A max across the catalog, not one kernel's value.
    int64_t workspaceSize = 0;
    auto result = graph->get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(workspaceSize, EXPECTED_WORKSPACE_BYTES);
}

// Plan build and execute
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

// global.benchmarking: the composite plan built when the knob is set

/// Drives global.benchmarking=1 through the frontend against the shipped pointwise
/// pack, verifying the numerical result against the CPU reference and confirming from
/// the plugin's own logs that the composite plan actually ran a sampling sweep and
/// resolved a winner once.
///
/// Which candidate wins is deliberately not asserted: the two block-size-64/256 FLOAT
/// candidates surviving knob filtering for this graph may be indistinguishable within
/// noise, and either winner is correct so long as it produces the right answer. What
/// must hold is that benchmarking happened at all -- otherwise the case would pass
/// identically with the feature removed.
TEST_F(IntegrationGpuKernelIngestor, ExecutesCorrectlyWithBenchmarkingEnabled)
{
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();

    auto graph = buildPointwiseAddGraph();

    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1});
    buildAndCompileWithKnobs(*graph, engineId(), knobSettings);

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    // buildPlan() took the benchmarking branch rather than the single-plan one, and it
    // had more than one candidate to choose between: a one-candidate sweep would prove
    // nothing about selection.
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark"))
        << "buildPlan() did not take the benchmarking branch. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark 1 candidate(s)"))
        << "expected more than one candidate to benchmark. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // The first execute() samples every candidate; the second reuses the cached winner.
    // Both must produce the correct result, and executeAndVerify() re-randomizes and
    // re-checks each time.
    executeAndVerify(*graph, workspace.get(), /*seed=*/0);

    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel"))
        << "the sampling sweep did not resolve a winner. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    const size_t selectionsAfterFirstExecute = countSelectionLogs(recorder);
    ASSERT_EQ(selectionsAfterFirstExecute, 1U)
        << "expected exactly one selection sweep. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    executeAndVerify(*graph, workspace.get(), /*seed=*/1);

    // The winner is resolved once for the plan's life: a second execute() must reuse it
    // rather than re-sample.
    EXPECT_EQ(countSelectionLogs(recorder), selectionsAfterFirstExecute)
        << "the second execute() re-sampled instead of reusing the winner. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // The capture guard restores the global log level and unregisters the callback,
    // including on an early ASSERT return above.
}

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
// Three packs, one provider: the topology commit 2 exists to prove
// ---------------------------------------------------------------------------

// The pack-based design's core claim: hipDNN routes each operation to the pack that
// claims it, with nothing above the packs aware any of them exist.
TEST_F(IntegrationGpuKernelIngestor, ResolvesEveryPointwiseOperationToTheOneEngine)
{
    auto addGraph = buildPointwiseAddGraph();
    ASSERT_EQ(addGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> addEngines;
    ASSERT_EQ(addGraph->get_ranked_engine_ids(addEngines).code, ErrorCode::OK);

    auto mulGraph = buildPointwiseMulGraph();
    ASSERT_EQ(mulGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> mulEngines;
    ASSERT_EQ(mulGraph->get_ranked_engine_ids(mulEngines).code, ErrorCode::OK);

    auto subGraph = buildPointwiseSubGraph();
    ASSERT_EQ(subGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> subEngines;
    ASSERT_EQ(subGraph->get_ranked_engine_ids(subEngines).code, ErrorCode::OK);

    const auto offers = [](const std::vector<int64_t>& engines, int64_t id) {
        return std::find(engines.begin(), engines.end(), id) != engines.end();
    };

    EXPECT_TRUE(offers(addEngines, engineId()));
    EXPECT_TRUE(offers(mulEngines, engineId()));
    EXPECT_TRUE(offers(subEngines, engineId()));
}

// Numeric proof, not just routing: a-b and b-a are both plausible, so only comparing
// against the CPU reference catches an operand swap in the third pack's binding.
TEST_F(IntegrationGpuKernelIngestor, ExecutesASubtractGraphThroughItsOwnPack)
{
    auto graph = buildPointwiseSubGraph();
    buildAndCompile(*graph, engineId());

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    executeAndVerify(*graph, workspace.get(), 0);
}

// Numeric, not just routing: a+b and a*b are both plausible for the same operands, so
// only the CPU reference catches the engine reaching the wrong pack's kernel.
TEST_F(IntegrationGpuKernelIngestor, ExecutesBothOperationsOfOneEngineThroughDifferentPacks)
{
    auto addGraph = buildPointwiseAddGraph();
    buildAndCompile(*addGraph, engineId());
    int64_t addWorkspaceSize = 0;
    ASSERT_EQ(addGraph->get_workspace_size(addWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace addWorkspace(static_cast<size_t>(addWorkspaceSize));
    executeAndVerify(*addGraph, addWorkspace.get(), 0);

    // Same engine id: the pack is chosen by the operation matcher, not by the caller.
    auto mulGraph = buildPointwiseMulGraph();
    buildAndCompile(*mulGraph, engineId());
    int64_t mulWorkspaceSize = 0;
    ASSERT_EQ(mulGraph->get_workspace_size(mulWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace mulWorkspace(static_cast<size_t>(mulWorkspaceSize));
    executeAndVerify(*mulGraph, mulWorkspace.get(), 1);

    // The engine's catalog is keyed per graph, so the add graph still answers after a
    // second pack of the same engine has run and cached its own.
    executeAndVerify(*addGraph, addWorkspace.get(), 2);
}

// Catalogs are cached under (graph, device) keys in the engine's state manager; running
// a third pack between two runs of the first proves no pack's cached state leaks into
// another's -- a failure mode that only exists once one descriptor set serves several.
TEST_F(IntegrationGpuKernelIngestor, ExecutesBothPacksInOneProcessWithoutInterference)
{
    auto addGraph = buildPointwiseAddGraph();
    buildAndCompile(*addGraph, engineId());
    int64_t addWorkspaceSize = 0;
    ASSERT_EQ(addGraph->get_workspace_size(addWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace addWorkspace(static_cast<size_t>(addWorkspaceSize));
    executeAndVerify(*addGraph, addWorkspace.get(), 0);

    auto subGraph = buildPointwiseSubGraph();
    buildAndCompile(*subGraph, engineId());
    int64_t subWorkspaceSize = 0;
    ASSERT_EQ(subGraph->get_workspace_size(subWorkspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace subWorkspace(static_cast<size_t>(subWorkspaceSize));
    executeAndVerify(*subGraph, subWorkspace.get(), 1);

    // Confirms the add graph still answers correctly after the sub graph ran.
    executeAndVerify(*addGraph, addWorkspace.get(), 2);
}

// ---------------------------------------------------------------------------
// A second engine, split by graph node type
// ---------------------------------------------------------------------------

// Numeric proof for the second engine: only the CPU reference catches a swapped operand
// or a wrong accumulation order in the naive kernel.
TEST_F(IntegrationGpuKernelIngestor, ExecutesAConvForwardGraphOnDevice)
{
    auto graph = buildConvFwdGraph();
    buildAndCompile(*graph, convEngineId());

    int64_t workspaceSize = 0;
    ASSERT_EQ(graph->get_workspace_size(workspaceSize).code, ErrorCode::OK);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    // C*R*S = 2*3*3: every output element is an 18-term sum, so it is held to an
    // 18-term tolerance rather than the pointwise default of bit-exactness.
    executeAndVerify(*graph, workspace.get(), 0, /*reductionLength=*/2 * 3 * 3);
}

// The claim the graph-node-type split exists to make: the two engines don't overlap.
// Complements ResolvesEveryPointwiseOperationToTheOneEngine, which already shows every
// pointwise operation lands on the one engine.
TEST_F(IntegrationGpuKernelIngestor, ResolvesAConvGraphToTheConvEngineAndNotThePointwiseOne)
{
    auto convGraph = buildConvFwdGraph();
    ASSERT_EQ(convGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> convEngines;
    ASSERT_EQ(convGraph->get_ranked_engine_ids(convEngines).code, ErrorCode::OK);

    auto pointwiseGraph = buildPointwiseAddGraph();
    ASSERT_EQ(pointwiseGraph->build_operation_graph(_handle).code, ErrorCode::OK);
    std::vector<int64_t> pointwiseEngines;
    ASSERT_EQ(pointwiseGraph->get_ranked_engine_ids(pointwiseEngines).code, ErrorCode::OK);

    const auto offers = [](const std::vector<int64_t>& engines, int64_t id) {
        return std::find(engines.begin(), engines.end(), id) != engines.end();
    };

    EXPECT_TRUE(offers(convEngines, convEngineId()));
    EXPECT_FALSE(offers(convEngines, engineId()));

    EXPECT_TRUE(offers(pointwiseEngines, engineId()));
    EXPECT_FALSE(offers(pointwiseEngines, convEngineId()));
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
