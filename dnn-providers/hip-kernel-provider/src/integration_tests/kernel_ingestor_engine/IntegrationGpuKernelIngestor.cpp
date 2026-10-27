// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/execution_plan_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../engines/kernel_ingestor_engine/serialization/ingestor_plan_generated.h"
#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

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

/// In epsilons of the fixture's element type. Elementwise ops accumulate nothing, so one
/// epsilon is the whole budget.
constexpr float POINTWISE_TOLERANCE_EPSILONS = 1.0f;

/// Maximum workspace across the pack's surviving kernels for a FLOAT graph.
constexpr int64_t EXPECTED_WORKSPACE_BYTES = 1024;

// In epsilons of the fixture's element type. Each output of the convolution graph is a sum
// of C*R*S = 2*3*3 terms.
constexpr float CONV_FWD_TOLERANCE_EPSILONS = 2 * 3 * 3;

// The fixed layout of the ingestor plan payload: the marker, then the format major and minor
// as little-endian 16-bit values. The header is 48 bytes, and the payload body follows it.
constexpr std::array<uint8_t, 4> INGESTOR_PLAN_MARKER = {'H', 'K', 'I', 'P'};
constexpr size_t INGESTOR_PLAN_FORMAT_MAJOR_OFFSET = 4;
constexpr size_t INGESTOR_PLAN_FORMAT_MINOR_OFFSET = 6;
constexpr size_t INGESTOR_PLAN_HEADER_SIZE = 48;

constexpr std::array<SaveForm, 2> SAVE_FORMS = {SaveForm::GRAPH_AND_PLAN, SaveForm::PLAN_ONLY};

const char* saveFormName(SaveForm form)
{
    return form == SaveForm::GRAPH_AND_PLAN ? "graph and plan" : "plan only";
}

// Checks that a save or a load failed in the plugin with status, and that the message holds
// phrase.
void expectPluginRefusal(const Error& error, hipdnnPluginStatus_t status, const std::string& phrase)
{
    EXPECT_EQ(error.code, ErrorCode::HIPDNN_BACKEND_ERROR) << error.err_msg;
    EXPECT_NE(error.err_msg.find(::toString(status)), std::string::npos)
        << "expected " << ::toString(status) << " in: " << error.err_msg;
    EXPECT_NE(error.err_msg.find(phrase), std::string::npos)
        << "expected '" << phrase << "' in: " << error.err_msg;
}

uint16_t readLittleEndianU16(const std::vector<uint8_t>& bytes, size_t offset)
{
    return static_cast<uint16_t>(bytes.at(offset) | (bytes.at(offset + 1) << 8));
}

void writeLittleEndianU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value)
{
    bytes.at(offset) = static_cast<uint8_t>(value & 0xFFU);
    bytes.at(offset + 1) = static_cast<uint8_t>(value >> 8);
}

// Finds the plugin payload in the bytes of a plan-only save. Checks that the payload is an
// ingestor plan.
void findIngestorPlanPayload(const std::vector<uint8_t>& saved, size_t& offset, size_t& size)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::Verifier verifier(saved.data(), saved.size());
    ASSERT_TRUE(data_objects::VerifySerializedExecutionPlanBuffer(verifier));
    const auto* payload = data_objects::GetSerializedExecutionPlan(saved.data())->plugin_payload();
    ASSERT_NE(payload, nullptr);
    ASSERT_GT(payload->size(), INGESTOR_PLAN_HEADER_SIZE);
    ASSERT_TRUE(
        std::equal(INGESTOR_PLAN_MARKER.begin(), INGESTOR_PLAN_MARKER.end(), payload->begin()))
        << "the plugin payload is not an ingestor plan";

    offset = static_cast<size_t>(payload->data() - saved.data());
    size = payload->size();
}

// Finds the stored code object in the bytes of a plan-only save of an ingestor plan.
void findCodeObject(const std::vector<uint8_t>& saved, size_t& offset, size_t& size)
{
    namespace fb = serialization::fb;

    size_t payloadOffset = 0;
    size_t payloadSize = 0;
    ASSERT_NO_FATAL_FAILURE(findIngestorPlanPayload(saved, payloadOffset, payloadSize));
    const size_t bodyOffset = payloadOffset + INGESTOR_PLAN_HEADER_SIZE;
    const size_t bodySize = payloadSize - INGESTOR_PLAN_HEADER_SIZE;

    // FlatBuffers reads need an aligned body, and the payload sits at any address.
    std::vector<uint64_t> aligned((bodySize + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    std::memcpy(aligned.data(), saved.data() + bodyOffset, bodySize);
    const auto* body = reinterpret_cast<const uint8_t*>(aligned.data());

    flatbuffers::Verifier verifier(body, bodySize);
    ASSERT_TRUE(fb::VerifyExecutionPlanBuffer(verifier));
    const auto* codeObject = fb::GetExecutionPlan(body)->kernel_image()->code_object();
    ASSERT_NE(codeObject, nullptr);
    ASSERT_GT(codeObject->size(), 0U);

    offset = bodyOffset + static_cast<size_t>(codeObject->data() - body);
    size = codeObject->size();
    ASSERT_LE(offset + size, payloadOffset + payloadSize);
}

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
/// from x, w and the attributes. uids are assigned in input order, so y takes 3.
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

/// How many sampling sweeps the composite plan has resolved, counted from the plugin's
/// logs. BenchmarkPlan ends every sweep with exactly one of these two lines: the winner,
/// or -- when no candidate yielded a usable time -- the fallback to the first candidate.
/// Both count: on Windows, where HIP reports negative event readings
/// (ROCm/rocm-systems#12925), bad readings can leave every candidate unusable.
size_t countSweepResolutions(const hipdnn_test_sdk::utilities::LogRecorderBase& recorder)
{
    const auto logs = recorder.getRecordedLogs();
    return static_cast<size_t>(std::count_if(logs.begin(), logs.end(), [](const auto& log) {
        return log.message.find("benchmarking selected kernel") != std::string::npos
               || log.message.find("benchmarking found no usable candidate") != std::string::npos;
    }));
}

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

    /// Offsets the seed by UID so the binary operands differ: `a + b` and `a + a` agree
    /// elementwise when both operands carry the same data, and this engine's catalog is
    /// entirely elementwise binary ops.
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   DEFAULT_MIN,
                                   DEFAULT_MAX,
                                   seed + static_cast<unsigned int>(tensorPair.first));
        }
    }

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
    GraphVerificationContext context(*graph);
    registerValidatorsForOutputs(context, POINTWISE_TOLERANCE_EPSILONS);

    for(int iteration = 0; iteration < testCase.iterations; ++iteration)
    {
        verifyBuiltGraph(context, static_cast<unsigned int>(iteration));
    }
}

// global.benchmarking: the composite plan built when the knob is set

/// Drives global.benchmarking=1 through the frontend against the shipped pointwise
/// pack, verifying the numerical result against the CPU reference and confirming from
/// the plugin's own logs that the composite plan actually ran a sampling sweep, exactly
/// once.
///
/// Which candidate wins is not asserted: the two block-size-64/256 FLOAT candidates
/// surviving knob filtering for this graph may be indistinguishable within noise, and
/// either is correct so long as it produces the right answer. A winner is required except
/// on Windows, where bad HIP event readings can leave both unusable and the plan then
/// serves the first. What must hold everywhere is that benchmarking happened at all --
/// otherwise the case would pass identically with the feature removed.
TEST_F(IntegrationGpuKernelIngestor, ExecutesCorrectlyWithBenchmarkingEnabled)
{
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();

    auto graph = buildPointwiseAddGraph();

    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1});
    buildAndCompileWithKnobs(*graph, engineId(), knobSettings);

    // The benchmarking branch ran with more than one candidate.
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark"))
        << "buildPlan() did not take the benchmarking branch. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark 1 candidate(s)"))
        << "expected more than one candidate to benchmark. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // The first execute() samples every candidate, the second reuses the cached winner;
    // verifyBuiltGraph() re-randomizes and re-checks each time.
    GraphVerificationContext context(*graph);
    registerValidatorsForOutputs(context, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(context, /*seed=*/0);

#ifndef _WIN32
    // HIP event timing is reliable here, so a sweep without a usable candidate is a bug.
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel"))
        << "the sampling sweep did not resolve a winner. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
#endif

    const size_t sweepsAfterFirstExecute = countSweepResolutions(recorder);
    ASSERT_EQ(sweepsAfterFirstExecute, 1U)
        << "expected exactly one sampling sweep. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    verifyBuiltGraph(context, /*seed=*/1);

    // The sweep resolves once for the plan's life: a second execute() must reuse its
    // choice rather than re-sample.
    EXPECT_EQ(countSweepResolutions(recorder), sweepsAfterFirstExecute)
        << "the second execute() re-sampled instead of reusing the winner. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // The capture guard restores the global log level and unregisters the callback,
    // including on an early ASSERT return above.
}

TEST_F(IntegrationGpuKernelIngestor, ExecutesTwoIndependentlyBuiltGraphsCorrectly)
{
    auto graphA = buildPointwiseAddGraph();
    buildAndCompile(*graphA);
    GraphVerificationContext contextA(*graphA);
    registerValidatorsForOutputs(contextA, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(contextA, 0);

    auto graphB = buildPointwiseAddGraph();
    buildAndCompile(*graphB);
    GraphVerificationContext contextB(*graphB);
    registerValidatorsForOutputs(contextB, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(contextB, 1);
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

    GraphVerificationContext context(*graph);
    registerValidatorsForOutputs(context, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(context, 0);
}

// Numeric, not just routing: a+b and a*b are both plausible for the same operands, so
// only the CPU reference catches the engine reaching the wrong pack's kernel.
TEST_F(IntegrationGpuKernelIngestor, ExecutesBothOperationsOfOneEngineThroughDifferentPacks)
{
    auto addGraph = buildPointwiseAddGraph();
    buildAndCompile(*addGraph, engineId());
    GraphVerificationContext addContext(*addGraph);
    registerValidatorsForOutputs(addContext, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(addContext, 0);

    // Same engine id: the pack is chosen by the operation matcher, not by the caller.
    auto mulGraph = buildPointwiseMulGraph();
    buildAndCompile(*mulGraph, engineId());
    GraphVerificationContext mulContext(*mulGraph);
    registerValidatorsForOutputs(mulContext, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(mulContext, 1);

    // The engine's catalog is keyed per graph, so the add graph still answers after a
    // second pack of the same engine cached its own.
    verifyBuiltGraph(addContext, 2);
}

// Catalogs are cached under (graph, device) keys in the engine's state manager; running
// a third pack between two runs of the first proves no pack's cached state leaks into
// another's -- a failure mode that only exists once one descriptor set serves several.
TEST_F(IntegrationGpuKernelIngestor, ExecutesBothPacksInOneProcessWithoutInterference)
{
    auto addGraph = buildPointwiseAddGraph();
    buildAndCompile(*addGraph, engineId());
    GraphVerificationContext addContext(*addGraph);
    registerValidatorsForOutputs(addContext, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(addContext, 0);

    auto subGraph = buildPointwiseSubGraph();
    buildAndCompile(*subGraph, engineId());
    GraphVerificationContext subContext(*subGraph);
    registerValidatorsForOutputs(subContext, POINTWISE_TOLERANCE_EPSILONS);
    verifyBuiltGraph(subContext, 1);

    verifyBuiltGraph(addContext, 2);
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

    // C*R*S = 2*3*3: every output element is an 18-term sum, and GPU and CPU accumulate
    // in different orders, so the budget is 18 epsilons rather than the elementwise one.
    GraphVerificationContext context(*graph);
    registerValidatorsForOutputs(context, CONV_FWD_TOLERANCE_EPSILONS);
    verifyBuiltGraph(context, 0);
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

// ---------------------------------------------------------------------------
// Saved plans
// ---------------------------------------------------------------------------

class IntegrationGpuKernelIngestorSaveLoad : public IntegrationGpuKernelIngestor
{
protected:
    // Builds graph with pinnedEngineId, validates one run and saves the plan in form. Then
    // it destroys graph, loads the plan with a new handle and validates two runs of the
    // loaded plan. Every run uses the CPU reference of the original graph. Then it checks
    // that the loaded plan cannot be saved again.
    void roundTripInAFreshHandle(std::shared_ptr<Graph> graph,
                                 int64_t pinnedEngineId,
                                 SaveForm form,
                                 float epsilonMultiple)
    {
        {
            const ScopedPluginLogCapture capture(this);
            ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, pinnedEngineId));
            // The load below must log no selection. This check proves that the capture
            // records selection lines.
            ASSERT_TRUE(capture.recorder().hasLogContaining(detail::SELECTED_KERNEL_LOG))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }

        int64_t servingEngineId = 0;
        auto result = graph->get_execution_plan_engine_id(servingEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(servingEngineId, pinnedEngineId);

        std::vector<BehaviorNote> notes;
        result = graph->get_behavior_notes_for_engine(pinnedEngineId, notes);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(notes.begin(),
                            notes.end(),
                            BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
                  notes.end());

        VerificationRun run;
        {
            GraphVerificationContext context(*graph);
            ASSERT_NO_FATAL_FAILURE(registerValidatorsForOutputs(context, epsilonMultiple));
            ASSERT_NO_FATAL_FAILURE(prepareVerification(context, /*seed=*/0, run));
        }
        ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *graph, _handle));

        std::vector<uint8_t> saved;
        {
            const ScopedPluginLogCapture capture(this);
            auto save = detail::saveInForm(*graph, form);
            ASSERT_EQ(save.second.code, ErrorCode::OK) << save.second.err_msg;
            saved = std::move(save.first);
            EXPECT_FALSE(capture.recorder().hasLogContaining(
                "does not support execution plan serialization"))
                << "Captured logs:\n"
                << capture.recorder().getRecordedLogsAsString();
        }
        if(form == SaveForm::GRAPH_AND_PLAN)
        {
            int contents = 0;
            ASSERT_EQ(
                hipdnnBackendGetSerializedBinaryContents_ext(saved.data(), saved.size(), &contents),
                HIPDNN_STATUS_SUCCESS);
            ASSERT_NE(contents & HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN, 0)
                << "the saved graph holds no execution plan";
        }

        int64_t originalWorkspaceSize = 0;
        result = graph->get_workspace_size(originalWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const std::weak_ptr<Graph> original = graph;
        graph.reset();
        ASSERT_TRUE(original.expired()) << "another owner keeps the original graph alive";

        const ScopedPluginLogCapture capture(this);
        ASSERT_NO_FATAL_FAILURE(detail::replaceHandleWithAFreshOne(_handle, _stream));

        auto loaded = std::make_shared<Graph>();
        result = detail::loadInForm(*loaded, _handle, saved, form);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        EXPECT_TRUE(capture.recorder().hasLogContaining(detail::CREATING_CONTAINER_LOG))
            << "the new handle did not get a new provider container. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
        detail::expectNoKernelSelectionLogged(capture.recorder());

        int64_t loadedEngineId = 0;
        result = loaded->get_execution_plan_engine_id(loadedEngineId);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedEngineId, pinnedEngineId);

        int64_t loadedWorkspaceSize = 0;
        result = loaded->get_workspace_size(loadedWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_EQ(loadedWorkspaceSize, originalWorkspaceSize);

        ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *loaded, _handle));
        ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *loaded, _handle));

        const auto resave = detail::saveInForm(*loaded, form);
        expectPluginRefusal(
            resave.second, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "loaded from a saved payload");
    }
};

TEST_F(IntegrationGpuKernelIngestorSaveLoad, RestoresAPointwiseGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(buildPointwiseAddGraph(),
                            engineId(),
                            SaveForm::GRAPH_AND_PLAN,
                            POINTWISE_TOLERANCE_EPSILONS);
}

TEST_F(IntegrationGpuKernelIngestorSaveLoad, RestoresAPointwisePlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(
        buildPointwiseAddGraph(), engineId(), SaveForm::PLAN_ONLY, POINTWISE_TOLERANCE_EPSILONS);
}

TEST_F(IntegrationGpuKernelIngestorSaveLoad, RestoresAConvFwdGraphAndPlanInAFreshHandle)
{
    roundTripInAFreshHandle(
        buildConvFwdGraph(), convEngineId(), SaveForm::GRAPH_AND_PLAN, CONV_FWD_TOLERANCE_EPSILONS);
}

TEST_F(IntegrationGpuKernelIngestorSaveLoad, RestoresAConvFwdPlanOnlyInAFreshHandle)
{
    roundTripInAFreshHandle(
        buildConvFwdGraph(), convEngineId(), SaveForm::PLAN_ONLY, CONV_FWD_TOLERANCE_EPSILONS);
}

class IntegrationGpuKernelIngestorSaveLoadRefusals : public IntegrationGpuKernelIngestor
{
protected:
    // Builds the convolution graph with the ConvFwd engine and saves its plan only.
    void saveAConvFwdPlanOnly(std::vector<uint8_t>& saved)
    {
        auto graph = buildConvFwdGraph();
        ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, convEngineId()));
        auto save = detail::saveInForm(*graph, SaveForm::PLAN_ONLY);
        ASSERT_EQ(save.second.code, ErrorCode::OK) << save.second.err_msg;
        saved = std::move(save.first);
    }
};

TEST_F(IntegrationGpuKernelIngestorSaveLoadRefusals,
       RefusesToSaveABenchmarkingPlanBeforeItsFirstExecute)
{
    auto graph = buildPointwiseAddGraph();
    std::vector<KnobSetting> knobSettings;
    knobSettings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1});
    {
        const ScopedPluginLogCapture capture(this);
        ASSERT_NO_FATAL_FAILURE(buildAndCompileWithKnobs(*graph, engineId(), knobSettings));
        ASSERT_TRUE(capture.recorder().hasLogContaining("' will benchmark "))
            << "the plan is not a benchmarking plan. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
        ASSERT_FALSE(capture.recorder().hasLogContaining("will benchmark 1 candidate(s)"))
            << "the plan has one candidate only. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
    }

    for(const SaveForm form : SAVE_FORMS)
    {
        SCOPED_TRACE(saveFormName(form));
        const auto save = detail::saveInForm(*graph, form);
        expectPluginRefusal(
            save.second, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, "has not chosen a kernel yet");
    }

    VerificationRun run;
    {
        GraphVerificationContext context(*graph);
        ASSERT_NO_FATAL_FAILURE(
            registerValidatorsForOutputs(context, POINTWISE_TOLERANCE_EPSILONS));
        ASSERT_NO_FATAL_FAILURE(prepareVerification(context, /*seed=*/0, run));
    }
    ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, *graph, _handle));

    for(const SaveForm form : SAVE_FORMS)
    {
        SCOPED_TRACE(saveFormName(form));
        const auto save = detail::saveInForm(*graph, form);
        EXPECT_EQ(save.second.code, ErrorCode::OK) << save.second.err_msg;
        EXPECT_FALSE(save.first.empty());
    }
}

// The FlatBuffers Verifier accepts a body with one changed code-object byte, so only the
// body digest finds that damage.
TEST_F(IntegrationGpuKernelIngestorSaveLoadRefusals,
       RefusesToLoadDamagedBytesOrAnotherFormatMajorVersion)
{
    std::vector<uint8_t> saved;
    ASSERT_NO_FATAL_FAILURE(saveAConvFwdPlanOnly(saved));

    const auto loadWithOneCodeObjectByteChanged = [this](std::vector<uint8_t> copy) {
        SCOPED_TRACE("one code-object byte changed");
        size_t codeOffset = 0;
        size_t codeSize = 0;
        ASSERT_NO_FATAL_FAILURE(findCodeObject(copy, codeOffset, codeSize));

        const size_t middleOfCode = codeOffset + (codeSize / 2);
        copy.at(middleOfCode) = static_cast<uint8_t>(copy.at(middleOfCode) ^ 0xFFU);

        auto loaded = std::make_shared<Graph>();
        const auto result = detail::loadInForm(*loaded, _handle, copy, SaveForm::PLAN_ONLY);
        expectPluginRefusal(
            result, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "ingestor plan data is damaged");
        EXPECT_NE(result.err_msg.find("body digest mismatch"), std::string::npos) << result.err_msg;
    };

    const auto loadWithAnotherFormatMajorVersion = [this](std::vector<uint8_t> copy) {
        SCOPED_TRACE("another format major version");
        size_t payloadOffset = 0;
        size_t payloadSize = 0;
        ASSERT_NO_FATAL_FAILURE(findIngestorPlanPayload(copy, payloadOffset, payloadSize));

        const uint16_t major
            = readLittleEndianU16(copy, payloadOffset + INGESTOR_PLAN_FORMAT_MAJOR_OFFSET);
        const uint16_t minor
            = readLittleEndianU16(copy, payloadOffset + INGESTOR_PLAN_FORMAT_MINOR_OFFSET);
        const auto otherMajor = static_cast<uint16_t>(major + 1);
        writeLittleEndianU16(copy, payloadOffset + INGESTOR_PLAN_FORMAT_MAJOR_OFFSET, otherMajor);

        auto loaded = std::make_shared<Graph>();
        const auto result = detail::loadInForm(*loaded, _handle, copy, SaveForm::PLAN_ONLY);
        expectPluginRefusal(result,
                            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                            "payload format version " + std::to_string(otherMajor) + "."
                                + std::to_string(minor) + " is not readable");
        EXPECT_NE(result.err_msg.find("reads format versions " + std::to_string(major) + "."),
                  std::string::npos)
            << result.err_msg;
    };

    ASSERT_NO_FATAL_FAILURE(loadWithOneCodeObjectByteChanged(saved));
    ASSERT_NO_FATAL_FAILURE(loadWithAnotherFormatMajorVersion(saved));
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
