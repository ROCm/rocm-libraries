// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#endif
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::test;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using namespace ::testing;

namespace hipdnn_frontend
{

// Static assert checks to verify Move and Copy semantics
// Ensure INode cannot be copied, only moved
static_assert(!std::is_copy_constructible_v<INode>, "INode must not be copy constructible");
static_assert(!std::is_copy_assignable_v<INode>, "INode must not be copy assignable");

// Ensure Graph cannot be copied, only moved (inherits deleted copy from INode or explicitly deleted)
static_assert(!std::is_copy_constructible_v<Graph>, "Graph must not be copy constructible");
static_assert(!std::is_copy_assignable_v<Graph>, "Graph must not be copy assignable");

// Optional: Explicitly verify that move semantics ARE available
static_assert(std::is_move_constructible_v<Graph>, "Graph must be move constructible");
static_assert(std::is_move_assignable_v<Graph>, "Graph must be move assignable");

// Utility class to access private/protected members of Graph for testing purposes
class GraphTestUtils : public Graph
{
public:
    GraphTestUtils() = default;

    using Graph::build_operation_graph_via_descriptors;
    using Graph::CompiledPlan;
    using Graph::compilePlanFromSpec;

    std::vector<std::shared_ptr<INode>>& getPrivateGraphSubnodes()
    {
        return _sub_nodes;
    }

    size_t getCompiledPlansCount() const
    {
        return _compiledPlans.size();
    }

    size_t getActivePlanIndex() const
    {
        return _activePlanIndex;
    }

    detail::ScopedHipdnnBackendDescriptor* getActiveEngineConfigPtr()
    {
        return activeEngineConfigPtr();
    }

    detail::ScopedHipdnnBackendDescriptor* getActiveExecutionPlanPtr()
    {
        return activeExecutionPlanPtr();
    }

    /// Inject a dummy entry into _compiledPlans (simulates create_execution_plans() having run).
    void injectDummyCompiledPlan()
    {
        CompiledPlan plan;
        plan.engineId = 0;
        _compiledPlans.push_back(std::move(plan));
    }

    /// Inject a dummy entry into _planSpecs (simulates add_engine_*() having run).
    void injectDummyPlanSpec()
    {
        // Fully-qualified to work around clang-tidy-20 name resolution difference
        ::hipdnn_frontend::autotune::detail::PlanSpec spec;
        spec.engineId = 0;
        _planSpecs.push_back(std::move(spec));
    }

    size_t getPlanSpecsCount() const
    {
        return _planSpecs.size();
    }

    /// Inject a plan spec with specific engine ID and workspace size.
    void injectPlanSpec(int64_t engineId, int64_t workspaceSize)
    {
        // Fully-qualified to work around clang-tidy-20 name resolution difference
        ::hipdnn_frontend::autotune::detail::PlanSpec spec;
        spec.engineId = engineId;
        spec.workspaceSize = workspaceSize;
        _planSpecs.push_back(std::move(spec));
    }

    /// Inject a plan spec with an explicit supportsExhaustive flag, so the
    /// EXHAUSTIVE priming loop runs (and can fail) for this engine.
    void injectPlanSpec(int64_t engineId, int64_t workspaceSize, bool supportsExhaustive)
    {
        ::hipdnn_frontend::autotune::detail::PlanSpec spec;
        spec.engineId = engineId;
        spec.workspaceSize = workspaceSize;
        spec.supportsExhaustive = supportsExhaustive;
        _planSpecs.push_back(std::move(spec));
    }

    /// Inject a compiled plan with specific engine ID and workspace size.
    void injectCompiledPlan(int64_t engineId, int64_t workspaceSize)
    {
        CompiledPlan plan;
        plan.engineId = engineId;
        plan.workspaceSize = workspaceSize;
        _compiledPlans.push_back(std::move(plan));
    }

    /// Inject a compiled plan with valid descriptors (backed by a fake raw
    /// pointer) and an explicit barred flag. Used to drive the compiled-plan
    /// autotune path past the descriptor-validity check without running the
    /// full create_execution_plans()/build_plans() mock sequence.
    void injectValidCompiledPlan(int64_t engineId, int64_t workspaceSize, bool barred)
    {
        CompiledPlan plan;
        plan.engineId = engineId;
        plan.workspaceSize = workspaceSize;
        plan.barred = barred;
        // Distinct non-null sentinel handles per engine; never dereferenced, only
        // used to make the descriptors test as "valid" past the validity check.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto engineConfigHandle = reinterpret_cast<hipdnnBackendDescriptor_t>(0xE0 + engineId);
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto executionPlanHandle = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF0 + engineId);
        plan.engineConfigDesc
            = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(engineConfigHandle);
        plan.executionPlanDesc
            = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(executionPlanHandle);
        _compiledPlans.push_back(std::move(plan));
    }

    /// Get the engine IDs from all plan specs (for assertions).
    std::vector<int64_t> getPlanSpecEngineIds() const
    {
        std::vector<int64_t> ids;
        ids.reserve(_planSpecs.size());
        for(const auto& s : _planSpecs)
        {
            ids.push_back(s.engineId);
        }
        return ids;
    }

    /// Get the engine IDs from all compiled plans (for assertions).
    std::vector<int64_t> getCompiledPlanEngineIds() const
    {
        std::vector<int64_t> ids;
        ids.reserve(_compiledPlans.size());
        for(const auto& p : _compiledPlans)
        {
            ids.push_back(p.engineId);
        }
        return ids;
    }

    /// Get barred flags for all compiled plans (for assertions).
    std::vector<bool> getCompiledPlanBarredFlags() const
    {
        std::vector<bool> flags;
        flags.reserve(_compiledPlans.size());
        for(const auto& p : _compiledPlans)
        {
            flags.push_back(p.barred);
        }
        return flags;
    }

    /// Get the stored workspace threshold (for assertions).
    int64_t getMaxWorkspaceAllowed() const
    {
        return _maxWorkspaceAllowed;
    }

    /// Get the stored barred engine IDs (for assertions).
    std::unordered_set<int64_t> getBarredEngineIds() const
    {
        return _barredEngineIds;
    }

    /// Get the supportsExhaustive flag stored on each plan spec (for assertions).
    std::vector<bool> getPlanSpecSupportsExhaustive() const
    {
        std::vector<bool> flags;
        flags.reserve(_planSpecs.size());
        for(const auto& s : _planSpecs)
        {
            flags.push_back(s.supportsExhaustive);
        }
        return flags;
    }

    /// Get the knob IDs stored on each plan spec (for assertions).
    std::vector<std::vector<std::string>> getPlanSpecKnobIds() const
    {
        std::vector<std::vector<std::string>> out;
        out.reserve(_planSpecs.size());
        for(const auto& s : _planSpecs)
        {
            std::vector<std::string> ids;
            ids.reserve(s.knobSettings.size());
            for(const auto& k : s.knobSettings)
            {
                ids.push_back(k.knobId());
            }
            out.push_back(std::move(ids));
        }
        return out;
    }

    /// Get the full (knobId, value) settings stored on each plan spec, as a
    /// set of pairs per spec (for exact-set-equality assertions against the
    /// production-produced plan-spec knob vector).
    std::vector<std::set<std::pair<std::string, KnobValueVariant>>> getPlanSpecKnobSettings() const
    {
        std::vector<std::set<std::pair<std::string, KnobValueVariant>>> out;
        out.reserve(_planSpecs.size());
        for(const auto& s : _planSpecs)
        {
            std::set<std::pair<std::string, KnobValueVariant>> settings;
            for(const auto& k : s.knobSettings)
            {
                settings.insert({k.knobId(), k.value()});
            }
            out.push_back(std::move(settings));
        }
        return out;
    }

    /// Test shim for the rankAndSelectWinner free function (drives the real
    /// production ranking/winner-selection code).
    static Error callRankAndSelectWinner(std::vector<AutotuneResult>& results,
                                         const AutotuneConfig& config,
                                         size_t& activePlanIndex)
    {
        return hipdnn_frontend::autotune::detail::rankAndSelectWinner(
            results, config, activePlanIndex);
    }

    // True when an execution plan descriptor is attached (created) to the active
    // compiled plan. The plan may not yet be finalized; use
    // isExecutionPlanFinalized() for that.
    bool hasExecutionPlan() const
    {
        const auto* activePlan = activeExecutionPlanPtr();
        return activePlan != nullptr && activePlan->valid();
    }

    // True only once the attached execution plan has been finalized (build_plans()/build()).
    bool isExecutionPlanFinalized() const
    {
        return _executionPlanFinalized;
    }

    // Test seams for the capability-gated serialize path. Setting the captured
    // engine id makes serialize() query that engine's behavior note (combo path
    // when it reports support); clearing it forces the graph-only fallback (plan
    // treated as not serializable, no query issued).
    void setSelectedEngineId(int64_t id)
    {
        _selectedEngineId = id;
    }
    void clearSelectedEngineId()
    {
        _selectedEngineId.reset();
    }

    std::optional<int64_t> selectedEngineIdForTest() const
    {
        return _selectedEngineId;
    }
};
}

// Creates a minimal batchnorm inference graph for testing. Used both by TestGraph
// fixture methods and by standalone helper functions.
static std::shared_ptr<TensorAttributes> createBasicBatchnormGraph(Graph& graph)
{
    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    return graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);
}

class TestGraph : public ::testing::Test
{
protected:
    // NiceMock suppresses warnings for backend calls that are not the focus
    // of individual tests (e.g., descriptor creation/finalization setup calls).
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    hipdnnHandle_t _handle;
    std::array<char, 256> _fakeDescs{};
    size_t _nextFakeDescIdx = 0;

    void SetUp() override
    {
        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        detail::IHipdnnBackend::setInstance(_mockBackend);
        _handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);

        // Default: all descriptor creation succeeds with unique fake pointers
        _nextFakeDescIdx = 0;
        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
                return HIPDNN_STATUS_SUCCESS;
            });

        // Default: all setAttribute succeeds
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

        // Default: all finalize succeeds
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

        // Default: all destroy succeeds
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestGraph, ValidateUnsetNodeComputeTypeUnsetGraphComputeType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::NOT_SET)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, GetBehaviorNotesForEngineFailsBeforeGraphFinalized)
{
    const Graph graph;
    std::vector<BehaviorNote> notes = {BehaviorNote::RUNTIME_COMPILATION};

    auto result = graph.get_behavior_notes_for_engine(0, notes);

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, GetBehaviorNotesForEngineReturnsNotes)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 3, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 3;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
            notes[1] = HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY;
            notes[2] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<BehaviorNote> notes = {BehaviorNote::SUPPORTS_GRAPH_CAPTURE};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(notes.size(), 3u);
    EXPECT_EQ(notes[0], BehaviorNote::RUNTIME_COMPILATION);
    EXPECT_EQ(notes[1], BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY);
    EXPECT_EQ(notes[2], BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
}

TEST_F(TestGraph, GetBehaviorNotesForEnginePropagatesCountQueryFailure)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<BehaviorNote> notes = {BehaviorNote::SUPPORTS_GRAPH_CAPTURE};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_bad());
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, GetBehaviorNotesForEnginePropagatesNoteQueryFailure)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<BehaviorNote> notes = {BehaviorNote::SUPPORTS_GRAPH_CAPTURE};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_bad());
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, GetBehaviorNotesForEnginePreservesUnknownNotes)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 3, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 3;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
            notes[1]
                = static_cast<hipdnnBackendBehaviorNote_t>(HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT + 1);
            notes[2] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<BehaviorNote> notes;
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(notes.size(), 3u);
    EXPECT_EQ(notes[0], BehaviorNote::RUNTIME_COMPILATION);
    EXPECT_EQ(notes[1], static_cast<BehaviorNote>(HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT + 1));
    EXPECT_EQ(notes[2], BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
}

TEST_F(TestGraph, GetBehaviorNotesForEngineRejectsMismatchedReturnedNoteCount)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 2, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
            notes[1] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<BehaviorNote> notes = {BehaviorNote::SUPPORTS_GRAPH_CAPTURE};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_bad());
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, GetBehaviorNotesForEngineRejectsNegativeNoteCount)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = -1;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<BehaviorNote> notes = {BehaviorNote::RUNTIME_COMPILATION};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_bad());
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, GetBehaviorNotesForEngineClearsOutputWhenNoNotes)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<BehaviorNote> notes = {BehaviorNote::SUPPORTS_GRAPH_CAPTURE};
    auto result = graph.get_behavior_notes_for_engine(7, notes);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(notes.empty());
}

TEST_F(TestGraph, SerializeCompiledPlanRejectsMissingExecutionPlan)
{
    const Graph graph;
    std::vector<uint8_t> data{1, 2, 3};

    auto result = graph.serialize_compiled_plan(data);

    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, SerializeCompiledPlanPropagatesSizeQueryFailure)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    ASSERT_TRUE(graph.deserialize_compiled_plan(_handle, serializedPlan).is_good());

    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<uint8_t> data;
    auto result = graph.serialize_compiled_plan(data);

    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, SerializeCompiledPlanRejectsZeroLengthPlan)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    ASSERT_TRUE(graph.deserialize_compiled_plan(_handle, serializedPlan).is_good());

    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t, size_t, size_t* planByteSize, uint8_t*) {
            *planByteSize = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data;
    auto result = graph.serialize_compiled_plan(data);

    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, SerializeCompiledPlanPropagatesSerializationFailure)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    ASSERT_TRUE(graph.deserialize_compiled_plan(_handle, serializedPlan).is_good());

    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t, size_t, size_t* planByteSize, uint8_t*) {
            *planByteSize = 4;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 4, _, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<uint8_t> data;
    auto result = graph.serialize_compiled_plan(data);

    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, SerializeCompiledPlanReturnsBackendBytes)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    const std::vector<uint8_t> expectedPlan{9, 8, 7};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    ASSERT_TRUE(graph.deserialize_compiled_plan(_handle, serializedPlan).is_good());

    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t, size_t, size_t* planByteSize, uint8_t*) {
            *planByteSize = 4;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(executionPlan, 4, _, _))
        .WillOnce([&expectedPlan](hipdnnBackendDescriptor_t,
                                  size_t requestedByteSize,
                                  size_t* planByteSize,
                                  uint8_t* serializedPlanBytes) {
            EXPECT_EQ(requestedByteSize, 4);
            std::copy(expectedPlan.begin(), expectedPlan.end(), serializedPlanBytes);
            *planByteSize = expectedPlan.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    auto [data, result] = graph.to_compiled_plan_binary();

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(data, expectedPlan);
}

TEST_F(TestGraph, DeserializeCompiledPlanPropagatesBackendFailure)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto result = graph.deserialize_compiled_plan(_handle, serializedPlan);

    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, DeserializeCompiledPlanClearsFrontendGraphState)
{
    GraphTestUtils graph;
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);
    const std::vector<uint8_t> serializedPlan{1, 2, 3};

    createBasicBatchnormGraph(graph);
    ASSERT_FALSE(graph.getPrivateGraphSubnodes().empty());

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });

    auto result = graph.from_compiled_plan_binary(_handle, serializedPlan);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(graph.getPrivateGraphSubnodes().empty());
}

#ifdef HIPDNN_ENABLE_SDPA
// Sets the override-shape-enabled attribute query that deserialize_compiled_plan
// issues on the restored execution plan, letting plan-only tests choose whether the
// restored plan reports override shapes as enabled. Uses ON_CALL so the sibling
// engine-id query on the same descriptor stays uninteresting under NiceMock.
static void setPlanOverrideShapeEnabledQuery(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend,
                                             hipdnnBackendDescriptor_t executionPlan,
                                             bool enabled)
{
    ON_CALL(mockBackend,
            backendGetAttribute(executionPlan,
                                HIPDNN_ATTR_EXECUTION_PLAN_IS_OVERRIDE_SHAPE_ENABLED_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                1,
                                _,
                                _))
        .WillByDefault([enabled](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t* elementCount,
                                 void* arrayOfElements) {
            *elementCount = 1;
            *static_cast<bool*>(arrayOfElements) = enabled;
            return HIPDNN_STATUS_SUCCESS;
        });
}

TEST_F(TestGraph, PlanOnlyOverrideExecuteWritesOverrideVariantPackAttributes)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, executionPlan, true);
    ASSERT_TRUE(graph.from_compiled_plan_binary(_handle, serializedPlan).is_good());

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = reinterpret_cast<void*>(0x1000);
    variantPack[2] = reinterpret_cast<void*>(0x2000);
    void* workspace = reinterpret_cast<void*>(0x3000);

    const std::vector<int64_t> overrideUids{1, 2};
    const std::vector<std::vector<int64_t>> overrideShapes{{2, 3}, {4, 5, 6}};
    const std::vector<std::vector<int64_t>> overrideStrides{{3, 1}, {30, 6, 1}};

    auto variantPackDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [variantPackDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variantPackDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                    HIPDNN_TYPE_VOID_PTR,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, _));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(overrideUids.size()),
                                    NotNull()))
        .WillOnce(Invoke([&overrideUids](hipdnnBackendDescriptor_t,
                                         hipdnnBackendAttributeName_t,
                                         hipdnnBackendAttributeType_t,
                                         int64_t count,
                                         const void* ptr) {
            EXPECT_EQ(count, static_cast<int64_t>(overrideUids.size()));
            const auto* uids = static_cast<const int64_t*>(ptr);
            EXPECT_EQ(std::vector<int64_t>(uids, uids + count), overrideUids);
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT,
                                    HIPDNN_TYPE_INT64,
                                    2,
                                    NotNull()))
        .WillOnce(Invoke([](hipdnnBackendDescriptor_t,
                            hipdnnBackendAttributeName_t,
                            hipdnnBackendAttributeType_t,
                            int64_t count,
                            const void* ptr) {
            EXPECT_EQ(count, 2);
            const auto* lengths = static_cast<const int64_t*>(ptr);
            EXPECT_EQ(std::vector<int64_t>(lengths, lengths + count), (std::vector<int64_t>{2, 3}));
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT,
                                    HIPDNN_TYPE_INT64,
                                    5,
                                    NotNull()))
        .WillOnce(Invoke([](hipdnnBackendDescriptor_t,
                            hipdnnBackendAttributeName_t,
                            hipdnnBackendAttributeType_t,
                            int64_t count,
                            const void* ptr) {
            const auto* shapes = static_cast<const int64_t*>(ptr);
            EXPECT_EQ(std::vector<int64_t>(shapes, shapes + count),
                      (std::vector<int64_t>{2, 3, 4, 5, 6}));
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT,
                                    HIPDNN_TYPE_INT64,
                                    5,
                                    NotNull()))
        .WillOnce(Invoke([](hipdnnBackendDescriptor_t,
                            hipdnnBackendAttributeName_t,
                            hipdnnBackendAttributeType_t,
                            int64_t count,
                            const void* ptr) {
            const auto* strides = static_cast<const int64_t*>(ptr);
            EXPECT_EQ(std::vector<int64_t>(strides, strides + count),
                      (std::vector<int64_t>{3, 1, 30, 6, 1}));
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend, backendFinalize(variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendExecute(_handle, executionPlan, variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto result = graph.execute(
        _handle, variantPack, workspace, overrideUids, overrideShapes, overrideStrides);

    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, PlanOnlyOverrideExecuteEmptyOverridesUsesLegacyVariantPackAttributes)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    ASSERT_TRUE(graph.from_compiled_plan_binary(_handle, serializedPlan).is_good());

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = reinterpret_cast<void*>(0x1000);
    void* workspace = reinterpret_cast<void*>(0x3000);
    const std::vector<int64_t> emptyUids;
    const std::vector<std::vector<int64_t>> emptyShapes;
    const std::vector<std::vector<int64_t>> emptyStrides;

    auto variantPackDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [variantPackDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variantPackDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend, backendSetAttribute(variantPackDesc, _, _, _, _))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_LENGTHS_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend, backendFinalize(variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendExecute(_handle, executionPlan, variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto result
        = graph.execute(_handle, variantPack, workspace, emptyUids, emptyShapes, emptyStrides);

    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, PlanOnlyOverrideExecuteRejectsStructuralValidationBeforeBackendDescriptor)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, executionPlan, true);
    ASSERT_TRUE(graph.from_compiled_plan_binary(_handle, serializedPlan).is_good());

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = reinterpret_cast<void*>(0x1000);
    const std::vector<int64_t> overrideUids{1};
    const std::vector<std::vector<int64_t>> overrideShapes{{2, 3}};
    const std::vector<std::vector<int64_t>> overrideStrides{{1}};

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .Times(0);

    auto result = graph.execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, PlanOnlyOverrideExecutePropagatesOverrideAttributeFailure)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, executionPlan, true);
    ASSERT_TRUE(graph.from_compiled_plan_binary(_handle, serializedPlan).is_good());

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = reinterpret_cast<void*>(0x1000);
    const std::vector<int64_t> overrideUids{1};
    const std::vector<std::vector<int64_t>> overrideShapes{{2, 3}};
    const std::vector<std::vector<int64_t>> overrideStrides{{3, 1}};

    auto variantPackDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [variantPackDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variantPackDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendSetAttribute(variantPackDesc, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            variantPackDesc, HIPDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES_EXT, HIPDNN_TYPE_INT64, 2, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR))
        .RetiresOnSaturation();
    EXPECT_CALL(*_mockBackend, backendFinalize(variantPackDesc)).Times(0);
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _)).Times(0);

    auto result = graph.execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);

    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestGraph, DeserializeCompiledPlanRecoversOverrideShapeEnabledFlag)
{
    const std::vector<uint8_t> serializedPlan{1, 2, 3};

    auto enabledPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_handle, _, _, _))
        .WillOnce(
            [enabledPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = enabledPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, enabledPlan, true);

    Graph enabledGraph;
    ASSERT_TRUE(enabledGraph.from_compiled_plan_binary(_handle, serializedPlan).is_good());
    EXPECT_TRUE(enabledGraph.is_override_shape_enabled());

    auto disabledPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x89ab);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_handle, _, _, _))
        .WillOnce(
            [disabledPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = disabledPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, disabledPlan, false);

    Graph disabledGraph;
    ASSERT_TRUE(disabledGraph.from_compiled_plan_binary(_handle, serializedPlan).is_good());
    EXPECT_FALSE(disabledGraph.is_override_shape_enabled());
}

TEST_F(TestGraph, PlanOnlyOverrideExecuteRejectedWhenOverrideShapeDisabled)
{
    Graph graph;
    const std::vector<uint8_t> serializedPlan{1, 2, 3};
    auto executionPlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4567);

    EXPECT_CALL(*_mockBackend,
                backendCreateAndDeserializeExecutionPlanExt(
                    _handle, _, serializedPlan.data(), serializedPlan.size()))
        .WillOnce(
            [executionPlan](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = executionPlan;
                return HIPDNN_STATUS_SUCCESS;
            });
    setPlanOverrideShapeEnabledQuery(*_mockBackend, executionPlan, false);
    ASSERT_TRUE(graph.from_compiled_plan_binary(_handle, serializedPlan).is_good());

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = reinterpret_cast<void*>(0x1000);
    const std::vector<int64_t> overrideUids{1};
    const std::vector<std::vector<int64_t>> overrideShapes{{2, 3}};
    const std::vector<std::vector<int64_t>> overrideStrides{{3, 1}};

    // Fail-fast must reject before constructing a variant pack or dispatching.
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _)).Times(0);

    auto result = graph.execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.get_message();
}
#endif

TEST_F(TestGraph, ValidateUnsetNodeComputeTypeSetGraphComputeType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeUnsetGraphIoType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::NOT_SET);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeUnsetGraphIntermediateType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::NOT_SET)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);
    in0->set_is_virtual(true);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeSetGraphDataTypes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);
    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::NOT_SET)
        .set_is_virtual(true);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::ADD);

    auto out0 = graph.pointwise(in0, in1, attributes);

    auto validationResult = graph.validate();

    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, SetAndGetAttributes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    EXPECT_EQ(graph.get_name(), "TestGraph");
    EXPECT_EQ(graph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(graph.get_io_data_type(), DataType::FLOAT);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    x->set_name("BatchnormNode::X");

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("BatchnormNode::SCALE").set_dim(derivedDims).set_stride(derivedStrides);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("BatchnormNode::BIAS").set_dim(derivedDims).set_stride(derivedStrides);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("BatchnormNode::EPSILON").set_value(0.001f);

    BatchnormAttributes attributes;
    attributes.set_name("BatchnormNode");
    attributes.set_epsilon(epsilon);

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    EXPECT_EQ(mean->get_name(), "BatchnormNode::MEAN");
    EXPECT_TRUE(mean->get_is_virtual());

    EXPECT_EQ(invVariance->get_name(), "BatchnormNode::INV_VARIANCE");
    EXPECT_TRUE(invVariance->get_is_virtual());

    EXPECT_FALSE(nextRunningMean);
    EXPECT_FALSE(nextRunningVariance);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormBackwardNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto dy = std::make_shared<TensorAttributes>();
    auto x = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();

    dy->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    scale->set_dim(derivedDims).set_stride(derivedStrides).set_data_type(DataType::FLOAT);

    BatchnormBackwardAttributes attributes;
    attributes.set_name("BatchnormBackwardNode");

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, attributes);

    EXPECT_EQ(dx->get_name(), "BatchnormBackwardNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    EXPECT_EQ(dscale->get_name(), "BatchnormBackwardNode::DSCALE");
    EXPECT_TRUE(dscale->get_is_virtual());

    EXPECT_EQ(dbias->get_name(), "BatchnormBackwardNode::DBIAS");
    EXPECT_TRUE(dbias->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormInferenceNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1});

    mean->set_dim(derivedDims).set_stride(derivedStrides);
    invVariance->set_dim(derivedDims).set_stride(derivedStrides);
    scale->set_dim(derivedDims).set_stride(derivedStrides);
    bias->set_dim(derivedDims).set_stride(derivedStrides);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormInferenceNodeVarianceExtCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1});

    auto variance = std::make_shared<TensorAttributes>();
    variance->set_dim({1, 2, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1});

    mean->set_dim(derivedDims).set_stride(derivedStrides);
    variance->set_dim(derivedDims).set_stride(derivedStrides);
    scale->set_dim(derivedDims).set_stride(derivedStrides);
    bias->set_dim(derivedDims).set_stride(derivedStrides);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5);

    BatchnormInferenceAttributesVarianceExt attributes;
    attributes.set_name("BatchnormNodeVariance");

    auto y = graph.batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNodeVariance::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationSingleInput)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationTwoInputs)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    auto in1 = std::make_shared<TensorAttributes>();

    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::ADD);

    auto out0 = graph.pointwise(in0, in1, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationThreeInputs)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    auto in1 = std::make_shared<TensorAttributes>();
    auto in2 = std::make_shared<TensorAttributes>();

    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in2->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::BINARY_SELECT);

    auto out0 = graph.pointwise(in0, in1, in2, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ConvolutionFwdNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvFpropAttributes attributes;
    attributes.set_name("ConvolutionFpropNode");
    attributes.set_pre_padding({1, 1});
    attributes.set_post_padding({1, 1});
    attributes.set_stride({1, 1});
    attributes.set_dilation({1, 1});

    auto y = graph.conv_fprop(x, w, attributes);

    EXPECT_EQ(y->get_name(), "ConvolutionFpropNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ConvolutionDgradNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvDgradAttributes attributes;
    attributes.set_name("ConvolutionDgradNode");
    attributes.set_pre_padding({1, 1});
    attributes.set_post_padding({1, 1});
    attributes.set_stride({1, 1});
    attributes.set_dilation({1, 1});

    auto dx = graph.conv_dgrad(dy, w, attributes);

    EXPECT_EQ(dx->get_name(), "ConvolutionDgradNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, MatmulNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5}).set_stride({5, 1}).set_data_type(DataType::FLOAT);

    MatmulAttributes attributes;
    attributes.set_name("MatmulNode");

    auto c = graph.matmul(a, b, attributes);

    EXPECT_EQ(c->get_name(), "MatmulNode::C");
    EXPECT_TRUE(c->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, LayernormNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 6, 4}).set_stride({24, 4, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5f);

    LayernormAttributes attributes;
    attributes.set_name("LayerNormNode");
    attributes.set_forward_phase(NormFwdPhase::INFERENCE);
    attributes.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "LayerNormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    // In inference mode, mean and inv_variance are not created
    EXPECT_EQ(mean, nullptr);
    EXPECT_EQ(invVariance, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("RMSNormNode");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);

    EXPECT_EQ(y->get_name(), "RMSNormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    EXPECT_EQ(invRms->get_name(), "RMSNormNode::INV_RMS");
    EXPECT_TRUE(invRms->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormNodeCreationWithBias)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("RMSNormWithBias");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_bias(bias);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);

    EXPECT_EQ(y->get_name(), "RMSNormWithBias::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormBackwardNodeCreation)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);

    // Create attributes (default: no dbias)
    RMSNormBackwardAttributes attributes;
    attributes.set_name("RMSNormBackwardNode");

    // Call graph method
    auto [dx, dscale, dbias] = graph.rmsnorm_backward(dy, x, scale, invRms, attributes);

    EXPECT_EQ(dx->get_name(), "RMSNormBackwardNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    EXPECT_EQ(dscale->get_name(), "RMSNormBackwardNode::DSCALE");
    EXPECT_TRUE(dscale->get_is_virtual());

    // dbias is not computed by default
    EXPECT_EQ(dbias, nullptr);

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormBackwardNodeCreationWithDbias)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);

    RMSNormBackwardAttributes attributes;
    attributes.set_name("RMSNormBackwardNode").set_compute_dbias(true);

    auto [dx, dscale, dbias] = graph.rmsnorm_backward(dy, x, scale, invRms, attributes);

    ASSERT_NE(dbias, nullptr);
    EXPECT_EQ(dbias->get_name(), "RMSNormBackwardNode::DBIAS");
    EXPECT_TRUE(dbias->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, LayernormNodeCreationTrainingPhase)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 6, 4}).set_stride({24, 4, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5f);

    LayernormAttributes attributes;
    attributes.set_name("LayernormNodeTraining");
    attributes.set_forward_phase(NormFwdPhase::TRAINING);
    attributes.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "LayernormNodeTraining::Y");
    EXPECT_TRUE(y->get_is_virtual());
    // In training mode, mean and inv_variance should be created
    ASSERT_NE(mean, nullptr);
    EXPECT_EQ(mean->get_name(), "LayernormNodeTraining::MEAN");
    ASSERT_NE(invVariance, nullptr);
    EXPECT_EQ(invVariance->get_name(), "LayernormNodeTraining::INV_VARIANCE");

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ResampleReturnsNullIndexWhenNotRequested)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 4, 4}).set_stride({48, 16, 4, 1}).set_data_type(DataType::FLOAT);

    ResampleFwdAttributes attributes;
    attributes.set_name("ResampleNode");
    attributes.set_resample_mode(ResampleMode::MAXPOOL);
    attributes.set_padding_mode(PaddingMode::ZERO_PAD);
    attributes.set_pre_padding({0, 0});
    attributes.set_post_padding({0, 0});
    attributes.set_stride({2, 2});
    attributes.set_window({2, 2});

    auto [y, index] = graph.resample(x, attributes);

    EXPECT_EQ(y->get_name(), "ResampleNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    EXPECT_EQ(index, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ResampleReturnsIndexWhenRequested)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 4, 4}).set_stride({48, 16, 4, 1}).set_data_type(DataType::FLOAT);

    ResampleFwdAttributes attributes;
    attributes.set_name("ResampleNodeWithIndex");
    attributes.set_resample_mode(ResampleMode::MAXPOOL);
    attributes.set_padding_mode(PaddingMode::ZERO_PAD);
    attributes.set_pre_padding({0, 0});
    attributes.set_post_padding({0, 0});
    attributes.set_stride({2, 2});
    attributes.set_window({2, 2});
    attributes.set_generate_index(true);

    auto [y, index] = graph.resample(x, attributes);

    EXPECT_EQ(y->get_name(), "ResampleNodeWithIndex::Y");
    EXPECT_TRUE(y->get_is_virtual());
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->get_name(), "ResampleNodeWithIndex::Index");
    EXPECT_TRUE(index->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(index->get_dim(), y->get_dim());
    EXPECT_EQ(index->get_stride(), y->get_stride());
}

// Test graph.tensor()
TEST_F(TestGraph, TensorGraphAttributes)
{
    auto tensor = Graph::tensor(TensorAttributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_stride({5, 6, 7, 8})
                                    .set_data_type(DataType::FLOAT)
                                    .set_is_virtual(false)
                                    .set_dim({1, 2, 3, 4}));

    EXPECT_EQ(tensor->get_data_type(), DataType::FLOAT);
    EXPECT_FALSE(tensor->get_is_virtual());
    EXPECT_EQ(tensor->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensor->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
}

// Test graph.tensorLike()
TEST_F(TestGraph, TensorLikeGraphAttributes)
{
    auto tensor = Graph::tensor(TensorAttributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_dim({1, 2, 3, 4})
                                    .set_stride({5, 6, 7, 8})
                                    .set_is_virtual(false)
                                    .set_data_type(DataType::FLOAT));

    auto tensorLike = Graph::tensor_like(tensor, "TensorLike");

    EXPECT_EQ(tensorLike->get_data_type(), DataType::FLOAT);
    EXPECT_FALSE(tensorLike->get_is_virtual());
    EXPECT_EQ(tensorLike->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensorLike->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensorLike->get_name(), "TensorLike");
    EXPECT_NE(tensorLike->get_uid(), 100);

    EXPECT_NE(tensorLike, tensor);

    auto tensorLikeNoName = Graph::tensor_like(tensorLike);
    EXPECT_EQ(tensorLikeNoName->get_name(), "");

    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
    EXPECT_NE(tensor->get_uid(), tensorLikeNoName->get_uid());
}

TEST_F(TestGraph, WillCorrectlyBuildOperationGraphDescriptor)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path setAttribute calls (ops, data types, name, etc.)
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());

    // Verify the handle is set on the operation graph descriptor
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce([this](hipdnnBackendDescriptor_t,
                         hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t,
                         int64_t,
                         const void* arrayOfElements) {
            hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);
            EXPECT_EQ(handle, this->_handle);
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result.is_good());
}

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsWhenNodeFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    graph.set_name("FailTest").set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_FPROP_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_FPROP_TENSOR_X_DIMS))
        .set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_FPROP_TENSOR_W_UID)
        .set_name("W")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_FPROP_TENSOR_W_DIMS))
        .set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_x(x);
    convAttrs.set_w(w);
    convAttrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_FPROP_CONV_DILATION));

    graph.conv_fprop(x, w, convAttrs);

    // All descriptor creation fails
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    // validate() now runs inside build_operation_graph_via_descriptors(), catching
    // the missing output tensor before descriptor creation is attempted
    EXPECT_EQ(result.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsWhenGraphCreateFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    graph.set_name("GraphCreateFail")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_FPROP_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_FPROP_TENSOR_X_DIMS))
        .set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_FPROP_TENSOR_W_UID)
        .set_name("W")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_FPROP_TENSOR_W_DIMS))
        .set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_x(x);
    convAttrs.set_w(w);
    convAttrs.set_compute_data_type(DataType::FLOAT);
    convAttrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_FPROP_CONV_DILATION));

    graph.conv_fprop(x, w, convAttrs);

    // Validate graph to propagate graph-level attributes (e.g. io_data_type)
    // to node/tensor attributes, matching the real build() flow
    ASSERT_TRUE(graph.validate().is_good());

    // Tensor/operation descriptors succeed, but graph descriptor creation fails
    EXPECT_CALL(*_mockBackend,
                backendCreateDescriptor(Ne(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR), _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsOnEmptyGraph)
{
    GraphTestUtils graph;
    graph.set_name("EmptyGraph").set_compute_data_type(DataType::FLOAT);

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, CreatingExecutionPlansFailsWithNoGraph)
{
    Graph graph;

    auto result = graph.create_execution_plans({HeuristicMode::FALLBACK});
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(),
              "Graph has not been built, build the operation graph first. Cannot create "
              "execution plan.");
}

TEST_F(TestGraph, CanSuccessfullyCreateExecutionPlans)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    std::vector<hipdnnBackendHeurMode_t> backendModes;
    backendModes.reserve(heurModes.size());
    for(const auto& mode : heurModes)
    {
        backendModes.push_back(toBackendType(mode));
    }
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    graph.build_operation_graph(_handle);

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce([&backendModes](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t count,
                                  const void* arrayOfElements) {
            EXPECT_EQ(count, static_cast<int64_t>(backendModes.size()));
            auto modesPtr = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
            for(size_t i = 0; i < backendModes.size(); ++i)
            {
                EXPECT_EQ(modesPtr[i], backendModes[i]);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    // Set up the mock to handle multiple calls with different arguments using .WillOnce()/.WillRepeatedly()
    // First call: elementCount query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto execPlanResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(execPlanResult.is_good());
}

TEST_F(TestGraph, PreferredEngineIdSelectsSpecificConfig)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    std::vector<hipdnnBackendHeurMode_t> backendModes;
    backendModes.reserve(heurModes.size());
    for(const auto& mode : heurModes)
    {
        backendModes.push_back(toBackendType(mode));
    }
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Set preferred engine ID
    const int64_t preferredEngineId = 42;
    graph.set_preferred_engine_id_ext(preferredEngineId);

    graph.build_operation_graph(_handle);

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce([&backendModes](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t count,
                                  const void* arrayOfElements) {
            EXPECT_EQ(count, static_cast<int64_t>(backendModes.size()));
            auto modesPtr = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
            for(size_t i = 0; i < backendModes.size(); ++i)
            {
                EXPECT_EQ(modesPtr[i], backendModes[i]);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    // First call: elementCount query - return 2 configs available
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2346);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3346);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2));

    // Get engine from second config (ID = 42 - our preferred one)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from second engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = preferredEngineId;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    auto executionPlanDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9877);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc1](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&executionPlanDesc2](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto execPlanResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(execPlanResult.is_good());
}

TEST_F(TestGraph, PreferredEngineIdFallsBackToTopConfig)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    std::vector<hipdnnBackendHeurMode_t> backendModes;
    backendModes.reserve(heurModes.size());
    for(const auto& mode : heurModes)
    {
        backendModes.push_back(toBackendType(mode));
    }
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Set preferred engine ID that doesn't exist
    const int64_t preferredEngineId = 999;
    graph.set_preferred_engine_id_ext(preferredEngineId);

    graph.build_operation_graph(_handle);

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce([&backendModes](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t count,
                                  const void* arrayOfElements) {
            EXPECT_EQ(count, static_cast<int64_t>(backendModes.size()));
            auto modesPtr = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
            for(size_t i = 0; i < backendModes.size(); ++i)
            {
                EXPECT_EQ(modesPtr[i], backendModes[i]);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    // First call: elementCount query - return 2 configs available
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2346);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3346);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine (neither will match preferred ID 999)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2));

    // Get engine from second config (ID = 42)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from second engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 42;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    auto executionPlanDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9877);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc1](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&executionPlanDesc2](hipdnnBackendDescriptorType_t,
                                        hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto execPlanResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(execPlanResult.is_good());
}

TEST_F(TestGraph, CheckSupportFailsIfNoExecutionPlanCreated)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    graph.build_operation_graph(_handle);

    auto result = graph.check_support();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(), "Execution plan descriptor is not created or invalid.");
}

TEST_F(TestGraph, CheckSupportSucceedsWhenExecutionPlanCreated)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    graph.create_execution_plans(heurModes);

    auto result = graph.check_support();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, ExecutionPlanisFinalizedAfterBuildPlans)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            if(elementCount != nullptr)
            {
                *elementCount = 1;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(executionPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto result = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(result.is_good());

    result = graph.build_plans();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, GetExecutionPlanEngineIdFailsWithNoPlan)
{
    const Graph graph;
    int64_t engineId = -1;
    auto result = graph.get_execution_plan_engine_id(engineId);
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(engineId, -1);
}

TEST_F(TestGraph, GetExecutionPlanEngineIdReturnsSelectedEngine)
{
    // get_execution_plan_engine_id returns the cached engine selected for the
    // plan (_selectedEngineId), set when the engine config is built.
    GraphTestUtils graph;
    graph.setSelectedEngineId(4242);

    int64_t engineId = -1;
    auto result = graph.get_execution_plan_engine_id(engineId);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(engineId, 4242);
}

TEST_F(TestGraph, WorkspaceSizeIsRetrievedFromExecutionPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    graph.create_execution_plans(heurModes);

    const int64_t workspaceSize = 123454;
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = workspaceSize;
            return HIPDNN_STATUS_SUCCESS;
        });

    int64_t workspaceSizeResult = 0;
    auto result = graph.get_workspace_size(workspaceSizeResult);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(workspaceSizeResult, workspaceSize);
}

TEST_F(TestGraph, ExecutePacksVariantPackAndPassesTheCorrectArguments)
{
    ::testing::FLAGS_gmock_verbose = "error";
    using ::testing::_;
    using ::testing::Invoke;
    using ::testing::NotNull;
    using ::testing::Return;

    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT).set_io_data_type(DataType::FLOAT);

    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(42)
        .set_name("InputTensor")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);
    auto outTensor = graph.pointwise(tensor, pointwiseAttributes);
    outTensor->set_data_type(DataType::FLOAT).set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8});
    auto valResult = graph.validate();
    ASSERT_TRUE(valResult.is_good()) << valResult.get_message();

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // create_execution_plans mocks
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2000);
    auto engineCfgDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3000);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4000);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce([&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = heurDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineCfgDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = engineCfgDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineCfgDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&execPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = execPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // build_plans mocks
    EXPECT_CALL(*_mockBackend, backendFinalize(engineCfgDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(execPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // get_workspace_size mock (called during build_plans and get_workspace_size)
    const int64_t expectedWorkspaceSize = 12345;
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(execPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    nullptr,
                                    _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* ptr) {
            *reinterpret_cast<int64_t*>(ptr) = expectedWorkspaceSize;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Prepare variant pack and workspace for execute
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_uid(42);

    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_uid(22);

    auto tensor3 = std::make_shared<TensorAttributes>();
    tensor3->set_uid(33);

    auto tensor4 = std::make_shared<TensorAttributes>();
    tensor4->set_uid(1);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[tensor1->get_uid()] = reinterpret_cast<void*>(0xDEADBEEF);
    variantPack[tensor2->get_uid()] = reinterpret_cast<void*>(0xBEEFBEEF);
    variantPack[tensor3->get_uid()] = reinterpret_cast<void*>(0xBEEFDEAD);
    variantPack[tensor4->get_uid()] = reinterpret_cast<void*>(0xDEADBEE);

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> variantPackForExec;
    variantPackForExec[tensor1] = reinterpret_cast<void*>(0xDEADBEEF);
    variantPackForExec[tensor2] = reinterpret_cast<void*>(0xBEEFBEEF);
    variantPackForExec[tensor3] = reinterpret_cast<void*>(0xBEEFDEAD);
    variantPackForExec[tensor4] = reinterpret_cast<void*>(0xDEADBEE);

    void* workspace = reinterpret_cast<void*>(0xCAFEBABE);

    auto variantPackDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [&variantPackDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variantPackDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                    HIPDNN_TYPE_VOID_PTR,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()))
        .WillOnce(Invoke([variantPack](hipdnnBackendDescriptor_t,
                                       hipdnnBackendAttributeName_t,
                                       hipdnnBackendAttributeType_t,
                                       int64_t count,
                                       const void* ptr) {
            EXPECT_EQ(count, 4);
            auto dataPtrs = static_cast<void* const*>(ptr);
            for(int i = 0; i < 4; i++)
            {
                auto targetValue = dataPtrs[i];
                auto it = std::find_if(
                    variantPack.begin(), variantPack.end(), [&targetValue](const auto& pair) {
                        return pair.second == targetValue;
                    });
                EXPECT_TRUE(it != variantPack.end());
            }

            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()))
        .WillOnce(Invoke([variantPack](hipdnnBackendDescriptor_t,
                                       hipdnnBackendAttributeName_t,
                                       hipdnnBackendAttributeType_t,
                                       int64_t count,
                                       const void* ptr) {
            EXPECT_EQ(count, 4);
            auto keys = static_cast<const int64_t*>(ptr);
            for(int i = 0; i < 4; i++)
            {
                EXPECT_TRUE(variantPack.find(keys[i]) != variantPack.end());
            }
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                    HIPDNN_TYPE_VOID_PTR,
                                    1,
                                    NotNull()))
        .WillOnce(Invoke([workspace](hipdnnBackendDescriptor_t,
                                     hipdnnBackendAttributeName_t,
                                     hipdnnBackendAttributeType_t,
                                     int64_t count,
                                     const void* ptr) {
            EXPECT_EQ(count, 1);
            auto workspacePtr = *static_cast<void* const*>(ptr);
            EXPECT_EQ(workspacePtr, workspace);
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend, backendFinalize(variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendExecute(_handle, execPlanDesc, variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Run the full sequence
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto planResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(planResult.is_good());

    auto supportResult = graph.check_support();
    EXPECT_TRUE(supportResult.is_good());

    auto buildPlansResult = graph.build_plans();
    EXPECT_TRUE(buildPlansResult.is_good());

    int64_t workspaceSize = 0;
    auto wsResult = graph.get_workspace_size(workspaceSize);
    EXPECT_TRUE(wsResult.is_good());
    EXPECT_EQ(workspaceSize, expectedWorkspaceSize);

    auto execResult = graph.execute(_handle, variantPackForExec, workspace);
    EXPECT_TRUE(execResult.is_good());
}

TEST_F(TestGraph, TopologicalSortSucceedsOnNormalGraph)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    EXPECT_TRUE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, TopologicalSortFailsWithOrphanedNode)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    // Connected nodes
    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Orphaned node (not connected to the main graph)
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x2->set_uid(2);
    BatchnormInferenceAttributes attributes3;
    attributes3.set_name("BatchnormNode3");
    auto y3 = graph.batchnorm_inference(x2, mean, invVariance, scale, bias, attributes3);

    EXPECT_FALSE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, TopologicalSortFailsOnCircularDependency)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Introduce a cycle: set x of the first node to y2
    auto& subNodes = graph.getPrivateGraphSubnodes();
    auto* batchNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(batchNode, nullptr);
    batchNode->attributes.set_x(y2);

    EXPECT_FALSE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, ValidateSortsNodesTopologically)
{
    GraphTestUtils graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    // Node 0: batchnorm1
    BatchnormInferenceAttributes bnAttrs1;
    bnAttrs1.set_name("batchnorm1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs1);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(x->get_dim());
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    // Node 1: pointwise1 (depends on batchnorm1)
    PointwiseAttributes pwAttrs1;
    pwAttrs1.set_name("pointwise1");
    pwAttrs1.set_mode(PointwiseMode::RELU_FWD);
    auto out1 = graph.pointwise(y1, pwAttrs1);

    // Node 2: pointwise2 (depends on batchnorm1)
    PointwiseAttributes pwAttrs2;
    pwAttrs2.set_name("pointwise2");
    pwAttrs2.set_mode(PointwiseMode::RELU_FWD);
    auto out2 = graph.pointwise(y1, pwAttrs2);

    //Node 3 Create a combined input by using ADD pointwise
    PointwiseAttributes pwAdd;
    pwAdd.set_name("add");
    pwAdd.set_mode(PointwiseMode::ADD);
    auto combined = graph.pointwise(out1, out2, pwAdd);

    // Node 4: batchnorm2 (depends on both pointwise1 and pointwise2)
    BatchnormInferenceAttributes bnAttrs2;
    bnAttrs2.set_name("batchnorm2");
    auto y2 = graph.batchnorm_inference(combined, mean, invVariance, scale, bias, bnAttrs2);

    // Node 5: pointwise3 (final node)
    PointwiseAttributes pwAttrs3;
    pwAttrs3.set_name("pointwise3");
    pwAttrs3.set_mode(PointwiseMode::RELU_FWD);
    auto out3 = graph.pointwise(y2, pwAttrs3);

    auto sortedSubnodesDueToGraphConstructionOrderCopy = graph.getPrivateGraphSubnodes();

    auto& subNodes = graph.getPrivateGraphSubnodes();
    // Deterministically scramble the node order.
    // std::shuffle was flaky — it could randomly produce the original order.
    std::swap(subNodes.front(), subNodes.back());
    std::swap(subNodes[1], subNodes[subNodes.size() - 2]);

    // Verify the nodes are no longer in the original order.
    bool notSortedAnymore = false;
    for(size_t i = 0; i < subNodes.size(); i++)
    {
        if(sortedSubnodesDueToGraphConstructionOrderCopy[i] != subNodes[i])
        {
            notSortedAnymore = true;
            break;
        }
    }
    EXPECT_TRUE(notSortedAnymore);

    auto result = graph.validate();
    EXPECT_TRUE(result.is_good()) << result.get_message();

    ASSERT_EQ(subNodes.size(), sortedSubnodesDueToGraphConstructionOrderCopy.size());
    EXPECT_EQ(subNodes[0], sortedSubnodesDueToGraphConstructionOrderCopy[0]);

    //due to randomiztion, its possible that these nodes swap positions because the graph is valid
    //regardless of the order of these two nodes because the graph has a diamond shape.
    if(subNodes[1] == sortedSubnodesDueToGraphConstructionOrderCopy[1])
    {
        EXPECT_EQ(subNodes[1], sortedSubnodesDueToGraphConstructionOrderCopy[1]);
        EXPECT_EQ(subNodes[2], sortedSubnodesDueToGraphConstructionOrderCopy[2]);
    }
    else
    {
        EXPECT_EQ(subNodes[1], sortedSubnodesDueToGraphConstructionOrderCopy[2]);
        EXPECT_EQ(subNodes[2], sortedSubnodesDueToGraphConstructionOrderCopy[1]);
    }

    EXPECT_EQ(subNodes[3], sortedSubnodesDueToGraphConstructionOrderCopy[3]);
    EXPECT_EQ(subNodes[4], sortedSubnodesDueToGraphConstructionOrderCopy[4]);
    EXPECT_EQ(subNodes[5], sortedSubnodesDueToGraphConstructionOrderCopy[5]);
}

TEST_F(TestGraph, ValidateFailsWithDuplicateTensorUids)
{
    GraphTestUtils graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(x->get_dim());
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    //validate graph is good.
    auto result = graph.validate();
    EXPECT_TRUE(result.is_good()) << result.get_message();

    //introduce duplicate uids
    mean->set_uid(1);
    invVariance->set_uid(1);
    scale->set_uid(1);
    bias->set_uid(2);
    y1->set_uid(2);
    y2->set_uid(3);

    result = graph.validate();
    EXPECT_FALSE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsPassesWithNoDuplicates)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsFailsWithDuplicates)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(2); // Duplicate UID
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3); // Duplicate UID

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.get_message().find("Duplicate tensor UIDs") != std::string::npos);
    EXPECT_TRUE(result.get_message().find('2') != std::string::npos);
    EXPECT_TRUE(result.get_message().find('3') != std::string::npos);
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsPassesWithReusedTensors)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    // These tensors will be reused across multiple batchnorm nodes
    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    // Node 1: Uses mean, invVariance, scale, bias
    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    // Node 2: REUSES the same mean, invVariance, scale, bias tensors
    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Should pass - same tensor objects being reused is fine
    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsFailsWithReusedUidsOnDifferentTensors)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    // Node 1 tensors
    auto mean1 = std::make_shared<TensorAttributes>();
    mean1->set_uid(2);
    auto invVariance1 = std::make_shared<TensorAttributes>();
    invVariance1->set_uid(3);
    auto scale1 = std::make_shared<TensorAttributes>();
    scale1->set_uid(4);
    auto bias1 = std::make_shared<TensorAttributes>();
    bias1->set_uid(5);

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean1, invVariance1, scale1, bias1, attributes1);

    // Node 2 tensors - DIFFERENT objects but SAME UIDs
    auto mean2 = std::make_shared<TensorAttributes>();
    mean2->set_uid(2); // Same UID as mean1 but different object
    auto invVariance2 = std::make_shared<TensorAttributes>();
    invVariance2->set_uid(3); // Same UID as invVariance1 but different object
    auto scale2 = std::make_shared<TensorAttributes>();
    scale2->set_uid(4); // Same UID as scale1 but different object
    auto bias2 = std::make_shared<TensorAttributes>();
    bias2->set_uid(5); // Same UID as bias1 but different object

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean2, invVariance2, scale2, bias2, attributes2);

    // Should fail - different tensor objects with same UIDs
    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.get_message().find("Duplicate tensor UIDs") != std::string::npos);
}

TEST_F(TestGraph, CheckTensorUidsSetPassesWhenAllUidsSet)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);
    y->set_uid(6);

    auto result = graph.checkTensorUidsSet();
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(result.code, ErrorCode::OK);
}

TEST_F(TestGraph, CheckTensorUidsSetFailsWhenUidMissing)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_name("X").set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_name("Mean");
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    auto result = graph.checkTensorUidsSet();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_NE(result.get_message().find("Tensors without UIDs"), std::string::npos)
        << result.get_message();
    EXPECT_NE(result.get_message().find("Mean"), std::string::npos) << result.get_message();
    EXPECT_NE(result.get_message().find("(unnamed)"), std::string::npos) << result.get_message();
}

TEST_F(TestGraph, BuildOperationGraphAllMissingTensorUids)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    // Before build_operation_graph, UIDs should not be set
    EXPECT_FALSE(x->has_uid());
    EXPECT_FALSE(mean->has_uid());
    EXPECT_FALSE(invVariance->has_uid());
    EXPECT_FALSE(scale->has_uid());
    EXPECT_FALSE(bias->has_uid());
    EXPECT_FALSE(y->has_uid());

    auto validateResult = graph.validate();
    ASSERT_TRUE(validateResult.is_good()) << validateResult.get_message();
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After build_operation_graph, all UIDs should be populated
    EXPECT_TRUE(x->has_uid());
    EXPECT_TRUE(mean->has_uid());
    EXPECT_TRUE(invVariance->has_uid());
    EXPECT_TRUE(scale->has_uid());
    EXPECT_TRUE(bias->has_uid());
    EXPECT_TRUE(y->has_uid());

    // Verify all UIDs are unique
    std::unordered_set<int64_t> uids;
    uids.insert(x->get_uid());
    uids.insert(mean->get_uid());
    uids.insert(invVariance->get_uid());
    uids.insert(scale->get_uid());
    uids.insert(bias->get_uid());
    uids.insert(y->get_uid());

    EXPECT_EQ(uids.size(), 6);
}

TEST_F(TestGraph, BuildOperationGraphPopulatesOnlyMissingUids)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(100); // Pre-set UID

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    mean->set_uid(200); // Pre-set UID

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    invVariance->set_uid(300); // Pre-set UID

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    scale->set_uid(400); // Pre-set UID

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    // bias does not have a UID set

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);
    y->set_uid(500); // Pre-set UID

    EXPECT_TRUE(x->has_uid());
    EXPECT_EQ(x->get_uid(), 100);
    EXPECT_TRUE(mean->has_uid());
    EXPECT_EQ(mean->get_uid(), 200);
    EXPECT_TRUE(invVariance->has_uid());
    EXPECT_EQ(invVariance->get_uid(), 300);
    EXPECT_TRUE(scale->has_uid());
    EXPECT_EQ(scale->get_uid(), 400);
    EXPECT_FALSE(bias->has_uid());
    EXPECT_TRUE(y->has_uid());
    EXPECT_EQ(y->get_uid(), 500);

    ASSERT_TRUE(graph.validate().is_good());
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After build_operation_graph, bias should have a UID
    EXPECT_TRUE(bias->has_uid());

    // All pre-existing UIDs should remain unchanged
    EXPECT_EQ(x->get_uid(), 100);
    EXPECT_EQ(mean->get_uid(), 200);
    EXPECT_EQ(invVariance->get_uid(), 300);
    EXPECT_EQ(scale->get_uid(), 400);
    EXPECT_EQ(y->get_uid(), 500);

    // The new UID for bias should be unique
    const int64_t biasUid = bias->get_uid();
    EXPECT_NE(biasUid, 100);
    EXPECT_NE(biasUid, 200);
    EXPECT_NE(biasUid, 300);
    EXPECT_NE(biasUid, 400);
    EXPECT_NE(biasUid, 500);
}

TEST_F(TestGraph, GetTensorsByUidReturnsMap)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(42)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("PointwiseNode");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_uid(99);

    // Get tensor map by UID
    auto tensorsByUid = graph.getTensorsByUid();

    // Map contains expected tensors
    EXPECT_EQ(tensorsByUid.size(), 2);

    auto itX = tensorsByUid.find(42);
    ASSERT_NE(itX, tensorsByUid.end());
    EXPECT_EQ(itX->second->get_uid(), 42);
    EXPECT_EQ(itX->second->get_name(), "X");
    EXPECT_EQ(itX->second, x);

    auto itY = tensorsByUid.find(99);
    ASSERT_NE(itY, tensorsByUid.end());
    EXPECT_EQ(itY->second->get_uid(), 99);
    EXPECT_EQ(itY->second, y);

    // Non-existent UID not found in map
    auto notFound = tensorsByUid.find(999);
    EXPECT_EQ(notFound, tensorsByUid.end());
}

TEST_F(TestGraph, GetTensorsByNameReturnsMap)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("PointwiseNode");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);

    // Get tensor map by name
    auto tensorsByName = graph.getTensorsByName();

    // Map contains expected tensors
    EXPECT_EQ(tensorsByName.size(), 2);

    auto itX = tensorsByName.find("InputTensor");
    ASSERT_NE(itX, tensorsByName.end());
    EXPECT_EQ(itX->second->get_name(), "InputTensor");
    EXPECT_EQ(itX->second, x);

    // Output tensor has auto-generated name
    auto itY = tensorsByName.find("PointwiseNode::OUT_0");
    ASSERT_NE(itY, tensorsByName.end());
    EXPECT_EQ(itY->second, y);

    // Non-existent name not found in map
    auto notFound = tensorsByName.find("NonExistentTensor");
    EXPECT_EQ(notFound, tensorsByName.end());
}

TEST_F(TestGraph, GetTensorsByUidAndNameIncludePeerStatTensors)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputX")
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("ScaleTensor")
        .set_dim({64})
        .set_stride({1})
        .set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3)
        .set_name("BiasTensor")
        .set_dim({64})
        .set_stride({1})
        .set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    // Create peer stat tensors (these go in the separate peer_stats vector, not outputs)
    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(20).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(21).set_name("PeerStat2");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("BN");
    bnAttrs.set_epsilon(epsilon);
    bnAttrs.set_peer_stats({peerStat1, peerStat2}); // Set peer stats separately

    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, bnAttrs);

    // Assign UIDs to output tensors
    y->set_uid(10);
    savedMean->set_uid(11);
    savedInvVariance->set_uid(12);

    // Test getTensorsByUid()
    auto tensorsByUid = graph.getTensorsByUid();

    // Should have:
    // Inputs: x(1), scale(2), bias(3), epsilon(4) = 4
    // Outputs: y(10), savedMean(11), savedInvVariance(12) = 3
    // Peer stats: peerStat1(20), peerStat2(21) = 2
    // Total = 9
    EXPECT_EQ(tensorsByUid.size(), 9);

    // Verify input tensors are present by UID
    EXPECT_NE(tensorsByUid.find(1), tensorsByUid.end()); // x
    EXPECT_NE(tensorsByUid.find(2), tensorsByUid.end()); // scale
    EXPECT_NE(tensorsByUid.find(3), tensorsByUid.end()); // bias
    EXPECT_NE(tensorsByUid.find(4), tensorsByUid.end()); // epsilon

    // Verify output tensors are present by UID
    EXPECT_NE(tensorsByUid.find(10), tensorsByUid.end()); // y
    EXPECT_NE(tensorsByUid.find(11), tensorsByUid.end()); // savedMean
    EXPECT_NE(tensorsByUid.find(12), tensorsByUid.end()); // savedInvVariance

    // Verify peer stat tensors are present by UID (tests the specialized gather override)
    EXPECT_NE(tensorsByUid.find(20), tensorsByUid.end()); // peerStat1
    EXPECT_NE(tensorsByUid.find(21), tensorsByUid.end()); // peerStat2

    // Verify the pointers match
    EXPECT_EQ(tensorsByUid[1], x);
    EXPECT_EQ(tensorsByUid[10], y);
    EXPECT_EQ(tensorsByUid[20], peerStat1);
    EXPECT_EQ(tensorsByUid[21], peerStat2);

    // Test getTensorsByName()
    auto tensorsByName = graph.getTensorsByName();

    // Should have all named tensors:
    // Inputs with names: InputX, ScaleTensor, BiasTensor (3) - epsilon has no name
    // Outputs with names: BN::Y, BN::MEAN, BN::INV_VARIANCE (3)
    // Peer stats with names: PeerStat1, PeerStat2 (2)
    // Total = 8
    EXPECT_EQ(tensorsByName.size(), 8);

    // Verify input tensors are present by name
    EXPECT_NE(tensorsByName.find("InputX"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("ScaleTensor"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BiasTensor"), tensorsByName.end());

    // Verify output tensors are present by name
    EXPECT_NE(tensorsByName.find("BN::Y"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BN::MEAN"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BN::INV_VARIANCE"), tensorsByName.end());

    // Verify peer stat tensors are present by name (tests the specialized gather override)
    EXPECT_NE(tensorsByName.find("PeerStat1"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("PeerStat2"), tensorsByName.end());

    // Verify the pointers match
    EXPECT_EQ(tensorsByName["InputX"], x);
    EXPECT_EQ(tensorsByName["BN::Y"], y);
    EXPECT_EQ(tensorsByName["PeerStat1"], peerStat1);
    EXPECT_EQ(tensorsByName["PeerStat2"], peerStat2);
}

TEST_F(TestGraph, GetRankedEngineIdsReturnsRankedList)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("RankedEngineIdsTestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    graph.build_operation_graph(_handle);

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count (3 engines available)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Create 3 engine config descriptors
    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2001);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2002);
    auto engineConfigDesc3 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2003);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3001);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3002);
    auto engineDesc3 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3003);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc3](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting engine configs
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    3,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Finalize and get IDs for each engine config
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 100;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 200;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc3))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc3,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc3](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc3;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc3, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 300;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(rankedEngineIds.size(), 3);
    EXPECT_EQ(rankedEngineIds[0], 100);
    EXPECT_EQ(rankedEngineIds[1], 200);
    EXPECT_EQ(rankedEngineIds[2], 300);
}

TEST_F(TestGraph, BuildMethodSucceedsWithValidGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("BuildMethodTestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock create_execution_plans
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock build_plans
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(executionPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Workspace query during build_plans(HEURISTICS_CHOICE)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* ptr) {
            *static_cast<int64_t*>(ptr) = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.build(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtFailsWithoutGraphBuilt)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(),
              "Graph has not been built, build the operation graph first. Cannot create "
              "execution plan.");
}

TEST_F(TestGraph, CreateExecutionPlanExtWithEmptySettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    const int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting knob count - return 0 (no knobs available)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Empty settings - should not call backendSetAttribute for knobs
    const std::vector<KnobSetting> settings;
    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, BuildMethodFailsWhenValidationFails)
{
    Graph graph;

    // Create an invalid graph (compute type not set)
    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::NOT_SET)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(in0, attributes);

    auto result = graph.build(_handle);
    EXPECT_FALSE(result.is_good());
}

// ============================================================================
// get_knobs_for_engine Tests
// ============================================================================

// Helper to set up mock expectations for a single knob info descriptor.
// Mocks all KNOB_INFO attributes that unpackKnobsFromDescriptors reads:
// knob ID, description, deprecated flag, default value type/value, and
// constraint attributes (absent by default for int/double, or empty for
// string valid values).
static void
    setupKnobDescriptorMock(std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
                            hipdnnBackendDescriptor_t knobDesc,
                            const std::string& knobId,
                            const std::string& description,
                            bool deprecated,
                            hipdnnBackendAttributeType_t valueType,
                            const KnobValueVariant& defaultValue)
{
    // Helper lambdas for mock attribute patterns

    // Mock string attribute: size-query + data-read
    auto mockString = [&](hipdnnBackendAttributeName_t attrName, const std::string& value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_CHAR, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                            Return(HIPDNN_STATUS_SUCCESS)));

        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc,
                                        attrName,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(value.size() + 1),
                                        _,
                                        NotNull()))
            .WillOnce(DoAll(
                SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                Invoke([value](hipdnnBackendDescriptor_t,
                               hipdnnBackendAttributeName_t,
                               hipdnnBackendAttributeType_t,
                               int64_t,
                               int64_t*,
                               void* out) { std::memcpy(out, value.c_str(), value.size() + 1); }),
                Return(HIPDNN_STATUS_SUCCESS)));
    };

    // Mock scalar attribute: single get call with typed value
    auto mockScalarInt64 = [&](hipdnnBackendAttributeName_t attrName, int64_t value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_INT64, 1, _, NotNull()))
            .WillOnce(DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* out) { *static_cast<int64_t*>(out) = value; }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockScalarDouble = [&](hipdnnBackendAttributeName_t attrName, double value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_DOUBLE, 1, _, NotNull()))
            .WillOnce(DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* out) { *static_cast<double*>(out) = value; }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockScalarBool = [&](hipdnnBackendAttributeName_t attrName, bool value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_BOOLEAN, 1, _, NotNull()))
            .WillOnce(DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                           hipdnnBackendAttributeName_t,
                                           hipdnnBackendAttributeType_t,
                                           int64_t,
                                           int64_t*,
                                           void* out) { *static_cast<bool*>(out) = value; }),
                            Return(HIPDNN_STATUS_SUCCESS)));
    };

    // Mock optional scalar absent (count query returns 0)
    auto mockOptAbsent = [&](hipdnnBackendAttributeName_t attrName,
                             hipdnnBackendAttributeType_t attrType) {
        EXPECT_CALL(*mockBackend, backendGetAttribute(knobDesc, attrName, attrType, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    };

    // Mock empty vector attribute (count query returns 0)
    auto mockEmptyVec = [&](hipdnnBackendAttributeName_t attrName) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_INT64, 0, _, nullptr))
            .WillOnce(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    };

    // 1. Knob ID (string)
    mockString(HIPDNN_ATTR_KNOB_INFO_TYPE, knobId);

    // 2. Description (string)
    mockString(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, description);

    // 3. Deprecated flag (bool)
    mockScalarBool(HIPDNN_ATTR_KNOB_INFO_DEPRECATED, deprecated);

    // 4. Default value type (int64 holding the type enum)
    mockScalarInt64(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, static_cast<int64_t>(valueType));

    // 5. Default value and constraint fields, dispatched on type
    if(valueType == HIPDNN_TYPE_INT64)
    {
        auto intVal = std::get<int64_t>(defaultValue);
        mockScalarInt64(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, intVal);

        // No constraints (all absent)
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);
        mockEmptyVec(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);
    }
    else if(valueType == HIPDNN_TYPE_DOUBLE)
    {
        auto doubleVal = std::get<double>(defaultValue);
        mockScalarDouble(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, doubleVal);

        // No constraints (all absent)
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_DOUBLE);
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_DOUBLE);
    }
    else if(valueType == HIPDNN_TYPE_CHAR)
    {
        auto strVal = std::get<std::string>(defaultValue);
        mockString(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, strVal);

        // No string max length
        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32);

        // No valid values string (not supported)
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc,
                                        HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                        HIPDNN_TYPE_CHAR,
                                        0,
                                        _,
                                        nullptr))
            .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));
    }
}

// Helper to build a graph and mock engine descriptor creation for knob queries.
// Returns the fake engine descriptor pointer.
static hipdnnBackendDescriptor_t
    buildGraphAndMockEngine(std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
                            Graph& graph,
                            hipdnnHandle_t handle,
                            int engineDescTimes = 1)
{
    createBasicBatchnormGraph(graph);
    EXPECT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*mockBackend, backendFinalize(_)).Times(AnyNumber());

    auto buildResult = graph.build_operation_graph(handle);
    EXPECT_TRUE(buildResult.is_good());

    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xE001);
    EXPECT_CALL(*mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(engineDescTimes)
        .WillRepeatedly(
            [engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(engineDescTimes)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(engineDescTimes)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*mockBackend, backendFinalize(engineDesc))
        .Times(engineDescTimes)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    return engineDesc;
}

// Helper to mock the KNOB_INFO array query on an engine descriptor.
// Returns count knob descriptors (fakeKnobDescs) when the engine is queried.
static void
    mockKnobInfoQuery(std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
                      hipdnnBackendDescriptor_t engineDesc,
                      const std::vector<hipdnnBackendDescriptor_t>& fakeKnobDescs)
{
    auto count = static_cast<int64_t>(fakeKnobDescs.size());

    EXPECT_CALL(*mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([count](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t* elementCount,
                          void*) {
            *elementCount = count;
            return HIPDNN_STATUS_SUCCESS;
        });

    if(count > 0)
    {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(engineDesc,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        count,
                                        _,
                                        NotNull()))
            .WillOnce([fakeKnobDescs](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t* elementCount,
                                      void* out) {
                *elementCount = static_cast<int64_t>(fakeKnobDescs.size());
                auto* arr = static_cast<hipdnnBackendDescriptor_t*>(out);
                for(size_t i = 0; i < fakeKnobDescs.size(); ++i)
                {
                    arr[i] = fakeKnobDescs[i];
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }
}

TEST_F(TestGraph, GetKnobsForEngineFailsWhenGraphNotBuilt)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    const int64_t engineId = 42;
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);
    EXPECT_FALSE(result.is_good());
    EXPECT_NE(result.get_message().find("Graph has not been built"), std::string::npos);
}

TEST_F(TestGraph, GetKnobsForEngineReturnsEmptyVectorWhenNoKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    // No knobs available
    mockKnobInfoQuery(_mockBackend, engineDesc, {});

    const int64_t engineId = 42;
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestGraph, GetKnobsForEngineReturnsKnobsWhenAvailable)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    // Two knob descriptors: 1 int, 1 double
    auto knobDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    auto knobDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA002);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobDesc1, knobDesc2});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc1,
                            "tile_size",
                            "Tile dimension",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{256}});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc2,
                            "learning_rate",
                            "Learning rate parameter",
                            false,
                            HIPDNN_TYPE_DOUBLE,
                            KnobValueVariant{0.01});

    const int64_t engineId = 42;
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(knobs.size(), 2u);
    EXPECT_EQ(knobs[0].knobId(), "tile_size");
    EXPECT_EQ(knobs[0].valueType(), KnobValueType::INT64);
    EXPECT_EQ(knobs[1].knobId(), "learning_rate");
    EXPECT_EQ(knobs[1].valueType(), KnobValueType::FLOAT64);
}

TEST_F(TestGraph, GetKnobsForEngineHandlesDeprecatedKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    auto knobDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA003);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobDesc1});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc1,
                            "old_tile_size",
                            "Deprecated tile knob",
                            true,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{128}});

    const int64_t engineId = 42;
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knobId(), "old_tile_size");
    EXPECT_TRUE(knobs[0].isDeprecated());
}

TEST_F(TestGraph, GetKnobsForEngineHandlesStringKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    auto knobDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA004);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobDesc1});

    // String knob with VALID_VALUES_STRING_EXT constraint requires custom setup
    // since setupKnobDescriptorMock defaults to NOT_SUPPORTED for valid values.
    // We set up the basic attributes manually and add the constraint mock.

    // Knob ID
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(knobDesc1, HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{10}), Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    knobDesc1, HIPDNN_ATTR_KNOB_INFO_TYPE, HIPDNN_TYPE_CHAR, 10, _, NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{10}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) { std::memcpy(out, "algorithm\0", 10); }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Description
    const std::string desc = "Algorithm choice";
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    knobDesc1, HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(desc.size() + 1)),
                        Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(knobDesc1,
                                    HIPDNN_ATTR_KNOB_INFO_DESCRIPTION,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(desc.size() + 1),
                                    _,
                                    NotNull()))
        .WillOnce(
            DoAll(SetArgPointee<4>(static_cast<int64_t>(desc.size() + 1)),
                  Invoke([desc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* out) { std::memcpy(out, desc.c_str(), desc.size() + 1); }),
                  Return(HIPDNN_STATUS_SUCCESS)));

    // Deprecated = false
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            knobDesc1, HIPDNN_ATTR_KNOB_INFO_DEPRECATED, HIPDNN_TYPE_BOOLEAN, 1, _, NotNull()))
        .WillOnce(DoAll(Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) { *static_cast<bool*>(out) = false; }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Default value type = CHAR
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(knobDesc1,
                                    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) {
                            *static_cast<int64_t*>(out) = static_cast<int64_t>(HIPDNN_TYPE_CHAR);
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Default value = "fast"
    const std::string defaultStr = "fast";
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            knobDesc1, HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(defaultStr.size() + 1)),
                        Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(knobDesc1,
                                    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE,
                                    HIPDNN_TYPE_CHAR,
                                    static_cast<int64_t>(defaultStr.size() + 1),
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(static_cast<int64_t>(defaultStr.size() + 1)),
                        Invoke([defaultStr](hipdnnBackendDescriptor_t,
                                            hipdnnBackendAttributeName_t,
                                            hipdnnBackendAttributeType_t,
                                            int64_t,
                                            int64_t*,
                                            void* out) {
                            std::memcpy(out, defaultStr.c_str(), defaultStr.size() + 1);
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // String max length: 20
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            knobDesc1, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(1), Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            knobDesc1, HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH, HIPDNN_TYPE_INT32, 1, _, NotNull()))
        .WillOnce(DoAll(Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* out) { *static_cast<int32_t*>(out) = 20; }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Valid values string: "fast\0accurate\0balanced\0"
    const std::string validValBuf
        = std::string("fast") + '\0' + "accurate" + '\0' + "balanced" + '\0';
    const auto validValBufLen = static_cast<int64_t>(validValBuf.size());

    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            knobDesc1, HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING, HIPDNN_TYPE_CHAR, 0, _, nullptr))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen), Return(HIPDNN_STATUS_SUCCESS)));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(knobDesc1,
                                    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING,
                                    HIPDNN_TYPE_CHAR,
                                    validValBufLen,
                                    _,
                                    NotNull()))
        .WillOnce(DoAll(SetArgPointee<4>(validValBufLen),
                        Invoke([&validValBuf](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t,
                                              int64_t*,
                                              void* out) {
                            std::memcpy(out, validValBuf.data(), validValBuf.size());
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    const int64_t engineId = 42;
    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knobId(), "algorithm");
    EXPECT_EQ(knobs[0].valueType(), KnobValueType::STRING);

    auto* defaultVal = std::get_if<std::string>(&knobs[0].defaultValue());
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(*defaultVal, "fast");

    auto* strConstraint = dynamic_cast<const StringConstraint*>(knobs[0].constraint());
    ASSERT_NE(strConstraint, nullptr);
    EXPECT_EQ(strConstraint->getMaxLength(), 20);
    const auto& validValues = strConstraint->getValidValues();
    EXPECT_EQ(validValues.size(), 3u);
    EXPECT_NE(validValues.find("fast"), validValues.end());
    EXPECT_NE(validValues.find("accurate"), validValues.end());
    EXPECT_NE(validValues.find("balanced"), validValues.end());
}

// ============================================================================
// get_knob_lookup_for_engine Tests
// ============================================================================

TEST_F(TestGraph, GetKnobLookupForEngineReturnsMapByKnobId)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    auto knobDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA005);
    auto knobDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA006);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobDesc1, knobDesc2});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc1,
                            "split_k",
                            "Split-K factor",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{2}});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc2,
                            "precision_mode",
                            "Precision tuning",
                            false,
                            HIPDNN_TYPE_DOUBLE,
                            KnobValueVariant{1.0});

    const int64_t engineId = 42;
    std::unordered_map<KnobType_t, Knob> knobMap;
    auto result = graph.get_knob_lookup_for_engine(engineId, knobMap);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(knobMap.size(), 2u);
    EXPECT_NE(knobMap.find("split_k"), knobMap.end());
    EXPECT_NE(knobMap.find("precision_mode"), knobMap.end());
    EXPECT_EQ(knobMap.at("split_k").valueType(), KnobValueType::INT64);
    EXPECT_EQ(knobMap.at("precision_mode").valueType(), KnobValueType::FLOAT64);
}

TEST_F(TestGraph, GetKnobLookupForEngineReturnsEmptyMapWhenNoKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    mockKnobInfoQuery(_mockBackend, engineDesc, {});

    const int64_t engineId = 42;
    std::unordered_map<KnobType_t, Knob> knobMap;
    auto result = graph.get_knob_lookup_for_engine(engineId, knobMap);
    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(knobMap.empty());
}

TEST_F(TestGraph, GetKnobLookupForEngineClearsPreExistingEntries)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    auto knobDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA007);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobDesc1});

    setupKnobDescriptorMock(_mockBackend,
                            knobDesc1,
                            "new_knob",
                            "New knob",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{10}});

    // Pre-populate the map with stale entries
    std::unordered_map<KnobType_t, Knob> knobMap;
    {
        auto [err, knob]
            = Knob::tryCreate("stale_knob", "Stale", KnobValueVariant{int64_t{0}}, false);
        ASSERT_TRUE(err.is_good());
        knobMap.emplace("stale_knob", std::move(knob));
    }
    ASSERT_EQ(knobMap.size(), 1u);
    ASSERT_NE(knobMap.find("stale_knob"), knobMap.end());

    const int64_t engineId = 42;
    auto result = graph.get_knob_lookup_for_engine(engineId, knobMap);
    EXPECT_TRUE(result.is_good()) << result.get_message();

    // Old entry should be gone, only new knob present
    EXPECT_EQ(knobMap.size(), 1u);
    EXPECT_EQ(knobMap.find("stale_knob"), knobMap.end());
    EXPECT_NE(knobMap.find("new_knob"), knobMap.end());
}

// ============================================================================
// create_execution_plan_ext with Knob Settings Tests
// ============================================================================

TEST_F(TestGraph, CreateExecutionPlanExtWithKnobSettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    // Engine descriptor is created twice in create_execution_plan_ext:
    // once in get_knob_lookup_for_engine, once in initializeEngineConfig
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle, 2);

    // Mock one int knob available on the engine
    auto knobInfoDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB001);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobInfoDesc});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc,
                            "tile_size",
                            "Tile dimension",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{256}});

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([engineConfigDesc](hipdnnBackendDescriptorType_t,
                                     hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock knob choice descriptor creation for the setting
    auto knobChoiceDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, _))
        .WillOnce(
            [knobChoiceDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = knobChoiceDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Knob choice: set knob ID (string)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice: set knob value (int64)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice finalize
    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config: set knob choices
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config finalize
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([executionPlanDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("tile_size", int64_t{512});

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtIgnoresUnsupportedKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle, 2);

    // Mock one int knob available on the engine
    auto knobInfoDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB002);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobInfoDesc});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc,
                            "supported_knob",
                            "Supported knob",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{100}});

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC002);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([engineConfigDesc](hipdnnBackendDescriptorType_t,
                                     hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock knob choice descriptor for the one supported setting
    auto knobChoiceDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD002);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, _))
        .WillOnce(
            [knobChoiceDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = knobChoiceDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Only 1 knob choice applied (the unsupported one is skipped)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF002);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([executionPlanDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("supported_knob", int64_t{200});
    settings.emplace_back("unsupported_knob", int64_t{999});

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtWithDeprecatedKnob)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle, 2);

    // Mock one deprecated int knob available on the engine
    auto knobInfoDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB003);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobInfoDesc});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc,
                            "legacy_tile_size",
                            "Legacy tile knob",
                            true,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{64}});

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC003);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([engineConfigDesc](hipdnnBackendDescriptorType_t,
                                     hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock knob choice descriptor for the deprecated knob setting
    auto knobChoiceDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD003);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, _))
        .WillOnce(
            [knobChoiceDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = knobChoiceDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Deprecated knobs still get applied
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF003);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([executionPlanDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("legacy_tile_size", int64_t{128});

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanWithInt64Knobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    // Engine descriptor is created twice in create_execution_plan_ext:
    // once in get_knob_lookup_for_engine, once in initializeEngineConfig
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle, 2);

    // Mock one int knob available on the engine
    auto knobInfoDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB010);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobInfoDesc});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc,
                            "global.deterministic",
                            "Enable deterministic execution",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{0}});

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC010);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([engineConfigDesc](hipdnnBackendDescriptorType_t,
                                     hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock knob choice descriptor creation for the setting
    auto knobChoiceDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD010);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, _))
        .WillOnce(
            [knobChoiceDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = knobChoiceDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Knob choice: set knob ID (string)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice: set knob value (int64)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    knobChoiceDesc, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice finalize
    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config: set knob choices
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config finalize
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF010);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([executionPlanDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtWithMultipleKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    // Engine descriptor is created twice in create_execution_plan_ext:
    // once in get_knob_lookup_for_engine, once in initializeEngineConfig
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle, 2);

    // Mock two int knobs available on the engine
    auto knobInfoDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB020);
    auto knobInfoDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB021);
    mockKnobInfoQuery(_mockBackend, engineDesc, {knobInfoDesc1, knobInfoDesc2});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc1,
                            "global.deterministic",
                            "Enable deterministic execution",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{0}});

    setupKnobDescriptorMock(_mockBackend,
                            knobInfoDesc2,
                            "performance.threads",
                            "Number of threads",
                            false,
                            HIPDNN_TYPE_INT64,
                            KnobValueVariant{int64_t{4}});

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC020);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([engineConfigDesc](hipdnnBackendDescriptorType_t,
                                     hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock knob choice descriptors for both settings
    auto knobChoiceDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD020);
    auto knobChoiceDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xD021);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, _))
        .WillOnce([knobChoiceDesc1](hipdnnBackendDescriptorType_t,
                                    hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = knobChoiceDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([knobChoiceDesc2](hipdnnBackendDescriptorType_t,
                                    hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = knobChoiceDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Knob choice: set knob IDs (string) for both
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice: set knob values (int64) for both
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Knob choice finalize for both
    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(knobChoiceDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config: set knob choices (2 knobs)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Engine config finalize
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xF020);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([executionPlanDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    const int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));
    settings.emplace_back("performance.threads", static_cast<int64_t>(8));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, GetKnobsForEngineHandlesCountMismatch)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngine(_mockBackend, graph, _handle);

    // Mock getting knob count - return 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting actual knob data - but return a count greater than requested,
    // which triggers a mismatch error in getDescriptorAttrDescArray
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* actualCount,
                     void*) {
            *actualCount = 3; // Mismatch: more than requested
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(42, knobs);

    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestGraph, EngineOverrideDoesNotReplaceExplicitlySetEngineId)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2).set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvFpropAttributes convAttr;
    convAttr.set_name("EngineOverrideConv")
        .set_pre_padding({1, 1})
        .set_post_padding({1, 1})
        .set_stride({1, 1})
        .set_dilation({1, 1});

    graph.conv_fprop(x, w, convAttr);

    constexpr int64_t EXPLICIT_ENGINE_ID = 42;
    graph.set_preferred_engine_id_ext(EXPLICIT_ENGINE_ID);

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    ASSERT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), EXPLICIT_ENGINE_ID);
}

TEST_F(TestGraph, SetPreferredEngineIdByName)
{
    // NOLINTNEXTLINE(misc-const-correctness)
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // Set by name
    graph.set_preferred_engine_id_ext(testEngineName);

    // Verify it was converted to the correct ID
    auto expectedId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedId);
}

TEST_F(TestGraph, SetPreferredEngineIdByEmptyStringClearsPreference)
{
    // NOLINTNEXTLINE(misc-const-correctness)
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // First set a preference
    graph.set_preferred_engine_id_ext(testEngineName);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());

    // Then clear it with empty string
    graph.set_preferred_engine_id_ext("");

    // Verify no preferred engine ID is set
    EXPECT_FALSE(graph.get_preferred_engine_id_ext().has_value());
}

TEST_F(TestGraph, SetPreferredEngineIdByNameThenById)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // Set by name first
    graph.set_preferred_engine_id_ext(testEngineName);

    // Then override with a different ID
    const int64_t overrideId = 999;
    graph.set_preferred_engine_id_ext(std::optional<int64_t>(overrideId));

    // Verify the ID overload took precedence
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), overrideId);
}

TEST_F(TestGraph, SetPreferredEngineIdByIdThenByName)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";
    auto expectedId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);

    // Set by ID first
    graph.set_preferred_engine_id_ext(std::optional<int64_t>(999));

    // Then override with name
    graph.set_preferred_engine_id_ext(testEngineName);

    // Verify the name overload took precedence
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedId);
}

TEST_F(TestGraph, MethodChaining)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_CHAINING";
    auto expectedEngineId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);

    // Test that all setters return reference to self for chaining
    auto& ref1 = graph.set_name("ChainedGraph");
    auto& ref2 = ref1.set_compute_data_type(DataType::FLOAT);
    auto& ref3 = ref2.set_intermediate_data_type(DataType::HALF);
    auto& ref4 = ref3.set_io_data_type(DataType::BFLOAT16);
    auto& ref5 = ref4.set_preferred_engine_id_ext(12345);

    // All references should point to the same object
    EXPECT_EQ(&graph, &ref1);
    EXPECT_EQ(&graph, &ref2);
    EXPECT_EQ(&graph, &ref3);
    EXPECT_EQ(&graph, &ref4);
    EXPECT_EQ(&graph, &ref5);

    // Verify all values were set correctly
    EXPECT_EQ(graph.get_name(), "ChainedGraph");
    EXPECT_EQ(graph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(graph.get_io_data_type(), DataType::BFLOAT16);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), 12345);

    // Test chaining with string overload
    auto& ref6 = graph.set_preferred_engine_id_ext(testEngineName);
    EXPECT_EQ(&graph, &ref6);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedEngineId);
}

// Test that get_ranked_engine_ids returns an error when the backend fails
// to finalize the heuristic descriptor. This ensures proper error propagation
// from the backend through the frontend API.
TEST_F(TestGraph, GetRankedEngineIdsFailsWhenHeuristicCreationFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Make finalize fail for the heuristic descriptor
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);

    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.get_message().find("Failed to finalize engine heuristic descriptor"),
              std::string::npos);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(TestGraph, MoveConstruction)
{
    Graph originalGraph;
    originalGraph.set_name("OriginalGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    // Move construct
    const Graph movedGraph(std::move(originalGraph));

    // Verify moved graph has the original state
    EXPECT_EQ(movedGraph.get_name(), "OriginalGraph");
    EXPECT_EQ(movedGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(movedGraph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(movedGraph.get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(originalGraph.get_name(), ""); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(originalGraph.getTensorsByName().empty());
}

TEST_F(TestGraph, MoveAssignment)
{
    // NOLINTNEXTLINE(misc-const-correctness)
    Graph originalGraph;
    originalGraph.set_name("OriginalGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    Graph movedGraph;
    movedGraph.set_name("TargetGraph");

    // Move assign
    movedGraph = std::move(originalGraph);

    // Verify moved graph has the original state
    EXPECT_EQ(movedGraph.get_name(), "OriginalGraph");
    EXPECT_EQ(movedGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(movedGraph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(movedGraph.get_io_data_type(), DataType::FLOAT);
}

TEST_F(TestGraph, MoveConstructionWithNodes)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithNodes")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Add a batchnorm node to the graph
    auto y = createBasicBatchnormGraph(originalGraph);
    EXPECT_NE(y, nullptr);

    // Get tensor count before move
    auto tensorsBeforeMove = originalGraph.getTensorsByName();
    const size_t tensorCountBeforeMove = tensorsBeforeMove.size();
    EXPECT_GT(tensorCountBeforeMove, 0);

    // Move construct
    const Graph movedGraph(std::move(originalGraph));

    // Verify moved graph has the nodes
    auto tensorsAfterMove = movedGraph.getTensorsByName();
    EXPECT_EQ(tensorsAfterMove.size(), tensorCountBeforeMove);

    // Verify graph name was moved
    EXPECT_EQ(movedGraph.get_name(), "TestGraph");
}

TEST_F(TestGraph, MoveAssignmentWithNodes)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithNodes")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Add a batchnorm node to the graph
    auto y = createBasicBatchnormGraph(originalGraph);
    EXPECT_NE(y, nullptr);

    // Get tensor count before move
    auto tensorsBeforeMove = originalGraph.getTensorsByName();
    const size_t tensorCountBeforeMove = tensorsBeforeMove.size();
    EXPECT_GT(tensorCountBeforeMove, 0);

    Graph movedGraph;
    movedGraph.set_name("TargetGraph");

    // Move assign
    movedGraph = std::move(originalGraph);

    // Verify moved graph has the nodes
    auto tensorsAfterMove = movedGraph.getTensorsByName();
    EXPECT_EQ(tensorsAfterMove.size(), tensorCountBeforeMove);

    // Verify graph name was moved
    EXPECT_EQ(movedGraph.get_name(), "TestGraph");
}

TEST_F(TestGraph, MoveConstructionWithPreferredEngineId)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithEngineId")
        .set_compute_data_type(DataType::FLOAT)
        .set_preferred_engine_id_ext(42);

    EXPECT_TRUE(originalGraph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(originalGraph.get_preferred_engine_id_ext().value(), 42);

    // Move construct
    const Graph movedGraph(std::move(originalGraph));

    // Verify preferred engine id was moved
    EXPECT_TRUE(movedGraph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(movedGraph.get_preferred_engine_id_ext().value(), 42);
}

TEST_F(TestGraph, MoveAssignmentToEmptyGraph)
{
    // NOLINTNEXTLINE(misc-const-correctness)
    Graph sourceGraph;
    sourceGraph.set_name("SourceGraph").set_compute_data_type(DataType::FLOAT);

    Graph targetGraph;
    // Target starts empty
    EXPECT_EQ(targetGraph.get_name(), "");

    // Move assign
    targetGraph = std::move(sourceGraph);

    // Target now has source's state
    EXPECT_EQ(targetGraph.get_name(), "SourceGraph");
    EXPECT_EQ(targetGraph.get_compute_data_type(), DataType::FLOAT);
}

#ifdef HIPDNN_ENABLE_SDPA
TEST_F(TestGraph, SdpaFwdNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16, 64}).set_stride({8192, 1024, 64, 1}).set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaNode");

    auto [o, stats] = graph.sdpa(q, k, v, attributes);

    EXPECT_EQ(o->get_name(), "SdpaNode::O");
    EXPECT_TRUE(o->get_is_virtual());
    EXPECT_EQ(stats, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, SdpaFwdNodeCreationWithStats)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16, 64}).set_stride({8192, 1024, 64, 1}).set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaNodeStats");
    attributes.set_generate_stats(true);

    auto [o, stats] = graph.sdpa(q, k, v, attributes);

    EXPECT_EQ(o->get_name(), "SdpaNodeStats::O");
    EXPECT_TRUE(o->get_is_virtual());
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->get_name(), "SdpaNodeStats::STATS");
    EXPECT_TRUE(stats->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}
#endif // HIPDNN_ENABLE_SDPA

TEST_F(TestGraph, CustomOpNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    CustomOpAttributes attributes;
    attributes.set_name("MyCustomOp").set_custom_op_id("example.my_add");

    auto outputs = graph.custom_op({inputA, inputB}, 2, attributes);

    EXPECT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0]->get_name(), "MyCustomOp::output_0");
    EXPECT_EQ(outputs[1]->get_name(), "MyCustomOp::output_1");
    EXPECT_TRUE(outputs[0]->get_is_virtual());
    EXPECT_TRUE(outputs[1]->get_is_virtual());

    // Custom ops are opaque — output dims must be set explicitly by the caller
    outputs[0]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    outputs[1]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, CustomOpValidateFailsWithNullInput)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto validInput = std::make_shared<TensorAttributes>();
    validInput->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    CustomOpAttributes attributes;
    attributes.set_name("NullInputOp").set_custom_op_id("example.my_add");

    // Pass a nullptr alongside a valid input
    auto outputs = graph.custom_op({validInput, nullptr}, 1, attributes);
    outputs[0]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    // This must return an error, not crash
    auto err = graph.validate();
    EXPECT_FALSE(err.is_good());
}

// ============================================================================
// is_supported_ext() Tests
// ============================================================================

TEST_F(TestGraph, IsSupportedReturnsTrueWhenEnginesAvailable)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    heurDesc, HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count (1 engine available)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, IsSupportedReturnsFalseWhenNoEngines)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    heurDesc, HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count: 0 engines available
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.is_supported_ext(_handle);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

TEST_F(TestGraph, BuildReturnsGraphNotSupportedWhenNoEngines)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count: 0 engines available — exercises getEngineConfigs() probe path.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.build(_handle);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

TEST_F(TestGraph, GetRankedEngineIdsReturnsGraphNotSupportedWhenNoEngines)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Pre-build the graph so get_ranked_engine_ids() proceeds past the
    // "graph not built" guard.
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count: 0 engines available — exercises the
    // getEngineConfigs() availableEngineCount==0 branch (GraphDetail.hpp:53-57)
    // through Graph::get_ranked_engine_ids().
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

TEST_F(TestGraph, GetRankedEngineIdsReturnsGraphNotSupportedWhenRetrievedCountZero)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Pre-build the graph so get_ranked_engine_ids() proceeds past the
    // "graph not built" guard.
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // First backendGetAttribute (probe with requestedElementCount=0, arrayOfElements=nullptr):
    // report 1 available engine so getEngineConfigs() proceeds past the first
    // count check and allocates a single engine-config descriptor.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second backendGetAttribute (retrieval with requestedElementCount>0,
    // arrayOfElements!=nullptr): report 0 retrieved — exercises the
    // count==0 branch at GraphDetail.hpp:87-91.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    Gt(int64_t{0}),
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::GRAPH_NOT_SUPPORTED);
}

TEST_F(TestGraph, IsSupportedAutoBuildsGraphIfNotBuilt)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());

    // Allow descriptor-path calls (tensor/op/graph descriptors, setAttribute, finalize)
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());

    // Mock heuristic descriptor creation
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    heurDesc, HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count (1 engine)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Call is_supported_ext without calling build_operation_graph first
    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, IsSupportedSkipsBuildIfAlreadyBuilt)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Pre-build the graph
    graph.validate();
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Now call is_supported_ext — it should skip validate+build_operation_graph
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(
                    heurDesc, HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT, HIPDNN_TYPE_BOOLEAN, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock engine count: 1 engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.is_supported_ext(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, IsSupportedPropagatesValidationErrors)
{
    Graph graph;

    // Create a graph with invalid configuration (NOT_SET compute data type)
    graph.set_name("InvalidGraph")
        .set_compute_data_type(DataType::NOT_SET)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    graph.pointwise(in0, attributes);

    // is_supported_ext should propagate the validation error
    auto result = graph.is_supported_ext(_handle);
    EXPECT_FALSE(result.is_good());
}

// ============================================================================
// Lowering tests (via public serialize API which triggers lower_to_backend)
// ============================================================================

TEST_F(TestGraph, AutoLowerCreatesDescriptorForConstSerialize)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // The graph has not been lowered yet — const serialize should fail
    std::vector<uint8_t> data;
    const auto& constGraph = graph;
    EXPECT_TRUE(constGraph.serialize(data).is_bad());

    // Non-const serialize auto-lowers via lower_to_backend()
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* buf) {
            *graphByteSize = 4;
            if(buf != nullptr && requestedSize >= 4)
            {
                std::memset(buf, 0xAA, 4);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.serialize(data);
    EXPECT_TRUE(result.is_good()) << result.get_message();

    // Now const serialize should succeed because descriptor exists from auto-lower
    std::vector<uint8_t> data2;
    EXPECT_TRUE(constGraph.serialize(data2).is_good());
    EXPECT_EQ(data, data2);
}

TEST_F(TestGraph, BuildOperationGraphReLowersExistingDescriptor)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // First build
    auto result1 = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result1.is_good()) << result1.get_message();

    // Second build — should succeed and replace the existing descriptor
    auto result2 = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result2.is_good()) << result2.get_message();
}

// ============================================================================
// Binary serialize/deserialize tests
// ============================================================================

TEST_F(TestGraph, ConstBinarySerializeFailsWithoutLoweredDescriptor)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    std::vector<uint8_t> data;
    const auto& constGraph = graph;
    auto result = constGraph.serialize(data);

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, NonConstBinarySerializeAutoLowersWhenNeeded)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            *graphByteSize = 16;
            if(data != nullptr && requestedSize >= 16)
            {
                std::memset(data, 0xAB, 16);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data;
    auto result = graph.serialize(data);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(data.size(), 16u);
    EXPECT_EQ(data[0], 0xAB);
}

TEST_F(TestGraph, BinarySerializePreservesExistingDescriptor)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            *graphByteSize = 8;
            if(data != nullptr && requestedSize >= 8)
            {
                std::memset(data, 0xCD, 8);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data1;
    auto result1 = graph.serialize(data1);
    EXPECT_TRUE(result1.is_good()) << result1.get_message();
    EXPECT_EQ(data1.size(), 8u);

    std::vector<uint8_t> data2;
    auto result2 = graph.serialize(data2);
    EXPECT_TRUE(result2.is_good()) << result2.get_message();
    EXPECT_EQ(data2.size(), 8u);
    EXPECT_EQ(data1, data2);
}

TEST_F(TestGraph, BinaryDeserializeFailsWhenBackendFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04};
    auto result = graph.deserialize(_handle, data);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, BinaryDeserializeForwardsCorrectDataToBackend)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    std::vector<uint8_t> capturedData;
    size_t capturedSize = 0;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&capturedData, &capturedSize](hipdnnBackendDescriptor_t*,
                                                 const uint8_t* serializedGraph,
                                                 size_t graphByteSize) {
            capturedData.assign(serializedGraph, serializedGraph + graphByteSize);
            capturedSize = graphByteSize;
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    const std::vector<uint8_t> inputData = {0xDE, 0xAD, 0xBE, 0xEF};
    graph.deserialize(_handle, inputData);

    EXPECT_EQ(capturedData, inputData);
    EXPECT_EQ(capturedSize, inputData.size());
}

TEST_F(TestGraph, BuildClearsExistingDescriptorFromSerialize)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            *graphByteSize = 16;
            if(data != nullptr && requestedSize >= 16)
            {
                std::memset(data, 0xAB, 16);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data;
    auto serializeResult = graph.serialize(data);
    EXPECT_TRUE(serializeResult.is_good()) << serializeResult.get_message();

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();
}

TEST_F(TestGraph, BinaryDeserializeEmptyData)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_BAD_PARAM));

    const std::vector<uint8_t> emptyData;
    auto result = graph.deserialize(_handle, emptyData);
    EXPECT_TRUE(result.is_bad());
}

// ============================================================================
// JSON serialize/deserialize tests
// ============================================================================

TEST_F(TestGraph, ConstJsonSerializeFailsWithoutLoweredDescriptor)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    std::string jsonData;
    const auto& constGraph = graph;
    auto result = constGraph.serialize(jsonData);

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, NonConstJsonSerializeAutoLowersWhenNeeded)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"test": true})";
                auto len = std::strlen(fakeJson) + 1; // include null terminator
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    std::string jsonData;
    auto result = graph.serialize(jsonData);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(jsonData, R"({"test": true})");
    // Verify null terminator was trimmed from std::string
    EXPECT_FALSE(jsonData.empty());
    EXPECT_NE(jsonData.back(), '\0');
}

TEST_F(TestGraph, JsonDeserializeFailsWhenBackendFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::string jsonData = R"({"graph": "test"})";
    auto result = graph.deserialize(_handle, jsonData);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, JsonDeserializeForwardsCorrectDataToBackend)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    std::string capturedJson;
    size_t capturedSize = 0;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillOnce([&capturedJson, &capturedSize](
                      hipdnnBackendDescriptor_t*, const char* jsonGraph, size_t jsonByteSize) {
            capturedJson = std::string(jsonGraph, jsonByteSize);
            capturedSize = jsonByteSize;
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    const std::string inputJson = R"({"operations": []})";
    graph.deserialize(_handle, inputJson);

    EXPECT_EQ(capturedJson, inputJson);
    EXPECT_EQ(capturedSize, inputJson.size());
}

TEST_F(TestGraph, JsonDeserializeEmptyString)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_BAD_PARAM));

    const std::string emptyJson;
    auto result = graph.deserialize(_handle, emptyJson);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, JsonDeserializeInvalidJson)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::string invalidJson = R"({not valid json})";
    auto result = graph.deserialize(_handle, invalidJson);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, ToBinaryReturnsDataAndSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            const std::vector<uint8_t> fakeData = {0x01, 0x02, 0x03};
            *graphByteSize = fakeData.size();
            if(data != nullptr && requestedSize >= fakeData.size())
            {
                std::memcpy(data, fakeData.data(), fakeData.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    auto [data, err] = graph.to_binary();
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_FALSE(data.empty());
}

TEST_F(TestGraph, ToBinaryConstReturnsDataAndSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            const std::vector<uint8_t> fakeData = {0x01, 0x02, 0x03};
            *graphByteSize = fakeData.size();
            if(data != nullptr && requestedSize >= fakeData.size())
            {
                std::memcpy(data, fakeData.data(), fakeData.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Auto-lower via non-const serialize, then test const overload
    std::vector<uint8_t> discard;
    graph.serialize(discard);

    const auto& constGraph = graph;
    auto [data, err] = constGraph.to_binary();
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_FALSE(data.empty());
}

TEST_F(TestGraph, ToJsonReturnsDataAndSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"test": true})";
                auto len = std::strlen(fakeJson) + 1;
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    auto [jsonData, err] = graph.to_json();
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_FALSE(jsonData.empty());
}

TEST_F(TestGraph, ToJsonConstReturnsDataAndSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"test": true})";
                auto len = std::strlen(fakeJson) + 1;
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    // Auto-lower via non-const serialize, then test const overload
    std::string discard;
    graph.serialize(discard);

    const auto& constGraph = graph;
    auto [jsonData, err] = constGraph.to_json();
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_FALSE(jsonData.empty());
}

// ============================================================================
// Handle-less deserialize tests (no handle, structure only)
// ============================================================================

TEST_F(TestGraph, DeserializeBinaryHandleLess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Serialize the graph to binary
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          size_t requestedSize,
                          size_t* graphByteSize,
                          uint8_t* data) {
            *graphByteSize = 8;
            if(data != nullptr && requestedSize >= 8)
            {
                std::memset(data, 0xBB, 8);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data;
    auto serResult = graph.serialize(data);
    ASSERT_TRUE(serResult.is_good()) << serResult.get_message();
    ASSERT_EQ(data.size(), 8u);

    // Verify handle-less deserialize forwards correct data to backend
    std::vector<uint8_t> capturedData;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&capturedData](hipdnnBackendDescriptor_t*,
                                  const uint8_t* serializedGraph,
                                  size_t graphByteSize) {
            capturedData.assign(serializedGraph, serializedGraph + graphByteSize);
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    Graph graph2;
    auto result = graph2.deserialize(data);

    EXPECT_EQ(capturedData, data);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, DeserializeStringHandleLess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Serialize the graph to JSON string
    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"test": true})";
                auto len = std::strlen(fakeJson) + 1;
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    std::string jsonData;
    auto serResult = graph.serialize(jsonData);
    ASSERT_TRUE(serResult.is_good()) << serResult.get_message();
    ASSERT_FALSE(jsonData.empty());

    // Verify handle-less deserialize forwards correct data to backend
    std::string capturedJson;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillOnce([&capturedJson](
                      hipdnnBackendDescriptor_t*, const char* jsonGraph, size_t jsonByteSize) {
            capturedJson = std::string(jsonGraph, jsonByteSize);
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    Graph graph2;
    auto result = graph2.deserialize(jsonData);

    EXPECT_EQ(capturedJson, jsonData);
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, DeserializeHandleLessThenBuild)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Handle-less deserialize fails because backend rejects the data
    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04};
    auto desResult = graph.deserialize(data);
    EXPECT_TRUE(desResult.is_bad());

    // The graph should still be buildable via the normal path
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();
}

TEST_F(TestGraph, DeserializeBinaryHandleLessSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock the backend create-and-deserialize to succeed with a fake descriptor.
    // The subsequent unpackGraphDescriptor call will fail because backendGetAttribute
    // is not fully mocked for graph unpacking, but this exercises the success path
    // of the deserialization backend call itself.
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    auto result = graph.deserialize(data);
    // Deserialization itself succeeds, but unpacking the graph descriptor fails
    // because backendGetAttribute is not fully mocked for the graph structure.
    EXPECT_TRUE(result.is_bad());
    EXPECT_NE(result.get_message().find("operation"), std::string::npos);
}

TEST_F(TestGraph, DeserializeStringHandleLessSuccess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t* desc, const char*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::string jsonData = R"({"graph": "test"})";
    auto result = graph.deserialize(jsonData);
    // Deserialization itself succeeds, but unpacking the graph descriptor fails
    // because backendGetAttribute is not fully mocked for the graph structure.
    EXPECT_TRUE(result.is_bad());
    EXPECT_NE(result.get_message().find("operation"), std::string::npos);
}

// ============================================================================
// nlohmann::json serialize/deserialize tests
// ============================================================================

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

#include <nlohmann/json.hpp>

TEST_F(TestGraph, SerializeToJsonObject)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Auto-lower needs the serialize backend call
    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"operations": [{"type": "batchnorm"}]})";
                auto len = std::strlen(fakeJson) + 1;
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    // Auto-lower (non-const serialize), then serialize to json object
    nlohmann::json j;
    auto result = graph.serialize(j);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_FALSE(j.empty());
    EXPECT_TRUE(j.contains("operations"));
}

TEST_F(TestGraph, SerializeToJsonObjectAutoLower)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char* data) {
                const char* fakeJson = R"({"graph": "auto-lowered"})";
                auto len = std::strlen(fakeJson) + 1;
                *graphByteSize = len;
                if(data != nullptr && requestedSize >= len)
                {
                    hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJson, len);
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    // Graph is not lowered — non-const serialize(nlohmann::json&) should auto-lower
    nlohmann::json j;
    auto result = graph.serialize(j);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_FALSE(j.empty());
    EXPECT_TRUE(j.contains("graph"));
    EXPECT_EQ(j["graph"], "auto-lowered");
}

TEST_F(TestGraph, DeserializeFromJsonObjectHandleLess)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    // Verify handle-less deserialize from nlohmann::json forwards data to backend
    std::string capturedJson;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillOnce([&capturedJson](
                      hipdnnBackendDescriptor_t*, const char* jsonGraph, size_t jsonByteSize) {
            capturedJson = std::string(jsonGraph, jsonByteSize);
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    const nlohmann::json j = {{"operations", nlohmann::json::array()}};
    auto result = graph.deserialize(j);

    // The nlohmann::json is serialized via dump() before forwarding
    EXPECT_EQ(capturedJson, j.dump());
    EXPECT_TRUE(result.is_bad());
}

TEST_F(TestGraph, DeserializeFromJsonObjectWithHandle)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    // Verify with-handle deserialize from nlohmann::json forwards data to backend
    std::string capturedJson;

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillOnce([&capturedJson](
                      hipdnnBackendDescriptor_t*, const char* jsonGraph, size_t jsonByteSize) {
            capturedJson = std::string(jsonGraph, jsonByteSize);
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    const nlohmann::json j = {{"graph", "test"}, {"operations", nlohmann::json::array()}};
    auto result = graph.deserialize(_handle, j);

    EXPECT_EQ(capturedJson, j.dump());
    EXPECT_TRUE(result.is_bad());
}

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB

// ============================================================================
// Error-path tests for to_binary / to_json
// ============================================================================

TEST_F(TestGraph, ToBinaryPropagatesSerializationSizeQueryError)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // The size-query call (requestedSize==0, data==nullptr) returns an error
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto [data, err] = graph.to_binary();
    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(data.empty());
}

TEST_F(TestGraph, ToBinaryPropagatesSerializationDataError)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Size query succeeds, but the actual data-fetch call fails
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, uint8_t*) {
                *graphByteSize = 16;
                if(requestedSize > 0)
                {
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    auto [data, err] = graph.to_binary();
    EXPECT_TRUE(err.is_bad());
}

TEST_F(TestGraph, ToJsonPropagatesSerializationSizeQueryError)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    auto [jsonData, err] = graph.to_json();
    EXPECT_TRUE(err.is_bad());
    EXPECT_TRUE(jsonData.empty());
}

TEST_F(TestGraph, ToJsonPropagatesSerializationDataError)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault(
            [](hipdnnBackendDescriptor_t, size_t requestedSize, size_t* graphByteSize, char*) {
                *graphByteSize = 32;
                if(requestedSize > 0)
                {
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                }
                return HIPDNN_STATUS_SUCCESS;
            });

    auto [jsonData, err] = graph.to_json();
    EXPECT_TRUE(err.is_bad());
}

// ============================================================================
// Round-trip serialization tests
// ============================================================================

// Helper: create a conv fprop + relu fusion graph (2 nodes, linear chain)
static void createConvReluFusionGraph(Graph& graph)
{
    graph.set_name("ConvReluFusionGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1})
        .set_data_type(DataType::FLOAT);

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("ConvFwd");
    convAttrs.set_pre_padding({1, 1});
    convAttrs.set_post_padding({1, 1});
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});

    auto convOut = graph.conv_fprop(x, w, convAttrs);

    PointwiseAttributes reluAttrs;
    reluAttrs.set_name("Relu");
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph.pointwise(convOut, reluAttrs);
    y->set_output(true).set_uid(3);
}

// Helper: create a pointwise chain graph (2 nodes: unary RELU -> binary ADD)
static void createPointwiseChainGraph(Graph& graph)
{
    graph.set_name("PointwiseChainGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({2, 4, 16, 16})
        .set_stride({1024, 256, 16, 1})
        .set_data_type(DataType::FLOAT);

    auto b = std::make_shared<TensorAttributes>();
    b->set_uid(2)
        .set_name("B")
        .set_dim({2, 4, 16, 16})
        .set_stride({1024, 256, 16, 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes reluAttrs;
    reluAttrs.set_name("Relu");
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto activated = graph.pointwise(x, reluAttrs);

    PointwiseAttributes addAttrs;
    addAttrs.set_name("Add");
    addAttrs.set_mode(PointwiseMode::ADD);

    auto y = graph.pointwise(activated, b, addAttrs);
    y->set_output(true).set_uid(3);
}

// Helper: create a diamond graph (4 nodes: BN -> 2x RELU fan-out -> ADD fan-in)
static void createDiamondGraph(Graph& graph)
{
    graph.set_name("DiamondGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 8, 8})
        .set_stride({128, 64, 8, 1})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes bnAttrs;
    bnAttrs.set_name("BN");

    auto bnOut = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs);

    PointwiseAttributes relu1Attrs;
    relu1Attrs.set_name("Relu1");
    relu1Attrs.set_mode(PointwiseMode::RELU_FWD);
    auto branch1 = graph.pointwise(bnOut, relu1Attrs);

    PointwiseAttributes relu2Attrs;
    relu2Attrs.set_name("Relu2");
    relu2Attrs.set_mode(PointwiseMode::RELU_FWD);
    auto branch2 = graph.pointwise(bnOut, relu2Attrs);

    PointwiseAttributes addAttrs;
    addAttrs.set_name("Add");
    addAttrs.set_mode(PointwiseMode::ADD);

    auto y = graph.pointwise(branch1, branch2, addAttrs);
    y->set_output(true).set_uid(6);
}

// Type to hold a graph builder function and a descriptive name
struct GraphTopologyParam
{
    std::string name;
    std::function<void(Graph&)> builder;

    friend std::ostream& operator<<(std::ostream& os, const GraphTopologyParam& p)
    {
        return os << p.name;
    }
};

using FrontendGraphFactory = hipdnn_test_sdk::utilities::FrontendGraphFactory;

class TestGraphSerializationRoundTrip : public TestGraph,
                                        public ::testing::WithParamInterface<GraphTopologyParam>
{
};

TEST_P(TestGraphSerializationRoundTrip, BinarySerializeDeserializeForwardsData)
{
    ::testing::FLAGS_gmock_verbose = "error";
    const auto& param = GetParam();

    // Build the graph
    Graph graph;
    param.builder(graph);

    // Mock binary serialization to return deterministic fake data
    const std::vector<uint8_t> fakeSerializedData = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE};

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&fakeSerializedData](hipdnnBackendDescriptor_t,
                                             size_t requestedSize,
                                             size_t* graphByteSize,
                                             uint8_t* data) {
            *graphByteSize = fakeSerializedData.size();
            if(data != nullptr && requestedSize >= fakeSerializedData.size())
            {
                std::memcpy(data, fakeSerializedData.data(), fakeSerializedData.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Serialize
    auto [data, serErr] = graph.to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    ASSERT_FALSE(data.empty());
    EXPECT_EQ(data, fakeSerializedData);

    // Mock deserialization to capture what gets forwarded
    std::vector<uint8_t> capturedData;

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([&capturedData](hipdnnBackendDescriptor_t*,
                                       const uint8_t* serializedGraph,
                                       size_t graphByteSize) {
            capturedData.assign(serializedGraph, serializedGraph + graphByteSize);
            // Return error so we don't need to mock the full unpack chain
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    // Deserialize via from_binary
    Graph graph2;
    auto desErr = graph2.from_binary(_handle, data);

    // The data should be forwarded correctly to the backend even if unpack fails
    EXPECT_EQ(capturedData, fakeSerializedData);
}

TEST_P(TestGraphSerializationRoundTrip, JsonSerializeDeserializeForwardsData)
{
    ::testing::FLAGS_gmock_verbose = "error";
    const auto& param = GetParam();

    // Build the graph
    Graph graph;
    param.builder(graph);

    // Mock JSON serialization to return fake JSON
    const char* fakeJsonStr = R"({"topology": "test", "name": "round-trip"})";

    ON_CALL(*_mockBackend, backendGetSerializedJsonGraphExt(_, _, _, _))
        .WillByDefault([fakeJsonStr](hipdnnBackendDescriptor_t,
                                     size_t requestedSize,
                                     size_t* graphByteSize,
                                     char* data) {
            auto len = std::strlen(fakeJsonStr) + 1;
            *graphByteSize = len;
            if(data != nullptr && requestedSize >= len)
            {
                hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(data, fakeJsonStr, len);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Serialize
    auto [jsonData, serErr] = graph.to_json();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    ASSERT_FALSE(jsonData.empty());
    EXPECT_EQ(jsonData, fakeJsonStr);

    // Mock deserialization to capture what gets forwarded
    std::string capturedJson;

    ON_CALL(*_mockBackend, backendCreateAndDeserializeJsonGraphExt(_, _, _))
        .WillByDefault([&capturedJson](
                           hipdnnBackendDescriptor_t*, const char* jsonGraph, size_t jsonByteSize) {
            capturedJson = std::string(jsonGraph, jsonByteSize);
            return HIPDNN_STATUS_INTERNAL_ERROR;
        });

    // Deserialize via from_json
    Graph graph2;
    auto desErr = graph2.from_json(_handle, jsonData);

    // The data should be forwarded correctly to the backend even if unpack fails
    EXPECT_EQ(capturedJson, jsonData);
}

// This needs to be a helper function rather than being defined inline so that we can use preprocessor directives in it.
// Embedding a directive within a macro has undefined behaviour.
static std::vector<GraphTopologyParam> getGraphTopologyParams()
{
    std::vector<GraphTopologyParam> params = {
        // Single-node topologies via FrontendGraphFactory
        {"BatchnormInference",
         [](Graph& g) { g = FrontendGraphFactory::createBatchnormInferenceGraph(); }},
        {"BatchnormTraining",
         [](Graph& g) { g = FrontendGraphFactory::createBatchnormTrainingGraph(); }},
        {"BatchnormBackward",
         [](Graph& g) { g = FrontendGraphFactory::createBatchnormBackwardGraph(); }},
        {"ConvForward", [](Graph& g) { g = FrontendGraphFactory::createConvForwardGraph(); }},
        {"ConvBackwardData",
         [](Graph& g) { g = FrontendGraphFactory::createConvBackwardDataGraph(); }},
        {"ConvBackwardWeights",
         [](Graph& g) { g = FrontendGraphFactory::createConvBackwardWeightsGraph(); }},
        {"PointwiseUnary", [](Graph& g) { g = FrontendGraphFactory::createPointwiseUnaryGraph(); }},
        {"PointwiseBinary",
         [](Graph& g) { g = FrontendGraphFactory::createPointwiseBinaryGraph(); }},
        {"Matmul", [](Graph& g) { g = FrontendGraphFactory::createMatmulGraph(); }},
        {"Layernorm", [](Graph& g) { g = FrontendGraphFactory::createLayernormGraph(); }},
        {"Rmsnorm", [](Graph& g) { g = FrontendGraphFactory::createRmsnormGraph(); }},
        {"Reduction", [](Graph& g) { g = FrontendGraphFactory::createReductionGraph(); }},
#ifdef HIPDNN_ENABLE_SDPA
        {"SdpaForward", [](Graph& g) { g = FrontendGraphFactory::createSdpaForwardGraph(); }},
        {"SdpaBackward", [](Graph& g) { g = FrontendGraphFactory::createSdpaBackwardGraph(); }},
#endif
        // Multi-node topologies
        {"ConvFwdBiasActiv",
         [](Graph& g) { g = FrontendGraphFactory::createConvFwdBiasActivGraph(); }},
        {"ConvReluFusion", createConvReluFusionGraph},
        {"PointwiseChain", createPointwiseChainGraph},
        {"Diamond", createDiamondGraph}};

    return params;
}

// NOLINTNEXTLINE(cert-err58-cpp)
INSTANTIATE_TEST_SUITE_P(GraphTopologies,
                         TestGraphSerializationRoundTrip,
                         ::testing::ValuesIn(getGraphTopologyParams()),
                         [](const ::testing::TestParamInfo<GraphTopologyParam>& info) {
                             return info.param.name;
                         });

// --- CompiledPlan Infrastructure Tests ---

// Verify that accessor helpers return nullptr / throw when no plans exist.
TEST_F(TestGraph, ActiveAccessorsReturnNullptrWhenEmpty)
{
    hipdnn_frontend::GraphTestUtils graph;
    EXPECT_EQ(graph.getCompiledPlansCount(), 0u);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_EQ(graph.getActiveEngineConfigPtr(), nullptr);
    EXPECT_EQ(graph.getActiveExecutionPlanPtr(), nullptr);
}

// Verify that the single-plan path (create_execution_plans) produces
// a vector of size 1 with the active index at 0.
TEST_F(TestGraph, SinglePlanPathProducesVectorOfSizeOne)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    // elementCount query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto execPlanResult = graph.create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_TRUE(execPlanResult.is_good());

    // Verify vector model invariants
    EXPECT_EQ(graph.getCompiledPlansCount(), 1u);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_NE(graph.getActiveEngineConfigPtr(), nullptr);
    EXPECT_NE(graph.getActiveExecutionPlanPtr(), nullptr);
}

// Verify create_execution_plan_ext() uses compilePlanFromSpec and produces
// a vector of size 1.
TEST_F(TestGraph, CreateExecutionPlanExtProducesVectorOfSizeOne)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    const int64_t engineId = 42;

    // Mock: get_knob_lookup_for_engine - return empty knobs
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xAA01);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillRepeatedly(
            [&engineDesc2](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc2;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Knob query returns 0 knobs
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            engineDesc2, HIPDNN_ATTR_ENGINE_KNOB_INFO, HIPDNN_TYPE_BACKEND_DESCRIPTOR, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            if(elementCount)
            {
                *elementCount = 0;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Engine config creation
    auto engineCfgDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xBB01);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce(
            [&engineCfgDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineCfgDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Execution plan creation
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xCC01);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce(
            [&execPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = execPlanDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    auto result = graph.create_execution_plan_ext(engineId, {});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // Verify vector model invariants after create_execution_plan_ext
    EXPECT_EQ(graph.getCompiledPlansCount(), 1u);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_NE(graph.getActiveEngineConfigPtr(), nullptr);
    EXPECT_NE(graph.getActiveExecutionPlanPtr(), nullptr);
}

// Verify that compilePlanFromSpec produces a valid CompiledPlan without
// modifying the Graph's _compiledPlans vector.
TEST_F(TestGraph, CompilePlanFromSpecProducesValidPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    graph.build_operation_graph(_handle);

    const int64_t engineId = 99;

    // Mock: engine descriptor for get_knob_lookup
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0xAA02);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillRepeatedly(
            [&engineDesc2](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc2;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Knob query returns 0 knobs
    EXPECT_CALL(
        *_mockBackend,
        backendGetAttribute(
            engineDesc2, HIPDNN_ATTR_ENGINE_KNOB_INFO, HIPDNN_TYPE_BACKEND_DESCRIPTOR, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            if(elementCount)
            {
                *elementCount = 0;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Engine config creation for compilePlanFromSpec
    auto engineCfgDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xBB02);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce(
            [&engineCfgDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineCfgDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Execution plan creation for compilePlanFromSpec
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xCC02);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce(
            [&execPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = execPlanDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // The Graph's compiled plans should be untouched
    EXPECT_EQ(graph.getCompiledPlansCount(), 0u);

    hipdnn_frontend::GraphTestUtils::CompiledPlan plan;
    auto result = graph.compilePlanFromSpec(engineId, {}, plan);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // Verify the compiled plan has valid descriptors and metadata
    EXPECT_NE(plan.engineConfigDesc, nullptr);
    EXPECT_NE(plan.executionPlanDesc, nullptr);
    EXPECT_EQ(plan.engineId, engineId);
    EXPECT_TRUE(plan.knobSettings.empty());
    EXPECT_EQ(plan.workspaceSize, -1); // Not yet queried

    // The Graph's compiled plans should still be empty - compilePlanFromSpec
    // does not modify Graph state.
    EXPECT_EQ(graph.getCompiledPlansCount(), 0u);
}

// Verify that deserialized compiled plan clears the vector and sets up
// a plan with no engine config (only execution plan).
TEST_F(TestGraph, DeserializeCompiledPlanUsesVectorModel)
{
    hipdnn_frontend::GraphTestUtils graph;

    // Set up mock for deserialize
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xDD01);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _))
        .WillOnce(
            [&execPlanDesc](
                hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = execPlanDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    const std::vector<uint8_t> fakeData = {0x01, 0x02};
    auto result = graph.deserialize_compiled_plan(_handle, fakeData);
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // Verify vector model
    EXPECT_EQ(graph.getCompiledPlansCount(), 1u);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_NE(graph.getActiveExecutionPlanPtr(), nullptr);
    // Engine config is null for deserialized plans
    EXPECT_EQ(graph.getActiveEngineConfigPtr(), nullptr);
}

// ---------------------------------------------------------------------------
// Mutual Exclusion Guard Tests
// ---------------------------------------------------------------------------

TEST_F(TestGraph, CreateExecutionPlansRejectsAfterAddEngine)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyPlanSpec();
    auto result = graph.create_execution_plans({HeuristicMode::FALLBACK});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, BuildRejectsAfterAddEngine)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyPlanSpec();
    auto result = graph.build(_handle);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, AddEngineConfigsRejectsAfterCreateExecutionPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    const std::vector<EngineConfigInfo> configs = {{42, "", {}, false, 0}};
    auto result = graph.add_engine_configs(configs);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, AddAllEnginesRejectsAfterCreateExecutionPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    auto result = graph.add_all_engines();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

// ---------------------------------------------------------------------------
// Plan-Indexed Access Tests
// ---------------------------------------------------------------------------

TEST_F(TestGraph, GetExecutionPlanCountReturnsZeroWhenEmpty)
{
    const hipdnn_frontend::GraphTestUtils graph;
    EXPECT_EQ(graph.get_execution_plan_count(), 0);
}

TEST_F(TestGraph, GetExecutionPlanCountAfterInject)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    graph.injectDummyCompiledPlan();
    EXPECT_EQ(graph.get_execution_plan_count(), 2);
}

TEST_F(TestGraph, GetWorkspaceSizePlanAtIndexOutOfBounds)
{
    const hipdnn_frontend::GraphTestUtils graph;
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(0), -1);
    EXPECT_EQ(graph.get_workspace_size_plan_at_index(-1), -1);
}

TEST_F(TestGraph, GetPlanNameAtIndexValid)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    std::string name;
    auto result = graph.get_plan_name_at_index(0, name);
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(name, "0x0");
}

TEST_F(TestGraph, GetPlanNameAtIndexOutOfBounds)
{
    const hipdnn_frontend::GraphTestUtils graph;
    std::string name;
    auto result = graph.get_plan_name_at_index(0, name);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, ExecutePlanAtIndexOutOfBounds)
{
    const hipdnn_frontend::GraphTestUtils graph;
    const std::unordered_map<int64_t, void*> variantPack;
    auto result = graph.execute_plan_at_index(_handle, variantPack, nullptr, 0);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, ExecutePlanAtIndexNegativeIndex)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    const std::unordered_map<int64_t, void*> variantPack;
    auto result = graph.execute_plan_at_index(_handle, variantPack, nullptr, -1);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, ExecutePlanAtIndexUncompiledPlan)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    const std::unordered_map<int64_t, void*> variantPack;
    auto result = graph.execute_plan_at_index(_handle, variantPack, nullptr, 0);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, GetAutotuneWorkspaceSizeReturnsZeroWhenEmpty)
{
    const hipdnn_frontend::GraphTestUtils graph;
    EXPECT_EQ(graph.get_autotune_workspace_size(), 0);
}

TEST_F(TestGraph, BuildPlanAtIndexOutOfBounds)
{
    hipdnn_frontend::GraphTestUtils graph;
    auto result = graph.build_plan_at_index(0);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

// --------------------------------------------------------------------------
// deselect_workspace_greater_than tests
// --------------------------------------------------------------------------

TEST_F(TestGraph, DeselectWorkspacePlanSpecPath)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 512);
    graph.injectPlanSpec(2, 1024);
    graph.injectPlanSpec(3, 2048);

    auto& ref = graph.deselect_workspace_greater_than(1024);
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getPlanSpecsCount(), 3u);
    EXPECT_EQ(graph.getMaxWorkspaceAllowed(), 1024);
}

TEST_F(TestGraph, DeselectWorkspaceCompiledPlanPath)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(1, 512);
    graph.injectCompiledPlan(2, 1024);
    graph.injectCompiledPlan(3, 2048);

    auto& ref = graph.deselect_workspace_greater_than(1024);
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getCompiledPlansCount(), 3u);
    EXPECT_EQ(graph.getMaxWorkspaceAllowed(), 1024);
}

TEST_F(TestGraph, DeselectWorkspaceRemovesAll)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 100);
    graph.injectPlanSpec(2, 200);

    graph.deselect_workspace_greater_than(50);
    EXPECT_EQ(graph.getPlanSpecsCount(), 2u);
    EXPECT_EQ(graph.getMaxWorkspaceAllowed(), 50);
}

TEST_F(TestGraph, DeselectWorkspaceRemovesNone)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 100);
    graph.injectPlanSpec(2, 200);

    graph.deselect_workspace_greater_than(300);
    EXPECT_EQ(graph.getPlanSpecsCount(), 2u);
}

TEST_F(TestGraph, DeselectWorkspaceNoOp)
{
    hipdnn_frontend::GraphTestUtils graph;
    auto& ref = graph.deselect_workspace_greater_than(100);
    EXPECT_EQ(&ref, &graph);
}

TEST_F(TestGraph, DeselectWorkspacePreservesUnqueried)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(1, -1); // workspaceSize not yet queried
    graph.injectCompiledPlan(2, 200);

    graph.deselect_workspace_greater_than(100);
    EXPECT_EQ(graph.getCompiledPlansCount(), 2u);
    EXPECT_EQ(graph.getMaxWorkspaceAllowed(), 100);
}

// --------------------------------------------------------------------------
// deselect_engines tests
// --------------------------------------------------------------------------

TEST_F(TestGraph, DeselectEnginesPlanSpecPath)
{
    using hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_ID;
    using hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(MIOPEN_ENGINE_ID, 0);
    graph.injectPlanSpec(HIPBLASLT_ENGINE_ID, 0);
    graph.injectPlanSpec(HIP_MLOPS_ENGINE_ID, 0);

    auto& ref = graph.deselect_engines({"MIOPEN_ENGINE"});
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getPlanSpecsCount(), 3u);
    auto barredIds = graph.getBarredEngineIds();
    EXPECT_EQ(barredIds.count(MIOPEN_ENGINE_ID), 1u);
}

TEST_F(TestGraph, DeselectEnginesCompiledPlanPath)
{
    using hipdnn_data_sdk::utilities::HIP_MLOPS_ENGINE_ID;
    using hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(MIOPEN_ENGINE_ID, 0);
    graph.injectCompiledPlan(HIPBLASLT_ENGINE_ID, 0);
    graph.injectCompiledPlan(HIP_MLOPS_ENGINE_ID, 0);

    graph.deselect_engines({"MIOPEN_ENGINE"});
    EXPECT_EQ(graph.getCompiledPlansCount(), 3u);
    auto barredIds = graph.getBarredEngineIds();
    EXPECT_EQ(barredIds.count(MIOPEN_ENGINE_ID), 1u);
}

TEST_F(TestGraph, DeselectEnginesUnknownNameSkipped)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 0);
    graph.injectPlanSpec(2, 0);

    graph.deselect_engines({"nonexistent_engine_xyz"});
    EXPECT_EQ(graph.getPlanSpecsCount(), 2u);
}

TEST_F(TestGraph, DeselectEnginesEmptyList)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 0);

    auto& ref = graph.deselect_engines(std::vector<std::string>{});
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getPlanSpecsCount(), 1u);
}

TEST_F(TestGraph, DeselectEnginesNoOp)
{
    hipdnn_frontend::GraphTestUtils graph;
    auto& ref = graph.deselect_engines({"MIOPEN_ENGINE"});
    EXPECT_EQ(&ref, &graph);
}

// ---------------------------------------------------------------------------
// Knob Validation at add_engine_*() Time
// ---------------------------------------------------------------------------

// Helper to mock knob info query that can be called multiple times.
// Uses WillRepeatedly instead of WillOnce, because add_engine() queries
// knobs for both engine validation and strict knob validation.
static void mockKnobInfoQueryRepeated(
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
    hipdnnBackendDescriptor_t engineDesc,
    const std::vector<hipdnnBackendDescriptor_t>& fakeKnobDescs)
{
    auto count = static_cast<int64_t>(fakeKnobDescs.size());

    EXPECT_CALL(*mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillRepeatedly([count](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* elementCount,
                                void*) {
            *elementCount = count;
            return HIPDNN_STATUS_SUCCESS;
        });

    if(count > 0)
    {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(engineDesc,
                                        HIPDNN_ATTR_ENGINE_KNOB_INFO,
                                        HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                        count,
                                        _,
                                        NotNull()))
            .WillRepeatedly([fakeKnobDescs](hipdnnBackendDescriptor_t,
                                            hipdnnBackendAttributeName_t,
                                            hipdnnBackendAttributeType_t,
                                            int64_t,
                                            int64_t* elementCount,
                                            void* out) {
                *elementCount = static_cast<int64_t>(fakeKnobDescs.size());
                auto* arr = static_cast<hipdnnBackendDescriptor_t*>(out);
                for(size_t i = 0; i < fakeKnobDescs.size(); ++i)
                {
                    arr[i] = fakeKnobDescs[i];
                }
                return HIPDNN_STATUS_SUCCESS;
            });
    }
}

// Helper to set up a knob descriptor mock with int range constraints.
// Similar to setupKnobDescriptorMock but uses WillRepeatedly and supports
// optional min/max constraints for integer knobs.
static void setupKnobDescriptorMockRepeated(
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
    hipdnnBackendDescriptor_t knobDesc,
    const std::string& knobId,
    const std::string& description,
    bool deprecated,
    hipdnnBackendAttributeType_t valueType,
    const KnobValueVariant& defaultValue,
    std::optional<int64_t> minValue = std::nullopt,
    std::optional<int64_t> maxValue = std::nullopt)
{
    // Mock string attribute (size-query + data-read), repeatable
    auto mockString = [&](hipdnnBackendAttributeName_t attrName, const std::string& value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_CHAR, 0, _, nullptr))
            .WillRepeatedly(DoAll(SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                                  Return(HIPDNN_STATUS_SUCCESS)));

        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc,
                                        attrName,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(value.size() + 1),
                                        _,
                                        NotNull()))
            .WillRepeatedly(DoAll(
                SetArgPointee<4>(static_cast<int64_t>(value.size() + 1)),
                Invoke([value](hipdnnBackendDescriptor_t,
                               hipdnnBackendAttributeName_t,
                               hipdnnBackendAttributeType_t,
                               int64_t,
                               int64_t*,
                               void* out) { std::memcpy(out, value.c_str(), value.size() + 1); }),
                Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockScalarInt64 = [&](hipdnnBackendAttributeName_t attrName, int64_t value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_INT64, 1, _, NotNull()))
            .WillRepeatedly(
                DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                     hipdnnBackendAttributeName_t,
                                     hipdnnBackendAttributeType_t,
                                     int64_t,
                                     int64_t*,
                                     void* out) { *static_cast<int64_t*>(out) = value; }),
                      Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockScalarBool = [&](hipdnnBackendAttributeName_t attrName, bool value) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_BOOLEAN, 1, _, NotNull()))
            .WillRepeatedly(DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                                 hipdnnBackendAttributeName_t,
                                                 hipdnnBackendAttributeType_t,
                                                 int64_t,
                                                 int64_t*,
                                                 void* out) { *static_cast<bool*>(out) = value; }),
                                  Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockOptAbsent = [&](hipdnnBackendAttributeName_t attrName,
                             hipdnnBackendAttributeType_t attrType) {
        EXPECT_CALL(*mockBackend, backendGetAttribute(knobDesc, attrName, attrType, 0, _, nullptr))
            .WillRepeatedly(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockOptPresent = [&](hipdnnBackendAttributeName_t attrName,
                              hipdnnBackendAttributeType_t attrType,
                              int64_t value) {
        // Count query returns 1 (present)
        EXPECT_CALL(*mockBackend, backendGetAttribute(knobDesc, attrName, attrType, 0, _, nullptr))
            .WillRepeatedly(DoAll(SetArgPointee<4>(1), Return(HIPDNN_STATUS_SUCCESS)));
        // Data read returns the value
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, attrType, 1, _, NotNull()))
            .WillRepeatedly(
                DoAll(Invoke([value](hipdnnBackendDescriptor_t,
                                     hipdnnBackendAttributeName_t,
                                     hipdnnBackendAttributeType_t,
                                     int64_t,
                                     int64_t*,
                                     void* out) { *static_cast<int64_t*>(out) = value; }),
                      Return(HIPDNN_STATUS_SUCCESS)));
    };

    auto mockEmptyVec = [&](hipdnnBackendAttributeName_t attrName) {
        EXPECT_CALL(*mockBackend,
                    backendGetAttribute(knobDesc, attrName, HIPDNN_TYPE_INT64, 0, _, nullptr))
            .WillRepeatedly(DoAll(SetArgPointee<4>(0), Return(HIPDNN_STATUS_SUCCESS)));
    };

    // Knob ID (string)
    mockString(HIPDNN_ATTR_KNOB_INFO_TYPE, knobId);

    // Description (string)
    mockString(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION, description);

    // Deprecated flag (bool)
    mockScalarBool(HIPDNN_ATTR_KNOB_INFO_DEPRECATED, deprecated);

    // Default value type
    mockScalarInt64(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE, static_cast<int64_t>(valueType));

    // Default value and constraints (only int64 supported in this helper)
    if(valueType == HIPDNN_TYPE_INT64)
    {
        auto intVal = std::get<int64_t>(defaultValue);
        mockScalarInt64(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE, intVal);

        if(minValue.has_value())
        {
            mockOptPresent(
                HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64, minValue.value());
        }
        else
        {
            mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, HIPDNN_TYPE_INT64);
        }

        if(maxValue.has_value())
        {
            mockOptPresent(
                HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64, maxValue.value());
        }
        else
        {
            mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, HIPDNN_TYPE_INT64);
        }

        mockOptAbsent(HIPDNN_ATTR_KNOB_INFO_STRIDE, HIPDNN_TYPE_INT64);
        mockEmptyVec(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT);
    }
}

// Helper to build a graph, mock engine descriptor creation for repeated use
// (e.g., add_engine() which creates engine descriptors multiple times).
static hipdnnBackendDescriptor_t buildGraphAndMockEngineRepeated(
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>>& mockBackend,
    Graph& graph,
    hipdnnHandle_t handle)
{
    createBasicBatchnormGraph(graph);
    EXPECT_TRUE(graph.validate().is_good());

    EXPECT_CALL(*mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*mockBackend, backendFinalize(_)).Times(AnyNumber());
    EXPECT_CALL(*mockBackend, backendGetAttribute(_, _, _, _, _, _)).Times(AnyNumber());

    auto buildResult = graph.build_operation_graph(handle);
    EXPECT_TRUE(buildResult.is_good());

    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xE001);
    EXPECT_CALL(*mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*mockBackend, backendFinalize(engineDesc))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    return engineDesc;
}

TEST_F(TestGraph, AddEngineRejectsInvalidKnobName)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});

    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    const int64_t engineId = 42;
    auto result = graph.add_engine(engineId, {KnobSetting("nonexistent_knob", int64_t{1})});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("nonexistent_knob"), std::string::npos)
        << "Error message should mention the invalid knob name: " << result.err_msg;
}

TEST_F(TestGraph, AddEngineRejectsOutOfRangeKnobValue)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});

    // Set up knob "tile_size" with range [1, 512]
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}},
                                    int64_t{1},
                                    int64_t{512});

    const int64_t engineId = 42;
    auto result = graph.add_engine(engineId, {KnobSetting("tile_size", int64_t{9999})});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("tile_size"), std::string::npos)
        << "Error message should mention the knob name: " << result.err_msg;
}

TEST_F(TestGraph, AddEngineAcceptsValidKnobSettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});

    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    const int64_t engineId = 42;
    auto result = graph.add_engine(engineId, {KnobSetting("tile_size", int64_t{256})});
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

// add_engine() derives PlanSpec::supportsExhaustive from the validation knobs.
// An engine exposing the benchmarking knob yields a spec with the flag set.
TEST_F(TestGraph, AddEnginePropagatesSupportsExhaustiveWhenBenchmarkingKnobPresent)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto benchKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {benchKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    benchKnobDesc,
                                    hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME,
                                    "Benchmarking knob",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{0}});

    ASSERT_TRUE(graph.add_engine(42, {}).is_good());

    const auto flags = graph.getPlanSpecSupportsExhaustive();
    ASSERT_EQ(flags.size(), 1u);
    EXPECT_TRUE(flags[0]);
}

// An engine that exposes only a non-benchmarking knob yields a spec with the
// flag cleared, matching the absence of the benchmarking knob.
TEST_F(TestGraph, AddEnginePropagatesSupportsExhaustiveFalseWithoutBenchmarkingKnob)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    ASSERT_TRUE(graph.add_engine(42, {KnobSetting("tile_size", int64_t{256})}).is_good());

    const auto flags = graph.getPlanSpecSupportsExhaustive();
    ASSERT_EQ(flags.size(), 1u);
    EXPECT_FALSE(flags[0]);
}

TEST_F(TestGraph, AddEngineSweepRejectsInvalidAxisKnob)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});

    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    const int64_t engineId = 42;
    KnobSweepAxis axis;
    axis.knobId = "nonexistent_knob";
    axis.values = {KnobValueVariant{int64_t{1}}};

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = engineId;
    sweepSpec.axes = {axis};

    auto result = graph.add_engine_sweep({sweepSpec});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("nonexistent_knob"), std::string::npos)
        << "Error message should mention the invalid knob name: " << result.err_msg;
}

// An empty-values sweep axis is dropped (knob takes its engine default) rather
// than eliminating the engine's entire sweep. The surviving axis still produces
// its product; the valueless knob is absent from every produced spec.
TEST_F(TestGraph, AddEngineSweepDropsEmptyValuesAxisAndKeepsOthers)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto tileKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    auto warpsKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA002);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {tileKnobDesc, warpsKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    tileKnobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{32}});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    warpsKnobDesc,
                                    "warps",
                                    "Warp count",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{4}});

    KnobSweepAxis tileAxis;
    tileAxis.knobId = "tile_size";
    tileAxis.values = {KnobValueVariant{int64_t{32}}, KnobValueVariant{int64_t{64}}};

    KnobSweepAxis warpsAxis;
    warpsAxis.knobId = "warps";
    warpsAxis.values = {}; // no values: dropped, warps takes engine default

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 42;
    sweepSpec.axes = {tileAxis, warpsAxis};

    auto result = graph.add_engine_sweep({sweepSpec});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    // One spec per tile value; the engine's sweep is NOT eliminated.
    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 2u);
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> expected
        = {{{"tile_size", KnobValueVariant{int64_t{32}}}},
           {{"tile_size", KnobValueVariant{int64_t{64}}}}};
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> produced(settings.begin(),
                                                                                settings.end());
    EXPECT_EQ(produced, expected)
        << "Specs must sweep tile_size only; the empty-values warps axis is dropped";
}

// When the only axis has empty values, the engine sweep falls back to a single
// fixedSettings-only spec instead of being eliminated.
TEST_F(TestGraph, AddEngineSweepSoleEmptyValuesAxisYieldsFixedSettingsSpec)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto warpsKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    auto tileKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA002);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {warpsKnobDesc, tileKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    warpsKnobDesc,
                                    "warps",
                                    "Warp count",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{4}});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    tileKnobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{32}});

    KnobSweepAxis warpsAxis;
    warpsAxis.knobId = "warps";
    warpsAxis.values = {}; // no values: dropped

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 42;
    sweepSpec.axes = {warpsAxis};
    sweepSpec.fixedSettings = {{"tile_size", KnobValueVariant{int64_t{32}}}};

    auto result = graph.add_engine_sweep({sweepSpec});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 1u);
    const std::set<std::pair<std::string, KnobValueVariant>> expected
        = {{"tile_size", KnobValueVariant{int64_t{32}}}};
    EXPECT_EQ(settings[0], expected)
        << "A sole empty-values axis must yield one fixedSettings-only spec";
}

// When a knob appears in both a (dropped) empty-values axis and fixedSettings,
// the fixedSettings value wins because the dropped axis contributes nothing.
TEST_F(TestGraph, AddEngineSweepEmptyValuesAxisFixedSettingsConflictResolvesToFixed)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto warpsKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {warpsKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    warpsKnobDesc,
                                    "warps",
                                    "Warp count",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{4}});

    KnobSweepAxis warpsAxis;
    warpsAxis.knobId = "warps";
    warpsAxis.values = {}; // no values: dropped

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 42;
    sweepSpec.axes = {warpsAxis};
    sweepSpec.fixedSettings = {{"warps", KnobValueVariant{int64_t{8}}}};

    auto result = graph.add_engine_sweep({sweepSpec});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 1u);
    const std::set<std::pair<std::string, KnobValueVariant>> expected
        = {{"warps", KnobValueVariant{int64_t{8}}}};
    EXPECT_EQ(settings[0], expected)
        << "fixedSettings value must win when its knob also has an empty-values axis";
}

TEST_F(TestGraph, AddEngineSweepStripsBenchmarkingKnobFromAxis)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto tileKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {tileKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    tileKnobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{32}});

    KnobSweepAxis tileAxis;
    tileAxis.knobId = "tile_size";
    tileAxis.values = {KnobValueVariant{int64_t{32}}, KnobValueVariant{int64_t{64}}};

    KnobSweepAxis benchmarkingAxis;
    benchmarkingAxis.knobId = autotune::detail::BENCHMARKING_KNOB_NAME;
    benchmarkingAxis.values = {KnobValueVariant{int64_t{1}}};

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 42;
    sweepSpec.axes = {tileAxis, benchmarkingAxis};

    auto result = graph.add_engine_sweep({sweepSpec});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 2u);
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> expected
        = {{{"tile_size", KnobValueVariant{int64_t{32}}}},
           {{"tile_size", KnobValueVariant{int64_t{64}}}}};
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> produced(settings.begin(),
                                                                                settings.end());
    EXPECT_EQ(produced, expected)
        << "Benchmarking knob axis must be stripped; only tile_size is swept";
    for(const auto& spec : settings)
    {
        EXPECT_EQ(spec.count({std::string(autotune::detail::BENCHMARKING_KNOB_NAME),
                              KnobValueVariant{int64_t{1}}}),
                  0u)
            << "No produced spec may contain the benchmarking knob";
    }
}

TEST_F(TestGraph, AddEngineSweepStripsBenchmarkingKnobFromFixedSettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto warpsKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    auto tileKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA002);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {warpsKnobDesc, tileKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    warpsKnobDesc,
                                    "warps",
                                    "Warp count",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{4}});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    tileKnobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{32}});

    KnobSweepAxis warpsAxis;
    warpsAxis.knobId = "warps";
    warpsAxis.values = {KnobValueVariant{int64_t{4}}, KnobValueVariant{int64_t{8}}};

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 42;
    sweepSpec.axes = {warpsAxis};
    sweepSpec.fixedSettings
        = {{autotune::detail::BENCHMARKING_KNOB_NAME, KnobValueVariant{int64_t{1}}},
           {"tile_size", KnobValueVariant{int64_t{32}}}};

    auto result = graph.add_engine_sweep({sweepSpec});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 2u);
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> expected
        = {{{"tile_size", KnobValueVariant{int64_t{32}}}, {"warps", KnobValueVariant{int64_t{4}}}},
           {{"tile_size", KnobValueVariant{int64_t{32}}}, {"warps", KnobValueVariant{int64_t{8}}}}};
    const std::set<std::set<std::pair<std::string, KnobValueVariant>>> produced(settings.begin(),
                                                                                settings.end());
    EXPECT_EQ(produced, expected)
        << "Benchmarking knob must be stripped from fixed settings; tile_size survives";
    for(const auto& spec : settings)
    {
        EXPECT_EQ(spec.count({std::string(autotune::detail::BENCHMARKING_KNOB_NAME),
                              KnobValueVariant{int64_t{1}}}),
                  0u)
            << "No produced spec may contain the benchmarking knob";
    }
}

// --------------------------------------------------------------------------
// get_workspace_size_plan_at_index error-returning overload tests
// --------------------------------------------------------------------------

TEST_F(TestGraph, GetWorkspaceSizePlanAtIndexErrorOverloadValid)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(42, 2048);
    graph.injectCompiledPlan(99, 4096);

    int64_t size = -1;
    auto result = graph.get_workspace_size_plan_at_index(0, size);
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(size, 2048);

    result = graph.get_workspace_size_plan_at_index(1, size);
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(size, 4096);
}

TEST_F(TestGraph, GetWorkspaceSizePlanAtIndexErrorOverloadOutOfBounds)
{
    const hipdnn_frontend::GraphTestUtils graph;
    int64_t size = -1;

    auto result = graph.get_workspace_size_plan_at_index(0, size);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);

    result = graph.get_workspace_size_plan_at_index(-1, size);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, GetWorkspaceSizePlanAtIndexErrorOverloadZeroWorkspace)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(1, 0);

    int64_t size = -1;
    auto result = graph.get_workspace_size_plan_at_index(0, size);
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(size, 0); // Valid: workspace size is 0, distinguishable from error
}

// --------------------------------------------------------------------------
// WindowSize / stabilityThreshold conditional validation tests (Task 15.7)
// --------------------------------------------------------------------------

TEST_F(TestGraph, WindowSizeValidationOnlyForRunUntilStable)
{
    hipdnn_frontend::GraphTestUtils graph;
    const std::unordered_map<int64_t, void*> emptyPack;

    // With FIXED_AVERAGE strategy, windowSize=1 should NOT trigger validation error.
    // It will fail later at upfront parameter validation (null handle), not at windowSize.
    {
        AutotuneConfig config;
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
        config.windowSize = 1; // Would fail if validation were unconditional

        auto result = graph.autotune(nullptr, emptyPack, nullptr, config);
        // Should fail at a later check (null handle), not at windowSize
        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
            << "Expected windowSize validation to be skipped for FIXED_AVERAGE, "
               "but got: "
            << result.err_msg;
        EXPECT_NE(result.err_msg.find("handle"), std::string::npos)
            << "Expected null-handle error, but got: " << result.err_msg;
    }

    // With RUN_UNTIL_STABLE strategy, windowSize=1 should trigger validation error.
    {
        AutotuneConfig config;
        config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
        config.windowSize = 1;

        auto result = graph.autotune(nullptr, emptyPack, nullptr, config);
        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
        EXPECT_NE(result.err_msg.find("windowSize"), std::string::npos)
            << "Error message should mention windowSize: " << result.err_msg;
    }
}

// ---------------------------------------------------------------------------
// Final Conformance and Cleanup
// ---------------------------------------------------------------------------

TEST_F(TestGraph, AddEngineConfigsRejectsBeforeGraphBuild)
{
    Graph graph;
    createBasicBatchnormGraph(graph);
    // Graph has NOT been built - hasReadyGraphDesc() is false

    const std::vector<EngineConfigInfo> configs = {{42, "", {}, false, 0}};
    const auto result = graph.add_engine_configs(configs);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("Graph has not been built"), std::string::npos)
        << "Error message should mention graph not built: " << result.err_msg;
}

TEST_F(TestGraph, AddEngineVariantsRejectsBeforeGraphBuild)
{
    Graph graph;
    createBasicBatchnormGraph(graph);
    // Graph has NOT been built - hasReadyGraphDesc() is false

    const std::vector<EngineVariant> variants = {{42, {}}};
    const auto result = graph.add_engine_variants(variants);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("Graph has not been built"), std::string::npos)
        << "Error message should mention graph not built: " << result.err_msg;
}

TEST_F(TestGraph, AddEngineSweepRejectsBeforeGraphBuild)
{
    Graph graph;
    createBasicBatchnormGraph(graph);
    // Graph has NOT been built - hasReadyGraphDesc() is false

    const std::vector<EngineSweepSpec> specs = {{42, {}, {}}};
    const auto result = graph.add_engine_sweep(specs);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("Graph has not been built"), std::string::npos)
        << "Error message should mention graph not built: " << result.err_msg;
}

TEST_F(TestGraph, AddEngineSweepSkipsUnsupportedEngine)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    EXPECT_TRUE(graph.validate().is_good());

    // Allow all descriptor setup calls during build_operation_graph()
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _)).Times(AnyNumber());

    const auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After building the graph, make engine descriptor finalize fail.
    // This causes get_knobs_for_engine() to fail for any engine, simulating
    // an unsupported engine.  The last-added EXPECT_CALL takes priority
    // over the generic backendFinalize(_) set above.
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xE001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    EngineSweepSpec sweepSpec;
    sweepSpec.engineId = 99;

    const auto result = graph.add_engine_sweep({sweepSpec});
    EXPECT_TRUE(result.is_good())
        << "add_engine_sweep() should skip unsupported engines (batch semantics), "
           "but got error: "
        << result.err_msg;
}

TEST_F(TestGraph, AddEngineVariantsSkipsUnsupportedEngine)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);
    EXPECT_TRUE(graph.validate().is_good());

    // Allow all descriptor setup calls during build_operation_graph()
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _)).Times(AnyNumber());

    const auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After building the graph, make engine descriptor finalize fail.
    // This causes get_knobs_for_engine() to fail for any engine, simulating
    // an unsupported engine.  The last-added EXPECT_CALL takes priority
    // over the generic backendFinalize(_) set above.
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xE001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    const std::vector<EngineVariant> variants = {{99, {}}};
    const auto result = graph.add_engine_variants(variants);
    EXPECT_TRUE(result.is_good())
        << "add_engine_variants() should skip unsupported engines (batch semantics), "
           "but got error: "
        << result.err_msg;
}

TEST_F(TestGraph, StabilityThresholdValidationOnlyForRunUntilStable)
{
    hipdnn_frontend::GraphTestUtils graph;
    const std::unordered_map<int64_t, void*> emptyPack;

    // With FIXED_AVERAGE strategy, invalid stabilityThreshold should NOT trigger error
    {
        AutotuneConfig config;
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
        config.timedIterations = 1;
        config.stabilityThreshold = 0.0f; // Would fail if validation were unconditional

        auto result = graph.autotune(nullptr, emptyPack, nullptr, config);
        // Should fail at a later check (null handle), not at stabilityThreshold
        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
            << "Expected stabilityThreshold validation to be skipped for FIXED_AVERAGE, "
               "but got: "
            << result.err_msg;
        EXPECT_NE(result.err_msg.find("handle"), std::string::npos)
            << "Expected null-handle error, but got: " << result.err_msg;
    }

    // With RUN_UNTIL_STABLE strategy, invalid stabilityThreshold should trigger error
    {
        AutotuneConfig config;
        config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
        config.stabilityThreshold = 0.0f;

        auto result = graph.autotune(nullptr, emptyPack, nullptr, config);
        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
        EXPECT_NE(result.err_msg.find("stabilityThreshold"), std::string::npos)
            << "Error message should mention stabilityThreshold: " << result.err_msg;
    }
}

TEST_F(TestGraph, AutotuneRejectsNullHandle)
{
    hipdnn_frontend::GraphTestUtils graph;
    const std::unordered_map<int64_t, void*> pack = {{0, reinterpret_cast<void*>(0x1)}};
    const auto result = graph.autotune(nullptr, pack, nullptr);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("handle"), std::string::npos)
        << "Error message should mention handle: " << result.err_msg;
}

TEST_F(TestGraph, AutotuneRejectsEmptyVariantPack)
{
    hipdnn_frontend::GraphTestUtils graph;
    const std::unordered_map<int64_t, void*> emptyPack;
    const auto result = graph.autotune(reinterpret_cast<hipdnnHandle_t>(0x1), emptyPack, nullptr);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("variantPack"), std::string::npos)
        << "Error message should mention variantPack: " << result.err_msg;
}

// ============================================================================
// Autotune UID Validation Tests
//
// These tests verify that autotune() checks the variantPack for all
// required non-virtual tensor UIDs after hasReadyGraphDesc() passes.
// The batchnorm graph has 5 non-virtual tensors (UIDs 1-5) and one
// virtual output tensor (auto-assigned UID, skipped by validation).
// ============================================================================

TEST_F(TestGraph, AutotuneRejectsMissingTensorUids)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);

    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << "build_operation_graph failed: " << buildResult.err_msg;

    graph.injectDummyPlanSpec();

    // Provide variantPack with only UID 1 (missing UIDs 2-5).
    // Use the general overload (with workspaceSize) because plan specs are present.
    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)}};
    const auto result = graph.autotune(_handle, pack, nullptr, int64_t{0});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("missing"), std::string::npos)
        << "Error message should mention missing UIDs: " << result.err_msg;
    // Verify specific missing UIDs are listed (sorted: 2, 3, 4, 5)
    EXPECT_NE(result.err_msg.find('2'), std::string::npos)
        << "Error message should list missing UID 2: " << result.err_msg;
    EXPECT_NE(result.err_msg.find('5'), std::string::npos)
        << "Error message should list missing UID 5: " << result.err_msg;
}

TEST_F(TestGraph, AutotuneAcceptsCompleteVariantPack)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);

    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << "build_operation_graph failed: " << buildResult.err_msg;

    graph.injectDummyPlanSpec();

    // Provide variantPack with all required UIDs (1-5).
    // Use the general overload (with workspaceSize) because plan specs are present.
    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};
    const auto result = graph.autotune(_handle, pack, nullptr, int64_t{0});
    // Should pass UID validation and fail at a later point (e.g., compilation).
    // The key assertion is that it does NOT fail with "missing" in the error message.
    if(result.is_bad())
    {
        EXPECT_EQ(result.err_msg.find("missing"), std::string::npos)
            << "Should not fail at UID validation: " << result.err_msg;
    }
}

TEST_F(TestGraph, AutotuneAcceptsExtraUidsInVariantPack)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);

    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << "build_operation_graph failed: " << buildResult.err_msg;

    graph.injectDummyPlanSpec();

    // Provide variantPack with all required UIDs (1-5) AND an extra UID 99.
    // Use the general overload (with workspaceSize) because plan specs are present.
    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)},
                                                     {99, reinterpret_cast<void*>(0x99)}};
    const auto result = graph.autotune(_handle, pack, nullptr, int64_t{0});
    // Extra UIDs should be silently ignored; should not fail at UID validation
    if(result.is_bad())
    {
        EXPECT_EQ(result.err_msg.find("missing"), std::string::npos)
            << "Extra UIDs should not cause UID validation failure: " << result.err_msg;
    }
}

// ============================================================================
// add_engines() batch method tests
// ============================================================================

TEST_F(TestGraph, AddEnginesCreatesMultiplePlanSpecs)
{
    using hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);

    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // Mock engine knob queries so add_engine() validation passes
    ON_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINE_KNOB_INFO, _, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t* elementCount,
                          void*) {
            if(elementCount)
            {
                *elementCount = 0;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.add_engines({MIOPEN_ENGINE_ID, HIPBLASLT_ENGINE_ID});
    EXPECT_TRUE(result.is_good()) << result.err_msg;
    EXPECT_EQ(graph.getPlanSpecsCount(), 2u);
}

TEST_F(TestGraph, AddEnginesReturnsErrorOnInvalidId)
{
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);

    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // Let the engine knob query fail for all engines to simulate invalid IDs
    ON_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINE_KNOB_INFO, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    auto result = graph.add_engines({99999});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

TEST_F(TestGraph, AddEnginesMutualExclusionGuard)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyCompiledPlan();
    auto result = graph.add_engines({1, 2});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("add_engines()"), std::string::npos)
        << "Error message should mention add_engines(): " << result.err_msg;
}

TEST_F(TestGraph, AddEnginesEmptyInputReturnsError)
{
    hipdnn_frontend::GraphTestUtils graph;
    auto result = graph.add_engines({});
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

// ============================================================================
// deselect_engines() by ID tests
// ============================================================================

TEST_F(TestGraph, DeselectEnginesByIdRemovesFromPlanSpecs)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(10, 0);
    graph.injectPlanSpec(20, 0);
    graph.injectPlanSpec(30, 0);

    auto& ref = graph.deselect_engines(std::vector<int64_t>{10, 30});
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getPlanSpecsCount(), 3u);
    auto barredIds = graph.getBarredEngineIds();
    EXPECT_EQ(barredIds, (std::unordered_set<int64_t>{10, 30}));
}

TEST_F(TestGraph, DeselectEnginesByIdRemovesFromCompiledPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(10, 0);
    graph.injectCompiledPlan(20, 0);
    graph.injectCompiledPlan(30, 0);

    graph.deselect_engines(std::vector<int64_t>{20});
    EXPECT_EQ(graph.getCompiledPlansCount(), 3u);
    auto barredIds = graph.getBarredEngineIds();
    EXPECT_EQ(barredIds, (std::unordered_set<int64_t>{20}));
}

TEST_F(TestGraph, DeselectEnginesByIdEmptyVector)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 0);

    auto& ref = graph.deselect_engines(std::vector<int64_t>{});
    EXPECT_EQ(&ref, &graph);
    EXPECT_EQ(graph.getPlanSpecsCount(), 1u);
}

// ============================================================================
// cuDNN-compatible autotune overload tests
// ============================================================================

TEST_F(TestGraph, Tier3AutotuneRequiresCompiledPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    // No compiled plans injected - should fail with "No autotuning candidates"
    std::unordered_map<int64_t, void*> pack = {{0, reinterpret_cast<void*>(0x1)}};
    auto result = graph.autotune(_handle, pack, nullptr, nullptr);
    EXPECT_TRUE(result.is_bad());
}

// ============================================================================
// Compiled-Plan-Only Guard Tests
//
// RFC 6.3: The compiled-plan autotune overloads (no workspaceSize parameter) are
// compiled-plan path only and must return an error if plan specs exist.
// ============================================================================

TEST_F(TestGraph, AutotuneTier1RejectsPlanSpecs)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyPlanSpec();

    const std::unordered_map<int64_t, void*> emptyPack;
    const AutotuneConfig config;

    auto result = graph.autotune(nullptr, emptyPack, nullptr, config);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("compiled-plan path only"), std::string::npos)
        << "Error should mention compiled-plan path: " << result.err_msg;
}

TEST_F(TestGraph, AutotuneTier1AcceptsCompiledPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectCompiledPlan(42, 1024);

    const std::unordered_map<int64_t, void*> dummyPack = {{1, reinterpret_cast<void*>(0x1)}};
    const AutotuneConfig config;

    auto result = graph.autotune(nullptr, dummyPack, nullptr, config);
    // Should fail at handle validation, NOT at the plan-spec guard
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("handle"), std::string::npos)
        << "Expected null-handle error (past the guard), got: " << result.err_msg;
}

TEST_F(TestGraph, AutotuneTier2AcceptsPlanSpecs)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyPlanSpec();

    const std::unordered_map<int64_t, void*> dummyPack = {{1, reinterpret_cast<void*>(0x1)}};
    const AutotuneConfig config;

    // The general overload accepts workspaceSize
    auto result = graph.autotune(nullptr, dummyPack, nullptr, int64_t{1024}, config);
    // Should fail at handle validation, NOT at plan-spec guard
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("handle"), std::string::npos)
        << "Expected null-handle error (no guard on Tier 2), got: " << result.err_msg;
}

TEST_F(TestGraph, AutotuneTier2RejectsNegativeWorkspaceSize)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectDummyPlanSpec();

    const std::unordered_map<int64_t, void*> dummyPack = {{1, reinterpret_cast<void*>(0x1)}};

    // Negative workspaceSize must be rejected at the general-overload entry point
    auto result = graph.autotune(nullptr, dummyPack, nullptr, int64_t{-1});
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("workspaceSize"), std::string::npos)
        << "Error should mention workspaceSize: " << result.err_msg;
}

// ============================================================================
// build_plans() guards empty _compiledPlans
// ============================================================================

TEST_F(TestGraph, BuildPlansHeuristicsChoiceRejectsEmptyCompiledPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    // No create_execution_plans() - _compiledPlans is empty.
    auto result = graph.build_plans(BuildPlanPolicy::HEURISTICS_CHOICE);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("create_execution_plans()"), std::string::npos)
        << "Error should mention create_execution_plans(): " << result.err_msg;
}

TEST_F(TestGraph, BuildPlansAllRejectsEmptyCompiledPlans)
{
    hipdnn_frontend::GraphTestUtils graph;
    auto result = graph.build_plans(BuildPlanPolicy::ALL);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
}

// ============================================================================
// All-plans-exceed-workspace and all-barred early returns (host, no GPU)
//
// Both early-returns fire before the benchmark loop, so no real GPU op runs;
// the mocked backend (TestGraph fixture) is sufficient.
// ============================================================================

TEST_F(TestGraph, AutotuneAllPlansExceedWorkspace)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // Compiled-plan path: two plans whose cached workspace exceeds the
    // requested runtime ceiling. maxWorkspaceSize-skipped plans are NOT pushed
    // into compiledPlanMap, so the empty-map early-return fires.
    graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/4096, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/20, /*workspaceSize=*/8192, /*barred=*/false);

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    std::vector<AutotuneResult> results;
    auto result
        = graph.autotune(_handle, pack, nullptr, int64_t{1024}, AutotuneConfig{}, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("workspace limit"), std::string::npos) << result.err_msg;
    ASSERT_EQ(results.size(), 2u);
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.succeeded);
        EXPECT_EQ(r.rank, -1);
        EXPECT_EQ(r.compiledPlanIndex, -1);
        EXPECT_GT(r.workspaceSize, 1024);
    }
}

TEST_F(TestGraph, AutotuneAllPlansBarredReturnsInvalidValueAndKeepsIndices)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // Compiled-plan path: all candidates are barred (e.g. by deselect_engines()).
    // Barred plans stay in _compiledPlans for cuDNN index-stability; the
    // all-barred check fires instead of the workspace early-return.
    const int candidateCount = 2;
    graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/512, /*barred=*/true);
    graph.injectValidCompiledPlan(/*engineId=*/20, /*workspaceSize=*/512, /*barred=*/true);

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, nullptr, AutotuneConfig{}, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("deselected"), std::string::npos) << result.err_msg;

    // *results populated with a skipped entry per barred plan.
    ASSERT_EQ(results.size(), static_cast<size_t>(candidateCount));
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.succeeded);
        EXPECT_EQ(r.rank, -1);
        EXPECT_EQ(r.compiledPlanIndex, -1);
    }

    // Index-stability: plans remain, and a barred index returns "is barred"
    // (NOT out-of-bounds).
    EXPECT_EQ(graph.get_execution_plan_count(), candidateCount);
    auto execResult = graph.execute_plan_at_index(_handle, pack, nullptr, 0);
    EXPECT_EQ(execResult.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(execResult.err_msg.find("is barred"), std::string::npos) << execResult.err_msg;
}

// A compiled plan that is both barred and exceeds maxWorkspaceSize must be
// surfaced exactly once - by the barred loop (carrying the "Plan barred"
// message), not also by the workspace-skip loop. A second non-barred,
// in-workspace plan keeps the compiled-plan map non-empty so the benchmark and
// result-building path (which contains the dedup) runs via the mock backend.
TEST_F(TestGraph, AutotuneBarredAndOversizedPlanAddedOnce)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // One benchmarkable plan (non-barred, within the limit) keeps the plan map
    // non-empty so the benchmark + result-building path runs.
    graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/512, /*barred=*/false);
    // The plan under test: barred (e.g. by deselect_engines()) AND over the
    // limit. It satisfies both the workspace-skip and barred conditions.
    const int64_t barredEngineId = 20;
    graph.injectValidCompiledPlan(
        /*engineId=*/barredEngineId, /*workspaceSize=*/8192, /*barred=*/true);

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    std::vector<AutotuneResult> results;
    void* workspace = reinterpret_cast<void*>(0x1000);
    auto result
        = graph.autotune(_handle, pack, workspace, int64_t{1024}, AutotuneConfig{}, {}, &results);

    ASSERT_TRUE(result.is_good()) << result.err_msg;

    // The barred + oversized plan must appear EXACTLY once, carrying the barred
    // message ("Plan barred"), not the workspace-skip message ("exceeds limit").
    int barredEntryCount = 0;
    for(const auto& r : results)
    {
        if(r.engineId == barredEngineId)
        {
            ++barredEntryCount;
            EXPECT_FALSE(r.succeeded);
            EXPECT_EQ(r.rank, -1);
            EXPECT_EQ(r.compiledPlanIndex, -1);
            EXPECT_NE(r.errorMessage.find("Plan barred"), std::string::npos) << r.errorMessage;
            EXPECT_EQ(r.errorMessage.find("exceeds limit"), std::string::npos) << r.errorMessage;
        }
    }
    EXPECT_EQ(barredEntryCount, 1);
}

TEST_F(TestGraph, AutotuneMixedBarredAndWorkspaceCeilingReportsCompositionAndSurfacesBoth)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    auto buildResult = graph.build_operation_graph(_handle);
    ASSERT_TRUE(buildResult.is_good()) << buildResult.err_msg;

    // This single test exercises all four compiled-plan non-benchmarked
    // dispositions at once: finalize-failed, barred, engineIdFilter-excluded,
    // and workspace-skipped. The compiled-plan filter loop checks them in that
    // order (finalize -> barred -> filter -> workspace), each via `continue`.

    // One plan with null descriptors -> fails the descriptor-validity check
    // (finalize-failed). injectCompiledPlan (no valid descriptors) is the hook.
    const int64_t finalizeFailEngineId = 5;
    graph.injectCompiledPlan(/*engineId=*/finalizeFailEngineId, /*workspaceSize=*/512);

    // Three plans barred via deselect (workspace within the ceiling so the only
    // reason each is non-benchmarkable is the bar).
    const std::vector<int64_t> barredEngineIds = {10, 11, 12};
    for(const int64_t engineId : barredEngineIds)
    {
        graph.injectValidCompiledPlan(/*engineId=*/engineId,
                                      /*workspaceSize=*/512,
                                      /*barred=*/true);
    }
    // One un-barred, in-workspace plan excluded by engineIdFilter (it is the one
    // valid plan left out of the filter set below).
    const int64_t filteredEngineId = 30;
    graph.injectValidCompiledPlan(/*engineId=*/filteredEngineId,
                                  /*workspaceSize=*/512,
                                  /*barred=*/false);
    // Two un-barred, valid plans whose workspace exceeds the runtime ceiling.
    const std::vector<int64_t> skippedEngineIds = {20, 21};
    for(const int64_t engineId : skippedEngineIds)
    {
        graph.injectValidCompiledPlan(/*engineId=*/engineId,
                                      /*workspaceSize=*/8192,
                                      /*barred=*/false);
    }

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    // engineIdFilter admits the barred and workspace-skipped engines so each
    // reaches its own disposition; filteredEngineId is deliberately omitted so
    // it surfaces as engineIdFilter-excluded. The finalize-failed plan is
    // rejected before the filter check, so it need not be listed.
    AutotuneConfig config;
    config.engineIdFilter = {10, 11, 12, 20, 21};

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, nullptr, int64_t{1024}, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("3 deselected"), std::string::npos) << result.err_msg;
    EXPECT_NE(result.err_msg.find("2 over the workspace limit"), std::string::npos)
        << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 failed to finalize"), std::string::npos) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 excluded by engineIdFilter"), std::string::npos)
        << result.err_msg;

    // *results carries every non-benchmarkable plan with its own reason.
    ASSERT_EQ(results.size(),
              barredEngineIds.size() + skippedEngineIds.size() + 2); // +finalize-fail +filtered
    int sawBarred = 0;
    int sawSkipped = 0;
    int sawFinalizeFailed = 0;
    int sawFiltered = 0;
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.succeeded);
        EXPECT_EQ(r.rank, -1);
        EXPECT_EQ(r.compiledPlanIndex, -1);
        // Priming never runs on the compiled-plan path, so every disposition here
        // reports ranExhaustive=false with no priming reason.
        EXPECT_FALSE(r.ranExhaustive);
        EXPECT_TRUE(r.exhaustiveNotRunReason.empty()) << r.exhaustiveNotRunReason;
        if(r.engineId == finalizeFailEngineId)
        {
            ++sawFinalizeFailed;
            EXPECT_NE(r.errorMessage.find("failed to finalize"), std::string::npos)
                << r.errorMessage;
        }
        else if(std::find(barredEngineIds.begin(), barredEngineIds.end(), r.engineId)
                != barredEngineIds.end())
        {
            ++sawBarred;
            EXPECT_NE(r.errorMessage.find("Plan barred"), std::string::npos) << r.errorMessage;
        }
        else if(r.engineId == filteredEngineId)
        {
            ++sawFiltered;
            EXPECT_NE(r.errorMessage.find("engineIdFilter"), std::string::npos) << r.errorMessage;
        }
        else if(std::find(skippedEngineIds.begin(), skippedEngineIds.end(), r.engineId)
                != skippedEngineIds.end())
        {
            ++sawSkipped;
            EXPECT_NE(r.errorMessage.find("exceeds limit"), std::string::npos) << r.errorMessage;
        }
    }
    EXPECT_EQ(sawFinalizeFailed, 1);
    EXPECT_EQ(sawBarred, static_cast<int>(barredEngineIds.size()));
    EXPECT_EQ(sawFiltered, 1);
    EXPECT_EQ(sawSkipped, static_cast<int>(skippedEngineIds.size()));
}

TEST_F(TestGraph, AutotunePlanSpecFinalizeFailuresReportedInCompositionSummary)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    // Built operation graph plus AnyNumber create/set/finalize/get default mocks
    // so the plan-spec compile and engine-config finalize succeed by default.
    buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    // Spec 0 and spec 1 fail to finalize; spec 2 finalizes but its compiled
    // workspace exceeds the runtime ceiling.
    const int64_t finalizeFailEngineIdA = 51;
    const int64_t finalizeFailEngineIdB = 52;
    const int64_t workspaceSkipEngineId = 53;
    const int64_t maxWorkspaceSize = 1024;
    const int64_t skippedCompiledWorkspace = 8192; // over the ceiling

    graph.injectPlanSpec(finalizeFailEngineIdA, /*workspaceSize=*/256);
    graph.injectPlanSpec(finalizeFailEngineIdB, /*workspaceSize=*/256);
    // Exhaustive-capable so the workspace-skipped result carries the capability;
    // STANDARD mode means no priming, so ranExhaustive stays false.
    graph.injectPlanSpec(workspaceSkipEngineId, /*workspaceSize=*/256, /*supportsExhaustive=*/true);

    // Hand out a distinct execution-plan descriptor handle per creation, in
    // compile order. The first two belong to the finalize-failure specs, the
    // third to the workspace-skip spec.
    auto execPlanDescA = reinterpret_cast<hipdnnBackendDescriptor_t>(0xEF01);
    auto execPlanDescB = reinterpret_cast<hipdnnBackendDescriptor_t>(0xEF02);
    auto execPlanDescC = reinterpret_cast<hipdnnBackendDescriptor_t>(0xEF03);
    auto execPlanHandles = std::make_shared<std::vector<hipdnnBackendDescriptor_t>>(
        std::vector<hipdnnBackendDescriptor_t>{execPlanDescA, execPlanDescB, execPlanDescC});
    auto execPlanCreateIdx = std::make_shared<size_t>(0);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly([execPlanHandles, execPlanCreateIdx](
                            hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
            const size_t i = *execPlanCreateIdx;
            // Reuse the last handle if creations exceed the planned count; the
            // test only relies on the first three being distinct.
            *descriptor = (*execPlanHandles)[std::min(i, execPlanHandles->size() - 1)];
            ++(*execPlanCreateIdx);
            return HIPDNN_STATUS_SUCCESS;
        });

    // Fail the execution-plan finalize for the two finalize-failure specs.
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDescA))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDescB))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    // The workspace-skip spec finalizes successfully.
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDescC))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // The surviving (finalized) plan's compiled workspace exceeds the ceiling, so
    // it is workspace-skipped. The workspace-size query is only reached for the
    // plan that finalizes (spec 2), so a single return value suffices.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = skippedCompiledWorkspace;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    const AutotuneConfig config; // STANDARD mode: no EXHAUSTIVE priming
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, workspace, maxWorkspaceSize, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("0 deselected"), std::string::npos) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 over the workspace limit"), std::string::npos)
        << result.err_msg;
    EXPECT_NE(result.err_msg.find("2 failed to finalize"), std::string::npos) << result.err_msg;

    // *results carries every excluded plan with its own reason: two
    // finalize-failed entries and one workspace-skipped entry.
    ASSERT_EQ(results.size(), 3u);
    int sawFinalizeFailed = 0;
    int sawWorkspaceSkipped = 0;
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.succeeded);
        EXPECT_EQ(r.rank, -1);
        if(r.engineId == finalizeFailEngineIdA || r.engineId == finalizeFailEngineIdB)
        {
            ++sawFinalizeFailed;
            EXPECT_NE(r.errorMessage.find("finalize"), std::string::npos) << r.errorMessage;
            // 2-arg injectPlanSpec (no capability) + STANDARD mode (no priming).
            EXPECT_FALSE(r.supportsExhaustive);
            EXPECT_FALSE(r.ranExhaustive);
            EXPECT_TRUE(r.exhaustiveNotRunReason.empty());
        }
        else if(r.engineId == workspaceSkipEngineId)
        {
            ++sawWorkspaceSkipped;
            EXPECT_NE(r.errorMessage.find("exceeds limit"), std::string::npos) << r.errorMessage;
            // Capability carried on this post-priming skip; STANDARD mode, no priming.
            EXPECT_TRUE(r.supportsExhaustive);
            EXPECT_FALSE(r.ranExhaustive);
            EXPECT_TRUE(r.exhaustiveNotRunReason.empty());
        }
    }
    EXPECT_EQ(sawFinalizeFailed, 2);
    EXPECT_EQ(sawWorkspaceSkipped, 1);
}

// Plan-spec path, EXHAUSTIVE + BENCHMARK_UNPRIMED: a spec that supports exhaustive
// priming fails its priming finalize (recording an exhaustiveNotRunReason) and then
// fails its real finalize. The finalize-failed result must carry the exhaustive
// outcome (supportsExhaustive, ranExhaustive, exhaustiveNotRunReason) alongside the
// finalize errorMessage, so the priming outcome is not lost. Execution-plan finalize
// is failed for the engine, which fails both the priming finalize and the real one.
TEST_F(TestGraph, AutotunePlanSpecExhaustiveFinalizeFailureRetainsPrimingReason)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t finalizeFailEngineId = 51;

    // supportsExhaustive=true so the priming loop runs (and its finalize fails).
    graph.injectPlanSpec(finalizeFailEngineId, /*workspaceSize=*/256, /*supportsExhaustive=*/true);

    // All execution-plan descriptors get the same sentinel handle; failing its
    // finalize fails both the priming plan's finalize and the real plan's finalize.
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xEF01);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [execPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = execPlanDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDesc))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.primingFailurePolicy = PrimingFailurePolicy::BENCHMARK_UNPRIMED;
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, workspace, int64_t{1024}, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 failed to finalize"), std::string::npos) << result.err_msg;

    ASSERT_EQ(results.size(), 1u);
    const auto& r = results.front();
    EXPECT_EQ(r.engineId, finalizeFailEngineId);
    EXPECT_FALSE(r.succeeded);
    EXPECT_EQ(r.rank, -1);
    EXPECT_EQ(r.compiledPlanIndex, -1);
    EXPECT_NE(r.errorMessage.find("failed finalize"), std::string::npos) << r.errorMessage;
    // The full exhaustive outcome recorded before the real finalize failure is
    // retained: the engine's capability, that priming did not run, and the reason.
    EXPECT_TRUE(r.supportsExhaustive);
    EXPECT_FALSE(r.ranExhaustive);
    EXPECT_NE(r.exhaustiveNotRunReason.find("Priming"), std::string::npos)
        << r.exhaustiveNotRunReason;
}

// Plan-spec path: a spec barred by deselect_engines() (engine-ID) and specs
// barred by deselect_workspace_greater_than() (compiled workspace over the
// deselect threshold) both surface as barredResults, and each carries the
// correct workspace-size pair:
//   - engine-deselected spec: never compiled, so workspaceSize == -1 (the
//     "not applicable" sentinel) and estimatedWorkspaceSize == the spec estimate.
//   - workspace-deselected spec: compiled, so workspaceSize == the real compiled
//     size and estimatedWorkspaceSize == the spec estimate (the two differ).
// A benchmarkable spec keeps the run OK so the barred entries surface in *results
// alongside the winner, and the composition summary reports "3 deselected".
TEST_F(TestGraph, AutotunePlanSpecDeselectBarredResultsRecordBothWorkspaceSizes)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    // Built operation graph plus AnyNumber create/set/finalize/get mocks so the
    // plan-spec compile, engine-config set-attr, and finalize all succeed.
    buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t fitEngineId = 61; // benchmarkable winner
    const int64_t engineDeselectId = 62; // barred by deselect_engines()
    const int64_t workspaceDeselectIdA = 63; // barred by deselect_workspace_greater_than()
    const int64_t workspaceDeselectIdB = 64; // barred by deselect_workspace_greater_than()

    const int64_t fitEstimate = 512;
    const int64_t engineDeselectEstimate = 700;
    const int64_t workspaceDeselectEstimateA = 800;
    const int64_t workspaceDeselectEstimateB = 900;

    const int64_t deselectThreshold = 2000; // deselect_workspace_greater_than()
    const int64_t fitCompiled = 1024; // within the deselect threshold
    const int64_t overCompiled = 8192; // over the deselect threshold

    // Insertion order drives compile order. The engine-deselected spec is
    // removed in the spec-filter loop and never compiled, so the compile-order
    // workspace queries are: fit, then the two workspace-deselect specs.
    graph.injectPlanSpec(fitEngineId, fitEstimate);
    // The barred specs are exhaustive-capable: their capability must surface on the
    // barred results. The engine-deselect bar is a pre-priming site and the
    // workspace-deselect bar is a post-priming site; STANDARD mode here means no
    // priming runs, so ranExhaustive stays false on all of them.
    graph.injectPlanSpec(engineDeselectId, engineDeselectEstimate, /*supportsExhaustive=*/true);
    graph.injectPlanSpec(
        workspaceDeselectIdA, workspaceDeselectEstimateA, /*supportsExhaustive=*/true);
    graph.injectPlanSpec(
        workspaceDeselectIdB, workspaceDeselectEstimateB, /*supportsExhaustive=*/true);

    graph.deselect_engines(std::vector<int64_t>{engineDeselectId});
    graph.deselect_workspace_greater_than(deselectThreshold);

    // Compiled workspace per plan in compile order: fit (within), then the two
    // workspace-deselect specs (over the deselect threshold).
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* out) {
            *static_cast<int64_t*>(out) = fitCompiled;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = overCompiled;
            return HIPDNN_STATUS_SUCCESS;
        });

    // The fitting plan is benchmarked through the mocked backend.
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<float*>(out) = 1.0f;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    const AutotuneConfig config; // STANDARD mode: no EXHAUSTIVE priming
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    // A runtime ceiling well above every compiled size, so only the deselect
    // filters (engine-ID and workspace-greater-than) bar plans. The
    // deselect-workspace check runs before the runtime guard, so the
    // workspace-deselected specs surface as barred (not workspace-skipped).
    const int64_t runtimeCeiling = 100000;
    auto result = graph.autotune(_handle, pack, workspace, runtimeCeiling, config, {}, &results);

    // The fitting plan succeeded, so autotune returns OK.
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Winner plus three barred specs.
    ASSERT_EQ(results.size(), 4u);

    auto findResult = [&](int64_t engineId) -> const AutotuneResult* {
        for(const auto& r : results)
        {
            if(r.engineId == engineId)
            {
                return &r;
            }
        }
        return nullptr;
    };

    const auto* fit = findResult(fitEngineId);
    const auto* engineDeselected = findResult(engineDeselectId);
    const auto* workspaceDeselectedA = findResult(workspaceDeselectIdA);
    const auto* workspaceDeselectedB = findResult(workspaceDeselectIdB);
    ASSERT_NE(fit, nullptr);
    ASSERT_NE(engineDeselected, nullptr);
    ASSERT_NE(workspaceDeselectedA, nullptr);
    ASSERT_NE(workspaceDeselectedB, nullptr);

    // Winner: benchmarked and within the threshold.
    EXPECT_TRUE(fit->succeeded);
    EXPECT_EQ(fit->rank, 0);
    EXPECT_EQ(fit->workspaceSize, fitCompiled);
    // Benchmarked winner in STANDARD mode: no priming, and this spec was injected
    // without exhaustive capability.
    EXPECT_FALSE(fit->supportsExhaustive);
    EXPECT_FALSE(fit->ranExhaustive);
    EXPECT_TRUE(fit->exhaustiveNotRunReason.empty());

    // Engine-deselected: never compiled. workspaceSize is the -1 N/A sentinel,
    // the estimate carries the spec estimate.
    EXPECT_FALSE(engineDeselected->succeeded);
    EXPECT_EQ(engineDeselected->rank, -1);
    EXPECT_EQ(engineDeselected->compiledPlanIndex, -1);
    EXPECT_NE(engineDeselected->errorMessage.find("Plan barred"), std::string::npos)
        << engineDeselected->errorMessage;
    EXPECT_EQ(engineDeselected->workspaceSize, -1);
    EXPECT_EQ(engineDeselected->estimatedWorkspaceSize, engineDeselectEstimate);
    // Capability carried; pre-priming bar, so priming never ran.
    EXPECT_TRUE(engineDeselected->supportsExhaustive);
    EXPECT_FALSE(engineDeselected->ranExhaustive);
    EXPECT_TRUE(engineDeselected->exhaustiveNotRunReason.empty());

    // Workspace-deselected: compiled, so workspaceSize is the real compiled size
    // and the estimate is the spec estimate; the two differ.
    for(const auto* r : {workspaceDeselectedA, workspaceDeselectedB})
    {
        EXPECT_FALSE(r->succeeded);
        EXPECT_EQ(r->rank, -1);
        EXPECT_EQ(r->compiledPlanIndex, -1);
        EXPECT_NE(r->errorMessage.find("Plan barred"), std::string::npos) << r->errorMessage;
        EXPECT_EQ(r->workspaceSize, overCompiled);
        EXPECT_NE(r->estimatedWorkspaceSize, r->workspaceSize);
        // Capability carried on this post-priming bar; STANDARD mode so no priming ran.
        EXPECT_TRUE(r->supportsExhaustive);
        EXPECT_FALSE(r->ranExhaustive);
        EXPECT_TRUE(r->exhaustiveNotRunReason.empty());
    }
    EXPECT_EQ(workspaceDeselectedA->estimatedWorkspaceSize, workspaceDeselectEstimateA);
    EXPECT_EQ(workspaceDeselectedB->estimatedWorkspaceSize, workspaceDeselectEstimateB);
}

// Plan-spec path: a spec whose compilePlanFromSpec fails surfaces as a
// compile-failed result instead of being silently dropped. Compilation is failed
// per-engine by intercepting the engine-descriptor HIPDNN_ATTR_ENGINE_GLOBAL_INDEX
// set-attribute (which carries the engine ID) and returning a backend error for
// the targeted engine. The compile-failed entry carries succeeded==false,
// rank==-1, compiledPlanIndex==-1, a "failed to compile" message, and the -1 N/A
// workspace sentinel; the composition summary reports "1 failed to compile".
TEST_F(TestGraph, AutotunePlanSpecCompileFailuresReportedInCompositionSummary)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t compileFailEngineId = 71;

    graph.injectPlanSpec(compileFailEngineId, /*workspaceSize=*/256);

    // Fail compilePlanFromSpec for the targeted engine by rejecting the engine
    // descriptor global-index set-attribute when its value matches the engine ID.
    // Every other engine ID succeeds. This more-specific expectation overrides the
    // broad GLOBAL_INDEX mock installed by buildGraphAndMockEngineRepeated.
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           const void* value) {
            const int64_t requestedEngineId = *static_cast<const int64_t*>(value);
            if(requestedEngineId == compileFailEngineId)
            {
                return HIPDNN_STATUS_INTERNAL_ERROR;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    const AutotuneConfig config; // STANDARD mode: no EXHAUSTIVE priming
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, workspace, int64_t{1024}, config, {}, &results);

    // The only candidate failed to compile, so no plan is benchmarkable and the
    // composition summary surfaces the compile-failed count.
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 failed to compile"), std::string::npos) << result.err_msg;

    ASSERT_EQ(results.size(), 1u);
    const auto& r = results.front();
    EXPECT_EQ(r.engineId, compileFailEngineId);
    EXPECT_FALSE(r.succeeded);
    EXPECT_EQ(r.rank, -1);
    EXPECT_EQ(r.compiledPlanIndex, -1);
    EXPECT_EQ(r.workspaceSize, -1);
    EXPECT_NE(r.errorMessage.find("failed compile"), std::string::npos) << r.errorMessage;
}

// Plan-spec path, EXHAUSTIVE + BENCHMARK_UNPRIMED: a spec that supports exhaustive
// priming fails its priming compile (recording an exhaustiveNotRunReason) and then
// fails its real compile. The compile-failed result must carry BOTH the compile
// errorMessage AND the earlier priming reason, so the priming outcome is not lost.
// The GLOBAL_INDEX set-attribute interception fails the targeted engine for both
// the priming compile and the real compile.
TEST_F(TestGraph, AutotunePlanSpecExhaustiveCompileFailureRetainsPrimingReason)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t compileFailEngineId = 71;

    // supportsExhaustive=true so the priming loop runs (and fails) for this spec.
    graph.injectPlanSpec(compileFailEngineId, /*workspaceSize=*/256, /*supportsExhaustive=*/true);

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           const void* value) {
            const int64_t requestedEngineId = *static_cast<const int64_t*>(value);
            if(requestedEngineId == compileFailEngineId)
            {
                return HIPDNN_STATUS_INTERNAL_ERROR;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.primingFailurePolicy = PrimingFailurePolicy::BENCHMARK_UNPRIMED;
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, workspace, int64_t{1024}, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 failed to compile"), std::string::npos) << result.err_msg;

    ASSERT_EQ(results.size(), 1u);
    const auto& r = results.front();
    EXPECT_EQ(r.engineId, compileFailEngineId);
    EXPECT_FALSE(r.succeeded);
    EXPECT_EQ(r.rank, -1);
    EXPECT_EQ(r.compiledPlanIndex, -1);
    EXPECT_NE(r.errorMessage.find("failed compile"), std::string::npos) << r.errorMessage;
    // The full exhaustive outcome recorded before the real compile failure is
    // retained: the engine's capability, that priming did not run, and the reason.
    EXPECT_TRUE(r.supportsExhaustive);
    EXPECT_FALSE(r.ranExhaustive);
    EXPECT_NE(r.exhaustiveNotRunReason.find("Priming compile failed"), std::string::npos)
        << r.exhaustiveNotRunReason;
}

// Plan-spec path, EXHAUSTIVE + BENCHMARK_UNPRIMED: an engine's priming SUCCEEDS
// (priming compile + finalize + execute all pass), then its REAL plan compile
// FAILS. The compile-failed result must carry ranExhaustive=true (priming ran),
// supportsExhaustive=true, and an empty exhaustiveNotRunReason. This exercises the
// ranExhaustive=true propagation through makeCompileFailedResult, which the
// priming-failed corners cannot reach.
//
// Discriminator: the priming compile applies the global.benchmarking knob; the real
// compile applies an empty knob set. A per-compile flag is set when the knob-choice
// type set-attribute carries the benchmarking knob id, then consulted at the
// engine-config finalize: priming (flag set) finalizes OK; the real compile (flag
// clear) fails to finalize, which fails the real compile.
TEST_F(TestGraph, AutotunePlanSpecExhaustivePrimingSucceedsRealCompileFailsRetainsRanExhaustive)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    // Engine exposes the benchmarking knob so priming applies it and it survives
    // validateAndFilterKnobSettings.
    auto benchKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {benchKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    benchKnobDesc,
                                    hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME,
                                    "Benchmarking knob",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{0}});

    // add_engine (NOT injectPlanSpec): the engine carries a real benchmarking knob
    // (so priming applies it) and its spec.knobSettings is empty (so the real
    // compile applies no knob and the discriminator flag stays clear).
    ASSERT_TRUE(graph.add_engine(42, {}).is_good());

    // Stable engine-config descriptor handle so the finalize hook can key on it.
    auto engineCfgDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xC001);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .Times(AnyNumber())
        .WillRepeatedly(
            [engineCfgDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineCfgDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    // Per-compile flag: set when this compile applies the benchmarking knob.
    auto sawBenchmarkingKnob = std::make_shared<bool>(false);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .Times(AnyNumber())
        .WillRepeatedly([sawBenchmarkingKnob](hipdnnBackendDescriptor_t,
                                              hipdnnBackendAttributeName_t,
                                              hipdnnBackendAttributeType_t,
                                              int64_t elementCount,
                                              const void* arrayOfElements) {
            const std::string knobId(static_cast<const char*>(arrayOfElements),
                                     static_cast<size_t>(elementCount));
            if(knobId.find(hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME)
               != std::string::npos)
            {
                *sawBenchmarkingKnob = true;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // Engine-config finalize: priming (flag set) passes; real compile (flag clear)
    // fails. Reset the flag each call so the next compile starts clean.
    EXPECT_CALL(*_mockBackend, backendFinalize(engineCfgDesc))
        .Times(AnyNumber())
        .WillRepeatedly([sawBenchmarkingKnob](hipdnnBackendDescriptor_t) {
            const bool priming = *sawBenchmarkingKnob;
            *sawBenchmarkingKnob = false;
            return priming ? HIPDNN_STATUS_SUCCESS : HIPDNN_STATUS_INTERNAL_ERROR;
        });

    // Priming execute succeeds (single backendExecute; no profiling dance here).
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    // Priming compiled workspace fits the budget so priming reaches execute.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = 256;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.primingFailurePolicy = PrimingFailurePolicy::BENCHMARK_UNPRIMED;
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, workspace, int64_t{1024}, config, {}, &results);

    // The engine-config finalize is part of compilePlanFromSpec, so failing it
    // fails the REAL compile (not the later execution-plan finalize): the spec
    // surfaces as a compile-failed result.
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    EXPECT_NE(result.err_msg.find("1 failed to compile"), std::string::npos) << result.err_msg;

    ASSERT_EQ(results.size(), 1u);
    const auto& r = results.front();
    EXPECT_EQ(r.engineId, 42);
    EXPECT_FALSE(r.succeeded);
    EXPECT_EQ(r.rank, -1);
    EXPECT_EQ(r.compiledPlanIndex, -1);
    EXPECT_NE(r.errorMessage.find("failed compile"), std::string::npos) << r.errorMessage;
    // Priming ran successfully, so ranExhaustive is retained as true on the
    // failed result, with no priming reason recorded.
    EXPECT_TRUE(r.supportsExhaustive);
    EXPECT_TRUE(r.ranExhaustive);
    EXPECT_TRUE(r.exhaustiveNotRunReason.empty()) << r.exhaustiveNotRunReason;
}

// Plan-spec path: a spec whose engine ID is excluded by config.engineIdFilter
// surfaces as a filtered result, while the in-scope engine still benchmarks and
// wins. The excluded entry carries succeeded==false, an "engineIdFilter" message,
// and the run returns OK because the in-scope plan succeeded. The composition
// counters are also exercised: the filtered spec contributes to filteredCount.
TEST_F(TestGraph, AutotunePlanSpecEngineIdFilterExcludedReported)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t inScopeEngineId = 81; // selected by the filter; benchmarks
    const int64_t excludedEngineId = 82; // excluded by the filter

    graph.injectPlanSpec(inScopeEngineId, /*workspaceSize=*/256);
    // The excluded spec is exhaustive-capable: its capability must still surface on
    // the filtered result even though it is excluded before any priming.
    graph.injectPlanSpec(excludedEngineId, /*workspaceSize=*/256, /*supportsExhaustive=*/true);

    // Compiled workspace fits the ceiling for the in-scope plan.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = 256;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<float*>(out) = 1.0f;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    AutotuneConfig config; // STANDARD mode: no EXHAUSTIVE priming
    config.engineIdFilter = {inScopeEngineId};
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    const int64_t runtimeCeiling = 100000;
    auto result = graph.autotune(_handle, pack, workspace, runtimeCeiling, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Winner plus one filtered spec.
    ASSERT_EQ(results.size(), 2u);

    auto findResult = [&](int64_t engineId) -> const AutotuneResult* {
        for(const auto& r : results)
        {
            if(r.engineId == engineId)
            {
                return &r;
            }
        }
        return nullptr;
    };

    const auto* inScope = findResult(inScopeEngineId);
    const auto* excluded = findResult(excludedEngineId);
    ASSERT_NE(inScope, nullptr);
    ASSERT_NE(excluded, nullptr);

    EXPECT_TRUE(inScope->succeeded);
    EXPECT_EQ(inScope->rank, 0);
    // Benchmarked winner in STANDARD mode with a non-exhaustive spec.
    EXPECT_FALSE(inScope->supportsExhaustive);
    EXPECT_FALSE(inScope->ranExhaustive);
    EXPECT_TRUE(inScope->exhaustiveNotRunReason.empty());

    EXPECT_FALSE(excluded->succeeded);
    EXPECT_EQ(excluded->rank, -1);
    EXPECT_EQ(excluded->compiledPlanIndex, -1);
    EXPECT_NE(excluded->errorMessage.find("engineIdFilter"), std::string::npos)
        << excluded->errorMessage;
    // Capability carried even though excluded; priming never ran (pre-priming site).
    EXPECT_TRUE(excluded->supportsExhaustive);
    EXPECT_FALSE(excluded->ranExhaustive);
    EXPECT_TRUE(excluded->exhaustiveNotRunReason.empty());
}

// Compiled-plan path: a plan whose engine ID is excluded by config.engineIdFilter
// surfaces as a filtered result instead of being silently dropped. Filtering to an
// engine ID absent from the compiled plans drops them all, so the composition
// summary reports "<N> excluded by engineIdFilter" and *results carries one
// filtered entry per excluded plan.
TEST_F(TestGraph, AutotuneCompiledPlanEngineIdFilterExcludedReported)
{
    ::testing::FLAGS_gmock_verbose = "error";

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    hipdnn_frontend::GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());

    graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/4096, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/20, /*workspaceSize=*/4096, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/30, /*workspaceSize=*/4096, /*barred=*/false);

    AutotuneConfig config;
    config.engineIdFilter = {999}; // no compiled plan has this engine ID

    std::vector<AutotuneResult> results;
    auto result = graph.autotune(_handle, pack, /*workspace=*/nullptr, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("3 excluded by engineIdFilter"), std::string::npos)
        << result.err_msg;

    // Every excluded plan surfaces its own filtered result.
    ASSERT_EQ(results.size(), 3u);
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.succeeded);
        EXPECT_EQ(r.rank, -1);
        EXPECT_EQ(r.compiledPlanIndex, -1);
        EXPECT_NE(r.errorMessage.find("engineIdFilter"), std::string::npos) << r.errorMessage;
    }
}

// ============================================================================
// Ranking tests driving the real autotune::detail::rankAndSelectWinner
// ============================================================================

namespace
{
AutotuneResult
    makeSucceededResult(int64_t engineId, float minTimeMs, int compiledPlanIndex, float avgTimeMs)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = true;
    r.minTimeMs = minTimeMs;
    r.avgTimeMs = avgTimeMs;
    r.compiledPlanIndex = compiledPlanIndex;
    return r;
}

AutotuneResult makeFailedResult(int64_t engineId)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.succeeded = false;
    return r;
}
} // namespace

TEST_F(TestGraph, RankAndSelectWinnerDefaultRankingByMinTime)
{
    // minTimeMs {30,10,20} for engines {0,1,2} -> winner index 1, order {1,2,0}.
    std::vector<AutotuneResult> results;
    results.push_back(makeSucceededResult(0, 30.0f, 0, 30.0f));
    results.push_back(makeSucceededResult(1, 10.0f, 1, 10.0f));
    results.push_back(makeSucceededResult(2, 20.0f, 2, 20.0f));

    size_t activePlanIndex = 99;
    const AutotuneConfig config;
    auto err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
        results, config, activePlanIndex);

    ASSERT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(activePlanIndex, 1u); // winner's compiledPlanIndex
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].engineId, 1);
    EXPECT_EQ(results[1].engineId, 2);
    EXPECT_EQ(results[2].engineId, 0);
    EXPECT_EQ(results[0].rank, 0);
    EXPECT_EQ(results[1].rank, 1);
    EXPECT_EQ(results[2].rank, 2);
}

TEST_F(TestGraph, RankAndSelectWinnerFailedEnginesRankAfterSucceeded)
{
    std::vector<AutotuneResult> results;
    results.push_back(makeSucceededResult(1, 5.0f, 0, 5.0f));
    results.push_back(makeFailedResult(2));
    results.push_back(makeSucceededResult(3, 1.0f, 1, 1.0f));

    size_t activePlanIndex = 99;
    const AutotuneConfig config;
    auto err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
        results, config, activePlanIndex);

    ASSERT_TRUE(err.is_good()) << err.err_msg;
    ASSERT_EQ(results.size(), 3u);
    // Both succeeded entries come first (ranked by minTimeMs), failed last.
    EXPECT_TRUE(results[0].succeeded);
    EXPECT_TRUE(results[1].succeeded);
    EXPECT_FALSE(results[2].succeeded);
    EXPECT_EQ(results[2].rank, -1);
    EXPECT_EQ(results[0].engineId, 3); // fastest
}

TEST_F(TestGraph, RankAndSelectWinnerThrowingRankingFnFallsBackToMinTime)
{
    // F12: a ranking function that throws must NOT propagate; rankAndSelectWinner
    // falls back to default minTimeMs ranking and still selects a winner.
    std::vector<AutotuneResult> results;
    results.push_back(makeSucceededResult(0, 30.0f, 0, 30.0f));
    results.push_back(makeSucceededResult(1, 10.0f, 1, 10.0f));
    results.push_back(makeSucceededResult(2, 20.0f, 2, 20.0f));

    AutotuneConfig config;
    config.rankingFn
        = [](std::vector<AutotuneResult>&) { throw std::runtime_error("ranking blew up"); };

    size_t activePlanIndex = 99;
    Error err{ErrorCode::OK, ""};
    EXPECT_NO_THROW(err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
                        results, config, activePlanIndex));
    ASSERT_TRUE(err.is_good()) << err.err_msg;
    // Fallback minTimeMs ranking: engine 1 (10ms) wins.
    EXPECT_EQ(activePlanIndex, 1u);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].engineId, 1);
}

TEST_F(TestGraph, RankAndSelectWinnerCustomRankingFnChangesWinnerThroughProductionCode)
{
    //   engine 0: minTime 10 (min-time winner), avgTime 30
    //   engine 1: minTime 20,                   avgTime 10 (avg-time winner)
    //   engine 2: minTime 30,                   avgTime 20
    //   engine 9: failed (no timings)
    //
    // Default min-time order: {0, 1, 2}, winner engine 0.
    // Avg-time order:         {1, 2, 0}, winner engine 1.
    std::vector<AutotuneResult> results;
    results.push_back(makeSucceededResult(0, 10.0f, 0, 30.0f));
    results.push_back(makeSucceededResult(1, 20.0f, 1, 10.0f));
    results.push_back(makeSucceededResult(2, 30.0f, 2, 20.0f));
    results.push_back(makeFailedResult(9));

    AutotuneConfig config;
    config.rankingFn = [](std::vector<AutotuneResult>& res) {
        std::stable_sort(
            res.begin(), res.end(), [](const AutotuneResult& a, const AutotuneResult& b) {
                return a.avgTimeMs < b.avgTimeMs;
            });
    };

    size_t activePlanIndex = 99;
    auto err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
        results, config, activePlanIndex);

    ASSERT_TRUE(err.is_good()) << err.err_msg;
    ASSERT_EQ(results.size(), 4u);

    // (i) Succeeded results reordered by avgTime (10, 20, 30), not minTime.
    EXPECT_EQ(results[0].engineId, 1);
    EXPECT_EQ(results[1].engineId, 2);
    EXPECT_EQ(results[2].engineId, 0);
    EXPECT_FLOAT_EQ(results[0].avgTimeMs, 10.0f);
    EXPECT_FLOAT_EQ(results[1].avgTimeMs, 20.0f);
    EXPECT_FLOAT_EQ(results[2].avgTimeMs, 30.0f);

    // (ii) Winner is the avg-time winner (engine 1, compiledPlanIndex 1), NOT the
    // default min-time winner (engine 0, compiledPlanIndex 0).
    EXPECT_EQ(activePlanIndex, 1u);
    EXPECT_EQ(results[0].engineId, 1);

    // (iii) Succeeded ranks are 0..n-1 in avg-time order.
    EXPECT_EQ(results[0].rank, 0);
    EXPECT_EQ(results[1].rank, 1);
    EXPECT_EQ(results[2].rank, 2);

    // Failed result stays last with rank -1; production passes only succeeded
    // results to the custom ranking fn, so it cannot reorder the failed entry.
    EXPECT_FALSE(results[3].succeeded);
    EXPECT_EQ(results[3].engineId, 9);
    EXPECT_EQ(results[3].rank, -1);
}

TEST_F(TestGraph, RankAndSelectWinnerContiguousRanksSucceededTimeOrderFailedMinusOne)
{
    // Succeeded entries (out of time order) interleaved with failures. After
    // ranking, succeeded entries take contiguous ranks 0..n-1 in ascending
    // minTimeMs order, and every failed entry takes rank -1.
    std::vector<AutotuneResult> results;
    results.push_back(makeSucceededResult(0, 30.0f, 0, 30.0f));
    results.push_back(makeFailedResult(1));
    results.push_back(makeSucceededResult(2, 10.0f, 1, 10.0f));
    results.push_back(makeFailedResult(3));
    results.push_back(makeSucceededResult(4, 20.0f, 2, 20.0f));

    size_t activePlanIndex = 99;
    const AutotuneConfig config;
    auto err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
        results, config, activePlanIndex);

    ASSERT_TRUE(err.is_good()) << err.err_msg;
    ASSERT_EQ(results.size(), 5u);

    // (a) Succeeded results first, contiguous ranks 0..2 in time order {2,4,0}.
    EXPECT_TRUE(results[0].succeeded);
    EXPECT_TRUE(results[1].succeeded);
    EXPECT_TRUE(results[2].succeeded);
    EXPECT_EQ(results[0].engineId, 2);
    EXPECT_EQ(results[1].engineId, 4);
    EXPECT_EQ(results[2].engineId, 0);
    EXPECT_EQ(results[0].rank, 0);
    EXPECT_EQ(results[1].rank, 1);
    EXPECT_EQ(results[2].rank, 2);

    // (b) Failed results last, each rank -1.
    EXPECT_FALSE(results[3].succeeded);
    EXPECT_FALSE(results[4].succeeded);
    EXPECT_EQ(results[3].rank, -1);
    EXPECT_EQ(results[4].rank, -1);
}

TEST_F(TestGraph, RankAndSelectWinnerAllFailedReturnsBackendErrorWithPinnedMessage)
{
    // (c) No succeeded results -> fatal error with the pinned message.
    std::vector<AutotuneResult> results;
    results.push_back(makeFailedResult(0));
    results.push_back(makeFailedResult(1));

    size_t activePlanIndex = 99;
    const AutotuneConfig config;
    auto err = hipdnn_frontend::GraphTestUtils::callRankAndSelectWinner(
        results, config, activePlanIndex);

    EXPECT_FALSE(err.is_good());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(err.err_msg, "All engines failed during autotuning. No winner selected.");

    // Failed entries are still ranked -1 even on the error path.
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].rank, -1);
    EXPECT_EQ(results[1].rank, -1);
}

// ============================================================================
// dedup / strip / max driving real production APIs
// ============================================================================

TEST_F(TestGraph, AddEngineDedupPreventsDuplicatePlanSpecs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    // Two identical (engineId, knobSettings) and one distinct engine -> 2 specs.
    EXPECT_TRUE(graph.add_engine(42, {KnobSetting("tile_size", int64_t{256})}).is_good());
    EXPECT_TRUE(graph.add_engine(42, {KnobSetting("tile_size", int64_t{256})}).is_good());
    EXPECT_TRUE(graph.add_engine(99, {KnobSetting("tile_size", int64_t{256})}).is_good());

    EXPECT_EQ(graph.getPlanSpecsCount(), 2u);
}

TEST_F(TestGraph, AddEngineStripsBenchmarkingKnobFromPlanSpec)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto knobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {knobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    knobDesc,
                                    "tile_size",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{256}});

    auto result = graph.add_engine(
        42,
        {KnobSetting("tile_size", int64_t{256}),
         KnobSetting(hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME, int64_t{1})});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto knobIds = graph.getPlanSpecKnobIds();
    ASSERT_EQ(knobIds.size(), 1u);
    EXPECT_EQ(std::count(knobIds[0].begin(),
                         knobIds[0].end(),
                         hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME),
              0)
        << "benchmarking knob must be stripped from the stored plan spec";
    EXPECT_NE(std::find(knobIds[0].begin(), knobIds[0].end(), "tile_size"), knobIds[0].end());
}

TEST_F(TestGraph, GetEstimatedMaxWorkspaceReturnsMaxOfPlanSpecs)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 100);
    graph.injectPlanSpec(2, 500);
    graph.injectPlanSpec(3, 200);

    int64_t maxWs = -1;
    auto result = graph.get_estimated_max_workspace_size(maxWs);
    ASSERT_TRUE(result.is_good()) << result.err_msg;
    EXPECT_EQ(maxWs, 500);
}

// ============================================================================
// engine-ID filter, variant->knob-settings conversion, exhaustive-priming
// knob injection - all driving the real production autotune APIs (host, no GPU).
//
// These replace deleted reimplemented-logic unit tests that re-ran std::find /
// a hand-copied strip loop and asserted on the copy. Each case below drives the
// production code path and asserts the production output.
// ============================================================================

// Drives the real compiled-plan engine-ID filter loop (autotuneImpl). The
// filter selects the subset of _compiledPlans whose engineId is in
// config.engineIdFilter; non-matching plans are excluded. Both behaviors are
// asserted via host-reachable early-returns that fire before the GPU benchmark
// loop, so no real GPU op runs:
//   - inclusion: a filtered-in plan that needs workspace forces the
//     "workspace pointer is null" early-return (proving it survived the filter);
//   - exclusion: filtering to an engine ID absent from _compiledPlans empties
//     the post-filter map, forcing the "No execution plans were benchmarkable"
//     early-return (proving non-matching plans are dropped).
TEST_F(TestGraph, EngineIdFilterSelectsSubset)
{
    ::testing::FLAGS_gmock_verbose = "error";

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    // Inclusion: only the filtered-in engine (20) survives. It carries a
    // non-zero workspace, so the post-filter workspace-null check fires -
    // which can only happen if engine 20 was selected by the filter.
    {
        hipdnn_frontend::GraphTestUtils graph;
        createBasicBatchnormGraph(graph);
        ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());

        graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/4096, /*barred=*/false);
        graph.injectValidCompiledPlan(/*engineId=*/20, /*workspaceSize=*/4096, /*barred=*/false);
        graph.injectValidCompiledPlan(/*engineId=*/30, /*workspaceSize=*/4096, /*barred=*/false);

        AutotuneConfig config;
        config.engineIdFilter = {20};

        std::vector<AutotuneResult> results;
        // The compiled-plan overload with a null workspace pointer.
        auto result = graph.autotune(_handle, pack, /*workspace=*/nullptr, config, {}, &results);

        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
        EXPECT_NE(result.err_msg.find("workspace pointer is null"), std::string::npos)
            << "Filtered-in engine 20 needs workspace; the null-workspace guard must "
               "fire only because the filter selected it: "
            << result.err_msg;
    }

    // Exclusion: filtering to an engine ID that is NOT among the compiled plans
    // drops every plan, so the post-filter map is empty.
    {
        hipdnn_frontend::GraphTestUtils graph;
        createBasicBatchnormGraph(graph);
        ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());

        graph.injectValidCompiledPlan(/*engineId=*/10, /*workspaceSize=*/4096, /*barred=*/false);
        graph.injectValidCompiledPlan(/*engineId=*/20, /*workspaceSize=*/4096, /*barred=*/false);
        graph.injectValidCompiledPlan(/*engineId=*/30, /*workspaceSize=*/4096, /*barred=*/false);

        AutotuneConfig config;
        config.engineIdFilter = {999}; // no compiled plan has this engine ID

        std::vector<AutotuneResult> results;
        auto result = graph.autotune(_handle, pack, /*workspace=*/nullptr, config, {}, &results);

        EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
        EXPECT_NE(result.err_msg.find("No execution plans were benchmarkable"), std::string::npos)
            << "Filtering to an absent engine ID must drop all plans: " << result.err_msg;
    }
}

// Drives the real add_engine_variants() map->vector knob conversion. The
// production code converts EngineVariant::knobSettings (a std::map) into the
// stored PlanSpec::knobSettings vector, stripping the benchmarking knob. The
// test asserts the produced plan-spec knob vector equals the expected
// (knobId, value) set exactly.
TEST_F(TestGraph, EngineVariantToKnobSettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    auto tileKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA001);
    auto splitKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA002);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {tileKnobDesc, splitKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    tileKnobDesc,
                                    "TILE_SIZE",
                                    "Tile dimension",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{128}});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    splitKnobDesc,
                                    "SPLIT_K",
                                    "Split-K factor",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{2}});

    EngineVariant variant;
    variant.engineId = 42;
    variant.knobSettings["TILE_SIZE"] = int64_t{128};
    variant.knobSettings["SPLIT_K"] = int64_t{2};
    // The benchmarking knob must be dropped by the production conversion.
    variant.knobSettings[hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME] = int64_t{1};

    auto result = graph.add_engine_variants({variant});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    auto settings = graph.getPlanSpecKnobSettings();
    ASSERT_EQ(settings.size(), 1u);

    const std::set<std::pair<std::string, KnobValueVariant>> expected = {
        {"TILE_SIZE", KnobValueVariant{int64_t{128}}}, {"SPLIT_K", KnobValueVariant{int64_t{2}}}};
    EXPECT_EQ(settings[0], expected)
        << "Produced plan-spec knob settings must equal the converted variant "
           "settings with the benchmarking knob stripped";
}

// Drives the real EXHAUSTIVE priming-knob injection in autotuneImpl. The
// production code copies the plan-spec knob settings and appends
// (global.benchmarking, 1) before compiling the priming plan. The injected
// knob set is captured at the production knob-choice descriptor calls
// (HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE / KNOB_VALUE) made during the priming
// compile, and asserted to equal the expected (knobId, value) set exactly.
//
// The engine is added with NO user knobs, so the only knob the priming compile
// applies is the injected benchmarking knob. The subsequent real-plan compile
// applies an empty knob set (no knob-choice calls), keeping the capture exact.
// maxWorkspaceSize=0 with a non-zero compiled workspace forces the workspace
// early-return before the GPU benchmark loop.
TEST_F(TestGraph, ExhaustivePrimingKnobInjection)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    auto engineDesc = buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    // The engine exposes the benchmarking knob so EXHAUSTIVE priming applies.
    auto benchKnobDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xB001);
    mockKnobInfoQueryRepeated(_mockBackend, engineDesc, {benchKnobDesc});
    setupKnobDescriptorMockRepeated(_mockBackend,
                                    benchKnobDesc,
                                    hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME,
                                    "Benchmarking knob",
                                    false,
                                    HIPDNN_TYPE_INT64,
                                    KnobValueVariant{int64_t{0}});

    // Add the engine with no user knobs: the only knob the priming compile
    // applies is the injected (global.benchmarking, 1).
    ASSERT_TRUE(graph.add_engine(42, {}).is_good());

    // Capture the knob-choice (id, value) pairs the production code sends to the
    // backend. The benchmarking attribute names are unique to knob-choice
    // descriptors, so matching on them isolates knob-choice writes.
    std::vector<std::string> capturedKnobIds;
    std::vector<int64_t> capturedKnobValues;
    ON_CALL(*_mockBackend,
            backendSetAttribute(_, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, HIPDNN_TYPE_CHAR, _, _))
        .WillByDefault([&capturedKnobIds](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t elementCount,
                                          const void* arrayOfElements) {
            capturedKnobIds.emplace_back(static_cast<const char*>(arrayOfElements),
                                         static_cast<size_t>(elementCount));
            return HIPDNN_STATUS_SUCCESS;
        });
    ON_CALL(*_mockBackend,
            backendSetAttribute(_, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, HIPDNN_TYPE_INT64, _, _))
        .WillByDefault([&capturedKnobValues](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             const void* arrayOfElements) {
            capturedKnobValues.push_back(*static_cast<const int64_t*>(arrayOfElements));
            return HIPDNN_STATUS_SUCCESS;
        });

    // Force the compiled real plan to exceed the workspace limit so autotune
    // early-returns before the GPU benchmark loop (the priming knob-choice
    // capture has already happened by then).
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = 4096;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;

    std::vector<AutotuneResult> results;
    // The general plan-spec overload with maxWorkspaceSize=0 (all plans exceed it).
    auto result
        = graph.autotune(_handle, pack, /*workspace=*/nullptr, int64_t{0}, config, {}, &results);

    // Early-returns at the workspace guard (priming already ran and captured).
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    // Exactly one knob was applied during priming: the injected benchmarking
    // knob with value 1. The real-plan compile applies an empty knob set.
    std::set<std::pair<std::string, int64_t>> captured;
    ASSERT_EQ(capturedKnobIds.size(), capturedKnobValues.size());
    for(size_t i = 0; i < capturedKnobIds.size(); ++i)
    {
        captured.insert({capturedKnobIds[i], capturedKnobValues[i]});
    }
    const std::set<std::pair<std::string, int64_t>> expected
        = {{hipdnn_frontend::autotune::detail::BENCHMARKING_KNOB_NAME, int64_t{1}}};
    EXPECT_EQ(captured, expected)
        << "EXHAUSTIVE priming must inject exactly (global.benchmarking, 1)";

    // The real plan exceeds the workspace limit and is skipped before the
    // benchmark loop, so the surfaced result is a workspace-skipped result. That
    // post-priming site now carries the engine's exhaustive disposition:
    //   - supportsExhaustive=true (the engine exposes the benchmarking knob),
    //   - ranExhaustive=false (priming itself was workspace-skipped because the
    //     priming plan's compiled workspace did not fit the null workspace), and
    //   - exhaustiveNotRunReason records that priming skip.
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].supportsExhaustive);
    EXPECT_FALSE(results[0].ranExhaustive);
    EXPECT_NE(results[0].exhaustiveNotRunReason.find("Priming execution skipped"),
              std::string::npos)
        << results[0].exhaustiveNotRunReason;
}

// Drives the post-compile workspace-growth skip branch in autotuneImpl
// alongside a sibling plan that fits, proving the guard discriminates per plan.
//
// Two engines reach the workspace query; both estimates fit within the limit,
// but only one stays within after compilation:
//   - engine A (the fitting plan): estimate 512, compiled 1024 <= limit 1500 ->
//     benchmarked and selected as the winner.
//   - engine B (the growing plan): estimate 1024 <= limit 1500 < compiled 4096 ->
//     skipped (not benchmarked) and surfaced as a failed AutotuneResult.
//
// Estimates are set directly via injectPlanSpec; per-plan compiled workspace is
// returned in compile order by the EXECUTION_PLAN_WORKSPACE_SIZE mock (A first,
// B second). STANDARD mode skips priming, and the benchmark path runs entirely
// through the mocked backend (execute + profiling elapsed-ms), so no GPU is
// needed. The scenario is non-degenerate: both estimates are positive and within
// a positive limit, and only B's compiled workspace grows past it.
TEST_F(TestGraph, AutotuneSkipsPlanWhoseCompiledWorkspaceGrowsPastLimit)
{
    ::testing::FLAGS_gmock_verbose = "error";
    hipdnn_frontend::GraphTestUtils graph;
    // Built operation graph plus AnyNumber create/set/finalize/get mocks so the
    // plan-spec compile, engine-config set-attr, and finalize all succeed.
    buildGraphAndMockEngineRepeated(_mockBackend, graph, _handle);

    const int64_t fitEngineId = 41;
    const int64_t growEngineId = 42;
    const int64_t fitEstimate = 512; // within the limit, compiles to a fitting size
    const int64_t growEstimate = 1024; // within the limit pre-compile, grows past it after
    const int64_t maxWorkspaceSize = 1500; // binding limit, between both estimates and growCompiled
    const int64_t fitCompiled = 1024; // post-compile, still within the limit
    const int64_t growCompiled = 4096; // post-compile, grows past the limit

    // Two plan specs with positive estimates that both fit pre-compile. The fit
    // engine is inserted first, so it is compiled (and workspace-queried) first.
    graph.injectPlanSpec(fitEngineId, fitEstimate);
    graph.injectPlanSpec(growEngineId, growEstimate);

    // Return the compiled workspace per plan in compile order: A (fits) then
    // B (grows). The compile loop iterates plan specs in insertion order.
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .Times(AnyNumber())
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* out) {
            *static_cast<int64_t*>(out) = fitCompiled;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<int64_t*>(out) = growCompiled;
            return HIPDNN_STATUS_SUCCESS;
        });

    // The fitting plan is benchmarked: its execution and profiling go through the
    // mocked backend. Execute succeeds; the profiling elapsed-ms query returns a
    // positive time so the benchmark produces a valid measurement.
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _))
        .Times(AnyNumber())
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, nullptr, _))
        .Times(AnyNumber())
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t*,
                           void* out) {
            *static_cast<float*>(out) = 1.0f;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::unordered_map<int64_t, void*> pack = {{1, reinterpret_cast<void*>(0x1)},
                                                     {2, reinterpret_cast<void*>(0x2)},
                                                     {3, reinterpret_cast<void*>(0x3)},
                                                     {4, reinterpret_cast<void*>(0x4)},
                                                     {5, reinterpret_cast<void*>(0x5)}};

    const AutotuneConfig config; // STANDARD mode: no EXHAUSTIVE priming

    // The fitting plan needs workspace, so a non-null workspace pointer is
    // required to pass the null-workspace guard. The pointer is never
    // dereferenced (execution is mocked).
    auto* const workspace = reinterpret_cast<void*>(0xC0FFEE);

    std::vector<AutotuneResult> results;
    // The general plan-spec overload with the binding workspace limit.
    auto result = graph.autotune(_handle, pack, workspace, maxWorkspaceSize, config, {}, &results);

    // The fitting plan succeeded, so autotune returns OK.
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Both engines appear in the results: the fitting winner and the skipped
    // grown plan.
    ASSERT_EQ(results.size(), 2u);

    auto findResult = [&](int64_t engineId) -> const AutotuneResult* {
        for(const auto& r : results)
        {
            if(r.engineId == engineId)
            {
                return &r;
            }
        }
        return nullptr;
    };

    const auto* fit = findResult(fitEngineId);
    const auto* grow = findResult(growEngineId);
    ASSERT_NE(fit, nullptr);
    ASSERT_NE(grow, nullptr);

    // The fitting plan was benchmarked: it succeeded, ranked winner, and its
    // compiled workspace stayed within the limit.
    EXPECT_TRUE(fit->succeeded);
    EXPECT_EQ(fit->rank, 0);
    EXPECT_EQ(fit->workspaceSize, fitCompiled);
    EXPECT_LE(fit->workspaceSize, maxWorkspaceSize);
    // Benchmarked winner in STANDARD mode with a non-exhaustive spec.
    EXPECT_FALSE(fit->supportsExhaustive);
    EXPECT_FALSE(fit->ranExhaustive);
    EXPECT_TRUE(fit->exhaustiveNotRunReason.empty());

    // The growing plan was skipped: not succeeded, rank -1, and its estimate
    // and compiled size DIFFER - the estimate fit, the compiled size grew past.
    EXPECT_FALSE(grow->succeeded);
    EXPECT_EQ(grow->rank, -1);
    EXPECT_EQ(grow->estimatedWorkspaceSize, growEstimate);
    EXPECT_EQ(grow->workspaceSize, growCompiled);
    EXPECT_NE(grow->estimatedWorkspaceSize, grow->workspaceSize);
    EXPECT_LE(grow->estimatedWorkspaceSize, maxWorkspaceSize);
    EXPECT_GT(grow->workspaceSize, maxWorkspaceSize);
    // Post-compile workspace skip in STANDARD mode with a non-exhaustive spec.
    EXPECT_FALSE(grow->supportsExhaustive);
    EXPECT_FALSE(grow->ranExhaustive);
    EXPECT_TRUE(grow->exhaustiveNotRunReason.empty());
}

// ============================================================================
// get_estimated_max_workspace_size() validation
// ============================================================================

TEST_F(TestGraph, GetEstimatedMaxWorkspaceSizeRejectsEmptyPlanSpecs)
{
    const hipdnn_frontend::GraphTestUtils graph;
    // No plan specs added - method should return INVALID_VALUE
    int64_t maxWs = -1;
    const auto result = graph.get_estimated_max_workspace_size(maxWs);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("No plan specs"), std::string::npos)
        << "Error message should explain the cause: " << result.err_msg;
}

TEST_F(TestGraph, GetEstimatedMaxWorkspaceSizeWithPlanSpecs)
{
    hipdnn_frontend::GraphTestUtils graph;
    graph.injectPlanSpec(1, 512);
    graph.injectPlanSpec(2, 2048);
    graph.injectPlanSpec(3, 1024);

    int64_t maxWs = 0;
    const auto result = graph.get_estimated_max_workspace_size(maxWs);
    EXPECT_EQ(result.code, ErrorCode::OK);
    EXPECT_EQ(maxWs, 2048);
}

// ============================================================================
// Serialize/deserialize graph-and-plan tests
//
// Pure unit tests on Mock_hipdnn_backend with the backend container framing
// fully mocked: they assert which serialize/deserialize C-API wrappers are
// invoked, with what arguments, and the resulting frontend state. No
// flatbuffers / container byte-format assertions are made here.
// ============================================================================

namespace
{
// Drives a GraphTestUtils to a state with a valid (lowered) graph descriptor and
// an execution plan: build_operation_graph() then create_execution_plans(), and
// (when buildPlans is true) build_plans() to finalize the plan. The serialize
// plan branch requires both descriptors valid and the plan finalized, which the
// deserialize-attach path cannot supply. Pass buildPlans=false to stop after the
// plan descriptor is created but before it is finalized.
void buildGraphWithExecutionPlan(GraphTestUtils& graph,
                                 ::testing::NiceMock<Mock_hipdnn_backend>& mockBackend,
                                 hipdnnHandle_t handle,
                                 bool buildPlans = true)
{
    using ::testing::_;
    using ::testing::Return;

    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(handle).is_good());

    ON_CALL(mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    ON_CALL(mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    // Generic getAttribute default: report a single element and succeed. This is
    // sufficient for the heuristic/engine-config queries that build_plans drives.
    ON_CALL(mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t* elementCount,
                          void*) {
            if(elementCount != nullptr)
            {
                *elementCount = 1;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    ON_CALL(mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault([heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = heurDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    ON_CALL(mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillByDefault(
            [engineConfigDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = engineConfigDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    ON_CALL(mockBackend,
            backendGetAttribute(engineConfigDesc,
                                HIPDNN_ATTR_ENGINECFG_ENGINE,
                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                1,
                                _,
                                _))
        .WillByDefault([engineDesc](hipdnnBackendDescriptor_t,
                                    hipdnnBackendAttributeName_t,
                                    hipdnnBackendAttributeType_t,
                                    int64_t,
                                    int64_t*,
                                    void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(mockBackend,
            backendGetAttribute(
                engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t*,
                          void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    ON_CALL(mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillByDefault(
            [executionPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = executionPlanDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    ASSERT_TRUE(graph.create_execution_plans(heurModes).is_good());

    if(!buildPlans)
    {
        // Plan descriptor created but intentionally left unfinalized.
        ASSERT_TRUE(graph.hasExecutionPlan());
        ASSERT_FALSE(graph.isExecutionPlanFinalized());
        return;
    }

    ASSERT_TRUE(graph.build_plans().is_good());
    ASSERT_TRUE(graph.hasExecutionPlan());
    ASSERT_TRUE(graph.isExecutionPlanFinalized());
}

void setupEngineWithNoKnobs(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend)
{
    ON_CALL(mockBackend,
            backendGetAttribute(
                _, HIPDNN_ATTR_ENGINE_KNOB_INFO, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, _, nullptr))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t* elementCount,
                          void*) {
            if(elementCount != nullptr)
            {
                *elementCount = 0;
            }
            return HIPDNN_STATUS_SUCCESS;
        });
}

void expectEngineSupportsPlanSerialization(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend)
{
    EXPECT_CALL(mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });
}

void expectGraphOnlySerialization(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend,
                                  const std::vector<uint8_t>& fakeGraphBytes)
{
    EXPECT_CALL(mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);
    EXPECT_CALL(mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .Times(2)
        .WillRepeatedly([&fakeGraphBytes](hipdnnBackendDescriptor_t,
                                          size_t requestedSize,
                                          size_t* graphByteSize,
                                          uint8_t* data) {
            *graphByteSize = fakeGraphBytes.size();
            if(data != nullptr && requestedSize >= fakeGraphBytes.size())
            {
                std::memcpy(data, fakeGraphBytes.data(), fakeGraphBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });
}

void expectComboContainerSerialization(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend,
                                       const std::vector<uint8_t>& fakeContainerBytes)
{
    EXPECT_CALL(mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _)).Times(0);
    EXPECT_CALL(mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _))
        .Times(2)
        .WillRepeatedly([&fakeContainerBytes](hipdnnBackendDescriptor_t,
                                              hipdnnBackendDescriptor_t,
                                              size_t requestedByteSize,
                                              size_t* blobByteSize,
                                              uint8_t* serializedBlob) {
            *blobByteSize = fakeContainerBytes.size();
            if(serializedBlob != nullptr && requestedByteSize >= fakeContainerBytes.size())
            {
                std::memcpy(serializedBlob, fakeContainerBytes.data(), fakeContainerBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });
}

// Installs backendGetAttribute ON_CALL defaults that reconstruct a single
// unary-pointwise operation graph during deserialize, letting the
// post-reconstruction contents-query + plan-attach logic run. Keying on the
// attribute name alone works because tensor and operation/graph attributes
// occupy disjoint name sets.
void setupSuccessfulPointwiseGraphUnpack(::testing::NiceMock<Mock_hipdnn_backend>& mockBackend)
{
    using ::testing::_;

    auto fakeOpDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA100);
    auto fakeTensorDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA200);

    ON_CALL(mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillByDefault([fakeOpDesc, fakeTensorDesc](hipdnnBackendDescriptor_t,
                                                    hipdnnBackendAttributeName_t attrName,
                                                    hipdnnBackendAttributeType_t,
                                                    int64_t requestedCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements) {
            const bool isCountQuery = (arrayOfElements == nullptr);
            auto setCount = [&](int64_t value) {
                if(elementCount != nullptr)
                {
                    *elementCount = value;
                }
            };

            switch(attrName)
            {
            // Operation-graph: exactly one operation descriptor.
            case HIPDNN_ATTR_OPERATIONGRAPH_OPS:
                if(isCountQuery)
                {
                    setCount(1);
                }
                else
                {
                    setCount(1);
                    *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = fakeOpDesc;
                }
                return HIPDNN_STATUS_SUCCESS;

            // Operation type: unary pointwise (single IN_0 input).
            case HIPDNN_ATTR_OPERATION_TYPE_EXT:
                setCount(1);
                *static_cast<hipdnnOperationType_ext_t*>(arrayOfElements)
                    = HIPDNN_OPERATION_TYPE_POINTWISE_EXT;
                return HIPDNN_STATUS_SUCCESS;

            // Mandatory pointwise tensors return a tensor descriptor.
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT:
            case HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT:
                setCount(1);
                if(!isCountQuery)
                {
                    *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = fakeTensorDesc;
                }
                return HIPDNN_STATUS_SUCCESS;

            // Optional pointwise tensors are absent.
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT:
            case HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT:
                setCount(0);
                return HIPDNN_STATUS_SUCCESS;

            // Pointwise mode (mandatory scalar).
            case HIPDNN_ATTR_POINTWISE_MODE:
                setCount(1);
                *static_cast<hipdnnPointwiseMode_t*>(arrayOfElements) = HIPDNN_POINTWISE_RELU_FWD;
                return HIPDNN_STATUS_SUCCESS;

            // Tensor scalar attributes.
            case HIPDNN_ATTR_TENSOR_UNIQUE_ID:
                setCount(1);
                *static_cast<int64_t*>(arrayOfElements) = 1;
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_DATA_TYPE:
                setCount(1);
                *static_cast<hipdnnDataType_t*>(arrayOfElements) = HIPDNN_DATA_FLOAT;
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_DIMENSIONS:
            case HIPDNN_ATTR_TENSOR_STRIDES:
                if(isCountQuery)
                {
                    setCount(4);
                }
                else
                {
                    setCount(4);
                    auto* dims = static_cast<int64_t*>(arrayOfElements);
                    for(int64_t i = 0; i < requestedCount && i < 4; ++i)
                    {
                        dims[i] = 1;
                    }
                }
                return HIPDNN_STATUS_SUCCESS;
            case HIPDNN_ATTR_TENSOR_IS_VIRTUAL:
            case HIPDNN_ATTR_TENSOR_IS_BY_VALUE:
                setCount(1);
                *static_cast<bool*>(arrayOfElements) = false;
                return HIPDNN_STATUS_SUCCESS;

            // Optional / unset attributes: report zero elements so the
            // unpacker treats them as absent.
            default:
                setCount(0);
                return HIPDNN_STATUS_SUCCESS;
            }
        });
}
} // namespace

// No plan built: the bare graph serializer is used (byte-identical to a legacy
// graph blob) and the combo container API is never invoked.
TEST_F(TestGraph, SerializeNoPlanUsesGraphSerializerByteIdentical)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    const std::vector<uint8_t> fakeGraphBytes = {0x01, 0x02, 0x03};

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&fakeGraphBytes](hipdnnBackendDescriptor_t,
                                         size_t requestedSize,
                                         size_t* graphByteSize,
                                         uint8_t* data) {
            *graphByteSize = fakeGraphBytes.size();
            if(data != nullptr && requestedSize >= fakeGraphBytes.size())
            {
                std::memcpy(data, fakeGraphBytes.data(), fakeGraphBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The combo container API must not be touched when no plan exists.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeGraphBytes);
}

// A plan is built: the combo container API (two-call size/fill) is used and the
// bare graph serializer is bypassed.
TEST_F(TestGraph, SerializeWithPlanUsesComboContainerApi)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    // Select an engine and report the serialization note so serialize() takes
    // the combo path (two-call count/fill behavior-note query).
    graph.setSelectedEngineId(10);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> fakeContainerBytes = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE};

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _))
        .WillByDefault([&fakeContainerBytes](hipdnnBackendDescriptor_t,
                                             hipdnnBackendDescriptor_t,
                                             size_t requestedByteSize,
                                             size_t* blobByteSize,
                                             uint8_t* serializedBlob) {
            *blobByteSize = fakeContainerBytes.size();
            if(serializedBlob != nullptr && requestedByteSize >= fakeContainerBytes.size())
            {
                std::memcpy(serializedBlob, fakeContainerBytes.data(), fakeContainerBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The plan path must not fall back to the bare graph serializer.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

// A plan descriptor was created (create_execution_plans()) but never finalized
// via build_plans()/build(): serialize() must gate on the finalized state and
// fall back to the bare graph serializer (byte-identical to a graph that never
// had a plan). The combo container API and the engine capability query are both
// short-circuited because the plan is not finalized. This guards against
// embedding an unfinalized plan (PR #7975 review feedback): serialize must gate
// on the finalized/compiled-plan state, not merely on plan-descriptor validity.
TEST_F(TestGraph, SerializeWithUnfinalizedPlanUsesGraphSerializer)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle, /*buildPlans=*/false);
    EXPECT_FALSE(graph.isExecutionPlanFinalized());

    // Select an engine that, if queried, would advertise support. This proves the
    // graph-only fallback is forced by the finalization gate, not by a missing
    // engine id or an unsupported engine.
    graph.setSelectedEngineId(10);

    const std::vector<uint8_t> fakeGraphBytes = {0x01, 0x02, 0x03};

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&fakeGraphBytes](hipdnnBackendDescriptor_t,
                                         size_t requestedSize,
                                         size_t* graphByteSize,
                                         uint8_t* data) {
            *graphByteSize = fakeGraphBytes.size();
            if(data != nullptr && requestedSize >= fakeGraphBytes.size())
            {
                std::memcpy(data, fakeGraphBytes.data(), fakeGraphBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The combo container API and the engine capability query must both be
    // skipped: the unfinalized-plan gate short-circuits before either is reached.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, _, _, _, _))
        .Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeGraphBytes);
}

TEST_F(TestGraph, CreateExecutionPlanExtResetsActivePlanSerializationState)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    ASSERT_TRUE(graph.isExecutionPlanFinalized());

    graph.setSelectedEngineId(777);

    ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
            return HIPDNN_STATUS_SUCCESS;
        });
    setupEngineWithNoKnobs(*_mockBackend);

    const int64_t engineId = -2;
    auto result = graph.create_execution_plan_ext(engineId, {});
    ASSERT_TRUE(result.is_good()) << result.get_message();

    EXPECT_EQ(graph.getCompiledPlansCount(), 1u);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_TRUE(graph.hasExecutionPlan());
    EXPECT_FALSE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), engineId);

    const std::vector<uint8_t> fakeGraphBytes = {0x01, 0x02, 0x03};
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, _, _, _, _))
        .Times(0);
    expectGraphOnlySerialization(*_mockBackend, fakeGraphBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeGraphBytes);
}

TEST_F(TestGraph, CreateExecutionPlanExtBuildPlansSerializesComboContainer)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    setupEngineWithNoKnobs(*_mockBackend);

    const int64_t engineId = -2;
    auto result = graph.create_execution_plan_ext(engineId, {});
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_FALSE(graph.isExecutionPlanFinalized());

    result = graph.build_plans();
    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), engineId);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0xAA, 0xBB, 0xCC};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, BuildPlansAllMarksFirstSuccessfulPlanSerializable)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    graph.injectValidCompiledPlan(/*engineId=*/-2, /*workspaceSize=*/-1, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/-3, /*workspaceSize=*/-1, /*barred=*/false);

    auto result = graph.build_plans(BuildPlanPolicy::ALL);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -2);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0xBA, 0xAD, 0xF0, 0x0D};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, BuildPlanAtIndexUpdatesSerializableActivePlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    graph.injectValidCompiledPlan(/*engineId=*/-2, /*workspaceSize=*/-1, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/-3, /*workspaceSize=*/-1, /*barred=*/false);

    auto result = graph.build_plan_at_index(1);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(graph.getActivePlanIndex(), 1u);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -3);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0x20, 0x21, 0x22};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, CompiledPlanAutotuneWinnerUpdatesSerializableActivePlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    graph.injectValidCompiledPlan(/*engineId=*/-2, /*workspaceSize=*/0, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/-3, /*workspaceSize=*/0, /*barred=*/false);

    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 0;
    config.rankingFn = [](std::vector<AutotuneResult>& results) {
        std::reverse(results.begin(), results.end());
    };

    std::vector<AutotuneResult> results;
    const std::unordered_map<int64_t, void*> variantPack = {{1, reinterpret_cast<void*>(0x1)},
                                                            {2, reinterpret_cast<void*>(0x2)},
                                                            {3, reinterpret_cast<void*>(0x3)},
                                                            {4, reinterpret_cast<void*>(0x4)},
                                                            {5, reinterpret_cast<void*>(0x5)}};
    auto result = graph.autotune(_handle, variantPack, nullptr, config, {}, &results);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(results.size(), 2u);
    // Compiled-plan path never primes, so every result (including the benchmarked
    // winner) reports ranExhaustive=false with no priming reason.
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.ranExhaustive);
        EXPECT_TRUE(r.exhaustiveNotRunReason.empty()) << r.exhaustiveNotRunReason;
    }
    EXPECT_EQ(graph.getActivePlanIndex(), 1u);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -3);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0x30, 0x31, 0x32};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, CompiledPlanAutotuneFailurePreservesActivePlanState)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    graph.injectValidCompiledPlan(/*engineId=*/-2, /*workspaceSize=*/0, /*barred=*/false);
    graph.injectValidCompiledPlan(/*engineId=*/-3, /*workspaceSize=*/0, /*barred=*/false);

    auto result = graph.build_plan_at_index(1);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(graph.getActivePlanIndex(), 1u);
    ASSERT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    ASSERT_EQ(*graph.selectedEngineIdForTest(), -3);

    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 0;
    config.engineIdFilter = {-999};

    std::vector<AutotuneResult> results;
    const std::unordered_map<int64_t, void*> variantPack = {{1, reinterpret_cast<void*>(0x1)},
                                                            {2, reinterpret_cast<void*>(0x2)},
                                                            {3, reinterpret_cast<void*>(0x3)},
                                                            {4, reinterpret_cast<void*>(0x4)},
                                                            {5, reinterpret_cast<void*>(0x5)}};
    result = graph.autotune(_handle, variantPack, nullptr, config, {}, &results);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(graph.getActivePlanIndex(), 1u);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -3);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0x33, 0x44, 0x55};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, PlanSpecAutotuneWinnerUpdatesSerializableActivePlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    setupEngineWithNoKnobs(*_mockBackend);

    EngineConfigInfo engineA;
    engineA.engineId = -2;
    engineA.estimatedWorkspaceSize = 0;
    EngineConfigInfo engineB;
    engineB.engineId = -3;
    engineB.estimatedWorkspaceSize = 0;
    auto addResult = graph.add_engine_configs({engineA, engineB});
    ASSERT_TRUE(addResult.is_good()) << addResult.get_message();

    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 0;
    config.rankingFn = [](std::vector<AutotuneResult>& results) {
        std::reverse(results.begin(), results.end());
    };

    std::vector<AutotuneResult> results;
    const std::unordered_map<int64_t, void*> variantPack = {{1, reinterpret_cast<void*>(0x1)},
                                                            {2, reinterpret_cast<void*>(0x2)},
                                                            {3, reinterpret_cast<void*>(0x3)},
                                                            {4, reinterpret_cast<void*>(0x4)},
                                                            {5, reinterpret_cast<void*>(0x5)}};
    auto result = graph.autotune(_handle, variantPack, nullptr, 0, config, {}, &results);
    ASSERT_TRUE(result.is_good()) << result.get_message();
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(graph.getActivePlanIndex(), 1u);
    EXPECT_TRUE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -3);

    expectEngineSupportsPlanSerialization(*_mockBackend);
    const std::vector<uint8_t> fakeContainerBytes = {0x40, 0x41, 0x42};
    expectComboContainerSerialization(*_mockBackend, fakeContainerBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeContainerBytes);
}

TEST_F(TestGraph, PlanSpecAutotuneFailureLeavesCompiledPlanUnfinalized)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    createBasicBatchnormGraph(graph);
    ASSERT_TRUE(graph.validate().is_good());
    ASSERT_TRUE(graph.build_operation_graph(_handle).is_good());
    setupEngineWithNoKnobs(*_mockBackend);

    ON_CALL(*_mockBackend,
            backendGetAttribute(
                _, HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t*,
                          void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 1024;
            return HIPDNN_STATUS_SUCCESS;
        });

    EngineConfigInfo engine;
    engine.engineId = -2;
    engine.estimatedWorkspaceSize = 1024;
    auto addResult = graph.add_engine_configs({engine});
    ASSERT_TRUE(addResult.is_good()) << addResult.get_message();

    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 0;

    std::vector<AutotuneResult> results;
    const std::unordered_map<int64_t, void*> variantPack = {{1, reinterpret_cast<void*>(0x1)},
                                                            {2, reinterpret_cast<void*>(0x2)},
                                                            {3, reinterpret_cast<void*>(0x3)},
                                                            {4, reinterpret_cast<void*>(0x4)},
                                                            {5, reinterpret_cast<void*>(0x5)}};

    auto result = graph.autotune(_handle, variantPack, nullptr, 2048, config, {}, &results);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(graph.getActivePlanIndex(), 0u);
    EXPECT_FALSE(graph.isExecutionPlanFinalized());
    ASSERT_TRUE(graph.selectedEngineIdForTest().has_value());
    EXPECT_EQ(*graph.selectedEngineIdForTest(), -2);

    const std::vector<uint8_t> fakeGraphBytes = {0x66, 0x77, 0x88};
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, _, _, _, _))
        .Times(0);
    expectGraphOnlySerialization(*_mockBackend, fakeGraphBytes);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, fakeGraphBytes);
}

TEST_F(TestGraph, SerializeCompiledPlanRejectsUnfinalizedActivePlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle, /*buildPlans=*/false);
    ASSERT_TRUE(graph.hasExecutionPlan());
    ASSERT_FALSE(graph.isExecutionPlanFinalized());

    EXPECT_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(_, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize_compiled_plan(data);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

// The combo size-query succeeds but reports a zero-length blob: serialize fails
// with a descriptive error and never issues the fill-call.
TEST_F(TestGraph, SerializeWithPlanRejectsZeroLengthCombo)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    ASSERT_TRUE(graph.hasExecutionPlan());
    // Select an engine and report the serialization note so serialize() takes
    // the combo path (two-call count/fill behavior-note query).
    graph.setSelectedEngineId(10);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    // The size-query succeeds but reports zero bytes; the fill-call must not run.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendDescriptor_t,
                     size_t,
                     size_t* blobByteSize,
                     uint8_t*) {
            *blobByteSize = 0;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, Gt(size_t{0}), _, _))
        .Times(0);

    std::vector<uint8_t> data;
    auto result = graph.serialize(data);

    EXPECT_TRUE(result.is_bad());
    EXPECT_NE(result.get_message().find("zero-length binary graph and plan"), std::string::npos)
        << result.get_message();
}

// The combo size-query failure is propagated.
TEST_F(TestGraph, SerializePropagatesComboSizeQueryFailure)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    // Select an engine and report the serialization note so serialize() takes
    // the combo path (two-call count/fill behavior-note query).
    graph.setSelectedEngineId(10);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<uint8_t> data;
    auto result = graph.serialize(data);

    EXPECT_TRUE(result.is_bad());
}

// The combo fill-call failure (size succeeds) is propagated.
TEST_F(TestGraph, SerializePropagatesComboFillFailure)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    // Select an engine and report the serialization note so serialize() takes
    // the combo path (two-call count/fill behavior-note query).
    graph.setSelectedEngineId(10);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendDescriptor_t,
                          size_t requestedByteSize,
                          size_t* blobByteSize,
                          uint8_t*) {
            *blobByteSize = 16;
            if(requestedByteSize > 0)
            {
                return HIPDNN_STATUS_INTERNAL_ERROR;
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<uint8_t> data;
    auto result = graph.serialize(data);

    EXPECT_TRUE(result.is_bad());
}

// A plan is built but no engine id is captured (e.g. the engine could not be
// recovered). Serialize must short-circuit on the gate's
// _selectedEngineId.has_value()==false branch -- without any behavior-note query --
// and fall through to the byte-identical legacy bare-graph blob, never invoking the
// combo container API. (The engine-reports-no-note path is covered separately by
// SerializeWithPlanQueriesEngineAndUsesGraphOnlyWhenUnsupported.)
TEST_F(TestGraph, SerializeWithPlanButNoSelectedEngineFallsBackToGraphOnly)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    // Clear the captured engine id so serialize treats the plan as not
    // serializable without a behavior-note query, forcing the graph-only path.
    ASSERT_TRUE(graph.hasExecutionPlan());
    graph.clearSelectedEngineId();

    const std::vector<uint8_t> fakeGraphBytes = {0x01, 0x02, 0x03};

    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&fakeGraphBytes](hipdnnBackendDescriptor_t,
                                         size_t requestedSize,
                                         size_t* graphByteSize,
                                         uint8_t* data) {
            *graphByteSize = fakeGraphBytes.size();
            if(data != nullptr && requestedSize >= fakeGraphBytes.size())
            {
                std::memcpy(data, fakeGraphBytes.data(), fakeGraphBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The combo container API must not be touched when the engine cannot
    // serialize the plan; the graph-only fallback is taken instead.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);

    EXPECT_TRUE(err.is_good()) << err.get_message();
    // Output is the bare-graph bytes: the graph-only path was taken.
    EXPECT_EQ(data, fakeGraphBytes);
}

// A plan exists but its engine cannot serialize it, so serialize falls through to
// the bare-graph path; the graph size-query then reports zero bytes and serialize
// fails with a descriptive error.
TEST_F(TestGraph, SerializeGraphOnlyRejectsZeroLengthGraphAfterPlanFallback)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    // Clear the captured engine id so serialize treats the plan as not
    // serializable without a behavior-note query, forcing the graph-only path.
    ASSERT_TRUE(graph.hasExecutionPlan());
    graph.clearSelectedEngineId();

    // The graph size-query succeeds but reports zero bytes.
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t, size_t, size_t* graphByteSize, uint8_t*) {
            *graphByteSize = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    // The combo container API must not be touched on the graph-only fallback.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto result = graph.serialize(data);

    EXPECT_TRUE(result.is_bad());
    EXPECT_NE(result.get_message().find("zero-length binary graph"), std::string::npos)
        << result.get_message();
}

// Exercises the real engineSupportsPlanSerialization() branch (freshly built
// plan, no test seam). The selected engine reports the serialization behavior
// note, so serialize() queries the engine, finds it supported, and takes the
// combo path; the bare graph serializer must not be invoked.
TEST_F(TestGraph, SerializeWithPlanQueriesEngineAndUsesComboWhenSupported)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    ASSERT_TRUE(graph.hasExecutionPlan());

    // The selected engine reports the serialization note (two-call count/fill).
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 1, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void* arrayOfElements) {
            *elementCount = 1;
            auto* notes = static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements);
            notes[0] = HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> combo = {0xAA, 0xBB, 0xCC};
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _))
        .WillByDefault([&combo](hipdnnBackendDescriptor_t,
                                hipdnnBackendDescriptor_t,
                                size_t requestedByteSize,
                                size_t* blobByteSize,
                                uint8_t* serializedBlob) {
            *blobByteSize = combo.size();
            if(serializedBlob != nullptr && requestedByteSize >= combo.size())
            {
                std::memcpy(serializedBlob, combo.data(), combo.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The supported engine takes the combo path; the bare serializer is bypassed.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, combo);
}

// Companion to the above: the selected engine reports no behavior notes, so
// serialize() queries the engine, finds the plan not serializable, and falls
// back to the bare graph serializer; the combo API must not be invoked.
TEST_F(TestGraph, SerializeWithPlanQueriesEngineAndUsesGraphOnlyWhenUnsupported)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    ASSERT_TRUE(graph.hasExecutionPlan());

    // The selected engine reports no notes (count-call returns 0).
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> bare = {0x01, 0x02, 0x03};
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&bare](hipdnnBackendDescriptor_t,
                               size_t requestedSize,
                               size_t* graphByteSize,
                               uint8_t* data) {
            *graphByteSize = bare.size();
            if(data != nullptr && requestedSize >= bare.size())
            {
                std::memcpy(data, bare.data(), bare.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The unsupported engine takes the graph-only fallback; the combo API is
    // never invoked.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, bare);
}

// The behavior-note query itself fails (count-call returns an error), so
// engineSupportsPlanSerialization() treats the plan as not serializable and
// serialize() falls back to the bare graph serializer; the combo API is never
// invoked.
TEST_F(TestGraph, SerializeWithPlanBehaviorNoteQueryFailureFallsBackToGraphOnly)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    buildGraphWithExecutionPlan(graph, *_mockBackend, _handle);
    ASSERT_TRUE(graph.hasExecutionPlan());

    // Keep the captured engine id so serialize queries the behavior note; the
    // count-call fails, driving engineSupportsPlanSerialization() to false.
    graph.setSelectedEngineId(10);
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE, HIPDNN_TYPE_BEHAVIOR_NOTE, 0, _, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::vector<uint8_t> bare = {0x01, 0x02, 0x03};
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&bare](hipdnnBackendDescriptor_t,
                               size_t requestedSize,
                               size_t* graphByteSize,
                               uint8_t* data) {
            *graphByteSize = bare.size();
            if(data != nullptr && requestedSize >= bare.size())
            {
                std::memcpy(data, bare.data(), bare.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    // The failed capability probe takes the graph-only fallback; the combo API is
    // never invoked.
    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> data;
    auto err = graph.serialize(data);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(data, bare);
}

// A GRAPH-only contents flag leaves the execution plan null and never calls the
// plan-deserialize wrapper.
TEST_F(TestGraph, DeserializeNoPlanLeavesExecutionPlanNull)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags = HIPDNN_SERIALIZED_CONTENT_GRAPH;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _)).Times(0);

    const std::vector<uint8_t> data = {0x10, 0x11, 0x12, 0x13};
    GraphTestUtils graph2;
    auto result = graph2.deserialize(_handle, data);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_FALSE(graph2.hasExecutionPlan());
    EXPECT_FALSE(graph2.getPrivateGraphSubnodes().empty());
}

// An EXECUTION_PLAN flag plus a handle attaches the plan; the plan-deserialize
// wrapper receives the whole blob.
TEST_F(TestGraph, DeserializeWithPlanFlagAndHandleAttachesPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags
                = HIPDNN_SERIALIZED_CONTENT_GRAPH | HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> data = {0x20, 0x21, 0x22, 0x23, 0x24};
    auto fakePlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x7777);

    const uint8_t* capturedPlanData = nullptr;
    size_t capturedPlanSize = 0;
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_handle, _, _, _))
        .WillOnce([&](hipdnnHandle_t,
                      hipdnnBackendDescriptor_t* descriptor,
                      const uint8_t* serializedPlan,
                      size_t planByteSize) {
            capturedPlanData = serializedPlan;
            capturedPlanSize = planByteSize;
            *descriptor = fakePlan;
            return HIPDNN_STATUS_SUCCESS;
        });

    // The attach path recovers the engine backing the plan so serialize() can
    // re-query its plan-serialization capability.
    ON_CALL(*_mockBackend,
            backendGetAttribute(fakePlan,
                                HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT,
                                HIPDNN_TYPE_INT64,
                                1,
                                _,
                                _))
        .WillByDefault([](hipdnnBackendDescriptor_t,
                          hipdnnBackendAttributeName_t,
                          hipdnnBackendAttributeType_t,
                          int64_t,
                          int64_t* elementCount,
                          void* arrayOfElements) {
            if(elementCount != nullptr)
            {
                *elementCount = 1;
            }
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    GraphTestUtils graph2;
    auto result = graph2.deserialize(_handle, data);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(graph2.hasExecutionPlan());
    // The backend extracts the plan blob from the container, so the whole blob
    // is handed to the plan-deserialize wrapper.
    EXPECT_EQ(capturedPlanData, data.data());
    EXPECT_EQ(capturedPlanSize, data.size());
}

// Mirrors DeserializeWithPlanFlagAndHandleAttachesPlan but the engine-id query on
// the attached plan FAILS. The plan still attaches, but with no engine recovered
// _selectedEngineId stays unset, so a subsequent serialize() must degrade to the
// graph-only path (Graph.hpp's failure branch): the combo container API is never
// invoked and the bare-graph bytes are emitted.
TEST_F(TestGraph, DeserializeAttachWithEngineIdQueryFailureSerializesGraphOnly)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags
                = HIPDNN_SERIALIZED_CONTENT_GRAPH | HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN;
            return HIPDNN_STATUS_SUCCESS;
        });

    const std::vector<uint8_t> data = {0x20, 0x21, 0x22, 0x23, 0x24};
    auto fakePlan = reinterpret_cast<hipdnnBackendDescriptor_t>(0x7777);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_handle, _, _, _))
        .WillByDefault(
            [&](hipdnnHandle_t, hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
                *descriptor = fakePlan;
                return HIPDNN_STATUS_SUCCESS;
            });

    // The engine-id recovery on attach fails, so _selectedEngineId stays unset.
    ON_CALL(*_mockBackend,
            backendGetAttribute(fakePlan,
                                HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT,
                                HIPDNN_TYPE_INT64,
                                1,
                                _,
                                _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    GraphTestUtils graph2;
    ASSERT_TRUE(graph2.deserialize(_handle, data).is_good());
    // The plan attaches even though its engine could not be recovered.
    ASSERT_TRUE(graph2.hasExecutionPlan());

    // Serialize must take the graph-only path: no combo container call, just the
    // bare-graph serializer.
    const std::vector<uint8_t> fakeGraphBytes = {0x01, 0x02, 0x03};
    ON_CALL(*_mockBackend, backendGetSerializedBinaryGraphExt(_, _, _, _))
        .WillByDefault([&fakeGraphBytes](hipdnnBackendDescriptor_t,
                                         size_t requestedSize,
                                         size_t* graphByteSize,
                                         uint8_t* out) {
            *graphByteSize = fakeGraphBytes.size();
            if(out != nullptr && requestedSize >= fakeGraphBytes.size())
            {
                std::memcpy(out, fakeGraphBytes.data(), fakeGraphBytes.size());
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendGetSerializedBinaryGraphAndPlanExt(_, _, _, _, _)).Times(0);

    std::vector<uint8_t> out;
    auto err = graph2.serialize(out);
    EXPECT_TRUE(err.is_good()) << err.get_message();
    EXPECT_EQ(out, fakeGraphBytes);
}

// An EXECUTION_PLAN flag but no handle drops the plan and never calls the
// plan-deserialize wrapper. (Warning text is asserted at the integration level.)
TEST_F(TestGraph, DeserializeWithPlanFlagNoHandleDropsPlanAndWarns)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags
                = HIPDNN_SERIALIZED_CONTENT_GRAPH | HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _)).Times(0);

    const std::vector<uint8_t> data = {0x30, 0x31, 0x32, 0x33};
    GraphTestUtils graph2;
    auto result = graph2.deserialize(data);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_FALSE(graph2.hasExecutionPlan());
}

// A failure of the contents-query wrapper is propagated.
TEST_F(TestGraph, DeserializePropagatesContentsQueryFailure)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::vector<uint8_t> data = {0x40, 0x41, 0x42, 0x43};
    GraphTestUtils graph2;
    auto result = graph2.deserialize(_handle, data);

    EXPECT_TRUE(result.is_bad());
}

// A failure of the plan-deserialize wrapper is propagated.
TEST_F(TestGraph, DeserializePropagatesPlanDeserializeFailure)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags
                = HIPDNN_SERIALIZED_CONTENT_GRAPH | HIPDNN_SERIALIZED_CONTENT_EXECUTION_PLAN;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    const std::vector<uint8_t> data = {0x50, 0x51, 0x52, 0x53};
    GraphTestUtils graph2;
    auto result = graph2.deserialize(_handle, data);

    EXPECT_TRUE(result.is_bad());
}

// A legacy bare-graph blob reconstructs the graph only; the backend reports
// GRAPH-only contents and the plan-deserialize wrapper is never called.
TEST_F(TestGraph, DeserializeLegacyBareGraphBlobReconstructsGraphNoPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    setupSuccessfulPointwiseGraphUnpack(*_mockBackend);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(0xA000);
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendGetSerializedBinaryContentsExt(_, _, _))
        .WillByDefault([](const uint8_t*, size_t, int* contentFlags) {
            *contentFlags = HIPDNN_SERIALIZED_CONTENT_GRAPH;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeExecutionPlanExt(_, _, _, _)).Times(0);

    const std::vector<uint8_t> data = {0xBB, 0xBB, 0xBB, 0xBB};
    GraphTestUtils graph2;
    auto result = graph2.deserialize(_handle, data);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_FALSE(graph2.getPrivateGraphSubnodes().empty());
    EXPECT_FALSE(graph2.hasExecutionPlan());
}
