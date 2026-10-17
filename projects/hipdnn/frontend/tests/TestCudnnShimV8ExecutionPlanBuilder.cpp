// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x ExecutionPlanBuilder against the shim's backend C API and the
// mock hipDNN backend: building without a plan handle, tags, notes, JSON and
// copies.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <nlohmann/json.hpp>
#endif

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

class TestCudnnShimV8ExecutionPlanBuilder : public hipdnn_shim_test::ShimV8PlanFixture
{
};

// Upstream's build() never sets CUDNN_ATTR_EXECUTION_PLAN_HANDLE; the plan runs
// on the graph's handle.
TEST_F(TestCudnnShimV8ExecutionPlanBuilder, PlanBuildsWithoutPlanHandle)
{
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);

    const fe::ExecutionPlan_v8 plan
        = fe::ExecutionPlanBuilder_v8().setEngineConfig(configs[1], graph.getTag()).build();

    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    EXPECT_EQ(plan.getTag(), "ConvFwd_eng1");
    const fe::VariantPack_v8 pack = variantPack();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS) << pack.get_error();

    EXPECT_EQ(cudnnBackendExecute(_handle, plan.get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_SUCCESS)
        << hipdnn_shim_test::lastShimError();
    ASSERT_EQ(executions().size(), 1U);
    EXPECT_EQ(static_cast<void*>(executions().back().handle), static_cast<void*>(_handle));
    EXPECT_EQ(executedEngineIds(), std::vector<std::int64_t>{1});

    // Another handle is still rejected.
    auto* const otherHandle = reinterpret_cast<cudnnHandle_t>(0x23456789);
    EXPECT_EQ(cudnnBackendExecute(otherHandle, plan.get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(executions().size(), 1U);
}

TEST_F(TestCudnnShimV8ExecutionPlanBuilder, PlanTagFormat)
{
    installEngineKnobs(2, {{"prov.tile_size", 8, 0, 64, 8, {}}});
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);

    EXPECT_EQ(planFor(configs[1], graph.getTag()).getTag(), "ConvFwd_eng1");
    // Without a tag, as PyTorch's quantized ops build.
    EXPECT_EQ(planFor(configs[1], "").getTag(), "eng1");
    EXPECT_EQ(planFor(graph, 0).getTag(), "ConvFwd_eng0");

    fe::Engine_v8 knobbed = engine(graph, 2);
    ASSERT_EQ(knobbed.getSupportedKnobs().size(), 1U);
    knobbed.getSupportedKnobs().front().setChoice(16);
    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(knobbed).build();
    const fe::ExecutionPlan_v8 plan = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    // The knob segment streams the shim's cudnnBackendKnobType_t integer.
    EXPECT_EQ(plan.getTag(),
              "ConvFwd_eng2_k" + std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_TILE_SIZE))
                  + "=16");
}

TEST_F(TestCudnnShimV8ExecutionPlanBuilder, PlanNotesAndWorkspace)
{
    const std::int64_t deterministic = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    const std::vector<std::int64_t> ids = {deterministic, 9};
    installHeuristicResults(ids);
    installBehaviorNotes(9, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    const fe::OperationGraph_v8 graph = convGraph();

    const fe::ExecutionPlan_v8 nondeterministic = planFor(graph, denseIndex(ids, 9));
    ASSERT_EQ(nondeterministic.get_status(), CUDNN_STATUS_SUCCESS) << nondeterministic.get_error();
    EXPECT_EQ(nondeterministic.getAllNumericNotes(),
              std::vector<cudnnBackendNumericalNote_t>{CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC});
    const auto& numeric = nondeterministic.getNumericNotes();
    EXPECT_EQ(numeric.front(), CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC);
    EXPECT_TRUE(std::all_of(numeric.begin() + 1, numeric.end(), [](auto note) {
        return note == CUDNN_NUMERICAL_NOTE_TYPE_COUNT;
    }));
    EXPECT_EQ(nondeterministic.getAllBehaviorNotes(),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(nondeterministic.getBehaviorNotes().front(), CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);
    EXPECT_EQ(nondeterministic.getWorkspaceSize(), WORKSPACE_SIZE);

    const fe::ExecutionPlan_v8 claimed = planFor(graph, denseIndex(ids, deterministic));
    ASSERT_EQ(claimed.get_status(), CUDNN_STATUS_SUCCESS) << claimed.get_error();
    EXPECT_TRUE(claimed.getAllNumericNotes().empty());
    EXPECT_EQ(claimed.getNumericNotes().front(), CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
    EXPECT_TRUE(claimed.getAllBehaviorNotes().empty());
    EXPECT_EQ(claimed.getWorkspaceSize(), WORKSPACE_SIZE);
}

// PyTorch stores plans by value; copies must keep using one descriptor.
TEST_F(TestCudnnShimV8ExecutionPlanBuilder, CopiedPlansShareDescriptor)
{
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::ExecutionPlan_v8 plan = planFor(graph, 1);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();

    const std::vector<fe::ExecutionPlan_v8> stored(1, plan);
    fe::ExecutionPlan_v8 assigned = planFor(graph, 2);
    assigned = stored.front();

    EXPECT_EQ(stored.front().get_raw_desc(), plan.get_raw_desc());
    EXPECT_EQ(assigned.get_raw_desc(), plan.get_raw_desc());
    EXPECT_EQ(assigned.getTag(), plan.getTag());
    EXPECT_EQ(assigned.getWorkspaceSize(), plan.getWorkspaceSize());
    EXPECT_EQ(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).size(), 2U);
}

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
// PyTorch parses the plan JSON for an sm120 errata predicate.
TEST_F(TestCudnnShimV8ExecutionPlanBuilder, JsonParsesWithNlohmann)
{
    installSerializedPlan({0x01, 0x02, 0x03});
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::ExecutionPlan_v8 plan = planFor(graph, 1);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();

    const std::string text = plan.getJsonRepresentation();
    ASSERT_FALSE(text.empty());
    EXPECT_EQ(text.back(), '\0');

    const nlohmann::json parsed = nlohmann::json::parse(text);
    ASSERT_TRUE(parsed.contains("engine"));
    EXPECT_TRUE(parsed["engine"].is_object());
    EXPECT_EQ(parsed["engine"].value("smVersion", -1), -1);
    EXPECT_EQ(parsed["engine"].value("globalIndex", -1), 1);
}
#endif

TEST_F(TestCudnnShimV8ExecutionPlanBuilder, LoadFromJsonRoundTrip)
{
    installSerializedPlan({0x01, 0x02, 0x03});
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::ExecutionPlan_v8 plan = planFor(graph, 2);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();

    const fe::ExecutionPlan_v8 loaded
        = fe::ExecutionPlanBuilder_v8().setHandle(_handle).loadFromJson(
            plan.getJsonRepresentation());

    ASSERT_EQ(loaded.get_status(), CUDNN_STATUS_SUCCESS) << loaded.get_error();
    // loadFromJson never seeds the tag with the graph's.
    EXPECT_EQ(loaded.getTag(), "eng2");
    EXPECT_EQ(loaded.getWorkspaceSize(), plan.getWorkspaceSize());
    EXPECT_EQ(loaded.getAllNumericNotes(), plan.getAllNumericNotes());
    EXPECT_NE(loaded.get_raw_desc(), plan.get_raw_desc());

    fe::ExecutionPlanBuilder_v8 noHandle;
    const BuilderFailure failure = builderFailureOf([&noHandle, &plan] {
        [[maybe_unused]] auto built = noHandle.loadFromJson(plan.getJsonRepresentation());
    });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8ExecutionPlanBuilder, KernelCacheNotSupported)
{
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    const std::size_t plansBefore = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).size();

    fe::ExecutionPlanBuilder_v8 builder;
    builder.setEngineConfig(configs[0], graph.getTag())
        .setKernelCache(std::make_shared<fe::KernelCache>());
    const BuilderFailure failure
        = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });

    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("KERNEL_CACHE"), std::string::npos) << failure.what;
    EXPECT_EQ(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).size(), plansBefore);
}

} // namespace
