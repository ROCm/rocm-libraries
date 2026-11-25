// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The cuDNN v8 shim reads the MIOpen engines' numerical notes through the backend attribute:
// MIOPEN_ENGINE_DETERMINISTIC reports no NONDETERMINISTIC note, MIOPEN_ENGINE reports it, and
// filtering on the note selects the deterministic engine. That engine's determinism is covered
// by IntegrationGpuDeterministic.cpp.

#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Filters.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Heuristics.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "IntegrationCudnnShimFixture.hpp"

using namespace hipdnn_data_sdk::utilities;

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// ============================================================================
// Test Fixture for cuDNN v8-Shim Numerical Notes
// Manages the cuDNN-shim handle lifecycle
// ============================================================================

class IntegrationGpuCudnnV8ShimDeterministic
    : public miopen_plugin::test_utilities::CudnnShimFixture<::testing::Test>
{
protected:
    static fe::Tensor makeTensor(std::int64_t uid, const std::vector<int64_t>& dims)
    {
        const std::vector<int64_t> strides = generateStrides(dims);
        return fe::TensorBuilder()
            .setDataType(fe::DataType_t::FLOAT)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .build();
    }

    // FP32 3x3 convolution forward, x {2,4,8,8} * w {8,4,3,3} -> y {2,8,8,8}.
    fe::OperationGraph buildConvForwardGraph() const
    {
        const std::array<int64_t, 2> ones = {1, 1};
        const fe::ConvDesc conv = fe::ConvDescBuilder()
                                      .setComputeType(fe::DataType_t::FLOAT)
                                      .setMathMode(CUDNN_CROSS_CORRELATION)
                                      .setSpatialDimCount(2)
                                      .setPrePadding(2, ones.data())
                                      .setPostPadding(2, ones.data())
                                      .setDilation(2, ones.data())
                                      .setSpatialStride(2, ones.data())
                                      .build();
        const fe::Tensor x = makeTensor(1, {2, 4, 8, 8});
        const fe::Tensor w = makeTensor(2, {8, 4, 3, 3});
        const fe::Tensor y = makeTensor(3, {2, 8, 8, 8});
        const fe::Operation op
            = fe::OperationBuilder(fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .setxDesc(x)
                  .setwDesc(w)
                  .setyDesc(y)
                  .setcDesc(conv)
                  .build();
        std::array<const fe::Operation*, 1> ops = {&op};
        return fe::OperationGraphBuilder()
            .setHandle(_handle)
            .setOperationGraph(static_cast<std::int64_t>(ops.size()), ops.data())
            .build();
    }

    static fe::EngineConfigList listEngineConfigs(fe::OperationGraph& graph)
    {
        fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                              .setOperationGraph(graph)
                                              .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                              .build();
        return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    }

    static fe::ExecutionPlan buildPlan(const fe::ManagedOpaqueDescriptor& config,
                                       const fe::OperationGraph& graph)
    {
        return fe::ExecutionPlanBuilder().setEngineConfig(config, graph.getTag()).build();
    }

    // The v8 shim exposes a plan's hipDNN engine name through its JSON.
    static bool planIsEngine(const fe::ExecutionPlan& plan, const char* engineName)
    {
        return plan.getJsonRepresentation().find(std::string(R"("name":")") + engineName + "\"")
               != std::string::npos;
    }
};

TEST_F(IntegrationGpuCudnnV8ShimDeterministic, ConvForwardEnginesReportNumericalNotes)
{
    fe::OperationGraph graph = buildConvForwardGraph();
    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    const fe::EngineConfigList configs = listEngineConfigs(graph);

    bool sawDeterministic = false;
    bool sawNondeterministic = false;
    for(const fe::ManagedOpaqueDescriptor& config : configs)
    {
        const fe::ExecutionPlan plan = buildPlan(config, graph);
        if(plan.get_status() != CUDNN_STATUS_SUCCESS)
        {
            continue;
        }
        const bool nondeterministic = fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
            config->get_backend_descriptor());
        if(planIsEngine(plan, MIOPEN_ENGINE_DETERMINISTIC_NAME))
        {
            sawDeterministic = true;
            EXPECT_FALSE(nondeterministic);
        }
        else if(planIsEngine(plan, MIOPEN_ENGINE_NAME))
        {
            sawNondeterministic = true;
            EXPECT_TRUE(nondeterministic);
        }
    }
    EXPECT_TRUE(sawDeterministic) << MIOPEN_ENGINE_DETERMINISTIC_NAME << " was not listed";
    EXPECT_TRUE(sawNondeterministic) << MIOPEN_ENGINE_NAME << " was not listed";
}

TEST_F(IntegrationGpuCudnnV8ShimDeterministic, FilterNondeterministicSelectsDeterministicEngine)
{
    fe::OperationGraph graph = buildConvForwardGraph();
    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    fe::EngineConfigList all = listEngineConfigs(graph);
    fe::EngineConfigList kept;

    // Configs the predicate rejects land in `kept`.
    fe::filter(all, kept, [](cudnnBackendDescriptor_t config) {
        return fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config);
    });

    bool keptDeterministic = false;
    for(const fe::ManagedOpaqueDescriptor& config : kept)
    {
        const fe::ExecutionPlan plan = buildPlan(config, graph);
        if(plan.get_status() != CUDNN_STATUS_SUCCESS)
        {
            continue;
        }
        EXPECT_FALSE(planIsEngine(plan, MIOPEN_ENGINE_NAME));
        keptDeterministic |= planIsEngine(plan, MIOPEN_ENGINE_DETERMINISTIC_NAME);
    }
    EXPECT_TRUE(keptDeterministic);

    bool droppedNondeterministic = false;
    for(const fe::ManagedOpaqueDescriptor& config : all)
    {
        const fe::ExecutionPlan plan = buildPlan(config, graph);
        if(plan.get_status() != CUDNN_STATUS_SUCCESS)
        {
            continue;
        }
        EXPECT_FALSE(planIsEngine(plan, MIOPEN_ENGINE_DETERMINISTIC_NAME));
        droppedNondeterministic |= planIsEngine(plan, MIOPEN_ENGINE_NAME);
    }
    EXPECT_TRUE(droppedNondeterministic);

    // The first config that survives the filter builds a plan on the deterministic engine.
    ASSERT_FALSE(kept.empty());
    const fe::ExecutionPlan chosen = buildPlan(kept.front(), graph);
    ASSERT_EQ(chosen.get_status(), CUDNN_STATUS_SUCCESS) << chosen.get_error();
    EXPECT_TRUE(planIsEngine(chosen, MIOPEN_ENGINE_DETERMINISTIC_NAME));
    for(const cudnnBackendNumericalNote_t note : chosen.getAllNumericNotes())
    {
        EXPECT_NE(note, CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC);
    }
}

} // namespace
