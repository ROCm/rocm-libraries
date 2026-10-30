// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Mock-backend fixture for the ported v0.x engine, plan and autotune
// builders: a conv graph over engines whose dense index equals their ID, and
// helpers to build configs, plans and variant packs through the builders only.
#pragma once

#include "ShimV8BuilderFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

class ShimV8BuilderPlanFixture : public ShimV8BuilderFixture
{
protected:
    static constexpr std::int64_t WORKSPACE_SIZE = 4096;
    static constexpr std::int64_t X_UID = 1;
    static constexpr std::int64_t W_UID = 2;
    static constexpr std::int64_t Y_UID = 3;

    void SetUp() override
    {
        ShimV8BuilderFixture::SetUp();
        installPlanWorkspace(WORKSPACE_SIZE);
        installEngines(3);
    }

    // Engines 0..count-1, ranked in ID order, so index i names engine i.
    void installEngines(std::int64_t count)
    {
        std::vector<std::int64_t> ids(static_cast<std::size_t>(count));
        std::iota(ids.begin(), ids.end(), std::int64_t{0});
        installHeuristicResults(std::move(ids));
    }

    // Dense index of `id` among `ids` (the shim numbers engines in ID order); an `id` that is
    // not among them fails the test and reads as -1.
    static std::int64_t denseIndex(std::vector<std::int64_t> ids, std::int64_t id)
    {
        std::sort(ids.begin(), ids.end());
        const auto found = std::lower_bound(ids.begin(), ids.end(), id);
        if(found == ids.end() || *found != id)
        {
            ADD_FAILURE() << "engine " << id << " is not in the engine space";
            return -1;
        }
        return static_cast<std::int64_t>(std::distance(ids.begin(), found));
    }

    // TILE_SIZE is the only knob the shim can show: BLOCK_SIZE has explicit valid
    // values, "int_knob" has no cuDNN type, and the two STAGES knobs collide.
    static std::vector<MockKnob> mixedKnobs()
    {
        return {
            {"prov.tile_size", 8, 0, 64, 8, {}},
            {"prov.block_size", 8, 0, 64, 1, {8, 16}},
            {"prov.int_knob", 1, 0, 10, 1, {}},
            {"first.stages", 1, 1, 4, 1, {}},
            {"second.stages", 1, 1, 4, 1, {}},
        };
    }

    // The tag segment a plan gets for one knob choice.
    static std::string knobSegment(cudnnBackendKnobType_t type, std::int64_t value)
    {
        return "_k" + std::to_string(static_cast<int>(type)) + "=" + std::to_string(value);
    }

    // Sets the choice of the reported knob of `type`, which must exist.
    static void choose(v8fe::Engine_v8& built, cudnnBackendKnobType_t type, std::int64_t choice)
    {
        auto& knobs = built.getSupportedKnobs();
        const auto knob = std::find_if(knobs.begin(), knobs.end(), [type](const auto& candidate) {
            return candidate.getKnobType() == type;
        });
        ASSERT_NE(knob, knobs.end());
        knob->setChoice(static_cast<std::uint64_t>(choice));
    }

    static v8fe::Operation_v8 convForward(const Dims& xDims, const Dims& wDims, const Dims& yDims)
    {
        return convolutionForward(
            tensor(X_UID, xDims), tensor(W_UID, wDims), tensor(Y_UID, yDims), convDesc());
    }

    // conv(x {2,4,8,8}, w {8,4,3,3}) -> y {2,8,8,8}; tag "ConvFwd_".
    v8fe::OperationGraph_v8 convGraph(std::int64_t channels = 4)
    {
        return graphOf(convForward({2, channels, 8, 8}, {8, channels, 3, 3}, {2, 8, 8, 8}));
    }

    static v8fe::EngineConfigList engineConfigs(v8fe::OperationGraph_v8& graph)
    {
        v8fe::EngineHeuristics_v8 heuristics = v8fe::EngineHeuristicsBuilder_v8()
                                                   .setOperationGraph(graph)
                                                   .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                                   .build();
        return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    }

    static v8fe::Engine_v8 engine(const v8fe::OperationGraph_v8& graph, std::int64_t index)
    {
        return v8fe::EngineBuilder_v8().setOperationGraph(graph).setGlobalEngineIdx(index).build();
    }

    static v8fe::ExecutionPlan_v8 planFor(const v8fe::ManagedOpaqueDescriptor& config,
                                          const std::string& tag)
    {
        return v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config, tag).build();
    }

    // Built through EngineBuilder and EngineConfigBuilder, as legacy samples do.
    static v8fe::ExecutionPlan_v8 planFor(const v8fe::OperationGraph_v8& graph, std::int64_t index)
    {
        const v8fe::Engine_v8 built = engine(graph, index);
        const v8fe::EngineConfig_v8 config
            = v8fe::EngineConfigBuilder_v8().setEngine(built).build();
        return v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    }

    static std::vector<std::string> tagsOf(const std::vector<v8fe::ExecutionPlan_v8>& plans)
    {
        std::vector<std::string> tags;
        tags.reserve(plans.size());
        for(const v8fe::ExecutionPlan_v8& plan : plans)
        {
            tags.push_back(plan.getTag());
        }
        return tags;
    }

    v8fe::VariantPack_v8 variantPack()
    {
        std::array<void*, 3> pointers = {&_x, &_w, &_y};
        const std::array<std::int64_t, 3> uids = {X_UID, W_UID, Y_UID};
        return v8fe::VariantPackBuilder_v8()
            .setWorkspacePointer(_workspace.data())
            .setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data())
            .setUids(static_cast<std::int64_t>(uids.size()), uids.data())
            .build();
    }

private:
    float _x = 0.0F;
    float _w = 0.0F;
    float _y = 0.0F;
    std::array<std::byte, WORKSPACE_SIZE> _workspace{};
};

} // namespace hipdnn_shim_test
