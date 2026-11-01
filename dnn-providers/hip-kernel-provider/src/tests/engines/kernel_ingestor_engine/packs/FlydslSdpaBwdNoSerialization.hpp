// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// Pins that a FlyDSL SDPA backward engine, merged or split, neither advertises nor performs
/// plan saving. Remove or invert it when RFC 0021 slot-program saving lands (split) or the
/// merged dispatch handler implements save/restore.
///
/// The engine's behavior notes exclude the serialization note, authored or computed, and no
/// dispatch handler of its packs (or its slots' packs) saves, which is what makes
/// captureIngestorPlan() refuse a plan over its kernels ("does not support saving"; the
/// capture refusals themselves are TestIngestorPlanCapture's). Building a plan over a real
/// kernel loads its code object and needs a device, so no plan is saved here.
inline void
    expectEngineRefusesPlanSerialization(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    constexpr auto NOTE
        = static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
    const auto hasNote = [&](const std::vector<int32_t>& notes) {
        return std::find(notes.begin(), notes.end(), NOTE) != notes.end();
    };

    EXPECT_FALSE(hasNote(set.engine.behaviorNotes)) << "the descriptor authors the note";
    EXPECT_FALSE(supportsExecutionPlanSerialization(set));
    EXPECT_FALSE(hasNote(engineWithComputedBehaviorNotes(set).behaviorNotes));

    // The packs that hold kernels: the engine's own, or each slot's.
    std::vector<const KernelDescriptorPack*> packs;
    for(const auto& pack : set.packs)
    {
        packs.push_back(&pack);
    }
    for(const auto& slot : set.slots)
    {
        for(const auto& pack : slot.packs)
        {
            packs.push_back(&pack);
        }
    }
    ASSERT_FALSE(packs.empty());

    registerNativeIngestorSymbols();
    for(const auto* pack : packs)
    {
        const auto dispatch
            = std::find_if(set.dispatches.begin(),
                           set.dispatches.end(),
                           [&](const auto& candidate) { return candidate.id == pack->dispatchId; });
        ASSERT_NE(dispatch, set.dispatches.end());
        const auto* handler = DispatchRegistry<Handle>::tryResolve(dispatch->dispatchSymbol);
        ASSERT_NE(handler, nullptr) << dispatch->dispatchSymbol;
        EXPECT_FALSE(handler->supportsSaving()) << dispatch->dispatchSymbol;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
