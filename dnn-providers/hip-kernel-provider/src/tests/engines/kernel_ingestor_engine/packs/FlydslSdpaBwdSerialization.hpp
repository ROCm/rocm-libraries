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

/// Pins that a FlyDSL SDPA backward engine, merged or split, reports the plan serialization
/// note, and that every dispatch handler behind it saves and restores every launch.
///
/// The note is computed, never authored: the descriptor leaves it out, and the engine adds it
/// when every handler of every pack, and of every slot's packs, opts in. Building a plan over
/// a real kernel loads its code object and needs a device, so no plan is saved here; the
/// device round trips are TestFlydslSdpaBwdHandlerRestore.cpp's.
inline void
    expectEngineReportsPlanSerialization(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    constexpr auto NOTE
        = static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION);
    const auto hasNote = [&](const std::vector<int32_t>& notes) {
        return std::find(notes.begin(), notes.end(), NOTE) != notes.end();
    };

    EXPECT_FALSE(hasNote(set.engine.behaviorNotes)) << "the descriptor authors the note";
    EXPECT_TRUE(supportsExecutionPlanSerialization(set));
    EXPECT_TRUE(hasNote(engineWithComputedBehaviorNotes(set).behaviorNotes));

    // The packs that hold kernels: the engine's own, or each slot's.
    std::vector<const KernelDescriptorPack*> packs;
    size_t packCount = set.packs.size();
    for(const auto& slot : set.slots)
    {
        packCount += slot.packs.size();
    }
    packs.reserve(packCount);
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
        EXPECT_TRUE(handler->supportsSaving()) << dispatch->dispatchSymbol;
        EXPECT_TRUE(handler->supportsSavingAllLaunches()) << dispatch->dispatchSymbol;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
