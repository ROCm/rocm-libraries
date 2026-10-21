// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/engine_filters.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// hipDNN engines declare no numerical notes (its ENGINE_NUMERICAL_NOTE GET is
// NOT_SUPPORTED); only a determinism claim lifts NONDETERMINISTIC.
inline std::vector<cudnnBackendNumericalNote_t> numericalNotes(std::int64_t engineId)
{
    if(isDeterminismClaimed(engineId))
    {
        return {};
    }
    return {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC};
}

// Only notes with a cuDNN counterpart, in hipDNN's order.
inline std::vector<cudnnBackendBehaviorNote_t>
    cudnnBehaviorNotes(const std::vector<hipdnnBackendBehaviorNote_t>& hipdnnNotes)
{
    std::vector<cudnnBackendBehaviorNote_t> notes;
    for(const hipdnnBackendBehaviorNote_t note : hipdnnNotes)
    {
        const std::optional<BehaviorNote_t> mapped
            = ::hipdnn_frontend::fromHipdnnBehaviorNote(note);
        const std::optional<cudnnBackendBehaviorNote_t> cudnnNote
            = mapped ? toCudnn(*mapped) : std::nullopt;
        if(cudnnNote)
        {
            notes.push_back(*cudnnNote);
        }
    }
    return notes;
}

// The ENGINE behind a finalized hipDNN ENGINECFG, as a wrapper the caller
// owns; null (with hipDNN's error drained) when the GET fails.
inline ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
    engineOfConfig(hipdnnBackendDescriptor_t engineConfig)
{
    hipdnnBackendDescriptor_t engine = nullptr;
    if(::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
           engineConfig,
           HIPDNN_ATTR_ENGINECFG_ENGINE,
           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
           1,
           nullptr,
           static_cast<void*>(&engine))
       != HIPDNN_STATUS_SUCCESS)
    {
        takeHipdnnMessage();
        return {};
    }
    return {engine};
}

// Whether the engine of a raw hipDNN ENGINECFG has `note`, for filters that only
// see the raw descriptor. Any failure answers false, as upstream's helpers do.
inline bool engineConfigHasNumericalNote(hipdnnBackendDescriptor_t engineConfig,
                                         cudnnBackendNumericalNote_t note) noexcept
{
    if(note != CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC)
    {
        return false;
    }
    try
    {
        const ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor engine
            = engineOfConfig(engineConfig);
        if(engine.get() == nullptr)
        {
            return false;
        }
        std::int64_t id = 0;
        if(::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
               engine.get(), HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &id)
           != HIPDNN_STATUS_SUCCESS)
        {
            takeHipdnnMessage();
            return false;
        }
        return !isDeterminismClaimed(id);
    }
    catch(...)
    {
        return false;
    }
}

inline bool engineConfigHasBehaviorNote(hipdnnBackendDescriptor_t engineConfig,
                                        cudnnBackendBehaviorNote_t note) noexcept
{
    try
    {
        const ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor engine
            = engineOfConfig(engineConfig);
        if(engine.get() == nullptr)
        {
            return false;
        }
        std::vector<hipdnnBackendBehaviorNote_t> hipdnnNotes;
        if(::hipdnn_frontend::detail::getDescriptorAttrVec(engine.get(),
                                                           HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                                           HIPDNN_TYPE_BEHAVIOR_NOTE,
                                                           hipdnnNotes,
                                                           "engine behavior notes")
               .is_bad())
        {
            takeHipdnnMessage();
            return false;
        }
        const std::vector<cudnnBackendBehaviorNote_t> notes = cudnnBehaviorNotes(hipdnnNotes);
        return std::find(notes.begin(), notes.end(), note) != notes.end();
    }
    catch(...)
    {
        return false;
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
