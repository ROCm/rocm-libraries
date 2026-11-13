// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// Only notes with a cuDNN counterpart, in hipDNN's order. `context` prefixes
// the error when the GET fails. An older same-major backend does not know
// HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE and answers HIPDNN_STATUS_NOT_SUPPORTED; that
// status alone reads as {NONDETERMINISTIC}, the conservative claim for an engine
// that declares nothing. Any other failure throws.
inline std::vector<cudnnBackendNumericalNote_t>
    readCudnnNumericalNotes(hipdnnBackendDescriptor_t engine, const std::string& context)
{
    const auto backend = ::hipdnn_frontend::detail::hipdnnBackend();

    std::int64_t count = 0;
    hipdnnStatus_t status = backend->backendGetAttribute(
        engine, HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE, HIPDNN_TYPE_NUMERICAL_NOTE, 0, &count, nullptr);
    if(status == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        takeHipdnnMessage();
        return {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC};
    }
    checkHipdnn(status, context + ": failed to get count for engine numerical notes");
    if(count < 0)
    {
        throw CompatibilityError(CUDNN_STATUS_INTERNAL_ERROR,
                                 context + ": negative count for engine numerical notes");
    }

    std::vector<hipdnnBackendNumericalNote_t> hipdnnNotes;
    if(count > 0)
    {
        hipdnnNotes.resize(static_cast<std::size_t>(count));
        std::int64_t actualCount = 0;
        status = backend->backendGetAttribute(engine,
                                              HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                              HIPDNN_TYPE_NUMERICAL_NOTE,
                                              count,
                                              &actualCount,
                                              hipdnnNotes.data());
        if(status == HIPDNN_STATUS_NOT_SUPPORTED)
        {
            takeHipdnnMessage();
            return {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC};
        }
        checkHipdnn(status, context + ": failed to get engine numerical notes");
        if(actualCount != count)
        {
            throw CompatibilityError(CUDNN_STATUS_INTERNAL_ERROR,
                                     context
                                         + ": element count mismatch for engine numerical notes");
        }
    }

    std::vector<cudnnBackendNumericalNote_t> notes;
    for(const hipdnnBackendNumericalNote_t note : hipdnnNotes)
    {
        const std::optional<NumericalNote_t> mapped
            = ::hipdnn_frontend::fromHipdnnNumericalNote(note);
        const std::optional<cudnnBackendNumericalNote_t> cudnnNote
            = mapped ? toCudnn(*mapped) : std::nullopt;
        if(cudnnNote)
        {
            notes.push_back(*cudnnNote);
        }
    }
    return notes;
}

// Only notes with a cuDNN counterpart, in hipDNN's order. `context` prefixes
// the error when the GET fails.
inline std::vector<cudnnBackendBehaviorNote_t>
    readCudnnBehaviorNotes(hipdnnBackendDescriptor_t engine, const std::string& context)
{
    std::vector<hipdnnBackendBehaviorNote_t> hipdnnNotes;
    // getDescriptorAttrVec drains hipDNN's last error when the GET fails.
    checkFrontend(::hipdnn_frontend::detail::getDescriptorAttrVec(engine,
                                                                  HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                                                  HIPDNN_TYPE_BEHAVIOR_NOTE,
                                                                  hipdnnNotes,
                                                                  "engine behavior notes"),
                  context);
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

// The ENGINE behind a finalized hipDNN ENGINECFG, as a wrapper the caller owns.
inline ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
    engineOfConfig(hipdnnBackendDescriptor_t engineConfig)
{
    hipdnnBackendDescriptor_t engine = nullptr;
    checkHipdnn(::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
                    engineConfig,
                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    nullptr,
                    static_cast<void*>(&engine)),
                "reading the engine of a hipDNN engine config");
    return {engine};
}

inline std::int64_t engineGlobalIndex(hipdnnBackendDescriptor_t engine)
{
    std::int64_t id = 0;
    checkHipdnn(::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
                    engine, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &id),
                "reading a hipDNN engine ID");
    return id;
}

// Whether the engine of a raw hipDNN ENGINECFG has `note`, for filters that only
// see the raw descriptor. Any failure answers false, as upstream's helpers do.
inline bool engineConfigHasNumericalNote(hipdnnBackendDescriptor_t engineConfig,
                                         cudnnBackendNumericalNote_t note) noexcept
{
    bool found = false;
    return guarded([&] {
               const std::vector<cudnnBackendNumericalNote_t> notes = readCudnnNumericalNotes(
                   engineOfConfig(engineConfig).get(), "reading engine numerical notes");
               found = std::find(notes.begin(), notes.end(), note) != notes.end();
           })
           && found;
}

inline bool engineConfigHasBehaviorNote(hipdnnBackendDescriptor_t engineConfig,
                                        cudnnBackendBehaviorNote_t note) noexcept
{
    bool found = false;
    return guarded([&] {
               const std::vector<cudnnBackendBehaviorNote_t> notes = readCudnnBehaviorNotes(
                   engineOfConfig(engineConfig).get(), "reading engine behavior notes");
               found = std::find(notes.begin(), notes.end(), note) != notes.end();
           })
           && found;
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
