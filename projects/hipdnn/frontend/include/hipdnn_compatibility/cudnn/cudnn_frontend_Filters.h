// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Filters.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Filters.h
 * @brief cuDNN v8 engine-config `filter` and the numerical/behavior note predicates over the
 *        shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Filters.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-avoid-c-arrays,performance-unnecessary-value-param,misc-const-correctness,bugprone-multi-level-implicit-pointer-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

// Configs for which filter_fn returns true stay in `from`; the rest move to `to`.
inline void filter(EngineConfigList& from,
                   EngineConfigList& to,
                   std::function<bool(cudnnBackendDescriptor_t)> filter_fn)
{
    auto p
        = std::stable_partition(from.begin(), from.end(), [filter_fn](ManagedOpaqueDescriptor& p) {
              return filter_fn(const_cast<cudnnBackendDescriptor_t>(p->get_backend_descriptor()));
          });
    to.insert(to.end(), std::make_move_iterator(p), std::make_move_iterator(from.end()));
    from.erase(p, from.end());
}

template <cudnnBackendNumericalNote_t NUMERIC_NOTE>
bool hasNumericalNote(cudnnBackendDescriptor_t engine_config)
{
    bool hasNumerics = false;
    auto status = CUDNN_STATUS_SUCCESS;
    ManagedOpaqueDescriptor engine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    cudnnBackendDescriptor_t engine_ = engine->get_backend_descriptor();
    int64_t engine_count = -1;
    status = detail::get_attribute(engine_config,
                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &engine_count,
                                   &engine_);
    if(status == CUDNN_STATUS_SUCCESS)
    {
        cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
        std::fill_n(notes, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
        int64_t elem_count = 0;
        detail::get_attribute(engine->get_backend_descriptor(),
                              CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                              CUDNN_TYPE_NUMERICAL_NOTE,
                              CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                              &elem_count,
                              notes);
        if(std::any_of(notes, notes + elem_count, [](cudnnBackendNumericalNote_t note) {
               return note == NUMERIC_NOTE;
           }))
        {
            hasNumerics = true;
        }
    }
    return hasNumerics;
}

template <cudnnBackendBehaviorNote_t BEHAVIOR_NOTE>
bool hasBehaviorNote(cudnnBackendDescriptor_t engine_config)
{
    bool hasBehavior = false;
    auto status = CUDNN_STATUS_SUCCESS;
    ManagedOpaqueDescriptor engine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    cudnnBackendDescriptor_t engine_ = engine->get_backend_descriptor();
    int64_t engine_count = -1;
    status = detail::get_attribute(engine_config,
                                   CUDNN_ATTR_ENGINECFG_ENGINE,
                                   CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &engine_count,
                                   &engine_);
    if(status == CUDNN_STATUS_SUCCESS)
    {
        cudnnBackendBehaviorNote_t notes[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
        std::fill_n(notes, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT);
        int64_t elem_count = 0;
        detail::get_attribute(engine->get_backend_descriptor(),
                              CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                              CUDNN_TYPE_BEHAVIOR_NOTE,
                              CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                              &elem_count,
                              notes);
        if(std::any_of(notes, notes + elem_count, [](cudnnBackendBehaviorNote_t note) {
               return note == BEHAVIOR_NOTE;
           }))
        {
            hasBehavior = true;
        }
    }
    return hasBehavior;
}

// NOLINTEND(readability-identifier-naming,modernize-avoid-c-arrays,performance-unnecessary-value-param,misc-const-correctness,bugprone-multi-level-implicit-pointer-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
