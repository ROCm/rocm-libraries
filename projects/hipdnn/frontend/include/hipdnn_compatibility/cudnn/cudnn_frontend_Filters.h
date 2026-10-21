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
 * @brief cuDNN v8 engine-config `filter` and the numerical/behavior note predicates over hipDNN.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/v8/notes.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Filters.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,performance-unnecessary-value-param):
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

// `engine_config` is the raw hipDNN ENGINECFG of a config; any failure answers false, as upstream.
template <cudnnBackendNumericalNote_t NUMERIC_NOTE>
bool hasNumericalNote(cudnnBackendDescriptor_t engine_config)
{
    return detail::v8::engineConfigHasNumericalNote(engine_config, NUMERIC_NOTE);
}

template <cudnnBackendBehaviorNote_t BEHAVIOR_NOTE>
bool hasBehaviorNote(cudnnBackendDescriptor_t engine_config)
{
    return detail::v8::engineConfigHasBehaviorNote(engine_config, BEHAVIOR_NOTE);
}

// NOLINTEND(readability-identifier-naming,performance-unnecessary-value-param)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
