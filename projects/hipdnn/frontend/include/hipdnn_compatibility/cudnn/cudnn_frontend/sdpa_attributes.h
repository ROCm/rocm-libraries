// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h — SDPA_attributes /
// SDPA_backward_attributes), used under the MIT license.

/**
 * @file sdpa_attributes.h
 * @brief cuDNN-shaped SDPA attribute aliases for the hipDNN compatibility shim.
 *
 * Like the rest of the shim, `SDPA_attributes` / `SDPA_backward_attributes` are
 * zero-cost `using` aliases onto the native hipDNN frontend types. The native
 * `SdpaAttributes` / `SdpaBackwardAttributes` accept the cuDNN spelling,
 * overloads, and semantics directly (including the deprecated `set_is_inference`,
 * the fused `set_dropout` overloads, and a fail-loudly path for setters with no
 * hipDNN equivalent such as `set_score_mod`), so no composition wrapper or
 * per-setter forwarding is needed here.
 *
 * @note Internal-to-shim; included unconditionally by `detail/graph_wrapper.h`,
 * but the body compiles only under HIPDNN_ENABLE_SDPA.
 */

#pragma once

#include <functional>
#include <memory>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

#ifdef HIPDNN_ENABLE_SDPA

#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming)

class Graph; // completed in detail/graph_wrapper.h

// cuDNN FE's programmable score modifier is a std::function over the graph
// object. hipDNN has no equivalent; the native SDPA attribute types accept any
// callable via a templated set_score_mod and fail loudly at validate(). The type
// is retained for source compatibility with hipified cuDNN consumers.
using AttentionScoreModifier_t = std::function<std::shared_ptr<Tensor_attributes>(
    std::shared_ptr<Graph>, std::shared_ptr<Tensor_attributes>)>;

using SDPA_attributes = hipdnn_frontend::graph::SdpaAttributes;
using SDPA_backward_attributes = hipdnn_frontend::graph::SdpaBackwardAttributes;

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph

#endif // HIPDNN_ENABLE_SDPA
