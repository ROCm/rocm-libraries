// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend/graph_properties.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file sdpa_attributes.h
 * @brief cuDNN FE SDPA attribute types, aliased to hipDNN's SDPA attribute types.
 */

#pragma once

#include <functional>
#include <memory>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

// Unconditional: only the SDPA node methods are gated on HIPDNN_ENABLE_SDPA, not
// these attribute types.
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming)

class Graph; // completed in detail/graph_wrapper.h

/// @brief Accepted for source compatibility; any `set_score_mod()` use fails at validate().
/// @note Namespace-scope here; upstream declares it inside SDPA_attributes.
using AttentionScoreModifier_t = std::function<std::shared_ptr<Tensor_attributes>(
    std::shared_ptr<Graph>, std::shared_ptr<Tensor_attributes>)>;

/// @brief Alias of hipdnn_frontend::graph::SdpaAttributes.
using SDPA_attributes = hipdnn_frontend::graph::SdpaAttributes;
/// @brief Alias of hipdnn_frontend::graph::SdpaBackwardAttributes.
using SDPA_backward_attributes = hipdnn_frontend::graph::SdpaBackwardAttributes;

/// @brief Alias of SDPA_attributes, as upstream. The sdpa_fp8 nodes are unsupported.
using SDPA_fp8_attributes = SDPA_attributes;
/// @brief Alias of SDPA_backward_attributes; upstream has a distinct class. The
/// sdpa_fp8_backward nodes are unsupported.
using SDPA_fp8_backward_attributes = SDPA_backward_attributes;

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
