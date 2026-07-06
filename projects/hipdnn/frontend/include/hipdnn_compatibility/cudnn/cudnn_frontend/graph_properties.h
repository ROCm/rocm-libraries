// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h), used under the MIT license.

/**
 * @file graph_properties.h
 * @brief Graph attribute-type aliases for the hipDNN cuDNN-compatibility shim.
 *
 * Brings hipDNN's v9 graph attribute types into `<shim_ns>::graph` by aliasing
 * them with `using` declarations — zero overhead, no shim-side state. A tensor
 * configured through the shim therefore *is* a hipDNN `TensorAttributes` and
 * flows into a wrapped hipDNN graph with no conversion (UID handling stays on
 * the hipDNN side).
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{

// hipDNN publishes cuDNN's `Tensor_attributes` spelling as a typedef; aliasing
// it lets consumer code using that name resolve through the shim.
using hipdnn_frontend::graph::Tensor_attributes;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
