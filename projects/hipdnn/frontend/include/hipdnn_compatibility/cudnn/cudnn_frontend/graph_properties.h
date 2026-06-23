// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h), used under the MIT license. The
// type name below intentionally mirrors cuDNN frontend so hipified v9 consumers
// compile unchanged.

/**
 * @file graph_properties.h
 * @brief Graph attribute-type aliases for the hipDNN cuDNN-compatibility shim.
 *
 * cuDNN frontend declares its v9 graph attribute structs (`Tensor_attributes`,
 * the per-node `*_attributes`, …) in `cudnn_frontend/graph_properties.h`. This
 * header mirrors that filename and brings the matching hipDNN types into
 * `<shim_ns>::graph` by **aliasing** them with `using` declarations — zero
 * overhead, no re-declaration, and no shim-side state (RFC 0012 §4.4.1).
 *
 * `cudnn_frontend::graph::Tensor_attributes` therefore *is*
 * `hipdnn_frontend::graph::TensorAttributes`: a tensor configured through the
 * shim flows into a wrapped hipDNN graph with no conversion, and tensor UID
 * handling stays entirely on the hipDNN side (`assignUnsetTensorUids()` at
 * build). The shim adds no per-tensor identity map or UID allocator
 * (RFC 0012 §4.4.1, §5.3).
 *
 * @par Setter coverage (RFC 0012 §4.4.1)
 * hipDNN's `TensorAttributes` publishes the cuDNN-named chained setters the v9
 * graph API uses: `set_dim`, `set_stride`, `set_data_type`, `set_uid`,
 * `set_is_virtual`, `set_output`, `set_name` — aliased here 1:1. Pass-by-value
 * tensors are expressed through hipDNN's typed `set_value(scalar)` /
 * scalar-constructor surface rather than a separate `set_is_pass_by_value(bool)`
 * flag. cuDNN FE's `set_is_pass_by_value` and `set_ragged_offset` have no
 * hipDNN counterpart yet (no ragged-tensor support); they are intentionally
 * out of this alias's scope and are added on the hipDNN side, then surfaced
 * here, when their consuming nodes land (RFC 0012 §7.x).
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{

// Tensor attributes aliased 1:1 (RFC 0012 §4.4.1). hipDNN publishes both the
// class name and cuDNN's `Tensor_attributes` spelling (a typedef); both are
// aliased so consumer code using either name resolves through the shim.
using hipdnn_frontend::graph::Tensor_attributes;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
