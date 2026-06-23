// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend_utils.h),
// used under the MIT license. The enum names below intentionally mirror the
// cuDNN frontend FE-namespace enums so hipified v9 consumers compile unchanged.

/**
 * @file cudnn_frontend_utils.h
 * @brief FE-namespace enum aliases for the hipDNN cuDNN-compatibility shim.
 *
 * cuDNN frontend uses two parallel enum families (RFC 0012 §4.3): the C-API
 * enums from `<cudnn.h>` and the FE-namespace C++ enums in `namespace
 * cudnn_frontend` (`DataType_t`, `PointwiseMode_t`, …). The v9 graph API
 * signatures use the **FE-namespace** family; this header makes those names
 * available under `<shim_ns>` by **aliasing** the matching hipDNN types with
 * `using` declarations — zero overhead, no re-declaration, and no numeric cast
 * between enum families (RFC 0012 §4.3, §4.3.1).
 *
 * hipDNN already publishes these enums as cuDNN-named `_t` typedefs (see
 * `hipdnn_frontend/Types.hpp`), each a name-superset of its cuDNN counterpart,
 * so the alias holds.
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

// FE-namespace enums that hipDNN publishes 1:1 (RFC 0012 §4.3). Aliased, not
// re-declared, so `cudnn_frontend::DataType_t` *is* `hipdnn_frontend::DataType_t`.
using hipdnn_frontend::AttentionImplementation_t;
using hipdnn_frontend::BehaviorNote_t;
using hipdnn_frontend::BuildPlanPolicy_t;
using hipdnn_frontend::ConvolutionMode_t;
using hipdnn_frontend::DataType_t;
using hipdnn_frontend::DiagonalAlignment_t;
using hipdnn_frontend::HeurMode_t;
using hipdnn_frontend::NormFwdPhase_t;
using hipdnn_frontend::PaddingMode_t;
using hipdnn_frontend::PointwiseMode_t;
using hipdnn_frontend::ReductionMode_t;
using hipdnn_frontend::ResampleMode_t;

// Other cuDNN FE-namespace enums (NumericalNote_t, NormMode_t, RngDistribution_t,
// DescriptorType_t, MoeGroupedMatmulMode_t, TensorReordering_t, ReshapeMode_t)
// are intentionally NOT aliased here: hipDNN does not yet publish them, and the
// nodes that use them are out of scope until later phases (RFC 0012 §7.3, §7.4).
// They are added on the hipDNN side and aliased when their node lands.

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

// The `graph` sub-namespace is populated by the `cudnn_frontend/*` headers the
// umbrella pulls in: `graph_properties.h` aliases the attribute types
// (`Tensor_attributes`, …). The `Graph` composition wrapper and the remaining
// per-node `*_attributes` aliases land in later phases (RFC 0012 §7.2).
// Declaring the namespace here keeps this header self-contained even when
// included on its own.
namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
