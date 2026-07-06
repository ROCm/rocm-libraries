// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend_utils.h),
// used under the MIT license.

/**
 * @file cudnn_frontend_utils.h
 * @brief FE-namespace enum aliases for the hipDNN cuDNN-compatibility shim.
 *
 * The v9 graph API signatures use cuDNN's FE-namespace enums (`DataType_t`,
 * `PointwiseMode_t`, …), not the C-API ones from `<cudnn.h>`. hipDNN already
 * publishes these as cuDNN-named `_t` typedefs (see `Types.hpp`), so this header
 * just aliases them into `<shim_ns>` with `using` declarations — zero overhead,
 * no numeric cast between enum families.
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/Types.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

// FE-namespace enums hipDNN publishes 1:1, aliased so e.g.
// `cudnn_frontend::DataType_t` *is* `hipdnn_frontend::DataType_t`.
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
// are not aliased yet: hipDNN does not publish them and their nodes are out of
// scope. They are aliased when their node lands.

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

// The `graph` sub-namespace is populated by the `cudnn_frontend/*` headers the
// umbrella pulls in; declared empty here so this header is self-contained when
// included on its own.
namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
