// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/pybind11.h>

#include "SdpaBwdSpec.hpp"

namespace ck_dsl_provider {

/// Translate an ``SdpaBwdSpec`` into the on-wire payload dict that the
/// Python ``ck_dsl_provider.compile_service`` consumes for the SDPA
/// backward op (``sdpa_fmha_bwd``).
///
/// The dict carries only the codegen-relevant fields: batch, the
/// head-shape triple (head_size / num_query_heads / num_kv_heads), the
/// dtype, the mask mode, and the two sequence lengths. The stride_*
/// scalars and the scale_* values are deliberately omitted -- they are
/// launch-time kernel arguments, not codegen inputs, so they neither
/// belong in the payload nor in the cache key.
///
/// **GIL discipline:** the caller MUST hold the GIL before invoking this
/// function. It allocates Python objects (py::dict, py::int_, etc.);
/// doing so without the GIL is undefined behaviour.
pybind11::dict sdpaBwdSpecToPayload(const SdpaBwdSpec& spec);

/// Translate an ``SdpaBwdSpec`` into the on-wire payload dict for the
/// LSE-prep op (``sdpa_lse_prep``).
///
/// The prep kernel only depends on batch, the query-head count, and the
/// query sequence length, so the dict carries just those three fields.
///
/// **GIL discipline:** the caller MUST hold the GIL before invoking this
/// function (it allocates Python objects).
pybind11::dict sdpaLsePrepSpecToPayload(const SdpaBwdSpec& spec);

}  // namespace ck_dsl_provider
