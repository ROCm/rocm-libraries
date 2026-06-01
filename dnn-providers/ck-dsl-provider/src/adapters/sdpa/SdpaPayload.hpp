// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/pybind11.h>

#include "SdpaSpec.hpp"

namespace ck_dsl_provider {

/// Translate an ``SdpaSpec`` into the on-wire payload dict that the
/// Python ``ck_dsl_provider.compile_service`` consumes for the SDPA
/// forward op.
///
/// The dict carries only the codegen-relevant fields: batch, the
/// head-shape triple (head_size / num_query_heads / num_kv_heads), the
/// dtype, the mask mode, and the two sequence lengths. The eight
/// stride_* scalars and scale_log2 are deliberately omitted -- they are
/// launch-time kernel arguments, not codegen inputs, so they neither
/// belong in the payload nor in the cache key.
///
/// **GIL discipline:** the caller MUST hold the GIL before invoking this
/// function. It allocates Python objects (py::dict, py::int_, etc.);
/// doing so without the GIL is undefined behaviour.
pybind11::dict sdpaSpecToPayload(const SdpaSpec& spec);

}  // namespace ck_dsl_provider
