// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/pybind11.h>

#include "ConvImplicitGemmSpec.hpp"

namespace ck_dsl_provider {

/// Translate a ``ConvImplicitGemmSpec`` into the on-wire payload dict
/// that ``ck_dsl_provider.compile_service.compile`` (I-7) will consume.
///
/// The dict shape mirrors the ``ImplicitGemmConvSpec`` Python
/// dataclass field-for-field so the Python side can splat it directly
/// via ``ImplicitGemmConvSpec(**payload)``. The nested ``problem``
/// entry is itself a dict mirroring ``ConvProblem``.
///
/// Optional fields (``lds_k_pad``, ``waves_per_eu``) are emitted as
/// ``py::none()`` when unset so the dataclass constructor takes its
/// own default. We deliberately omit ``lds_layout`` (always None on
/// the dataclass for M1) so the dataclass constructor re-derives it
/// from ``async_dma`` / ``lds_k_pad`` / ``tile_k``.
///
/// **GIL discipline:** the caller MUST hold the GIL before invoking
/// this function. It allocates Python objects (py::dict, py::int_,
/// etc.); doing so without the GIL is undefined behaviour.
pybind11::dict convImplicitGemmSpecToPayload(const ConvImplicitGemmSpec& spec);

}  // namespace ck_dsl_provider
