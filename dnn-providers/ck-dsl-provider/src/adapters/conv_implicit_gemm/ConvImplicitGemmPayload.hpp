// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "../../python/CompilePayload.hpp"
#include "ConvImplicitGemmSpec.hpp"

namespace ck_dsl_provider {

/// Translate a ``ConvImplicitGemmSpec`` into the interpreter-neutral payload
/// that ``ck_dsl_provider.compile_service.compile`` will consume.
///
/// The dict shape mirrors the ``ImplicitGemmConvSpec`` Python dataclass
/// field-for-field (the Python side splats it via ``ImplicitGemmConvSpec(
/// **payload)``); the nested ``problem`` entry mirrors ``ConvProblem``.
/// Optional fields (``lds_k_pad``, ``waves_per_eu``) are emitted as None when
/// unset so the dataclass takes its own default; ``lds_layout`` is omitted so
/// the dataclass re-derives it.
///
/// Unlike the previous pybind version, this builds no interpreter objects and
/// needs no lock held; ``CompileServiceBridge`` marshals the result to mp_obj_t.
PayloadDict convImplicitGemmSpecToPayload(const ConvImplicitGemmSpec& spec);

}  // namespace ck_dsl_provider
