// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — TensorDesc struct. No runtime, no CK deps.
//
// Common vocabulary type for resolved tensor metadata.
// Captures dtype, name, rank, and layout — used as the internal intermediate
// between user-facing Signature structs and make_spec().

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>

#include <string_view>

namespace rocm_ck {

/// Resolved metadata for a single tensor operand.
/// Plain aggregate — no constructors, no methods.
/// std::string_view makes it non-structural (can't be NTTP), but that's fine:
/// TensorDesc is an internal intermediate, never used as a template parameter.
struct TensorDesc
{
    std::string_view name;       // "A", "bias", "query"
    DataType dtype;              // element type
    int rank      = 2;           // dimensions
    Layout layout = Layout::Row; // default: row-major rank-2
};

} // namespace rocm_ck
