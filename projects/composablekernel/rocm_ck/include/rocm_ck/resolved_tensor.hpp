// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — ResolvedTensor, ResolvedQuantization. No runtime, no CK deps.
//
// A ResolvedTensor is the output of Signature::resolve() — a tensor whose
// metadata is fully determined. In the user-facing Signature, tensors can
// have Layout::Auto (inherit from operator slot) and omit fields that have
// sensible defaults. After resolution, every field is concrete.
//
// The base fields (name, dtype, rank, layout) describe a plain dense tensor.
// This covers most operands: GEMM inputs/outputs, bias vectors, activations.
//
// Some tensors carry additional metadata beyond the dense description.
// Block-quantized tensors (e.g., INT4 weights) need a scale tensor and
// group size. Rather than encoding every possible extension as top-level
// fields, we use optional sub-structs — present only when relevant. This
// keeps the common case clean (most tensors are just dense) while allowing
// future extensions (sparsity metadata, tiling hints) without bloating
// every ResolvedTensor.
//
// ResolvedTensor is the bridge between the Signature (user intent) and
// makeSpec() (kernel configuration). It's a plain aggregate — no methods,
// no validation. Resolution validates; this type just carries the result.
//
// std::string_view makes ResolvedTensor non-structural (can't be an NTTP),
// which is fine — it's a transient intermediate, never a template parameter.
// PhysicalTensor uses FixedString for the NTTP use case.

#pragma once

#include <rocm_ck/datatype.hpp>
#include <rocm_ck/layout.hpp>

#include <optional>
#include <string_view>

namespace rocm_ck {

// Present when a tensor carries block-quantized data (e.g., INT4 weights).
// The scale tensor is a separate entry in the Signature; this struct ties
// the quantized tensor to its scale.
struct ResolvedQuantization
{
    std::string_view scale_name;
    DataType scale_dtype;
    int group_size; // elements per quantization group
};

struct ResolvedTensor
{
    std::string_view name;
    DataType dtype;
    int rank                                     = 2;
    Layout layout                                = Layout::Row;
    std::optional<ResolvedQuantization> quantize = std::nullopt;
};

} // namespace rocm_ck
