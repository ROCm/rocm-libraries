// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Pointwise-activation op adapter -- the fourth proof op, extending the catalog
// engine to unary elementwise activations. It decodes a single-node
// PointwiseAttributes graph into a ProblemShape keyed by dtype/activation/numel
// and resolves the (A, C, N) launch ABI the rocKE elementwise .co expects.
//
// The shipped gfx1151 kernels are the CK Tile 21_elementwise parity kernels:
// one contiguous pass over `numel` elements applying a fused per-element unary op
// (compute in f32). The activation is baked into each .co at build time, so
// family.json carries one kernel per (activation, dtype, tuning) and constrains
// the decoded "activation" token exactly. numel is a runtime i32 arg, so a single
// .co serves any numel that is a multiple of its `vec` (row-start alignment).
//
// v1 accepts exactly two activation modes, both of which the rocKE elementwise
// builder implements as unary ops:
//   * SWISH_FWD (SiLU), when swish_beta is absent or == 1.0  -> token "silu"
//   * GELU_APPROX_TANH_FWD (tanh-approximation GELU)         -> token "gelu_tanh"
// Everything else fails closed (declines): exact erf GELU_FWD (no builder op yet),
// any binary/ternary pointwise (in_1/in_2 set), a non-unit swish beta, a
// non-contiguous tensor, a mixed or unsupported dtype, or a numel that overflows
// int32.

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class ActivationAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "pointwise";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
