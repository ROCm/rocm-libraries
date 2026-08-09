// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// RMS-norm op adapter -- the second proof op, added to demonstrate the catalog
// engine generalizes past GEMM. It decodes a single-node RMSNormAttributes graph
// into a ProblemShape keyed by dtype/M/N and resolves the
// (X, Gamma, Y, M, N, eps) launch ABI the rocKE rmsnorm2d .co expects.
//
// The shipped gfx1151 kernel is the CK Tile 10_rmsnorm2d parity kernel:
//   rms[m] = sqrt(sum_n(X[m,n]^2) / N + eps);  Y[m,n] = X[m,n] / rms[m] * Gamma[n]
// i.e. per-row RMS norm over the last dim of a 2D [M,N] activation, with a
// per-column weight Gamma[N] -- the Llama/Mistral/Gemma RMSNorm. N is baked into
// the kernel at build time, so the family.json constrains N to an exact value
// (exercising exact-match applicability, unlike GEMM's multiple_of predicates).
//
// The adapter fails closed (declines) on anything outside that shape: non-2D
// input, non-row-major strides, a reduction that is not exactly the last dim, a
// bias or inv_rms output (our kernel has neither), a non-f16 dtype, or a
// runtime-user-supplied epsilon (we bake epsilon at plan-build time).

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class RmsNormAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "rmsnorm";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
