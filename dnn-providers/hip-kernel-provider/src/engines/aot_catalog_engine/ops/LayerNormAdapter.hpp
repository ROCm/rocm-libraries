// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// LayerNorm op adapter -- the third proof op, extending the catalog engine past
// GEMM and RMSNorm. It decodes a single-node LayernormAttributes graph into a
// ProblemShape keyed by dtype/M/N and resolves the (X, Gamma, Beta, Y, M, N, eps)
// launch ABI the rocKE layernorm2d .co expects.
//
// The shipped gfx1151 kernel is the CK Tile 02_layernorm2d parity kernel:
//   mean[m]    = sum_n(X[m,n]) / N
//   inv_std[m] = 1 / sqrt(sum_n((X[m,n]-mean[m])^2)/N + eps)
//   Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * Gamma[n] + Beta[n]
// i.e. per-row LayerNorm over the last dim of a 2D [M,N] activation, with a
// per-column weight Gamma[N] and bias Beta[N] -- the standard transformer
// LayerNorm. N is baked into the kernel at build time, so family.json constrains
// N to an exact value (like rmsnorm2d). The extra Beta pointer vs RMSNorm is the
// one structural difference in the ABI.
//
// The adapter fails closed (declines) on anything outside that shape: non-2D
// input, non-row-major strides, a Gamma/Beta that is not [1,N], a mixed or
// unsupported dtype, a mean/inv_variance stat output (our forward kernel saves
// neither), a non-inference forward phase, or a runtime-user-supplied epsilon
// (we bake epsilon at plan-build time).

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class LayerNormAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "layernorm";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
