// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GEMM (matmul) op adapter -- the first proof op. Decodes a single-node
// MatmulAttributes graph into a ProblemShape keyed by dtype/M/N/K, and resolves
// the (A,B,C,M,N,K) launch ABI the catalog .co expects.
//
// RCR layout: the gfx1151 wmma_gemm we ship computes C[M,N] = A[M,K] * B^T with
// A row-major [M,K], B row-major [N,K] (the weight), C row-major [M,N] -- i.e.
// exactly nn.Linear (y = x @ W^T, W = [out_features, in_features]), our real
// ComfyUI target. The adapter therefore reads B as [N,K] and fails closed
// (declines) on any graph that does not fit that shape. Batched matmul,
// other layouts, and non-f16 dtypes are left for later.

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class GemmAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "matmul";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
