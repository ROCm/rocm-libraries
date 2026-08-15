// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Forward-convolution (implicit-GEMM) op adapter. Decodes a single-node
// ConvolutionFwdAttributes graph into a ProblemShape and resolves the runtime
// launch ABI the catalog .co expects.
//
// Runtime-generic model: the shipped conv .co bakes only the tile/perf config
// and reads ALL problem geometry (N, C, K, Hi, Wi, R, S, stride, pad, dilation)
// from runtime scalar args, masking partial tiles at the M/N/K boundaries. One
// .co per tile config therefore serves ANY 2-D forward-conv shape. Genericity
// comes from boundary masking, not alignment (see the family README).
//
// Fact-publishing (SDPA-style): decode() enforces only universal safety
// invariants -- rank-4 operands, NHWC/KRSC/NHWK packed layout, symmetric
// padding, a single supported dtype, self-consistent Ho/Wo -- and publishes
// every structural fact (groups, conv_mode, the raw geometry, the derived GEMM
// extents, the *_bytes buffer sizes) as a ProblemShape key. Each family.json
// kernel opts into a subset via `constraints` + `args_signature`, so ONE adapter
// serves gfx1151 (WMMA, groups==1) and the KA team's CDNA kernels (MFMA,
// grouped, split-k) with zero C++ divergence.

#pragma once

#include "ops/IOpAdapter.hpp"

namespace aot_catalog_engine::ops
{

class ConvFpropAdapter : public IOpAdapter
{
public:
    const char* opKind() const override
    {
        return "conv_fprop";
    }

    std::optional<catalog::ProblemShape> decode(const IGraph& graph) const override;

    catalog::LaunchBindings buildBindings(const IGraph& graph,
                                          const catalog::ProblemShape& problem,
                                          const catalog::KernelEntry& kernel) const override;

    launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                    const catalog::KernelEntry& kernel) const override;
};

} // namespace aot_catalog_engine::ops
