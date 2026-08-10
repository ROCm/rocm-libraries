// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Pluggable op adapter: the piece that knows how to read one hipDNN op out of a
// graph. Generalizes PR #9207's SDPA-monomorphic upper half so ops are additive
// (matmul now; sdpa/conv later) and the matcher itself stays op-agnostic.
//
// decode() doubles as the Tier-D applicability escape hatch: it is arbitrary
// C++ that gates on node type, decodes the shape, and returns nullopt for
// anything unsupported. Our team writes adapters; kernel authors never touch
// them for data-only catalog changes.

#pragma once

#include <optional>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "catalog/CatalogTypes.hpp"
#include "launch/LaunchAbi.hpp"

namespace aot_catalog_engine::ops
{

using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;

class IOpAdapter
{
public:
    virtual ~IOpAdapter() = default;

    // op_kind this adapter handles; matched against a family's op_kind.
    virtual const char* opKind() const = 0;

    // Decode the graph into a selection ProblemShape. Returns nullopt when the
    // graph is not this adapter's op or is otherwise unsupported.
    virtual std::optional<catalog::ProblemShape> decode(const IGraph& graph) const = 0;

    // Resolve the chosen kernel's argument bindings (arg name -> tensor uid /
    // scalar value) from the graph and decoded problem.
    virtual catalog::LaunchBindings buildBindings(const IGraph& graph,
                                                  const catalog::ProblemShape& problem,
                                                  const catalog::KernelEntry& kernel) const
        = 0;

    // Build the grid symbol table (e.g. M,N,K) the kernel's grid formula uses.
    virtual launch::SymbolTable gridSymbols(const catalog::ProblemShape& problem,
                                            const catalog::KernelEntry& kernel) const
        = 0;
};

} // namespace aot_catalog_engine::ops
