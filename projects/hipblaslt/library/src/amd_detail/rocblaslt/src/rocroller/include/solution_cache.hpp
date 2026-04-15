// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm.hpp"
#include "kernel_type.hpp"
#include "solution_selection.hpp"

struct ProblemDims
{
    size_t m, n, k;
};

class SolutionCache
{
public:
    /// Register a kernel in the cache.
    ///
    /// Kernels with a ShapeCondition are appended alongside existing entries for
    /// the same (KernelType, SolutionIndexParameters) bucket.  Unconditional
    /// kernels (no ShapeCondition) replace any prior unconditional entry.
    ///
    /// During lookup, conditional kernels are checked first in the order they
    /// were added; the first whose condition matches the problem dimensions
    /// wins.  If none match, the unconditional kernel is returned as a fallback.
    void addKernel(const KernelType&              kernelType,
                   const SolutionIndexParameters& params,
                   std::shared_ptr<GemmKernel>    kernel);

    /// Look up a kernel for a given (KernelType, SolutionIndexParameters) pair.
    ///
    /// When @p dims is provided the cache first checks conditional kernels
    /// (in insertion order) and returns the first match.  If no conditional
    /// kernel matches — or @p dims is not provided — the unconditional
    /// fallback kernel is returned.
    std::optional<std::shared_ptr<GemmKernel>> getKernel(const KernelType&              kernelType,
                                                         const SolutionIndexParameters& params,
                                                         std::optional<ProblemDims>     dims
                                                         = std::nullopt);

private:
    // Map of kernels that have already been generated.
    // The first level is indexed by KernelType.
    // The second level is indexed by a hash of SolutionIndexParameters.
    // The value is a vector of GemmKernels: conditional (shape-heuristic)
    // entries followed by at most one unconditional fallback entry.
    std::unordered_map<KernelType,
                       std::unordered_map<int, std::vector<std::shared_ptr<GemmKernel>>>>
        m_generatedKernels;
};
