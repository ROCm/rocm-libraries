// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "solution_cache.hpp"

void SolutionCache::addKernel(const KernelType&              kernelType,
                              const SolutionIndexParameters& params,
                              std::shared_ptr<GemmKernel>    kernel)
{
    auto existingKernelType = m_generatedKernels.find(kernelType);
    if(existingKernelType == m_generatedKernels.end())
    {
        m_generatedKernels[kernelType] = {};
    }

    auto  index = parametersToIndex(params);
    auto& vec   = m_generatedKernels[kernelType][index];

    if(kernel->shapeCondition.has_value())
    {
        // Conditional kernels are appended so that multiple shape-specific
        // heuristics can coexist in the same bucket.  During lookup they are
        // evaluated in insertion order; the first match wins.
        vec.push_back(kernel);
    }
    else
    {
        // Unconditional (fallback) kernels: at most one per bucket.
        // If one already exists, replace it rather than accumulating duplicates.
        for(auto& existing : vec)
        {
            if(!existing->shapeCondition.has_value())
            {
                existing = kernel;
                return;
            }
        }
        vec.push_back(kernel);
    }
}

std::optional<std::shared_ptr<GemmKernel>>
    SolutionCache::getKernel(const KernelType&              kernelType,
                             const SolutionIndexParameters& params,
                             std::optional<ProblemDims>      dims)
{
    auto existingKernelType = m_generatedKernels.find(kernelType);
    if(existingKernelType == m_generatedKernels.end())
    {
        return std::nullopt;
    }

    auto index = parametersToIndex(params);
    auto it    = existingKernelType->second.find(index);

    if(it == existingKernelType->second.end())
        return std::nullopt;

    const auto& kernels = it->second;

    // Tier 1: if problem dimensions are known, check conditional kernels
    // first (in insertion order).  The first whose heuristic matches wins.
    if(dims.has_value())
    {
        for(const auto& k : kernels)
        {
            if(k->shapeCondition.has_value()
               && k->shapeCondition->matches(dims->m, dims->n, dims->k))
            {
                return k;
            }
        }
    }

    // Tier 2: fall back to the unconditional kernel.
    for(const auto& k : kernels)
    {
        if(!k->shapeCondition.has_value())
        {
            return k;
        }
    }

    return std::nullopt;
}
