// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "solution_cache.hpp"

void SolutionCache::addKernel(const KernelType&              kernelType,
                              const SolutionIndexParameters& params,
                              std::shared_ptr<GemmKernel>    kernel)
{
    auto  index = parametersToIndex(params);
    auto& vec   = m_generatedKernels[kernelType][index];

    if(kernel->staticShape.has_value() || kernel->shapeCondition.has_value())
    {
        vec.push_back(kernel);
    }
    else
    {
        for(auto& existing : vec)
        {
            if(!existing->staticShape.has_value() && !existing->shapeCondition.has_value())
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
                             std::optional<ProblemDims>     dims)
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

    if(dims.has_value())
    {
        for(const auto& k : kernels)
        {
            if(k->staticShape.has_value() && k->staticShape->matches(dims->m, dims->n, dims->k))
            {
                std::cout << "Loading kernel from cache (static): " << k->module->getKernelName()
                          << std::endl;
                return k;
            }
        }

        for(const auto& k : kernels)
        {
            if(k->shapeCondition.has_value()
               && k->shapeCondition->matches(dims->m, dims->n, dims->k))
            {
                std::cout << "Loading kernel from cache: " << k->module->getKernelName()
                          << std::endl;
                return k;
            }
        }
    }

    for(const auto& k : kernels)
    {
        if(!k->staticShape.has_value() && !k->shapeCondition.has_value())
        {
            if(k->module.has_value())
            {
                std::cout << "Loading kernel from cache: " << k->module->getKernelName()
                          << std::endl;
            }
            return k;
        }
    }

    return std::nullopt;
}
