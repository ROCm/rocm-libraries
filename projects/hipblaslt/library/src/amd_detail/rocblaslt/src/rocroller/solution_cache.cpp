// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "solution_cache.hpp"

void SolutionCache::addKernel(const KernelType& kernelType, const SolutionIndexParameters& params, std::shared_ptr<GemmKernel> kernel)
{
    auto existingKernelType = m_generatedKernels.find(kernelType);
    if(existingKernelType == m_generatedKernels.end())
    {
        m_generatedKernels[kernelType] = {};
    }

    auto index = parametersToIndex(params);

    m_generatedKernels[kernelType][index] = kernel;
}

std::optional<std::shared_ptr<GemmKernel>> SolutionCache::getKernel(const KernelType& kernelType, const SolutionIndexParameters& params)
{
    auto existingKernelType = m_generatedKernels.find(kernelType);
    if(existingKernelType == m_generatedKernels.end())
    {
        return std::nullopt;
    }

    auto index = parametersToIndex(params);

    auto kernel = existingKernelType->second.find(index);

    if (kernel == existingKernelType->second.end())
        return std::nullopt;
    else
        return kernel->second;
}
