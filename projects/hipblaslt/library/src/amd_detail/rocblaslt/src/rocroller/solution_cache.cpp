/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

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
