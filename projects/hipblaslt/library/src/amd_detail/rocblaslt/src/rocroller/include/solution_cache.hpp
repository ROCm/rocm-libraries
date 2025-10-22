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

#pragma once

#include "gemm.hpp"
#include "kernel_type.hpp"
#include "solution_selection.hpp"

class SolutionCache
{
    public:
    void addKernel(const KernelType& kernelType, const SolutionIndexParameters& params, std::shared_ptr<GemmKernel> kernel);
    std::optional<std::shared_ptr<GemmKernel>> getKernel(const KernelType& kernelType, const SolutionIndexParameters& params);

    private:
    // Map of kernels that have already been generated.
    // The first level of the map is indexed with a KernelType.
    // The second level of the map is indexed with a hash value of a
    // SolutionIndexParameters type.
    // The value is a GemmKernel.
    std::unordered_map<KernelType, std::unordered_map<int, std::shared_ptr<GemmKernel>>> m_generatedKernels;
};
