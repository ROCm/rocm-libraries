// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
