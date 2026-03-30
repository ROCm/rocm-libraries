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
    void addKernel(const KernelType&              kernelType,
                   const SolutionIndexParameters& params,
                   std::shared_ptr<GemmKernel>    kernel);

    std::optional<std::shared_ptr<GemmKernel>> getKernel(const KernelType&              kernelType,
                                                         const SolutionIndexParameters& params,
                                                         std::optional<ProblemDims>     dims
                                                         = std::nullopt);

private:
    std::unordered_map<KernelType,
                       std::unordered_map<int, std::vector<std::shared_ptr<GemmKernel>>>>
        m_generatedKernels;
};
