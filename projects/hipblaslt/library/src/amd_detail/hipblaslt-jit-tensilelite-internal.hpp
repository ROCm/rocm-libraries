// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "hipblaslt-jit-gemm-internal.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    struct ExecutionContext
    {
        hipblasLtHandle_t handle;
        void*             workspace;
        size_t            workspaceBytes;
        hipStream_t       stream;
    };
    struct PreparedLaunch
    {
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::vector<TensileLite::KernelInvocation>         kernels;
        hipStream_t                                        preparedStream;
        bool                                               streamBound = false;
        hipblasStatus_t run(hipStream_t stream, hipEvent_t start, hipEvent_t stop) const;
    };
    struct Bundle
    {
        uint64_t                               process = 0;
        int                                    device  = -1;
        hipDeviceProp_t                        properties{};
        std::shared_ptr<TensileLite::Hardware> hardware;
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                                           library;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::string                                        manifest, kernel;
        std::string                                        name() const
        {
            return library->solutions.at(0)->solutionName;
        }
        std::string kernelNames() const
        {
            return kernel;
        }
        hipblasStatus_t support(const GemmProblem&, size_t, size_t&, Diagnostics&) const;
        hipblasStatus_t prepare(const GemmProblem&,
                                const ExecutionContext&,
                                std::shared_ptr<const PreparedLaunch>&,
                                Diagnostics&) const;
    };
    hipblasStatus_t generateBundle(
        const Options&, const hipDeviceProp_t&, int, std::shared_ptr<Bundle>&, Diagnostics&);
}
