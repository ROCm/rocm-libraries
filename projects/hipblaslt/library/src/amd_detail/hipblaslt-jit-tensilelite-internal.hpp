// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-gemm-internal.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    struct Bundle final : jit::detail::KernelBundle
    {
        int                                    device = -1;
        hipDeviceProp_t                        properties{};
        std::shared_ptr<TensileLite::Hardware> hardware;
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                                           library;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::string                                        manifest, kernel;
        std::string_view                                   operationKind() const noexcept override
        {
            return jit::detail::GemmRequest::operation;
        }
        std::string name() const override
        {
            return library->solutions.at(0)->solutionName;
        }
        std::string kernelNames() const override
        {
            return kernel;
        }
        hipblasStatus_t support(const jit::detail::OperationRequest&,
                                size_t,
                                size_t&,
                                Diagnostics&) const override;
        hipblasStatus_t prepare(const jit::detail::OperationRequest&,
                                const jit::detail::ExecutionContext&,
                                std::shared_ptr<const jit::detail::PreparedLaunch>&,
                                Diagnostics&) const override;
    };
}
