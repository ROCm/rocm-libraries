// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_jit
{
    // A one-solution TensileLite library entry and, once loaded, its code objects.
    struct TensileBundle final : KernelBundle
    {
        using Diagnostics      = hipblaslt_ext::experimental::jit::Diagnostics;
        using ExecutionContext = hipblaslt_ext::experimental::jit::detail::ExecutionContext;
        using PreparedLaunch   = hipblaslt_ext::experimental::jit::detail::PreparedLaunch;

        std::shared_ptr<TensileLite::Hardware> hardware;
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                                           library;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::string                                        kernel;

        std::string_view operationKind() const noexcept override
        {
            return hipblaslt_ext::experimental::jit::detail::GemmRequest::operation;
        }
        std::string name() const override
        {
            return library->solutions.at(0)->solutionName;
        }
        std::string kernelNames() const override
        {
            return kernel;
        }
        hipblasStatus_t
            support(const OperationRequest&, size_t, size_t&, Diagnostics&) const override;
        hipblasStatus_t prepare(const OperationRequest&,
                                const ExecutionContext&,
                                std::shared_ptr<const PreparedLaunch>&,
                                Diagnostics&) const override;
    };

    // Passes code objects the backend already built through unchanged.
    std::shared_ptr<const CodeObjectBuilder> makePrebuiltBuilder();

    // Loads CustomKernel entries from any backend into TensileBundles.
    std::shared_ptr<const SolutionLoader> makeTensileLoader();
}
