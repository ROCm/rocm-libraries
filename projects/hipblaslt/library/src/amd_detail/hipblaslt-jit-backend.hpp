// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"
#include <memory>
#include <string_view>

// Private, compiled-in provider interface. This is not a stable plugin ABI.
namespace hipblaslt_ext::experimental::jit::detail
{
    struct OperationRequest
    {
        virtual ~OperationRequest()                    = default;
        virtual std::string_view kind() const noexcept = 0;
    };

    struct KernelBundle
    {
        virtual ~KernelBundle()                                 = default;
        virtual std::string_view operationKind() const noexcept = 0;
        virtual int              solutionIndex() const noexcept = 0; // local, in its entry
        virtual std::string      name() const                   = 0;
        virtual std::string      kernelNames() const            = 0;
        virtual hipblasStatus_t  support(const OperationRequest& request,
                                         size_t                  workspaceLimit,
                                         size_t&                 workspaceBytes,
                                         Diagnostics&            diagnostics) const
            = 0;
    };

    struct BackendAccess
    {
        static Backend make(std::shared_ptr<const hipblaslt_jit::Jit> jit)
        {
            Backend backend;
            backend.jit = std::move(jit);
            return backend;
        }
        static const auto& get(const Backend& backend)
        {
            return backend.jit;
        }
    };

    struct RequestAccess
    {
        static Request make(std::shared_ptr<const OperationRequest> operation)
        {
            Request request;
            request.implementation = std::move(operation);
            return request;
        }
        static const auto& get(const Request& request)
        {
            return request.implementation;
        }
    };

    struct SolutionAccess
    {
        static Solution make(std::shared_ptr<const CompiledSolution> compiled)
        {
            Solution solution;
            solution.implementation = std::move(compiled);
            return solution;
        }
        static const auto& get(const Solution& solution)
        {
            return solution.implementation;
        }
    };

}
