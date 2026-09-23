// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt-jit.hpp>
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

    struct Target
    {
        int             device = -1;
        hipDeviceProp_t properties{};
    };

    struct ExecutionContext
    {
        hipblasLtHandle_t handle;
        void*             workspace;
        size_t            workspaceBytes;
        hipStream_t       stream;
    };

    struct PreparedLaunch
    {
        virtual ~PreparedLaunch() = default;
        virtual hipblasStatus_t run(hipStream_t stream, hipEvent_t start, hipEvent_t stop) const
            = 0;
    };

    struct KernelBundle
    {
        virtual ~KernelBundle()                                 = default;
        virtual std::string_view operationKind() const noexcept = 0;
        virtual std::string      name() const                   = 0;
        virtual std::string      kernelNames() const            = 0;
        virtual hipblasStatus_t  support(const OperationRequest& request,
                                         size_t                  workspaceLimit,
                                         size_t&                 workspaceBytes,
                                         Diagnostics&            diagnostics) const
            = 0;
        // Resolve every required symbol before submitting any GPU work.
        virtual hipblasStatus_t prepare(const OperationRequest&                request,
                                        const ExecutionContext&                execution,
                                        std::shared_ptr<const PreparedLaunch>& launch,
                                        Diagnostics&                           diagnostics) const
            = 0;
    };

    struct BackendImplementation
    {
        virtual ~BackendImplementation()               = default;
        virtual std::string_view name() const noexcept = 0;
        virtual hipblasStatus_t  compile(const OperationRequest&              request,
                                         const Target&                        target,
                                         size_t                               workspaceLimit,
                                         std::shared_ptr<const KernelBundle>& bundle,
                                         Diagnostics&                         diagnostics) const
            = 0;
    };

    struct BackendAccess
    {
        static Backend make(std::shared_ptr<const BackendImplementation> implementation)
        {
            Backend backend;
            backend.implementation = std::move(implementation);
            return backend;
        }
        static const auto& get(const Backend& backend)
        {
            return backend.implementation;
        }
    };
    struct CompiledSolution
    {
        Target                                       target;
        std::shared_ptr<const OperationRequest>      request;
        std::shared_ptr<const BackendImplementation> backend;
        std::shared_ptr<const KernelBundle>          bundle;
        uint64_t                                     process        = 0;
        size_t                                       workspaceLimit = 0;
        size_t                                       workspaceBytes = 0;
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
