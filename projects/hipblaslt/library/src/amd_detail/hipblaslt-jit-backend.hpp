// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <string>

namespace hipblaslt_jit
{
    class Jit;
}

// GEMM-only provider interface. This is not a stable plugin ABI.
namespace hipblaslt_ext::experimental::jit
{
    struct Diagnostics
    {
        std::string backend;
        std::string message;
    };

    namespace detail
    {
        struct GemmRequest;
        struct CompiledSolution;
    }

    // Backend factories configure an opaque, copyable solution generator.
    class Backend
    {
    public:
        Backend() = default;
        static Backend create(std::shared_ptr<const hipblaslt_jit::Jit> jit)
        {
            Backend backend;
            backend.m_jit = std::move(jit);
            return backend;
        }
        const std::shared_ptr<const hipblaslt_jit::Jit>& get() const
        {
            return m_jit;
        }

    private:
        std::shared_ptr<const hipblaslt_jit::Jit> m_jit;
    };

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
    class Request
    {
    public:
        Request() = default;
        static Request create(std::shared_ptr<const detail::GemmRequest> gemm)
        {
            Request request;
            request.m_gemm = std::move(gemm);
            return request;
        }
        const std::shared_ptr<const detail::GemmRequest>& get() const
        {
            return m_gemm;
        }

    private:
        std::shared_ptr<const detail::GemmRequest> m_gemm;
    };

    namespace detail
    {
        struct KernelBundle
        {
            virtual ~KernelBundle()                            = default;
            virtual int         solutionIndex() const noexcept = 0; // local, in its entry
            virtual std::string name() const                   = 0;
            virtual std::string kernelNames() const            = 0;
            virtual hipblasStatus_t support(const GemmRequest& request,
                                            size_t             workspaceLimit,
                                            size_t&            workspaceBytes,
                                            Diagnostics&       diagnostics) const
                = 0;
        };
    }
}
