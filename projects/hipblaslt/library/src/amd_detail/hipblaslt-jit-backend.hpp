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

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
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
