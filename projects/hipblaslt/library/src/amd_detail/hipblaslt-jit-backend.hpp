// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"
#include <memory>
#include <string>

// GEMM-only provider interface. This is not a stable plugin ABI.
namespace hipblaslt_ext::experimental::jit::detail
{
    struct GemmRequest;

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
