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

}
