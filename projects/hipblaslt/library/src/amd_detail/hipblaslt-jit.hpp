// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_jit
{
    class Jit;
}

// Not installed.
namespace hipblaslt_ext::experimental::jit
{
    namespace detail
    {
        struct BackendAccess;
        struct OperationRequest;
        struct RequestAccess;
        struct CompiledSolution;
        struct SolutionAccess;
    }

    // Backend factories configure an opaque, copyable solution generator.
    class Backend
    {
    public:
        Backend() = default;

    private:
        std::shared_ptr<const hipblaslt_jit::Jit> jit;
        friend struct detail::BackendAccess;
    };

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
    class Request
    {
    public:
        Request() = default;

    private:
        std::shared_ptr<const detail::OperationRequest> implementation;
        friend struct detail::RequestAccess;
    };

    // Owns an executable kernel bundle, including its backend and helper modules.
    // An operation adapter converts it to an execution API's algorithm type.
    class Solution
    {
    public:
        Solution() = default;

    private:
        std::shared_ptr<const detail::CompiledSolution> implementation;
        friend struct detail::SolutionAccess;
    };

    struct Diagnostics
    {
        std::string backend;
        std::string message;
    };
}
