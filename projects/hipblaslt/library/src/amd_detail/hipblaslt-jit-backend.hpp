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

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
    namespace detail
    {
        struct GemmRequest;
    }
}
