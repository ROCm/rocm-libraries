// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <string>

namespace hipblaslt_jit
{
    class Jit;

    struct Diagnostics
    {
        std::string message;
    };

    struct GemmRequest;
}
