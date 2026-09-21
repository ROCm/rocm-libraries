// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>

namespace rocRoller
{
    enum class ArgumentType : int
    {
        Value = 0,
        Size,
        Stride,

        Count
    };
}
