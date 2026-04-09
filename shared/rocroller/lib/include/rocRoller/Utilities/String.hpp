// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    Generator<std::string> EscapeComment(std::string comment, int indent = 0);
}

#include <rocRoller/Utilities/String_impl.hpp>
