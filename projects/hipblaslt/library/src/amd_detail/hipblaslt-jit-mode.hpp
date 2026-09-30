// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>

namespace hipblaslt_jit
{
    // When heuristic queries and hipblasLtMatmul without an algorithm use JIT solutions.
    enum class Mode
    {
        Off, // HIPBLASLT_JIT unset, empty or 0
        Fallback, // 1: pre-tuned solutions first, JIT solutions for any shortfall
        Forced, // 2: JIT solutions only
    };

    // Parses a HIPBLASLT_JIT value into mode. False for anything other than
    // unset, empty, "0", "1" or "2", which leaves mode Off.
    inline bool parseMode(const char* value, Mode& mode) noexcept
    {
        mode = Mode::Off;
        if(!value || !*value || std::strcmp(value, "0") == 0)
            return true;
        if(std::strcmp(value, "1") == 0)
            mode = Mode::Fallback;
        else if(std::strcmp(value, "2") == 0)
            mode = Mode::Forced;
        return mode != Mode::Off;
    }

    // The process mode, read from HIPBLASLT_JIT on first use and never again.
    // Privileged processes ignore the variable, and builds without
    // HIPBLASLT_ENABLE_JIT always return Off. The first call writes one warning
    // to stderr when the value is invalid or ignored.
    Mode mode() noexcept;
}
