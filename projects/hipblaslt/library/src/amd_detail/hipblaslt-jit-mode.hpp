// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>

namespace hipblaslt_jit
{
    // When heuristic queries consult the JIT library. Read from HIPBLASLT_JIT.
    enum class Mode
    {
        Off, // unset, empty or 0: heuristic queries do not consult JIT
        Fallback, // 1: JIT fills the shortfall after the Equality rows
        Forced, // 2: the JIT library only
    };

    // Parses a HIPBLASLT_JIT value. False for anything other than unset, empty,
    // "0", "1" or "2", which leaves mode Off.
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
    // An invalid value warns once and leaves the mode Off.
    Mode mode() noexcept;
}
