// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-mode.hpp"
#include "rocblaslt_secure_env.hpp"

#include <iostream>

namespace hipblaslt_jit
{
    namespace
    {
        Mode readMode() noexcept
        {
            const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT");
            Mode        parsed;
            const bool  valid = parseMode(value, parsed);
#ifdef HIPBLASLT_ENABLE_JIT
            if(!valid)
                std::cerr << "hipblaslt warning: HIPBLASLT_JIT=" << value
                          << " is not 0, 1 or 2; JIT is off" << std::endl;
            return parsed;
#else
            if(!valid || parsed != Mode::Off)
                std::cerr << "hipblaslt warning: HIPBLASLT_JIT=" << value
                          << " is ignored: hipBLASLt was built without HIPBLASLT_ENABLE_JIT"
                          << std::endl;
            return Mode::Off;
#endif
        }
    }

    Mode mode() noexcept
    {
        static const Mode value = readMode();
        return value;
    }
}
