// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace hipblaslt_clients
{
#ifdef _WIN32
    inline int setEnv(const char* name, const char* value, int overwrite)
    {
        if(!overwrite && std::getenv(name))
            return 0;
        return _putenv_s(name, value);
    }

    inline int unsetEnv(const char* name)
    {
        return _putenv_s(name, "");
    }

    inline int processId()
    {
        return _getpid();
    }
#else
    inline int setEnv(const char* name, const char* value, int overwrite)
    {
        return ::setenv(name, value, overwrite);
    }

    inline int unsetEnv(const char* name)
    {
        return ::unsetenv(name);
    }

    inline int processId()
    {
        return ::getpid();
    }
#endif
}
