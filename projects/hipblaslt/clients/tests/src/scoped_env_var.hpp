// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <optional>
#include <string>

// Sets or unsets an environment variable for the lifetime of the object and
// restores the previous value, so an early return or a failed assertion does
// not leak it into later tests. A null value unsets the variable.
class ScopedEnvVar
{
public:
    ScopedEnvVar(const char* name, const char* value)
        : m_name(name)
    {
        if(const char* cur = std::getenv(name))
            m_old = cur;
        set(value);
    }

    ~ScopedEnvVar()
    {
        set(m_old ? m_old->c_str() : nullptr);
    }

    ScopedEnvVar(const ScopedEnvVar&)            = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    void set(const char* value)
    {
#if defined(_WIN32)
        _putenv_s(m_name.c_str(), value ? value : "");
#else
        if(value)
            setenv(m_name.c_str(), value, 1);
        else
            unsetenv(m_name.c_str());
#endif
    }

    std::string                m_name;
    std::optional<std::string> m_old;
};
