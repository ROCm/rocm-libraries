// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <optional>
#include <string>

// Sets an environment variable, or unsets it for nullptr, until the end of the scope.
class ScopedEnv
{
public:
    ScopedEnv(const char* name, const char* value)
        : m_name(name)
    {
        if(const char* old = std::getenv(name))
            m_saved = old;
        set(value);
    }

    ~ScopedEnv()
    {
        set(m_saved ? m_saved->c_str() : nullptr);
    }

    ScopedEnv(const ScopedEnv&)            = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    void set(const char* value)
    {
#ifdef _WIN32
        _putenv_s(m_name, value ? value : "");
#else
        if(value)
            setenv(m_name, value, 1);
        else
            unsetenv(m_name);
#endif
    }

    const char*                m_name;
    std::optional<std::string> m_saved;
};
