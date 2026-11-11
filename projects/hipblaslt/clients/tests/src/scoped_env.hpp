// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "client_process.hpp"

#include <cstdlib>
#include <optional>
#include <string>

namespace test_env
{
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
            if(value)
                hipblaslt_clients::setEnv(m_name, value, 1);
            else
                hipblaslt_clients::unsetEnv(m_name);
        }

        const char*                m_name;
        std::optional<std::string> m_saved;
    };
} // namespace test_env
