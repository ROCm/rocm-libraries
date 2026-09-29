// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace hipblaslt_jit
{
    // 64-bit FNV-1a for names and change detection; it is not collision resistant.
    class Fnv1a
    {
    public:
        // Ends each field with a separator so that field boundaries change the hash.
        Fnv1a& add(std::string_view field) noexcept
        {
            for(unsigned char c : field)
                m_hash = (m_hash ^ c) * 1099511628211ull;
            m_hash = (m_hash ^ 0xff) * 1099511628211ull;
            return *this;
        }
        std::string hex() const
        {
            char text[17];
            std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(m_hash));
            return text;
        }

    private:
        uint64_t m_hash = 14695981039346656037ull;
    };
}
