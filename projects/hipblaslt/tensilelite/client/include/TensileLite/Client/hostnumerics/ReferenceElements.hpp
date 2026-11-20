// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>

namespace TensileLite::Client
{
    // The output elements a TensileLite reference computes: every element, or
    // about `count` elements at a prime stride (see primeIndexRange).
    class ReferenceElements
    {
    public:
        static ReferenceElements all()
        {
            return ReferenceElements(std::nullopt);
        }
        static ReferenceElements sampled(size_t count)
        {
            return ReferenceElements(count);
        }
        std::optional<size_t> sampleCount() const
        {
            return m_sampleCount;
        }

    private:
        explicit ReferenceElements(std::optional<size_t> sampleCount)
            : m_sampleCount(sampleCount)
        {
        }
        std::optional<size_t> m_sampleCount;
    };
} // namespace TensileLite::Client
