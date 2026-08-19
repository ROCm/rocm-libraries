// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/generation_primitives.hpp>

namespace hipblaslt::host_validation
{
    inline constexpr uint64_t defaultInitializationSeed    = 69'069;
    inline constexpr uint64_t oneSpecialInitializationSeed = 12'345;

    namespace compatibility
    {
        inline constexpr uint64_t integerExactMatrixBRandomDomain      = 1'000'003;
        inline constexpr size_t   integerExactMatrixBDeviceIndexOffset = 1'000'003;
        inline constexpr size_t   legacyComplexImaginaryIndexOffset    = 1'000'000;

        inline constexpr uint32_t oneSpecialLcgMultiplier = 1'103'515'245u;
        inline constexpr uint32_t oneSpecialLcgIncrement  = 12'345u;
        inline constexpr unsigned oneSpecialLcgValueShift = 16;
        inline constexpr int      oneSpecialValueCount    = 3;

        // Adjusts the seed so changing the random stream ID does not change
        // the generated values.
        inline constexpr uint64_t seedForRandomDomain(
            uint64_t seed,
            uint64_t sourceDomain,
            uint64_t targetDomain
            = roc::host_validation::generation_random_domain_version_1::realComponent) noexcept
        {
            constexpr uint64_t offset
                = roc::host_validation::counter_random_version_1::domainOffset;
            return seed ^ (sourceDomain + offset) ^ (targetDomain + offset);
        }
    } // namespace compatibility
} // namespace hipblaslt::host_validation
