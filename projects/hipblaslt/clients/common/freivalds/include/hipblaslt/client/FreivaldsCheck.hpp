// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <roc/hostnumerics/freivalds.hpp>

namespace hipblaslt::client
{
    inline roc::hostnumerics::FreivaldsCheckOptions
        freivaldsCheckOptions(int                                     architectureMajor,
                              const roc::hostnumerics::FreivaldsPlan& plan,
                              size_t                                  reductionLength,
                              roc::hostnumerics::ScalarType           inputTypeA,
                              roc::hostnumerics::ScalarType           inputTypeB,
                              roc::hostnumerics::ScalarType           computeType)
    {
        using roc::hostnumerics::ScalarType;
        const auto narrow = [](ScalarType type) {
            return type == ScalarType::Float16 || type == ScalarType::BFloat16;
        };
        if(architectureMajor != 11 || computeType != ScalarType::Float32 || !narrow(inputTypeA)
           || !narrow(inputTypeB) || reductionLength == 0)
            return {};

        // GFX11 DOT2 complements terms opposing C's sign without an increment
        // (rocm-systems#12056). Alignment, complement and FP32 rounding cost at
        // most six unit roundoffs per pair. WMMA executes eight pairs, including
        // masked K tails; ten more cover the client's linear epilogue operations.
        // Absolute-sum bounds include scaling and prevent cancellation from
        // concealing the magnitude that determines the accumulator's grid.
        const double tiles = double(reductionLength / 16) + (reductionLength % 16 != 0);
        const double terms = (48 * tiles + 10) * 0x1p-24;
        const auto&  stats = plan.statistics();
        const double magnitude
            = std::max(stats.maximumPartialMagnitude, stats.maximumResultMagnitude);
        const double error
            = terms < 1 ? magnitude * terms / (1 - terms) : std::numeric_limits<double>::infinity();
        return {error == 0 ? 0 : std::nextafter(error, std::numeric_limits<double>::infinity())};
    }
}
