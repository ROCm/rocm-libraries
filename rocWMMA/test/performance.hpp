/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_PERFORMANCE_HPP
#define ROCWMMA_PERFORMANCE_HPP

#include <rocwmma/internal/types.hpp>

namespace rocwmma
{

    // Architectures
    class ArchGfx908;
    class ArchGfx90a;
    class Vega20;
    class DefaultArch;

    template <typename GfxArch, typename DataT>
    struct MfmaPerfTraits;

    template <>
    struct MfmaPerfTraits<DefaultArch, int8_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, float8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, bfloat8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, float8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, bfloat8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, bfloat16_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, float16_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, float32_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, xfloat32_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<DefaultArch, float64_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    // gfx908
    template <>
    struct MfmaPerfTraits<ArchGfx908, int8_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, float8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, bfloat8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, float8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, bfloat8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, bfloat16_t>
    {
        enum : uint32_t
        {
            Multiplier = 512
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, float16_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, float32_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, xfloat32_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx908, float64_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    // gfx90a
    template <>
    struct MfmaPerfTraits<ArchGfx90a, int8_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, float8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, bfloat8_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, float8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, bfloat8_fnuz_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, bfloat16_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, float16_t>
    {
        enum : uint32_t
        {
            Multiplier = 1024
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, float32_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, xfloat32_t>
    {
        enum : uint32_t
        {
            Multiplier = 0
        };
    };

    template <>
    struct MfmaPerfTraits<ArchGfx90a, float64_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

#if !ROCWMMA_TESTS_NO_HALF
    template <typename GfxArch>
    struct MfmaPerfTraits<GfxArch, hfloat16_t> : public MfmaPerfTraits<GfxArch, float16_t>
    {
    };
#endif // !ROCWMMA_TESTS_NO_HALF

    template <typename GfxArch, typename DataT>
    struct VALUPerfTraits;

    // MI-100
    template <>
    struct VALUPerfTraits<ArchGfx908, int8_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx908, bfloat16_t>
    {
        enum : uint32_t
        {
            Multiplier = 128
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx908, float16_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx908, float32_t>
    {
        enum : uint32_t
        {
            Multiplier = 128
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx908, float64_t>
    {
        enum : uint32_t
        {
            Multiplier = 64
        };
    };

    // ArchGfx90a
    template <>
    struct VALUPerfTraits<ArchGfx90a, int8_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx90a, bfloat16_t>
    {
        enum : uint32_t
        {
            Multiplier = 128
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx90a, float16_t>
    {
        enum : uint32_t
        {
            Multiplier = 256
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx90a, float32_t>
    {
        enum : uint32_t
        {
            Multiplier = 128
        };
    };

    template <>
    struct VALUPerfTraits<ArchGfx90a, float64_t>
    {
        enum : uint32_t
        {
            Multiplier = 128
        };
    };

#if !ROCWMMA_TESTS_NO_HALF
    template <typename GfxArch>
    struct VALUPerfTraits<GfxArch, hfloat16_t> : public VALUPerfTraits<GfxArch, float16_t>
    {
    };
#endif // !ROCWMMA_TESTS_NO_HALF

    inline double calculateGFlops(uint32_t m, uint32_t n, uint32_t k)
    {
        return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)
               * 1.0e-9;
    }

    inline double calculateTFlopsPerSec(uint32_t m, uint32_t n, uint32_t k, double elapsedTimeMs)
    {
        return calculateGFlops(m, n, k) / elapsedTimeMs;
    }

    template <typename InputT,
              typename GfxArch                               = DefaultArch,
              template <typename, typename> class PerfTraits = rocwmma::MfmaPerfTraits>
    inline double calculatePeakGFlopsPerSec(uint32_t freqMHz, uint32_t cuCount)
    {
        return static_cast<double>(PerfTraits<GfxArch, InputT>::Multiplier)
               * static_cast<double>(cuCount) * static_cast<double>(freqMHz) * 1.0e-3;
    }

} // namespace rocwmma

#endif // ROCWMMA_PERFORMANCE_HPP
