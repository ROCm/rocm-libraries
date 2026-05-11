/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#pragma once

#include "miopen_limits.hpp"
#include "miopen_cstdint.hpp"

#ifdef __HIPCC_RTC__
#define WORKAROUND_IGNORE_ROCRAND_INCLUDES 1
#else
#define WORKAROUND_IGNORE_ROCRAND_INCLUDES 0
#endif

#if WORKAROUND_IGNORE_ROCRAND_INCLUDES == 1
// disable math.h from rocrand (it conflicts with hiptrc)
// NOLINTNEXTLINE
#define _GLIBCXX_MATH_H 1
#endif

// disable normal-distribution shortcuts (it bloats prng state)
#define ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

// Use inlined rocrand header for runtime compilation to avoid external dependency
// For host code compilation, use the regular rocrand header.
// Suppress warnings from rocrand's own headers: in TheRock builds rocrand/stage/include
// is injected globally as -I (not -isystem), so rocrand headers are not treated as system
// includes and their internal issues trigger our warnings.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundef"
#ifdef MIOPEN_HIP_RUNTIME_COMPILE
#include "rocrand_xorwow_inlined.h"
#else
#include <rocrand/rocrand_xorwow.h>
#endif
#pragma clang diagnostic pop

namespace prng {

// based on splitmix64
inline constexpr uint64_t hash(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
};

// borrowed from <rocrand/rocrand_uniform.h>
// rocrand_uniform.h has too many dependencies and device code
inline constexpr float uniform_distribution(unsigned int v)
{
    constexpr float rocrand_2pow32_inv =
#ifdef ROCRAND_2POW32_INV
        ROCRAND_2POW32_INV;
#else
        2.3283064e-10f;
#endif

    return rocrand_2pow32_inv + (static_cast<float>(v) * rocrand_2pow32_inv);
}

inline constexpr float xorwow_uniform(rocrand_device::xorwow_engine* state)
{
    return uniform_distribution(state->next());
}

} // namespace prng
