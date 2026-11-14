/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <limits>
#include <type_traits>

#include "rocblas_utility.hpp"

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
#define ROCSOLVER_HAS_BUILTIN_MUL_OVERFLOW 1
#endif
#endif

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    Overflow-checked integer arithmetic helpers, used when computing
 *    buffer sizes and strides that must not silently wrap around.
 * ===========================================================================
 */

// Base case: a single factor cannot overflow on its own. T is deduced from out
// alone so callers can pass factors of any integer type (e.g. rocblas_int
// dimensions multiplied into a size_t) without casting at the call site. T is
// required to be unsigned so the result type matches size/stride math and the
// overflow check behaves identically with or without the compiler builtin.
template <typename T, typename U>
bool rocsolver_mul_overflow(T& out, U last)
{
    static_assert(std::is_unsigned<T>::value, "checked multiplication requires an unsigned type");
    out = static_cast<T>(last);
    return false;
}

// Multiplies all factors into out, returning true when the result overflows T
// and false when it fits. On overflow out holds the wrapped low-order bits
// (matching C23 ckd_mul) and callers must not use it. The factors are folded
// from the tail inward, short-circuiting the moment a product overflows.
template <typename T, typename U, typename... Us>
bool rocsolver_mul_overflow(T& out, U first, Us... rest)
{
    static_assert(std::is_unsigned<T>::value, "checked multiplication requires an unsigned type");
    T partial;
    if(rocsolver_mul_overflow(partial, rest...))
        return true;

#ifdef ROCSOLVER_HAS_BUILTIN_MUL_OVERFLOW
    return __builtin_mul_overflow(static_cast<T>(first), partial, &out);
#else
    out = static_cast<T>(first) * partial;
    return first != 0 && out / static_cast<T>(first) != partial;
#endif
}

// Multiplies all factors and returns the product, saturated to the maximum
// value of T on overflow. Intended for workspace-size math: a saturated size
// flows into the device allocator and fails cleanly with a memory error,
// instead of a wrapped-small size that would under-allocate the workspace.
template <typename T, typename... Us>
T rocsolver_saturating_mul(Us... factors)
{
    T out;
    if(rocsolver_mul_overflow(out, factors...))
        return std::numeric_limits<T>::max();
    return out;
}

ROCSOLVER_END_NAMESPACE

#undef ROCSOLVER_HAS_BUILTIN_MUL_OVERFLOW
