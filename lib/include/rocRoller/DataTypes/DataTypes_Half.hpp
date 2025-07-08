/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#pragma once

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_fp16.h>
#endif

#include <functional>

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
//TODO: If ROCROLLER_USE_HIP is defined, invoke and use the HIP compiler
#if defined(ROCROLLER_USE_HIP) || defined(ROCROLLER_USE_FLOAT16_BUILTIN)
    /**
 * \ingroup DataTypes
 */
    using Half = __half;
#define ROCROLLER_USE_HALF
#else
    /**
 * \ingroup DataTypes
 */
    struct Half : public DistinctType<uint16_t, Half>
    {
    };
#endif
} // namespace rocRoller

namespace std
{
    inline ostream& operator<<(ostream& stream, const rocRoller::Half val)
    {
        return stream << static_cast<float>(val);
    }

    template <>
    struct hash<rocRoller::Half>
    {
        inline size_t operator()(rocRoller::Half const& h) const noexcept
        {
            hash<float> float_hash;
            return float_hash(static_cast<float>(h));
        }
    };

    template <>
    struct is_floating_point<rocRoller::Half> : std::true_type
    {
    };

    template <>
    struct is_signed<rocRoller::Half> : std::true_type
    {
    };
} // namespace std
