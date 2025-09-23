/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <Tensile/DistinctType.hpp>

namespace TensileLite
{
#if defined(TENSILE_USE_HIP) || defined(TENSILE_USE_FLOAT16_BUILTIN)
    /**
 * \ingroup DataTypes
 */
    using Half = _Float16;
#define TENSILE_USE_HALF
#else
    /**
 * \ingroup DataTypes
 */
    struct Half : public DistinctType<uint16_t, Half>
    {
    };
#endif
} // namespace TensileLite

namespace std
{
    inline ostream& operator<<(ostream& stream, const TensileLite::Half val)
    {
        return stream << static_cast<float>(val);
    }

    inline std::string to_string(const TensileLite::Half val)
    {
        return std::to_string(static_cast<float>(val));
    }
} // namespace std
