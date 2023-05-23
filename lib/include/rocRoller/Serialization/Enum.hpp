/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2023 Advanced Micro Devices, Inc.
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

#include <type_traits>

#include "../DataTypes/DataTypes.hpp"

#include "Base.hpp"

namespace rocRoller
{
    namespace Serialization
    {
        /**
         * Provides serialization for any enumeration type that adheres to the CCountedEnum concept;
         * i.e. provides a Count value and toString.
         */
        template <CCountedEnum Enum, typename IO>
        struct EnumTraits<Enum, IO>
        {
            using iot        = IOTraits<IO>;
            using underlying = std::underlying_type_t<Enum>;

            static void enumeration(IO& io, Enum& ref)
            {
                for(int i = 0; i < static_cast<int>(Enum::Count); i++)
                {
                    auto value = static_cast<Enum>(i);
                    auto str   = toString(value);
                    iot::enumCase(io, ref, str.c_str(), value);
                }
            }
        };

    } // namespace Serialization
} // namespace Tensile
