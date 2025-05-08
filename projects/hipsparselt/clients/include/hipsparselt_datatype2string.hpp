/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

#include "auxiliary.hpp"
#include "hipsparselt_ostream.hpp"
#include <hipsparselt/hipsparselt.h>
#include <string>

enum class hipsparselt_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
    special    = 444,
};

constexpr auto hipsparselt_initialization2string(hipsparselt_initialization init)
{
    switch(init)
    {
    case hipsparselt_initialization::rand_int:
        return "rand_int";
    case hipsparselt_initialization::trig_float:
        return "trig_float";
    case hipsparselt_initialization::hpl:
        return "hpl";
    case hipsparselt_initialization::special:
        return "special";
    }
    return "invalid";
}

inline hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream& os,
                                                hipsparselt_initialization    init)
{
    return os << hipsparselt_initialization2string(init);
}

// clang-format off
inline hipsparselt_initialization string2hipsparselt_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? hipsparselt_initialization::rand_int   :
        value == "trig_float" ? hipsparselt_initialization::trig_float :
        value == "hpl"        ? hipsparselt_initialization::hpl        :
        value == "special"    ? hipsparselt_initialization::special        :
        static_cast<hipsparselt_initialization>(-1);
}
// clang-format on
