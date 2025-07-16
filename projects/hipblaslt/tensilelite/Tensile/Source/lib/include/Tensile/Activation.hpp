/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    /**
 * \ingroup Tensile
 * \defgroup ActivationTypes Activation Type Info
 *
 * @brief Definitions and metadata on supported activation types.
 */

    /**
 * \ingroup ActivationTypes
 * @{
 */

    /**
 * Data Type
 */
    enum class ActivationType : uint32_t
    {
        None = 0,
        Abs,
        Clippedrelu,
        Gelu,
        Leakyrelu,
        Relu,
        Sigmoid,
        Tanh,
        DGelu,
        Geluscaling,
        Silu,
        Swish,
        Clamp,
        All,
        Hipblaslt_all,
        Exp, // Verification use only.
        Count
    };

    std::string   ToString(ActivationType d);
    std::ostream& operator<<(std::ostream& stream, const ActivationType& t);
    std::istream& operator>>(std::istream& stream, ActivationType& t);

    int getAdditionalArgNum(ActivationType d);
}
