/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <initializer_list>

#include <hiptensor/hiptensor_types.h>

namespace hiptensor
{
    // Every elementwise op (permute, binary, trinary) shares one rule: all tensors have the same
    // data type T, and the scalar type is T, or f32 when T is a 16-bit type.
    inline bool isSupportedElementwiseTypeCombination(
        std::initializer_list<hiptensorDataType_t> tensorTypes, hiptensorDataType_t scalarType)
    {
        if(tensorTypes.size() == 0)
        {
            return false;
        }

        auto const type = *tensorTypes.begin();
        if(std::any_of(
               tensorTypes.begin(), tensorTypes.end(), [type](auto t) { return t != type; }))
        {
            return false;
        }

        switch(type)
        {
        case HIPTENSOR_R_16F:
        case HIPTENSOR_R_16BF:
            return scalarType == type || scalarType == HIPTENSOR_R_32F;
        case HIPTENSOR_R_32F:
        case HIPTENSOR_R_64F:
            return scalarType == type;
        default:
            return false;
        }
    }
} // namespace hiptensor
