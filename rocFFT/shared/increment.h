// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_INCREMENT_H
#define ROCFFT_INCREMENT_H

#include <algorithm>
#include <tuple>
#include <vector>

// Helper functions to iterate over a buffer in row-major order.
// Indexes may be given as either a tuple or vector of sizes.  They
// return true if the index was successfully incremented to move to
// the next element in the buffer.

template <typename T1, typename T2>
static bool increment_base(T1& index, const T2& length)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    if(index < length - 1)
    {
        ++index;
        return true;
    }
    index = 0;
    return false;
}

// Increment the index (row-major) for looping over 1, 2, and 3 dimensions length.
template <typename T1, typename T2>
static bool increment_rowmajor(T1& index, const T2& length)
{
    static_assert(std::is_integral<T1>::value, "Integral required.");
    static_assert(std::is_integral<T2>::value, "Integral required.");
    return increment_base(index, length);
}

template <typename T1, typename T2>
static bool increment_rowmajor(std::tuple<T1, T1>& index, const std::tuple<T2, T2>& length)
{
    if(increment_base(std::get<1>(index), std::get<1>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<0>(index), std::get<0>(length));
}

template <typename T1, typename T2>
static bool increment_rowmajor(std::tuple<T1, T1, T1>& index, const std::tuple<T2, T2, T2>& length)
{
    if(increment_base(std::get<2>(index), std::get<2>(length)))
        // we incremented ok, nothing further to do
        return true;
    if(increment_base(std::get<1>(index), std::get<1>(length)))
        // we incremented ok, nothing further to do
        return true;
    // otherwise, we rolled over
    return increment_base(std::get<0>(index), std::get<0>(length));
}

// Increment row-major index over arbitrary dimension length
template <typename T1, typename T2>
bool increment_rowmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = length.size(); idim-- > 0;)
    {
        if(index[idim] < length[idim])
        {
            if((++index[idim]) == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            // we know we were able to increment something and didn't hit the end
            return true;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

#endif
