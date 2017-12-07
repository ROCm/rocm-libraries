// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_PAIR_HPP_
#define ROCPRIM_INTRINSICS_PAIR_HPP_

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<typename X, typename Y>
struct Pair
{
    X x;
    Y y;

    Pair() [[hc]]
    {
    }

    Pair(X key, Y value) [[hc]] : x(key), y(value)
    {
    }

    bool operator==(const Pair<X, Y>& obj) const [[hc]]
    {
        return (x == obj.x) && (y == obj.y);
    }

    bool operator!=(const Pair<X, Y>& obj) const [[hc]]
    {
        return !operator==(obj);
    }

    bool operator<(const Pair<X, Y>& obj) const [[hc]]
    {
        return (x < obj.x);
    }

    bool operator>(const Pair<X, Y>& obj) const [[hc]]
    {
        return (x > obj.x);
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_PAIR_HPP_
