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

#ifndef ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_
#define ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<class T>
struct double_buffer
{
    T * d_buffers[2];

    int selector;

    ROCPRIM_HOST_DEVICE inline
    double_buffer()
    {
        selector = 0;
        d_buffers[0] = nullptr;
        d_buffers[1] = nullptr;
    }

    ROCPRIM_HOST_DEVICE inline
    double_buffer(T * d_current, T * d_alternate)
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    ROCPRIM_HOST_DEVICE inline
    T * current() const
    {
        return d_buffers[selector];
    }

    ROCPRIM_HOST_DEVICE inline
    T * alternate() const
    {
        return d_buffers[selector ^ 1];
    }
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_
