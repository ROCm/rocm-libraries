// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DATA_DESCRIPTOR_TEST_ACCESS_H
#define DATA_DESCRIPTOR_TEST_ACCESS_H

#include "data_set_descriptor.h"
#include <vector>

// Befriended by data_descriptor_t to let rocfft-internal-test drive its private members
struct data_descriptor_test_access
{
    static void reorder_length_axes(data_descriptor_t& layout, const std::vector<size_t>& order)
    {
        layout.reorder_length_axes(order);
    }
    static void clear(data_descriptor_t& layout)
    {
        layout.clear();
    }
    static void full_range_reset(data_descriptor_t&         layout,
                                 const std::vector<size_t>& lengths,
                                 size_t                     batch)
    {
        layout.full_range_reset(lengths, {}, {batch}, {}, false);
    }
    // mimics friends setting an array type without a precision
    static void set_array_type(data_descriptor_t& layout, rocfft_array_type array_type)
    {
        layout.array_type = array_type;
    }
};

#endif // DATA_DESCRIPTOR_TEST_ACCESS_H
