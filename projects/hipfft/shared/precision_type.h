// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_PRECISION_TYPE_H
#define ROCFFT_PRECISION_TYPE_H

#include "array_predicate.h"
#include "rocfft/rocfft.h"

static size_t real_type_size(rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_half:
        return 2;
    case rocfft_precision_single:
        return 4;
    case rocfft_precision_double:
        return 8;
    }
}

static size_t complex_type_size(rocfft_precision precision)
{
    return real_type_size(precision) * 2;
}

static const char* precision_name(rocfft_precision precision)
{
    switch(precision)
    {
    case rocfft_precision_half:
        return "half";
    case rocfft_precision_single:
        return "single";
    case rocfft_precision_double:
        return "double";
    }
}

static size_t element_size(rocfft_precision precision, rocfft_array_type array_type)
{
    return array_type_is_complex(array_type) ? complex_type_size(precision)
                                             : real_type_size(precision);
}

// offset a pointer by a number of elements, given the elements'
// precision and type (complex or not)
static void* ptr_offset(void* p, size_t elems, rocfft_precision precision, rocfft_array_type type)
{
    return static_cast<char*>(p) + elems * element_size(precision, type);
}
#endif
