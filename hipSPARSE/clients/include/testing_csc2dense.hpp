/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef TESTING_CSC2DENSE_HPP
#define TESTING_CSC2DENSE_HPP

#include "testing_csx2dense.hpp"

template <typename T>
void testing_csc2dense_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    static constexpr hipsparseDirection_t DIRA = HIPSPARSE_DIRECTION_COLUMN;
    testing_csx2dense_bad_arg<DIRA, T>(hipsparseXcsc2dense<T>);
#endif
}

template <typename T>
hipsparseStatus_t testing_csc2dense(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
    static constexpr hipsparseDirection_t DIRA = HIPSPARSE_DIRECTION_COLUMN;
    return testing_csx2dense<DIRA, T>(argus, hipsparseXcsc2dense<T>, hipsparseXdense2csc<T>);
#else
    return HIPSPARSE_STATUS_SUCCESS;
#endif
}

#endif // TESTING_CSC2DENSE
