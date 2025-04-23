/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_LOCAL_VECTOR_HPP
#define TESTING_LOCAL_VECTOR_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_vector_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // LocalVector object
    LocalVector<T> vec;

    // SetDataPtr
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != (NULL|__null)*");
        ASSERT_DEATH(vec.SetDataPtr(&null_ptr, "", safe_size),
                     ".*Assertion.*ptr != (NULL|__null)*");
    }

    // LeaveDataPtr
    {
        T* vdata = nullptr;
        allocate_host(safe_size, &vdata);
        ASSERT_DEATH(vec.LeaveDataPtr(&vdata), ".*Assertion.*ptr == (NULL|__null)*");
        free_host(&vdata);
    }

    // CopyFromData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyFromData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // CopyToData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyToData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // GetContinuousValues
    {
        vec.Allocate("", safe_size);
        T* null_T = nullptr;
        ASSERT_DEATH(vec.GetContinuousValues(0, safe_size, null_T),
                     ".*Assertion.*values != (NULL|__null)*");
    }

    // ExtractCoarseMapping
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*map != (NULL|__null)*");
        free_host(&vint);
    }

    // ExtractCoarseBoundary
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*boundary != (NULL|__null)*");
        free_host(&vint);
    }

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_VECTOR_HPP
