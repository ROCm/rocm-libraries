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
#ifndef TESTING_PARALLEL_MANAGER_HPP
#define TESTING_PARALLEL_MANAGER_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_parallel_manager_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    ParallelManager pm;

    int* idata = nullptr;
    allocate_host(safe_size, &idata);

    // SetMPICommunicator
    {
        void* null_ptr = nullptr;
        ASSERT_DEATH(pm.SetMPICommunicator(null_ptr), ".*Assertion.*comm != (NULL|__null)*");
    }

    // SetBoundaryIndex
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetBoundaryIndex(safe_size, null_int),
                     ".*Assertion.*index != (NULL|__null)*");
    }

    // SetReceivers
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetReceivers(safe_size, null_int, idata),
                     ".*Assertion.*recvs != (NULL|__null)*");
        ASSERT_DEATH(pm.SetReceivers(safe_size, idata, null_int),
                     ".*Assertion.*recv_offset != (NULL|__null)*");
    }

    // SetSenders
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetSenders(safe_size, null_int, idata),
                     ".*Assertion.*sends != (NULL|__null)*");
        ASSERT_DEATH(pm.SetSenders(safe_size, idata, null_int),
                     ".*Assertion.*send_offset != (NULL|__null)*");
    }

    free_host(&idata);

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_PARALLEL_MANAGER_HPP
