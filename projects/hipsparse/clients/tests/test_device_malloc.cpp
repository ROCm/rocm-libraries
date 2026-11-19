/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse_test_unique_ptr.hpp"
#include "utility.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// A failed hipMalloc used to return nullptr. device_malloc must throw instead,
// so the caller cannot continue with an invalid pointer.
//
// The pre_checkin prefix is the name CI selects. TheRock matches pre_checkin/*
// and Jenkins matches *checkin*. A plain TEST() has no such prefix, so it
// never runs in either lane.
class device_malloc : public testing::TestWithParam<int>
{
};

TEST_P(device_malloc, throws_on_allocation_failure)
{
    EXPECT_THROW(hipsparse_test::device_malloc(SIZE_MAX), std::runtime_error);
    // Reset the last error so later tests don't see the failed hipMalloc.
    (void)hipGetLastError();
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, device_malloc, testing::Values(0));
