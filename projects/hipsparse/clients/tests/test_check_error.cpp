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

#include "utility.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

// Host-only. The pre_checkin prefix is the name CI selects. TheRock matches
// pre_checkin/* and Jenkins matches *checkin*. A plain TEST() has no such
// prefix, so it never runs in either lane.
class check_error : public testing::TestWithParam<int>
{
};

namespace
{
    void expect_message_has_location_and_call(const std::runtime_error& error, const char* call)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(__FILE__), std::string::npos) << message;
        EXPECT_NE(message.find(call), std::string::npos) << message;
    }
}

TEST_P(check_error, throws_with_location)
{
    EXPECT_NO_THROW(CHECK_HIP_ERROR(hipSuccess));
    EXPECT_THROW(CHECK_HIP_ERROR(hipErrorInvalidValue), std::runtime_error);
    try
    {
        CHECK_HIP_ERROR(hipErrorInvalidValue);
        FAIL() << "CHECK_HIP_ERROR did not throw";
    }
    catch(const std::runtime_error& error)
    {
        expect_message_has_location_and_call(error, "hipErrorInvalidValue");
    }

    EXPECT_NO_THROW(CHECK_HIPSPARSE_ERROR(HIPSPARSE_STATUS_SUCCESS));
    EXPECT_THROW(CHECK_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INVALID_VALUE), std::runtime_error);
    try
    {
        CHECK_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INVALID_VALUE);
        FAIL() << "CHECK_HIPSPARSE_ERROR did not throw";
    }
    catch(const std::runtime_error& error)
    {
        expect_message_has_location_and_call(error, "HIPSPARSE_STATUS_INVALID_VALUE");
    }
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, check_error, testing::Values(0));
