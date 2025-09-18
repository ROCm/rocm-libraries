/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_identity.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <vector>

int identity_N_range[] = {0, 33, 242, 623, 1000};

class parameterized_identity : public testing::TestWithParam<int>
{
protected:
    parameterized_identity() {}
    virtual ~parameterized_identity() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_identity_arguments(int n)
{
    Arguments arg;
    arg.N      = n;
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(identity_bad_arg, identity)
{
    testing_identity_bad_arg();
}

TEST_P(parameterized_identity, identity)
{
    Arguments arg = setup_identity_arguments(GetParam());

    hipsparseStatus_t status = testing_identity(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(identity, parameterized_identity, testing::ValuesIn(identity_N_range));
#endif
