/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_gtsv.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int> gtsv_tuple;

int gtsv_M_range[] = {512};
int gtsv_N_range[] = {512};

class parameterized_gtsv : public testing::TestWithParam<gtsv_tuple>
{
protected:
    parameterized_gtsv() {}
    virtual ~parameterized_gtsv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gtsv_arguments(gtsv_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.timing = 0;
    return arg;
}

TEST(gtsv_bad_arg, gtsv_float)
{
    testing_gtsv2_bad_arg<float>();
}

TEST_P(parameterized_gtsv, gtsv_float)
{
    Arguments arg = setup_gtsv_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv, gtsv_double)
{
    Arguments arg = setup_gtsv_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv, gtsv_float_complex)
{
    Arguments arg = setup_gtsv_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv, gtsv_double_complex)
{
    Arguments arg = setup_gtsv_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gtsv,
                         parameterized_gtsv,
                         testing::Combine(testing::ValuesIn(gtsv_M_range),
                                          testing::ValuesIn(gtsv_N_range)));
