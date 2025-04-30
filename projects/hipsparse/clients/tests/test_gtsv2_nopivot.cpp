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

#include "testing_gtsv2_nopivot.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int> gtsv2_nopivot_tuple;

int gtsv2_nopivot_M_range[] = {512};
int gtsv2_nopivot_N_range[] = {512};

class parameterized_gtsv2_nopivot : public testing::TestWithParam<gtsv2_nopivot_tuple>
{
protected:
    parameterized_gtsv2_nopivot() {}
    virtual ~parameterized_gtsv2_nopivot() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gtsv2_nopivot_arguments(gtsv2_nopivot_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.timing = 0;
    return arg;
}

TEST(gtsv2_nopivot_bad_arg, gtsv2_nopivot_float)
{
    testing_gtsv2_nopivot_bad_arg<float>();
}

TEST_P(parameterized_gtsv2_nopivot, gtsv2_nopivot_float)
{
    Arguments arg = setup_gtsv2_nopivot_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_nopivot<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_nopivot, gtsv2_nopivot_double)
{
    Arguments arg = setup_gtsv2_nopivot_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_nopivot<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_nopivot, gtsv2_nopivot_float_complex)
{
    Arguments arg = setup_gtsv2_nopivot_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_nopivot<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gtsv2_nopivot, gtsv2_nopivot_double_complex)
{
    Arguments arg = setup_gtsv2_nopivot_arguments(GetParam());

    hipsparseStatus_t status = testing_gtsv2_nopivot<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gtsv2_nopivot,
                         parameterized_gtsv2_nopivot,
                         testing::Combine(testing::ValuesIn(gtsv2_nopivot_M_range),
                                          testing::ValuesIn(gtsv2_nopivot_N_range)));
