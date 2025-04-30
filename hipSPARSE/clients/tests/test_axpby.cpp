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

#include "testing_axpby.hpp"
#include <hipsparse.h>

typedef hipsparseIndexBase_t               base;
typedef std::tuple<int, int, double, base> axpby_tuple;

int axpby_N_range[]   = {22031};
int axpby_nnz_range[] = {0, 5, 1000, 10000};

std::vector<double> axpby_alpha_range = {1.0, 0.0};

base axpby_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_axpby : public testing::TestWithParam<axpby_tuple>
{
protected:
    parameterized_axpby() {}
    virtual ~parameterized_axpby() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_axpby_arguments(axpby_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
TEST(axpby_bad_arg, axpby_float)
{
    testing_axpby_bad_arg();
}

TEST_P(parameterized_axpby, axpby_i32_float)
{
    Arguments arg = setup_axpby_arguments(GetParam());

    hipsparseStatus_t status = testing_axpby<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpby, axpby_i64_double)
{
    Arguments arg = setup_axpby_arguments(GetParam());

    hipsparseStatus_t status = testing_axpby<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpby, axpby_i32_float_complex)
{
    Arguments arg = setup_axpby_arguments(GetParam());

    hipsparseStatus_t status = testing_axpby<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpby, axpby_i64_double_complex)
{
    Arguments arg = setup_axpby_arguments(GetParam());

    hipsparseStatus_t status = testing_axpby<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(axpby,
                         parameterized_axpby,
                         testing::Combine(testing::ValuesIn(axpby_N_range),
                                          testing::ValuesIn(axpby_nnz_range),
                                          testing::ValuesIn(axpby_alpha_range),
                                          testing::ValuesIn(axpby_idx_base_range)));
#endif
