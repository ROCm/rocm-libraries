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

#include "testing_axpyi.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <vector>

typedef hipsparseIndexBase_t               base;
typedef std::tuple<int, int, double, base> axpyi_tuple;

int axpyi_N_range[]   = {15332};
int axpyi_nnz_range[] = {0, 10, 500, 7111};

std::vector<double> axpyi_alpha_range = {1.0, 0.0};

base axpyi_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_axpyi : public testing::TestWithParam<axpyi_tuple>
{
protected:
    parameterized_axpyi() {}
    virtual ~parameterized_axpyi() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_axpyi_arguments(axpyi_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(axpyi_bad_arg, axpyi_float)
{
    testing_axpyi_bad_arg<float>();
}

TEST_P(parameterized_axpyi, axpyi_float)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpyi, axpyi_double)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpyi, axpyi_float_complex)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_axpyi, axpyi_double_complex)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    hipsparseStatus_t status = testing_axpyi<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(axpyi,
                         parameterized_axpyi,
                         testing::Combine(testing::ValuesIn(axpyi_N_range),
                                          testing::ValuesIn(axpyi_nnz_range),
                                          testing::ValuesIn(axpyi_alpha_range),
                                          testing::ValuesIn(axpyi_idx_base_range)));
#endif
