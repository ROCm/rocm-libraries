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

#include "testing_scatter.hpp"

#include <hipsparse.h>

typedef hipsparseIndexBase_t       base;
typedef std::tuple<int, int, base> scatter_tuple;

int  scatter_N_range[]        = {12000, 15332, 22031};
int  scatter_nnz_range[]      = {0, 5, 10, 500, 1000, 7111, 10000};
base scatter_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_scatter : public testing::TestWithParam<scatter_tuple>
{
protected:
    parameterized_scatter() {}
    virtual ~parameterized_scatter() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_scatter_arguments(scatter_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
TEST(scatter_bad_arg, scatter_float)
{
    testing_scatter_bad_arg();
}

TEST_P(parameterized_scatter, scatter_i32_float)
{
    Arguments arg = setup_scatter_arguments(GetParam());

    hipsparseStatus_t status = testing_scatter<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_scatter, scatter_i64_double)
{
    Arguments arg = setup_scatter_arguments(GetParam());

    hipsparseStatus_t status = testing_scatter<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_scatter, scatter_i32_float_complex)
{
    Arguments arg = setup_scatter_arguments(GetParam());

    hipsparseStatus_t status = testing_scatter<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_scatter, scatter_i64_double_complex)
{
    Arguments arg = setup_scatter_arguments(GetParam());

    hipsparseStatus_t status = testing_scatter<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(scatter,
                         parameterized_scatter,
                         testing::Combine(testing::ValuesIn(scatter_N_range),
                                          testing::ValuesIn(scatter_nnz_range),
                                          testing::ValuesIn(scatter_idx_base_range)));
#endif
