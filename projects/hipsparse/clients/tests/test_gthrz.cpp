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

#include "testing_gthrz.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <vector>

typedef hipsparseIndexBase_t       base;
typedef std::tuple<int, int, base> gthrz_tuple;

int gthrz_N_range[]   = {12000, 15332, 22031};
int gthrz_nnz_range[] = {0, 5, 10, 500, 1000, 7111, 10000};

base gthrz_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_gthrz : public testing::TestWithParam<gthrz_tuple>
{
protected:
    parameterized_gthrz() {}
    virtual ~parameterized_gthrz() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gthrz_arguments(gthrz_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(gthrz_bad_arg, gthrz_float)
{
    testing_gthrz_bad_arg<float>();
}

TEST_P(parameterized_gthrz, gthrz_float)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    hipsparseStatus_t status = testing_gthrz<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthrz, gthrz_double)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    hipsparseStatus_t status = testing_gthrz<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthrz, gthrz_float_complex)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    hipsparseStatus_t status = testing_gthrz<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gthrz, gthrz_double_complex)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    hipsparseStatus_t status = testing_gthrz<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gthrz,
                         parameterized_gthrz,
                         testing::Combine(testing::ValuesIn(gthrz_N_range),
                                          testing::ValuesIn(gthrz_nnz_range),
                                          testing::ValuesIn(gthrz_idx_base_range)));
#endif