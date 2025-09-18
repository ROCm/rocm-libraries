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

#include "testing_gemvi.hpp"

#include <hipsparse.h>
#include <string>

typedef std::tuple<int, int, int, double, double, hipsparseOperation_t, hipsparseIndexBase_t>
    gemvi_tuple;

int gemvi_M_range[]   = {1291};
int gemvi_N_range[]   = {724};
int gemvi_nnz_range[] = {237};

double gemvi_alpha_range[] = {-0.5, 2.0};
double gemvi_beta_range[]  = {0.5, 0.0};

hipsparseOperation_t gemvi_trans_range[]    = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseIndexBase_t gemvi_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_gemvi : public testing::TestWithParam<gemvi_tuple>
{
protected:
    parameterized_gemvi() {}
    virtual ~parameterized_gemvi() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gemvi_arguments(gemvi_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.nnz    = std::get<2>(tup);
    arg.alpha  = std::get<3>(tup);
    arg.beta   = std::get<4>(tup);
    arg.transA = std::get<5>(tup);
    arg.baseA  = std::get<6>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(gemvi_bad_arg, gemvi_float)
{
    testing_gemvi_bad_arg();
}

TEST_P(parameterized_gemvi, gemvi_float)
{
    Arguments arg = setup_gemvi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemvi<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemvi, gemvi_double)
{
    Arguments arg = setup_gemvi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemvi<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemvi, gemvi_float_complex)
{
    Arguments arg = setup_gemvi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemvi<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gemvi, gemvi_double_complex)
{
    Arguments arg = setup_gemvi_arguments(GetParam());

    hipsparseStatus_t status = testing_gemvi<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gemvi,
                         parameterized_gemvi,
                         testing::Combine(testing::ValuesIn(gemvi_M_range),
                                          testing::ValuesIn(gemvi_N_range),
                                          testing::ValuesIn(gemvi_nnz_range),
                                          testing::ValuesIn(gemvi_alpha_range),
                                          testing::ValuesIn(gemvi_beta_range),
                                          testing::ValuesIn(gemvi_trans_range),
                                          testing::ValuesIn(gemvi_idx_base_range)));
#endif
