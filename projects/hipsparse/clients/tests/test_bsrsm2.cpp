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

#include "hipsparse_arguments.hpp"
#include "testing_bsrsm2.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int,
                   int,
                   int,
                   double,
                   hipsparseDirection_t,
                   hipsparseIndexBase_t,
                   hipsparseOperation_t,
                   hipsparseOperation_t>
    bsrsm2_tuple;
typedef std::tuple<int,
                   int,
                   double,
                   hipsparseDirection_t,
                   hipsparseIndexBase_t,
                   hipsparseOperation_t,
                   hipsparseOperation_t,
                   std::string>
    bsrsm2_bin_tuple;

int    bsrsm2_M_range[]     = {50};
int    bsrsm2_N_range[]     = {15};
int    bsrsm2_dim_range[]   = {2, 3};
double bsrsm2_alpha_range[] = {2.0};

hipsparseIndexBase_t bsrsm2_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseDirection_t bsrsm2_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};
hipsparseOperation_t bsrsm2_transA_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
hipsparseOperation_t bsrsm2_transB_range[]
    = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};

std::string bsrsm2_bin[] = {"nos3.bin"};

class parameterized_bsrsm2 : public testing::TestWithParam<bsrsm2_tuple>
{
protected:
    parameterized_bsrsm2() {}
    virtual ~parameterized_bsrsm2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsrsm2_bin : public testing::TestWithParam<bsrsm2_bin_tuple>
{
protected:
    parameterized_bsrsm2_bin() {}
    virtual ~parameterized_bsrsm2_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsrsm2_arguments(bsrsm2_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.block_dim = std::get<2>(tup);
    arg.alpha     = std::get<3>(tup);
    arg.dirA      = std::get<4>(tup);
    arg.baseA     = std::get<5>(tup);
    arg.transA    = std::get<6>(tup);
    arg.transB    = std::get<7>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_bsrsm2_arguments(bsrsm2_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = std::get<0>(tup);
    arg.block_dim = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.dirA      = std::get<3>(tup);
    arg.baseA     = std::get<4>(tup);
    arg.transA    = std::get<5>(tup);
    arg.transB    = std::get<6>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<7>(tup);

    // Get current executables absolute path

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(bsrsm2_bad_arg, bsrsm2_float)
{
    testing_bsrsm2_bad_arg();
}

TEST_P(parameterized_bsrsm2, bsrsm2_float)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrsm2, bsrsm2_double)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrsm2, bsrsm2_float_complex)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrsm2, bsrsm2_double_complex)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrsm2_bin, bsrsm2_bin_float)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsrsm2_bin, bsrsm2_bin_double)
{
    Arguments arg = setup_bsrsm2_arguments(GetParam());

    hipsparseStatus_t status = testing_bsrsm2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(bsrsm2,
                         parameterized_bsrsm2,
                         testing::Combine(testing::ValuesIn(bsrsm2_M_range),
                                          testing::ValuesIn(bsrsm2_N_range),
                                          testing::ValuesIn(bsrsm2_dim_range),
                                          testing::ValuesIn(bsrsm2_alpha_range),
                                          testing::ValuesIn(bsrsm2_dir_range),
                                          testing::ValuesIn(bsrsm2_idxbase_range),
                                          testing::ValuesIn(bsrsm2_transA_range),
                                          testing::ValuesIn(bsrsm2_transB_range)));

INSTANTIATE_TEST_SUITE_P(bsrsm2_bin,
                         parameterized_bsrsm2_bin,
                         testing::Combine(testing::ValuesIn(bsrsm2_N_range),
                                          testing::ValuesIn(bsrsm2_dim_range),
                                          testing::ValuesIn(bsrsm2_alpha_range),
                                          testing::ValuesIn(bsrsm2_dir_range),
                                          testing::ValuesIn(bsrsm2_idxbase_range),
                                          testing::ValuesIn(bsrsm2_transA_range),
                                          testing::ValuesIn(bsrsm2_transB_range),
                                          testing::ValuesIn(bsrsm2_bin)));
#endif
