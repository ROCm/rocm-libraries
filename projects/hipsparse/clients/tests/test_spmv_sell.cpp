/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "testing_spmv_sell.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseSpMVAlg_t>
    spmv_sell_tuple;
typedef std::tuple<int,
                   double,
                   double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseSpMVAlg_t,
                   std::string>
    spmv_sell_bin_tuple;

int spmv_sell_M_range[]          = {50};
int spmv_sell_N_range[]          = {84};
int spmv_sell_slice_size_range[] = {2, 3, 4};

std::vector<double> spmv_sell_alpha_range = {2.0};
std::vector<double> spmv_sell_beta_range  = {1.0};

hipsparseOperation_t spmv_sell_transA_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseIndexBase_t spmv_sell_idxbase_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

#if(!defined(CUDART_VERSION))
hipsparseSpMVAlg_t spmv_sell_alg_range[] = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_SELL_ALG1};
#else
#if(CUDART_VERSION >= 12011)
hipsparseSpMVAlg_t spmv_sell_alg_range[] = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_SELL_ALG1};
#endif
#endif

std::string spmv_sell_bin[] = {"nos1.bin",
                               "nos2.bin",
                               "nos3.bin",
                               "nos4.bin",
                               "nos5.bin",
                               "nos6.bin",
                               "nos7.bin",
                               "Chebyshev4.bin",
                               "shipsec1.bin"};

class parameterized_spmv_sell : public testing::TestWithParam<spmv_sell_tuple>
{
protected:
    parameterized_spmv_sell() {}
    virtual ~parameterized_spmv_sell() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spmv_sell_bin : public testing::TestWithParam<spmv_sell_bin_tuple>
{
protected:
    parameterized_spmv_sell_bin() {}
    virtual ~parameterized_spmv_sell_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spmv_sell_arguments(spmv_sell_tuple tup)
{
    Arguments arg;
    arg.M          = std::get<0>(tup);
    arg.N          = std::get<1>(tup);
    arg.slice_size = std::get<2>(tup);
    arg.alpha      = std::get<3>(tup);
    arg.beta       = std::get<4>(tup);
    arg.transA     = std::get<5>(tup);
    arg.baseA      = std::get<6>(tup);
    arg.spmv_alg   = std::get<7>(tup);
    arg.timing     = 0;
    return arg;
}

Arguments setup_spmv_sell_arguments(spmv_sell_bin_tuple tup)
{
    Arguments arg;
    arg.M          = -99;
    arg.N          = -99;
    arg.slice_size = std::get<0>(tup);
    arg.alpha      = std::get<1>(tup);
    arg.beta       = std::get<2>(tup);
    arg.transA     = std::get<3>(tup);
    arg.baseA      = std::get<4>(tup);
    arg.spmv_alg   = std::get<5>(tup);
    arg.timing     = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(bin_file);

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)
TEST(spmv_sell_bad_arg, spmv_sell_float)
{
    testing_spmv_sell_bad_arg();
}

TEST_P(parameterized_spmv_sell, spmv_sell_i32_float)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_sell, spmv_sell_i64_double)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_sell, spmv_sell_i32_float_complex)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_sell, spmv_sell_i64_double_complex)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_sell_bin, spmv_sell_bin_i32_float)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_sell_bin, spmv_sell_bin_i64_double)
{
    Arguments arg = setup_spmv_sell_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_sell<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spmv_sell,
                         parameterized_spmv_sell,
                         testing::Combine(testing::ValuesIn(spmv_sell_M_range),
                                          testing::ValuesIn(spmv_sell_N_range),
                                          testing::ValuesIn(spmv_sell_slice_size_range),
                                          testing::ValuesIn(spmv_sell_alpha_range),
                                          testing::ValuesIn(spmv_sell_beta_range),
                                          testing::ValuesIn(spmv_sell_transA_range),
                                          testing::ValuesIn(spmv_sell_idxbase_range),
                                          testing::ValuesIn(spmv_sell_alg_range)));

INSTANTIATE_TEST_SUITE_P(spmv_sell_bin,
                         parameterized_spmv_sell_bin,
                         testing::Combine(testing::ValuesIn(spmv_sell_slice_size_range),
                                          testing::ValuesIn(spmv_sell_alpha_range),
                                          testing::ValuesIn(spmv_sell_beta_range),
                                          testing::ValuesIn(spmv_sell_transA_range),
                                          testing::ValuesIn(spmv_sell_idxbase_range),
                                          testing::ValuesIn(spmv_sell_alg_range),
                                          testing::ValuesIn(spmv_sell_bin)));
#endif
