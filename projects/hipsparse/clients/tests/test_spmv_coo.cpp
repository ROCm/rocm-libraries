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

#include "hipsparse_arguments.hpp"
#include "testing_spmv_coo.hpp"

#include <hipsparse.h>

typedef std::
    tuple<int, int, double, double, hipsparseOperation_t, hipsparseIndexBase_t, hipsparseSpMVAlg_t>
        spmv_coo_tuple;
typedef std::tuple<double,
                   double,
                   hipsparseOperation_t,
                   hipsparseIndexBase_t,
                   hipsparseSpMVAlg_t,
                   std::string>
    spmv_coo_bin_tuple;

int spmv_coo_M_range[] = {50};
int spmv_coo_N_range[] = {84};

std::vector<double> spmv_coo_alpha_range = {2.0};
std::vector<double> spmv_coo_beta_range  = {1.0};

hipsparseOperation_t spmv_coo_transA_range[] = {HIPSPARSE_OPERATION_NON_TRANSPOSE};
hipsparseIndexBase_t spmv_coo_idxbase_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
#if(!defined(CUDART_VERSION))
hipsparseSpMVAlg_t spmv_coo_alg_range[]
    = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_COO_ALG1, HIPSPARSE_SPMV_COO_ALG2};
#else
#if(CUDART_VERSION >= 12000)
hipsparseSpMVAlg_t spmv_coo_alg_range[]
    = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_COO_ALG1, HIPSPARSE_SPMV_COO_ALG2};
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
hipsparseSpMVAlg_t spmv_coo_alg_range[]
    = {HIPSPARSE_SPMV_ALG_DEFAULT, HIPSPARSE_SPMV_COO_ALG1, HIPSPARSE_SPMV_COO_ALG2};
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
hipsparseSpMVAlg_t spmv_coo_alg_range[] = {HIPSPARSE_MV_ALG_DEFAULT, HIPSPARSE_COOMV_ALG};
#endif
#endif

std::string spmv_coo_bin[] = {"nos1.bin",
                              "nos2.bin",
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin",
                              "Chebyshev4.bin",
                              "shipsec1.bin"};

class parameterized_spmv_coo : public testing::TestWithParam<spmv_coo_tuple>
{
protected:
    parameterized_spmv_coo() {}
    virtual ~parameterized_spmv_coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spmv_coo_bin : public testing::TestWithParam<spmv_coo_bin_tuple>
{
protected:
    parameterized_spmv_coo_bin() {}
    virtual ~parameterized_spmv_coo_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spmv_coo_arguments(spmv_coo_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.beta     = std::get<3>(tup);
    arg.transA   = std::get<4>(tup);
    arg.baseA    = std::get<5>(tup);
    arg.spmv_alg = std::get<6>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_spmv_coo_arguments(spmv_coo_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = -99;
    arg.alpha    = std::get<0>(tup);
    arg.beta     = std::get<1>(tup);
    arg.transA   = std::get<2>(tup);
    arg.baseA    = std::get<3>(tup);
    arg.spmv_alg = std::get<4>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spmv_coo_bad_arg, spmv_coo_float)
{
    testing_spmv_coo_bad_arg();
}

TEST_P(parameterized_spmv_coo, spmv_coo_i32_float)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_coo, spmv_coo_i64_double)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_coo, spmv_coo_i32_float_complex)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_coo, spmv_coo_i64_double_complex)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_coo_bin, spmv_coo_bin_i32_float)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spmv_coo_bin, spmv_coo_bin_i64_double)
{
    Arguments arg = setup_spmv_coo_arguments(GetParam());

    hipsparseStatus_t status = testing_spmv_coo<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(spmv_coo,
                         parameterized_spmv_coo,
                         testing::Combine(testing::ValuesIn(spmv_coo_M_range),
                                          testing::ValuesIn(spmv_coo_N_range),
                                          testing::ValuesIn(spmv_coo_alpha_range),
                                          testing::ValuesIn(spmv_coo_beta_range),
                                          testing::ValuesIn(spmv_coo_transA_range),
                                          testing::ValuesIn(spmv_coo_idxbase_range),
                                          testing::ValuesIn(spmv_coo_alg_range)));

INSTANTIATE_TEST_SUITE_P(spmv_coo_bin,
                         parameterized_spmv_coo_bin,
                         testing::Combine(testing::ValuesIn(spmv_coo_alpha_range),
                                          testing::ValuesIn(spmv_coo_beta_range),
                                          testing::ValuesIn(spmv_coo_transA_range),
                                          testing::ValuesIn(spmv_coo_idxbase_range),
                                          testing::ValuesIn(spmv_coo_alg_range),
                                          testing::ValuesIn(spmv_coo_bin)));
#endif
