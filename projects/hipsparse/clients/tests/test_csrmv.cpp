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

#include "testing_csrmv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseOperation_t                                 trans;
typedef hipsparseIndexBase_t                                 base;
typedef std::tuple<int, int, double, double, trans, base>    csrmv_tuple;
typedef std::tuple<double, double, trans, base, std::string> csrmv_bin_tuple;

int csr_M_range[] = {0, 500, 7111};
int csr_N_range[] = {0, 842, 4441};

std::vector<double> csr_alpha_range = {3.0};
std::vector<double> csr_beta_range  = {1.0};

// clang-format off
trans csr_trans_range[]   = {HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             HIPSPARSE_OPERATION_TRANSPOSE,
                             HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE};
// clang-format on
base csr_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csr_bin[] = {"nos1.bin",
                         "nos2.bin",
                         "nos3.bin",
                         "nos4.bin",
                         "nos5.bin",
                         "nos6.bin",
                         "nos7.bin",
                         "Chebyshev4.bin",
                         "shipsec1.bin"};

class parameterized_csrmv : public testing::TestWithParam<csrmv_tuple>
{
protected:
    parameterized_csrmv() {}
    virtual ~parameterized_csrmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrmv_bin : public testing::TestWithParam<csrmv_bin_tuple>
{
protected:
    parameterized_csrmv_bin() {}
    virtual ~parameterized_csrmv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmv_arguments(csrmv_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.beta   = std::get<3>(tup);
    arg.transA = std::get<4>(tup);
    arg.baseA  = std::get<5>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrmv_arguments(csrmv_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.alpha  = std::get<0>(tup);
    arg.beta   = std::get<1>(tup);
    arg.transA = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(csrmv_bad_arg, csrmv_float)
{
    testing_csrmv_bad_arg<float>();
}

TEST_P(parameterized_csrmv, csrmv_float)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv, csrmv_double)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv, csrmv_float_complex)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv, csrmv_double_complex)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv_bin, csrmv_bin_float)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmv_bin, csrmv_bin_double)
{
    Arguments arg = setup_csrmv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrmv,
                         parameterized_csrmv,
                         testing::Combine(testing::ValuesIn(csr_M_range),
                                          testing::ValuesIn(csr_N_range),
                                          testing::ValuesIn(csr_alpha_range),
                                          testing::ValuesIn(csr_beta_range),
                                          testing::ValuesIn(csr_trans_range),
                                          testing::ValuesIn(csr_idxbase_range)));

INSTANTIATE_TEST_SUITE_P(csrmv_bin,
                         parameterized_csrmv_bin,
                         testing::Combine(testing::ValuesIn(csr_alpha_range),
                                          testing::ValuesIn(csr_beta_range),
                                          testing::ValuesIn(csr_trans_range),
                                          testing::ValuesIn(csr_idxbase_range),
                                          testing::ValuesIn(csr_bin)));
#endif
