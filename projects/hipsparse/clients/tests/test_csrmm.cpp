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

#include "testing_csrmm.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>

typedef hipsparseIndexBase_t                                             base;
typedef hipsparseOperation_t                                             trans;
typedef std::tuple<int, int, int, double, double, base, trans, trans>    csrmm_tuple;
typedef std::tuple<int, double, double, base, trans, trans, std::string> csrmm_bin_tuple;

int csrmm_M_range[] = {0, 275, 1059};
int csrmm_N_range[] = {0, 78};
int csrmm_K_range[] = {0, 173, 1375};

double csrmm_alpha_range[] = {-0.5};
double csrmm_beta_range[]  = {0.5};

base  csrmm_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
trans csrmm_transA_range[]  = {HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              HIPSPARSE_OPERATION_TRANSPOSE,
                              HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE};
trans csrmm_transB_range[]  = {HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              HIPSPARSE_OPERATION_TRANSPOSE,
                              HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE};

std::string csrmm_bin[] = {"rma10.bin", "nos3.bin", "nos5.bin", "nos7.bin"};

class parameterized_csrmm : public testing::TestWithParam<csrmm_tuple>
{
protected:
    parameterized_csrmm() {}
    virtual ~parameterized_csrmm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrmm_bin : public testing::TestWithParam<csrmm_bin_tuple>
{
protected:
    parameterized_csrmm_bin() {}
    virtual ~parameterized_csrmm_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmm_arguments(csrmm_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.K      = std::get<2>(tup);
    arg.alpha  = std::get<3>(tup);
    arg.beta   = std::get<4>(tup);
    arg.baseA  = std::get<5>(tup);
    arg.transA = std::get<6>(tup);
    arg.transB = std::get<7>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csrmm_arguments(csrmm_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = std::get<0>(tup);
    arg.K      = -99;
    arg.alpha  = std::get<1>(tup);
    arg.beta   = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.transA = std::get<4>(tup);
    arg.transB = std::get<5>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(csrmm_bad_arg, csrmm_float)
{
    testing_csrmm_bad_arg<float>();
}

TEST_P(parameterized_csrmm, csrmm_float)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm, csrmm_double)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm, csrmm_float_complex)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm, csrmm_double_complex)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm_bin, csrmm_bin_float)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrmm_bin, csrmm_bin_double)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    hipsparseStatus_t status = testing_csrmm<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrmm,
                         parameterized_csrmm,
                         testing::Combine(testing::ValuesIn(csrmm_M_range),
                                          testing::ValuesIn(csrmm_N_range),
                                          testing::ValuesIn(csrmm_K_range),
                                          testing::ValuesIn(csrmm_alpha_range),
                                          testing::ValuesIn(csrmm_beta_range),
                                          testing::ValuesIn(csrmm_idxbase_range),
                                          testing::ValuesIn(csrmm_transA_range),
                                          testing::ValuesIn(csrmm_transB_range)));

INSTANTIATE_TEST_SUITE_P(csrmm_bin,
                         parameterized_csrmm_bin,
                         testing::Combine(testing::ValuesIn(csrmm_N_range),
                                          testing::ValuesIn(csrmm_alpha_range),
                                          testing::ValuesIn(csrmm_beta_range),
                                          testing::ValuesIn(csrmm_idxbase_range),
                                          testing::ValuesIn(csrmm_transA_range),
                                          testing::ValuesIn(csrmm_transB_range),
                                          testing::ValuesIn(csrmm_bin)));
#endif
