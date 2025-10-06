/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csrsv2.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t   base;
typedef hipsparseOperation_t   op;
typedef hipsparseDiagType_t    diag;
typedef hipsparseFillMode_t    fill;
typedef hipsparseSolvePolicy_t policy;

typedef std::tuple<int, double, base, op, diag, fill, policy>         csrsv2_tuple;
typedef std::tuple<double, base, op, diag, fill, policy, std::string> csrsv2_bin_tuple;

int csrsv2_M_range[] = {0, 647};

double csrsv2_alpha_range[] = {2.3};

base   csrsv2_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
op     csrsv2_op_range[]      = {HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE};
diag   csrsv2_diag_range[]    = {HIPSPARSE_DIAG_TYPE_NON_UNIT};
fill   csrsv2_fill_range[]    = {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER};
policy csrsv2_policy_range[]  = {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL};

std::string csrsv2_bin[] = {"rma10.bin",
                            "mc2depi.bin",
                            "nos1.bin",
                            "nos2.bin",
                            "nos3.bin",
                            "nos4.bin",
                            "nos5.bin",
                            "nos6.bin",
                            "sme3Dc.bin"};

class parameterized_csrsv2 : public testing::TestWithParam<csrsv2_tuple>
{
protected:
    parameterized_csrsv2() {}
    virtual ~parameterized_csrsv2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrsv2_bin : public testing::TestWithParam<csrsv2_bin_tuple>
{
protected:
    parameterized_csrsv2_bin() {}
    virtual ~parameterized_csrsv2_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrsv2_arguments(csrsv2_tuple tup)
{
    Arguments arg;
    arg.M            = std::get<0>(tup);
    arg.alpha        = std::get<1>(tup);
    arg.baseA        = std::get<2>(tup);
    arg.transA       = std::get<3>(tup);
    arg.diag_type    = std::get<4>(tup);
    arg.fill_mode    = std::get<5>(tup);
    arg.solve_policy = std::get<6>(tup);
    arg.timing       = 0;
    return arg;
}

Arguments setup_csrsv2_arguments(csrsv2_bin_tuple tup)
{
    Arguments arg;
    arg.M            = -99;
    arg.alpha        = std::get<0>(tup);
    arg.baseA        = std::get<1>(tup);
    arg.transA       = std::get<2>(tup);
    arg.diag_type    = std::get<3>(tup);
    arg.fill_mode    = std::get<4>(tup);
    arg.solve_policy = std::get<5>(tup);
    arg.timing       = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
TEST(csrsv2_bad_arg, csrsv2_float)
{
    testing_csrsv2_bad_arg<float>();
}

TEST_P(parameterized_csrsv2, csrsv2_float)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsv2, csrsv2_double)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsv2, csrsv2_float_complex)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsv2, csrsv2_double_complex)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsv2_bin, csrsv2_bin_float)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsv2_bin, csrsv2_bin_double)
{
    Arguments arg = setup_csrsv2_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsv2<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrsv2,
                         parameterized_csrsv2,
                         testing::Combine(testing::ValuesIn(csrsv2_M_range),
                                          testing::ValuesIn(csrsv2_alpha_range),
                                          testing::ValuesIn(csrsv2_idxbase_range),
                                          testing::ValuesIn(csrsv2_op_range),
                                          testing::ValuesIn(csrsv2_diag_range),
                                          testing::ValuesIn(csrsv2_fill_range),
                                          testing::ValuesIn(csrsv2_policy_range)));

INSTANTIATE_TEST_SUITE_P(csrsv2_bin,
                         parameterized_csrsv2_bin,
                         testing::Combine(testing::ValuesIn(csrsv2_alpha_range),
                                          testing::ValuesIn(csrsv2_idxbase_range),
                                          testing::ValuesIn(csrsv2_op_range),
                                          testing::ValuesIn(csrsv2_diag_range),
                                          testing::ValuesIn(csrsv2_fill_range),
                                          testing::ValuesIn(csrsv2_policy_range),
                                          testing::ValuesIn(csrsv2_bin)));
#endif
