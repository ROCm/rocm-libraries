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

#include "testing_csrilu02.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                                                     base;
typedef hipsparseSolvePolicy_t                                                   solve_policy;
typedef std::tuple<int, int, double, double, double, base, solve_policy>         csrilu02_tuple;
typedef std::tuple<int, double, double, double, base, solve_policy, std::string> csrilu02_bin_tuple;

int    csrilu02_M_range[]          = {0, 50, 647};
int    csrilu02_boost_range[]      = {0, 1};
double csrilu02_boost_tol_range[]  = {0.5};
double csrilu02_boost_val_range[]  = {0.3};
double csrilu02_boost_vali_range[] = {0.2};

base         csrilu02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
solve_policy csrilu02_solve_policy_range[]
    = {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL};

#if(!defined(CUDART_VERSION))
std::string csrilu02_bin[] = {
    "mac_econ_fwd500.bin", "rma10.bin", "nos1.bin", "nos2.bin", "nos3.bin", "nos5.bin", "nos6.bin"};
#elif(CUDART_VERSION >= 11080)
std::string csrilu02_bin[] = {"mac_econ_fwd500.bin", "nos3.bin", "nos5.bin", "nos6.bin"};
#else
// Note: There was a bug in csrilu02 where an infinite loop could occur on large matrices.
// This was fixed in cusparse 11.8
std::string csrilu02_bin[] = {"nos3.bin", "nos5.bin", "nos6.bin"};
#endif

class parameterized_csrilu02 : public testing::TestWithParam<csrilu02_tuple>
{
protected:
    parameterized_csrilu02() {}
    virtual ~parameterized_csrilu02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrilu02_bin : public testing::TestWithParam<csrilu02_bin_tuple>
{
protected:
    parameterized_csrilu02_bin() {}
    virtual ~parameterized_csrilu02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilu02_arguments(csrilu02_tuple tup)
{
    Arguments arg;
    arg.M            = std::get<0>(tup);
    arg.numericboost = std::get<1>(tup);
    arg.boosttol     = std::get<2>(tup);
    arg.boostval     = std::get<3>(tup);
    arg.boostvali    = std::get<4>(tup);
    arg.baseA        = std::get<5>(tup);
    arg.solve_policy = std::get<6>(tup);
    arg.timing       = 0;
    return arg;
}

Arguments setup_csrilu02_arguments(csrilu02_bin_tuple tup)
{
    Arguments arg;
    arg.M            = -99;
    arg.numericboost = std::get<0>(tup);
    arg.boosttol     = std::get<1>(tup);
    arg.boostval     = std::get<2>(tup);
    arg.boostvali    = std::get<3>(tup);
    arg.baseA        = std::get<4>(tup);
    arg.solve_policy = std::get<5>(tup);
    arg.timing       = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(csrilu02_bad_arg, csrilu02_float)
{
    testing_csrilu02_bad_arg<float>();
}

TEST_P(parameterized_csrilu02, csrilu02_float)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02, csrilu02_double)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02, csrilu02_float_complex)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02, csrilu02_double_complex)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02_bin, csrilu02_bin_float)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilu02_bin, csrilu02_bin_double)
{
    Arguments arg = setup_csrilu02_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilu02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrilu02,
                         parameterized_csrilu02,
                         testing::Combine(testing::ValuesIn(csrilu02_M_range),
                                          testing::ValuesIn(csrilu02_boost_range),
                                          testing::ValuesIn(csrilu02_boost_tol_range),
                                          testing::ValuesIn(csrilu02_boost_val_range),
                                          testing::ValuesIn(csrilu02_boost_vali_range),
                                          testing::ValuesIn(csrilu02_idxbase_range),
                                          testing::ValuesIn(csrilu02_solve_policy_range)));

INSTANTIATE_TEST_SUITE_P(csrilu02_bin,
                         parameterized_csrilu02_bin,
                         testing::Combine(testing::ValuesIn(csrilu02_boost_range),
                                          testing::ValuesIn(csrilu02_boost_tol_range),
                                          testing::ValuesIn(csrilu02_boost_val_range),
                                          testing::ValuesIn(csrilu02_boost_vali_range),
                                          testing::ValuesIn(csrilu02_idxbase_range),
                                          testing::ValuesIn(csrilu02_solve_policy_range),
                                          testing::ValuesIn(csrilu02_bin)));
#endif
