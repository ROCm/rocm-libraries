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

#include "testing_bsric02.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                                  base;
typedef hipsparseDirection_t                                  dir;
typedef hipsparseSolvePolicy_t                                solve_policy;
typedef std::tuple<int, int, dir, base, solve_policy>         bsric02_tuple;
typedef std::tuple<int, dir, base, solve_policy, std::string> bsric02_bin_tuple;

int bsric02_M_range[]   = {50, 426};
int bsric02_dim_range[] = {3, 5, 9};

base         bsric02_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
dir          bsric02_dir_range[]     = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};
solve_policy bsric02_solve_policy_range[]
    = {HIPSPARSE_SOLVE_POLICY_NO_LEVEL, HIPSPARSE_SOLVE_POLICY_USE_LEVEL};

std::string bsric02_bin[] = {"nos4.bin", "nos6.bin", "nos7.bin"};

class parameterized_bsric02 : public testing::TestWithParam<bsric02_tuple>
{
protected:
    parameterized_bsric02() {}
    virtual ~parameterized_bsric02() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_bsric02_bin : public testing::TestWithParam<bsric02_bin_tuple>
{
protected:
    parameterized_bsric02_bin() {}
    virtual ~parameterized_bsric02_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bsric02_arguments(bsric02_tuple tup)
{
    Arguments arg;
    arg.M            = std::get<0>(tup);
    arg.block_dim    = std::get<1>(tup);
    arg.dirA         = std::get<2>(tup);
    arg.baseA        = std::get<3>(tup);
    arg.solve_policy = std::get<4>(tup);
    arg.timing       = 0;
    return arg;
}

Arguments setup_bsric02_arguments(bsric02_bin_tuple tup)
{
    Arguments arg;
    arg.M            = -99;
    arg.block_dim    = std::get<0>(tup);
    arg.dirA         = std::get<1>(tup);
    arg.baseA        = std::get<2>(tup);
    arg.solve_policy = std::get<3>(tup);
    arg.timing       = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

    // Get current executables absolute path

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(bsric02_bad_arg, bsric02_float)
{
    testing_bsric02_bad_arg<float>();
}

TEST_P(parameterized_bsric02, bsric02_float)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsric02, bsric02_double)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsric02, bsric02_float_complex)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsric02, bsric02_double_complex)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsric02_bin, bsric02_bin_float)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_bsric02_bin, bsric02_bin_double)
{
    Arguments arg = setup_bsric02_arguments(GetParam());

    hipsparseStatus_t status = testing_bsric02<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(bsric02,
                         parameterized_bsric02,
                         testing::Combine(testing::ValuesIn(bsric02_M_range),
                                          testing::ValuesIn(bsric02_dim_range),
                                          testing::ValuesIn(bsric02_dir_range),
                                          testing::ValuesIn(bsric02_idxbase_range),
                                          testing::ValuesIn(bsric02_solve_policy_range)));

INSTANTIATE_TEST_SUITE_P(bsric02_bin,
                         parameterized_bsric02_bin,
                         testing::Combine(testing::ValuesIn(bsric02_dim_range),
                                          testing::ValuesIn(bsric02_dir_range),
                                          testing::ValuesIn(bsric02_idxbase_range),
                                          testing::ValuesIn(bsric02_solve_policy_range),
                                          testing::ValuesIn(bsric02_bin)));
#endif
