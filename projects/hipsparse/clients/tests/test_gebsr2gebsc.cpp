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

#include "testing_gebsr2gebsc.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int, int, hipsparseAction_t, hipsparseIndexBase_t> gebsr2gebsc_tuple;
typedef std::tuple<int, int, hipsparseAction_t, hipsparseIndexBase_t, std::string>
    gebsr2gebsc_bin_tuple;

int gebsr2gebsc_M_range[] = {0, 10, 872};
int gebsr2gebsc_N_range[] = {0, 33, 623};

int gebsr2gebsc_row_block_dim_range[] = {1, 2, 7};
int gebsr2gebsc_col_block_dim_range[] = {1, 3, 4};

hipsparseAction_t gebsr2gebsc_action_range[]
    = {HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_ACTION_SYMBOLIC};

hipsparseIndexBase_t gebsr2gebsc_csr_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string gebsr2gebsc_bin[] = {"scircuit.bin",
                                 "nos1.bin",
                                 "nos2.bin",
                                 "nos3.bin",
                                 "nos4.bin",
                                 "nos5.bin",
                                 "nos6.bin",
                                 "nos7.bin"};

class parameterized_gebsr2gebsc : public testing::TestWithParam<gebsr2gebsc_tuple>
{
protected:
    parameterized_gebsr2gebsc() {}
    virtual ~parameterized_gebsr2gebsc() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gebsr2gebsc_bin : public testing::TestWithParam<gebsr2gebsc_bin_tuple>
{
protected:
    parameterized_gebsr2gebsc_bin() {}
    virtual ~parameterized_gebsr2gebsc_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gebsr2gebsc_arguments(gebsr2gebsc_tuple tup)
{
    Arguments arg;
    arg.M              = std::get<0>(tup);
    arg.N              = std::get<1>(tup);
    arg.row_block_dimA = std::get<2>(tup);
    arg.col_block_dimA = std::get<3>(tup);
    arg.action         = std::get<4>(tup);
    arg.baseA          = std::get<5>(tup);
    arg.timing         = 0;
    return arg;
}

Arguments setup_gebsr2gebsc_arguments(gebsr2gebsc_bin_tuple tup)
{
    Arguments arg;
    arg.M              = -99;
    arg.N              = -99;
    arg.row_block_dimA = std::get<0>(tup);
    arg.col_block_dimA = std::get<1>(tup);
    arg.action         = std::get<2>(tup);
    arg.baseA          = std::get<3>(tup);
    arg.timing         = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(gebsr2gebsc_bad_arg, gebsr2gebsc)
{
    testing_gebsr2gebsc_bad_arg<float>();
}

TEST_P(parameterized_gebsr2gebsc, gebsr2gebsc_float)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsc, gebsr2gebsc_double)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsc, gebsr2gebsc_float_complex)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsc, gebsr2gebsc_double_complex)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsc_bin, gebsr2gebsc_bin_float)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2gebsc_bin, gebsr2gebsc_bin_double)
{
    Arguments arg = setup_gebsr2gebsc_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2gebsc<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gebsr2gebsc,
                         parameterized_gebsr2gebsc,
                         testing::Combine(testing::ValuesIn(gebsr2gebsc_M_range),
                                          testing::ValuesIn(gebsr2gebsc_N_range),
                                          testing::ValuesIn(gebsr2gebsc_row_block_dim_range),
                                          testing::ValuesIn(gebsr2gebsc_col_block_dim_range),
                                          testing::ValuesIn(gebsr2gebsc_action_range),
                                          testing::ValuesIn(gebsr2gebsc_csr_base_range)));

INSTANTIATE_TEST_SUITE_P(gebsr2gebsc_bin,
                         parameterized_gebsr2gebsc_bin,
                         testing::Combine(testing::ValuesIn(gebsr2gebsc_row_block_dim_range),
                                          testing::ValuesIn(gebsr2gebsc_col_block_dim_range),
                                          testing::ValuesIn(gebsr2gebsc_action_range),
                                          testing::ValuesIn(gebsr2gebsc_csr_base_range),
                                          testing::ValuesIn(gebsr2gebsc_bin)));
