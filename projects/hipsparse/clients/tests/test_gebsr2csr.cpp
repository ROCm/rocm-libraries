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

#include "testing_gebsr2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::
    tuple<int, int, int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t>
        gebsr2csr_tuple;
typedef std::
    tuple<int, int, hipsparseIndexBase_t, hipsparseIndexBase_t, hipsparseDirection_t, std::string>
        gebsr2csr_bin_tuple;

// Random matrices
int gebsr2csr_M_range[]             = {872, 21453};
int gebsr2csr_N_range[]             = {623, 29285};
int gebsr2csr_row_block_dim_range[] = {2, 4, 8};
int gebsr2csr_col_block_dim_range[] = {2, 4, 8};

hipsparseIndexBase_t gebsr2csr_csr_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO};

hipsparseIndexBase_t gebsr2csr_bsr_base_range[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t gebsr2csr_dir_range[] = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

// Matrices from files (float and double)
int gebsr2csr_row_block_dim_range_bin[] = {2, 3};
int gebsr2csr_col_block_dim_range_bin[] = {3, 4};

hipsparseIndexBase_t gebsr2csr_csr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t gebsr2csr_bsr_base_range_bin[] = {HIPSPARSE_INDEX_BASE_ONE};

hipsparseDirection_t gebsr2csr_dir_range_bin[]
    = {HIPSPARSE_DIRECTION_ROW, HIPSPARSE_DIRECTION_COLUMN};

std::string gebsr2csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_gebsr2csr : public testing::TestWithParam<gebsr2csr_tuple>
{
protected:
    parameterized_gebsr2csr() {}
    virtual ~parameterized_gebsr2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_gebsr2csr_bin : public testing::TestWithParam<gebsr2csr_bin_tuple>
{
protected:
    parameterized_gebsr2csr_bin() {}
    virtual ~parameterized_gebsr2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gebsr2csr_arguments(gebsr2csr_tuple tup)
{
    Arguments arg;
    arg.M              = std::get<0>(tup);
    arg.N              = std::get<1>(tup);
    arg.row_block_dimA = std::get<2>(tup);
    arg.col_block_dimA = std::get<3>(tup);
    arg.baseA          = std::get<4>(tup);
    arg.baseB          = std::get<5>(tup);
    arg.dirA           = std::get<6>(tup);
    arg.timing         = 0;
    return arg;
}

Arguments setup_gebsr2csr_arguments(gebsr2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M              = -99;
    arg.N              = -99;
    arg.row_block_dimA = std::get<0>(tup);
    arg.col_block_dimA = std::get<1>(tup);
    arg.baseA          = std::get<2>(tup);
    arg.baseB          = std::get<3>(tup);
    arg.dirA           = std::get<4>(tup);
    arg.timing         = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(gebsr2csr_bad_arg, gebsr2csr)
{
    testing_gebsr2csr_bad_arg<float>();
}

TEST_P(parameterized_gebsr2csr, gebsr2csr_float)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2csr, gebsr2csr_double)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2csr, gebsr2csr_float_complex)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2csr, gebsr2csr_double_complex)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2csr_bin, gebsr2csr_bin_float)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_gebsr2csr_bin, gebsr2csr_bin_double)
{
    Arguments arg = setup_gebsr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_gebsr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gebsr2csr,
                         parameterized_gebsr2csr,
                         testing::Combine(testing::ValuesIn(gebsr2csr_M_range),
                                          testing::ValuesIn(gebsr2csr_N_range),
                                          testing::ValuesIn(gebsr2csr_row_block_dim_range),
                                          testing::ValuesIn(gebsr2csr_col_block_dim_range),
                                          testing::ValuesIn(gebsr2csr_bsr_base_range),
                                          testing::ValuesIn(gebsr2csr_csr_base_range),
                                          testing::ValuesIn(gebsr2csr_dir_range)));

INSTANTIATE_TEST_SUITE_P(gebsr2csr_bin,
                         parameterized_gebsr2csr_bin,
                         testing::Combine(testing::ValuesIn(gebsr2csr_row_block_dim_range_bin),
                                          testing::ValuesIn(gebsr2csr_col_block_dim_range_bin),
                                          testing::ValuesIn(gebsr2csr_bsr_base_range_bin),
                                          testing::ValuesIn(gebsr2csr_csr_base_range_bin),
                                          testing::ValuesIn(gebsr2csr_dir_range_bin),
                                          testing::ValuesIn(gebsr2csr_bin)));
