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

#include "testing_csr2hyb.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t, hipsparseHybPartition_t, int> csr2hyb_tuple;
typedef std::tuple<hipsparseIndexBase_t, hipsparseHybPartition_t, int, std::string>
    csr2hyb_bin_tuple;

int csr2hyb_M_range[] = {0, 10, 500, 872, 1000};
int csr2hyb_N_range[] = {0, 33, 242, 623, 1000};

hipsparseIndexBase_t csr2hyb_idx_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

hipsparseHybPartition_t csr2hyb_partition[]
    = {HIPSPARSE_HYB_PARTITION_AUTO, HIPSPARSE_HYB_PARTITION_MAX, HIPSPARSE_HYB_PARTITION_USER};

int csr2hyb_ELL_range[] = {-33, -1, 0, INT32_MAX};

std::string csr2hyb_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csr2hyb : public testing::TestWithParam<csr2hyb_tuple>
{
protected:
    parameterized_csr2hyb() {}
    virtual ~parameterized_csr2hyb() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2hyb_bin : public testing::TestWithParam<csr2hyb_bin_tuple>
{
protected:
    parameterized_csr2hyb_bin() {}
    virtual ~parameterized_csr2hyb_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2hyb_arguments(csr2hyb_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.baseA     = std::get<2>(tup);
    arg.part      = std::get<3>(tup);
    arg.ell_width = std::get<4>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csr2hyb_arguments(csr2hyb_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.baseA     = std::get<0>(tup);
    arg.part      = std::get<1>(tup);
    arg.ell_width = std::get<2>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(csr2hyb_bad_arg, csr2hyb)
{
    testing_csr2hyb_bad_arg<float>();
}

TEST_P(parameterized_csr2hyb, csr2hyb_float)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb, csr2hyb_double)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb, csr2hyb_float_complex)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb, csr2hyb_double_complex)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb_bin, csr2hyb_bin_float)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2hyb_bin, csr2hyb_bin_double)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2hyb<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csr2hyb,
                         parameterized_csr2hyb,
                         testing::Combine(testing::ValuesIn(csr2hyb_M_range),
                                          testing::ValuesIn(csr2hyb_N_range),
                                          testing::ValuesIn(csr2hyb_idx_base_range),
                                          testing::ValuesIn(csr2hyb_partition),
                                          testing::ValuesIn(csr2hyb_ELL_range)));

INSTANTIATE_TEST_SUITE_P(csr2hyb_bin,
                         parameterized_csr2hyb_bin,
                         testing::Combine(testing::ValuesIn(csr2hyb_idx_base_range),
                                          testing::ValuesIn(csr2hyb_partition),
                                          testing::ValuesIn(csr2hyb_ELL_range),
                                          testing::ValuesIn(csr2hyb_bin)));
#endif
