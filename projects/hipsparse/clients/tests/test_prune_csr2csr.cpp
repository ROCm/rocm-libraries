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

#include "testing_prune_csr2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, double, hipsparseIndexBase_t, hipsparseIndexBase_t>
    prune_csr2csr_tuple;
typedef std::tuple<double, hipsparseIndexBase_t, hipsparseIndexBase_t, std::string>
    prune_csr2csr_bin_tuple;

int    prune_csr2csr_M_range[]         = {10, 500, 872, 27463, 35327};
int    prune_csr2csr_N_range[]         = {33, 242, 623, 29837, 22645};
double prune_csr2csr_threshold_range[] = {0.0, 0.0012, 0.08736, 0.33333, 0.5, 1.7};

hipsparseIndexBase_t prune_csr2csr_base_A_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

hipsparseIndexBase_t prune_csr2csr_base_C_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string prune_csr2csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_prune_csr2csr : public testing::TestWithParam<prune_csr2csr_tuple>
{
protected:
    parameterized_prune_csr2csr() {}
    virtual ~parameterized_prune_csr2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_prune_csr2csr_bin : public testing::TestWithParam<prune_csr2csr_bin_tuple>
{
protected:
    parameterized_prune_csr2csr_bin() {}
    virtual ~parameterized_prune_csr2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_prune_csr2csr_arguments(prune_csr2csr_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.threshold = std::get<2>(tup);
    arg.baseA     = std::get<3>(tup);
    arg.baseB     = std::get<4>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_prune_csr2csr_arguments(prune_csr2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.threshold = std::get<0>(tup);
    arg.baseA     = std::get<1>(tup);
    arg.baseB     = std::get<2>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(prune_csr2csr_bad_arg, prune_csr2csr)
{
    testing_prune_csr2csr_bad_arg<float>();
}

TEST_P(parameterized_prune_csr2csr, prune_csr2csr_float)
{
    Arguments arg = setup_prune_csr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr, prune_csr2csr_double)
{
    Arguments arg = setup_prune_csr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr_bin, prune_csr2csr_bin_float)
{
    Arguments arg = setup_prune_csr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_csr2csr_bin, prune_csr2csr_bin_double)
{
    Arguments arg = setup_prune_csr2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_csr2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(prune_csr2csr,
                         parameterized_prune_csr2csr,
                         testing::Combine(testing::ValuesIn(prune_csr2csr_M_range),
                                          testing::ValuesIn(prune_csr2csr_N_range),
                                          testing::ValuesIn(prune_csr2csr_threshold_range),
                                          testing::ValuesIn(prune_csr2csr_base_A_range),
                                          testing::ValuesIn(prune_csr2csr_base_C_range)));

INSTANTIATE_TEST_SUITE_P(prune_csr2csr_bin,
                         parameterized_prune_csr2csr_bin,
                         testing::Combine(testing::ValuesIn(prune_csr2csr_threshold_range),
                                          testing::ValuesIn(prune_csr2csr_base_A_range),
                                          testing::ValuesIn(prune_csr2csr_base_C_range),
                                          testing::ValuesIn(prune_csr2csr_bin)));
#endif
