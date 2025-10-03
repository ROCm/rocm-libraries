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

#include "testing_hyb2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t>    hyb2csr_tuple;
typedef std::tuple<hipsparseIndexBase_t, std::string> hyb2csr_bin_tuple;

int hyb2csr_M_range[] = {0, 10, 500, 872, 1000};
int hyb2csr_N_range[] = {0, 33, 242, 623, 1000};

hipsparseIndexBase_t hyb2csr_idx_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string hyb2csr_bin[] = {"rma10.bin",
                             "mac_econ_fwd500.bin",
                             "nos1.bin",
                             "nos2.bin",
                             "nos3.bin",
                             "nos4.bin",
                             "nos5.bin",
                             "nos6.bin",
                             "nos7.bin",
                             "sme3Dc.bin"};

class parameterized_hyb2csr : public testing::TestWithParam<hyb2csr_tuple>
{
protected:
    parameterized_hyb2csr() {}
    virtual ~parameterized_hyb2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_hyb2csr_bin : public testing::TestWithParam<hyb2csr_bin_tuple>
{
protected:
    parameterized_hyb2csr_bin() {}
    virtual ~parameterized_hyb2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_hyb2csr_arguments(hyb2csr_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_hyb2csr_arguments(hyb2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.baseA  = std::get<0>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<1>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
TEST(hyb2csr_bad_arg, hyb2csr)
{
    testing_hyb2csr_bad_arg<float>();
}

TEST_P(parameterized_hyb2csr, hyb2csr_float)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hyb2csr, hyb2csr_double)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hyb2csr, hyb2csr_float_complex)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hyb2csr, hyb2csr_double_complex)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hyb2csr_bin, hyb2csr_bin_float)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_hyb2csr_bin, hyb2csr_bin_double)
{
    Arguments arg = setup_hyb2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_hyb2csr<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(hyb2csr,
                         parameterized_hyb2csr,
                         testing::Combine(testing::ValuesIn(hyb2csr_M_range),
                                          testing::ValuesIn(hyb2csr_N_range),
                                          testing::ValuesIn(hyb2csr_idx_base_range)));

INSTANTIATE_TEST_SUITE_P(hyb2csr_bin,
                         parameterized_hyb2csr_bin,
                         testing::Combine(testing::ValuesIn(hyb2csr_idx_base_range),
                                          testing::ValuesIn(hyb2csr_bin)));
#endif
