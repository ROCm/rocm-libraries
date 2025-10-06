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

#include "testing_csr2csr_compress.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, double, hipsparseIndexBase_t>    csr2csr_compress_tuple;
typedef std::tuple<double, hipsparseIndexBase_t, std::string> csr2csr_compress_bin_tuple;

int    csr2csr_compress_M_range[]     = {10, 500, 872, 9375, 30327};
int    csr2csr_compress_N_range[]     = {33, 242, 623, 9184, 30645};
double csr2csr_compress_alpha_range[] = {0.0, 0.08736, 0.33333, 1.7};

hipsparseIndexBase_t csr2csr_compress_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csr2csr_compress_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_csr2csr_compress : public testing::TestWithParam<csr2csr_compress_tuple>
{
protected:
    parameterized_csr2csr_compress() {}
    virtual ~parameterized_csr2csr_compress() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2csr_compress_bin : public testing::TestWithParam<csr2csr_compress_bin_tuple>
{
protected:
    parameterized_csr2csr_compress_bin() {}
    virtual ~parameterized_csr2csr_compress_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2csr_compress_arguments(csr2csr_compress_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.baseA  = std::get<3>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csr2csr_compress_arguments(csr2csr_compress_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.N      = -99;
    arg.alpha  = std::get<0>(tup);
    arg.baseA  = std::get<1>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<2>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(csr2csr_compress_bad_arg, csr2csr_compress)
{
    testing_csr2csr_compress_bad_arg<float>();
}

TEST_P(parameterized_csr2csr_compress, csr2csr_compress_float)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csr_compress, csr2csr_compress_double)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csr_compress, csr2csr_compress_float_complex)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csr_compress, csr2csr_compress_double_complex)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csr_compress_bin, csr2csr_compress_bin_float)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csr2csr_compress_bin, csr2csr_compress_bin_double)
{
    Arguments arg = setup_csr2csr_compress_arguments(GetParam());

    hipsparseStatus_t status = testing_csr2csr_compress<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csr2csr_compress,
                         parameterized_csr2csr_compress,
                         testing::Combine(testing::ValuesIn(csr2csr_compress_M_range),
                                          testing::ValuesIn(csr2csr_compress_N_range),
                                          testing::ValuesIn(csr2csr_compress_alpha_range),
                                          testing::ValuesIn(csr2csr_compress_base_range)));

INSTANTIATE_TEST_SUITE_P(csr2csr_compress_bin,
                         parameterized_csr2csr_compress_bin,
                         testing::Combine(testing::ValuesIn(csr2csr_compress_alpha_range),
                                          testing::ValuesIn(csr2csr_compress_base_range),
                                          testing::ValuesIn(csr2csr_compress_bin)));
