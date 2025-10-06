/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csru2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t>    csru2csr_tuple;
typedef std::tuple<hipsparseIndexBase_t, std::string> csru2csr_bin_tuple;

int csru2csr_M_range[] = {51314};
int csru2csr_N_range[] = {12963};

hipsparseIndexBase_t csru2csr_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csru2csr_bin[] = {"nos3.bin"};

class parameterized_csru2csr : public testing::TestWithParam<csru2csr_tuple>
{
protected:
    parameterized_csru2csr() {}
    virtual ~parameterized_csru2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csru2csr_bin : public testing::TestWithParam<csru2csr_bin_tuple>
{
protected:
    parameterized_csru2csr_bin() {}
    virtual ~parameterized_csru2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csru2csr_arguments(csru2csr_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_csru2csr_arguments(csru2csr_bin_tuple tup)
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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(csru2csr_bad_arg, csru2csr)
{
    testing_csru2csr_bad_arg();
}

TEST_P(parameterized_csru2csr, csru2csr_float)
{
    Arguments arg = setup_csru2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_csru2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csru2csr_bin, csru2csr_bin_float)
{
    Arguments arg = setup_csru2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_csru2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csru2csr,
                         parameterized_csru2csr,
                         testing::Combine(testing::ValuesIn(csru2csr_M_range),
                                          testing::ValuesIn(csru2csr_N_range),
                                          testing::ValuesIn(csru2csr_base)));

INSTANTIATE_TEST_SUITE_P(csru2csr_bin,
                         parameterized_csru2csr_bin,
                         testing::Combine(testing::ValuesIn(csru2csr_base),
                                          testing::ValuesIn(csru2csr_bin)));
#endif
