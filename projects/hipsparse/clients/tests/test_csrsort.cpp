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

#include "testing_csrsort.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int, hipsparseIndexBase_t>    csrsort_tuple;
typedef std::tuple<int, hipsparseIndexBase_t, std::string> csrsort_bin_tuple;

#if defined(__HIP_PLATFORM_AMD__)
int csrsort_M_range[] = {0, 10, 500, 872, 1000};
int csrsort_N_range[] = {0, 33, 242, 623, 1000};
int csrsort_perm[]    = {0, 1};
#elif defined(__HIP_PLATFORM_NVIDIA__)
// cusparse does not allow without permutation
int csrsort_M_range[] = {10, 500, 872, 1000};
int csrsort_N_range[] = {33, 242, 623, 1000};
int csrsort_perm[]    = {1};
#endif

hipsparseIndexBase_t csrsort_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrsort_bin[] = {"rma10.bin",
                             "bibd_22_8.bin",
                             "scircuit.bin",
                             "bmwcra_1.bin",
                             "nos1.bin",
                             "nos3.bin",
                             "nos5.bin",
                             "nos7.bin",
                             "Chebyshev4.bin",
                             "webbase-1M.bin"};

class parameterized_csrsort : public testing::TestWithParam<csrsort_tuple>
{
protected:
    parameterized_csrsort() {}
    virtual ~parameterized_csrsort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrsort_bin : public testing::TestWithParam<csrsort_bin_tuple>
{
protected:
    parameterized_csrsort_bin() {}
    virtual ~parameterized_csrsort_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrsort_arguments(csrsort_tuple tup)
{
    Arguments arg;
    arg.M       = std::get<0>(tup);
    arg.N       = std::get<1>(tup);
    arg.permute = std::get<2>(tup);
    arg.baseA   = std::get<3>(tup);
    arg.timing  = 0;
    return arg;
}

Arguments setup_csrsort_arguments(csrsort_bin_tuple tup)
{
    Arguments arg;
    arg.M       = -99;
    arg.N       = -99;
    arg.permute = std::get<0>(tup);
    arg.baseA   = std::get<1>(tup);
    arg.timing  = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<2>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(csrsort_bad_arg, csrsort)
{
    testing_csrsort_bad_arg();
}

TEST_P(parameterized_csrsort, csrsort)
{
    Arguments arg = setup_csrsort_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrsort_bin, csrsort_bin)
{
    Arguments arg = setup_csrsort_arguments(GetParam());

    hipsparseStatus_t status = testing_csrsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrsort,
                         parameterized_csrsort,
                         testing::Combine(testing::ValuesIn(csrsort_M_range),
                                          testing::ValuesIn(csrsort_N_range),
                                          testing::ValuesIn(csrsort_perm),
                                          testing::ValuesIn(csrsort_base)));

INSTANTIATE_TEST_SUITE_P(csrsort_bin,
                         parameterized_csrsort_bin,
                         testing::Combine(testing::ValuesIn(csrsort_perm),
                                          testing::ValuesIn(csrsort_base),
                                          testing::ValuesIn(csrsort_bin)));
