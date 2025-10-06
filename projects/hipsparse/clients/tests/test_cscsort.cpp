/* ************************************************************************
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_cscsort.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, int, hipsparseIndexBase_t>    cscsort_tuple;
typedef std::tuple<int, hipsparseIndexBase_t, std::string> cscsort_bin_tuple;

int                  cscsort_M_range[] = {0, 10, 500, 872, 1000};
int                  cscsort_N_range[] = {0, 33, 242, 623, 1000};
int                  cscsort_perm[]    = {0, 1};
hipsparseIndexBase_t cscsort_base[]    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string cscsort_bin[] = {"mac_econ_fwd500.bin",
                             "mc2depi.bin",
                             "ASIC_320k.bin",
                             "nos2.bin",
                             "nos4.bin",
                             "nos6.bin",
                             "amazon0312.bin",
                             "sme3Dc.bin",
                             "shipsec1.bin"};

class parameterized_cscsort : public testing::TestWithParam<cscsort_tuple>
{
protected:
    parameterized_cscsort() {}
    virtual ~parameterized_cscsort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_cscsort_bin : public testing::TestWithParam<cscsort_bin_tuple>
{
protected:
    parameterized_cscsort_bin() {}
    virtual ~parameterized_cscsort_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cscsort_arguments(cscsort_tuple tup)
{
    Arguments arg;
    arg.M       = std::get<0>(tup);
    arg.N       = std::get<1>(tup);
    arg.permute = std::get<2>(tup);
    arg.baseA   = std::get<3>(tup);
    arg.timing  = 0;
    return arg;
}

Arguments setup_cscsort_arguments(cscsort_bin_tuple tup)
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

TEST(cscsort_bad_arg, cscsort)
{
    testing_cscsort_bad_arg();
}

TEST_P(parameterized_cscsort, cscsort)
{
    Arguments arg = setup_cscsort_arguments(GetParam());

    hipsparseStatus_t status = testing_cscsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_cscsort_bin, cscsort_bin)
{
    Arguments arg = setup_cscsort_arguments(GetParam());

    hipsparseStatus_t status = testing_cscsort(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(cscsort,
                         parameterized_cscsort,
                         testing::Combine(testing::ValuesIn(cscsort_M_range),
                                          testing::ValuesIn(cscsort_N_range),
                                          testing::ValuesIn(cscsort_perm),
                                          testing::ValuesIn(cscsort_base)));

INSTANTIATE_TEST_SUITE_P(cscsort_bin,
                         parameterized_cscsort_bin,
                         testing::Combine(testing::ValuesIn(cscsort_perm),
                                          testing::ValuesIn(cscsort_base),
                                          testing::ValuesIn(cscsort_bin)));
