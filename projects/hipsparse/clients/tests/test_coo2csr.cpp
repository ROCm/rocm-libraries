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

#include "testing_coo2csr.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t>    coo2csr_tuple;
typedef std::tuple<hipsparseIndexBase_t, std::string> coo2csr_bin_tuple;

int coo2csr_M_range[] = {0, 10, 500, 872, 1000};
int coo2csr_N_range[] = {0, 33, 242, 623, 1000};

hipsparseIndexBase_t coo2csr_idx_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string coo2csr_bin[] = {"rma10.bin",
                             "mac_econ_fwd500.bin",
                             "bibd_22_8.bin",
                             "mc2depi.bin",
                             "scircuit.bin",
                             "ASIC_320k.bin",
                             "bmwcra_1.bin",
                             "nos1.bin",
                             "nos2.bin",
                             "nos3.bin",
                             "nos4.bin",
                             "nos5.bin",
                             "nos6.bin",
                             "nos7.bin",
                             "amazon0312.bin",
                             "Chebyshev4.bin",
                             "sme3Dc.bin",
                             "webbase-1M.bin",
                             "shipsec1.bin"};

class parameterized_coo2csr : public testing::TestWithParam<coo2csr_tuple>
{
protected:
    parameterized_coo2csr() {}
    virtual ~parameterized_coo2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_coo2csr_bin : public testing::TestWithParam<coo2csr_bin_tuple>
{
protected:
    parameterized_coo2csr_bin() {}
    virtual ~parameterized_coo2csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coo2csr_arguments(coo2csr_tuple tup)
{
    Arguments arg;
    arg.M      = std::get<0>(tup);
    arg.N      = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

Arguments setup_coo2csr_arguments(coo2csr_bin_tuple tup)
{
    Arguments arg;
    arg.M                = -99;
    arg.N                = -99;
    arg.baseA            = std::get<0>(tup);
    arg.timing           = 0;
    std::string bin_file = std::get<1>(tup);

    arg.set_filename(get_filename(bin_file));

    return arg;
}

TEST(coo2csr_bad_arg, coo2csr)
{
    testing_coo2csr_bad_arg();
}

TEST_P(parameterized_coo2csr, coo2csr_float)
{
    Arguments arg = setup_coo2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_coo2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_coo2csr_bin, coo2csr_bin_float)
{
    Arguments arg = setup_coo2csr_arguments(GetParam());

    hipsparseStatus_t status = testing_coo2csr<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(coo2csr,
                         parameterized_coo2csr,
                         testing::Combine(testing::ValuesIn(coo2csr_M_range),
                                          testing::ValuesIn(coo2csr_N_range),
                                          testing::ValuesIn(coo2csr_idx_base_range)));

INSTANTIATE_TEST_SUITE_P(coo2csr_bin,
                         parameterized_coo2csr_bin,
                         testing::Combine(testing::ValuesIn(coo2csr_idx_base_range),
                                          testing::ValuesIn(coo2csr_bin)));
