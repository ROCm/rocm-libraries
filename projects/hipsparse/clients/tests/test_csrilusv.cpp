/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_csrilusv.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t base;

typedef std::tuple<base, std::string> csrilusv_bin_tuple;

base csrilusv_idxbase_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

std::string csrilusv_bin[] = {"scircuit.bin",
#if defined(__HIP_PLATFORM_AMD__)
                              //                              "bmwcra_1.bin",
                              "nos1.bin",
#endif
                              "nos6.bin",
                              "amazon0312.bin"};

class parameterized_csrilusv_bin : public testing::TestWithParam<csrilusv_bin_tuple>
{
protected:
    parameterized_csrilusv_bin() {}
    virtual ~parameterized_csrilusv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilusv_arguments(csrilusv_bin_tuple tup)
{
    Arguments arg;
    arg.M      = -99;
    arg.baseA  = std::get<0>(tup);
    arg.timing = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<1>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST_P(parameterized_csrilusv_bin, csrilusv_bin_float)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilusv<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_csrilusv_bin, csrilusv_bin_double)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    hipsparseStatus_t status = testing_csrilusv<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(csrilusv_bin,
                         parameterized_csrilusv_bin,
                         testing::Combine(testing::ValuesIn(csrilusv_idxbase_range),
                                          testing::ValuesIn(csrilusv_bin)));
#endif
