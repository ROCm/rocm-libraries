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

#include "testing_gather.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, hipsparseIndexBase_t> gather_tuple;

int gather_N_range[]   = {15332};
int gather_nnz_range[] = {500};

hipsparseIndexBase_t gather_base[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_gather : public testing::TestWithParam<gather_tuple>
{
protected:
    parameterized_gather() {}
    virtual ~parameterized_gather() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gather_arguments(gather_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.baseA  = std::get<2>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
TEST(gather_bad_arg, gather)
{
    testing_gather_bad_arg();
}

TEST_P(parameterized_gather, gather_float)
{
    Arguments arg = setup_gather_arguments(GetParam());

    hipsparseStatus_t status = testing_gather<int, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(gather,
                         parameterized_gather,
                         testing::Combine(testing::ValuesIn(gather_N_range),
                                          testing::ValuesIn(gather_nnz_range),
                                          testing::ValuesIn(gather_base)));
#endif
