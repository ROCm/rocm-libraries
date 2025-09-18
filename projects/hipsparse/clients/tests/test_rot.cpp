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

#include "testing_rot.hpp"

#include <hipsparse.h>

typedef hipsparseIndexBase_t                       base;
typedef std::tuple<int, int, double, double, base> rot_tuple;

int rot_N_range[]   = {12000, 15332, 22031};
int rot_nnz_range[] = {0, 5, 10, 500, 1000, 7111, 10000};

double rot_c_range[] = {-2.0, 0.0, 1.0};
double rot_s_range[] = {-3.0, 0.0, 4.0};

base rot_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_rot : public testing::TestWithParam<rot_tuple>
{
protected:
    parameterized_rot() {}
    virtual ~parameterized_rot() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rot_arguments(rot_tuple tup)
{
    Arguments arg;
    arg.N      = std::get<0>(tup);
    arg.nnz    = std::get<1>(tup);
    arg.alpha  = std::get<2>(tup);
    arg.beta   = std::get<3>(tup);
    arg.baseA  = std::get<4>(tup);
    arg.timing = 0;
    return arg;
}

#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
TEST(rot_bad_arg, rot_float)
{
    testing_rot_bad_arg();
}

TEST_P(parameterized_rot, rot_i32_float)
{
    Arguments arg = setup_rot_arguments(GetParam());

    hipsparseStatus_t status = testing_rot<int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_rot, rot_i64_double)
{
    Arguments arg = setup_rot_arguments(GetParam());

    hipsparseStatus_t status = testing_rot<int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(rot,
                         parameterized_rot,
                         testing::Combine(testing::ValuesIn(rot_N_range),
                                          testing::ValuesIn(rot_nnz_range),
                                          testing::ValuesIn(rot_c_range),
                                          testing::ValuesIn(rot_s_range),
                                          testing::ValuesIn(rot_idx_base_range)));
#endif
