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

#include "testing_prune_dense2csr_by_percentage.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <vector>

typedef hipsparseIndexBase_t                    base;
typedef std::tuple<int, int, int, double, base> prune_dense2csr_by_percentage_tuple;
int    prune_dense2csr_by_percentage_M_range[]  = {0, 10, 500, 872, 1000};
int    prune_dense2csr_by_percentage_N_range[]  = {0, 33, 242, 623, 1000};
int    prune_dense2csr_by_percentage_LD_range[] = {1000};
double prune_dense2csr_by_percentage_range[]    = {0.1, 55.0, 67.0};
base   prune_dense2csr_by_percentage_idx_base_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

class parameterized_prune_dense2csr_by_percentage
    : public testing::TestWithParam<prune_dense2csr_by_percentage_tuple>
{
protected:
    parameterized_prune_dense2csr_by_percentage() {}
    virtual ~parameterized_prune_dense2csr_by_percentage() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_prune_dense2csr_by_percentage_arguments(prune_dense2csr_by_percentage_tuple tup)
{
    Arguments arg;
    arg.M          = std::get<0>(tup);
    arg.N          = std::get<1>(tup);
    arg.lda        = std::get<2>(tup);
    arg.percentage = std::get<3>(tup);
    arg.baseA      = std::get<4>(tup);
    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
TEST(prune_dense2csr_by_percentage_bad_arg, prune_dense2csr_by_percentage)
{
    testing_prune_dense2csr_by_percentage_bad_arg<float>();
}

TEST_P(parameterized_prune_dense2csr_by_percentage, prune_dense2csr_by_percentage_float)
{
    Arguments arg = setup_prune_dense2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_dense2csr_by_percentage<float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_prune_dense2csr_by_percentage, prune_dense2csr_by_percentage_double)
{
    Arguments arg = setup_prune_dense2csr_by_percentage_arguments(GetParam());

    hipsparseStatus_t status = testing_prune_dense2csr_by_percentage<double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    prune_dense2csr_by_percentage,
    parameterized_prune_dense2csr_by_percentage,
    testing::Combine(testing::ValuesIn(prune_dense2csr_by_percentage_M_range),
                     testing::ValuesIn(prune_dense2csr_by_percentage_N_range),
                     testing::ValuesIn(prune_dense2csr_by_percentage_LD_range),
                     testing::ValuesIn(prune_dense2csr_by_percentage_range),
                     testing::ValuesIn(prune_dense2csr_by_percentage_idx_base_range)));
#endif
