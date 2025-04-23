/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_local_matrix.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, std::string> local_matrix_conversions_tuple;
typedef std::tuple<int, int>              local_matrix_allocations_tuple;
typedef std::tuple<int, int>              local_matrix_zero_tuple;

int         local_matrix_conversions_size[]     = {10, 17, 21};
int         local_matrix_conversions_blockdim[] = {4, 7, 11};
std::string local_matrix_type[]                 = {"Laplacian2D", "PermutedIdentity", "Random"};

int local_matrix_allocations_size[]     = {100, 1475, 2524};
int local_matrix_allocations_blockdim[] = {4, 7, 11};

class parameterized_local_matrix_conversions
    : public testing::TestWithParam<local_matrix_conversions_tuple>
{
protected:
    parameterized_local_matrix_conversions() {}
    virtual ~parameterized_local_matrix_conversions() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_local_matrix_conversions_arguments(local_matrix_conversions_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.blockdim    = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    return arg;
}

class parameterized_local_matrix_allocations
    : public testing::TestWithParam<local_matrix_allocations_tuple>
{
protected:
    parameterized_local_matrix_allocations() {}
    virtual ~parameterized_local_matrix_allocations() {}
    virtual void SetUp() override
    {
        if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                               "ROCALUTION_EMULATION_REGRESSION",
                               "ROCALUTION_EMULATION_EXTENDED"}))
        {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() {}
};

Arguments setup_local_matrix_allocations_arguments(local_matrix_allocations_tuple tup)
{
    Arguments arg;
    arg.size     = std::get<0>(tup);
    arg.blockdim = std::get<1>(tup);
    return arg;
}

TEST(local_matrix_bad_args, local_matrix)
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED"}))
    {
        GTEST_SKIP();
    }

    testing_local_matrix_bad_args<float>();
}

TEST_P(parameterized_local_matrix_conversions, local_matrix_conversions_float)
{
    Arguments arg = setup_local_matrix_conversions_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_conversions<float>(arg), true);
}

TEST_P(parameterized_local_matrix_conversions, local_matrix_conversions_double)
{
    Arguments arg = setup_local_matrix_conversions_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_conversions<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_conversions,
                        parameterized_local_matrix_conversions,
                        testing::Combine(testing::ValuesIn(local_matrix_conversions_size),
                                         testing::ValuesIn(local_matrix_conversions_blockdim),
                                         testing::ValuesIn(local_matrix_type)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_allocations_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_allocations<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_allocations_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_allocations<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_allocations,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_zero_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_zero<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_zero_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_zero<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_zero,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));

TEST_P(parameterized_local_matrix_allocations, local_matrix_set_data_ptr_float)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_set_data_ptr<float>(arg), true);
}

TEST_P(parameterized_local_matrix_allocations, local_matrix_set_data_ptr_double)
{
    Arguments arg = setup_local_matrix_allocations_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_set_data_ptr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_set_data_ptr,
                        parameterized_local_matrix_allocations,
                        testing::Combine(testing::ValuesIn(local_matrix_allocations_size),
                                         testing::ValuesIn(local_matrix_allocations_blockdim)));
