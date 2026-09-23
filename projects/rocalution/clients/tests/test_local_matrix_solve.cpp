/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_local_matrix_solve.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, unsigned int>       local_matrix_gen_solve_tuple;
typedef std::tuple<int, unsigned int, bool> local_matrix_tri_solve_tuple;

static const int          local_matrix_solve_size[]      = {10, 17, 21};
static const unsigned int local_matrix_solve_format[]    = {1};
static const bool         local_matrix_solve_unit_diag[] = {false, true};

class parameterized_local_matrix_lusolve
    : public testing::TestWithParam<local_matrix_gen_solve_tuple>
{
protected:
    parameterized_local_matrix_lusolve() {}
    virtual ~parameterized_local_matrix_lusolve() {}
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

class parameterized_local_matrix_llsolve
    : public testing::TestWithParam<local_matrix_gen_solve_tuple>
{
protected:
    parameterized_local_matrix_llsolve() {}
    virtual ~parameterized_local_matrix_llsolve() {}
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

class parameterized_local_matrix_lsolve
    : public testing::TestWithParam<local_matrix_tri_solve_tuple>
{
protected:
    parameterized_local_matrix_lsolve() {}
    virtual ~parameterized_local_matrix_lsolve() {}
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

class parameterized_local_matrix_usolve
    : public testing::TestWithParam<local_matrix_tri_solve_tuple>
{
protected:
    parameterized_local_matrix_usolve() {}
    virtual ~parameterized_local_matrix_usolve() {}
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

static Arguments setup_local_matrix_solve_arguments(local_matrix_gen_solve_tuple tup)
{
    Arguments arg;
    arg.size   = std::get<0>(tup);
    arg.format = std::get<1>(tup);
    return arg;
}

static Arguments setup_local_matrix_solve_arguments(local_matrix_tri_solve_tuple tup)
{
    Arguments arg;
    arg.size      = std::get<0>(tup);
    arg.format    = std::get<1>(tup);
    arg.unit_diag = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_local_matrix_lusolve, local_matrix_lusolve_float)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_lusolve<float>(arg), true);
}

TEST_P(parameterized_local_matrix_lusolve, local_matrix_lusolve_double)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_lusolve<double>(arg), true);
}

TEST_P(parameterized_local_matrix_llsolve, local_matrix_llsolve_float)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_llsolve<float>(arg), true);
}

TEST_P(parameterized_local_matrix_llsolve, local_matrix_llsolve_double)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_llsolve<double>(arg), true);
}

TEST_P(parameterized_local_matrix_lsolve, local_matrix_lsolve_float)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_lsolve<float>(arg), true);
}

TEST_P(parameterized_local_matrix_lsolve, local_matrix_lsolve_double)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_lsolve<double>(arg), true);
}

TEST_P(parameterized_local_matrix_usolve, local_matrix_usolve_float)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_usolve<float>(arg), true);
}

TEST_P(parameterized_local_matrix_usolve, local_matrix_usolve_double)
{
    Arguments arg = setup_local_matrix_solve_arguments(GetParam());
    ASSERT_EQ(testing_local_matrix_usolve<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(local_matrix_solve,
                        parameterized_local_matrix_lusolve,
                        testing::Combine(testing::ValuesIn(local_matrix_solve_size),
                                         testing::ValuesIn(local_matrix_solve_format)));

INSTANTIATE_TEST_CASE_P(local_matrix_solve,
                        parameterized_local_matrix_llsolve,
                        testing::Combine(testing::ValuesIn(local_matrix_solve_size),
                                         testing::ValuesIn(local_matrix_solve_format)));

INSTANTIATE_TEST_CASE_P(local_matrix_solve,
                        parameterized_local_matrix_lsolve,
                        testing::Combine(testing::ValuesIn(local_matrix_solve_size),
                                         testing::ValuesIn(local_matrix_solve_format),
                                         testing::ValuesIn(local_matrix_solve_unit_diag)));

INSTANTIATE_TEST_CASE_P(local_matrix_solve,
                        parameterized_local_matrix_usolve,
                        testing::Combine(testing::ValuesIn(local_matrix_solve_size),
                                         testing::ValuesIn(local_matrix_solve_format),
                                         testing::ValuesIn(local_matrix_solve_unit_diag)));
