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

#include "testing_pairwise_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int, int, int, int, int> pwamg_tuple;

std::vector<int>          pwamg_size           = {63, 134};
std::vector<std::string>  pwamg_smoother       = {"Jacobi"}; //, "MCILU"};
std::vector<unsigned int> pwamg_format         = {1, 7};
std::vector<int>          pwamg_pre_iter       = {1, 2};
std::vector<int>          pwamg_post_iter      = {1, 2};
std::vector<int>          pwamg_ordering       = {0, 1, 2, 3, 4, 5};
std::vector<int>          pwamg_rebuildnumeric = {0, 1};
std::vector<int>          pwamg_use_acc        = {1};

// Function to update tests if environment variable is set
void update_pwamg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        pwamg_size.clear();
        pwamg_smoother.clear();
        pwamg_format.clear();
        pwamg_pre_iter.clear();
        pwamg_post_iter.clear();
        pwamg_ordering.clear();
        pwamg_rebuildnumeric.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        pwamg_size.push_back(63);
        pwamg_smoother.push_back("Jacobi");
        pwamg_format.insert(pwamg_format.end(), {1, 7});
        pwamg_pre_iter.insert(pwamg_pre_iter.end(), {1, 2});
        pwamg_post_iter.insert(pwamg_post_iter.end(), {1, 2});
        pwamg_ordering.insert(pwamg_ordering.end(), {0, 1, 2, 3, 4, 5});
        pwamg_rebuildnumeric.insert(pwamg_rebuildnumeric.end(), {0, 1});
        pwamg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        pwamg_size.push_back(134);
        pwamg_smoother.push_back("Jacobi");
        pwamg_format.push_back(1);
        pwamg_pre_iter.push_back(1);
        pwamg_post_iter.push_back(1);
        pwamg_ordering.push_back(1);
        pwamg_rebuildnumeric.push_back(0);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        pwamg_size.push_back(63);
        pwamg_smoother.push_back("Jacobi");
        pwamg_format.push_back(7);
        pwamg_pre_iter.push_back(2);
        pwamg_post_iter.push_back(2);
        pwamg_ordering.insert(pwamg_ordering.end(), {1, 2, 3});
        pwamg_rebuildnumeric.push_back(0);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        pwamg_size.push_back(134);
        pwamg_smoother.push_back("Jacobi");
        pwamg_format.push_back(1);
        pwamg_pre_iter.push_back(1);
        pwamg_post_iter.push_back(2);
        pwamg_ordering.insert(pwamg_ordering.end(), {4, 5});
        pwamg_rebuildnumeric.insert(pwamg_rebuildnumeric.end(), {0, 1});
    }
}

struct PWAMGInitializer
{
    PWAMGInitializer()
    {
        update_pwamg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
PWAMGInitializer pwamg_initializer;

class parameterized_pairwise_amg : public testing::TestWithParam<pwamg_tuple>
{
protected:
    parameterized_pairwise_amg() {}
    virtual ~parameterized_pairwise_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_pwamg_arguments(pwamg_tuple tup)
{
    Arguments arg;
    arg.size           = std::get<0>(tup);
    arg.smoother       = std::get<1>(tup);
    arg.format         = std::get<2>(tup);
    arg.pre_smooth     = std::get<3>(tup);
    arg.post_smooth    = std::get<4>(tup);
    arg.ordering       = std::get<5>(tup);
    arg.rebuildnumeric = std::get<6>(tup);
    arg.use_acc        = std::get<7>(tup);
    return arg;
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_float)
{
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg<float>(arg), true);
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_double)
{
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg<double>(arg), true);
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_2_double)
{
    if(!is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        // Skip this test if not in code coverage mode
        GTEST_SKIP() << "Skipping pairwise_amg_2_double test unless in code coverage mode.";
    }
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg_2<double>(arg), true);
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_3_double)
{
    if(!is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        // Skip this test if not in code coverage mode
        GTEST_SKIP() << "Skipping pairwise_amg_3_double test unless in code coverage mode.";
    }
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg_3<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(pairwise_amg,
                        parameterized_pairwise_amg,
                        testing::Combine(testing::ValuesIn(pwamg_size),
                                         testing::ValuesIn(pwamg_smoother),
                                         testing::ValuesIn(pwamg_format),
                                         testing::ValuesIn(pwamg_pre_iter),
                                         testing::ValuesIn(pwamg_post_iter),
                                         testing::ValuesIn(pwamg_ordering),
                                         testing::ValuesIn(pwamg_rebuildnumeric),
                                         testing::ValuesIn(pwamg_use_acc)));
