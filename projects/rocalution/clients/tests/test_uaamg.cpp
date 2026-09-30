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

#include "testing_uaamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::
    tuple<int, int, int, std::string, std::string, std::string, unsigned int, int, int, int, int>
        uaamg_tuple;

std::vector<int>          uaamg_size             = {22, 63, 134, 157};
std::vector<int>          uaamg_pre_iter         = {2};
std::vector<int>          uaamg_post_iter        = {2};
std::vector<std::string>  uaamg_smoother         = {"FSAI" /*, "ILU"*/};
std::vector<std::string>  uaamg_coarsening_strat = {"Greedy", "PMIS"};
std::vector<std::string>  uaamg_matrix_type      = {"Laplacian2D", "Laplacian3D"};
std::vector<unsigned int> uaamg_format           = {1, 6};
std::vector<int>          uaamg_cycle            = {2};
std::vector<int>          uaamg_scaling          = {1};
std::vector<int>          uaamg_rebuildnumeric   = {0, 1};
std::vector<int>          uaamg_use_acc          = {1};

// Function to update tests if environment variable is set
void update_uaamg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        uaamg_size.clear();
        uaamg_smoother.clear();
        uaamg_format.clear();
        uaamg_pre_iter.clear();
        uaamg_post_iter.clear();
        uaamg_cycle.clear();
        uaamg_scaling.clear();
        uaamg_rebuildnumeric.clear();
        uaamg_coarsening_strat.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        uaamg_size.push_back(22);
        uaamg_smoother.push_back("FSAI");
        uaamg_format.insert(uaamg_format.end(), {1, 6});
        uaamg_pre_iter.push_back(2);
        uaamg_post_iter.push_back(2);
        uaamg_cycle.push_back(2);
        uaamg_scaling.push_back(1);
        uaamg_rebuildnumeric.insert(uaamg_rebuildnumeric.end(), {0, 1});
        uaamg_coarsening_strat.insert(uaamg_coarsening_strat.end(), {"Greedy", "PMIS"});
        uaamg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        uaamg_size.push_back(63);
        uaamg_smoother.push_back("FSAI");
        uaamg_format.push_back(1);
        uaamg_pre_iter.push_back(2);
        uaamg_post_iter.push_back(2);
        uaamg_cycle.push_back(0);
        uaamg_scaling.push_back(0);
        uaamg_rebuildnumeric.push_back(0);
        uaamg_coarsening_strat.insert(uaamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        uaamg_size.push_back(134);
        uaamg_smoother.push_back("FSAI");
        uaamg_format.push_back(6);
        uaamg_pre_iter.push_back(2);
        uaamg_post_iter.push_back(2);
        uaamg_cycle.push_back(2);
        uaamg_scaling.push_back(1);
        uaamg_rebuildnumeric.push_back(1);
        uaamg_coarsening_strat.insert(uaamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        uaamg_size.push_back(157);
        uaamg_smoother.push_back("FSAI");
        uaamg_format.push_back(6);
        uaamg_pre_iter.push_back(2);
        uaamg_post_iter.push_back(2);
        uaamg_cycle.push_back(2);
        uaamg_scaling.push_back(1);
        uaamg_rebuildnumeric.push_back(0);
        uaamg_coarsening_strat.insert(uaamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
}

struct UAAMGInitializer
{
    UAAMGInitializer()
    {
        update_uaamg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
UAAMGInitializer uaamg_initializer;

class parameterized_uaamg : public testing::TestWithParam<uaamg_tuple>
{
protected:
    parameterized_uaamg() {}
    virtual ~parameterized_uaamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_uaamg_arguments(uaamg_tuple tup)
{
    Arguments arg;
    arg.size                = std::get<0>(tup);
    arg.pre_smooth          = std::get<1>(tup);
    arg.post_smooth         = std::get<2>(tup);
    arg.smoother            = std::get<3>(tup);
    arg.coarsening_strategy = std::get<4>(tup);
    arg.matrix_type         = std::get<5>(tup);
    arg.format              = std::get<6>(tup);
    arg.cycle               = std::get<7>(tup);
    arg.ordering            = std::get<8>(tup);
    arg.rebuildnumeric      = std::get<9>(tup);
    arg.use_acc             = std::get<10>(tup);

    return arg;
}

TEST_P(parameterized_uaamg, uaamg_float)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<float>(arg), true);
}

TEST_P(parameterized_uaamg, uaamg_double)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(uaamg,
                        parameterized_uaamg,
                        testing::Combine(testing::ValuesIn(uaamg_size),
                                         testing::ValuesIn(uaamg_pre_iter),
                                         testing::ValuesIn(uaamg_post_iter),
                                         testing::ValuesIn(uaamg_smoother),
                                         testing::ValuesIn(uaamg_coarsening_strat),
                                         testing::ValuesIn(uaamg_matrix_type),
                                         testing::ValuesIn(uaamg_format),
                                         testing::ValuesIn(uaamg_cycle),
                                         testing::ValuesIn(uaamg_scaling),
                                         testing::ValuesIn(uaamg_rebuildnumeric),
                                         testing::ValuesIn(uaamg_use_acc)));
