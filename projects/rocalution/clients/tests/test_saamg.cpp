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

#include "testing_saamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::
    tuple<int, int, int, std::string, std::string, std::string, unsigned int, int, int, int, int>
        saamg_tuple;

std::vector<int>          saamg_size             = {22, 63, 134, 207};
std::vector<int>          saamg_pre_iter         = {2};
std::vector<int>          saamg_post_iter        = {2};
std::vector<std::string>  saamg_smoother         = {"FSAI", "SPAI"};
std::vector<std::string>  saamg_coarsening_strat = {"Greedy", "PMIS"};
std::vector<std::string>  saamg_matrix_type      = {"Laplacian2D"};
std::vector<unsigned int> saamg_format           = {1, 6};
std::vector<int>          saamg_cycle            = {2};
std::vector<int>          saamg_scaling          = {1};
std::vector<int>          saamg_rebuildnumeric   = {0, 1};
std::vector<int>          saamg_use_acc          = {1};

// Function to update tests if environment variable is set
void update_saamg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        saamg_size.clear();
        saamg_smoother.clear();
        saamg_format.clear();
        saamg_pre_iter.clear();
        saamg_post_iter.clear();
        saamg_cycle.clear();
        saamg_scaling.clear();
        saamg_rebuildnumeric.clear();
        saamg_coarsening_strat.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        saamg_size.push_back(22);
        saamg_smoother.insert(saamg_smoother.end(), {"FSAI", "SPAI"});
        saamg_format.insert(saamg_format.end(), {1, 6});
        saamg_pre_iter.push_back(2);
        saamg_post_iter.push_back(2);
        saamg_cycle.push_back(2);
        saamg_scaling.push_back(1);
        saamg_rebuildnumeric.insert(saamg_rebuildnumeric.end(), {0, 1});
        saamg_coarsening_strat.insert(saamg_coarsening_strat.end(), {"Greedy", "PMIS"});
        saamg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        saamg_size.push_back(63);
        saamg_smoother.push_back("FSAI");
        saamg_format.push_back(6);
        saamg_pre_iter.push_back(2);
        saamg_post_iter.push_back(2);
        saamg_cycle.push_back(2);
        saamg_scaling.push_back(1);
        saamg_rebuildnumeric.push_back(0);
        saamg_coarsening_strat.insert(saamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        saamg_size.push_back(63);
        saamg_smoother.push_back("SPAI");
        saamg_format.push_back(1);
        saamg_pre_iter.push_back(1);
        saamg_post_iter.push_back(1);
        saamg_cycle.push_back(2);
        saamg_scaling.push_back(1);
        saamg_rebuildnumeric.push_back(1);
        saamg_coarsening_strat.insert(saamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        saamg_size.push_back(134);
        saamg_smoother.push_back("FSAI");
        saamg_format.push_back(1);
        saamg_pre_iter.push_back(1);
        saamg_post_iter.push_back(2);
        saamg_cycle.push_back(0);
        saamg_scaling.push_back(0);
        saamg_rebuildnumeric.push_back(0);
        saamg_coarsening_strat.insert(saamg_coarsening_strat.end(), {"Greedy", "PMIS"});
    }
}

struct SAAMGInitializer
{
    SAAMGInitializer()
    {
        update_saamg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
SAAMGInitializer saamg_initializer;

class parameterized_saamg : public testing::TestWithParam<saamg_tuple>
{
protected:
    parameterized_saamg() {}
    virtual ~parameterized_saamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_saamg_arguments(saamg_tuple tup)
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

TEST_P(parameterized_saamg, saamg_float)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<float>(arg), true);
}

TEST_P(parameterized_saamg, saamg_double)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(saamg,
                        parameterized_saamg,
                        testing::Combine(testing::ValuesIn(saamg_size),
                                         testing::ValuesIn(saamg_pre_iter),
                                         testing::ValuesIn(saamg_post_iter),
                                         testing::ValuesIn(saamg_smoother),
                                         testing::ValuesIn(saamg_coarsening_strat),
                                         testing::ValuesIn(saamg_matrix_type),
                                         testing::ValuesIn(saamg_format),
                                         testing::ValuesIn(saamg_cycle),
                                         testing::ValuesIn(saamg_scaling),
                                         testing::ValuesIn(saamg_rebuildnumeric),
                                         testing::ValuesIn(saamg_use_acc)));
