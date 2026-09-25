/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_hybrid_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <utility>
#include <vector>

// Problem: (grid size, matrix type)
typedef std::pair<int, std::string> hybrid_amg_problem;

typedef std::tuple<hybrid_amg_problem, int, std::string, int, int, int> hybrid_amg_tuple;

// Problem sizes are chosen such that the unsmoothed aggregation AMG creates at
// least 3 levels without a level limit
std::vector<hybrid_amg_problem> hybrid_amg_problems
    = {hybrid_amg_problem(134, "Laplacian2D"), hybrid_amg_problem(40, "Laplacian3D")};
std::vector<int>         hybrid_amg_max_levels       = {2, 3};
std::vector<std::string> hybrid_amg_coarsening_strat = {"Greedy", "PMIS"};
std::vector<int>         hybrid_amg_cycle            = {0, 2};
std::vector<int>         hybrid_amg_rebuildnumeric   = {0, 1};
std::vector<int>         hybrid_amg_use_acc          = {1};

// Function to update tests if environment variable is set
void update_hybrid_amg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        hybrid_amg_max_levels.clear();
        hybrid_amg_coarsening_strat.clear();
        hybrid_amg_cycle.clear();
        hybrid_amg_rebuildnumeric.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        hybrid_amg_max_levels.push_back(2);
        hybrid_amg_coarsening_strat.push_back("PMIS");
        hybrid_amg_cycle.push_back(2);
        hybrid_amg_rebuildnumeric.insert(hybrid_amg_rebuildnumeric.end(), {0, 1});
        hybrid_amg_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        hybrid_amg_max_levels.push_back(2);
        hybrid_amg_coarsening_strat.push_back("PMIS");
        hybrid_amg_cycle.push_back(0);
        hybrid_amg_rebuildnumeric.push_back(0);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        hybrid_amg_max_levels.insert(hybrid_amg_max_levels.end(), {2, 3});
        hybrid_amg_coarsening_strat.push_back("PMIS");
        hybrid_amg_cycle.push_back(2);
        hybrid_amg_rebuildnumeric.push_back(1);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        hybrid_amg_max_levels.insert(hybrid_amg_max_levels.end(), {2, 3});
        hybrid_amg_coarsening_strat.insert(hybrid_amg_coarsening_strat.end(), {"Greedy", "PMIS"});
        hybrid_amg_cycle.push_back(2);
        hybrid_amg_rebuildnumeric.push_back(0);
    }
}

struct HybridAMGInitializer
{
    HybridAMGInitializer()
    {
        update_hybrid_amg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
HybridAMGInitializer hybrid_amg_initializer;

class parameterized_hybrid_amg : public testing::TestWithParam<hybrid_amg_tuple>
{
protected:
    parameterized_hybrid_amg() {}
    virtual ~parameterized_hybrid_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_hybrid_amg_arguments(hybrid_amg_tuple tup)
{
    Arguments arg;
    arg.size                = std::get<0>(tup).first;
    arg.matrix_type         = std::get<0>(tup).second;
    arg.max_levels          = std::get<1>(tup);
    arg.coarsening_strategy = std::get<2>(tup);
    arg.cycle               = std::get<3>(tup);
    arg.rebuildnumeric      = std::get<4>(tup);
    arg.use_acc             = std::get<5>(tup);

    return arg;
}

TEST_P(parameterized_hybrid_amg, hybrid_amg_float)
{
    Arguments arg = setup_hybrid_amg_arguments(GetParam());
    ASSERT_EQ(testing_hybrid_amg<float>(arg), true);
}

TEST_P(parameterized_hybrid_amg, hybrid_amg_double)
{
    Arguments arg = setup_hybrid_amg_arguments(GetParam());
    ASSERT_EQ(testing_hybrid_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(hybrid_amg,
                        parameterized_hybrid_amg,
                        testing::Combine(testing::ValuesIn(hybrid_amg_problems),
                                         testing::ValuesIn(hybrid_amg_max_levels),
                                         testing::ValuesIn(hybrid_amg_coarsening_strat),
                                         testing::ValuesIn(hybrid_amg_cycle),
                                         testing::ValuesIn(hybrid_amg_rebuildnumeric),
                                         testing::ValuesIn(hybrid_amg_use_acc)));
