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

#include "testing_kappa_cycle.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<std::string, int, int> kappa_cycle_tuple;

std::vector<std::string> kappa_cycle_amg     = {"UAAMG", "SAAMG", "RSAMG"};
std::vector<int>         kappa_cycle_size    = {100};
std::vector<int>         kappa_cycle_use_acc = {1};

// Function to update tests if environment variable is set
void update_kappa_cycle()
{
    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        kappa_cycle_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        kappa_cycle_amg = {"UAAMG"};
    }
}

struct KappaCycleInitializer
{
    KappaCycleInitializer()
    {
        update_kappa_cycle();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
KappaCycleInitializer kappa_cycle_initializer;

class parameterized_kappa_cycle : public testing::TestWithParam<kappa_cycle_tuple>
{
protected:
    parameterized_kappa_cycle() {}
    virtual ~parameterized_kappa_cycle() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_kappa_cycle_arguments(kappa_cycle_tuple tup)
{
    Arguments arg;
    arg.precond = std::get<0>(tup);
    arg.size    = std::get<1>(tup);
    arg.use_acc = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_kappa_cycle, kappa_cycle_float)
{
    Arguments arg = setup_kappa_cycle_arguments(GetParam());
    testing_kappa_cycle<float>(arg);
}

TEST_P(parameterized_kappa_cycle, kappa_cycle_double)
{
    Arguments arg = setup_kappa_cycle_arguments(GetParam());
    testing_kappa_cycle<double>(arg);
}

INSTANTIATE_TEST_CASE_P(kappa_cycle,
                        parameterized_kappa_cycle,
                        testing::Combine(testing::ValuesIn(kappa_cycle_amg),
                                         testing::ValuesIn(kappa_cycle_size),
                                         testing::ValuesIn(kappa_cycle_use_acc)));
