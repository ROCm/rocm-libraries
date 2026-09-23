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

#include "testing_fgmres.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, std::string, unsigned int, int> fgmres_tuple;

std::vector<int>          fgmres_size    = {7, 63};
std::vector<int>          fgmres_basis   = {20, 60};
std::vector<std::string>  fgmres_precond = {"None", "SPAI", "TNS", "Jacobi", "GS", "ILUT", "MCGS"};
std::vector<unsigned int> fgmres_format  = {1, 4, 5, 7};
std::vector<int>          fgmres_use_acc = {1};

// Function to update tests if environment variable is set
void update_fgmres()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        fgmres_size.clear();
        fgmres_basis.clear();
        fgmres_precond.clear();
        fgmres_format.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        fgmres_size.push_back(7);
        fgmres_basis.push_back(20);
        fgmres_precond.insert(fgmres_precond.end(),
                              {"None", "SPAI", "TNS", "Jacobi", "GS", "ILUT", "MCGS"});
        fgmres_format.insert(fgmres_format.end(), {1, 4, 5, 7});
        fgmres_use_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        fgmres_size.push_back(7);
        fgmres_basis.push_back(20);
        fgmres_precond.insert(fgmres_precond.end(), {"None", "TNS"});
        fgmres_format.push_back(5);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        fgmres_size.push_back(63);
        fgmres_basis.push_back(60);
        fgmres_precond.insert(fgmres_precond.end(), {"SPAI", "Jacobi"});
        fgmres_format.push_back(1);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        fgmres_size.push_back(63);
        fgmres_basis.push_back(60);
        fgmres_precond.insert(fgmres_precond.end(), {"GS", "ILUT"});
        fgmres_format.insert(fgmres_format.end(), {4, 7});
    }
}

struct FGMRESInitializer
{
    FGMRESInitializer()
    {
        update_fgmres();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
FGMRESInitializer fgmres_initializer;

class parameterized_fgmres : public testing::TestWithParam<fgmres_tuple>
{
protected:
    parameterized_fgmres() {}
    virtual ~parameterized_fgmres() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fgmres_arguments(fgmres_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.index   = std::get<1>(tup);
    arg.precond = std::get<2>(tup);
    arg.format  = std::get<3>(tup);
    arg.use_acc = std::get<4>(tup);
    return arg;
}

TEST_P(parameterized_fgmres, fgmres_float)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<float>(arg), true);
}

TEST_P(parameterized_fgmres, fgmres_double)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fgmres,
                        parameterized_fgmres,
                        testing::Combine(testing::ValuesIn(fgmres_size),
                                         testing::ValuesIn(fgmres_basis),
                                         testing::ValuesIn(fgmres_precond),
                                         testing::ValuesIn(fgmres_format),
                                         testing::ValuesIn(fgmres_use_acc)));
