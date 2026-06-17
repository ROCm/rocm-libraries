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

#include "testing_inversion.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, int> inversion_tuple;

std::vector<int>          inversion_size             = {7, 16, 21};
std::vector<unsigned int> inversion_format           = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  inversion_matrix_type      = {"Laplacian2D", "PermutedIdentity"};
std::vector<int>          inversion_use_host_and_acc = {0};

// Function to update tests if environment variable is set
void update_inversion()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        inversion_size.clear();
        inversion_format.clear();
        inversion_matrix_type.clear();
        inversion_use_host_and_acc.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        inversion_size.push_back(7);
        inversion_format.insert(inversion_format.end(), {1, 2, 3, 4, 5, 6, 7});
        inversion_matrix_type.push_back("Laplacian2D");
        inversion_use_host_and_acc.push_back(1);
    }
    else if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                                "ROCALUTION_EMULATION_REGRESSION",
                                "ROCALUTION_EMULATION_EXTENDED"}))
    {
        inversion_use_host_and_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        inversion_size.push_back(16);
        inversion_format.push_back(1);
        inversion_matrix_type.push_back("Laplacian2D");
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        inversion_size.insert(inversion_size.end(), {7, 16});
        inversion_format.insert(inversion_format.end(), {2, 3});
        inversion_matrix_type.insert(inversion_matrix_type.end(),
                                     {"Laplacian2D", "PermutedIdentity"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        inversion_size.insert(inversion_size.end(), {21});
        inversion_format.insert(inversion_format.end(), {4, 5, 6, 7});
        inversion_matrix_type.insert(inversion_matrix_type.end(),
                                     {"Laplacian2D", "PermutedIdentity"});
    }
}

struct InversionInitializer
{
    InversionInitializer()
    {
        update_inversion();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
InversionInitializer inversion_initializer;

class parameterized_inversion : public testing::TestWithParam<inversion_tuple>
{
protected:
    parameterized_inversion() {}
    virtual ~parameterized_inversion() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_inversion_arguments(inversion_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.format      = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    arg.use_acc     = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_inversion, inversion_float)
{
    Arguments arg = setup_inversion_arguments(GetParam());
    ASSERT_EQ(testing_inversion<float>(arg), true);
}

TEST_P(parameterized_inversion, inversion_double)
{
    Arguments arg = setup_inversion_arguments(GetParam());
    ASSERT_EQ(testing_inversion<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(inversion,
                        parameterized_inversion,
                        testing::Combine(testing::ValuesIn(inversion_size),
                                         testing::ValuesIn(inversion_format),
                                         testing::ValuesIn(inversion_matrix_type),
                                         testing::ValuesIn(inversion_use_host_and_acc)));
