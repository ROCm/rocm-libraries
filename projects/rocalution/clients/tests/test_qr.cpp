/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_qr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, unsigned int, std::string, int> qr_tuple;

std::vector<int>          qr_size             = {7, 16, 21};
std::vector<unsigned int> qr_format           = {1, 2, 3, 4, 5, 6, 7};
std::vector<std::string>  qr_matrix_type      = {"Laplacian2D", "PermutedIdentity"};
std::vector<int>          qr_use_host_and_acc = {0};

// Function to update tests if environment variable is set
void update_qr()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        qr_size.clear();
        qr_format.clear();
        qr_matrix_type.clear();
        qr_use_host_and_acc.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        qr_size.push_back(7);
        qr_format.insert(qr_format.end(), {1, 2, 3, 4, 5, 6, 7});
        qr_matrix_type.push_back("Laplacian2D");
        qr_use_host_and_acc.push_back(1);
    }
    else if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                                "ROCALUTION_EMULATION_REGRESSION",
                                "ROCALUTION_EMULATION_EXTENDED"}))
    {
        qr_use_host_and_acc.push_back(0);
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        qr_size.push_back(16);
        qr_format.push_back(2);
        qr_matrix_type.push_back("Laplacian2D");
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        qr_size.insert(qr_size.end(), {7, 16});
        qr_format.insert(qr_format.end(), {1, 3});
        qr_matrix_type.insert(qr_matrix_type.end(), {"Laplacian2D", "PermutedIdentity"});
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        qr_size.insert(qr_size.end(), {16, 21});
        qr_format.insert(qr_format.end(), {4, 5, 6, 7});
        qr_matrix_type.insert(qr_matrix_type.end(), {"Laplacian2D", "PermutedIdentity"});
    }
}

struct QRInitializer
{
    QRInitializer()
    {
        update_qr();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
QRInitializer qr_initializer;

class parameterized_qr : public testing::TestWithParam<qr_tuple>
{
protected:
    parameterized_qr() {}
    virtual ~parameterized_qr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_qr_arguments(qr_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.format      = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    arg.use_acc     = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_qr, qr_float)
{
    Arguments arg = setup_qr_arguments(GetParam());
    ASSERT_EQ(testing_qr<float>(arg), true);
}

TEST_P(parameterized_qr, qr_double)
{
    Arguments arg = setup_qr_arguments(GetParam());
    ASSERT_EQ(testing_qr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(qr,
                        parameterized_qr,
                        testing::Combine(testing::ValuesIn(qr_size),
                                         testing::ValuesIn(qr_format),
                                         testing::ValuesIn(qr_matrix_type),
                                         testing::ValuesIn(qr_use_host_and_acc)));
