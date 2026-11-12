/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_rsamg_truncation.hpp"
#include "testing_ruge_stueben_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, std::string, unsigned int, int, int, int, int, int, int, std::string>
    rsamg_tuple;

// size, format, use_acc, interpolation type, truncation factor, max elements per row
typedef std::tuple<int, unsigned int, int, std::string, float, int> rsamg_interp_tuple;

// rows, use_acc, truncation factor, max elements per row
typedef std::tuple<int, int, float, int> rsamg_trunc_tuple;

// rows, truncation factor, max elements per row
typedef std::tuple<int, float, int> rsamg_trunc_parity_tuple;

std::vector<int>          rsamg_size                = {63, 134};
std::vector<std::string>  rsamg_smoother            = {"Jacobi"};
std::vector<unsigned int> rsamg_format              = {1, 7};
std::vector<int>          rsamg_pre_iter            = {1, 2};
std::vector<int>          rsamg_post_iter           = {1, 2};
std::vector<int>          rsamg_cycle               = {0, 1};
std::vector<int>          rsamg_scaling             = {0, 1};
std::vector<int>          rsamg_rebuildnumeric      = {0, 1};
std::vector<int>          rsamg_use_acc             = {1};
std::vector<std::string>  rsamg_coarsening_strategy = {"PMIS"};

// The interpolation operators are swept separately, so that adding one does not multiply
// the whole parameter matrix above. Truncation is a post-pass shared by all of them, and
// crossing its two parameters covers each of them on its own as well as both together.
// ExtPI_FF1 is the classical Ext+I operator with the FF1 limit enabled, which is the only
// operator the limit applies to, so it is one more entry here rather than a dimension of
// its own. These keep their own size and format so that the emulation modes below, which
// trim the matrix above, cannot leave them empty. Both backends are always swept, because
// the host implementations are OpenMP parallel and are not covered anywhere else.
std::vector<std::string> rsamg_interpolation_type
    = {"Direct", "ExtPI", "ExtPI_FF1", "MMExtPI", "MMExtPE"};
std::vector<int>          rsamg_interp_size      = {63};
std::vector<unsigned int> rsamg_interp_format    = {1};
std::vector<int>          rsamg_interp_use_acc   = {0, 1};
std::vector<float>        rsamg_interp_trunc     = {0.0f, 0.2f};
std::vector<int>          rsamg_interp_max_elmts = {0, 4};

// The truncation post-pass has an exact specification, so it is checked directly on a
// small operator rather than through the convergence of a whole hierarchy
std::vector<int>   rsamg_trunc_size      = {8};
std::vector<int>   rsamg_trunc_use_acc   = {0, 1};
std::vector<float> rsamg_trunc_factor    = {0.0f, 0.2f, 0.5f};
std::vector<int>   rsamg_trunc_max_elmts = {0, 2, 4};

// Function to update tests if environment variable is set
void update_rsamg()
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED",
                           "ROCALUTION_CODE_COVERAGE"}))
    {
        rsamg_size.clear();
        rsamg_smoother.clear();
        rsamg_format.clear();
        rsamg_pre_iter.clear();
        rsamg_post_iter.clear();
        rsamg_cycle.clear();
        rsamg_scaling.clear();
        rsamg_rebuildnumeric.clear();
    }

    if(is_env_var_set("ROCALUTION_CODE_COVERAGE"))
    {
        rsamg_size.push_back(63);
        rsamg_smoother.push_back("Jacobi");
        rsamg_format.insert(rsamg_format.end(), {1, 7});
        rsamg_pre_iter.insert(rsamg_pre_iter.end(), {1, 2});
        rsamg_post_iter.insert(rsamg_post_iter.end(), {1, 2});
        rsamg_cycle.insert(rsamg_cycle.end(), {0, 1});
        rsamg_scaling.insert(rsamg_scaling.end(), {0, 1});
        rsamg_rebuildnumeric.insert(rsamg_rebuildnumeric.end(), {0, 1});
        rsamg_use_acc.push_back(0);
        rsamg_coarsening_strategy.push_back("Greedy");
    }

    if(is_env_var_set("ROCALUTION_EMULATION_SMOKE"))
    {
        rsamg_size.push_back(63);
        rsamg_smoother.push_back("Jacobi");
        rsamg_format.push_back(3);
        rsamg_pre_iter.push_back(1);
        rsamg_post_iter.push_back(1);
        rsamg_cycle.push_back(0);
        rsamg_scaling.push_back(0);
        rsamg_rebuildnumeric.push_back(0);

        // One point of the truncation sweep is enough to keep a smoke run quick
        rsamg_interp_trunc     = {0.2f};
        rsamg_interp_max_elmts = {4};
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_REGRESSION"))
    {
        rsamg_size.push_back(134);
        rsamg_smoother.push_back("Jacobi");
        rsamg_format.push_back(1);
        rsamg_pre_iter.push_back(2);
        rsamg_post_iter.push_back(2);
        rsamg_cycle.insert(rsamg_cycle.end(), {0, 1});
        rsamg_scaling.insert(rsamg_scaling.end(), {0, 1});
        rsamg_rebuildnumeric.push_back(0);
    }
    else if(is_env_var_set("ROCALUTION_EMULATION_EXTENDED"))
    {
        rsamg_size.push_back(134);
        rsamg_smoother.push_back("Jacobi");
        rsamg_format.push_back(7);
        rsamg_pre_iter.push_back(1);
        rsamg_post_iter.push_back(2);
        rsamg_cycle.insert(rsamg_cycle.end(), {0, 1});
        rsamg_scaling.insert(rsamg_scaling.end(), {0, 1});
        rsamg_rebuildnumeric.insert(rsamg_rebuildnumeric.end(), {0, 1});
    }
}

struct RSAMGInitializer
{
    RSAMGInitializer()
    {
        update_rsamg();
    }
};

// Create a global instance of the initializer, so the environment is checked and updated before tests.
RSAMGInitializer rsamg_initializer;

class parameterized_ruge_stueben_amg : public testing::TestWithParam<rsamg_tuple>
{
protected:
    parameterized_ruge_stueben_amg() {}
    virtual ~parameterized_ruge_stueben_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_arguments(rsamg_tuple tup)
{
    Arguments arg;
    arg.size                = std::get<0>(tup);
    arg.smoother            = std::get<1>(tup);
    arg.format              = std::get<2>(tup);
    arg.pre_smooth          = std::get<3>(tup);
    arg.post_smooth         = std::get<4>(tup);
    arg.cycle               = std::get<5>(tup);
    arg.ordering            = std::get<6>(tup);
    arg.rebuildnumeric      = std::get<7>(tup);
    arg.use_acc             = std::get<8>(tup);
    arg.coarsening_strategy = std::get<9>(tup);
    return arg;
}

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_float)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<float>(arg), true);
}

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_double)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(ruge_stueben_amg,
                        parameterized_ruge_stueben_amg,
                        testing::Combine(testing::ValuesIn(rsamg_size),
                                         testing::ValuesIn(rsamg_smoother),
                                         testing::ValuesIn(rsamg_format),
                                         testing::ValuesIn(rsamg_pre_iter),
                                         testing::ValuesIn(rsamg_post_iter),
                                         testing::ValuesIn(rsamg_cycle),
                                         testing::ValuesIn(rsamg_scaling),
                                         testing::ValuesIn(rsamg_rebuildnumeric),
                                         testing::ValuesIn(rsamg_use_acc),
                                         testing::ValuesIn(rsamg_coarsening_strategy)));

class parameterized_ruge_stueben_amg_interpolation
    : public testing::TestWithParam<rsamg_interp_tuple>
{
protected:
    parameterized_ruge_stueben_amg_interpolation() {}
    virtual ~parameterized_ruge_stueben_amg_interpolation() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_interpolation_arguments(rsamg_interp_tuple tup)
{
    Arguments arg;
    arg.size               = std::get<0>(tup);
    arg.format             = std::get<1>(tup);
    arg.use_acc            = std::get<2>(tup);
    arg.interpolation_type = std::get<3>(tup);
    arg.trunc_factor       = std::get<4>(tup);
    arg.p_max_elmts        = std::get<5>(tup);

    // Every interpolation operator here needs the PMIS C/F splitting
    arg.coarsening_strategy = "PMIS";
    arg.smoother            = "Jacobi";
    arg.pre_smooth          = 1;
    arg.post_smooth         = 1;
    arg.cycle               = 0;
    arg.ordering            = 0;
    arg.rebuildnumeric      = 0;

    return arg;
}

TEST_P(parameterized_ruge_stueben_amg_interpolation, ruge_stueben_amg_interpolation_float)
{
    Arguments arg = setup_rsamg_interpolation_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<float>(arg), true);
}

TEST_P(parameterized_ruge_stueben_amg_interpolation, ruge_stueben_amg_interpolation_double)
{
    Arguments arg = setup_rsamg_interpolation_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(ruge_stueben_amg_interpolation,
                        parameterized_ruge_stueben_amg_interpolation,
                        testing::Combine(testing::ValuesIn(rsamg_interp_size),
                                         testing::ValuesIn(rsamg_interp_format),
                                         testing::ValuesIn(rsamg_interp_use_acc),
                                         testing::ValuesIn(rsamg_interpolation_type),
                                         testing::ValuesIn(rsamg_interp_trunc),
                                         testing::ValuesIn(rsamg_interp_max_elmts)));

class parameterized_rsamg_truncation : public testing::TestWithParam<rsamg_trunc_tuple>
{
protected:
    parameterized_rsamg_truncation() {}
    virtual ~parameterized_rsamg_truncation() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_truncation_arguments(rsamg_trunc_tuple tup)
{
    Arguments arg;
    arg.size         = std::get<0>(tup);
    arg.use_acc      = std::get<1>(tup);
    arg.trunc_factor = std::get<2>(tup);
    arg.p_max_elmts  = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_rsamg_truncation, rsamg_truncation_float)
{
    Arguments arg = setup_rsamg_truncation_arguments(GetParam());
    ASSERT_EQ(testing_rsamg_truncation<float>(arg), true);
}

TEST_P(parameterized_rsamg_truncation, rsamg_truncation_double)
{
    Arguments arg = setup_rsamg_truncation_arguments(GetParam());
    ASSERT_EQ(testing_rsamg_truncation<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(rsamg_truncation,
                        parameterized_rsamg_truncation,
                        testing::Combine(testing::ValuesIn(rsamg_trunc_size),
                                         testing::ValuesIn(rsamg_trunc_use_acc),
                                         testing::ValuesIn(rsamg_trunc_factor),
                                         testing::ValuesIn(rsamg_trunc_max_elmts)));

class parameterized_rsamg_truncation_parity
    : public testing::TestWithParam<rsamg_trunc_parity_tuple>
{
protected:
    parameterized_rsamg_truncation_parity() {}
    virtual ~parameterized_rsamg_truncation_parity() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_truncation_parity_arguments(rsamg_trunc_parity_tuple tup)
{
    Arguments arg;
    arg.size         = std::get<0>(tup);
    arg.trunc_factor = std::get<1>(tup);
    arg.p_max_elmts  = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_rsamg_truncation_parity, rsamg_truncation_parity_float)
{
    Arguments arg = setup_rsamg_truncation_parity_arguments(GetParam());
    ASSERT_EQ(testing_rsamg_truncation_parity<float>(arg), true);
}

TEST_P(parameterized_rsamg_truncation_parity, rsamg_truncation_parity_double)
{
    Arguments arg = setup_rsamg_truncation_parity_arguments(GetParam());
    ASSERT_EQ(testing_rsamg_truncation_parity<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(rsamg_truncation_parity,
                        parameterized_rsamg_truncation_parity,
                        testing::Combine(testing::ValuesIn(rsamg_trunc_size),
                                         testing::ValuesIn(rsamg_trunc_factor),
                                         testing::ValuesIn(rsamg_trunc_max_elmts)));
