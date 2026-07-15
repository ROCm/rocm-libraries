/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_local_stencil.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
/*
typedef std::tuple<int, int, int, int, bool, int, bool> backend_tuple;

int backend_rank[] = {-1, 0, 13};
int backend_dev_node[] = {-1, 0, 13};
int backend_dev[] = {-1, 0, 13};
int backend_omp_threads[] = {-1, 0, 8};
bool backend_affinity[] = {true, false};
int backend_omp_threshold[] = {-1, 0, 20000};
bool backend_disable_acc[] = {true, false};

class parameterized_backend : public testing::TestWithParam<backend_tuple>
{
    protected:
    parameterized_backend() {}
    virtual ~parameterized_backend() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_backend_arguments(backend_tuple tup)
{
    Arguments arg;
    arg.rank          = std::get<0>(tup);
    arg.dev_per_node  = std::get<1>(tup);
    arg.dev           = std::get<2>(tup);
    arg.omp_nthreads  = std::get<3>(tup);
    arg.omp_affinity  = std::get<4>(tup);
    arg.omp_threshold = std::get<5>(tup);
    arg.use_acc       = std::get<6>(tup);
    return arg;
}
*/
TEST(local_stencil_bad_args, local_stencil)
{
    if(is_any_env_var_set({"ROCALUTION_EMULATION_SMOKE",
                           "ROCALUTION_EMULATION_REGRESSION",
                           "ROCALUTION_EMULATION_EXTENDED"}))
    {
        GTEST_SKIP();
    }

    testing_local_stencil_bad_args<float>();
}
/*
TEST_P(parameterized_backend, backend)
{
    Arguments arg = setup_backend_arguments(GetParam());
    testing_backend(arg);
}

INSTANTIATE_TEST_CASE_P(backend,
                        parameterized_backend,
                        testing::Combine(testing::ValuesIn(backend_rank),
                                         testing::ValuesIn(backend_dev_node),
                                         testing::ValuesIn(backend_dev),
                                         testing::ValuesIn(backend_omp_threads),
                                         testing::ValuesIn(backend_affinity),
                                         testing::ValuesIn(backend_omp_threshold),
                                         testing::ValuesIn(backend_disable_acc)));
*/
