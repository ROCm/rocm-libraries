/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "common/auxiliary/testing_orgxl_ungxl.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgql_tuple;

// each n_size_range vector is a {N, K}

// each m_size_range vector is a {M, lda}

// case when m = 0 and n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> n_size_range = {
    // quick return
    {0, 0},
    // always invalid
    {-1, 1},
    {1, -1},
    {10, 20},
    // invalid for case *
    {30, 25},
    // normal (valid) samples
    {10, 10},
    {20, 20},
};

const vector<vector<int>> m_size_range = {
    // quick return
    {0, 1},
    // always invalid
    {-1, 1},
    {20, 5},
    // invalid for case *
    {25, 25},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130}};

// for daily_lapack tests
const vector<vector<int>> large_n_size_range
    = {{164, 130}, {198, 198}, {130, 130}, {220, 140}, {400, 200}};

const vector<vector<int>> large_m_size_range = {{400, 640}, {640, 640}, {1000, 1000}, {2000, 2000}};

Arguments orgql_setup_arguments(orgql_tuple tup)
{
    vector<int> m_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", m_size[0]);
    arg.set<rocblas_int>("lda", m_size[1]);

    arg.set<rocblas_int>("n", n_size[0]);
    arg.set<rocblas_int>("k", n_size[1]);

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class ORGXL_UNGXL : public ::TestWithParam<orgql_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgql_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_orgxl_ungxl_bad_arg<T, BLOCKED>();

        testing_orgxl_ungxl<T, BLOCKED>(arg);
    }
};

class ORG2L : public ORGXL_UNGXL<false>
{
};

class UNG2L : public ORGXL_UNGXL<false>
{
};

class ORGQL : public ORGXL_UNGXL<true>
{
};

class UNGQL : public ORGXL_UNGXL<true>
{
};

// non-batch tests

TEST_P(ORG2L, __float)
{
    run_tests<float>();
}

TEST_P(ORG2L, __double)
{
    run_tests<double>();
}

TEST_P(UNG2L, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNG2L, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGQL, __float)
{
    run_tests<float>();
}

TEST_P(ORGQL, __double)
{
    run_tests<double>();
}

TEST_P(UNGQL, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGQL, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORG2L,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORG2L,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNG2L,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNG2L,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGQL,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGQL,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGQL,
                         Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGQL,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));
