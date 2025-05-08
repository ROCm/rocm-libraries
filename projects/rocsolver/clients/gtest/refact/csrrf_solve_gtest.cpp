/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/refact/testing_csrrf_solve.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>, printable_char> csrrf_solve_tuple;

// each n_range vector is {n, ldb}

// each nnz_range vector is {nnzT, nrhs}

// if mode = '1', then the factorization is LU
// if mode = '2', then the factorization is Cholesky

// case when n = 0 and nnz = 10 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> n_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 2},
    // normal (valid) samples
    {20, 20},
    {50, 60},
};
const vector<vector<int>> nnz_range = {
    // quick return
    {10, 0},
    // invalid
    {-1, 1},
    {20, -1},
    // normal (valid) samples
    {60, 1},
    {60, 10},
    {60, 30},
    {100, 1},
    {100, 10},
    {100, 30},
    {140, 1},
    {140, 10},
    {140, 30},
};

const vector<printable_char> mode_range = {
    '1', // for LU
    '2', // for Cholesky
};

// for daily_lapack tests
const vector<vector<int>> large_n_range = {
    // normal (valid) samples
    {100, 110},
    {250, 250},
};
const vector<vector<int>> large_nnz_range = {
    // normal (valid) samples
    {300, 1}, {300, 10}, {300, 30}, {500, 1}, {500, 10}, {500, 30}, {700, 1}, {700, 10}, {700, 30},
};

Arguments csrrf_solve_setup_arguments(csrrf_solve_tuple tup)
{
    vector<int> n_v = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);
    int mode = std::get<2>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v[0]);
    arg.set<rocblas_int>("ldb", n_v[1]);
    arg.set<rocblas_int>("nnzT", nnz_v[0]);
    arg.set<rocblas_int>("nrhs", nnz_v[1]);
    arg.set<char>("rfinfo_mode", mode);

    arg.timing = 0;

    return arg;
}

class CSRRF_SOLVE : public ::TestWithParam<csrrf_solve_tuple>
{
protected:
    void SetUp() override
    {
        if(rocsolver_create_rfinfo(nullptr, nullptr) == rocblas_status_not_implemented)
            GTEST_SKIP() << "Sparse functionality is not enabled";
    }
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_solve_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nnzT") == 10)
            testing_csrrf_solve_bad_arg<T>();

        testing_csrrf_solve<T>(arg);
    }
};

// non-batch tests

TEST_P(CSRRF_SOLVE, __float)
{
    run_tests<float>();
}

TEST_P(CSRRF_SOLVE, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_SOLVE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_SOLVE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRRF_SOLVE,
                         Combine(ValuesIn(large_n_range),
                                 ValuesIn(large_nnz_range),
                                 ValuesIn(mode_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRRF_SOLVE,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range), ValuesIn(mode_range)));
