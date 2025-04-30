/* **************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_stebz.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> stebz_tuple;

// each size_range vector is a {n, ord}
// if ord = 1, then order eigenvalues by blocks
// if ord = 0, then order eigenvalues of the entire matrix

// each ops_range vector is a {rng, vl, vu, il, iu, tol}
// if rng = 0, then find all eigenvalues
// if rng = 1, then find eigenavlues in (vl, vu]
// if rng = 2, then find the il-th to the iu-th eigenvalue

// Note: all tests are prepared with diagonally dominant matrices that have random diagonal
// elements in [-20, -11] U [11, 20], and off-diagonal elements in [-0.4, 0.5].
// Thus, all the eigenvalues are guaranteed to be in [-20, 20]

// case when n == 0, ord == 0, and rng == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {1, 1},
    {15, 0},
    {20, 1},
    {64, 0}};
const vector<vector<int>> ops_range = {
    // always invalid
    {1, 2, 1, 0, 0, 0},
    {2, 0, 0, 0, -1, 0},
    {2, 0, 0, 1, 80, 0},
    // valid only when n=0
    {2, 0, 0, 1, 0, 0},
    // valid only when n>0
    {2, 0, 0, 1, 5, 0},
    {2, 0, 0, 1, 15, 0},
    {2, 0, 0, 7, 12, 0},
    // always valid samples
    {0, 0, 0, 0, 0, 0},
    {1, -15, -5, 0, 0, -1},
    {1, -15, 15, 0, 0, 1},
    {1, -5, 5, 0, 0, 0},
    {1, 5, 15, 0, 0, -1},
    {1, 35, 55, 0, 0, 0}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{120, 1}, {256, 0}, {350, 1}, {512, 0}, {1024, 1}};
const vector<vector<int>> large_ops_range
    = {{0, 0, 0, 0, 0, -1}, {1, -15, 15, 0, 0, 0}, {1, -25, 0, 0, 0, 1},
       {1, 0, 15, 0, 0, 0}, {2, 0, 0, 50, 75, -1}, {2, 0, 0, 1, 25, 0}};

Arguments stebz_setup_arguments(stebz_tuple tup)
{
    Arguments arg;

    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    arg.set<rocblas_int>("n", size[0]);
    arg.set<char>("eorder", (size[1] == 0 ? 'E' : 'B'));

    arg.set<char>("erange", (op[0] == 0 ? 'A' : (op[0] == 1 ? 'V' : 'I')));
    arg.set<double>("vl", op[1]);
    arg.set<double>("vu", op[2]);
    arg.set<rocblas_int>("il", op[3]);
    arg.set<rocblas_int>("iu", op[4]);
    arg.set<double>("abstol", op[5]);

    arg.timing = 0;

    return arg;
}

class STEBZ : public ::TestWithParam<stebz_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = stebz_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("eorder") == 'E'
           && arg.peek<char>("erange") == 'A')
            testing_stebz_bad_arg<T>();

        testing_stebz<T>(arg);
    }
};

// non-batch tests

TEST_P(STEBZ, __float)
{
    run_tests<float>();
}

TEST_P(STEBZ, __double)
{
    run_tests<double>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEBZ,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_ops_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, STEBZ, Combine(ValuesIn(size_range), ValuesIn(ops_range)));
