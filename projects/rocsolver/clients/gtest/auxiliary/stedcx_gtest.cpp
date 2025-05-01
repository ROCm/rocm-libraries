/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_stedcx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> stedcx_tuple;

// each size_range vector is a {n, ldc}

// each ops_range vector is a {rng, vl, vu, il, iu}
// if rng = 0, then find all eigenvalues
// if rng = 1, then find eigenvalues in (vl, vu]
// if rng = 2, then find the il-th to the iu-th eigenvalue

// Note: all tests are prepared with diagonally dominant matrices that have random diagonal
// elements in [-20, -11] U [11, 20], and off-diagonal elements in [-0.4, 0.5].
// Thus, all the eigenvalues are guaranteed to be in [-20, 20]

// case when n == 0, and rng == 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {2, 1},
    // normal (valid) samples
    {1, 1},
    {15, 20},
    {20, 20},
    {60, 64}};

const vector<vector<int>> ops_range = {
    // always invalid
    {1, 2, 1, 0, 0},
    {2, 0, 0, 0, -1},
    {2, 0, 0, 1, 80},
    // valid only when n=0
    {2, 0, 0, 1, 0},
    // valid only when n>1
    {2, 0, 0, 4, 8},
    {2, 0, 0, 3, 15},
    {2, 0, 0, 5, 10},
    // always valid samples
    {0, 0, 0, 0, 0},
    {1, -10, -7, 0, 0},
    {1, -15, 15, 0, 0},
    {1, -5, 5, 0, 0},
    {1, 5, 15, 0, 0},
    {1, 35, 55, 0, 0}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{120, 120}, {256, 270}, {350, 350}};

const vector<vector<int>> large_ops_range
    = {{0, 0, 0, 0, 0},  {1, -15, 15, 0, 0}, {1, -25, 0, 0, 0},
       {1, 0, 15, 0, 0}, {2, 0, 0, 50, 75},  {2, 0, 0, 1, 25}};

Arguments stedcx_setup_arguments(stedcx_tuple tup)
{
    Arguments arg;

    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldc", size[1]);

    arg.set<char>("erange", (op[0] == 0 ? 'A' : (op[0] == 1 ? 'V' : 'I')));
    arg.set<double>("vl", op[1]);
    arg.set<double>("vu", op[2]);
    arg.set<rocblas_int>("il", op[3]);
    arg.set<rocblas_int>("iu", op[4]);

    // case evect = N is not implemented for now.
    // it could be added if stedcx goes to public API
    arg.set<char>("evect", 'I');

    arg.timing = 0;

    return arg;
}

class STEDCX : public ::TestWithParam<stedcx_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = stedcx_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("erange") == 'A')
            testing_stedcx_bad_arg<T>();

        testing_stedcx<T>(arg);
    }
};

// non-batch tests

TEST_P(STEDCX, __float)
{
    run_tests<float>();
}

TEST_P(STEDCX, __double)
{
    run_tests<double>();
}

TEST_P(STEDCX, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEDCX, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEDCX,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_ops_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, STEDCX, Combine(ValuesIn(size_range), ValuesIn(ops_range)));
