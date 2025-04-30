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

#include "common/auxiliary/testing_ormbr_unmbr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> ormbr_tuple;

// each size_range vector is a {M, N, K}

// each store_range vector is a {lda, ldc, s, t, st}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if ldc = -1, then ldc < limit (invalid size)
// if ldc = 0, then ldc = limit
// if ldc = 1, then ldc > limit
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'
// if t = 0, then trans = 'N'
// if t = 1, then trans = 'T'
// if t = 2, then trans = 'C'
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'

// case when m = 0, n = 1, side = 'L', trans = 'T' and storev = 'C'
// will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> store_range = {
    // invalid
    {-1, 0, 0, 0, 0},
    {0, -1, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 0, 0, 0},
    {1, 1, 0, 0, 1},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 2, 1},
    {0, 0, 1, 0, 0},
    {0, 0, 1, 0, 1},
    {0, 0, 1, 1, 0},
    {0, 0, 1, 1, 1},
    {0, 0, 1, 2, 0},
    {0, 0, 1, 2, 1},
};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    // invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // normal (valid) samples
    {10, 30, 5},
    {20, 5, 10},
    {20, 20, 25},
    {50, 50, 30},
    {70, 40, 40},
};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {
    {200, 150, 100}, {270, 270, 270}, {400, 400, 405}, {800, 500, 300}, {1500, 1000, 300},
};

Arguments ormbr_setup_arguments(ormbr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    rocblas_int m = size[0];
    rocblas_int n = size[1];
    rocblas_int k = size[2];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("k", k);

    rocblas_int nq = store[2] == 0 ? m : n;

    if(store[4] == 0)
        arg.set<rocblas_int>("lda", nq + store[0] * 10);
    else
        arg.set<rocblas_int>("lda", min(nq, k) + store[0] * 10);
    arg.set<rocblas_int>("ldc", m + store[1] * 10);
    arg.set<char>("side", store[2] == 0 ? 'L' : 'R');
    arg.set<char>("trans", (store[3] == 0 ? 'N' : (store[3] == 1 ? 'T' : 'C')));
    arg.set<char>("storev", store[4] == 0 ? 'C' : 'R');

    arg.timing = 0;

    return arg;
}

class ORMBR_UNMBR : public ::TestWithParam<ormbr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = ormbr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 1
           && arg.peek<char>("side") == 'L' && arg.peek<char>("trans") == 'T'
           && arg.peek<char>("storev") == 'C')
            testing_ormbr_unmbr_bad_arg<T>();

        testing_ormbr_unmbr<T>(arg);
    }
};

class ORMBR : public ORMBR_UNMBR
{
};

class UNMBR : public ORMBR_UNMBR
{
};

// non-batch tests

TEST_P(ORMBR, __float)
{
    run_tests<float>();
}

TEST_P(ORMBR, __double)
{
    run_tests<double>();
}

TEST_P(UNMBR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNMBR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORMBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORMBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNMBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNMBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));
