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

#include "common/auxiliary/testing_orgbr_ungbr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgbr_tuple;

// each size_range is a {M, N, K};

// each store_range vector is a {lda, st}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'

// case when m = 0, n = 0 and storev = 'C' will also execute the bad arguments
// test (null handle, null pointers and invalid values)

const vector<vector<int>> store_range = {
    // always invalid
    {-1, 0},
    {-1, 1},
    // normal (valid) samples
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // always quick return
    {0, 0, 0},
    // quick return for storev = 'R' invalid for 'C'
    {0, 1, 0},
    // quick return for storev = 'C' invalid for 'R'
    {1, 0, 0},
    // always invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // invalid for storev = 'C'
    {10, 30, 5},
    // invalid for storev = 'R'
    {30, 10, 5},
    // always invalid
    {30, 10, 20},
    {10, 30, 20},
    // normal (valid) samples
    {30, 30, 0},
    {20, 20, 20},
    {50, 50, 50},
    {100, 100, 50}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{150, 150, 100}, {270, 270, 270},    {400, 400, 400},
       {800, 800, 300}, {1000, 1000, 1000}, {1500, 1500, 800}};

Arguments orgbr_setup_arguments(orgbr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", size[0]);
    arg.set<rocblas_int>("n", size[1]);
    arg.set<rocblas_int>("k", size[2]);

    arg.set<rocblas_int>("lda", size[0] + store[0] * 10);
    arg.set<char>("storev", store[1] == 1 ? 'R' : 'C');

    arg.timing = 0;

    return arg;
}

class ORGBR_UNGBR : public ::TestWithParam<orgbr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgbr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0
           && arg.get<char>("storev") == 'C')
            testing_orgbr_ungbr_bad_arg<T>();

        testing_orgbr_ungbr<T>(arg);
    }
};

class ORGBR : public ORGBR_UNGBR
{
};

class UNGBR : public ORGBR_UNGBR
{
};

// non-batch tests

TEST_P(ORGBR, __float)
{
    run_tests<float>();
}

TEST_P(ORGBR, __double)
{
    run_tests<double>();
}

TEST_P(UNGBR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGBR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         ORGBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         UNGBR,
                         Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGBR, Combine(ValuesIn(size_range), ValuesIn(store_range)));
