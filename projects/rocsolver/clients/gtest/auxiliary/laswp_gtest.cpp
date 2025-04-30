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

#include "common/auxiliary/testing_laswp.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> laswp_tuple;

// each range1 vector is a {n,lda}

// each range2 vector is a {k1,k2,inc}

// case when n = 0, k1 = 1 and k2 = 3  will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> range1 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 0},
    // normal (valid) samples
    {10, 100},
    {20, 100},
    {30, 100}};
const vector<vector<int>> range2 = {
    // invalid
    {0, 1, 1},
    {1, 0, 1},
    {1, 2, 0},
    {2, 1, 1},
    // normal (valid) samples
    {1, 3, 1},
    {3, 5, 2},
    {5, 10, -1},
    {3, 12, -2}};

// for daily_lapack tests
const vector<vector<int>> large_range1 = {{192, 100}, {250, 100}, {500, 100}, {1500, 100}};
const vector<vector<int>> large_range2 = {{1, 50, 1}, {5, 60, 2}, {3, 70, -1}, {20, 100, -2}};

Arguments laswp_setup_arguments(laswp_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> pivots = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<rocblas_int>("k1", pivots[0]);
    arg.set<rocblas_int>("k2", pivots[1]);
    arg.set<rocblas_int>("incx", pivots[2]);

    arg.timing = 0;

    return arg;
}

class LASWP : public ::TestWithParam<laswp_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = laswp_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("k1") == 1
           && arg.peek<rocblas_int>("k2") == 3)
            testing_laswp_bad_arg<T>();

        testing_laswp<T>(arg);
    }
};

// non-batch tests

TEST_P(LASWP, __float)
{
    run_tests<float>();
}

TEST_P(LASWP, __double)
{
    run_tests<double>();
}

TEST_P(LASWP, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LASWP, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LASWP, Combine(ValuesIn(large_range1), ValuesIn(large_range2)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LASWP, Combine(ValuesIn(range1), ValuesIn(range2)));
