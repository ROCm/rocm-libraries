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

#include "common/auxiliary/testing_larf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using larf_tuple = std::tuple<vector<I>, vector<I>>;

// each size_range vector is a {M,N,lda}

// each incx_range vector is a {incx,s}
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'

// case when M == 0 and incx == 0  also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<int>> incx_range = {
    // invalid
    {0, 0},
    // normal (valid) samples
    {-10, 0},
    {-5, 1},
    {-1, 0},
    {1, 1},
    {5, 0},
    {10, 1}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5},
    // normal (valid) samples
    {12, 20, 12},
    {20, 15, 20},
    {35, 35, 50}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5},
    // normal (valid) samples
    {12, 20, 12},
    {20, 15, 20},
    {35, 35, 50}};
const vector<vector<int64_t>> incx_range_64 = {
    // invalid
    {0, 0},
    // normal (valid) samples
    {-10, 0},
    {-5, 1},
    {-1, 0},
    {1, 1},
    {5, 0},
    {10, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

template <typename I>
Arguments larf_setup_arguments(larf_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    vector<I> inc = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("m", matrix_size[0]);
    arg.set<I>("n", matrix_size[1]);
    arg.set<I>("lda", matrix_size[2]);

    arg.set<I>("incx", inc[0]);
    arg.set<char>("side", inc[1] == 1 ? 'R' : 'L');

    arg.timing = 0;

    return arg;
}

template <typename I>
class LARF_BASE : public ::TestWithParam<larf_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = larf_setup_arguments(this->GetParam());

        if(arg.peek<I>("m") == 0 && arg.peek<I>("incx") == 0)
            testing_larf_bad_arg<T, I>();

        testing_larf<T, I>(arg);
    }
};

class LARF : public LARF_BASE<rocblas_int>
{
};

class LARF_64 : public LARF_BASE<int64_t>
{
};

// non-batch tests

TEST_P(LARF, __float)
{
    run_tests<float>();
}

TEST_P(LARF, __double)
{
    run_tests<double>();
}

TEST_P(LARF, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARF, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LARF_64, __float)
{
    run_tests<float>();
}

TEST_P(LARF_64, __double)
{
    run_tests<double>();
}

TEST_P(LARF_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LARF_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(incx_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARF_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(incx_range_64)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARF_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(incx_range_64)));
