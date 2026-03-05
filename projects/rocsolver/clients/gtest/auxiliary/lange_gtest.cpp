/* **************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_lange.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using lange_tuple = std::tuple<vector<I>, char>;

// each size_range vector is a {M, N, lda}

// each norm_type is one of: '1', 'F', 'I', 'M'

// case when M == 0 also executes the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> norm_range = {'1', 'F', 'I', 'M'};

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

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

template <typename I>
Arguments lange_setup_arguments(lange_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char norm_type = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("m", matrix_size[0]);
    arg.set<I>("n", matrix_size[1]);
    arg.set<I>("lda", matrix_size[2]);

    arg.set<char>("norm_type", norm_type);

    arg.timing = 0;

    return arg;
}

template <typename I>
class LANGE_BASE : public ::TestWithParam<lange_tuple<I>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lange_setup_arguments(this->GetParam());

        if(arg.peek<I>("m") == 0)
            testing_lange_bad_arg<T, I>();

        testing_lange<T, I>(arg);
    }
};

class LANGE : public LANGE_BASE<rocblas_int>
{
};

class LANGE_64 : public LANGE_BASE<int64_t>
{
};

// non-batch tests

TEST_P(LANGE, __float)
{
    run_tests<float>();
}

TEST_P(LANGE, __double)
{
    run_tests<double>();
}

TEST_P(LANGE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANGE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(LANGE_64, __float)
{
    run_tests<float>();
}

TEST_P(LANGE_64, __double)
{
    run_tests<double>();
}

TEST_P(LANGE_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LANGE_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANGE,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(norm_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANGE,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(norm_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LANGE_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(norm_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LANGE_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(norm_range)));
