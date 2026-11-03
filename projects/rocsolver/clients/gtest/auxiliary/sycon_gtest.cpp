/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_sycon.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using sycon_tuple = std::tuple<vector<I>, printable_char>;

// each size_range vector is a {n, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// each uplo_range is a {uplo}

// case when n == 0 and uplo = L also executes the bad arguments test

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid sizes
    {-1, 1, 0},
    {10, 5, 0},
    // normal (valid) samples
    {1, 1, 0},
    {1, 1, 1},
    {2, 2, 0},
    {12, 12, 0},
    {20, 25, 1},
    {33, 50, 0},
    {50, 50, 1}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 0},
    // invalid sizes
    {-1, 1, 0},
    {10, 5, 0},
    // normal (valid) samples
    {1, 1, 0},
    {1, 1, 1},
    {2, 2, 0},
    {12, 12, 0},
    {20, 25, 1},
    {33, 50, 0},
    {50, 50, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 0}, {640, 800, 1}, {1024, 1200, 0}, {2048, 2050, 0}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{192, 192, 0}, {640, 800, 1}, {1024, 1200, 0}, {2547, 2550, 0}};

template <typename I>
Arguments sycon_setup_arguments(sycon_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("lda", matrix_size[1]);
    arg.set<char>("uplo", uplo);

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

template <typename I>
class SYCON_BASE : public ::TestWithParam<sycon_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = sycon_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<char>("uplo") == 'L')
            testing_sycon_bad_arg<T, I>();

        testing_sycon<T, I>(arg);
    }
};

class SYCON : public SYCON_BASE<rocblas_int>
{
};

class SYCON_64 : public SYCON_BASE<int64_t>
{
};

TEST_P(SYCON, __float)
{
    run_tests<float>();
}

TEST_P(SYCON, __double)
{
    run_tests<double>();
}

TEST_P(SYCON, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SYCON, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(SYCON_64, __float)
{
    run_tests<float>();
}

TEST_P(SYCON_64, __double)
{
    run_tests<double>();
}

TEST_P(SYCON_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SYCON_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYCON,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYCON,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYCON_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYCON_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(uplo_range)));
