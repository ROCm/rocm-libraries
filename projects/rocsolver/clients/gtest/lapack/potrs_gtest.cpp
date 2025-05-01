/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/lapack/testing_potrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using potrs_tuple = tuple<vector<I>, vector<I>>;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, uplo};
// if uplo = 0 then upper
// if uplo = 1 then lower

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};
const vector<vector<int>> matrix_sizeB_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 1},
};
const vector<vector<int64_t>> matrix_sizeA_range_64 = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};
const vector<vector<int64_t>> matrix_sizeB_range_64 = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 1},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range
    = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};
const vector<vector<int>> large_matrix_sizeB_range = {
    {100, 0}, {150, 0}, {200, 1}, {524, 1}, {1000, 0},
};
const vector<vector<int64_t>> large_matrix_sizeA_range_64
    = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};
const vector<vector<int64_t>> large_matrix_sizeB_range_64 = {
    {100, 0}, {150, 0}, {200, 1}, {524, 1}, {1000, 0},
};

template <typename I>
Arguments potrs_setup_arguments(potrs_tuple<I> tup)
{
    vector<I> matrix_sizeA = std::get<0>(tup);
    vector<I> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_sizeA[0]);
    arg.set<I>("nrhs", matrix_sizeB[0]);
    arg.set<I>("lda", matrix_sizeA[1]);
    arg.set<I>("ldb", matrix_sizeA[2]);

    if(matrix_sizeB[1] == 0)
        arg.set<char>("uplo", 'U');
    else
        arg.set<char>("uplo", 'L');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <typename I>
class POTRS_BASE : public ::TestWithParam<potrs_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = potrs_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<I>("nrhs") == 0)
            testing_potrs_bad_arg<BATCHED, STRIDED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_potrs<BATCHED, STRIDED, T, I>(arg);
    }
};

class POTRS : public POTRS_BASE<rocblas_int>
{
};

class POTRS_64 : public POTRS_BASE<int64_t>
{
};

// non-batch tests

TEST_P(POTRS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRS_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRS_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRS_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRS_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POTRS, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(POTRS, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(POTRS, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(POTRS, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(POTRS_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(POTRS_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(POTRS_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(POTRS_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(POTRS, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTRS, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTRS, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTRS, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(POTRS_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTRS_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTRS_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTRS_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRS_64,
                         Combine(ValuesIn(large_matrix_sizeA_range_64),
                                 ValuesIn(large_matrix_sizeB_range_64)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRS_64,
                         Combine(ValuesIn(matrix_sizeA_range_64), ValuesIn(matrix_sizeB_range_64)));
