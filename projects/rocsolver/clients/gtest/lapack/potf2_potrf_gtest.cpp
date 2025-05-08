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

#include "common/lapack/testing_potf2_potrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using potrf_tuple = tuple<vector<I>, printable_char>;

// each size_range vector is a {N, lda, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each uplo_range is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {10, 2, 0},
    // normal (valid) samples
    {10, 10, 1},
    {20, 30, 0},
    {50, 50, 1},
    {70, 80, 0}};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {10, 2, 0},
    // normal (valid) samples
    {10, 10, 1},
    {20, 30, 0},
    {50, 50, 1},
    {70, 80, 0}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0}, {640, 960, 1}, {1000, 1000, 0}, {1024, 1024, 1}, {2000, 2000, 0},
};

const vector<vector<int64_t>> large_matrix_size_range_64 = {
    {192, 192, 0}, {640, 960, 1}, {1000, 1000, 0}, {1024, 1024, 1}, {2000, 2000, 0},
};

template <typename I>
Arguments potrf_setup_arguments(potrf_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

template <bool BLOCKED, typename I>
class POTF2_POTRF : public ::TestWithParam<potrf_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = potrf_setup_arguments(this->GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<I>("n") == 0)
            testing_potf2_potrf_bad_arg<BATCHED, STRIDED, BLOCKED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_potf2_potrf<BATCHED, STRIDED, BLOCKED, T, I>(arg);

        arg.singular = 0;
        testing_potf2_potrf<BATCHED, STRIDED, BLOCKED, T, I>(arg);
    }
};

class POTF2 : public POTF2_POTRF<false, rocblas_int>
{
};

class POTRF : public POTF2_POTRF<true, rocblas_int>
{
};

class POTF2_64 : public POTF2_POTRF<false, int64_t>
{
};

class POTRF_64 : public POTF2_POTRF<true, int64_t>
{
};

// non-batch tests

TEST_P(POTF2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTF2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTF2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTF2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTF2_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTF2_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTF2_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTF2_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRF_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POTF2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(POTF2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(POTF2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(POTF2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(POTRF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(POTRF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(POTRF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(POTRF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(POTF2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTF2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTF2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTF2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(POTF2_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTF2_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTF2_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTF2_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(POTRF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTRF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTRF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTRF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(POTRF_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POTRF_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POTRF_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POTRF_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTF2_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTF2_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(uplo_range)));
