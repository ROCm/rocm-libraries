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

#include "common/lapack/testing_geqr2_geqrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using geqrf_tuple = tuple<vector<I>, I>;

// each matrix_size_range is a {m, lda}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int64_t> n_size_range_64 = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {64, 98, 130, 220, 400};

const vector<vector<int64_t>> large_matrix_size_range_64 = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

const vector<int64_t> large_n_size_range_64 = {64, 98, 130, 220, 400};

template <typename I>
Arguments geqrf_setup_arguments(geqrf_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    I n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("m", matrix_size[0]);
    arg.set<I>("n", n_size);
    arg.set<I>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED, typename I>
class GEQR2_GEQRF : public ::TestWithParam<geqrf_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = geqrf_setup_arguments(this->GetParam());

        if(arg.peek<I>("m") == 0 && arg.peek<I>("n") == 0)
            testing_geqr2_geqrf_bad_arg<BATCHED, STRIDED, BLOCKED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_geqr2_geqrf<BATCHED, STRIDED, BLOCKED, T, I>(arg);
    }
};

class GEQR2 : public GEQR2_GEQRF<false, rocblas_int>
{
};

class GEQRF : public GEQR2_GEQRF<true, rocblas_int>
{
};

class GEQR2_64 : public GEQR2_GEQRF<false, int64_t>
{
};

class GEQRF_64 : public GEQR2_GEQRF<true, int64_t>
{
};

// non-batch tests

TEST_P(GEQR2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEQR2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEQR2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEQR2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GEQRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEQRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEQRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEQRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEQR2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEQR2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEQR2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEQR2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GEQRF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEQRF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEQRF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEQRF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEQR2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEQR2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEQR2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEQR2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GEQRF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEQRF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEQRF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEQRF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// ptr_batched tests

TEST_P(GEQRF, ptr_batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(GEQRF, ptr_batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(GEQRF, ptr_batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(GEQRF, ptr_batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

// 64-bit API

// non-batch tests

TEST_P(GEQR2_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEQR2_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEQR2_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEQR2_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GEQRF_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEQRF_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEQRF_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEQRF_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEQR2_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEQR2_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEQR2_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEQR2_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GEQRF_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEQRF_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEQRF_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEQRF_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEQR2_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEQR2_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEQR2_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEQR2_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GEQRF_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEQRF_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEQRF_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEQRF_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// ptr_batched tests

TEST_P(GEQRF_64, ptr_batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(GEQRF_64, ptr_batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(GEQRF_64, ptr_batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(GEQRF_64, ptr_batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQR2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQR2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQR2_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(large_n_size_range_64)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQR2_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(n_size_range_64)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQRF_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(large_n_size_range_64)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQRF_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(n_size_range_64)));
