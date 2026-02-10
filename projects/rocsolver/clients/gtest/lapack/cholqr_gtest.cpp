/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/lapack/testing_cholqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

/**
 * Test parameter tuple: {m, lda, ldr}, n, algo
 *
 * Algorithm modes:
 *   '1' = Basic CholeskyQR1 - fast but may fail on ill-conditioned matrices
 *   '2' = CholeskyQR2 - more robust, does two iterations (default)
 *   '3' = Shifted CholeskyQR3 with internally computed shifts - most robust
 *   '4' = Shifted CholeskyQR3 with user-provided sigma shifts
 */
template <typename I>
using cholqr_tuple = tuple<vector<I>, I, char>;

// case when m = n = 0 and alg_select = 1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// ============================================================================
// Test sizes for checkin_lapack tests (small/quick tests)
// ============================================================================

// Matrix sizes: {m, lda, ldr}
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {20, 5, 20}, // invalid lda
    {20, 20, 5}, // invalid ldr
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 70},
    {130, 130, 130},
    {150, 200, 150}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {20, 5, 20}, // invalid lda
    {20, 20, 5}, // invalid ldr
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 70},
    {130, 130, 130},
    {150, 200, 150}};

const vector<int64_t> n_size_range_64 = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

// Algorithm choices for testing
const vector<char> algo_range = {
    '1', // cholqr1
    '2', // cholqr2
    '3' // cholqr3_compute
};

// ============================================================================
// Test sizes for daily_lapack tests (larger/longer tests)
// ============================================================================

const vector<vector<int>> large_matrix_size_range = {
    {152, 152, 152},
    {640, 640, 640},
    {1000, 1024, 1000},
};

const vector<int> large_n_size_range = {64, 98, 130, 220, 400};

const vector<vector<int64_t>> large_matrix_size_range_64 = {
    {152, 152, 152},
    {640, 640, 640},
    {1000, 1024, 1000},
};

const vector<int64_t> large_n_size_range_64 = {64, 98, 130, 220, 400};

// ============================================================================
// Argument setup functions
// ============================================================================

template <typename I>
Arguments cholqr_setup_arguments(cholqr_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    I n_size = std::get<1>(tup);
    char algo = std::get<2>(tup);

    Arguments arg;

    arg.set<I>("m", matrix_size[0]);
    arg.set<I>("n", n_size);
    arg.set<I>("lda", matrix_size[1]);
    arg.set<I>("ldr", matrix_size.size() > 2 ? matrix_size[2] : n_size);

    // Set the algorithm
    arg.set<char>("alg_select", algo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

// ============================================================================
// Test class templates
// ============================================================================

template <typename I>
class CHOLQR_BASE : public ::TestWithParam<cholqr_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = cholqr_setup_arguments(this->GetParam());

        if(arg.peek<I>("m") == 0 && arg.peek<I>("n") == 0 && arg.peek<char>("alg_select") == '1')
            testing_cholqr_bad_arg<BATCHED, STRIDED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_cholqr<BATCHED, STRIDED, T, I>(arg);
    }
};

class CHOLQR : public CHOLQR_BASE<rocblas_int>
{
};

class CHOLQR_64 : public CHOLQR_BASE<int64_t>
{
};

// ============================================================================
// 32-bit integer API tests
// ============================================================================

// non-batch tests

TEST_P(CHOLQR, __float)
{
    run_tests<false, false, float>();
}

TEST_P(CHOLQR, __double)
{
    run_tests<false, false, double>();
}

TEST_P(CHOLQR, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(CHOLQR, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(CHOLQR, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(CHOLQR, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(CHOLQR, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(CHOLQR, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(CHOLQR, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(CHOLQR, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(CHOLQR, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(CHOLQR, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// ============================================================================
// 64-bit integer API tests
// ============================================================================

// non-batch tests

TEST_P(CHOLQR_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(CHOLQR_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(CHOLQR_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(CHOLQR_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(CHOLQR_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(CHOLQR_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(CHOLQR_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(CHOLQR_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(CHOLQR_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(CHOLQR_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(CHOLQR_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(CHOLQR_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// ============================================================================
// Test suite instantiations
// ============================================================================

// Checkin tests: all algorithms, smaller sizes
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CHOLQR,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(n_size_range),
                                 ValuesIn(algo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CHOLQR_64,
                         Combine(ValuesIn(matrix_size_range_64),
                                 ValuesIn(n_size_range_64),
                                 ValuesIn(algo_range)));

// Daily tests: reduced algorithms, larger sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CHOLQR,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(large_n_size_range),
                                 ValuesIn(algo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CHOLQR_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(large_n_size_range_64),
                                 ValuesIn(algo_range)));
