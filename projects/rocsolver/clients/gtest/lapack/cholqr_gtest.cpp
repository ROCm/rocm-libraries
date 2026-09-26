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

#include "common/lapack/testing_cholqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using cholqr_tuple = tuple<vector<I>, int>;

/** Test parameter tuple: {{m, n, lda, ldw}, singular}
    the value of singular is used to manipulate the conditioning of the input matrix
    to test the different algorithm configs (choleskyQR1, choleskyQR2, shifted_choleskyQR3)

    - If singular = 0, the input matrix is non singular (cond(A) small).
    - If singular = 1, the matrix is singular (has zero columns/rows, i.e cond(A) = inf or very large)
    - If singular = s > 1, then a matrix A with repeated columns/rows is diagonally loaded as
      A = A + I * eps * 10^(s - 1) to gradually reduce cond(A). **/

// case when m = n = 0 and singular = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// ============================================================================
// Size cases for checkin_lapack tests (small/quick tests)
// ============================================================================
const vector<int> singular_range = {
    0, // test with rocsolver_cholqr_shift_none, cholnum = 1
    1, // test with rocsolver_cholqr_shift_computed, cholnum = 2
    3, // test with rocsolver_cholqr_shift_computed, cholnum = 2,3
    5 // test with rocsolver_cholqr_shift_none, cholnum = 1,2
    // cases 3 and 5 are tested with different cholnum values to verify that
    // the orthogonality error improved.
};

const vector<vector<int>> known_bug_size_range = {
    // normal (valid) samples
    {15, 15, 15, 15},
};

const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1}, // m = 0
    {1, 0, 1, 1}, // n = 0
    {0, 0, 1, 1}, // m = n = 0
    // invalid
    {-1, 1, 1, 1}, // invalid size
    {20, 5, 10, 5}, // invalid lda
    {20, 10, 20, 5}, // invalid ldr (m > n)
    {10, 20, 10, 5}, // invalid ldr (n > m)
    // normal (valid) samples
    {30, 30, 100, 30},
    {40, 40, 40, 100},
    {100, 30, 130, 30},
    {20, 80, 20, 20},
    {10, 100, 40, 80},
};

const vector<vector<int64_t>> known_bug_size_range_64 = {
    // normal (valid) samples
    {15, 15, 15, 15},
};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 1, 1}, // m = 0
    {1, 0, 1, 1}, // n = 0
    {0, 0, 1, 1}, // m = n = 0
    // invalid
    {-1, 1, 1, 1}, // invalid size
    {20, 5, 10, 5}, // invalid lda
    {20, 10, 20, 5}, // invalid ldr (m > n)
    {10, 20, 10, 5}, // invalid ldr (n > m)
    // normal (valid) samples
    {30, 30, 100, 30},
    {40, 40, 40, 100},
    {100, 30, 130, 30},
    {20, 80, 20, 20},
    {10, 100, 40, 80},
};

// ============================================================================
// Test sizes for daily_lapack tests (larger/longer tests)
// ============================================================================
const vector<int> large_singular_range = {
    0, // test with rocsolver_cholqr_shift_none, cholnum = 1
    1, // test with rocsolver_cholqr_shift_computed, cholnum = 2
};

const vector<vector<int>> large_matrix_size_range
    = {{152, 152, 152, 152}, {640, 800, 640, 640}, {1024, 256, 1000, 256}};

const vector<vector<int64_t>> large_matrix_size_range_64
    = {{152, 152, 152, 152}, {640, 800, 640, 640}, {1024, 256, 1000, 256}};

// ============================================================================
// Argument setup functions
// ============================================================================
template <typename I>
Arguments cholqr_setup_arguments(cholqr_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    int singular = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("m", matrix_size[0]);
    arg.set<I>("n", matrix_size[1]);
    arg.set<I>("lda", matrix_size[2]);
    arg.set<I>("ldw", matrix_size[3]);

    // Set the algorithm
    arg.singular = singular;
    if(singular == 0)
    {
        arg.set<char>("cholshift", 'N');
        arg.set<int>("cholnum", 1);
    }
    else if(singular == 1)
    {
        arg.set<char>("cholshift", 'C');
        arg.set<int>("cholnum", 2);
    }
    else if(singular == 3)
    {
        arg.set<char>("cholshift", 'C');
        arg.set<int>("cholnum", 3);
    }
    else
    {
        arg.set<char>("cholshift", 'N');
        arg.set<int>("cholnum", 2);
    }

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

        if(arg.peek<I>("m") == 0 && arg.peek<I>("n") == 0 && arg.singular == 0)
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
                         Combine(ValuesIn(matrix_size_range), ValuesIn(singular_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CHOLQR_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(singular_range)));

// Daily tests: reduced algorithms, larger sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CHOLQR,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_singular_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CHOLQR_64,
                         Combine(ValuesIn(large_matrix_size_range_64),
                                 ValuesIn(large_singular_range)));

// Known bug tests
INSTANTIATE_TEST_SUITE_P(known_bug,
                         CHOLQR,
                         Combine(ValuesIn(known_bug_size_range), ValuesIn(singular_range)));

INSTANTIATE_TEST_SUITE_P(known_bug,
                         CHOLQR_64,
                         Combine(ValuesIn(known_bug_size_range_64), ValuesIn(singular_range)));
