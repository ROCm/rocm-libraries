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

#include "common/lapack/testing_hseqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> hseqr_tuple;

// each matrix_size_range vector is a {n, ldh, ldz, mtype}
// mtype = 0: random matrix,
// mtype = 1: matrix with clustered eigenvalues,
// mtype = 2: matrix with a cluster of eigenvalues near zero,
// mtype = 3: upper triangular matrix,
// mtype = 4: balanced matrix with isolated eigenvalues (ilo > 1, ihi < n),
// mtype = 5: balanced upper triangular matrix (ilo = ihi),
// mtype = 6: permuted matrix with a small active block (ihi - ilo + 1 ~ n/3),
// mtype = 8 to 13: as 4, with a NaN or an infinite entry set after the reduction to
// Hessenberg form: in H(ilo,ilo) (8), H(ihi,ihi-1) (9), Inf in the real part (10) or in the
// imaginary part (11) of a middle entry (info = ihi at once); with an exactly zero
// subdiagonal entry, a NaN in the top block (12, the bottom block is processed as usual
// and info = its first row - 1) or in the coupling between the blocks (13, info = 0),
// mtype = 14: mixed batch of classes 4 and 12,
// mtype = 15 to 17: as 4, with exactly zero subdiagonal entries set after the reduction:
// upper triangular active block with Inf and NaN diagonal entries (15, 1x1 blocks deflate
// at once: info = 0 and W = diag(H)), a 1x1 block with an infinite entry between two
// irreducible blocks (16, info = 0), and a NaN above a trailing 1x1 block (17, info = ihi-1
// and W(ihi) = H(ihi,ihi))
// (n > 75 uses the multishift QR algorithm)
// (see hseqr_genMatrix; the input is reduced to Hessenberg form with the host GEHRD)

// each op_range vector is a {job, compz}

// case when n = 0, job = S and compz = I will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range
    = {{'S', 'I'}, {'S', 'V'}, {'S', 'N'}, {'E', 'N'}, {'E', 'I'}, {'E', 'V'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {10, 5, 10, 0},
    {10, 10, 5, 0},
    // normal (valid) samples
    {1, 1, 1, 0},
    {2, 2, 2, 0},
    {3, 3, 3, 0},
    {10, 10, 10, 0},
    {35, 40, 35, 0},
    {75, 75, 80, 0},
    {40, 40, 40, 1},
    {60, 60, 60, 2},
    {30, 30, 30, 3},
    {48, 48, 48, 4},
    {20, 20, 20, 5},
    {76, 76, 76, 0},
    {80, 80, 80, 6},
    {100, 100, 100, 6},
    {100, 100, 100, 0},
    // NaN or Inf in the active block
    {50, 50, 50, 8},
    {120, 120, 120, 8},
    {50, 50, 50, 9},
    {120, 120, 120, 9},
    {50, 50, 50, 10},
    {120, 120, 120, 10},
    {50, 50, 50, 11},
    {120, 120, 120, 11},
    {50, 50, 50, 12},
    {120, 120, 120, 12},
    {50, 50, 50, 13},
    {120, 120, 120, 13},
    {50, 50, 50, 14},
    {120, 120, 120, 14},
    // non-finite entries in 1x1 blocks, and a trailing 1x1 block below a NaN
    {50, 50, 50, 15},
    {120, 120, 120, 15},
    {50, 50, 50, 16},
    {120, 120, 120, 16},
    {50, 50, 50, 17},
    {120, 120, 120, 17},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {150, 150, 150, 1},
    {200, 200, 200, 2},
    {250, 250, 250, 4},
    {300, 320, 300, 0},
    {500, 500, 500, 2},
    {800, 800, 800, 0},
    // n >= 3000 (128 shifts recommended by IPARMQ, capped at the deflation window)
    {3000, 3000, 3000, 0},
};

Arguments hseqr_setup_arguments(hseqr_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("ldh", matrix_size[1]);
    arg.set<rocblas_int>("ldz", matrix_size[2]);
    arg.set<rocblas_int>("mtype", matrix_size[3]);
    arg.set<char>("schur_job", op[0]);
    arg.set<char>("compz", op[1]);

    // only testing standard use case/defaults for strides (as the other functions)

    arg.timing = 0;

    return arg;
}

template <rocblas_int MODE>
class HSEQR_BASE : public ::TestWithParam<hseqr_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = hseqr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("schur_job") == 'S'
           && arg.peek<char>("compz") == 'I')
            testing_hseqr_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        arg.alg_mode = MODE;
        testing_hseqr<BATCHED, STRIDED, T>(arg);
    }
};

class HSEQR : public HSEQR_BASE<0>
{
};

// hybrid mode: the core of the aggressive early deflation runs on the host
// (only matrices with n > 75, which use the multishift algorithm, are affected)
class HSEQR_HYBRID : public HSEQR_BASE<1>
{
};

// non-batch tests

TEST_P(HSEQR, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HSEQR, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(HSEQR, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HSEQR, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(HSEQR, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HSEQR, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// hybrid mode tests

TEST_P(HSEQR_HYBRID, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HSEQR_HYBRID, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(HSEQR_HYBRID, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HSEQR_HYBRID, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(HSEQR_HYBRID, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HSEQR_HYBRID, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HSEQR,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HSEQR,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

// hybrid mode (a subset of the sizes)
const vector<vector<int>> hybrid_size_range = {
    {76, 76, 76, 0},     {100, 100, 100, 6},  {200, 200, 200, 0},  {200, 200, 200, 2},
    {250, 250, 250, 4},  {120, 120, 120, 8},  {120, 120, 120, 12}, {120, 120, 120, 13},
    {120, 120, 120, 15}, {120, 120, 120, 16}, {120, 120, 120, 17},
};

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HSEQR_HYBRID,
                         Combine(ValuesIn(hybrid_size_range), ValuesIn(op_range)));

// hybrid mode, large sizes (n >= 3000: more shifts than the deflation window gives)
const vector<vector<int>> large_hybrid_size_range = {
    {800, 800, 800, 0},
    {3000, 3000, 3000, 0},
};

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HSEQR_HYBRID,
                         Combine(ValuesIn(large_hybrid_size_range), ValuesIn(op_range)));
