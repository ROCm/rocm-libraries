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

#include "common/lapack/testing_geev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> geev_tuple;

// each matrix_size_range vector is a {n, lda, ldv, mtype[, gap]}
// (ldv is the leading dimension of the computed eigenvectors; the one not
// referenced, if any, is set to 1; with gap > 0 the strides are n + gap for W and
// ld*n + gap for A, VL and VR, and the gaps between the members must not be changed)
// mtype = 0: random matrix,
// mtype = 1: matrix with clustered eigenvalues (1e-3 * random + I),
// mtype = 2: badly scaled matrix D^(-1) * random * D,
// mtype = 3: random upper Hessenberg matrix (ill-conditioned eigenvalues),
// mtype = 4: random upper triangular matrix (GEBAL isolates all the eigenvalues),
// mtype = 5: random matrix with huge entries (1e300, or 1e35 in single precision),
// mtype = 6: random matrix with tiny entries (1e-300, or 1e-35 in single precision),
// mtype = 7: matrix with isolated eigenvalues and bad scaling (ilo > 1, ihi < n),
// mtype = 8: matrix with isolated eigenvalues and a NaN in the active block (info > 0)
// mtype = 9: mixed batch, classes 0, 7 and 8 (different ranges ilo:ihi in one batch),
// mtype = 10: matrix with a NaN outside the active block (in the coupling with an
//             isolated eigenvalue; info = 0),
// mtype = 11: mixed batch, classes 0 and 10 (a single matrix is of class 10)
// (n > 75 uses the multishift QR algorithm in HSEQR)
// (see geev_genMatrix)

// each op_range vector is a {jobvl, jobvr}

// case when n = 0, jobvl = V and jobvr = V will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'N'}, {'N', 'V'}, {'V', 'N'}, {'V', 'V'}};

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
    {2, 2, 2, 1},
    {3, 3, 3, 4},
    {10, 10, 10, 0},
    {10, 10, 10, 7},
    {20, 20, 20, 5},
    {20, 20, 20, 6},
    {33, 40, 35, 2},
    {40, 40, 40, 3},
    {75, 75, 80, 0},
    {76, 76, 76, 0},
    {100, 100, 100, 1},
    {100, 100, 100, 2},
    {100, 100, 100, 5},
    {150, 150, 150, 3},
    {150, 150, 150, 6},
    {150, 150, 150, 7},
    {200, 200, 200, 4},
    {200, 200, 200, 0},
    // NaN in the active block (info > 0 at once; only the isolated eigenvalues are computed)
    {10, 10, 10, 8},
    {120, 120, 120, 8},
    // batches whose matrices have different ranges ilo:ihi (GEHRD in groups), and a NaN
    // outside the active block (alone, and next to a matrix with a larger range)
    {20, 20, 20, 9},
    {120, 120, 120, 9},
    {4, 4, 4, 10},
    {4, 4, 4, 11},
    {100, 100, 100, 10},
    {100, 100, 100, 11},
    // strides larger than n or ld*n
    {33, 40, 35, 2, 7},
    {100, 100, 100, 7, 7},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {300, 300, 300, 0},
    {500, 512, 500, 3},
    {1000, 1000, 1000, 2},
    // (blocked GEHRD in groups)
    {600, 600, 600, 9},
};

// hybrid mode (a subset of the sizes)
const vector<vector<int>> hybrid_size_range = {
    {76, 76, 76, 0}, {100, 100, 100, 7}, {150, 150, 150, 3}, {250, 250, 250, 0}, {120, 120, 120, 9},
};

Arguments geev_setup_arguments(geev_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldvl", op[0] == 'N' ? 1 : matrix_size[2]);
    arg.set<rocblas_int>("ldvr", op[1] == 'N' ? 1 : matrix_size[2]);
    arg.set<rocblas_int>("mtype", matrix_size[3]);
    arg.set<char>("jobvl", op[0]);
    arg.set<char>("jobvr", op[1]);

    // strides: defaults (n or ld*n), or with a gap between the members
    if(matrix_size.size() > 4 && matrix_size[4] > 0)
    {
        const rocblas_int n = matrix_size[0], gap = matrix_size[4];
        const rocblas_stride ldvl = arg.peek<rocblas_int>("ldvl");
        const rocblas_stride ldvr = arg.peek<rocblas_int>("ldvr");
        arg.set<rocblas_stride>("strideA", rocblas_stride(matrix_size[1]) * n + gap);
        arg.set<rocblas_stride>("strideW", rocblas_stride(n) + gap);
        arg.set<rocblas_stride>("strideVL", ldvl * n + gap);
        arg.set<rocblas_stride>("strideVR", ldvr * n + gap);
    }

    arg.timing = 0;

    return arg;
}

template <rocblas_int MODE>
class GEEV_BASE : public ::TestWithParam<geev_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = geev_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("jobvl") == 'V'
           && arg.peek<char>("jobvr") == 'V')
            testing_geev_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        arg.alg_mode = MODE;
        testing_geev<BATCHED, STRIDED, T>(arg);
    }
};

class GEEV : public GEEV_BASE<0>
{
};

// hybrid mode: GEEV follows the algorithm mode of HSEQR
// (only matrices with n > 75, which use the multishift algorithm, are affected)
class GEEV_HYBRID : public GEEV_BASE<1>
{
};

// non-batch tests

TEST_P(GEEV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEEV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEEV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEEV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEEV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEEV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// hybrid mode tests

TEST_P(GEEV_HYBRID, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEEV_HYBRID, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GEEV_HYBRID, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEEV_HYBRID, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GEEV_HYBRID, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEEV_HYBRID, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEEV,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEV,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEV_HYBRID,
                         Combine(ValuesIn(hybrid_size_range), ValuesIn(op_range)));
