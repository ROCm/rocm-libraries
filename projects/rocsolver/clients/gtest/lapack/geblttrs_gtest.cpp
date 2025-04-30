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

#include "common/lapack/testing_geblttrs_npvt.hpp"
#include "common/lapack/testing_geblttrs_npvt_interleaved.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> geblttrs_tuple;

// each matrix_size_range vector is a {nb, nblocks, nrhs, lda, ldb, ldc, ldx}

// case when nb = 0, nblocks = 0, and nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1, 1, 1, 1},
    {1, 0, 1, 1, 1, 1, 1},
    {1, 1, 0, 1, 1, 1, 1},
    // invalid
    {-1, 1, 1, 1, 1, 1, 1},
    {1, -1, 1, 1, 1, 1, 1},
    {1, 1, -1, 1, 1, 1, 1},
    {10, 2, 1, 1, 1, 1, 1},
    // normal (valid) samples
    {32, 1, 10, 32, 32, 32, 32},
    {16, 2, 10, 20, 16, 16, 16},
    {10, 7, 20, 10, 20, 10, 10},
    {10, 10, 20, 10, 10, 20, 20},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{32, 6, 10, 32, 32, 32, 32},
                                                     {50, 10, 10, 60, 50, 50, 50},
                                                     {32, 10, 20, 32, 40, 32, 40},
                                                     {32, 20, 20, 32, 32, 40, 32}};

Arguments geblttrs_setup_arguments(geblttrs_tuple tup, bool interleaved)
{
    Arguments arg;

    arg.set<rocblas_int>("nb", tup[0]);
    arg.set<rocblas_int>("nblocks", tup[1]);
    arg.set<rocblas_int>("nrhs", tup[2]);

    if(!interleaved)
    {
        arg.set<rocblas_int>("lda", tup[3]);
        arg.set<rocblas_int>("ldb", tup[4]);
        arg.set<rocblas_int>("ldc", tup[5]);
        arg.set<rocblas_int>("ldx", tup[6]);
    }
    else
    {
        // normal use case is covered by non-interleaved tests
        rocblas_int bc = 3;

        arg.set<rocblas_int>("inca", bc);
        arg.set<rocblas_int>("incb", bc);
        arg.set<rocblas_int>("incc", bc);
        arg.set<rocblas_int>("incx", bc);

        arg.set<rocblas_int>("lda", bc * tup[3]);
        arg.set<rocblas_int>("ldb", bc * tup[4]);
        arg.set<rocblas_int>("ldc", bc * tup[5]);
        arg.set<rocblas_int>("ldx", bc * tup[6]);

        arg.set<rocblas_stride>("strideA", 1);
        arg.set<rocblas_stride>("strideB", 1);
        arg.set<rocblas_stride>("strideC", 1);
        arg.set<rocblas_stride>("strideX", 1);
    }

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GEBLTTRS_NPVT : public ::TestWithParam<geblttrs_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = geblttrs_setup_arguments(GetParam(), false);

        if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("nblocks") == 0
           && arg.peek<rocblas_int>("nrhs") == 0)
            testing_geblttrs_npvt_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_geblttrs_npvt<BATCHED, STRIDED, T>(arg);
    }
};

class GEBLTTRS_NPVT_INTERLEAVED : public ::TestWithParam<geblttrs_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = geblttrs_setup_arguments(GetParam(), true);

        if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("nblocks") == 0
           && arg.peek<rocblas_int>("nrhs") == 0)
            testing_geblttrs_npvt_interleaved_bad_arg<T>();

        arg.batch_count = 3;
        testing_geblttrs_npvt_interleaved<T>(arg);
    }
};

// non-batch tests

TEST_P(GEBLTTRS_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBLTTRS_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBLTTRS_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBLTTRS_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEBLTTRS_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEBLTTRS_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEBLTTRS_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEBLTTRS_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEBLTTRS_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEBLTTRS_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEBLTTRS_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEBLTTRS_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEBLTTRS_NPVT_INTERLEAVED, interleaved_batched__float)
{
    run_tests<float>();
}

TEST_P(GEBLTTRS_NPVT_INTERLEAVED, interleaved_batched__double)
{
    run_tests<double>();
}

TEST_P(GEBLTTRS_NPVT_INTERLEAVED, interleaved_batched__float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEBLTTRS_NPVT_INTERLEAVED, interleaved_batched__double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEBLTTRS_NPVT, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEBLTTRS_NPVT, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEBLTTRS_NPVT_INTERLEAVED, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEBLTTRS_NPVT_INTERLEAVED, ValuesIn(matrix_size_range));
