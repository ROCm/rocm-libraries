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

#include "common/lapack/testing_getf2_getrf.hpp"
#include "common/lapack/testing_getf2_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {32, 32, 0},
    {50, 50, 1},
    {70, 100, 0}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    40,
    100,
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0},
    {640, 640, 1},
    {1000, 1024, 0},
};

const vector<int> large_n_size_range = {
    45, 64, 520, 1024, 2000,
};

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

template <bool BLOCKED, typename I>
class GETF2_GETRF : public ::TestWithParam<getrf_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_getf2_getrf_bad_arg<BATCHED, STRIDED, BLOCKED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getf2_getrf<BATCHED, STRIDED, BLOCKED, T, I>(arg);

        arg.singular = 0;
        testing_getf2_getrf<BATCHED, STRIDED, BLOCKED, T, I>(arg);
    }
};

template <bool BLOCKED, typename I>
class GETF2_GETRF_NPVT : public ::TestWithParam<getrf_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_getf2_getrf_npvt_bad_arg<BATCHED, STRIDED, BLOCKED, T, I>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getf2_getrf_npvt<BATCHED, STRIDED, BLOCKED, T, I>(arg);

        arg.singular = 0;
        testing_getf2_getrf_npvt<BATCHED, STRIDED, BLOCKED, T, I>(arg);
    }
};

class GETF2 : public GETF2_GETRF<false, rocblas_int>
{
};

class GETRF : public GETF2_GETRF<true, rocblas_int>
{
};

class GETF2_NPVT : public GETF2_GETRF_NPVT<false, rocblas_int>
{
};

class GETRF_NPVT : public GETF2_GETRF_NPVT<true, rocblas_int>
{
};

class GETF2_64 : public GETF2_GETRF<false, int64_t>
{
};

class GETRF_64 : public GETF2_GETRF<true, int64_t>
{
};

class GETF2_NPVT_64 : public GETF2_GETRF_NPVT<false, int64_t>
{
};

class GETRF_NPVT_64 : public GETF2_GETRF_NPVT<true, int64_t>
{
};

// non-batch tests
TEST_P(GETF2_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETF2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETF2_NPVT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2_NPVT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2_NPVT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_NPVT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_NPVT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETF2_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETF2_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETF2_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETF2_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRF_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests
TEST_P(GETF2_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETF2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETF2_NPVT_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2_NPVT_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2_NPVT_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF_NPVT_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF_NPVT_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETF2_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETF2_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETF2_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETF2_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRF_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases
TEST_P(GETF2_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETF2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETF2_NPVT_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2_NPVT_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2_NPVT_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2_NPVT_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF_NPVT_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF_NPVT_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF_NPVT_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF_NPVT_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETF2_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETF2_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETF2_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETF2_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRF_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2_NPVT_64,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2_NPVT_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT_64,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2_64,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_64,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
