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

#include "common/lapack/testing_gehd2_gehrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> gehrd_tuple;

// each matrix_size_range is a {n, lda, ilo, ihi}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1},
    // invalid
    {-1, 1, 1, 1},
    {20, 5, 1, 1},
    {20, 20, 1, 21},
    {20, 20, 0, 20},
    // normal (valid) samples
    {5, 5, 1, 5},
    // {50, 50},
    // {70, 100},
    // {130, 130},
    // {150, 200},
};

const vector<vector<int>> large_matrix_size_range = {
    {152, 152, 1, 152},
    {640, 640, 1, 640},
    {1000, 1024, 1, 1000},
};

Arguments gehrd_setup_arguments(gehrd_tuple tup)
{
    Arguments arg;

    arg.set<rocblas_int>("n", tup[0]);
    arg.set<rocblas_int>("lda", tup[1]);
    arg.set<rocblas_int>("ilo", tup[2]);
    arg.set<rocblas_int>("ihi", tup[3]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class GEHD2_GEHRD : public ::TestWithParam<gehrd_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gehrd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_gehd2_gehrd_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gehd2_gehrd<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class GEHD2 : public GEHD2_GEHRD<false>
{
};

class GEHRD : public GEHD2_GEHRD<true>
{
};

// non-batch tests

TEST_P(GEHD2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEHD2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEHD2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEHD2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GEHRD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEHRD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEHRD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEHRD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEHD2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEHD2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEHD2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEHD2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GEHRD, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEHRD, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEHRD, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEHRD, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEHD2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEHD2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEHD2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEHD2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GEHRD, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEHRD, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEHRD, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEHRD, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEHD2, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEHD2, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEHRD, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEHRD, ValuesIn(matrix_size_range));
