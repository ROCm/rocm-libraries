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

#include "common/lapack/testing_sygsx_hegsx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> sygst_tuple;

// each matrix_size_range is a {n, lda, ldb}

// each type_range is a {itype, uplo}

// case when n = 0, itype = 1, and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> type_range
    = {{'1', 'U'}, {'1', 'L'}, {'2', 'U'}, {'2', 'L'}, {'3', 'U'}, {'3', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {20, 5, 5},
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 110},
    {130, 130, 130},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152, 152},
    {640, 640, 640},
    {1000, 1024, 1024},
};

Arguments sygst_setup_arguments(sygst_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<printable_char> type = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("uplo", type[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class SYGSX_HEGSX : public ::TestWithParam<sygst_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygst_setup_arguments(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("uplo") == 'U'
           && arg.peek<rocblas_int>("n") == 0)
            testing_sygsx_hegsx_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_sygsx_hegsx<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class SYGS2 : public SYGSX_HEGSX<false>
{
};

class HEGS2 : public SYGSX_HEGSX<false>
{
};

class SYGST : public SYGSX_HEGSX<true>
{
};

class HEGST : public SYGSX_HEGSX<true>
{
};

// non-batch tests

TEST_P(SYGS2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGS2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGS2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGS2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYGST, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGST, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGST, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGST, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYGS2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYGS2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEGS2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEGS2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(SYGST, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYGST, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEGST, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEGST, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYGS2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYGS2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEGS2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEGS2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(SYGST, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYGST, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEGST, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEGST, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGS2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGS2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGS2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGS2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
