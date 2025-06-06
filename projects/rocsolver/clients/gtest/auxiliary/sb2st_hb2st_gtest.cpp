/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_sb2st_hb2st.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, printable_char> sb2st_hb2st_tuple;

// each matrix_size_range is a {n, lda, nb}

// case when n = 0 and nb = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1},
    {1, 1, 0},
    // invalid
    {-1, 1, 1},
    {1, 1, -1},
    {20, 15, 5},
    // normal (valid) samples
    {10, 10, 2},
    {20, 20, 20},
    {20, 24, 18},
    {128, 128, 32},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{152, 152, 152}, {640, 720, 64}, {1000, 1024, 128}, {512, 512, 312}};

Arguments sb2st_hb2st_setup_arguments(sb2st_hb2st_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("nb", matrix_size[2]);
    arg.set<char>("uplo", uplo);

    arg.timing = 0;

    return arg;
}

class SB2ST_HB2ST : public ::TestWithParam<sb2st_hb2st_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = sb2st_hb2st_setup_arguments(GetParam());

        //  if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("n") == 0)
        //      testing_sb2st_hb2st_bad_arg<T>();

        testing_sb2st_hb2st<T>(arg);
    }
};

// non-batch tests

TEST_P(SB2ST_HB2ST, __float)
{
    run_tests<float>();
}

TEST_P(SB2ST_HB2ST, __double)
{
    run_tests<double>();
}

TEST_P(SB2ST_HB2ST, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SB2ST_HB2ST, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SB2ST_HB2ST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SB2ST_HB2ST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
