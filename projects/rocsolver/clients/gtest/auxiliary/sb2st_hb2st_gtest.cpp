/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename I>
using sb2st_hb2st_tuple = std::tuple<vector<I>, printable_char>;

// each matrix_size_range is a {n, kd, ldab}
// ldab >= 3*kd - 1

// case when n = 0 and kd = 1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 2},
    // invalid
    {-1, 1, 2}, // n invalid
    {1, 0, 1}, // kd invalid
    {20, 5, 13}, // ldab invalid, ldab >= 14
    // normal (valid) samples
    {10, 2, 5}, // ldab >= 5
    {20, 20, 59}, // ldab >= 59
    {128, 32, 96}, // ldab >= 95
    {128, 40, 120}, // ldab >= 119
};

const vector<vector<int64_t>> matrix_size_range_64 = {
    // quick return
    {0, 1, 2},
    // invalid
    {-1, 1, 2}, // n invalid
    {1, 0, 1}, // kd invalid
    {20, 5, 13}, // ldab invalid, ldab >= 14
    // normal (valid) samples
    {10, 2, 5}, // ldab >= 5
    {20, 20, 59}, // ldab >= 59
    {128, 32, 96}, // ldab >= 95
    {128, 40, 120}, // ldab >= 119
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 64, 192}, // ldab >= 191
    {640, 64, 192}, // ldab >= 191
    {1000, 128, 384}, // ldab >= 383
    {512, 312, 960}, // ldab >= 935
};

const vector<vector<int64_t>> large_matrix_size_range_64 = {
    {152, 64, 192}, // ldab >= 191
    {640, 64, 192}, // ldab >= 191
    {1000, 128, 384}, // ldab >= 383
    {512, 312, 960}, // ldab >= 935
};

template <typename I>
Arguments sb2st_hb2st_setup_arguments(sb2st_hb2st_tuple<I> tup)
{
    vector<I> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", matrix_size[0]);
    arg.set<I>("kd", matrix_size[1]);
    arg.set<I>("ldab", matrix_size[2]);
    arg.set<char>("uplo", uplo);

    arg.timing = 0;

    return arg;
}

template <typename I>
class SB2ST_HB2ST_BASE : public ::TestWithParam<sb2st_hb2st_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = sb2st_hb2st_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<I>("kd") == 1)
            testing_sb2st_hb2st_bad_arg<T, I>();

        testing_sb2st_hb2st<T, I>(arg);
    }
};

class SB2ST_HB2ST : public SB2ST_HB2ST_BASE<rocblas_int>
{
};

class SB2ST_HB2ST_64 : public SB2ST_HB2ST_BASE<int64_t>
{
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

TEST_P(SB2ST_HB2ST_64, __float)
{
    run_tests<float>();
}

TEST_P(SB2ST_HB2ST_64, __double)
{
    run_tests<double>();
}

TEST_P(SB2ST_HB2ST_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SB2ST_HB2ST_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SB2ST_HB2ST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SB2ST_HB2ST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SB2ST_HB2ST_64,
                         Combine(ValuesIn(large_matrix_size_range_64), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SB2ST_HB2ST_64,
                         Combine(ValuesIn(matrix_size_range_64), ValuesIn(uplo_range)));
