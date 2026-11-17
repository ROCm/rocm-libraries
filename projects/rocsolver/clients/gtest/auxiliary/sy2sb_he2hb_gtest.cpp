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

#include "common/auxiliary/testing_sy2sb_he2hb.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using sy2sb_he2hb_tuple = std::tuple<vector<I>, vector<I>>;

// each matrix_size_range is a {n, lda, uplo}
// if uplo = 0 then lower
// if uplo = 1 then upper
// if uplo = 2 then full

// each blk_range is a {kd, nb}

// case when n = 0, kd = 0, uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {10, 10, 0},
    {10, 15, 0},
    {20, 20, 0}, // lower
    {20, 20, 1}, // upper
    {20, 20, 2}}; // full

const vector<vector<int64_t>> size_range_64 = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {10, 10, 0},
    {10, 15, 0},
    {20, 20, 0}, // lower
    {20, 20, 1}, // upper
    {20, 20, 2}}; // full

const vector<vector<int>> blk_range = {
    // invalid
    {0, 1},
    {4, 2},
    // normal (valid) samples
    {1, 10},
    {10, 10},
    {2, 2},
    {2, 4},
    {2, 8},
    {3, 6},
    {3, 9}};

const vector<vector<int64_t>> blk_range_64 = {
    // invalid
    {0, 1},
    {4, 2},
    // normal (valid) samples
    {1, 10},
    {10, 10},
    {2, 2},
    {2, 4},
    {2, 8},
    {3, 6},
    {3, 9}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{256, 256, 0}, {640, 640, 0},   {640, 640, 1},
                                              {640, 640, 2}, {1024, 1024, 0}, {2048, 2048, 0}};

const vector<vector<int64_t>> large_size_range_64 = {
    {256, 256, 0}, {640, 640, 0}, {640, 640, 1}, {640, 640, 2}, {1024, 1024, 0}, {2048, 2048, 0}};

const vector<vector<int>> large_blk_range = {{16, 32}, {16, 64}, {32, 64}, {32, 128}};

const vector<vector<int64_t>> large_blk_range_64 = {{16, 32}, {16, 64}, {32, 64}, {32, 128}};

template <typename I>
Arguments sy2sb_he2hb_setup_arguments(sy2sb_he2hb_tuple<I> tup)
{
    vector<I> size = std::get<0>(tup);
    vector<I> blk = std::get<1>(tup);

    Arguments arg;

    arg.set<I>("n", size[0]);
    arg.set<I>("lda", size[1]);
    arg.set<I>("kd", blk[0]);
    arg.set<I>("nb", blk[1]);

    if(size[2] == 0)
        arg.set<char>("uplo", 'L');
    else if(size[2] == 1)
        arg.set<char>("uplo", 'U');
    else if(size[2] == 2)
        arg.set<char>("uplo", 'F');

    arg.timing = 0;

    return arg;
}

template <typename I>
class SY2SB_HE2HB_BASE : public ::TestWithParam<sy2sb_he2hb_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = sy2sb_he2hb_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0 && arg.peek<I>("kd") == 0 && arg.peek<char>("uplo") == 'L')
            testing_sy2sb_he2hb_bad_arg<T, I>();

        testing_sy2sb_he2hb<T, I>(arg);
    }
};

class SY2SB_HE2HB : public SY2SB_HE2HB_BASE<rocblas_int>
{
};

class SY2SB_HE2HB_64 : public SY2SB_HE2HB_BASE<int64_t>
{
};

// non-batch tests

TEST_P(SY2SB_HE2HB, __float)
{
    run_tests<float>();
}

TEST_P(SY2SB_HE2HB, __double)
{
    run_tests<double>();
}

TEST_P(SY2SB_HE2HB, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SY2SB_HE2HB, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(SY2SB_HE2HB_64, __float)
{
    run_tests<float>();
}

TEST_P(SY2SB_HE2HB_64, __double)
{
    run_tests<double>();
}

TEST_P(SY2SB_HE2HB_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(SY2SB_HE2HB_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SY2SB_HE2HB,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_blk_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SY2SB_HE2HB,
                         Combine(ValuesIn(size_range), ValuesIn(blk_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SY2SB_HE2HB_64,
                         Combine(ValuesIn(large_size_range_64), ValuesIn(large_blk_range_64)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SY2SB_HE2HB_64,
                         Combine(ValuesIn(size_range_64), ValuesIn(blk_range_64)));
