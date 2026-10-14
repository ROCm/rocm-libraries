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

#include "common/auxiliary/testing_orghr_unghr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using orghr_tuple = vector<I>;

// each size_range vector is a {n, lda, ilo, ihi}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 1},
    {5, 5, 0, 3},
    {5, 5, 3, 6},
    {5, 5, 3, 2},
    {5, 3, 2, 5},
    // normal (valid) samples
    {1, 1, 1, 1},
    {5, 5, 3, 3},
    {5, 5, 1, 5},
    {10, 10, 3, 10},
    {20, 25, 1, 20},
    {50, 50, 10, 45}};

const vector<vector<int64_t>> size_range_64 = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 1},
    {5, 5, 0, 3},
    {5, 5, 3, 6},
    {5, 5, 3, 2},
    {5, 3, 2, 5},
    // normal (valid) samples
    {1, 1, 1, 1},
    {5, 5, 3, 3},
    {5, 5, 1, 5},
    {10, 10, 3, 10},
    {20, 25, 1, 20},
    {50, 50, 10, 45}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{192, 192, 1, 192}, {500, 600, 50, 450}, {640, 640, 100, 540}, {1000, 1024, 1, 1000}};

const vector<vector<int64_t>> large_size_range_64
    = {{192, 192, 1, 192}, {500, 600, 50, 450}, {640, 640, 100, 540}, {1000, 1024, 1, 1000}};

template <typename I>
Arguments orghr_setup_arguments(orghr_tuple<I> tup)
{
    Arguments arg;

    arg.set<I>("n", tup[0]);
    arg.set<I>("lda", tup[1]);
    arg.set<I>("ilo", tup[2]);
    arg.set<I>("ihi", tup[3]);

    arg.timing = 0;

    return arg;
}

template <typename I>
class ORGHR_UNGHR : public ::TestWithParam<orghr_tuple<I>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = orghr_setup_arguments(this->GetParam());

        if(arg.peek<I>("n") == 0)
            testing_orghr_unghr_bad_arg<T, I>();

        testing_orghr_unghr<T, I>(arg);
    }
};

class ORGHR : public ORGHR_UNGHR<rocblas_int>
{
};

class UNGHR : public ORGHR_UNGHR<rocblas_int>
{
};

class ORGHR_64 : public ORGHR_UNGHR<int64_t>
{
};

class UNGHR_64 : public ORGHR_UNGHR<int64_t>
{
};

// non-batch tests

TEST_P(ORGHR, __float)
{
    run_tests<float>();
}

TEST_P(ORGHR, __double)
{
    run_tests<double>();
}

TEST_P(UNGHR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGHR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

// 64-bit API

TEST_P(ORGHR_64, __float)
{
    run_tests<float>();
}

TEST_P(ORGHR_64, __double)
{
    run_tests<double>();
}

TEST_P(UNGHR_64, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGHR_64, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORGHR, ValuesIn(large_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGHR, ValuesIn(size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNGHR, ValuesIn(large_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGHR, ValuesIn(size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, ORGHR_64, ValuesIn(large_size_range_64));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, ORGHR_64, ValuesIn(size_range_64));

INSTANTIATE_TEST_SUITE_P(daily_lapack, UNGHR_64, ValuesIn(large_size_range_64));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, UNGHR_64, ValuesIn(size_range_64));
