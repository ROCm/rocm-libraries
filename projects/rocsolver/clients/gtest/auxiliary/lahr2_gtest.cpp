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

#include "common/auxiliary/testing_lahr2.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// each size_range entry is {n, k, nb, lda, ldt, ldy}
// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> lahr2_size_range = {
    // quick return
    {0, 1, 1, 1, 1, 1},
    // invalid
    {-1, 1, 1, 5, 1, 5},
    {5, 0, 1, 5, 1, 5},
    {5, 1, 0, 5, 1, 5},
    {5, 5, 1, 5, 6, 5}, // k >= n
    {5, 1, 6, 5, 6, 5}, // nb >= n - k + 1
    {5, 1, 2, 4, 2, 5}, // lda < n
    {5, 1, 2, 5, 1, 5}, // ldt < nb
    {5, 1, 2, 5, 2, 4}, // ldy < n
    // normal (valid) samples
    {10, 1, 4, 10, 4, 10},
    {20, 2, 6, 20, 6, 20},
    {50, 5, 10, 50, 20, 50},
    {80, 10, 20, 80, 20, 100},
    {100, 10, 30, 100, 30, 120},
};

// for daily_lapack tests
const vector<vector<int>> lahr2_large_size_range = {
    {200, 10, 40, 200, 40, 200},
    {512, 20, 60, 512, 60, 512},
    {1000, 50, 100, 1024, 100, 1024},
};

Arguments lahr2_setup_arguments(vector<int> sz)
{
    Arguments arg;

    arg.set<rocblas_int>("n", sz[0]);
    arg.set<rocblas_int>("k", sz[1]);
    arg.set<rocblas_int>("nb", sz[2]);
    arg.set<rocblas_int>("lda", sz[3]);
    arg.set<rocblas_int>("ldt", sz[4]);
    arg.set<rocblas_int>("ldy", sz[5]);

    arg.timing = 0;

    return arg;
}

class LAHR2 : public ::TestWithParam<vector<int>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lahr2_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_lahr2_bad_arg<T>();

        testing_lahr2<T>(arg);
    }
};

// non-batch tests

TEST_P(LAHR2, __float)
{
    run_tests<float>();
}

TEST_P(LAHR2, __double)
{
    run_tests<double>();
}

TEST_P(LAHR2, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAHR2, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LAHR2, ValuesIn(lahr2_large_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LAHR2, ValuesIn(lahr2_size_range));
