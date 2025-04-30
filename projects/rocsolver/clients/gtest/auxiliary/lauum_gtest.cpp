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

#include "common/auxiliary/testing_lauum.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<printable_char, vector<int>> lauum_tuple;

// each range is a {n,lda}

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 0},
    // normal (valid) samples
    {3, 3},
    {10, 10},
    {20, 20},
    {30, 30},
    {10, 20},
    {20, 30}};
const vector<vector<int>> large_size_range = {{100, 100}, {200, 200}};
const vector<printable_char> uplo_range = {'L', 'U'};

Arguments lauum_setup_arguments(lauum_tuple tup)
{
    char uplo = std::get<0>(tup);
    vector<int> size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("uplo", uplo);

    arg.timing = 0;

    return arg;
}

class LAUUM : public ::TestWithParam<lauum_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lauum_setup_arguments(GetParam());

        if((arg.peek<rocblas_int>("n") == 0) && (arg.peek<char>("uplo") == 'L'))
            testing_lauum_bad_arg<T>();

        testing_lauum<T>(arg);
    }
};

// non-batch tests

TEST_P(LAUUM, __float)
{
    run_tests<float>();
}

TEST_P(LAUUM, __double)
{
    run_tests<double>();
}

TEST_P(LAUUM, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LAUUM, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LAUUM,
                         Combine(ValuesIn(uplo_range), ValuesIn(large_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LAUUM, Combine(ValuesIn(uplo_range), ValuesIn(size_range)));
