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

#include "common/auxiliary/testing_stedc.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, printable_char> stedc_tuple;

// each size_range vector is a {N, ldc}

// each op_range vector is a {e}

// case when N == 0 and evect == N will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<printable_char> op_range = {'N', 'I', 'V'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    // invalid for case evect != N
    {2, 1},
    // normal (valid) samples
    {12, 12},
    {20, 30},
    {35, 40}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{192, 192}, {250, 250}, {256, 270}, {300, 300}};

Arguments stedc_setup_arguments(stedc_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    char op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldc", size[1]);

    arg.set<char>("evect", op);

    arg.timing = 0;

    return arg;
}

class STEDC : public ::TestWithParam<stedc_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = stedc_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N')
            testing_stedc_bad_arg<T>();

        testing_stedc<T>(arg);
    }
};

// non-batch tests

TEST_P(STEDC, __float)
{
    run_tests<float>();
}

TEST_P(STEDC, __double)
{
    run_tests<double>();
}

TEST_P(STEDC, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEDC, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEDC,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEDC,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));
