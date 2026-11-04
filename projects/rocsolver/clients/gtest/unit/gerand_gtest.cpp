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

#include "common/unit/testing_gerand.hpp"

using ::testing::TestWithParam;
using ::testing::ValuesIn;

// each size_range vector is a {M, N, lda}

// case when M == 0 also executes the bad arguments test

// for checkin_lapack tests
const std::vector<std::vector<int>> matrix_size_range = {
    // quick return
    {0, 10, 1},
    {10, 0, 10},
    // invalid
    {-1, 10, 1},
    {10, -1, 10},
    {10, 10, 5}, // lda < m, invalid
    // normal (valid) samples
    {12, 20, 12},
    {20, 15, 20},
    {35, 35, 50}, // lda > m (tests padding)
};

// for daily_lapack tests
const std::vector<std::vector<int>> large_matrix_size_range
    = {{192, 192, 192}, {640, 300, 700}, {1024, 2000, 1024}, {2547, 2547, 2550}};

Arguments gerand_setup_arguments(std::vector<int> matrix_size)
{
    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", matrix_size[1]);
    arg.set<rocblas_int>("lda", matrix_size[2]);

    arg.timing = 0;

    return arg;
}

class GERAND : public ::TestWithParam<std::vector<int>>
{
protected:
    void TearDown() override {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = gerand_setup_arguments(this->GetParam());

        if(arg.peek<rocblas_int>("m") == 0)
        {
            testing_gerand_bad_arg<T>();
        }

        testing_gerand<T>(arg);
    }
};

// non-batch tests (gerand is a pure host function, no batching)

TEST_P(GERAND, __float)
{
    run_tests<float>();
}

TEST_P(GERAND, __double)
{
    run_tests<double>();
}

TEST_P(GERAND, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GERAND, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GERAND, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GERAND, ValuesIn(matrix_size_range));
