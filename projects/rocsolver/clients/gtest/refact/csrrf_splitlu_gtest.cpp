/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/refact/testing_csrrf_splitlu.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int> csrrf_splitlu_tuple;

// case when n = 0 and nnz = 0 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<int> n_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    20,
    50,
};
const vector<int> nnz_range = {
    // matrix zero
    0,
    // invalid
    -1,
    // normal (valid) samples
    60,
    100,
    140,
};

// for daily_lapack tests
const vector<int> large_n_range = {
    // normal (valid) samples
    100, 250, 3000, 8000};
const vector<int> large_nnz_range = {
    // normal (valid) samples
    300, 500, 700000, 2000000};

Arguments csrrf_splitlu_setup_arguments(csrrf_splitlu_tuple tup)
{
    int n = std::get<0>(tup);
    int nnz = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("nnzT", nnz);

    arg.timing = 0;

    return arg;
}

class CSRRF_SPLITLU : public ::TestWithParam<csrrf_splitlu_tuple>
{
protected:
    void SetUp() override
    {
        if(rocsolver_create_rfinfo(nullptr, nullptr) == rocblas_status_not_implemented)
            GTEST_SKIP() << "Sparse functionality is not enabled";
    }
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_splitlu_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nnzT") == 0)
            testing_csrrf_splitlu_bad_arg<T>();

        testing_csrrf_splitlu<T>(arg);
    }
};

// non-batch tests

TEST_P(CSRRF_SPLITLU, __float)
{
    run_tests<float>();
}

TEST_P(CSRRF_SPLITLU, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_SPLITLU, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_SPLITLU, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRRF_SPLITLU,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRRF_SPLITLU,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
