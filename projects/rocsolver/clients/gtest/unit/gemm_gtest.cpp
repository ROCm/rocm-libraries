/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/unit/testing_gemm.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

template <typename I>
using gemm_tuple = tuple<vector<I>, I, vector<I>, char, char, vector<double>>;

// each m_size_range is a {m, lda, ldc}
// each k_size_range is a {k, ldb}

// for checkin_lapack tests
const vector<vector<int>> m_size_range = {
    // normal (valid) samples
    {50, 50, 50},
    {150, 200, 180}};

const vector<vector<int>> k_size_range = {
    // normal (valid) samples
    {50, 50},
    {150, 200}};

const vector<int> n_size_range = {16, 150};

const vector<vector<int64_t>> m_size_range_64 = {
    // normal (valid) samples
    {50, 50, 50},
    {150, 200, 180}};

const vector<vector<int64_t>> k_size_range_64 = {
    // normal (valid) samples
    {50, 50},
    {150, 200}};

const vector<int64_t> n_size_range_64 = {16, 150};

// for daily_lapack tests
const vector<vector<int>> large_m_size_range = {
    {1000, 1024, 1024},
};

const vector<vector<int>> large_k_size_range = {
    {1000, 1024},
};

const vector<int> large_n_size_range = {1000};

const vector<vector<int64_t>> large_m_size_range_64 = {
    {1000, 1024, 1024},
};

const vector<vector<int64_t>> large_k_size_range_64 = {
    {1000, 1024},
};

const vector<int64_t> large_n_size_range_64 = {1000};

const vector<char> all_operations = {
    'N',
    'T',
    'C',
};

const vector<vector<double>> alpha_beta = {{1.5, -2.0}};

template <typename T, typename I>
Arguments gemm_setup_arguments(gemm_tuple<I> tup)
{
    vector<I> m_size = std::get<0>(tup);
    I n_size = std::get<1>(tup);
    vector<I> k_size = std::get<2>(tup);
    char transA = std::get<3>(tup);
    char transB = std::get<4>(tup);
    vector<double> scalars = std::get<5>(tup);

    Arguments arg;

    arg.set<I>("m", m_size[0]);
    arg.set<I>("n", n_size);
    arg.set<I>("k", k_size[0]);
    arg.set<I>("lda", m_size[1]);
    arg.set<I>("ldb", k_size[1]);
    arg.set<I>("ldc", m_size[2]);
    arg.set<char>("transA", transA);
    arg.set<char>("transB", transB);
    arg.set<T>("alpha", scalars[0]);
    arg.set<T>("beta", scalars[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <typename I>
class GEMM_BASE : public ::TestWithParam<gemm_tuple<I>>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gemm_setup_arguments<T>(this->GetParam());

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gemm<BATCHED, STRIDED, T, I>(arg);
    }
};

class GEMM : public GEMM_BASE<rocblas_int>
{
};

class GEMM_64 : public GEMM_BASE<int64_t>
{
};

// non-batch tests

TEST_P(GEMM, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEMM, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEMM, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEMM, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEMM, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEMM, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEMM, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEMM, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEMM, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEMM, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEMM, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEMM, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// 64-bit API

// non-batch tests

TEST_P(GEMM_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEMM_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEMM_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEMM_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEMM_64, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEMM_64, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEMM_64, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEMM_64, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEMM_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEMM_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEMM_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEMM_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEMM,
                         Combine(ValuesIn(large_m_size_range),
                                 ValuesIn(large_n_size_range),
                                 ValuesIn(large_k_size_range),
                                 ValuesIn(all_operations),
                                 ValuesIn(all_operations),
                                 ValuesIn(alpha_beta)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEMM,
                         Combine(ValuesIn(m_size_range),
                                 ValuesIn(n_size_range),
                                 ValuesIn(k_size_range),
                                 ValuesIn(all_operations),
                                 ValuesIn(all_operations),
                                 ValuesIn(alpha_beta)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEMM_64,
                         Combine(ValuesIn(large_m_size_range_64),
                                 ValuesIn(large_n_size_range_64),
                                 ValuesIn(large_k_size_range_64),
                                 ValuesIn(all_operations),
                                 ValuesIn(all_operations),
                                 ValuesIn(alpha_beta)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEMM_64,
                         Combine(ValuesIn(m_size_range_64),
                                 ValuesIn(n_size_range_64),
                                 ValuesIn(k_size_range_64),
                                 ValuesIn(all_operations),
                                 ValuesIn(all_operations),
                                 ValuesIn(alpha_beta)));
