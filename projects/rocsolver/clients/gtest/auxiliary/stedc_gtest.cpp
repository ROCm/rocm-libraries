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
        ASSERT_EQ(hipGetLastError(), hipSuccess);
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

// scale invariance

// Scaling a tridiagonal matrix by an exact power of two must scale its eigenvalues by the
// same factor and leave its eigenvectors alone. STEDC normalizes before the divide, so
// both runs below reduce to the identical problem and the results must agree exactly.
//
// n must be at least STEDC_MIN_DC_SIZE, or STEDC defers to STEQR and the divide-and-conquer
// path under test never runs.
template <typename T>
void stedc_scale_invariance(const rocblas_int n, const int expo)
{
    using S = decltype(std::real(T{}));

    rocblas_local_handle handle;
    const rocblas_evect evect = rocblas_evect_tridiagonal;
    const rocblas_int ldc = n;
    const size_t size_C = size_t(ldc) * n;

    host_strided_batch_vector<S> hD(n, 1, n, 1);
    host_strided_batch_vector<S> hE(n, 1, n, 1);
    host_strided_batch_vector<S> hDScaled(n, 1, n, 1);
    host_strided_batch_vector<S> hEScaled(n, 1, n, 1);
    host_strided_batch_vector<S> hDRef(n, 1, n, 1);
    host_strided_batch_vector<S> hDRes(n, 1, n, 1);
    host_strided_batch_vector<T> hCRef(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hCRes(size_C, 1, size_C, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);

    device_strided_batch_vector<S> dD(n, 1, n, 1);
    device_strided_batch_vector<S> dE(n, 1, n, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    rocblas_init<S>(hD, true);
    rocblas_init<S>(hE, true);
    for(rocblas_int i = 0; i < n - 1; ++i)
        hE[0][i] -= 4;
    hE[0][n - 1] = 0;

    for(rocblas_int i = 0; i < n; ++i)
    {
        hDScaled[0][i] = std::ldexp(hD[0][i], -expo);
        hEScaled[0][i] = std::ldexp(hE[0][i], -expo);
    }

    CHECK_HIP_ERROR(dD.transfer_from(hD));
    CHECK_HIP_ERROR(dE.transfer_from(hE));
    CHECK_ROCBLAS_ERROR(
        rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRef.transfer_from(dD));
    CHECK_HIP_ERROR(hCRef.transfer_from(dC));
    CHECK_HIP_ERROR(hInfo.transfer_from(dInfo));
    ASSERT_EQ(hInfo[0][0], 0);

    // same matrix, scaled down by 2^expo
    CHECK_HIP_ERROR(dD.transfer_from(hDScaled));
    CHECK_HIP_ERROR(dE.transfer_from(hEScaled));
    CHECK_ROCBLAS_ERROR(
        rocsolver_stedc(handle, evect, n, dD.data(), dE.data(), dC.data(), ldc, dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hInfo.transfer_from(dInfo));
    ASSERT_EQ(hInfo[0][0], 0);

    for(rocblas_int i = 0; i < n; ++i)
        ASSERT_EQ(std::ldexp(hDRes[0][i], expo), hDRef[0][i]) << "eigenvalue " << i;

    for(size_t i = 0; i < size_C; ++i)
        ASSERT_EQ(hCRes[0][i], hCRef[0][i]) << "eigenvector entry " << i;
}

typedef std::tuple<int, int> stedc_scale_tuple;

// sizes are at least STEDC_MIN_DC_SIZE so that the divide-and-conquer path is used
const vector<int> scale_size_range = {16, 35, 64};
const vector<int> large_scale_size_range = {192, 300};

// exponents to scale the matrix down by
const vector<int> scale_exponent_range = {10, 30};

class STEDC_SCALE : public ::TestWithParam<stedc_scale_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        stedc_scale_invariance<T>(std::get<0>(GetParam()), std::get<1>(GetParam()));
    }
};

TEST_P(STEDC_SCALE, __float)
{
    run_tests<float>();
}

TEST_P(STEDC_SCALE, __double)
{
    run_tests<double>();
}

TEST_P(STEDC_SCALE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEDC_SCALE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEDC_SCALE,
                         Combine(ValuesIn(large_scale_size_range), ValuesIn(scale_exponent_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEDC_SCALE,
                         Combine(ValuesIn(scale_size_range), ValuesIn(scale_exponent_range)));
