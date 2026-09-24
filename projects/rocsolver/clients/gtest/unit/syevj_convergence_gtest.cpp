/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * *************************************************************************/

/* Regression tests for the Jacobi convergence criterion.
 *
 * The shared syevj/gesvdj test harness only generates diagonally dominant
 * matrices, which cannot reach these paths; each case below is built
 * explicitly. */

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_test.hpp"

using namespace std;

namespace
{

/* A = [[1, 0.5, 0], [0.5, 1, 0], [0, 0, K]], padded to n with unit diagonal.
   Exact singular values: K, 1.5, (n-3) ones, 0.5. */
template <typename T>
vector<T> dominant_axis_matrix(rocblas_int n, double K)
{
    vector<T> A(size_t(n) * n, T(0));
    for(rocblas_int i = 0; i < n; i++)
        A[i + i * n] = T(1);
    A[0 + 1 * n] = T(0.5);
    A[1 + 0 * n] = T(0.5);
    A[(n - 1) + (n - 1) * n] = T(K);
    return A;
}

vector<double> exact_sigma(rocblas_int n, double K)
{
    vector<double> s;
    s.push_back(K);
    s.push_back(1.5);
    for(rocblas_int i = 2; i < n - 1; i++)
        s.push_back(1.0);
    s.push_back(0.5);
    sort(s.begin(), s.end(), greater<double>());
    return s;
}

/* Runs gesvdj and returns the computed singular values, n_sweeps and info. */
template <typename T>
void run_gesvdj(rocblas_int n,
                const vector<T>& hA,
                vector<T>& hS,
                rocblas_int& n_sweeps,
                rocblas_int& info,
                rocblas_int max_sweeps = 100)
{
    rocblas_local_handle handle;

    device_strided_batch_vector<T> dA(size_t(n) * n, 1, size_t(n) * n, 1);
    device_strided_batch_vector<T> dS(n, 1, n, 1);
    device_strided_batch_vector<T> dU(size_t(n) * n, 1, size_t(n) * n, 1);
    device_strided_batch_vector<T> dV(size_t(n) * n, 1, size_t(n) * n, 1);
    device_strided_batch_vector<T> dres(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dsweeps(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dres.memcheck());
    CHECK_HIP_ERROR(dsweeps.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    CHECK_HIP_ERROR(hipMemcpy(dA.data(), hA.data(), sizeof(T) * hA.size(), hipMemcpyHostToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_gesvdj(false, handle, rocblas_svect_all, rocblas_svect_all, n, n,
                                         dA.data(), n, rocblas_stride(n) * n, T(0), dres.data(),
                                         max_sweeps, dsweeps.data(), dS.data(), rocblas_stride(n),
                                         dU.data(), n, rocblas_stride(n) * n, dV.data(), n,
                                         rocblas_stride(n) * n, dinfo.data(), 1));

    hS.resize(n);
    CHECK_HIP_ERROR(hipMemcpy(hS.data(), dS.data(), sizeof(T) * n, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&n_sweeps, dsweeps.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&info, dinfo.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost));
}

/* Runs syevj directly (no normal equations), so the input scale reaches the
   convergence test unsquared. */
template <typename T>
void run_syevj(rocblas_int n, const vector<T>& hA, vector<T>& hW, rocblas_int& n_sweeps, rocblas_int& info)
{
    rocblas_local_handle handle;

    device_strided_batch_vector<T> dA(size_t(n) * n, 1, size_t(n) * n, 1);
    device_strided_batch_vector<T> dW(n, 1, n, 1);
    device_strided_batch_vector<T> dres(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dsweeps(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dres.memcheck());
    CHECK_HIP_ERROR(dsweeps.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    CHECK_HIP_ERROR(hipMemcpy(dA.data(), hA.data(), sizeof(T) * hA.size(), hipMemcpyHostToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_syevj_heevj(
        false, handle, rocblas_esort_ascending, rocblas_evect_original, rocblas_fill_upper, n,
        dA.data(), n, rocblas_stride(n) * n, T(0), dres.data(), 100, dsweeps.data(), dW.data(),
        rocblas_stride(n), dinfo.data(), 1));

    hW.resize(n);
    CHECK_HIP_ERROR(hipMemcpy(hW.data(), dW.data(), sizeof(T) * n, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&n_sweeps, dsweeps.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&info, dinfo.data(), sizeof(rocblas_int), hipMemcpyDeviceToHost));
}

} // namespace

/* A single large entry must not mask an off-diagonal block that has not been
   rotated. Before the fix this returned after zero sweeps with the singular
   values of the unrotated matrix, sqrt(1.25) and 0.75/sqrt(1.25). */
TEST(checkin_lapack, SYEVJ_dominant_axis_does_not_mask_block)
{
    for(rocblas_int n : {3, 4, 8, 32, 59})
    {
        auto hA = dominant_axis_matrix<float>(n, 4096.0);
        auto exact = exact_sigma(n, 4096.0);

        vector<float> hS;
        rocblas_int n_sweeps = -1, info = -1;
        run_gesvdj<float>(n, hA, hS, n_sweeps, info);

        vector<double> got(hS.begin(), hS.end());
        sort(got.begin(), got.end(), greater<double>());

        EXPECT_EQ(info, 0) << "n = " << n;
        EXPECT_GT(n_sweeps, 0) << "n = " << n << ": the Jacobi loop never ran";
        for(rocblas_int i = 0; i < n; i++)
            EXPECT_NEAR(got[i], exact[i], exact[i] * 1e-5) << "n = " << n << ", sigma " << i;
    }
}

/* GESVDJ documents decreasing order; an unrotated matrix also breaks the sort,
   because SYEVJ orders by the diagonal it was handed. */
TEST(checkin_lapack, SYEVJ_dominant_axis_singular_values_are_sorted)
{
    const rocblas_int n = 5;
    auto hA = dominant_axis_matrix<float>(n, 4096.0);

    vector<float> hS;
    rocblas_int n_sweeps = -1, info = -1;
    run_gesvdj<float>(n, hA, hS, n_sweeps, info);

    for(rocblas_int i = 1; i < n; i++)
        EXPECT_LE(hS[i], hS[i - 1]) << "singular values not in decreasing order at " << i;
}

/* The small-size kernel used to test the sweep counter with a condition that
   was always true, so info never reported non-convergence for n <= 58. */
TEST(checkin_lapack, SYEVJ_reports_non_convergence)
{
    const rocblas_int n = 8;

    // dense and far from diagonal, so that one sweep cannot diagonalize it
    vector<float> hA(size_t(n) * n);
    for(rocblas_int i = 0; i < n; i++)
        for(rocblas_int j = 0; j < n; j++)
            hA[i + j * n] = 1.0f / float(i + j + 1);

    vector<float> hS;
    rocblas_int n_sweeps = -1, info = -1;

    // establish that this matrix really does need more than one sweep
    run_gesvdj<float>(n, hA, hS, n_sweeps, info);
    ASSERT_EQ(info, 0);
    ASSERT_GT(n_sweeps, 1) << "test matrix converges too fast to exercise the cap";

    run_gesvdj<float>(n, hA, hS, n_sweeps, info, 1);
    EXPECT_EQ(info, 1) << "capped at one sweep, info must report non-convergence";
}

/* The convergence test squares magnitudes on its cheap path, which halves the
   usable exponent range. Well-scaled matrices near the top of fp32 must still
   converge rather than be declared diagonal. */
TEST(checkin_lapack, SYEVJ_extreme_scale_still_converges)
{
    const rocblas_int n = 4;

    for(double scale : {1.0e0, 1.0e18, 1.0e24, 1.0e30, 1.0e37})
    {
        // diag = scale, one off-diagonal pair = scale/2; eigenvalues 0.5 and 1.5 times scale
        vector<float> hA(size_t(n) * n, 0.0f);
        for(rocblas_int i = 0; i < n; i++)
            hA[i + i * n] = float(scale);
        hA[0 + 1 * n] = float(0.5 * scale);
        hA[1 + 0 * n] = float(0.5 * scale);

        vector<float> hW;
        rocblas_int n_sweeps = -1, info = -1;
        run_syevj<float>(n, hA, hW, n_sweeps, info);

        EXPECT_EQ(info, 0) << "scale = " << scale;
        EXPECT_GT(n_sweeps, 0) << "scale = " << scale << ": the Jacobi loop never ran";
        EXPECT_NEAR(double(hW[0]), 0.5 * scale, 0.5 * scale * 1e-5) << "scale = " << scale;
    }
}
