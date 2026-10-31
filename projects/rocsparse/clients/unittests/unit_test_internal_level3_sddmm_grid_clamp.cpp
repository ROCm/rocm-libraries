/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

//
// Grid clamp / grid-stride coverage for rocsparse_sddmm (AISPARSE-673).
//
// Every sddmm launch now sizes grid.x through rocsparse::get_grid_size_x,
// which clamps the block count to min(handle->properties.maxGridSize[0],
// (2^32 - 1) / blockDim.x), and every kernel behind it grid-strides over
// grid.x. Reaching that clamp for real would need ~1.1e9 nonzeros (COO / COO
// AoS / ELL) or ~1.3e8 rows (CSR / CSC),
// so these tests shrink the limit on the handle instead and drive a SMALL
// problem through the same code path. The point is that the grid is smaller than
// the work, not that the work is large. This is the AISPARSE-699/700/702 idiom.
//
// The whole footprint is a 64x64 output with k <= 8 and at most two batches, a
// few tens of kilobytes. There is no device-memory guard and no size-based skip:
// every case runs on every GPU.
//
// TARGET: rocsparse-unit-test-device (needs a real device, the internal handle
// definition and the level-3 device headers).
//

#include "unit_test_utils.hpp"

#include "rocsparse.h"
#include "unit_test_grid_clamp.hpp"

// The dense-sample kernels that are driven directly below (library/src/level3 is
// on this target's include path).
#include "rocsparse_sddmm_csx_kernel.hpp"
#include "rocsparse_sddmm_ell_kernel.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_indextype  IT   = rocsparse_indextype_i32;
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;
    constexpr rocsparse_datatype   DT   = rocsparse_datatype_f32_r;

    // Small on purpose: the clamp is forced by shrinking maxGridSize[0], not by
    // making the problem big.
    constexpr int32_t test_m   = 64;
    constexpr int32_t test_n   = 64;
    constexpr int32_t test_k   = 4;
    constexpr int32_t test_nnz = test_m * test_n;

    // Written into C before the call. beta = 0 is applied by the kernel, not by a
    // memset, so any coefficient the grid fails to reach keeps this value and a
    // dropped block shows up as a sentinel rather than as a plausible number.
    constexpr float sentinel = -1.0f;

    using coord_list = std::vector<std::pair<int32_t, int32_t>>;

    // In batch b, A is m x k column major with A(i,0) = (b+1)*(i+1) and zero
    // elsewhere; B is k x n column major with B(0,j) = j+1 and zero elsewhere. So
    // batch b of A*B is exactly (b+1)*(i+1)*(j+1): distinct for every coefficient
    // and batch and exactly representable in float at these sizes, so a block that
    // samples the WRONG (i,j) or the wrong batch is caught just as reliably as one
    // that samples nothing.
    std::vector<float> host_A(int32_t k, int32_t batch_count)
    {
        std::vector<float> a(static_cast<size_t>(test_m) * k * batch_count, 0.0f);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            for(int32_t i = 0; i < test_m; ++i)
            {
                a[static_cast<size_t>(b) * test_m * k + i] = static_cast<float>((b + 1) * (i + 1));
            }
        }
        return a;
    }

    std::vector<float> host_B(int32_t k, int32_t batch_count)
    {
        std::vector<float> bm(static_cast<size_t>(k) * test_n * batch_count, 0.0f);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            for(int32_t j = 0; j < test_n; ++j)
            {
                bm[static_cast<size_t>(b) * k * test_n + static_cast<size_t>(j) * k]
                    = static_cast<float>(j + 1);
            }
        }
        return bm;
    }

    float expected_at(int32_t i, int32_t j, int32_t batch = 0)
    {
        return static_cast<float>(batch + 1) * static_cast<float>(i + 1)
               * static_cast<float>(j + 1);
    }

    std::vector<float> sentinel_values(int32_t batch_count = 1)
    {
        return std::vector<float>(static_cast<size_t>(test_nnz) * batch_count, sentinel);
    }

    // Every (i,j) of the 64x64 output is stored, so the checks cover it entirely.
    // Row major order: CSR, COO and COO AoS.
    coord_list row_major_coords()
    {
        coord_list c;
        c.reserve(test_nnz);
        for(int32_t i = 0; i < test_m; ++i)
        {
            for(int32_t j = 0; j < test_n; ++j)
            {
                c.emplace_back(i, j);
            }
        }
        return c;
    }

    // Column major order: CSC, and also ELL. ELL_IND_ROW places ell column el of
    // row i at el * m + i, and with a full ell_width = n the column index of that
    // entry is exactly el.
    coord_list column_major_coords()
    {
        coord_list c;
        c.reserve(test_nnz);
        for(int32_t j = 0; j < test_n; ++j)
        {
            for(int32_t i = 0; i < test_m; ++i)
            {
                c.emplace_back(i, j);
            }
        }
        return c;
    }

    // Run the full buffer_size -> preprocess -> compute pipeline for `matC` with
    // grid.x capped at `max_grid_x`, then check every stored coefficient of every
    // batch. `coords[idx]` is the (row, column) the idx-th stored value belongs to,
    // which is what makes this checker format agnostic. With batch_count > 1 the
    // caller has already set the strided batch on `matC`, whose values hold
    // batch_count runs of coords.size() entries.
    void run_and_check(rocsparse_handle      handle,
                       rocsparse_spmat_descr matC,
                       float*                c_val,
                       const coord_list&     coords,
                       int                   max_grid_x,
                       int32_t               k           = test_k,
                       int32_t               batch_count = 1)
    {
        device_vector<float> dA{host_A(k, batch_count)};
        device_vector<float> dB{host_B(k, batch_count)};
        ASSERT_TRUE(dA.ptr && dB.ptr);

        rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(
                      &mA, test_m, k, test_m, dA.ptr, DT, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mB, k, test_n, k, dB.ptr, DT, rocsparse_order_column),
            rocsparse_status_success);
        if(batch_count > 1)
        {
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(
                          mA, batch_count, static_cast<int64_t>(test_m) * k),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_dnmat_set_strided_batch(
                          mB, batch_count, static_cast<int64_t>(k) * test_n),
                      rocsparse_status_success);
        }

        const float alpha = 1.0f, beta = 0.0f;
        size_t      buffer_size = 0;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              &alpha,
                                              mA,
                                              mB,
                                              &beta,
                                              matC,
                                              DT,
                                              rocsparse_sddmm_alg_default,
                                              &buffer_size),
                  rocsparse_status_success);

        device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(tmp.ptr);

        ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             &alpha,
                                             mA,
                                             mB,
                                             &beta,
                                             matC,
                                             DT,
                                             rocsparse_sddmm_alg_default,
                                             tmp.ptr),
                  rocsparse_status_success);

        {
            const ScopedMaxGridSizeX cap(handle, max_grid_x);

            ASSERT_EQ(rocsparse_sddmm(handle,
                                      rocsparse_operation_none,
                                      rocsparse_operation_none,
                                      &alpha,
                                      mA,
                                      mB,
                                      &beta,
                                      matC,
                                      DT,
                                      rocsparse_sddmm_alg_default,
                                      tmp.ptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        const std::vector<float> got = to_host(c_val, coords.size() * batch_count);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            for(size_t idx = 0; idx < coords.size(); ++idx)
            {
                const int32_t i   = coords[idx].first;
                const int32_t j   = coords[idx].second;
                const float   val = got[b * coords.size() + idx];

                ASSERT_EQ(val, expected_at(i, j, b))
                    << "stored coefficient " << idx << " of batch " << b << " is C(" << i << ","
                    << j << "); k = " << k << ", grid.x was capped at " << max_grid_x
                    << ". A grid smaller than the work must be covered by the kernel's "
                       "grid-stride loop"
                    << (val == sentinel ? "; this coefficient was never written." : ".");
            }
        }

        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    }

    // With k = 4 the default-algorithm launches pick 128 nonzeros per block for
    // COO / COO AoS / ELL (4096 nonzeros -> 32 blocks); k = 1 and k = 8 pick 512
    // and 64 (8 and 64 blocks). CSR / CSC take 8 rows/columns per block for every
    // k (64 -> 8 blocks); k only changes the split inside the wavefront. So:
    //   - the device limit leaves the grid untouched (one sweep, the pre-existing
    //     behaviour, which must not change),
    //   - 3 blocks makes the sweep count ragged,
    //   - 1 block reaches everything past the first sweep only by striding.
    std::vector<int> grid_caps(rocsparse_handle handle)
    {
        return {handle->properties.maxGridSize[0], 3, 1};
    }

    // Full 64x64 CSR pattern, repeated batch_count times back to back.
    template <typename I, typename J>
    void check_csr(rocsparse_handle handle, int32_t k, int32_t batch_count)
    {
        std::vector<I> row_ptr(static_cast<size_t>(test_m + 1) * batch_count);
        std::vector<J> col_ind(static_cast<size_t>(test_nnz) * batch_count);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            for(int32_t i = 0; i <= test_m; ++i)
            {
                row_ptr[static_cast<size_t>(b) * (test_m + 1) + i] = static_cast<I>(i) * test_n;
            }
            for(int32_t i = 0; i < test_m; ++i)
            {
                for(int32_t j = 0; j < test_n; ++j)
                {
                    col_ind[static_cast<size_t>(b) * test_nnz + i * test_n + j] = j;
                }
            }
        }

        for(int cap : grid_caps(handle))
        {
            SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);

            device_vector<I>     d_ptr{row_ptr};
            device_vector<J>     d_ind{col_ind};
            device_vector<float> d_val{sentinel_values(batch_count)};
            ASSERT_TRUE(d_ptr.ptr && d_ind.ptr && d_val.ptr);

            rocsparse_spmat_descr matC = nullptr;
            ASSERT_EQ(rocsparse_create_csr_descr(&matC,
                                                 test_m,
                                                 test_n,
                                                 test_nnz,
                                                 d_ptr,
                                                 d_ind,
                                                 d_val,
                                                 it_of<I>(),
                                                 it_of<J>(),
                                                 BASE,
                                                 DT),
                      rocsparse_status_success);
            if(batch_count > 1)
            {
                ASSERT_EQ(rocsparse_csr_set_strided_batch(matC, batch_count, test_m + 1, test_nnz),
                          rocsparse_status_success);
            }
            ASSERT_NO_FATAL_FAILURE(
                run_and_check(handle, matC, d_val.ptr, row_major_coords(), cap, k, batch_count));
            EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
        }
    }

    // Full 64x64 COO pattern in row major order, repeated batch_count times back
    // to back (COO has a single batch stride for indices and values).
    template <typename I>
    void check_coo(rocsparse_handle handle, int32_t k, int32_t batch_count)
    {
        std::vector<I> rows(static_cast<size_t>(test_nnz) * batch_count);
        std::vector<I> cols(static_cast<size_t>(test_nnz) * batch_count);
        for(int32_t b = 0; b < batch_count; ++b)
        {
            for(int32_t i = 0; i < test_m; ++i)
            {
                for(int32_t j = 0; j < test_n; ++j)
                {
                    rows[static_cast<size_t>(b) * test_nnz + i * test_n + j] = i;
                    cols[static_cast<size_t>(b) * test_nnz + i * test_n + j] = j;
                }
            }
        }

        for(int cap : grid_caps(handle))
        {
            SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);

            device_vector<I>     d_row{rows}, d_col{cols};
            device_vector<float> d_val{sentinel_values(batch_count)};
            ASSERT_TRUE(d_row.ptr && d_col.ptr && d_val.ptr);

            rocsparse_spmat_descr matC = nullptr;
            ASSERT_EQ(
                rocsparse_create_coo_descr(
                    &matC, test_m, test_n, test_nnz, d_row, d_col, d_val, it_of<I>(), BASE, DT),
                rocsparse_status_success);
            if(batch_count > 1)
            {
                ASSERT_EQ(rocsparse_coo_set_strided_batch(matC, batch_count, test_nnz),
                          rocsparse_status_success);
            }
            ASSERT_NO_FATAL_FAILURE(
                run_and_check(handle, matC, d_val.ptr, row_major_coords(), cap, k, batch_count));
            EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
        }
    }
}

class SddmmGridClamp : public HandleTest
{
};

// ---------------------------------------------------------------------------
// End to end, all five formats, with grid.x capped below what the work needs.
// ---------------------------------------------------------------------------

TEST_F(SddmmGridClamp, csr)
{
    check_csr<int32_t, int32_t>(handle, test_k, 1);
}

TEST_F(SddmmGridClamp, csc)
{
    std::vector<int32_t> col_ptr(test_n + 1), row_ind(test_nnz);
    for(int32_t j = 0; j <= test_n; ++j)
    {
        col_ptr[j] = j * test_m;
    }
    for(int32_t j = 0; j < test_n; ++j)
    {
        for(int32_t i = 0; i < test_m; ++i)
        {
            row_ind[j * test_m + i] = i;
        }
    }

    for(int cap : grid_caps(handle))
    {
        SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);

        device_vector<int32_t> d_ptr{col_ptr}, d_ind{row_ind};
        device_vector<float>   d_val{sentinel_values()};
        ASSERT_TRUE(d_ptr.ptr && d_ind.ptr && d_val.ptr);

        rocsparse_spmat_descr matC = nullptr;
        ASSERT_EQ(rocsparse_create_csc_descr(
                      &matC, test_m, test_n, test_nnz, d_ptr, d_ind, d_val, IT, IT, BASE, DT),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(run_and_check(handle, matC, d_val.ptr, column_major_coords(), cap));
        EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    }
}

TEST_F(SddmmGridClamp, coo)
{
    check_coo<int32_t>(handle, test_k, 1);
}

TEST_F(SddmmGridClamp, coo_aos)
{
    // Array-of-structs COO: interleaved (row, column) pairs.
    std::vector<int32_t> ind(2 * static_cast<size_t>(test_nnz));
    for(int32_t i = 0; i < test_m; ++i)
    {
        for(int32_t j = 0; j < test_n; ++j)
        {
            ind[2 * static_cast<size_t>(i * test_n + j)]     = i;
            ind[2 * static_cast<size_t>(i * test_n + j) + 1] = j;
        }
    }

    for(int cap : grid_caps(handle))
    {
        SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);

        device_vector<int32_t> d_ind{ind};
        device_vector<float>   d_val{sentinel_values()};
        ASSERT_TRUE(d_ind.ptr && d_val.ptr);

        rocsparse_spmat_descr matC = nullptr;
        ASSERT_EQ(rocsparse_create_coo_aos_descr(
                      &matC, test_m, test_n, test_nnz, d_ind, d_val, IT, BASE, DT),
                  rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(run_and_check(handle, matC, d_val.ptr, row_major_coords(), cap));
        EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    }
}

TEST_F(SddmmGridClamp, ell)
{
    // Full ell_width = n, so nnz = m * n and every coefficient is stored.
    std::vector<int32_t> ell_ind(test_nnz);
    for(int32_t el = 0; el < test_n; ++el)
    {
        for(int32_t i = 0; i < test_m; ++i)
        {
            ell_ind[el * test_m + i] = el;
        }
    }

    for(int cap : grid_caps(handle))
    {
        SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);

        device_vector<int32_t> d_ind{ell_ind};
        device_vector<float>   d_val{sentinel_values()};
        ASSERT_TRUE(d_ind.ptr && d_val.ptr);

        rocsparse_spmat_descr matC = nullptr;
        ASSERT_EQ(
            rocsparse_create_ell_descr(&matC, test_m, test_n, d_ind, d_val, test_n, IT, BASE, DT),
            rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(run_and_check(handle, matC, d_val.ptr, column_major_coords(), cap));
        EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    }
}

// ---------------------------------------------------------------------------
// COO and CSR with 64-bit indices, other dot-product widths, and a strided
// batch, so the grid-stride loop also runs inside the hipBlockIdx_y batch loop.
// ---------------------------------------------------------------------------

TEST_F(SddmmGridClamp, csr_i64_i32)
{
    check_csr<int64_t, int32_t>(handle, test_k, 1);
}

TEST_F(SddmmGridClamp, csr_i64_i64)
{
    check_csr<int64_t, int64_t>(handle, test_k, 1);
}

TEST_F(SddmmGridClamp, csr_k1)
{
    check_csr<int32_t, int32_t>(handle, 1, 1);
}

TEST_F(SddmmGridClamp, csr_k8)
{
    check_csr<int32_t, int32_t>(handle, 8, 1);
}

TEST_F(SddmmGridClamp, csr_batch2)
{
    check_csr<int32_t, int32_t>(handle, test_k, 2);
}

TEST_F(SddmmGridClamp, coo_i64)
{
    check_coo<int64_t>(handle, test_k, 1);
}

TEST_F(SddmmGridClamp, coo_k1)
{
    check_coo<int32_t>(handle, 1, 1);
}

TEST_F(SddmmGridClamp, coo_k8)
{
    check_coo<int32_t>(handle, 8, 1);
}

TEST_F(SddmmGridClamp, coo_batch2)
{
    check_coo<int32_t>(handle, test_k, 2);
}

// ---------------------------------------------------------------------------
// The CSR/CSC dense-sample kernel, driven directly.
//
// sddmm_csx_sample_kernel is only reached through rocsparse_sddmm_alg_dense,
// which routes the multiply through rocBLAS and is therefore unreachable in a
// --no-rocblas build. The kernel is launchable on its own, so these cases call
// it with a deliberately undersized grid, which is exactly the situation the
// clamp in rocsparse_sddmm_{csr,csc}.cpp creates.
// ---------------------------------------------------------------------------

namespace
{
    constexpr rocsparse_int smpl_nb     = 512;
    constexpr rocsparse_int smpl_groups = 32; // NTHREADS_PER_GROUP
    // 512/32 = 16 groups per block, so each block covers 16 rows or columns.
    constexpr int32_t smpl_full_grid = (test_m + 16 - 1) / 16;

    // Launch the sample kernel the way rocsparse_sddmm_csc.cpp does: M = m, N = n,
    // csx_ptr = column offsets, csx_ind = row indices, and the dense buffer is
    // m x n column major with leading dimension m.
    std::vector<float> run_csx_sample_csc_shape(int32_t grid_x)
    {
        std::vector<int32_t> col_ptr(test_n + 1), row_ind(test_nnz);
        std::vector<float>   dense(static_cast<size_t>(test_m) * test_n);
        for(int32_t j = 0; j <= test_n; ++j)
        {
            col_ptr[j] = j * test_m;
        }
        for(int32_t j = 0; j < test_n; ++j)
        {
            for(int32_t i = 0; i < test_m; ++i)
            {
                row_ind[j * test_m + i]                    = i;
                dense[static_cast<size_t>(j) * test_m + i] = expected_at(i, j);
            }
        }

        device_vector<int32_t> d_ptr{col_ptr}, d_ind{row_ind};
        device_vector<float>   d_dense{dense};
        device_vector<float>   d_val{sentinel_values()};
        if(!d_ptr.ptr || !d_ind.ptr || !d_dense.ptr || !d_val.ptr)
        {
            return {};
        }

        hipLaunchKernelGGL(
            (rocsparse::
                 sddmm_csx_sample_kernel<smpl_nb, smpl_groups, rocsparse_direction_column, float>),
            dim3(static_cast<uint32_t>(grid_x)),
            dim3(smpl_nb),
            0,
            0,
            test_m,
            test_n,
            test_nnz,
            d_dense.ptr,
            test_m,
            d_val.ptr,
            d_ptr.ptr,
            d_ind.ptr,
            BASE);

        if(hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        {
            return {};
        }
        return to_host(d_val.ptr, test_nnz);
    }

    // And the way rocsparse_sddmm_csr.cpp does: the arguments are swapped (M = n,
    // N = m, lda = n) so that the same direction_column instantiation walks the
    // ROWS of a CSR matrix over a dense buffer holding C transposed.
    std::vector<float> run_csx_sample_csr_shape(int32_t grid_x)
    {
        std::vector<int32_t> row_ptr(test_m + 1), col_ind(test_nnz);
        std::vector<float>   dense(static_cast<size_t>(test_m) * test_n);
        for(int32_t i = 0; i <= test_m; ++i)
        {
            row_ptr[i] = i * test_n;
        }
        for(int32_t i = 0; i < test_m; ++i)
        {
            for(int32_t j = 0; j < test_n; ++j)
            {
                col_ind[i * test_n + j] = j;
                // n x m column major, i.e. C transposed, which is what the swapped
                // gemm in rocsparse_sddmm_csr.cpp produces.
                dense[static_cast<size_t>(i) * test_n + j] = expected_at(i, j);
            }
        }

        device_vector<int32_t> d_ptr{row_ptr}, d_ind{col_ind};
        device_vector<float>   d_dense{dense};
        device_vector<float>   d_val{sentinel_values()};
        if(!d_ptr.ptr || !d_ind.ptr || !d_dense.ptr || !d_val.ptr)
        {
            return {};
        }

        hipLaunchKernelGGL(
            (rocsparse::
                 sddmm_csx_sample_kernel<smpl_nb, smpl_groups, rocsparse_direction_column, float>),
            dim3(static_cast<uint32_t>(grid_x)),
            dim3(smpl_nb),
            0,
            0,
            test_n,
            test_m,
            test_nnz,
            d_dense.ptr,
            test_n,
            d_val.ptr,
            d_ptr.ptr,
            d_ind.ptr,
            BASE);

        if(hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        {
            return {};
        }
        return to_host(d_val.ptr, test_nnz);
    }

    void expect_sampled(const std::vector<float>& got, const coord_list& coords, int32_t grid_x)
    {
        ASSERT_EQ(got.size(), static_cast<size_t>(test_nnz));
        for(size_t idx = 0; idx < coords.size(); ++idx)
        {
            ASSERT_EQ(got[idx], expected_at(coords[idx].first, coords[idx].second))
                << "stored coefficient " << idx << " is C(" << coords[idx].first << ","
                << coords[idx].second << ") and was not sampled with grid.x = " << grid_x
                << " (the full grid is " << smpl_full_grid << " blocks).";
        }
    }
}

TEST(internal_level3_sddmm_sample, csc_shape_full_grid)
{
    ASSERT_NO_FATAL_FAILURE(expect_sampled(
        run_csx_sample_csc_shape(smpl_full_grid), column_major_coords(), smpl_full_grid));
}

TEST(internal_level3_sddmm_sample, csc_shape_single_block)
{
    ASSERT_NO_FATAL_FAILURE(expect_sampled(run_csx_sample_csc_shape(1), column_major_coords(), 1));
}

TEST(internal_level3_sddmm_sample, csr_shape_full_grid)
{
    ASSERT_NO_FATAL_FAILURE(expect_sampled(
        run_csx_sample_csr_shape(smpl_full_grid), row_major_coords(), smpl_full_grid));
}

TEST(internal_level3_sddmm_sample, csr_shape_single_block)
{
    ASSERT_NO_FATAL_FAILURE(expect_sampled(run_csx_sample_csr_shape(1), row_major_coords(), 1));
}

// ---------------------------------------------------------------------------
// The ELL dense-sample launch, driven through sddmm_ell_sample_launch.
//
// Like the CSR/CSC sample, it is only reached through rocsparse_sddmm_alg_dense,
// which needs rocBLAS. sddmm_ell_sample_launch is the host code
// rocsparse_sddmm_ell.cpp calls after the gemm: it clamps grid.x with
// get_grid_size_x and picks the grid-stride kernel only when the clamp binds.
// With ell_width = n = 64 and 16 ell columns per block the full grid is 4
// blocks, so the device limit takes the straight-line kernel and the caps of 3
// and 1 take the grid-stride one.
// ---------------------------------------------------------------------------

namespace
{
    template <rocsparse_int WF_SIZE>
    void check_ell_sample(rocsparse_handle handle, int max_grid_x)
    {
        std::vector<int32_t> ell_ind(test_nnz);
        std::vector<float>   dense(static_cast<size_t>(test_m) * test_n);
        for(int32_t el = 0; el < test_n; ++el)
        {
            for(int32_t i = 0; i < test_m; ++i)
            {
                ell_ind[el * test_m + i]                    = el;
                dense[static_cast<size_t>(el) * test_m + i] = expected_at(i, el);
            }
        }

        device_vector<int32_t> d_ind{ell_ind};
        device_vector<float>   d_dense{dense};
        device_vector<float>   d_val{sentinel_values()};
        ASSERT_TRUE(d_ind.ptr && d_dense.ptr && d_val.ptr);

        {
            const ScopedMaxGridSizeX cap(handle, max_grid_x);

            (void)hipGetLastError();
            ASSERT_EQ((rocsparse::sddmm_ell_sample_launch<16, WF_SIZE, float>(handle,
                                                                              test_m,
                                                                              test_n,
                                                                              d_dense.ptr,
                                                                              test_m,
                                                                              test_n,
                                                                              d_val.ptr,
                                                                              d_ind.ptr,
                                                                              BASE)),
                      rocsparse_status_success);
            ASSERT_EQ(hipGetLastError(), hipSuccess);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        const std::vector<float> got    = to_host(d_val.ptr, test_nnz);
        const coord_list         coords = column_major_coords();
        for(size_t idx = 0; idx < coords.size(); ++idx)
        {
            ASSERT_EQ(got[idx], expected_at(coords[idx].first, coords[idx].second))
                << "stored coefficient " << idx << " is C(" << coords[idx].first << ","
                << coords[idx].second << "); grid.x was capped at " << max_grid_x << " (WF_SIZE "
                << WF_SIZE << ", the full grid is 4 blocks)"
                << (got[idx] == sentinel ? "; this coefficient was never written." : ".");
        }
    }
}

TEST_F(SddmmGridClamp, ell_dense_sample_wf32)
{
    for(int cap : grid_caps(handle))
    {
        SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);
        ASSERT_NO_FATAL_FAILURE(check_ell_sample<32>(handle, cap));
    }
}

TEST_F(SddmmGridClamp, ell_dense_sample_wf64)
{
    for(int cap : grid_caps(handle))
    {
        SCOPED_TRACE(testing::Message() << "grid.x capped at " << cap);
        ASSERT_NO_FATAL_FAILURE(check_ell_sample<64>(handle, cap));
    }
}
