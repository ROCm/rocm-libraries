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
// Regression tests for the conversion-directory grid arithmetic hardened by
// AISPARSE-684, AISPARSE-685 and AISPARSE-686.
//
//   684  rocsparse_gebsr2csr_nnz        (rocsparse_gebsr2gebsr.cpp,
//                                        gebsr2csr_device.h)
//   685  csr2csc permute, csr2coo       (rocsparse_csr2csc.cpp, csr2csc_device.h,
//                                        rocsparse_csr2coo.cpp, csr2coo_device.h)
//   686  convert_array, extract,        (rocsparse_convert_array.cpp,
//        create_identity_permutation     rocsparse_extract_alg_default.cpp,
//                                        rocsparse_identity.cpp,
//                                        identity_device.h)
//
// FOCUS. Every site above sized grid.x from a length it could not express, and
// then had nothing behind the grid it launched:
//
//   684 formed `mb * row_block_dim` in rocsparse_int on BOTH sides -- once on the
//       host to size the grid and once in the kernel as the bound the grid was
//       checked against -- so above INT_MAX rows the product overflowed (signed,
//       so undefined behaviour), the launch was undersized and the guard was
//       corrupted by the same wrong value. The CSR row pointer and column index
//       arrays were left partially filled with no error reported.
//
//   685 csr2csc sized grid.x correctly from an int64_t nnz and then computed the
//       kernel's global id into a rocsparse_int. csr2coo did the arithmetic
//       correctly in int64_t -- that cast is the reference idiom of this epic --
//       and then assigned the result into a dim3, which narrows back to unsigned
//       int, and indexed rows with hipBlockIdx_x alone.
//
//   686 narrowed block counts formed from a size_t or an int64_t index type into
//       a dim3 with no clamp, and computed every global id as
//       `hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x`, which is unsigned-int
//       arithmetic and wraps at 2^32 whatever it is assigned to.
//
// WHY THE THRESHOLDS ARE NOT TESTED DIRECTLY. 684 needs mb * row_block_dim above
// 2^31 (2.1 billion CSR rows), 685 needs 2.1e9 non-zeros, 686 needs 1.1e12 to
// 2.2e12 elements. None of that fits the 15 GB gfx1201 this runs on, and 686 in
// particular is unreachable on any machine, which is why it is P3. What IS
// reachable -- and what the fixes actually consist of -- is the pairing of
//
//   (a) a grid.x clamped against handle->properties.maxGridSize[0], and
//   (b) a grid-stride loop that covers the length the clamp dropped.
//
// Every case below shrinks maxGridSize[0] to 3 blocks with ScopedMaxGridSizeX so
// the real launch is clamped far below what the length asks for, and then checks
// that the whole length was still processed. Remove a stride loop and the case
// covering it fails; remove a clamp and the launch asks for a grid the device
// limit forbids. The 64-bit widening of the products and indices themselves is
// not separately observable at these sizes -- it is the precondition that lets a
// stride loop address past the wrap at all -- so it is carried by review and by
// the -fsanitize=undefined run recorded on AISPARSE-684, not by a test.
//
// There is deliberately NO device-memory guard and no size-based skip anywhere in
// this file: the largest allocation is 10007 int64_t (80 kB), so every case runs
// on every GPU.
//
// TARGET. The csr2coo / csr2csc / identity / extract / gebsr2gebsr entry points
// are driven through the public C API, which reaches the real library kernels.
// rocsparse::convert_array is internal, but rocsparse_convert_array.cpp is
// already compiled into this binary (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES), so
// it can be called directly. This is the GPU binary, not the host-only
// rocsparse-unit-test.
//
#include "unit_test_utils.hpp"

#include "unit_test_conversion_grid_clamp.hpp"

// Internal declaration of the one helper under test with no public entry point.
// Relative path because library/src/conversion is not on this target's include
// path; the same idiom is already used by unit_test_internal_collective_extras_*.
#include "../../library/src/conversion/rocsparse_convert_array.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace rocsparse_ut;
namespace
{
    // A grid.x limit far below what any length below asks for, so the great
    // majority of the work is reached only by the grid-stride loop.
    constexpr int clamped_grid_x = 3;

    using ConversionGrids = HandleTest;

    // First index whose value differs from the expectation, or -1 if all match.
    // One precise gtest failure instead of thousands of EXPECT_EQ macros.
    template <typename T>
    int64_t first_mismatch(const std::vector<T>& got, const std::vector<T>& expected)
    {
        if(got.size() != expected.size())
        {
            return 0;
        }
        for(size_t i = 0; i < got.size(); ++i)
        {
            if(got[i] != expected[i])
            {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
    }
}
// ===========================================================================
// AISPARSE-684 -- gebsr2csr_nnz
//
// rocsparse_gebsr2csr_nnz has no public entry point, but rocsparse_gebsr2gebsr_nnz
// calls it for every row_block_dim_C above 32 -- below that it takes a separate
// fast path that never touches this kernel. The intermediate CSR row pointer it
// builds is then consumed by csr2gebsr_nnz, so a row the kernel never wrote
// shows up as a wrong non-zero-block count for C.
//
// mb = 4 with row_block_dim_A = 64 is 256 CSR rows, one wavefront each. At 256
// threads per block and a 32-wide wavefront that is 8 rows per block and 32
// blocks, against a clamp of 3. The shape is the ticket's own: a large
// row_block_dim against a modest mb is what makes this product reachable at all.
// ===========================================================================

TEST_F(ConversionGrids, gebsr2csr_nnz_grid_stride)
{
    constexpr rocsparse_int mb              = 4;
    constexpr rocsparse_int nb              = 4;
    constexpr rocsparse_int row_block_dim_A = 64;
    constexpr rocsparse_int col_block_dim_A = 64;
    constexpr rocsparse_int row_block_dim_C = 33; // > 32: forces the gebsr2csr path
    constexpr rocsparse_int col_block_dim_C = 33;

    // Dense block pattern: every block row holds all nb block columns.
    constexpr rocsparse_int nnzb = mb * nb;

    std::vector<rocsparse_int> bsr_row_ptr_A(mb + 1);
    std::vector<rocsparse_int> bsr_col_ind_A(nnzb);
    for(rocsparse_int r = 0; r <= mb; ++r)
    {
        bsr_row_ptr_A[r] = r * nb;
    }
    for(rocsparse_int r = 0; r < mb; ++r)
    {
        for(rocsparse_int c = 0; c < nb; ++c)
        {
            bsr_col_ind_A[r * nb + c] = c;
        }
    }
    const std::vector<float> bsr_val_A(size_t{nnzb} * row_block_dim_A * col_block_dim_A, 1.0f);

    device_vector<rocsparse_int> d_row_ptr_A(bsr_row_ptr_A);
    device_vector<rocsparse_int> d_col_ind_A(bsr_col_ind_A);
    device_vector<float>         d_val_A(bsr_val_A);

    const rocsparse_int m    = mb * row_block_dim_A; // 256
    const rocsparse_int n    = nb * col_block_dim_A; // 256
    const rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    const rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    device_vector<rocsparse_int> d_row_ptr_C(std::vector<rocsparse_int>(mb_c + 1, -1));

    rocsparse_mat_descr descr_A = nullptr;
    rocsparse_mat_descr descr_C = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_A), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_C), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                 rocsparse_direction_row,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 descr_A,
                                                 d_val_A.ptr,
                                                 d_row_ptr_A.ptr,
                                                 d_col_ind_A.ptr,
                                                 row_block_dim_A,
                                                 col_block_dim_A,
                                                 row_block_dim_C,
                                                 col_block_dim_C,
                                                 &buffer_size),
              rocsparse_status_success);

    device_vector<char> d_buffer(buffer_size ? buffer_size : size_t{1});
    ASSERT_NE(d_buffer.ptr, nullptr);

    // Zeroed so the check below is deterministic: any CSR row the kernel fails to
    // write stays 0 instead of holding whatever was last in that allocation.
    UT_CHECK_HIP(hipMemset(d_buffer.ptr, 0, buffer_size));

    rocsparse_int nnzb_C = -1;
    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                            rocsparse_direction_row,
                                            mb,
                                            nb,
                                            nnzb,
                                            descr_A,
                                            d_row_ptr_A.ptr,
                                            d_col_ind_A.ptr,
                                            row_block_dim_A,
                                            col_block_dim_A,
                                            descr_C,
                                            d_row_ptr_C.ptr,
                                            row_block_dim_C,
                                            col_block_dim_C,
                                            &nnzb_C,
                                            d_buffer.ptr),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    // The intermediate CSR row pointer gebsr2csr_nnz_kernel produced, read back
    // from the front of the caller-supplied temp buffer. rocsparse_gebsr2gebsr_nnz
    // lays the buffer out as [csr_row_ptr (m + 1), csr_col_ind (...)]
    // (rocsparse_gebsr2gebsr.cpp, the row_block_dim_C > 32 branch), so this is a
    // direct read of the kernel's own output rather than an inference from the
    // final block count. A is dense, so every CSR row holds nb * col_block_dim_A
    // entries. This is the assertion that fails deterministically when the kernel
    // does not cover every row; nnzb_C below can survive a short launch depending
    // on what the downstream csr2gebsr_nnz makes of a partly written array.
    std::vector<rocsparse_int> expected_csr_row_ptr(m + 1);
    for(rocsparse_int r = 0; r <= m; ++r)
    {
        expected_csr_row_ptr[r] = r * (nb * col_block_dim_A);
    }
    EXPECT_EQ(first_mismatch(to_host(reinterpret_cast<const rocsparse_int*>(d_buffer.ptr),
                                     static_cast<size_t>(m) + 1),
                             expected_csr_row_ptr),
              -1)
        << "gebsr2csr_nnz_kernel did not write all " << m
        << " CSR row pointers with grid.x clamped to " << clamped_grid_x << " blocks";

    // A is dense, so C is dense too: every one of the mb_c block rows holds all
    // nb_c block columns.
    EXPECT_EQ(nnzb_C, mb_c * nb_c)
        << "gebsr2csr_nnz_kernel did not cover all " << m << " CSR rows with grid.x clamped to "
        << clamped_grid_x << " blocks";

    std::vector<rocsparse_int> expected_row_ptr_C(mb_c + 1);
    for(rocsparse_int r = 0; r <= mb_c; ++r)
    {
        expected_row_ptr_C[r] = r * nb_c;
    }
    EXPECT_EQ(first_mismatch(to_host(d_row_ptr_C), expected_row_ptr_C), -1)
        << "gebsr2gebsr_nnz produced a wrong C row pointer with grid.x clamped to "
        << clamped_grid_x << " blocks";

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_A), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_C), rocsparse_status_success);
}
namespace
{
    // A CSR sparsity pattern with `per_row` entries in every row, except that
    // row `long_row` (when >= 0) gets `long_row_nnz` entries instead. Columns
    // are 0, 1, 2, ... within each row, so the transpose is easy to predict.
    struct csr_pattern
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<float>         val;
        rocsparse_int              m   = 0;
        rocsparse_int              n   = 0;
        rocsparse_int              nnz = 0;

        csr_pattern(rocsparse_int m_,
                    rocsparse_int per_row,
                    rocsparse_int long_row     = -1,
                    rocsparse_int long_row_nnz = 0)
            : m(m_)
        {
            row_ptr.push_back(0);
            for(rocsparse_int r = 0; r < m; ++r)
            {
                const rocsparse_int count = (r == long_row) ? long_row_nnz : per_row;
                for(rocsparse_int k = 0; k < count; ++k)
                {
                    col_ind.push_back(k);
                    val.push_back(static_cast<float>(r * 1000 + k));
                }
                row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
            }
            nnz = row_ptr.back();
            n   = (per_row > long_row_nnz) ? per_row : long_row_nnz;
        }

        // Row index of every non-zero, in CSR order: exactly what csr2coo must
        // produce.
        std::vector<rocsparse_int> expected_coo_row_ind() const
        {
            std::vector<rocsparse_int> expected(nnz);
            for(rocsparse_int r = 0; r < m; ++r)
            {
                for(rocsparse_int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
                {
                    expected[k] = r;
                }
            }
            return expected;
        }
    };
}

// ===========================================================================
// AISPARSE-685 -- csr2coo
//
// csr2coo_kernel runs 256 threads per block and assigns one wavefront of
// WF_SIZE threads per row, so a block covers 256 / WF_SIZE rows and the launcher
// asks for ceil(WF_SIZE * m / 256) blocks. The launcher picks WF_SIZE from
// nnz / m, so the eight call sites are reached by varying the density; every
// case below is shaped to land on one of them and to need more blocks than the
// clamp allows.
//
// This kernel is the one with barriers: two __syncthreads() around a shared
// all_short_rows flag and a per-wavefront short_rows array, plus a third added
// at the bottom of the new stride loop to separate one iteration's reads of
// those from the next iteration's writes. The stride bound is built only from
// hipBlockIdx_x, hipGridDim_x, the kernel argument m and compile-time constants,
// so every thread of a block runs the same number of iterations and reaches all
// three barriers together.
// ===========================================================================

namespace
{
    void run_csr2coo_clamped(rocsparse_handle handle, const csr_pattern& p, const char* which)
    {
        device_vector<rocsparse_int> d_row_ptr(p.row_ptr);
        device_vector<rocsparse_int> d_coo_row_ind(std::vector<rocsparse_int>(p.nnz, -1));
        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_coo_row_ind.ptr, nullptr);

        {
            ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
            ASSERT_EQ(rocsparse_csr2coo(handle,
                                        d_row_ptr.ptr,
                                        p.nnz,
                                        p.m,
                                        d_coo_row_ind.ptr,
                                        rocsparse_index_base_zero),
                      rocsparse_status_success);
            UT_CHECK_HIP(hipDeviceSynchronize());
        }

        EXPECT_EQ(first_mismatch(to_host(d_coo_row_ind), p.expected_coo_row_ind()), -1)
            << "csr2coo (" << which << ") did not cover all " << p.m
            << " rows with grid.x clamped to " << clamped_grid_x << " blocks";
    }
}

// nnz/m selects the wavefront size: <4 -> 2, <8 -> 4, <16 -> 8, <32 -> 16,
// <64 -> 32, <128 -> 64, <256 -> 128, else 256. One case per call site.
TEST_F(ConversionGrids, csr2coo_grid_stride_wf2)
{
    run_csr2coo_clamped(handle, csr_pattern(1000, 1), "wf2");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf4)
{
    run_csr2coo_clamped(handle, csr_pattern(500, 5), "wf4");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf8)
{
    run_csr2coo_clamped(handle, csr_pattern(400, 10), "wf8");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf16)
{
    run_csr2coo_clamped(handle, csr_pattern(300, 20), "wf16");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf32)
{
    run_csr2coo_clamped(handle, csr_pattern(200, 40), "wf32");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf64)
{
    run_csr2coo_clamped(handle, csr_pattern(100, 100), "wf64");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf128)
{
    run_csr2coo_clamped(handle, csr_pattern(100, 200), "wf128");
}

TEST_F(ConversionGrids, csr2coo_grid_stride_wf256)
{
    run_csr2coo_clamped(handle, csr_pattern(50, 300), "wf256");
}

// The long-row branch. A row with more than 8 * WF_SIZE non-zeros clears the
// shared all_short_rows flag, which sends the whole block through the second
// half of the kernel after the second __syncthreads(). This is the case that
// would deadlock or read stale shared state if the stride bound were not
// block-uniform, so it is pinned separately: 600 rows of one entry each with a
// single 5000-entry row gives nnz/m == 9, which lands on WF_SIZE 8 and a
// long-row threshold of 64.
TEST_F(ConversionGrids, csr2coo_grid_stride_long_row_branch)
{
    run_csr2coo_clamped(handle, csr_pattern(600, 1, 300, 5000), "long row");
}

// ===========================================================================
// AISPARSE-685 -- csr2csc
//
// csr2csc_permute_kernel runs 512 threads per block, one thread per non-zero, so
// 2048 non-zeros ask for 4 blocks against a clamp of 3. The action must be
// numeric: the symbolic path never reaches this kernel.
// ===========================================================================

TEST_F(ConversionGrids, csr2csc_grid_stride)
{
    const csr_pattern p(64, 32); // 64 x 32, 2048 non-zeros, every row identical

    device_vector<rocsparse_int> d_row_ptr(p.row_ptr);
    device_vector<rocsparse_int> d_col_ind(p.col_ind);
    device_vector<float>         d_val(p.val);
    device_vector<rocsparse_int> d_csc_col_ptr(std::vector<rocsparse_int>(p.n + 1, -1));
    device_vector<rocsparse_int> d_csc_row_ind(std::vector<rocsparse_int>(p.nnz, -1));
    device_vector<float>         d_csc_val(std::vector<float>(p.nnz, -1.0f));

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_csr2csc_buffer_size(handle,
                                            p.m,
                                            p.n,
                                            p.nnz,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            rocsparse_action_numeric,
                                            &buffer_size),
              rocsparse_status_success);

    device_vector<char> d_buffer(buffer_size);
    ASSERT_NE(d_buffer.ptr, nullptr);

    {
        ScopedMaxGridSizeX clamp(handle, clamped_grid_x);
        ASSERT_EQ(rocsparse_scsr2csc(handle,
                                     p.m,
                                     p.n,
                                     p.nnz,
                                     d_val.ptr,
                                     d_row_ptr.ptr,
                                     d_col_ind.ptr,
                                     d_csc_val.ptr,
                                     d_csc_row_ind.ptr,
                                     d_csc_col_ptr.ptr,
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_buffer.ptr),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());
    }

    // Host transpose of the same pattern: column c holds rows 0..m-1 in order,
    // because every row of the source has the identical column list 0..n-1.
    std::vector<rocsparse_int> expected_col_ptr(p.n + 1);
    std::vector<rocsparse_int> expected_row_ind(p.nnz);
    std::vector<float>         expected_val(p.nnz);
    for(rocsparse_int c = 0; c <= p.n; ++c)
    {
        expected_col_ptr[c] = c * p.m;
    }
    for(rocsparse_int c = 0; c < p.n; ++c)
    {
        for(rocsparse_int r = 0; r < p.m; ++r)
        {
            expected_row_ind[c * p.m + r] = r;
            expected_val[c * p.m + r]     = static_cast<float>(r * 1000 + c);
        }
    }

    EXPECT_EQ(first_mismatch(to_host(d_csc_col_ptr), expected_col_ptr), -1)
        << "csr2csc column pointers wrong with grid.x clamped to " << clamped_grid_x;
    EXPECT_EQ(first_mismatch(to_host(d_csc_row_ind), expected_row_ind), -1)
        << "csr2csc_permute_kernel did not cover all " << p.nnz
        << " non-zeros (row indices) with grid.x clamped to " << clamped_grid_x;
    EXPECT_EQ(first_mismatch(to_host(d_csc_val), expected_val), -1)
        << "csr2csc_permute_kernel did not cover all " << p.nnz
        << " non-zeros (values) with grid.x clamped to " << clamped_grid_x;
}
