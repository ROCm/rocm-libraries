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
// Forced-clamp tests for the bsrgemm scaling path, C = beta * D (AISPARSE-676).
//
// FOCUS. With alpha == nullptr and beta != nullptr, bsrgemm_scal_core copies the
// column indices of D with bsrgemm_copy and scales the block values with
// bsrgemm_copy_scale. Both clamp grid.x with rocsparse::get_grid_size_x, and
// bsrgemm_copy_scale takes its GRID_STRIDE instantiation only when the clamp
// binds, so a normally sized problem never runs the looping variant.
//
// WHY NOT TEST THE REAL THRESHOLD. With 1024-thread blocks the clamp binds at
// (2^32 - 1) / 1024 = 4,194,303 blocks, i.e. about 2^32 block values. These
// tests shrink handle->properties.maxGridSize[0] with ScopedMaxGridSizeX to 1,
// 3, 7 and 64 blocks instead, so a problem of 20000 blocks takes the clamped,
// looping path. The memory-guarded spgemm_bsr_extra case covers the real
// > 2^31 element count.
//
// WHAT MAKES EACH CASE LOAD-BEARING. nnzb_D = 20000 blocks spans 20 column-index
// blocks and block_dim^2 * 20000 / 1024 >= 79 value blocks, so every limit
// clamps the scaling launch and limits 1, 3 and 7 also clamp the index copy. A
// straight-line launch on the clamped grid leaves the tail of C unwritten.
//
// EXACT ARITHMETIC. Values and beta are small integers, so C compares exactly.
//
// TARGET: rocsparse-unit-test-device.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit
// get_grid_size_x clamps grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // grid.x limits the tests shrink maxGridSize[0] to. 0 means the device limit.
    constexpr int clamped_limits[] = {1, 3, 7, 64};

    constexpr int64_t scal_blocksize = 1024;

    constexpr rocsparse_int mb             = 5000;
    constexpr rocsparse_int nb             = 64;
    constexpr rocsparse_int kb             = 16;
    constexpr rocsparse_int blocks_per_row = 4;
    constexpr rocsparse_int nnzb_D         = mb * blocks_per_row;

    constexpr float beta_value = 3.0f;

    rocsparse_status bsrgemm_buffer_size(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_int             block_dim,
                                         const float*              beta,
                                         const rocsparse_mat_descr descr,
                                         const rocsparse_int*      ptr_D,
                                         const rocsparse_int*      ind_D,
                                         rocsparse_mat_info        info,
                                         size_t*                   buffer_size)
    {
        return rocsparse_sbsrgemm_buffer_size(handle,
                                              dir,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              mb,
                                              nb,
                                              kb,
                                              block_dim,
                                              nullptr,
                                              nullptr,
                                              0,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              0,
                                              nullptr,
                                              nullptr,
                                              beta,
                                              descr,
                                              nnzb_D,
                                              ptr_D,
                                              ind_D,
                                              info,
                                              buffer_size);
    }

    rocsparse_status bsrgemm_buffer_size(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_int             block_dim,
                                         const double*             beta,
                                         const rocsparse_mat_descr descr,
                                         const rocsparse_int*      ptr_D,
                                         const rocsparse_int*      ind_D,
                                         rocsparse_mat_info        info,
                                         size_t*                   buffer_size)
    {
        return rocsparse_dbsrgemm_buffer_size(handle,
                                              dir,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              mb,
                                              nb,
                                              kb,
                                              block_dim,
                                              nullptr,
                                              nullptr,
                                              0,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              0,
                                              nullptr,
                                              nullptr,
                                              beta,
                                              descr,
                                              nnzb_D,
                                              ptr_D,
                                              ind_D,
                                              info,
                                              buffer_size);
    }

    rocsparse_status bsrgemm_compute(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             block_dim,
                                     const float*              beta,
                                     const rocsparse_mat_descr descr,
                                     const float*              val_D,
                                     const rocsparse_int*      ptr_D,
                                     const rocsparse_int*      ind_D,
                                     float*                    val_C,
                                     const rocsparse_int*      ptr_C,
                                     rocsparse_int*            ind_C,
                                     rocsparse_mat_info        info,
                                     void*                     buffer)
    {
        return rocsparse_sbsrgemm(handle,
                                  dir,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  mb,
                                  nb,
                                  kb,
                                  block_dim,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  beta,
                                  descr,
                                  nnzb_D,
                                  val_D,
                                  ptr_D,
                                  ind_D,
                                  descr,
                                  val_C,
                                  ptr_C,
                                  ind_C,
                                  info,
                                  buffer);
    }

    rocsparse_status bsrgemm_compute(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             block_dim,
                                     const double*             beta,
                                     const rocsparse_mat_descr descr,
                                     const double*             val_D,
                                     const rocsparse_int*      ptr_D,
                                     const rocsparse_int*      ind_D,
                                     double*                   val_C,
                                     const rocsparse_int*      ptr_C,
                                     rocsparse_int*            ind_C,
                                     rocsparse_mat_info        info,
                                     void*                     buffer)
    {
        return rocsparse_dbsrgemm(handle,
                                  dir,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  mb,
                                  nb,
                                  kb,
                                  block_dim,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  beta,
                                  descr,
                                  nnzb_D,
                                  val_D,
                                  ptr_D,
                                  ind_D,
                                  descr,
                                  val_C,
                                  ptr_C,
                                  ind_C,
                                  info,
                                  buffer);
    }

    // D has blocks_per_row blocks in every block row, at distinct block columns.
    struct HostBsr
    {
        std::vector<rocsparse_int> ptr;
        std::vector<rocsparse_int> ind;
        std::vector<double>        val;
    };

    HostBsr make_d(rocsparse_int block_dim)
    {
        HostBsr d;
        d.ptr.resize(mb + 1);
        d.ind.resize(nnzb_D);
        d.val.resize(static_cast<size_t>(nnzb_D) * block_dim * block_dim);
        for(rocsparse_int i = 0; i <= mb; ++i)
        {
            d.ptr[i] = i * blocks_per_row;
        }
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            for(rocsparse_int k = 0; k < blocks_per_row; ++k)
            {
                d.ind[i * blocks_per_row + k] = (i + k * (nb / blocks_per_row)) % nb;
            }
        }
        for(size_t e = 0; e < d.val.size(); ++e)
        {
            d.val[e] = static_cast<double>(e % 13) - 6.0;
        }
        return d;
    }

    // Runs C = beta * D with grid.x clamped to limit blocks (0: device limit).
    template <typename T>
    bool device_scal(rocsparse_handle       handle,
                     rocsparse_direction    dir,
                     rocsparse_int          block_dim,
                     rocsparse_pointer_mode mode,
                     const HostBsr&         d,
                     int                    limit,
                     HostBsr&               c)
    {
        std::unique_ptr<ScopedMaxGridSizeX> clamp;
        if(limit > 0)
        {
            clamp.reset(new ScopedMaxGridSizeX(handle, limit));
        }

        const std::vector<T> h_val_D(d.val.begin(), d.val.end());
        const T              h_beta = static_cast<T>(beta_value);

        device_vector<rocsparse_int> d_ptr_D(d.ptr);
        device_vector<rocsparse_int> d_ind_D(d.ind);
        device_vector<T>             d_val_D(h_val_D);
        device_vector<rocsparse_int> d_ptr_C(static_cast<size_t>(mb + 1));
        device_vector<T>             d_beta(std::vector<T>{h_beta});

        rocsparse_mat_descr descr = nullptr;
        rocsparse_mat_info  info  = nullptr;
        EXPECT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_set_pointer_mode(handle, mode), rocsparse_status_success);
        const T* beta = (mode == rocsparse_pointer_mode_host) ? &h_beta : d_beta.ptr;

        void*            buffer      = nullptr;
        size_t           buffer_size = 0;
        rocsparse_int    nnzb_C      = 0;
        rocsparse_status status      = bsrgemm_buffer_size(
            handle, dir, block_dim, beta, descr, d_ptr_D, d_ind_D, info, &buffer_size);
        if(status == rocsparse_status_success
           && hipMalloc(&buffer, std::max<size_t>(buffer_size, 1)) != hipSuccess)
        {
            buffer = nullptr;
            status = rocsparse_status_memory_error;
        }
        if(status == rocsparse_status_success)
        {
            status = rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
        }
        if(status == rocsparse_status_success)
        {
            status = rocsparse_bsrgemm_nnzb(handle,
                                            dir,
                                            rocsparse_operation_none,
                                            rocsparse_operation_none,
                                            mb,
                                            nb,
                                            kb,
                                            block_dim,
                                            nullptr,
                                            0,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            0,
                                            nullptr,
                                            nullptr,
                                            descr,
                                            nnzb_D,
                                            d_ptr_D,
                                            d_ind_D,
                                            descr,
                                            d_ptr_C,
                                            &nnzb_C,
                                            info,
                                            buffer);
        }
        if(status == rocsparse_status_success)
        {
            status = rocsparse_set_pointer_mode(handle, mode);
        }

        device_vector<rocsparse_int> d_ind_C(static_cast<size_t>(std::max(nnzb_C, 1)));
        device_vector<T> d_val_C(static_cast<size_t>(std::max(nnzb_C, 1)) * block_dim * block_dim);
        // Poison C: hipMalloc may hand back the previous run's buffer, whose
        // stale values would hide elements this run never writes.
        if(status == rocsparse_status_success
           && (hipMemset(d_ind_C.ptr, 0xFF, d_ind_C.n * sizeof(rocsparse_int)) != hipSuccess
               || hipMemset(d_val_C.ptr, 0xFF, d_val_C.n * sizeof(T)) != hipSuccess))
        {
            status = rocsparse_status_internal_error;
        }
        if(status == rocsparse_status_success)
        {
            status = bsrgemm_compute(handle,
                                     dir,
                                     block_dim,
                                     beta,
                                     descr,
                                     d_val_D,
                                     d_ptr_D,
                                     d_ind_D,
                                     d_val_C,
                                     d_ptr_C,
                                     d_ind_C,
                                     info,
                                     buffer);
        }
        if(status == rocsparse_status_success && hipDeviceSynchronize() != hipSuccess)
        {
            status = rocsparse_status_internal_error;
        }
        if(buffer != nullptr)
        {
            (void)hipFree(buffer);
        }
        EXPECT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        if(status != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_bsrgemm failed with status " << status << " at limit "
                          << limit;
            return false;
        }

        const std::vector<T> h_val_C
            = to_host<T>(d_val_C.ptr, static_cast<size_t>(nnzb_C) * block_dim * block_dim);
        c.ptr = to_host(d_ptr_C);
        c.ind = to_host<rocsparse_int>(d_ind_C.ptr, static_cast<size_t>(nnzb_C));
        c.val.assign(h_val_C.begin(), h_val_C.end());
        return true;
    }

    std::string first_difference(const HostBsr& got, const HostBsr& want)
    {
        std::ostringstream msg;
        if(got.ptr != want.ptr)
        {
            msg << "row pointers differ";
        }
        else if(got.ind.size() != want.ind.size())
        {
            msg << "nnzb " << got.ind.size() << " != " << want.ind.size();
        }
        else
        {
            for(size_t k = 0; k < want.ind.size(); ++k)
            {
                if(got.ind[k] != want.ind[k])
                {
                    msg << "col_ind[" << k << "] = " << got.ind[k] << ", want " << want.ind[k];
                    return msg.str();
                }
            }
            for(size_t e = 0; e < want.val.size(); ++e)
            {
                if(got.val[e] != want.val[e])
                {
                    msg << "val[" << e << "] of " << want.val.size() << " = " << got.val[e]
                        << ", want " << want.val[e];
                    return msg.str();
                }
            }
        }
        return msg.str();
    }

    template <typename T>
    void check_scal(rocsparse_handle       handle,
                    rocsparse_int          block_dim,
                    rocsparse_direction    dir,
                    rocsparse_pointer_mode mode)
    {
        const HostBsr d    = make_d(block_dim);
        HostBsr       want = d;
        for(double& v : want.val)
        {
            v *= beta_value;
        }

        // Every limit must clamp the scaling launch; limits 1, 3 and 7 also clamp
        // the column-index copy.
        const int64_t value_blocks
            = (static_cast<int64_t>(want.val.size()) - 1) / scal_blocksize + 1;
        ASSERT_GT(value_blocks, clamped_limits[3]);
        ASSERT_GT((static_cast<int64_t>(nnzb_D) - 1) / scal_blocksize + 1, clamped_limits[2]);

        HostBsr unclamped;
        ASSERT_TRUE(device_scal<T>(handle, dir, block_dim, mode, d, 0, unclamped));
        const std::string control = first_difference(unclamped, want);
        ASSERT_TRUE(control.empty()) << "unclamped control: " << control;

        for(const int limit : clamped_limits)
        {
            HostBsr got;
            ASSERT_TRUE(device_scal<T>(handle, dir, block_dim, mode, d, limit, got));
            const std::string diff = first_difference(got, want);
            EXPECT_TRUE(diff.empty()) << "grid.x clamped to " << limit << " blocks: " << diff;
        }
    }

    using BsrgemmScalGrids = HandleTest;
}

TEST_F(BsrgemmScalGrids, bd2_row_f32_host)
{
    check_scal<float>(handle, 2, rocsparse_direction_row, rocsparse_pointer_mode_host);
}

TEST_F(BsrgemmScalGrids, bd2_column_f64_device)
{
    check_scal<double>(handle, 2, rocsparse_direction_column, rocsparse_pointer_mode_device);
}

TEST_F(BsrgemmScalGrids, bd3_row_f64_host)
{
    check_scal<double>(handle, 3, rocsparse_direction_row, rocsparse_pointer_mode_host);
}

TEST_F(BsrgemmScalGrids, bd4_column_f32_host)
{
    check_scal<float>(handle, 4, rocsparse_direction_column, rocsparse_pointer_mode_host);
}

TEST_F(BsrgemmScalGrids, bd8_row_f64_device)
{
    check_scal<double>(handle, 8, rocsparse_direction_row, rocsparse_pointer_mode_device);
}

TEST_F(BsrgemmScalGrids, bd16_row_f32_host)
{
    check_scal<float>(handle, 16, rocsparse_direction_row, rocsparse_pointer_mode_host);
}
