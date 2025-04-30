/*! \file */
/* ************************************************************************
* Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_enum.hpp"
#include "testing.hpp"
#include <iomanip>

template <typename T>
void testing_bsric0_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    size_t h_buffer_size;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nnzb        = safe_size;
    const rocsparse_mat_descr descr       = local_descr;
    T*                        bsr_val     = (T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    rocsparse_mat_info        info        = local_info;
    rocsparse_analysis_policy analysis    = rocsparse_analysis_policy_force;
    rocsparse_solve_policy    solve       = rocsparse_solve_policy_auto;
    rocsparse_solve_policy    policy      = rocsparse_solve_policy_auto;
    size_t*                   buffer_size = &h_buffer_size;
    void*                     temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE \
    handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size

#define PARAMS_ANALYSIS                                                                         \
    handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, \
        solve, temp_buffer

#define PARAMS                                                                                \
    handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info, policy, \
        temp_buffer

    bad_arg_analysis(rocsparse_bsric0_buffer_size<T>, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_bsric0_analysis<T>, PARAMS_ANALYSIS);
    bad_arg_analysis(rocsparse_bsric0<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0<T>(PARAMS), rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0<T>(PARAMS), rocsparse_status_invalid_size);
    block_dim = safe_size;

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS

    // Test rocsparse_bsric0_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(nullptr, info, &position),
                            rocsparse_status_invalid_handle);

    // Test rocsparse_bsric0_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_buffer_size<T>(handle,
                                                            rocsparse_direction_row,
                                                            safe_size,
                                                            safe_size,
                                                            descr,
                                                            nullptr,
                                                            bsr_row_ptr,
                                                            nullptr,
                                                            safe_size,
                                                            info,
                                                            buffer_size),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_analysis<T>(handle,
                                                         rocsparse_direction_row,
                                                         safe_size,
                                                         safe_size,
                                                         descr,
                                                         nullptr,
                                                         bsr_row_ptr,
                                                         nullptr,
                                                         safe_size,
                                                         info,
                                                         rocsparse_analysis_policy_reuse,
                                                         rocsparse_solve_policy_auto,
                                                         temp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0<T>(handle,
                                                rocsparse_direction_row,
                                                safe_size,
                                                safe_size,
                                                descr,
                                                nullptr,
                                                bsr_row_ptr,
                                                nullptr,
                                                safe_size,
                                                info,
                                                rocsparse_solve_policy_auto,
                                                temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsric0(const Arguments& arg)
{
    static constexpr bool       toint     = false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, toint, full_rank);

    rocsparse_int             M         = arg.M;
    rocsparse_int             N         = arg.N;
    rocsparse_int             block_dim = arg.block_dim;
    rocsparse_analysis_policy apol      = arg.apol;
    rocsparse_solve_policy    spol      = arg.spol;
    rocsparse_index_base      base      = arg.baseA;
    rocsparse_direction       direction = arg.direction;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    // Allocate host memory for output BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr;
    host_vector<rocsparse_int> hbsr_col_ind;
    host_vector<T>             hbsr_val_1;

    // Generate BSR matrix on host (or read from file)
    rocsparse_int nnzb;
    matrix_factory.init_bsr(
        hbsr_row_ptr, hbsr_col_ind, hbsr_val_1, direction, Mb, Nb, nnzb, block_dim, base);
    M = Mb * block_dim;

    host_vector<T> hbsr_val_orig(hbsr_val_1);
    host_vector<T> hbsr_val_gold(hbsr_val_1);
    host_vector<T> hbsr_val_2(hbsr_val_1);

    // Allocate device memory for BSR matrix
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val_1(size_t(nnzb) * block_dim * block_dim);
    device_vector<T>             dbsr_val_2(size_t(nnzb) * block_dim * block_dim);

    // Copy BSR matrix from host to device
    CHECK_HIP_ERROR(hipMemcpy(dbsr_row_ptr,
                              hbsr_row_ptr.data(),
                              sizeof(rocsparse_int) * (Mb + 1),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_col_ind, hbsr_col_ind.data(), sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                              hbsr_val_1.data(),
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val_2,
                              hbsr_val_2.data(),
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyHostToDevice));

    // Allocate host memory for pivots
    host_vector<rocsparse_int> hanalysis_pivot_1(1);
    host_vector<rocsparse_int> hanalysis_pivot_2(1);
    host_vector<rocsparse_int> hanalysis_pivot_gold(1);
    host_vector<rocsparse_int> hsolve_pivot_1(1);
    host_vector<rocsparse_int> hsolve_pivot_2(1);
    host_vector<rocsparse_int> hsolve_pivot_gold(1);

    // Allocate device memory for pivots
    device_vector<rocsparse_int> danalysis_pivot_2(1);
    device_vector<rocsparse_int> dsolve_pivot_2(1);

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_buffer_size<T>(handle,
                                                          direction,
                                                          Mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val_1,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          block_dim,
                                                          info,
                                                          &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Perform analysis step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_1,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));
        {
            auto st = rocsparse_bsric0_zero_pivot(handle, info, hanalysis_pivot_1);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                                 : rocsparse_status_success);
        }

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_2,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, danalysis_pivot_2),
                                (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);

        // Perform solve step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsric0<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_1,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           spol,
                                                           dbuffer));
        {
            auto st = rocsparse_bsric0_zero_pivot(handle, info, hsolve_pivot_1);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);
        }

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsric0<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_2,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           spol,
                                                           dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, dsolve_pivot_2),
                                (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_1,
                                  dbsr_val_1,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_2,
                                  dbsr_val_2,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hanalysis_pivot_2, danalysis_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hsolve_pivot_2, dsolve_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // CPU bsric0
        rocsparse_int numerical_pivot;
        rocsparse_int structural_pivot;
        host_bsric0<T>(direction,
                       Mb,
                       block_dim,
                       hbsr_row_ptr,
                       hbsr_col_ind,
                       hbsr_val_gold,
                       base,
                       &structural_pivot,
                       &numerical_pivot);

        hanalysis_pivot_gold[0] = structural_pivot;

        // Solve pivot gives the first numerical or structural non-invertible block
        if(structural_pivot == -1)
        {
            hsolve_pivot_gold[0] = numerical_pivot;
        }
        else if(numerical_pivot == -1)
        {
            hsolve_pivot_gold[0] = structural_pivot;
        }
        else
        {
            hsolve_pivot_gold[0] = std::min(numerical_pivot, structural_pivot);
        }

        // Check pivots
        hanalysis_pivot_gold.unit_check(hanalysis_pivot_1);
        hanalysis_pivot_gold.unit_check(hanalysis_pivot_2);
        hsolve_pivot_gold.unit_check(hsolve_pivot_1);
        hsolve_pivot_gold.unit_check(hsolve_pivot_2);

        // Check solution vector if no pivot has been found
        if(hanalysis_pivot_gold[0] == -1 && hsolve_pivot_gold[0] == -1)
        {
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save(
                    "P pointer mode host", hbsr_val_1, "P pointer mode device", hbsr_val_2);
            }
            hbsr_val_gold.near_check(hbsr_val_1);
            hbsr_val_gold.near_check(hbsr_val_2);
        }
        else
        {
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Pivot analysis pointer mode host",
                                                hanalysis_pivot_1,
                                                "Pivot analysis pointer mode device",
                                                hanalysis_pivot_2,
                                                "Pivot solve pointer mode host",
                                                hsolve_pivot_1,
                                                "Pivot solve pointer mode device",
                                                hsolve_pivot_2);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                      hbsr_val_orig,
                                      sizeof(T) * nnzb * block_dim * block_dim,
                                      hipMemcpyHostToDevice));

            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                               direction,
                                                               Mb,
                                                               nnzb,
                                                               descr,
                                                               dbsr_val_1,
                                                               dbsr_row_ptr,
                                                               dbsr_col_ind,
                                                               block_dim,
                                                               info,
                                                               apol,
                                                               spol,
                                                               dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0<T>(handle,
                                                      direction,
                                                      Mb,
                                                      nnzb,
                                                      descr,
                                                      dbsr_val_1,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      block_dim,
                                                      info,
                                                      spol,
                                                      dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_clear(handle, info));
        }

        CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                  hbsr_val_orig,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyHostToDevice));

        double gpu_analysis_time_used = get_time_us();

        // Analysis run
        CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_1,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));

        gpu_analysis_time_used = (get_time_us() - gpu_analysis_time_used);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, hanalysis_pivot_1),
                                (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);

        double gpu_solve_time_used = 0;

        // Solve run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                      hbsr_val_orig,
                                      sizeof(T) * nnzb * block_dim * block_dim,
                                      hipMemcpyHostToDevice));

            double temp = get_time_us();
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0<T>(handle,
                                                      direction,
                                                      Mb,
                                                      nnzb,
                                                      descr,
                                                      dbsr_val_1,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      block_dim,
                                                      info,
                                                      spol,
                                                      dbuffer));
            gpu_solve_time_used += (get_time_us() - temp);
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, hsolve_pivot_1),
                                (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        gpu_solve_time_used = gpu_solve_time_used / number_hot_calls;

        double gbyte_count = bsric0_gbyte_count<T>(Mb, block_dim, nnzb);

        rocsparse_int pivot = -1;
        if(hanalysis_pivot_1[0] == -1)
        {
            pivot = hsolve_pivot_1[0];
        }
        else if(hsolve_pivot_1[0] == -1)
        {
            pivot = hanalysis_pivot_1[0];
        }
        else
        {
            pivot = std::min(hanalysis_pivot_1[0], hsolve_pivot_1[0]);
        }

        double gpu_gbyte = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::pivot,
                            pivot,
                            display_key_t::bdir,
                            rocsparse_direction2string(direction),
                            display_key_t::analysis_policy,
                            rocsparse_analysis2string(apol),
                            display_key_t::solve_policy,
                            rocsparse_solve2string(spol),
                            display_key_t::analysis_time_ms,
                            get_gpu_time_msec(gpu_analysis_time_used),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Clear bsric0 meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                             \
    template void testing_bsric0_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsric0<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsric0_extra(const Arguments& arg) {}
