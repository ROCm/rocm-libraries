/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_bsrgeam_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;
    T                   h_alpha   = 0.6;
    T                   h_beta    = 0.2;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_B;
    rocsparse_local_mat_descr local_descr_C;

    rocsparse_handle          handle        = local_handle;
    rocsparse_direction       dir           = rocsparse_direction_row;
    rocsparse_int             mb            = safe_size;
    rocsparse_int             nb            = safe_size;
    rocsparse_int             block_dim     = safe_size;
    const T*                  alpha         = &h_alpha;
    const rocsparse_mat_descr descr_A       = local_descr_A;
    rocsparse_int             nnzb_A        = safe_size;
    const T*                  bsr_val_A     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)0x4;
    const T*                  beta          = &h_beta;
    const rocsparse_mat_descr descr_B       = local_descr_B;
    rocsparse_int             nnzb_B        = safe_size;
    const T*                  bsr_val_B     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        bsr_val_C     = (T*)0x4;

    host_dense_vector<rocsparse_int> hbsr_row_ptr_C(safe_size + 1);
    hbsr_row_ptr_C[0] = 0;
    hbsr_row_ptr_C[1] = 1;
    device_dense_vector<rocsparse_int> dbsr_row_ptr_C(hbsr_row_ptr_C);

    rocsparse_int* bsr_row_ptr_C = (rocsparse_int*)dbsr_row_ptr_C;
    rocsparse_int* bsr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int* nnzb_C        = (rocsparse_int*)0x4;

#define PARAMS_NNZB                                                                         \
    handle, dir, mb, nb, block_dim, descr_A, nnzb_A, bsr_row_ptr_A, bsr_col_ind_A, descr_B, \
        nnzb_B, bsr_row_ptr_B, bsr_col_ind_B, descr_C, bsr_row_ptr_C, nnzb_C

#define PARAMS                                                                                  \
    handle, dir, mb, nb, block_dim, alpha, descr_A, nnzb_A, bsr_val_A, bsr_row_ptr_A,           \
        bsr_col_ind_A, beta, descr_B, nnzb_B, bsr_val_B, bsr_row_ptr_B, bsr_col_ind_B, descr_C, \
        bsr_val_C, bsr_row_ptr_C, bsr_col_ind_C

    bad_arg_analysis(rocsparse_bsrgeam_nnzb, PARAMS_NNZB);
    bad_arg_analysis(rocsparse_bsrgeam<T>, PARAMS);
    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_A, val));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_B, val));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_C, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam_nnzb(PARAMS_NNZB),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_B, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_C, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_B, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_C, rocsparse_matrix_type_general));

    // Check block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam_nnzb(PARAMS_NNZB), rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam<T>(PARAMS), rocsparse_status_invalid_size);
    block_dim = safe_size;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_A, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_B, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_C, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam_nnzb(PARAMS_NNZB),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgeam<T>(PARAMS), rocsparse_status_requires_sorted_storage);

#undef PARAMS
#undef PARAMS_NNZB
}

template <typename T>
void testing_bsrgeam(const Arguments& arg)
{
    rocsparse_direction                dir       = arg.direction;
    rocsparse_int                      M         = arg.M;
    rocsparse_int                      N         = arg.N;
    rocsparse_int                      block_dim = arg.block_dim;
    rocsparse_index_base               baseA     = arg.baseA;
    rocsparse_index_base               baseB     = arg.baseB;
    rocsparse_index_base               baseC     = arg.baseC;
    static constexpr bool              full_rank = false;
    rocsparse_matrix_factory<T>        matrix_factory(arg, arg.timing ? false : true, full_rank);
    rocsparse_matrix_factory_random<T> matrix_factory_random(full_rank);

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));

    // Allocate host memory for matrices
    host_vector<rocsparse_int> hbsr_row_ptr_A;
    host_vector<rocsparse_int> hbsr_col_ind_A;
    host_vector<T>             hbsr_val_A;
    host_vector<rocsparse_int> hbsr_row_ptr_B;
    host_vector<rocsparse_int> hbsr_col_ind_B;
    host_vector<T>             hbsr_val_B;

    // Sample matrix
    rocsparse_int nnzb_A;
    rocsparse_int nnzb_B;
    rocsparse_int hnnzb_C_gold;
    rocsparse_int hnnzb_C_1;
    rocsparse_int hnnzb_C_2;

    // Sample A
    matrix_factory.init_bsr(
        hbsr_row_ptr_A, hbsr_col_ind_A, hbsr_val_A, dir, Mb, Nb, nnzb_A, block_dim, baseA);

    // Sample B
    matrix_factory_random.init_gebsr(hbsr_row_ptr_B,
                                     hbsr_col_ind_B,
                                     hbsr_val_B,
                                     dir,
                                     Mb,
                                     Nb,
                                     nnzb_B,
                                     block_dim,
                                     block_dim,
                                     baseB,
                                     rocsparse_matrix_type_general,
                                     rocsparse_fill_mode_lower,
                                     rocsparse_storage_mode_sorted);

    // Allocate device memory
    device_vector<rocsparse_int> dbsr_row_ptr_A(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind_A(nnzb_A);
    device_vector<T>             dbsr_val_A(size_t(nnzb_A) * block_dim * block_dim);
    device_vector<rocsparse_int> dbsr_row_ptr_B(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind_B(nnzb_B);
    device_vector<T>             dbsr_val_B(size_t(nnzb_B) * block_dim * block_dim);
    device_scalar<T>             d_alpha(h_alpha);
    device_scalar<T>             d_beta(h_beta);
    device_vector<rocsparse_int> dbsr_row_ptr_C_1(Mb + 1);
    device_vector<rocsparse_int> dbsr_row_ptr_C_2(Mb + 1);
    device_vector<rocsparse_int> dnnzb_C_2(1);

    // Copy data from CPU to device
    dbsr_row_ptr_A.transfer_from(hbsr_row_ptr_A);
    dbsr_col_ind_A.transfer_from(hbsr_col_ind_A);
    dbsr_val_A.transfer_from(hbsr_val_A);
    dbsr_row_ptr_B.transfer_from(hbsr_row_ptr_B);
    dbsr_col_ind_B.transfer_from(hbsr_col_ind_B);
    dbsr_val_B.transfer_from(hbsr_val_B);

    if(arg.unit_check)
    {
        // Obtain nnzb of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgeam_nnzb(handle,
                                                     dir,
                                                     Mb,
                                                     Nb,
                                                     block_dim,
                                                     descrA,
                                                     nnzb_A,
                                                     dbsr_row_ptr_A,
                                                     dbsr_col_ind_A,
                                                     descrB,
                                                     nnzb_B,
                                                     dbsr_row_ptr_B,
                                                     dbsr_col_ind_B,
                                                     descrC,
                                                     dbsr_row_ptr_C_1,
                                                     &hnnzb_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgeam_nnzb(handle,
                                                     dir,
                                                     Mb,
                                                     Nb,
                                                     block_dim,
                                                     descrA,
                                                     nnzb_A,
                                                     dbsr_row_ptr_A,
                                                     dbsr_col_ind_A,
                                                     descrB,
                                                     nnzb_B,
                                                     dbsr_row_ptr_B,
                                                     dbsr_col_ind_B,
                                                     descrC,
                                                     dbsr_row_ptr_C_2,
                                                     dnnzb_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hbsr_row_ptr_C_1(Mb + 1);
        host_vector<rocsparse_int> hbsr_row_ptr_C_2(Mb + 1);

        hbsr_row_ptr_C_1.transfer_from(dbsr_row_ptr_C_1);
        hbsr_row_ptr_C_2.transfer_from(dbsr_row_ptr_C_2);

        CHECK_HIP_ERROR(
            hipMemcpy(&hnnzb_C_2, dnnzb_C_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // CPU bsrgemm_nnzb
        host_vector<rocsparse_int> hbsr_row_ptr_C_gold(Mb + 1);
        host_bsrgeam_nnzb<T>(dir,
                             Mb,
                             Nb,
                             block_dim,
                             h_alpha,
                             hbsr_row_ptr_A,
                             hbsr_col_ind_A,
                             h_beta,
                             hbsr_row_ptr_B,
                             hbsr_col_ind_B,
                             hbsr_row_ptr_C_gold,
                             &hnnzb_C_gold,
                             baseA,
                             baseB,
                             baseC);

        // Check nnz of C
        unit_check_scalar(hnnzb_C_gold, hnnzb_C_1);
        unit_check_scalar(hnnzb_C_gold, hnnzb_C_2);

        // Check row pointers of C
        hbsr_row_ptr_C_gold.unit_check(hbsr_row_ptr_C_1);
        hbsr_row_ptr_C_gold.unit_check(hbsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<rocsparse_int> dbsr_col_ind_C_1(hnnzb_C_1);
        device_vector<rocsparse_int> dbsr_col_ind_C_2(hnnzb_C_2);
        device_vector<T>             dbsr_val_C_1(size_t(hnnzb_C_1) * block_dim * block_dim);
        device_vector<T>             dbsr_val_C_2(size_t(hnnzb_C_2) * block_dim * block_dim);

        // Perform matrix matrix multiplication

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgeam<T>(handle,
                                                   dir,
                                                   Mb,
                                                   Nb,
                                                   block_dim,
                                                   h_alpha,
                                                   descrA,
                                                   nnzb_A,
                                                   dbsr_val_A,
                                                   dbsr_row_ptr_A,
                                                   dbsr_col_ind_A,
                                                   h_beta,
                                                   descrB,
                                                   nnzb_B,
                                                   dbsr_val_B,
                                                   dbsr_row_ptr_B,
                                                   dbsr_col_ind_B,
                                                   descrC,
                                                   dbsr_val_C_1,
                                                   dbsr_row_ptr_C_1,
                                                   dbsr_col_ind_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgeam<T>(handle,
                                                   dir,
                                                   Mb,
                                                   Nb,
                                                   block_dim,
                                                   d_alpha,
                                                   descrA,
                                                   nnzb_A,
                                                   dbsr_val_A,
                                                   dbsr_row_ptr_A,
                                                   dbsr_col_ind_A,
                                                   d_beta,
                                                   descrB,
                                                   nnzb_B,
                                                   dbsr_val_B,
                                                   dbsr_row_ptr_B,
                                                   dbsr_col_ind_B,
                                                   descrC,
                                                   dbsr_val_C_2,
                                                   dbsr_row_ptr_C_2,
                                                   dbsr_col_ind_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hbsr_col_ind_C_1(hnnzb_C_1);
        host_vector<rocsparse_int> hbsr_col_ind_C_2(hnnzb_C_2);
        host_vector<T>             hbsr_val_C_1(size_t(hnnzb_C_1) * block_dim * block_dim);
        host_vector<T>             hbsr_val_C_2(size_t(hnnzb_C_2) * block_dim * block_dim);

        hbsr_col_ind_C_1.transfer_from(dbsr_col_ind_C_1);
        hbsr_col_ind_C_2.transfer_from(dbsr_col_ind_C_2);
        hbsr_val_C_1.transfer_from(dbsr_val_C_1);
        hbsr_val_C_2.transfer_from(dbsr_val_C_2);

        // CPU bsrgemm
        host_vector<rocsparse_int> hbsr_col_ind_C_gold(hnnzb_C_gold);
        host_vector<T>             hbsr_val_C_gold(size_t(hnnzb_C_gold) * block_dim * block_dim);
        host_bsrgeam<T>(dir,
                        Mb,
                        Nb,
                        block_dim,
                        h_alpha,
                        hbsr_row_ptr_A,
                        hbsr_col_ind_A,
                        hbsr_val_A,
                        h_beta,
                        hbsr_row_ptr_B,
                        hbsr_col_ind_B,
                        hbsr_val_B,
                        hbsr_row_ptr_C_gold,
                        hbsr_col_ind_C_gold,
                        hbsr_val_C_gold,
                        baseA,
                        baseB,
                        baseC);

        // Check C
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("hbsr_row_ptr_C_1", hbsr_row_ptr_C_1);
            rocsparse_reproducibility::save("hbsr_col_ind_C_1", hbsr_col_ind_C_1);
            rocsparse_reproducibility::save("hbsr_val_C_1", hbsr_val_C_1);

            rocsparse_reproducibility::save("hbsr_row_ptr_C_2", hbsr_row_ptr_C_2);
            rocsparse_reproducibility::save("hbsr_col_ind_C_2", hbsr_col_ind_C_2);
            rocsparse_reproducibility::save("hbsr_val_C_2", hbsr_val_C_2);
        }

        hbsr_col_ind_C_gold.unit_check(hbsr_col_ind_C_1);
        hbsr_col_ind_C_gold.unit_check(hbsr_col_ind_C_2);
        hbsr_val_C_gold.near_check(hbsr_val_C_1);
        hbsr_val_C_gold.near_check(hbsr_val_C_2);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int nnzb_C;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgeam_nnzb(handle,
                                                     dir,
                                                     Mb,
                                                     Nb,
                                                     block_dim,
                                                     descrA,
                                                     nnzb_A,
                                                     dbsr_row_ptr_A,
                                                     dbsr_col_ind_A,
                                                     descrB,
                                                     nnzb_B,
                                                     dbsr_row_ptr_B,
                                                     dbsr_col_ind_B,
                                                     descrC,
                                                     dbsr_row_ptr_C_1,
                                                     &nnzb_C));

        device_vector<rocsparse_int> dbsr_col_ind_C(nnzb_C);
        device_vector<T>             dbsr_val_C(block_dim * block_dim * nnzb_C);

        const double gpu_analysis_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_bsrgeam_nnzb,
                                               handle,
                                               dir,
                                               Mb,
                                               Nb,
                                               block_dim,
                                               descrA,
                                               nnzb_A,
                                               dbsr_row_ptr_A,
                                               dbsr_col_ind_A,
                                               descrB,
                                               nnzb_B,
                                               dbsr_row_ptr_B,
                                               dbsr_col_ind_B,
                                               descrC,
                                               dbsr_row_ptr_C_1,
                                               &nnzb_C);

        const double gpu_solve_time_used = rocsparse_clients::run_benchmark(arg,
                                                                            rocsparse_bsrgeam<T>,
                                                                            handle,
                                                                            dir,
                                                                            Mb,
                                                                            Nb,
                                                                            block_dim,
                                                                            h_alpha,
                                                                            descrA,
                                                                            nnzb_A,
                                                                            dbsr_val_A,
                                                                            dbsr_row_ptr_A,
                                                                            dbsr_col_ind_A,
                                                                            h_beta,
                                                                            descrB,
                                                                            nnzb_B,
                                                                            dbsr_val_B,
                                                                            dbsr_row_ptr_B,
                                                                            dbsr_col_ind_B,
                                                                            descrC,
                                                                            dbsr_val_C,
                                                                            dbsr_row_ptr_C_1,
                                                                            dbsr_col_ind_C);

        double gflop_count
            = bsrgeam_gflop_count<T>(block_dim, nnzb_A, nnzb_B, nnzb_C, h_alpha, h_beta);
        double gbyte_count
            = bsrgeam_gbyte_count<T>(Mb, block_dim, nnzb_A, nnzb_B, nnzb_C, h_alpha, h_beta);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnzb_A,
                            nnzb_A,
                            display_key_t::nnzb_B,
                            nnzb_B,
                            display_key_t::nnzb_C,
                            nnzb_C,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::analysis_time_ms,
                            get_gpu_time_msec(gpu_analysis_time_used),
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsrgeam_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrgeam<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrgeam_extra(const Arguments& arg) {}
