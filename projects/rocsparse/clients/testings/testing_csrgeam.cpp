/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename T>
void testing_csrgeam_bad_arg(const Arguments& arg)
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
    rocsparse_int             m             = safe_size;
    rocsparse_int             n             = safe_size;
    const T*                  alpha         = &h_alpha;
    const rocsparse_mat_descr descr_A       = local_descr_A;
    rocsparse_int             nnz_A         = safe_size;
    const T*                  csr_val_A     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;
    const T*                  beta          = &h_beta;
    const rocsparse_mat_descr descr_B       = local_descr_B;
    rocsparse_int             nnz_B         = safe_size;
    const T*                  csr_val_B     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        csr_val_C     = (T*)0x4;

    host_dense_vector<rocsparse_int> hcsr_row_ptr_C(safe_size + 1);
    hcsr_row_ptr_C[0] = 0;
    hcsr_row_ptr_C[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr_C(hcsr_row_ptr_C);

    rocsparse_int* csr_row_ptr_C = (rocsparse_int*)dcsr_row_ptr_C;
    rocsparse_int* csr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int* nnz_C         = (rocsparse_int*)0x4;

#define PARAMS_NNZ                                                                             \
    handle, m, n, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, \
        csr_col_ind_B, descr_C, csr_row_ptr_C, nnz_C

#define PARAMS                                                                                   \
    handle, m, n, alpha, descr_A, nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, beta, descr_B, \
        nnz_B, csr_val_B, csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C,       \
        csr_col_ind_C

    bad_arg_analysis(rocsparse_csrgeam_nnz, PARAMS_NNZ);
    bad_arg_analysis(rocsparse_csrgeam<T>, PARAMS);

    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_A, val));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_B, val));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_C, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgeam_nnz(PARAMS_NNZ),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrgeam<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_B, rocsparse_matrix_type_general));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr_C, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_A, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_B, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_C, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgeam_nnz(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgeam<T>(PARAMS), rocsparse_status_requires_sorted_storage);

#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csrgeam(const Arguments& arg)
{
    rocsparse_int                      M         = arg.M;
    rocsparse_int                      N         = arg.N;
    rocsparse_index_base               baseA     = arg.baseA;
    rocsparse_index_base               baseB     = arg.baseB;
    rocsparse_index_base               baseC     = arg.baseC;
    static constexpr bool              full_rank = false;
    rocsparse_matrix_factory<T>        matrix_factory(arg, arg.timing ? false : true, full_rank);
    rocsparse_matrix_factory_random<T> matrix_factory_random(full_rank);

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));

    // Allocate host memory for matrices
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_B;
    host_vector<rocsparse_int> hcsr_col_ind_B;
    host_vector<T>             hcsr_val_B;

    // Sample matrix
    rocsparse_int nnz_A = 4;
    rocsparse_int nnz_B = 4;
    rocsparse_int hnnz_C_gold;
    rocsparse_int hnnz_C_1;
    rocsparse_int hnnz_C_2;

    // Sample A
    matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz_A, baseA);

    // Sample B
    matrix_factory_random.init_csr(hcsr_row_ptr_B,
                                   hcsr_col_ind_B,
                                   hcsr_val_B,
                                   M,
                                   N,
                                   nnz_B,
                                   baseB,
                                   rocsparse_matrix_type_general,
                                   rocsparse_fill_mode_lower,
                                   rocsparse_storage_mode_sorted);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_B(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_B(nnz_B);
    device_vector<T>             dcsr_val_B(nnz_B);
    device_scalar<T>             d_alpha(h_alpha);
    device_scalar<T>             d_beta(h_beta);
    device_vector<rocsparse_int> dcsr_row_ptr_C_1(M + 1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_2(M + 1);
    device_vector<rocsparse_int> dnnz_C_2(1);

    // Copy data from CPU to device
    dcsr_row_ptr_A.transfer_from(hcsr_row_ptr_A);
    dcsr_col_ind_A.transfer_from(hcsr_col_ind_A);
    dcsr_val_A.transfer_from(hcsr_val_A);
    dcsr_row_ptr_B.transfer_from(hcsr_row_ptr_B);
    dcsr_col_ind_B.transfer_from(hcsr_col_ind_B);
    dcsr_val_B.transfer_from(hcsr_val_B);

    if(arg.unit_check)
    {
        // Obtain nnz of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_2,
                                                    dnnz_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr_C_1(M + 1);
        host_vector<rocsparse_int> hcsr_row_ptr_C_2(M + 1);

        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_2, dnnz_C_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        hcsr_row_ptr_C_1.transfer_from(dcsr_row_ptr_C_1);
        hcsr_row_ptr_C_2.transfer_from(dcsr_row_ptr_C_2);

        // CPU csrgemm_nnz
        host_vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);
        host_csrgeam_nnz<T, rocsparse_int, rocsparse_int>(M,
                                                          N,
                                                          h_alpha,
                                                          hcsr_row_ptr_A,
                                                          hcsr_col_ind_A,
                                                          h_beta,
                                                          hcsr_row_ptr_B,
                                                          hcsr_col_ind_B,
                                                          hcsr_row_ptr_C_gold,
                                                          &hnnz_C_gold,
                                                          baseA,
                                                          baseB,
                                                          baseC);

        // Check nnz of C
        unit_check_scalar(hnnz_C_gold, hnnz_C_1);
        unit_check_scalar(hnnz_C_gold, hnnz_C_2);

        // Check row pointers of C
        hcsr_row_ptr_C_gold.unit_check(hcsr_row_ptr_C_1);
        hcsr_row_ptr_C_gold.unit_check(hcsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<rocsparse_int> dcsr_col_ind_C_1(hnnz_C_1);
        device_vector<rocsparse_int> dcsr_col_ind_C_2(hnnz_C_2);
        device_vector<T>             dcsr_val_C_1(hnnz_C_1);
        device_vector<T>             dcsr_val_C_2(hnnz_C_2);

        // Perform matrix matrix multiplication

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrgeam<T>(handle,
                                                            M,
                                                            N,
                                                            h_alpha,
                                                            descrA,
                                                            nnz_A,
                                                            dcsr_val_A,
                                                            dcsr_row_ptr_A,
                                                            dcsr_col_ind_A,
                                                            h_beta,
                                                            descrB,
                                                            nnz_B,
                                                            dcsr_val_B,
                                                            dcsr_row_ptr_B,
                                                            dcsr_col_ind_B,
                                                            descrC,
                                                            dcsr_val_C_1,
                                                            dcsr_row_ptr_C_1,
                                                            dcsr_col_ind_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrgeam<T>(handle,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            descrA,
                                                            nnz_A,
                                                            dcsr_val_A,
                                                            dcsr_row_ptr_A,
                                                            dcsr_col_ind_A,
                                                            d_beta,
                                                            descrB,
                                                            nnz_B,
                                                            dcsr_val_B,
                                                            dcsr_row_ptr_B,
                                                            dcsr_col_ind_B,
                                                            descrC,
                                                            dcsr_val_C_2,
                                                            dcsr_row_ptr_C_2,
                                                            dcsr_col_ind_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_col_ind_C_1(hnnz_C_1);
        host_vector<rocsparse_int> hcsr_col_ind_C_2(hnnz_C_2);
        host_vector<T>             hcsr_val_C_1(hnnz_C_1);
        host_vector<T>             hcsr_val_C_2(hnnz_C_2);

        hcsr_col_ind_C_1.transfer_from(dcsr_col_ind_C_1);
        hcsr_col_ind_C_2.transfer_from(dcsr_col_ind_C_2);
        hcsr_val_C_1.transfer_from(dcsr_val_C_1);
        hcsr_val_C_2.transfer_from(dcsr_val_C_2);

        // CPU csrgemm
        host_vector<rocsparse_int> hcsr_col_ind_C_gold(hnnz_C_gold);
        host_vector<T>             hcsr_val_C_gold(hnnz_C_gold);
        host_csrgeam<T, rocsparse_int, rocsparse_int>(M,
                                                      N,
                                                      h_alpha,
                                                      hcsr_row_ptr_A,
                                                      hcsr_col_ind_A,
                                                      hcsr_val_A,
                                                      h_beta,
                                                      hcsr_row_ptr_B,
                                                      hcsr_col_ind_B,
                                                      hcsr_val_B,
                                                      hcsr_row_ptr_C_gold,
                                                      hcsr_col_ind_C_gold,
                                                      hcsr_val_C_gold,
                                                      baseA,
                                                      baseB,
                                                      baseC);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("hcsr_row_ptr_C_1", hcsr_row_ptr_C_1);
            rocsparse_reproducibility::save("hcsr_col_ind_C_1", hcsr_col_ind_C_1);
            rocsparse_reproducibility::save("hcsr_val_C_1", hcsr_val_C_1);

            rocsparse_reproducibility::save("hcsr_row_ptr_C_2", hcsr_row_ptr_C_2);
            rocsparse_reproducibility::save("hcsr_col_ind_C_2", hcsr_col_ind_C_2);
            rocsparse_reproducibility::save("hcsr_val_C_2", hcsr_val_C_2);
        }

        // Check C
        hcsr_col_ind_C_gold.unit_check(hcsr_col_ind_C_1);
        hcsr_col_ind_C_gold.unit_check(hcsr_col_ind_C_2);

        hcsr_val_C_gold.near_check(hcsr_val_C_1);
        hcsr_val_C_gold.near_check(hcsr_val_C_2);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int nnz_C;
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &nnz_C));

        device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
        device_vector<T>             dcsr_val_C(nnz_C);

        const double gpu_analysis_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_csrgeam_nnz,
                                               handle,
                                               M,
                                               N,
                                               descrA,
                                               nnz_A,
                                               dcsr_row_ptr_A,
                                               dcsr_col_ind_A,
                                               descrB,
                                               nnz_B,
                                               dcsr_row_ptr_B,
                                               dcsr_col_ind_B,
                                               descrC,
                                               dcsr_row_ptr_C_1,
                                               &nnz_C);

        const double gpu_solve_time_used = rocsparse_clients::run_benchmark(arg,
                                                                            rocsparse_csrgeam<T>,
                                                                            handle,
                                                                            M,
                                                                            N,
                                                                            h_alpha,
                                                                            descrA,
                                                                            nnz_A,
                                                                            dcsr_val_A,
                                                                            dcsr_row_ptr_A,
                                                                            dcsr_col_ind_A,
                                                                            h_beta,
                                                                            descrB,
                                                                            nnz_B,
                                                                            dcsr_val_B,
                                                                            dcsr_row_ptr_B,
                                                                            dcsr_col_ind_B,
                                                                            descrC,
                                                                            dcsr_val_C,
                                                                            dcsr_row_ptr_C_1,
                                                                            dcsr_col_ind_C);

        double gflop_count = csrgeam_gflop_count<T>(nnz_A, nnz_B, nnz_C, h_alpha, h_beta);
        double gbyte_count = csrgeam_gbyte_count<T>(M, nnz_A, nnz_B, nnz_C, h_alpha, h_beta);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nnz_B,
                            nnz_B,
                            display_key_t::nnz_C,
                            nnz_C,
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
    template void testing_csrgeam_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrgeam<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrgeam_extra(const Arguments& arg) {}
