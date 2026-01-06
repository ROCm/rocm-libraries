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

#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_spitsv_csr_bad_arg(const Arguments& arg)
{

    J                         m             = 100;
    J                         n             = 100;
    I                         nnz           = 100;
    const T                   local_alpha   = T(0.6);
    const T*                  alpha         = &local_alpha;
    rocsparse_int*            host_nmaxiter = (rocsparse_int*)0x4;
    const floating_data_t<T>* host_tol      = (const floating_data_t<T>*)0x4;
    floating_data_t<T>*       host_history  = (floating_data_t<T>*)0x4;

    rocsparse_operation  trans = rocsparse_operation_none;
    rocsparse_index_base base  = rocsparse_index_base_zero;
    rocsparse_spitsv_alg alg   = rocsparse_spitsv_alg_default;

    // Index and data type
    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Spitsv structures
    rocsparse_local_spmat local_A(
        m, n, nnz, (void*)0x4, (void*)0x4, (void*)0x4, itype, jtype, base, compute_type);

    rocsparse_local_dnvec local_x(m, (void*)0x4, compute_type);
    rocsparse_local_dnvec local_y(m, (void*)0x4, compute_type);

    int       nargs_to_exclude   = 4;
    const int args_to_exclude[4] = {2, 3, 12, 13};

    rocsparse_handle      handle = local_handle;
    rocsparse_spmat_descr mat    = local_A;
    rocsparse_dnvec_descr x      = local_x;
    rocsparse_dnvec_descr y      = local_y;

    size_t  local_buffer_size;
    size_t* buffer_size = &local_buffer_size;
    void*   temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                     \
    handle, host_nmaxiter, host_tol, host_history, trans, alpha, mat, x, y, compute_type, alg, \
        stage, buffer_size, temp_buffer

#define PARAMS_ANALYSIS                                                                        \
    handle, host_nmaxiter, host_tol, host_history, trans, alpha, mat, x, y, compute_type, alg, \
        stage, buffer_size, temp_buffer

#define PARAMS_SOLVE                                                                           \
    handle, host_nmaxiter, host_tol, host_history, trans, alpha, mat, x, y, compute_type, alg, \
        stage, buffer_size, temp_buffer

    rocsparse_spitsv_stage stage = rocsparse_spitsv_stage_buffer_size;
    select_bad_arg_analysis(
        rocsparse_spitsv, nargs_to_exclude, args_to_exclude, PARAMS_BUFFER_SIZE);
    stage = rocsparse_spitsv_stage_preprocess;
    select_bad_arg_analysis(rocsparse_spitsv, nargs_to_exclude, args_to_exclude, PARAMS_ANALYSIS);
    stage = rocsparse_spitsv_stage_compute;
    select_bad_arg_analysis(rocsparse_spitsv, nargs_to_exclude, args_to_exclude, PARAMS_SOLVE);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE
}

template <typename I, typename J, typename T>
void testing_spitsv_csr(const Arguments& arg)
{
    auto tol = get_near_check_tol<T>(arg);
    //
    // Set nmaxiter.
    //
    static constexpr rocsparse_int s_nmaxiter       = 200;
    rocsparse_int                  host_nmaxiter[1] = {s_nmaxiter};

    //
    // Tolerance for the iterative method.
    //
    floating_data_t<T> tol_iterative = static_cast<floating_data_t<T>>(1.0e-6);
    if(std::is_same<floating_data_t<T>, double>{})
        tol_iterative = static_cast<floating_data_t<T>>(1.0e-14);
    floating_data_t<T> host_tol[1] = {tol_iterative};
    floating_data_t<T> host_history[s_nmaxiter];

    J                    M       = arg.M;
    J                    N       = arg.N;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_spitsv_alg alg     = arg.spitsv_alg;
    rocsparse_diag_type  diag    = arg.diag;
    rocsparse_fill_mode  uplo    = arg.uplo;

    T halpha = arg.get_alpha<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_matrix_factory<T, I, J> matrix_factory(arg, false, true);

    // Allocate host memory for matrix
    host_vector<I> hcsr_row_ptr;
    host_vector<J> hcsr_col_ind;
    host_vector<T> hcsr_val;

    // Sample matrix
    I nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz_A, base);
    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    floating_data_t<T> mx = 0;
    for(rocsparse_int i = 0; i < nnz_A; ++i)
        mx = std::max(mx, std::abs(hcsr_val[i]));
    if(mx > 0)
    {
        for(rocsparse_int i = 0; i < nnz_A; ++i)
            hcsr_val[i] /= mx;
    }

    // Allocate host memory for vectors
    host_vector<T> hx(M);
    host_vector<T> hy_1(M);
    host_vector<T> hy_2(M);
    host_vector<T> hy_gold(M);

    // Initialize data on CPU
    rocsparse_init<T>(hx, M, 1, 1, arg.convert_to_int);
    rocsparse_init<T>(hy_1, M, 1, 1, arg.convert_to_int);

    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<I> dcsr_row_ptr(M + 1);
    device_vector<J> dcsr_col_ind(nnz_A);
    device_vector<T> dcsr_val(nnz_A);
    device_vector<T> dx(M);
    device_vector<T> dy_1(M);
    device_vector<T> dy_2(M);
    device_vector<T> dalpha(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(I) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(
        M, N, nnz_A, dcsr_row_ptr, dcsr_col_ind, dcsr_val, itype, jtype, base, ttype);

    rocsparse_local_dnvec x(M, dx, ttype);
    rocsparse_local_dnvec y1(M, dy_1, ttype);
    rocsparse_local_dnvec y2(M, dy_2, ttype);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    // Query Spitsv buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                           host_nmaxiter,
                                           host_tol,
                                           host_history,
                                           trans_A,
                                           &halpha,
                                           A,
                                           x,
                                           y1,
                                           ttype,
                                           alg,
                                           rocsparse_spitsv_stage_buffer_size,
                                           &buffer_size,
                                           nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    // Perform analysis on host
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                           host_nmaxiter,
                                           host_tol,
                                           host_history,
                                           trans_A,
                                           &halpha,
                                           A,
                                           x,
                                           y1,
                                           ttype,
                                           alg,
                                           rocsparse_spitsv_stage_preprocess,
                                           nullptr,
                                           dbuffer));

    // Perform analysis on device
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                           host_nmaxiter,
                                           host_tol,
                                           host_history,
                                           trans_A,
                                           dalpha,
                                           A,
                                           x,
                                           y2,
                                           ttype,
                                           alg,
                                           rocsparse_spitsv_stage_preprocess,
                                           nullptr,
                                           dbuffer));

    if(arg.unit_check)
    {
        // Solve on host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_HIP_ERROR(hipMemset(dy_1, 0, sizeof(T) * M));
        host_nmaxiter[0] = s_nmaxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                               host_nmaxiter,
                                               host_tol,
                                               host_history,
                                               trans_A,
                                               &halpha,
                                               A,
                                               x,
                                               y1,
                                               ttype,
                                               alg,
                                               rocsparse_spitsv_stage_compute,
                                               &buffer_size,
                                               dbuffer));

        // Solve on device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemset(dy_2, 0, sizeof(T) * M));
        host_nmaxiter[0] = s_nmaxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                               host_nmaxiter,
                                               host_tol,
                                               host_history,
                                               trans_A,
                                               dalpha,
                                               A,
                                               x,
                                               y2,
                                               ttype,
                                               alg,
                                               rocsparse_spitsv_stage_compute,
                                               &buffer_size,
                                               dbuffer));

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));
        // CPU csrsv
        J analysis_pivot = -1;
        J solve_pivot    = -1;
        host_csrsv<I, J, T>(trans_A,
                            M,
                            nnz_A,
                            halpha,
                            hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            hx,
                            (int64_t)1,
                            hy_gold,
                            diag,
                            uplo,
                            base,
                            &analysis_pivot,
                            &solve_pivot);

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save(
                    "Y pointe mode host", hy_1, "Y pointe mode device", hy_2);
            }
            hy_gold.near_check(hy_1, tol);
            hy_gold.near_check(hy_2, tol);
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
            CHECK_HIP_ERROR(hipMemset(dy_1, 0, sizeof(T) * M));
            host_nmaxiter[0] = s_nmaxiter;

            CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                                   host_nmaxiter,
                                                   host_tol,
                                                   host_history,
                                                   trans_A,
                                                   &halpha,
                                                   A,
                                                   x,
                                                   y1,
                                                   ttype,
                                                   alg,
                                                   rocsparse_spitsv_stage_compute,
                                                   &buffer_size,
                                                   dbuffer));
        }

        double gpu_time_used = 0;

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemset(dy_1, 0, sizeof(T) * M));
            host_nmaxiter[0]          = s_nmaxiter;
            double gpu_time_used_iter = get_time_us();
            CHECK_ROCSPARSE_ERROR(rocsparse_spitsv(handle,
                                                   host_nmaxiter,
                                                   host_tol,
                                                   host_history,
                                                   trans_A,
                                                   &halpha,
                                                   A,
                                                   x,
                                                   y1,
                                                   ttype,
                                                   alg,
                                                   rocsparse_spitsv_stage_compute,
                                                   &buffer_size,
                                                   dbuffer));
            gpu_time_used_iter = (get_time_us() - gpu_time_used_iter);
            gpu_time_used += gpu_time_used_iter;
        }
        gpu_time_used /= number_hot_calls;

        double gflop_count = spsv_gflop_count(M, nnz_A, diag);
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrsv_gbyte_count<T>(M, nnz_A);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::algorithm,
                            rocsparse_spitsvalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_spitsv_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spitsv_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_spitsv_csr_extra(const Arguments& arg) {}
