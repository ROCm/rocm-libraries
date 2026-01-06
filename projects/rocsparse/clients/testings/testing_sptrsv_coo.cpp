/* ************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_sptrsv.hpp"
#include "testing.hpp"

void rocsparse_clients::sptrsv_analysis(rocsparse_handle       handle,
                                        rocsparse_sptrsv_descr sptrsv_descr,
                                        rocsparse_spmat_descr  A,
                                        rocsparse_dnvec_descr  x,
                                        rocsparse_dnvec_descr  y,
                                        rocsparse_error*       p_error)
{
    void*  buffer;
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(
        handle, sptrsv_descr, A, x, y, rocsparse_sptrsv_stage_analysis, &buffer_size, p_error));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                           sptrsv_descr,
                                           A,
                                           x,
                                           y,
                                           rocsparse_sptrsv_stage_analysis,
                                           buffer_size,
                                           buffer,
                                           p_error));

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
}

void rocsparse_clients::sptrsv_compute(rocsparse_handle       handle,
                                       rocsparse_sptrsv_descr sptrsv_descr,
                                       rocsparse_spmat_descr  A,
                                       rocsparse_dnvec_descr  x,
                                       rocsparse_dnvec_descr  y,
                                       rocsparse_pointer_mode pointer_mode,
                                       const void*            alpha,
                                       rocsparse_error*       p_error)
{

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, pointer_mode));

    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
        handle, sptrsv_descr, rocsparse_sptrsv_input_scalar_alpha, alpha, sizeof(alpha), p_error));

    void*  buffer;
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(
        handle, sptrsv_descr, A, x, y, rocsparse_sptrsv_stage_compute, &buffer_size, p_error));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                           sptrsv_descr,
                                           A,
                                           x,
                                           y,
                                           rocsparse_sptrsv_stage_compute,
                                           buffer_size,
                                           buffer,
                                           p_error));

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
}

template <typename I, typename T>
void testing_sptrsv_coo_bad_arg(const Arguments& arg)
{
    //
    // Bad args of sptrsm is already tested in testing_sptrsv_csr_bad_arg
    //
}

template <typename I, typename T>
void testing_sptrsv_coo(const Arguments& arg)
{
    if(arg.M != arg.N)
    {
        return;
    }

    const rocsparse_operation   trans_A     = arg.transA;
    const rocsparse_index_base  base        = arg.baseA;
    const rocsparse_sptrsv_alg  alg         = arg.sptrsv_alg;
    const rocsparse_diag_type   diag        = arg.diag;
    const rocsparse_fill_mode   uplo        = arg.uplo;
    const rocsparse_matrix_type matrix_type = arg.matrix_type;

    //
    // Create handle.
    //
    rocsparse_local_handle handle(arg);

    //
    // Create host matrix.
    //
    host_coo_matrix<T, I> hA;
    {
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg);
        matrix_factory.init_coo(hA);
    }

    const I M = hA.m;
    if(M != hA.n)
    {
        return;
    }

    //
    // Create host data.
    //
    host_scalar<T> halpha(arg.get_alpha<T>());

    host_dense_vector<T> hx(M);
    rocsparse_init<T>(hx, M, 1, 1);
    //
    // Create device data.
    //
    device_coo_matrix<T, I> dA(hA);
    device_dense_vector<T>  dx(hx);
    device_dense_vector<T>  dy(M);
    device_scalar<T>        dalpha(halpha);

    rocsparse_local_spmat A(dA);
    rocsparse_local_dnvec x(dx);
    rocsparse_local_dnvec y(dy);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
        A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

    rocsparse_error p_error[1] = {nullptr};

    rocsparse_sptrsv_descr sptrsv_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsv_descr(&sptrsv_descr));

    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_operation,
                                                     &trans_A,
                                                     sizeof(trans_A),
                                                     p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
        handle, sptrsv_descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), p_error));

    {
        const rocsparse_datatype ttype = get_datatype<T>();
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                         sptrsv_descr,
                                                         rocsparse_sptrsv_input_scalar_datatype,
                                                         &ttype,
                                                         sizeof(ttype),
                                                         p_error));
    }

    {
        const rocsparse_datatype ttype = get_datatype<T>();
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                         sptrsv_descr,
                                                         rocsparse_sptrsv_input_compute_datatype,
                                                         &ttype,
                                                         sizeof(ttype),
                                                         p_error));
    }

    {
        const rocsparse_analysis_policy apol = arg.apol;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                         sptrsv_descr,
                                                         rocsparse_sptrsv_input_analysis_policy,
                                                         &apol,
                                                         sizeof(apol),
                                                         p_error));
    }

    rocsparse_clients::sptrsv_analysis(handle, sptrsv_descr, A, x, y, p_error);

    if(arg.unit_check)
    {

        // CPU coosv
        host_dense_vector<T> hy(M);
        I                    analysis_pivot = -1;
        I                    solve_pivot    = -1;

        host_coosv(trans_A,
                   hA.m,
                   hA.nnz,
                   *halpha,
                   hA.row_ind.data(),
                   hA.col_ind.data(),
                   hA.val.data(),
                   hx.data(),
                   hy.data(),
                   diag,
                   uplo,
                   base,
                   &analysis_pivot,
                   &solve_pivot);

        const bool comparable = (analysis_pivot == -1 && solve_pivot == -1);

        rocsparse_clients::sptrsv_compute(
            handle, sptrsv_descr, A, x, y, rocsparse_pointer_mode_host, halpha, p_error);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }

        if(comparable)
        {
            hy.near_check(dy);
        }

        //
        // Solve on device.
        //
        rocsparse_clients::sptrsv_compute(
            handle, sptrsv_descr, A, x, y, rocsparse_pointer_mode_device, dalpha, p_error);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode device", dy);
        }

        if(comparable)
        {
            hy.near_check(dy);
        }
    }

    if(arg.timing)
    {
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(
            handle, sptrsv_descr, A, x, y, rocsparse_sptrsv_stage_compute, &buffer_size, p_error));
        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                         sptrsv_descr,
                                                         rocsparse_sptrsv_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(halpha.data()),
                                                         p_error));

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_sptrsv,
                                               handle,
                                               sptrsv_descr,
                                               A,
                                               x,
                                               y,
                                               rocsparse_sptrsv_stage_compute,
                                               buffer_size,
                                               buffer,
                                               p_error);

        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

        const double gflop_count = spsv_gflop_count(hA.m, hA.nnz, diag);
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = coosv_gbyte_count<T>(hA.m, hA.nnz);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::nnz_A,
                            hA.nnz,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::algorithm,
                            rocsparse_sptrsvalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                 \
    template void testing_sptrsv_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsv_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_sptrsv_coo_extra(const Arguments& arg) {}
