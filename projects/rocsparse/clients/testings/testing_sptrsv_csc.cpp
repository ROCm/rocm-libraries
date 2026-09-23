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
#include "rocsparse_clients_dnvec_descr.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "rocsparse_clients_sptrsv.hpp"
#include "testing.hpp"

namespace
{
    //
    // Batched CPU reference for a CSC triangular solve.
    //
    // A CSC matrix is the transpose of the CSR matrix sharing the same arrays, so
    // host_cscsv internally forwards to host_csrsv with the transpose operation and
    // the fill mode flipped (and, for the conjugate transpose, the values
    // conjugated). Each batch entry is solved independently using the per-batch
    // strides of the matrix values and of the x / y dense vectors.
    //
    template <typename T, typename I, typename J>
    void sptrsv_host_csc(int64_t                                  batch_count,
                         const T*                                 halpha,
                         rocsparse_operation                      operation,
                         rocsparse_clients::spmat_descr<T, I, J>& A,
                         const rocsparse_clients::dnvec_descr<T>& x,
                         rocsparse_clients::dnvec_descr<T>&       y,
                         const rocsparse_diag_type                diag,
                         const rocsparse_fill_mode                uplo,
                         int64_t*                                 symbolic,
                         int64_t*                                 exact)
    {
        auto& host = A.template as<rocsparse_format_csc>().host();

        for(int64_t i = 0; i < batch_count; ++i)
        {
            const T* p_val = host.val.data() + i * A.get_stride();
            const T* p_hx  = x.host().data() + i * x.get_stride();
            T*       p_hy  = y.host().data() + i * y.get_stride();

            J analysis_pivot = -1;
            J solve_pivot    = -1;
            host_cscsv<I, J, T>(operation,
                                host.m,
                                host.nnz,
                                *halpha,
                                host.ptr,
                                host.ind,
                                p_val,
                                p_hx,
                                (int64_t)1,
                                p_hy,
                                diag,
                                uplo,
                                host.base,
                                &analysis_pivot,
                                &solve_pivot);

            symbolic[i] = analysis_pivot;
            exact[i]    = solve_pivot;
        }
    }
}

template <typename I, typename J, typename T>
void testing_sptrsv_csc_bad_arg(const Arguments& arg)
{
    //
    // Bad args of sptrsv is already tested in testing_sptrsv_csr_bad_arg
    //
}

template <typename I, typename J, typename T>
void testing_sptrsv_csc(const Arguments& arg)
{
    rocsparse_error* p_error = nullptr;

    if(arg.M != arg.N)
    {
        return;
    }

    const int64_t batch_count   = (arg.batch_count > 1) ? arg.batch_count : 1;
    const int64_t batch_count_A = (arg.batch_count_A > 0) ? arg.batch_count_A : batch_count;
    const int64_t batch_count_x = (arg.batch_count_B > 0) ? arg.batch_count_B : batch_count;

    bool full_rank = true;

    rocsparse_clients::spmat_descr<T, I, J> A(arg, batch_count_A, full_rank);
    if(false == A.is_square())
    {
        return;
    }
    const int64_t                     M = A.get_nrows();
    rocsparse_clients::dnvec_descr<T> x(M, batch_count_x, M);
    rocsparse_clients::dnvec_descr<T> y(M, batch_count, M);

    host_scalar<T>   halpha(arg.get_alpha<T>());
    device_scalar<T> dalpha(halpha);

    const rocsparse_analysis_policy apol        = arg.apol;
    const rocsparse_datatype        ttype       = get_datatype<T>();
    const rocsparse_operation       operation   = arg.transA;
    const rocsparse_sptrsv_alg      alg         = arg.sptrsv_alg;
    const rocsparse_diag_type       diag        = arg.diag;
    const rocsparse_fill_mode       uplo        = arg.uplo;
    const rocsparse_matrix_type     matrix_type = arg.matrix_type;

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
        A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

    //
    // Create handle.
    //
    rocsparse_local_handle handle(arg);
    hipStream_t            stream{};
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    // The batch count is carried by the strided-batch attributes of the sparse
    // matrix and the dense vectors, so the sptrsv descriptor itself only needs the
    // operation, algorithm, datatypes and analysis policy.
    rocsparse_sptrsv_descr sptrsv_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsv_descr(&sptrsv_descr));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_operation,
                                                     &operation,
                                                     sizeof(operation),
                                                     p_error));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
        handle, sptrsv_descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), p_error));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_scalar_datatype,
                                                     &ttype,
                                                     sizeof(ttype),
                                                     p_error));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_compute_datatype,
                                                     &ttype,
                                                     sizeof(ttype),
                                                     p_error));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_analysis_policy,
                                                     &apol,
                                                     sizeof(apol),
                                                     p_error));

    rocsparse_clients::sptrsv_analysis(handle, sptrsv_descr, A, x, y, p_error);

    host_dense_vector<int64_t> host_symbolic_position(batch_count);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_get_output(handle,
                                                      sptrsv_descr,
                                                      rocsparse_sptrsv_output_singularity_position,
                                                      host_symbolic_position,
                                                      sizeof(int64_t),
                                                      p_error));

    {
        device_dense_vector<int64_t> device_symbolic_position(batch_count);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_sptrsv_get_output(handle,
                                        sptrsv_descr,
                                        rocsparse_sptrsv_output_singularity_position,
                                        device_symbolic_position,
                                        sizeof(int64_t),
                                        p_error));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        host_symbolic_position.unit_check(device_symbolic_position);
    }

    if(arg.unit_check)
    {
        host_dense_vector<int64_t> cpu_symbolic_position(batch_count);
        host_dense_vector<int64_t> cpu_numeric_position(batch_count);

        // CPU gold check: host_cscsv wraps host_csrsv, handling the CSC->CSR
        // mapping (transpose operation, flipped fill mode and, for the conjugate
        // transpose, the value conjugation) internally, once per batch entry.
        sptrsv_host_csc<T, I, J>(batch_count,
                                 halpha,
                                 operation,
                                 A,
                                 x,
                                 y,
                                 diag,
                                 uplo,
                                 cpu_symbolic_position,
                                 cpu_numeric_position);

        for(auto mode : {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
        {
            void* alpha = (mode == rocsparse_pointer_mode_host) ? halpha : dalpha;

            rocsparse_clients::sptrsv_compute(handle, sptrsv_descr, A, x, y, mode, alpha, p_error);

            host_dense_vector<int64_t> host_numeric_position(batch_count);
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_sptrsv_get_output(handle,
                                            sptrsv_descr,
                                            rocsparse_sptrsv_output_singularity_position,
                                            host_numeric_position,
                                            sizeof(int64_t),
                                            p_error));
            {
                device_dense_vector<int64_t> device_numeric_position(batch_count);

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsv_get_output(handle,
                                                sptrsv_descr,
                                                rocsparse_sptrsv_output_singularity_position,
                                                device_numeric_position,
                                                sizeof(int64_t),
                                                p_error));

                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                host_numeric_position.unit_check(device_numeric_position);
            }

            cpu_symbolic_position.unit_check(host_symbolic_position);
            cpu_numeric_position.unit_check(host_numeric_position);

            if(ROCSPARSE_REPRODUCIBILITY)
            {
                if(rocsparse_pointer_mode_host == mode)
                {
                    rocsparse_reproducibility::save("Y pointer mode host", y.device());
                }
                else
                {
                    rocsparse_reproducibility::save("Y pointer mode device", y.device());
                }
            }

            y.near_check_values(host_symbolic_position, host_numeric_position);
        }
    }

    if(arg.timing)
    {
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(
            handle, sptrsv_descr, A, x, y, rocsparse_sptrsv_stage_compute, &buffer_size, p_error));
        device_dense_vector<char> buffer(buffer_size);

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

        int64_t A_m   = A.get_nrows();
        int64_t A_nnz = 0;
        {
            auto& device = A.template as<rocsparse_format_csc>().device();
            A_nnz        = device.nnz;
        }

        const double gflop_count = spsv_gflop_count(A_m, A_nnz, diag);
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = csrsv_gbyte_count<T>(A_m, A_nnz);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            A_m,
                            display_key_t::nnz_A,
                            A_nnz,
                            display_key_t::alpha,
                            *halpha,
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_sptrsv_csc_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsv_csc<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

void testing_sptrsv_csc_extra(const Arguments& arg) {}
