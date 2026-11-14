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

        // In-place solve: x and y are the same vector.
        if(batch_count_x == batch_count)
        {
            rocsparse_clients::dnvec_descr<T> xy(M, batch_count, M);
            xy.device().transfer_from(x.host());
            xy.host().transfer_from(y.host());
            void* alpha = halpha;
            rocsparse_clients::sptrsv_compute(
                handle, sptrsv_descr, A, xy, xy, rocsparse_pointer_mode_host, alpha, p_error);
            host_dense_vector<int64_t> numeric_position(batch_count);
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_sptrsv_get_output(handle,
                                            sptrsv_descr,
                                            rocsparse_sptrsv_output_singularity_position,
                                            numeric_position,
                                            sizeof(int64_t),
                                            p_error));
            xy.near_check_values(host_symbolic_position, numeric_position);
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

namespace
{
    // Small CSC matrix built on the host for the tests of testing_sptrsv_csc_extra.
    struct csc_case
    {
        int32_t                m{};
        int32_t                n{};
        std::vector<int32_t>   ptr;
        std::vector<int32_t>   ind;
        std::vector<double>    val;
        rocsparse_index_base   base{rocsparse_index_base_zero};
        rocsparse_fill_mode    uplo{rocsparse_fill_mode_lower};
        rocsparse_diag_type    diag{rocsparse_diag_type_non_unit};
        rocsparse_matrix_type  type{rocsparse_matrix_type_triangular};
        rocsparse_storage_mode storage{rocsparse_storage_mode_sorted};
    };

    // Banded m x m matrix with bandwidth 2; the diagonal entry of column missing is left out and
    // the one of column zero is set to zero (-1: none).
    csc_case banded_csc(int32_t m, int32_t missing, int32_t zero, rocsparse_index_base base)
    {
        csc_case c;
        c.m    = m;
        c.n    = m;
        c.base = base;
        c.ptr.push_back(base);
        for(int32_t j = 0; j < m; ++j)
        {
            for(int32_t i = std::max(0, j - 2); i <= std::min(m - 1, j + 2); ++i)
            {
                if(i == j && j == missing)
                {
                    continue;
                }
                c.ind.push_back(i + base);
                c.val.push_back((i == j) ? ((j == zero) ? 0.0 : 4.0) : 1.0 + 0.1 * (i - j));
            }
            c.ptr.push_back(static_cast<int32_t>(c.ind.size()) + base);
        }
        return c;
    }

    // Runs the analysis and the solve with x = 1. change_uplo is set on the matrix between the
    // analysis and the solve. With solve_diags, one solve per diagonal type, on the same
    // descriptor. status is the first status that is not success.
    void sptrsv_csc_run(const Arguments&                        arg,
                        const csc_case&                         c,
                        std::vector<double>&                    hy,
                        int64_t&                                symbolic,
                        int64_t&                                numeric,
                        rocsparse_status&                       status,
                        bool                                    change_uplo = false,
                        const std::vector<rocsparse_diag_type>& solve_diags = {})
    {
        rocsparse_error*       p_error = nullptr;
        rocsparse_local_handle handle(arg);
        const int32_t          nnz = static_cast<int32_t>(c.ind.size());

        device_vector<int32_t> dptr(c.n + 1);
        device_vector<int32_t> dind(std::max(nnz, 1));
        device_vector<double>  dval(std::max(nnz, 1));
        device_vector<double>  dx(std::max(c.m, c.n));
        device_vector<double>  dy(std::max(c.m, c.n));
        CHECK_HIP_ERROR(
            hipMemcpy(dptr, c.ptr.data(), sizeof(int32_t) * (c.n + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dind, c.ind.data(), sizeof(int32_t) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dval, c.val.data(), sizeof(double) * nnz, hipMemcpyHostToDevice));
        const std::vector<double> hx(std::max(c.m, c.n), 1.0);
        CHECK_HIP_ERROR(
            hipMemcpy(dx, hx.data(), sizeof(double) * hx.size(), hipMemcpyHostToDevice));

        const rocsparse_datatype ttype = rocsparse_datatype_f64_r;
        rocsparse_local_spmat    A(c.m,
                                c.n,
                                nnz,
                                dptr,
                                dind,
                                dval,
                                rocsparse_indextype_i32,
                                rocsparse_indextype_i32,
                                c.base,
                                ttype,
                                true);
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &c.uplo, sizeof(c.uplo)));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &c.diag, sizeof(c.diag)));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_matrix_type, &c.type, sizeof(c.type)));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
            A, rocsparse_spmat_storage_mode, &c.storage, sizeof(c.storage)));
        rocsparse_local_dnvec x(c.m, dx, ttype);
        rocsparse_local_dnvec y(c.m, dy, ttype);

        const rocsparse_operation       trans = rocsparse_operation_none;
        const rocsparse_sptrsv_alg      alg   = rocsparse_sptrsv_alg_default;
        const rocsparse_analysis_policy apol  = rocsparse_analysis_policy_force;
        rocsparse_sptrsv_descr          descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsv_descr(&descr));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_operation, &trans, sizeof(trans), p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_scalar_datatype, &ttype, sizeof(ttype), p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                         descr,
                                                         rocsparse_sptrsv_input_compute_datatype,
                                                         &ttype,
                                                         sizeof(ttype),
                                                         p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_analysis_policy, &apol, sizeof(apol), p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // The descriptor keeps a pointer to alpha until the solve.
        const double alpha  = 1.0;
        const void*  palpha = &alpha;
        status              = rocsparse_status_success;

        std::vector<std::pair<rocsparse_sptrsv_stage, rocsparse_diag_type>> stages{
            {rocsparse_sptrsv_stage_analysis, c.diag}};
        for(auto diag :
            solve_diags.empty() ? std::vector<rocsparse_diag_type>{c.diag} : solve_diags)
        {
            stages.emplace_back(rocsparse_sptrsv_stage_compute, diag);
        }
        for(auto [stage, diag] : stages)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));
            if(stage == rocsparse_sptrsv_stage_compute)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsv_set_input(handle,
                                               descr,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               palpha,
                                               sizeof(palpha),
                                               p_error));
                if(change_uplo)
                {
                    const rocsparse_fill_mode other = (c.uplo == rocsparse_fill_mode_lower)
                                                          ? rocsparse_fill_mode_upper
                                                          : rocsparse_fill_mode_lower;
                    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
                        A, rocsparse_spmat_fill_mode, &other, sizeof(other)));
                }
            }
            size_t buffer_size{};
            status = rocsparse_sptrsv_buffer_size(
                handle, descr, A, x, y, stage, &buffer_size, p_error);
            if(status != rocsparse_status_success)
            {
                break;
            }
            device_dense_vector<char> buffer(buffer_size);
            status = rocsparse_sptrsv(handle, descr, A, x, y, stage, buffer_size, buffer, p_error);
            if(status != rocsparse_status_success)
            {
                break;
            }
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            int64_t& position = (stage == rocsparse_sptrsv_stage_analysis) ? symbolic : numeric;
            CHECK_ROCSPARSE_ERROR(
                rocsparse_sptrsv_get_output(handle,
                                            descr,
                                            rocsparse_sptrsv_output_singularity_position,
                                            &position,
                                            sizeof(int64_t),
                                            p_error));
        }
        hy.resize(c.m);
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(double) * c.m, hipMemcpyDeviceToHost));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsv_descr(descr));
    }
}

// Missing and zero diagonal entries, and the arguments the CSC solve without transpose rejects.
void testing_sptrsv_csc_extra(const Arguments& arg)
{
    const int32_t m = 64;
    for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(auto uplo : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
        {
            for(auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
            {
                for(auto [missing, zero] : std::initializer_list<std::pair<int32_t, int32_t>>{
                        {-1, -1}, {17, -1}, {-1, 41}, {17, 41}})
                {
                    csc_case c = banded_csc(m, missing, zero, base);
                    c.uplo     = uplo;
                    c.diag     = diag;

                    std::vector<double> hy;
                    int64_t             symbolic = -2;
                    int64_t             numeric  = -2;
                    rocsparse_status    status;
                    sptrsv_csc_run(arg, c, hy, symbolic, numeric, status);
                    CHECK_ROCSPARSE_ERROR(status);

                    const std::vector<double> hx(m, 1.0);
                    std::vector<double>       hy_gold(m);
                    int32_t                   struct_pivot  = -1;
                    int32_t                   numeric_pivot = -1;
                    host_cscsv<int32_t, int32_t, double>(rocsparse_operation_none,
                                                         m,
                                                         static_cast<int32_t>(c.ind.size()),
                                                         1.0,
                                                         c.ptr.data(),
                                                         c.ind.data(),
                                                         c.val.data(),
                                                         hx.data(),
                                                         1,
                                                         hy_gold.data(),
                                                         diag,
                                                         uplo,
                                                         base,
                                                         &struct_pivot,
                                                         &numeric_pivot);
                    unit_check_scalar<int64_t>(struct_pivot, symbolic);
                    unit_check_scalar<int64_t>(numeric_pivot, numeric);
                    if(struct_pivot == -1 && numeric_pivot == -1)
                    {
                        near_check_segments<double>(m, hy_gold.data(), hy.data());
                    }
                }
            }
        }
    }

    std::vector<double> hy;
    int64_t             symbolic = -2;
    int64_t             numeric  = -2;
    rocsparse_status    status;
    {
        // Not square.
        csc_case c = banded_csc(8, -1, -1, rocsparse_index_base_zero);
        c.n        = 7;
        c.ptr.resize(8);
        c.ind.resize(c.ptr[7]);
        c.val.resize(c.ptr[7]);
        sptrsv_csc_run(arg, c, hy, symbolic, numeric, status);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_size, status);
    }
    {
        csc_case c = banded_csc(8, -1, -1, rocsparse_index_base_zero);
        c.type     = rocsparse_matrix_type_symmetric;
        sptrsv_csc_run(arg, c, hy, symbolic, numeric, status);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_not_implemented, status);
        c.type = rocsparse_matrix_type_hermitian;
        sptrsv_csc_run(arg, c, hy, symbolic, numeric, status);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_not_implemented, status);
    }
    {
        csc_case c = banded_csc(8, -1, -1, rocsparse_index_base_zero);
        c.storage  = rocsparse_storage_mode_unsorted;
        sptrsv_csc_run(arg, c, hy, symbolic, numeric, status);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_requires_sorted_storage, status);
    }
    {
        // The fill mode changes after the analysis: there is no analysis for the solve.
        csc_case c = banded_csc(8, -1, -1, rocsparse_index_base_zero);
        sptrsv_csc_run(arg, c, hy, symbolic, numeric, status, true);
        EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_pointer, status);
    }
    {
        // A unit-diagonal solve between the analysis and a non-unit solve keeps the structural
        // pivot of the analysis.
        for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        {
            csc_case c = banded_csc(64, 17, -1, base);
            sptrsv_csc_run(arg,
                           c,
                           hy,
                           symbolic,
                           numeric,
                           status,
                           false,
                           {rocsparse_diag_type_non_unit,
                            rocsparse_diag_type_unit,
                            rocsparse_diag_type_non_unit});
            CHECK_ROCSPARSE_ERROR(status);
            unit_check_scalar<int64_t>(17 + base, symbolic);
            unit_check_scalar<int64_t>(17 + base, numeric);
        }
    }
}
