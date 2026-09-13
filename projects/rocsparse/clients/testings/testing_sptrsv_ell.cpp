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

#include "rocsparse_clients_sptrsv.hpp"
#include "testing.hpp"

// Reads the generic singularity outputs in host and in device pointer mode and
// returns them, after checking that both pointer modes report the same thing.
static void query_sptrsv_ell_singularity(rocsparse_handle       handle,
                                         rocsparse_sptrsv_descr sptrsv_descr,
                                         rocsparse_error*       p_error,
                                         int64_t*               position,
                                         rocsparse_singularity* type)
{
    hipStream_t stream{};
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    rocsparse_pointer_mode pointer_mode;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &pointer_mode));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_get_output(handle,
                                                      sptrsv_descr,
                                                      rocsparse_sptrsv_output_singularity_position,
                                                      position,
                                                      sizeof(*position),
                                                      p_error));
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_get_output(
        handle, sptrsv_descr, rocsparse_sptrsv_output_singularity, type, sizeof(*type), p_error));

    int64_t               device_side_position = -2;
    rocsparse_singularity device_side_type     = static_cast<rocsparse_singularity>(-1);
    {
        device_dense_vector<int64_t> device_position(1);

        // device_dense_vector cannot be instantiated on an enumeration, its guard
        // pages are initialized from a random floating point value.
        rocsparse_singularity* device_type{};
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&device_type, sizeof(rocsparse_singularity)));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_sptrsv_get_output(handle,
                                        sptrsv_descr,
                                        rocsparse_sptrsv_output_singularity_position,
                                        device_position,
                                        sizeof(int64_t),
                                        p_error));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_get_output(handle,
                                                          sptrsv_descr,
                                                          rocsparse_sptrsv_output_singularity,
                                                          device_type,
                                                          sizeof(rocsparse_singularity),
                                                          p_error));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hipMemcpy(&device_side_position,
                                  device_position,
                                  sizeof(device_side_position),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            &device_side_type, device_type, sizeof(device_side_type), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(rocsparse_hipFree(device_type));
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, pointer_mode));

    if(device_side_position != *position || device_side_type != *type)
    {
        std::cout << "singularity output differs between pointer modes: host type " << *type
                  << " position " << *position << ", device type " << device_side_type
                  << " position " << device_side_position << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }
}

// For the cases below, where the singularity of the matrix is known up front.
static void expect_sptrsv_ell_singularity(rocsparse_handle       handle,
                                          rocsparse_sptrsv_descr sptrsv_descr,
                                          rocsparse_error*       p_error,
                                          int64_t                expected_position,
                                          rocsparse_singularity  expected_type)
{
    int64_t               position;
    rocsparse_singularity type;
    query_sptrsv_ell_singularity(handle, sptrsv_descr, p_error, &position, &type);

    if(position != expected_position || type != expected_type)
    {
        std::cout << "singularity output mismatch: expected type " << expected_type << " position "
                  << expected_position << ", got type " << type << " position " << position
                  << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

template <typename I, typename T>
void testing_sptrsv_ell_bad_arg(const Arguments& arg)
{
}

template <typename I, typename T>
void testing_sptrsv_ell(const Arguments& arg)
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

    I M = arg.M;
    I N = arg.N;

    rocsparse_local_handle handle(arg);

    host_ell_matrix<T, I> hA;
    {
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg);
        matrix_factory.init_ell(hA, M, N, base);
    }

    if(M != N)
    {
        return;
    }

    if(M != hA.n)
    {
        return;
    }

    I nnz_A = 0;
    for(I i = 0; i < M; ++i)
    {
        for(I p = 0; p < hA.width; ++p)
        {
            const int64_t idx = (int64_t)p * hA.m + i;
            const I       col = hA.ind[idx] - hA.base;
            if(col >= 0 && col < hA.n)
            {
                ++nnz_A;
            }
        }
    }

    host_scalar<T> halpha(arg.get_alpha<T>());

    host_dense_vector<T> hx(M);
    rocsparse_init<T>(hx, M, 1, 1);
    device_ell_matrix<T, I> dA(hA);
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

    int64_t          analysis_zero_pivot;
    rocsparse_status analysis_pivot_status
        = rocsparse_sptrsv_get_output(handle,
                                      sptrsv_descr,
                                      rocsparse_sptrsv_output_zero_pivot_position,
                                      &analysis_zero_pivot,
                                      sizeof(analysis_zero_pivot),
                                      p_error);
    if(analysis_pivot_status != rocsparse_status_zero_pivot)
    {
        CHECK_ROCSPARSE_ERROR(analysis_pivot_status);
    }
    // check consistency.
    if((analysis_pivot_status == rocsparse_status_zero_pivot) && (analysis_zero_pivot == -1))
    {
        std::cout << "inconsistent1 zero pivot detected during analysis status "
                  << analysis_pivot_status << " value " << analysis_zero_pivot << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    if((analysis_pivot_status != rocsparse_status_zero_pivot) && (analysis_zero_pivot != -1))
    {
        std::cout << "inconsistent2 zero pivot detected during analysis status "
                  << analysis_pivot_status << " value " << analysis_zero_pivot << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    // Analysis can only expose a structural singularity, and it must be the row
    // the zero-pivot output already reported.
    expect_sptrsv_ell_singularity(handle,
                                  sptrsv_descr,
                                  p_error,
                                  analysis_zero_pivot,
                                  (analysis_zero_pivot == -1) ? rocsparse_singularity_none
                                                              : rocsparse_singularity_symbolic);

    if(arg.unit_check)
    {
        host_dense_vector<T> hy(M);
        I                    analysis_pivot = -1;
        I                    solve_pivot    = -1;

        host_ellsv<I, T>(M,
                         hA.n,
                         *halpha,
                         hA.ind,
                         hA.val,
                         hA.width,
                         hx,
                         (int64_t)1,
                         hy,
                         diag,
                         uplo,
                         base,
                         &analysis_pivot,
                         &solve_pivot);

        if(analysis_zero_pivot != analysis_pivot)
        {
            std::cout << "analysis pivot failed: reference analysis pivot position = "
                      << analysis_pivot << ", calculated zero pivot position "
                      << analysis_zero_pivot << std::endl;
            CHECK_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }

        const bool comparable = (analysis_pivot == -1 && solve_pivot == -1);

        rocsparse_clients::sptrsv_compute(
            handle, sptrsv_descr, A, x, y, rocsparse_pointer_mode_host, halpha, p_error);

        int64_t          solve_zero_pivot;
        rocsparse_status solve_pivot_status
            = rocsparse_sptrsv_get_output(handle,
                                          sptrsv_descr,
                                          rocsparse_sptrsv_output_zero_pivot_position,
                                          &solve_zero_pivot,
                                          sizeof(solve_zero_pivot),
                                          p_error);
        // check consistency.
        if((solve_pivot_status == rocsparse_status_zero_pivot) && (solve_zero_pivot == -1))
        {
            std::cout << "inconsistent zero pivot detected during solve " << std::endl;
            CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

        if((solve_pivot_status != rocsparse_status_zero_pivot) && (solve_zero_pivot != -1))
        {
            std::cout << "inconsistent zero pivot detected during solve " << std::endl;
            CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

        if(solve_zero_pivot != solve_pivot)
        {
            std::cout << "solve pivot failed: reference solve pivot position = " << solve_pivot
                      << ", calculated zero pivot position " << solve_zero_pivot << std::endl;
            CHECK_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }

        // After compute the singularity can be structural or numeric depending on
        // which row comes first, so only the invariants are checked here: the
        // position agrees with the zero-pivot output, and a detected pivot is
        // classified instead of being reported as a clean matrix. The cases in
        // testing_sptrsv_ell_extra pin down the classification itself.
        {
            int64_t               position;
            rocsparse_singularity type;
            query_sptrsv_ell_singularity(handle, sptrsv_descr, p_error, &position, &type);

            const bool clean      = (type == rocsparse_singularity_none);
            const bool classified = (type == rocsparse_singularity_symbolic
                                     || type == rocsparse_singularity_numeric_exact);

            if(position != solve_zero_pivot || clean != (solve_zero_pivot == -1)
               || (!clean && !classified))
            {
                std::cout << "singularity output after compute disagrees with the zero-pivot "
                             "output: singularity type "
                          << type << " position " << position << ", zero pivot position "
                          << solve_zero_pivot << std::endl;
                CHECK_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
        }

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        if(comparable)
        {
            hy.near_check(dy);
        }

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

        const double gflop_count = spsv_gflop_count(hA.m, nnz_A, diag);
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = csrsv_gbyte_count<T>(hA.m, nnz_A);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            hA.m,
                            display_key_t::nnz_A,
                            nnz_A,
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
    template void testing_sptrsv_ell_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsv_ell<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

static void testing_sptrsv_ell_extra_case(const host_ell_matrix<float>& hA,
                                          int64_t                       expected_analysis_position,
                                          rocsparse_singularity         expected_analysis_type,
                                          int64_t                       expected_solve_position,
                                          rocsparse_singularity         expected_solve_type)
{
    const int32_t                   M           = hA.m;
    const rocsparse_operation       trans_A     = rocsparse_operation_none;
    const rocsparse_sptrsv_alg      alg         = rocsparse_sptrsv_alg_default;
    const rocsparse_diag_type       diag        = rocsparse_diag_type_non_unit;
    const rocsparse_fill_mode       uplo        = rocsparse_fill_mode_lower;
    const rocsparse_matrix_type     matrix_type = rocsparse_matrix_type_general;
    const rocsparse_analysis_policy apol        = rocsparse_analysis_policy_force;

    rocsparse_local_handle   handle;
    host_scalar<float>       halpha(1.0f);
    host_dense_vector<float> hx(M);
    rocsparse_init<float>(hx, M, 1, 1);

    device_ell_matrix<float>   dA(hA);
    device_dense_vector<float> dx(hx);
    device_dense_vector<float> dy(M);

    rocsparse_local_spmat A(dA);
    rocsparse_local_dnvec x(dx);
    rocsparse_local_dnvec y(dy);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
        A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

    rocsparse_error        p_error[1] = {nullptr};
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
        const rocsparse_datatype ttype = get_datatype<float>();
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
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                     sptrsv_descr,
                                                     rocsparse_sptrsv_input_analysis_policy,
                                                     &apol,
                                                     sizeof(apol),
                                                     p_error));

    rocsparse_clients::sptrsv_analysis(handle, sptrsv_descr, A, x, y, p_error);
    expect_sptrsv_ell_singularity(
        handle, sptrsv_descr, p_error, expected_analysis_position, expected_analysis_type);

    rocsparse_clients::sptrsv_compute(
        handle, sptrsv_descr, A, x, y, rocsparse_pointer_mode_host, halpha, p_error);
    expect_sptrsv_ell_singularity(
        handle, sptrsv_descr, p_error, expected_solve_position, expected_solve_type);

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
}

void testing_sptrsv_ell_extra(const Arguments&)
{
    auto pad = [](host_ell_matrix<float>& hA) {
        for(int32_t i = 0; i < hA.m * hA.width; ++i)
        {
            hA.ind[i] = static_cast<int32_t>(-1);
            hA.val[i] = 0.0f;
        }
    };
    auto set = [](host_ell_matrix<float>& hA, int32_t row, int32_t slot, int32_t col, float val) {
        hA.ind[(int64_t)slot * hA.m + row] = col;
        hA.val[(int64_t)slot * hA.m + row] = val;
    };

    {
        host_ell_matrix<float> hA(3, 3, 2, rocsparse_index_base_zero);
        pad(hA);
        set(hA, 0, 0, 0, 2.0f);
        set(hA, 1, 0, 0, 1.0f);
        set(hA, 1, 1, 1, 3.0f);
        set(hA, 2, 0, 2, 4.0f);
        testing_sptrsv_ell_extra_case(
            hA, -1, rocsparse_singularity_none, -1, rocsparse_singularity_none);
    }

    {
        // Missing diagonal at row 1: symbolic singularity after analysis.
        host_ell_matrix<float> hA(3, 3, 2, rocsparse_index_base_zero);
        pad(hA);
        set(hA, 0, 0, 0, 2.0f);
        set(hA, 1, 0, 0, 1.0f);
        set(hA, 2, 0, 1, 3.0f);
        set(hA, 2, 1, 2, 4.0f);
        testing_sptrsv_ell_extra_case(
            hA, 1, rocsparse_singularity_symbolic, 1, rocsparse_singularity_symbolic);
    }

    {
        // Explicit zero on the diagonal at row 1: numeric_exact after compute.
        host_ell_matrix<float> hA(3, 3, 2, rocsparse_index_base_zero);
        pad(hA);
        set(hA, 0, 0, 0, 2.0f);
        set(hA, 1, 0, 0, 1.0f);
        set(hA, 1, 1, 1, 0.0f);
        set(hA, 2, 0, 1, 3.0f);
        set(hA, 2, 1, 2, 4.0f);
        testing_sptrsv_ell_extra_case(
            hA, -1, rocsparse_singularity_none, 1, rocsparse_singularity_numeric_exact);
    }
}
