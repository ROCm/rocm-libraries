/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

//
// Host extract function.
//
template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
static void test_extract_compute_csx(const host_csx_matrix<DIRECTION, T, I, J>& S_,
                                     host_csx_matrix<DIRECTION, T, I, J>&       T_,
                                     rocsparse_fill_mode                        T_fill_mode_,
                                     rocsparse_diag_type                        T_diag_)
{
    const int64_t size = (DIRECTION == rocsparse_direction_row) ? S_.m : S_.n;

    auto predicate = [T_fill_mode_, T_diag_](J i, J j) {
        return (DIRECTION == rocsparse_direction_row)
                   ? ((T_fill_mode_ == rocsparse_fill_mode_lower)
                          ? ((T_diag_ == rocsparse_diag_type_unit) ? (i > j) : (i >= j))
                          : ((T_diag_ == rocsparse_diag_type_unit) ? (i < j) : (i <= j)))
                   : ((T_fill_mode_ == rocsparse_fill_mode_lower)
                          ? ((T_diag_ == rocsparse_diag_type_unit) ? (i < j) : (i <= j))
                          : ((T_diag_ == rocsparse_diag_type_unit) ? (i > j) : (i >= j)));
    };

    for(J i = 0; i <= size; ++i)
    {
        T_.ptr[i] = 0;
    }

    for(J i = 0; i < size; ++i)
    {
        for(I k = S_.ptr[i] - S_.base; k < S_.ptr[i + 1] - S_.base; ++k)
        {
            const J j = S_.ind[k] - S_.base;
            if(predicate(i, j))
            {
                T_.ptr[i + 1] += 1;
            }
        }
    }

    for(J i = 2; i <= size; ++i)
    {
        T_.ptr[i] += T_.ptr[i - 1];
    }

    const I nnz = T_.ptr[size];
    T_.define(T_.m, T_.n, nnz, T_.base);

    for(J i = 0; i < size; ++i)
    {
        for(I k = S_.ptr[i] - S_.base; k < S_.ptr[i + 1] - S_.base; ++k)
        {
            const J j = S_.ind[k] - S_.base;
            if(predicate(i, j))
            {
                T_.ind[T_.ptr[i]] = j + T_.base;
                T_.val[T_.ptr[i]] = S_.val[k];
                T_.ptr[i] += 1;
            }
        }
    }

    for(J i = size; i > 0; --i)
    {
        T_.ptr[i] = T_.ptr[i - 1];
    }
    T_.ptr[0] = 0;

    for(J i = 0; i <= size; ++i)
    {
        T_.ptr[i] += T_.base;
    }
}

//
// Testing bad arguments.
//
template <typename I, typename J, typename T>
void testing_extract_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle  local_handle;
    rocsparse_spmat_descr   source            = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr   target            = (rocsparse_spmat_descr)0x4;
    rocsparse_handle        handle            = local_handle;
    rocsparse_extract_alg   alg               = rocsparse_extract_alg_default;
    rocsparse_extract_stage stage             = rocsparse_extract_stage_analysis;
    size_t                  local_buffer_size = 100;
    void*                   buffer            = (void*)0x4;
    {
        rocsparse_extract_descr* descr = (rocsparse_extract_descr*)0x4;
        bad_arg_analysis(rocsparse_create_extract_descr, descr, source, target, alg);
    }

    {
        rocsparse_extract_descr descr = (rocsparse_extract_descr)0x4;

        {
            size_t* buffer_size_in_bytes = &local_buffer_size;
            bad_arg_analysis(rocsparse_extract_buffer_size,
                             handle,
                             descr,
                             source,
                             target,
                             stage,
                             buffer_size_in_bytes);
        }

        {
            size_t               buffer_size_in_bytes              = local_buffer_size;
            static constexpr int nargs_to_exclude                  = 1;
            static constexpr int args_to_exclude[nargs_to_exclude] = {5};

            select_bad_arg_analysis(rocsparse_extract,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    handle,
                                    descr,
                                    source,
                                    target,
                                    stage,
                                    buffer_size_in_bytes,
                                    buffer);
        }
    }
}

template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
static void testing_rocsparse_spmat_extract(rocsparse_handle                       handle,
                                            rocsparse_const_spmat_descr            A,
                                            device_csx_matrix<DIRECTION, T, I, J>& device_B,
                                            rocsparse_spmat_descr                  B,
                                            rocsparse_extract_alg                  alg)
{

    hipStream_t stream;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    rocsparse_extract_descr extract_descr;

    //
    // Create descriptor.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_create_extract_descr(&extract_descr, A, B, alg));
    //
    // Analysis: get buffer size.
    //
    size_t buffer_size = 0;
    void*  buffer      = nullptr;
    CHECK_ROCSPARSE_ERROR(rocsparse_extract_buffer_size(
        handle, extract_descr, A, B, rocsparse_extract_stage_analysis, &buffer_size));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    //
    // Analysis: execute.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_extract(
        handle, extract_descr, A, B, rocsparse_extract_stage_analysis, buffer_size, buffer));

    int64_t nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_extract_nnz(handle, extract_descr, &nnz));

    //
    // Synchronize to get nnz.
    //
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    buffer      = nullptr;
    buffer_size = 0;

    //
    // Realloc.
    //
    device_B.define(device_B.m, device_B.n, nnz, device_B.base);

    rocsparse_format B_format;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_format(B, &B_format));
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_nnz(B, nnz));
    switch(B_format)
    {
    case rocsparse_format_csr:
    {
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr_set_pointers(B, device_B.ptr, device_B.ind, device_B.val));
        break;
    }

    case rocsparse_format_csc:
    {
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csc_set_pointers(B, device_B.ptr, device_B.ind, device_B.val));
        break;
    }

    case rocsparse_format_bell:
    case rocsparse_format_ell:
    case rocsparse_format_bsr:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        break;
    }
    }

    //
    // Compute: get buffer size.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_extract_buffer_size(
        handle, extract_descr, A, B, rocsparse_extract_stage_compute, &buffer_size));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    //
    // Compute: execute.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_extract(
        handle, extract_descr, A, B, rocsparse_extract_stage_compute, buffer_size, buffer));

    //
    // Synchronize.
    //
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    //
    // Destroy algorithm descriptor.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_extract_descr(extract_descr));
}

template <typename I, typename J, typename T>
static void init_csx(const Arguments&                   arg,
                     host_csr_matrix<T, I, J>&          A,
                     rocsparse_matrix_factory<T, I, J>& matrix_factory)
{
    J m = arg.M;
    J n = arg.N;
    matrix_factory.init_csr(A, m, n, arg.baseA);
}

template <typename I, typename J, typename T>
static void init_csx(const Arguments&                   arg,
                     host_csc_matrix<T, I, J>&          A,
                     rocsparse_matrix_factory<T, I, J>& matrix_factory)
{
    J m = arg.M;
    J n = arg.N;
    matrix_factory.init_csc(A, m, n, arg.baseA);
}

template <rocsparse_direction DIRECTION, typename I, typename J, typename T>
static void testing_extract_csx_template(const Arguments& arg)
{
    rocsparse_local_handle      handle(arg);
    const rocsparse_extract_alg alg = rocsparse_extract_alg_default;

    //
    // Create matrix A.
    //
    host_csx_matrix<DIRECTION, T, I, J> host_A;
    {
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);
        init_csx(arg, host_A, matrix_factory);
    }

    device_csx_matrix<DIRECTION, T, I, J> device_A(host_A);
    rocsparse_local_spmat                 A(device_A);

    //
    // Create matrix B.
    //
    host_csx_matrix<DIRECTION, T, I, J>   host_B(host_A.m, host_A.n, 0, arg.baseB);
    device_csx_matrix<DIRECTION, T, I, J> device_B(host_B);
    rocsparse_local_spmat                 B(device_B);

    const rocsparse_fill_mode fill_mode = arg.uplo;
    const rocsparse_diag_type diag_type = arg.diag;

    //
    // Set attributes.
    //
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
            B, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
            B, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));

        rocsparse_matrix_type matrix_type = rocsparse_matrix_type_triangular;
        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
            B, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));
    }

    if(arg.unit_check)
    {
        //
        // Do the extraction.
        //

        testing_rocsparse_spmat_extract(handle, A, device_B, B, alg);

        //
        // Host calculation.
        //

        test_extract_compute_csx(host_A, host_B, fill_mode, diag_type);
        //
        // Compare.
        //
        host_B.unit_check(device_B);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            //
            // To fill
            //
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            //
            // To fill
            //
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = 0; // csr2csc_gbyte_count<T>(M, N, nnz, action);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            host_A.m,
                            display_key_t::N,
                            host_A.n,
                            display_key_t::nnz,
                            host_A.nnz,
                            display_key_t::fill_mode,
                            rocsparse_fillmode2string(arg.uplo),
                            display_key_t::diag_type,
                            rocsparse_diagtype2string(arg.diag),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

template <typename I, typename J, typename T>
void testing_extract(const Arguments& arg)
{
    switch(arg.formatA)
    {
    case rocsparse_format_csr:
    {
        testing_extract_csx_template<rocsparse_direction_row, I, J, T>(arg);
        break;
    }
    case rocsparse_format_csc:
    {
        testing_extract_csx_template<rocsparse_direction_column, I, J, T>(arg);
        break;
    }
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    case rocsparse_format_bsr:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        std::cerr << "not implemented" << std::endl;
        break;
    }
    }
}

void testing_extract_extra(const Arguments& arg) {}

#define INSTANTIATE(ITYPE, JTYPE, TYPE)                                              \
    template void testing_extract_bad_arg<ITYPE, JTYPE, TYPE>(const Arguments& arg); \
    template void testing_extract<ITYPE, JTYPE, TYPE>(const Arguments& arg)

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
