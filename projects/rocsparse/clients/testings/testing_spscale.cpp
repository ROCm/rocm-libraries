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

#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_spscale_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle = local_handle;
    J                m      = safe_size;
    J                n      = safe_size;
    I                nnz    = safe_size;

    const T     local_alpha = static_cast<T>(1);
    const void* alpha       = (const void*)&local_alpha;

    void* csr_row_ptr_A = (void*)0x4;
    void* csr_col_ind_A = (void*)0x4;
    void* csr_val_A     = (void*)0x4;
    void* csr_row_ptr_C = (void*)0x4;
    void* csr_col_ind_C = (void*)0x4;
    void* csr_val_C     = (void*)0x4;

    rocsparse_index_base base = rocsparse_index_base_zero;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // SpScale structures
    rocsparse_local_spmat local_A(
        m, n, nnz, csr_row_ptr_A, csr_col_ind_A, csr_val_A, itype, jtype, base, ttype);
    rocsparse_local_spmat local_C(
        m, n, nnz, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, ttype);

    rocsparse_spmat_descr mat_A = local_A;
    rocsparse_spmat_descr mat_C = local_C;

    size_t  local_buffer_size = 0;
    size_t* buffer_size       = &local_buffer_size;
    void*   temp_buffer       = (void*)0x4;

    // Buffer size: all arguments are checked.
    bad_arg_analysis(rocsparse_spscale_buffer_size, handle, alpha, mat_A, mat_C, buffer_size);

    // Compute: buffer_size (arg 4) and temp_buffer (arg 5) are not checked.
    {
        static const int nex   = 2;
        static const int ex[2] = {4, 5};
        select_bad_arg_analysis(rocsparse_spscale,
                                nex,
                                ex,
                                handle,
                                alpha,
                                mat_A,
                                mat_C,
                                local_buffer_size,
                                temp_buffer);
    }
}

// Generic driver shared by every format. It takes an already-initialized host matrix \p hA,
// allocates C with the same layout, runs rocsparse_spscale in host and device pointer mode and
// validates the result against a host reference C = alpha * A.
template <typename T, typename HostMatrix, typename DeviceMatrix>
static void testing_spscale_dispatch(const Arguments& arg, HostMatrix& hA)
{
    T  h_alpha     = arg.get_alpha<T>();
    T* h_alpha_ptr = &h_alpha;

    device_vector<T> d_alpha(1);
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    T* d_alpha_ptr = d_alpha;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Declare device matrix A.
    DeviceMatrix dA(hA);

    // Declare and set up C with the same layout as A but without copying its content: the
    // structure and the scaled values are produced by rocsparse_spscale itself.
    DeviceMatrix dC(dA, false);

    // Declare local spmat.
    rocsparse_local_spmat mat_A(dA), mat_C(dC);

    // Query buffer size and allocate.
    size_t buffer_size_in_bytes;
    void*  buffer;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spscale_buffer_size(handle, h_alpha_ptr, mat_A, mat_C, &buffer_size_in_bytes));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

    if(arg.unit_check)
    {
        // Compute C on host: C = alpha * A (same layout as A, values scaled by alpha).
        HostMatrix   hC(hA);
        const size_t nvalues = hC.val.size();
        for(size_t i = 0; i < nvalues; ++i)
        {
            hC.val[i] = h_alpha * hA.val[i];
        }

        // Compute C on device multiple times (host pointer mode).
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spscale(handle, h_alpha_ptr, mat_A, mat_C, buffer_size_in_bytes, buffer));
            hC.near_check(dC);
        }

        // Compute C on device multiple times (device pointer mode).
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spscale(handle, d_alpha_ptr, mat_A, mat_C, buffer_size_in_bytes, buffer));
            hC.near_check(dC);
        }
    }

    if(arg.timing)
    {
        int32_t number_cold_calls = 2;
        int32_t number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int32_t iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spscale(handle, h_alpha_ptr, mat_A, mat_C, buffer_size_in_bytes, buffer));
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int32_t iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spscale(handle, h_alpha_ptr, mat_A, mat_C, buffer_size_in_bytes, buffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        // C = alpha * A : read A's values and write C's values.
        const double nvalues     = static_cast<double>(hA.val.size());
        double       gbyte_count = (sizeof(T) * (2.0 * nvalues)) / 1e9;

        double gpu_gbyte = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            arg.M,
                            display_key_t::N,
                            arg.N,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
}

template <typename I, typename J, typename T>
void testing_spscale(const Arguments& arg)
{
    rocsparse_index_base base = arg.baseA;

    const bool            to_int    = arg.timing ? false : true;
    static constexpr bool full_rank = false;

    switch(arg.formatA)
    {
    case rocsparse_format_coo:
    {
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_coo_matrix<T, I>             hA;
        matrix_factory.init_coo(hA, m, n, base);
        testing_spscale_dispatch<T, host_coo_matrix<T, I>, device_coo_matrix<T, I>>(arg, hA);
        break;
    }
    case rocsparse_format_coo_aos:
    {
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_coo_aos_matrix<T, I>         hA;
        matrix_factory.init_coo_aos(hA, m, n, base);
        testing_spscale_dispatch<T, host_coo_aos_matrix<T, I>, device_coo_aos_matrix<T, I>>(arg,
                                                                                            hA);
        break;
    }
    case rocsparse_format_csr:
    {
        J                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        host_csr_matrix<T, I, J>          hA;
        matrix_factory.init_csr(hA, m, n, base);
        testing_spscale_dispatch<T, host_csr_matrix<T, I, J>, device_csr_matrix<T, I, J>>(arg, hA);
        break;
    }
    case rocsparse_format_csc:
    {
        J                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        host_csc_matrix<T, I, J>          hA;
        matrix_factory.init_csc(hA, m, n, base);
        testing_spscale_dispatch<T, host_csc_matrix<T, I, J>, device_csc_matrix<T, I, J>>(arg, hA);
        break;
    }
    case rocsparse_format_bsr:
    {
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        host_gebsr_matrix<T, I, J>        hA;
        J                                 block_dim = arg.block_dim;
        J                                 mb        = (arg.M + block_dim - 1) / block_dim;
        J                                 nb        = (arg.N + block_dim - 1) / block_dim;
        matrix_factory.init_gebsr(hA, mb, nb, block_dim, block_dim, base);
        testing_spscale_dispatch<T, host_gebsr_matrix<T, I, J>, device_gebsr_matrix<T, I, J>>(arg,
                                                                                              hA);
        break;
    }
    case rocsparse_format_ell:
    {
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_ell_matrix<T, I>             hA;
        matrix_factory.init_ell(hA, m, n, base);
        testing_spscale_dispatch<T, host_ell_matrix<T, I>, device_ell_matrix<T, I>>(arg, hA);
        break;
    }
    case rocsparse_format_sell:
    {
        // The SELL matrix factory requires a single index type (I == J).
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_sell_matrix<T, I, I>         hA;
        matrix_factory.init_sell(hA, m, n, arg.sell_slice_size, base);
        testing_spscale_dispatch<T, host_sell_matrix<T, I, I>, device_sell_matrix<T, I, I>>(arg,
                                                                                            hA);
        break;
    }
    case rocsparse_format_bell:
    {
        // Blocked-ELL is not supported by rocsparse_spscale yet and is excluded from the tests.
        CHECK_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        break;
    }
    }
}

void testing_spscale_extra(const Arguments& arg) {}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template void testing_spscale_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spscale<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
