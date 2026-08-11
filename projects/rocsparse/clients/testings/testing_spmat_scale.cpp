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
void testing_spmat_scale_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle = local_handle;
    J                m      = safe_size;
    J                n      = safe_size;
    I                nnz    = safe_size;

    T local_alpha = static_cast<T>(1);

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

    // Sparse matrix descriptors: source (A) and target (C).
    rocsparse_local_spmat local_A(
        m, n, nnz, csr_row_ptr_A, csr_col_ind_A, csr_val_A, itype, jtype, base, ttype);
    rocsparse_local_spmat local_C(
        m, n, nnz, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, ttype);

    rocsparse_const_spmat_descr source = local_A;
    rocsparse_spmat_descr       target = local_C;

    // alpha is a self-describing size-one scalar dense vector descriptor.
    rocsparse_local_dnvec       alpha_scalar(static_cast<int64_t>(1), &local_alpha, ttype);
    rocsparse_const_dnvec_descr alpha = alpha_scalar;

    // p_error is an optional output error descriptor and may be null.
    rocsparse_error p_error[1] = {nullptr};

    // Signature: rocsparse_spmat_scale(handle, alpha, source, target, p_error). p_error (arg 4)
    // is optional and not checked.
    {
        static const int nex   = 1;
        static const int ex[1] = {4};
        select_bad_arg_analysis(
            rocsparse_spmat_scale, nex, ex, handle, alpha, source, target, p_error);
    }

    // Consistency checks between source and target that the generic bad-arg harness does not
    // exercise.
    {
        // Format mismatch: source is CSR, target is COO.
        rocsparse_local_spmat local_C_coo(
            m, n, nnz, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, base, ttype);
        rocsparse_spmat_descr target_coo = local_C_coo;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_scale(handle, alpha, source, target_coo, p_error),
                                rocsparse_status_not_implemented);

        // Dimension mismatch: target has a different number of rows.
        rocsparse_local_spmat local_C_rows(
            m + 1, n, nnz, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, ttype);
        rocsparse_spmat_descr target_rows = local_C_rows;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_scale(handle, alpha, source, target_rows, p_error),
                                rocsparse_status_invalid_size);

        // Data-type mismatch between source and target.
        const rocsparse_datatype other_ttype = (ttype == rocsparse_datatype_f32_r)
                                                   ? rocsparse_datatype_f64_r
                                                   : rocsparse_datatype_f32_r;
        rocsparse_local_spmat    local_C_dtype(
            m, n, nnz, csr_row_ptr_C, csr_col_ind_C, csr_val_C, itype, jtype, base, other_ttype);
        rocsparse_spmat_descr target_dtype = local_C_dtype;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_scale(handle, alpha, source, target_dtype, p_error),
                                rocsparse_status_type_mismatch);

        // Note: a differing index base or index type between source and target is intentionally
        // accepted, since rocsparse_spmat_scale only operates on the value arrays.

        // Data-type mismatch between alpha and the matrices.
        rocsparse_local_dnvec       alpha_other(static_cast<int64_t>(1), &local_alpha, other_ttype);
        rocsparse_const_dnvec_descr alpha_bad = alpha_other;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_scale(handle, alpha_bad, source, target, p_error),
                                rocsparse_status_type_mismatch);
    }
}

// Generic driver shared by every format. It takes an already-initialized host matrix \p hA,
// allocates C with the same layout, runs rocsparse_spmat_scale with host and device scalar alpha
// (and in place) and validates the result against a host reference C = alpha * A.
template <typename T, typename HostMatrix, typename DeviceMatrix>
static void testing_spmat_scale_dispatch(const Arguments& arg, HostMatrix& hA)
{
    T h_alpha = arg.get_alpha<T>();

    device_vector<T> d_alpha(1);
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // alpha as a self-describing scalar descriptor, in host and device memory.
    rocsparse_dnvec_descr alpha_host;
    CHECK_ROCSPARSE_ERROR(rocsparse_dnvec_descr_create_scalar(handle,
                                                              &alpha_host,
                                                              rocsparse_pointer_mode_host,
                                                              get_datatype<T>(),
                                                              &h_alpha,
                                                              &h_alpha,
                                                              nullptr));
    rocsparse_dnvec_descr alpha_device;
    CHECK_ROCSPARSE_ERROR(rocsparse_dnvec_descr_create_scalar(handle,
                                                              &alpha_device,
                                                              rocsparse_pointer_mode_device,
                                                              get_datatype<T>(),
                                                              (const void*)(const T*)d_alpha,
                                                              (void*)(T*)d_alpha,
                                                              nullptr));

    // Declare device matrix A (the source).
    DeviceMatrix dA(hA);

    // Target C carries the same sparsity pattern as A (spmat_scale does not copy the pattern; it
    // only writes C's values = alpha * A's values).
    DeviceMatrix dC(dA);

    rocsparse_local_spmat mat_A(dA), mat_C(dC);

    if(arg.unit_check)
    {
        // Compute C on host: C = alpha * A (same layout as A, values scaled by alpha).
        HostMatrix   hC(hA);
        const size_t nvalues = hC.val.size();
        for(size_t i = 0; i < nvalues; ++i)
        {
            hC.val[i] = h_alpha * hA.val[i];
        }

        // Out-of-place, host scalar alpha (needed by the hipSPARSE SpGEAM use case).
        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmat_scale(handle, alpha_host, mat_A, mat_C, nullptr));
            hC.near_check(dC);
        }

        // Out-of-place, device scalar alpha.
        for(int32_t i = 0; i < 2; i++)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spmat_scale(handle, alpha_device, mat_A, mat_C, nullptr));
            hC.near_check(dC);
        }

        // In-place scaling: target == source. Use a fresh copy of A so the reference still holds.
        DeviceMatrix          dInplace(dA);
        rocsparse_local_spmat mat_inplace(dInplace);
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_scale(handle, alpha_host, mat_inplace, mat_inplace, nullptr));
        hC.near_check(dInplace);
    }

    if(arg.timing)
    {
        int32_t number_cold_calls = 2;
        int32_t number_hot_calls  = arg.iters;

        // Warm up
        for(int32_t iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmat_scale(handle, alpha_host, mat_A, mat_C, nullptr));
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int32_t iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmat_scale(handle, alpha_host, mat_A, mat_C, nullptr));
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

    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(alpha_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_dnvec_descr(alpha_device));
}

template <typename I, typename J, typename T>
void testing_spmat_scale(const Arguments& arg)
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
        testing_spmat_scale_dispatch<T, host_coo_matrix<T, I>, device_coo_matrix<T, I>>(arg, hA);
        break;
    }
    case rocsparse_format_coo_aos:
    {
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_coo_aos_matrix<T, I>         hA;
        matrix_factory.init_coo_aos(hA, m, n, base);
        testing_spmat_scale_dispatch<T, host_coo_aos_matrix<T, I>, device_coo_aos_matrix<T, I>>(arg,
                                                                                                hA);
        break;
    }
    case rocsparse_format_csr:
    {
        J                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        host_csr_matrix<T, I, J>          hA;
        matrix_factory.init_csr(hA, m, n, base);
        testing_spmat_scale_dispatch<T, host_csr_matrix<T, I, J>, device_csr_matrix<T, I, J>>(arg,
                                                                                              hA);
        break;
    }
    case rocsparse_format_csc:
    {
        J                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg, to_int, full_rank);
        host_csc_matrix<T, I, J>          hA;
        matrix_factory.init_csc(hA, m, n, base);
        testing_spmat_scale_dispatch<T, host_csc_matrix<T, I, J>, device_csc_matrix<T, I, J>>(arg,
                                                                                              hA);
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
        testing_spmat_scale_dispatch<T, host_gebsr_matrix<T, I, J>, device_gebsr_matrix<T, I, J>>(
            arg, hA);
        break;
    }
    case rocsparse_format_ell:
    {
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_ell_matrix<T, I>             hA;
        matrix_factory.init_ell(hA, m, n, base);
        testing_spmat_scale_dispatch<T, host_ell_matrix<T, I>, device_ell_matrix<T, I>>(arg, hA);
        break;
    }
    case rocsparse_format_sell:
    {
        // The SELL matrix factory requires a single index type (I == J).
        I                                 m = arg.M, n = arg.N;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_sell_matrix<T, I, I>         hA;
        matrix_factory.init_sell(hA, m, n, arg.sell_slice_size, base);
        testing_spmat_scale_dispatch<T, host_sell_matrix<T, I, I>, device_sell_matrix<T, I, I>>(arg,
                                                                                                hA);
        break;
    }
    case rocsparse_format_bell:
    {
        // The Blocked-ELL matrix factory requires a single index type (I == J).
        I                                 m = arg.M, n = arg.N;
        I                                 block_dim = arg.block_dim;
        rocsparse_matrix_factory<T, I, I> matrix_factory(arg, to_int, full_rank);
        host_bell_matrix<T, I>            hA;
        matrix_factory.init_bell(hA, m, n, block_dim, base);
        testing_spmat_scale_dispatch<T, host_bell_matrix<T, I>, device_bell_matrix<T, I>>(arg, hA);
        break;
    }
    }
}

void testing_spmat_scale_extra(const Arguments& arg) {}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                  \
    template void testing_spmat_scale_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmat_scale<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
