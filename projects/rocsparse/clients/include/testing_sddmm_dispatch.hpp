/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "auto_testing_bad_arg.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_matrix_type_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csr_matrix<U, I, J>;
};

template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csc, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csc_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csc_matrix<U, I, J>;
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_matrix<U, I>;
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_aos_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_aos_matrix<U, I>;
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_ell, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_ell_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_ell_matrix<U, I>;
};

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename B,
          typename C,
          typename T>
struct testing_sddmm_dispatch_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename A, typename B, typename C, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_csr, I, J, A, B, C, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csr, I, J, C>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<C, I, J>& matrix_factory,
                                      host_sparse_matrix<C>&             hC,
                                      Ts&&... ts)
    {
        matrix_factory.init_csr(hC, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<A>& hA,
                                 const host_dense_matrix<B>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<C>&      hC)
    {
        rocsparse_host<T, I, J, A, B, C>::csrddmm(
            trans_A,
            trans_B,
            hA.order,
            hB.order,
            hC.m,
            hC.n,
            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
            hC.nnz,
            h_alpha,
            hA,
            hA.ld,
            hB,
            hB.ld,
            h_beta,
            hC.ptr,
            hC.ind,
            hC.val,
            hC.base);
    }
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename A, typename B, typename C, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_csc, I, J, A, B, C, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csc, I, J, C>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<C, I, J>& matrix_factory,
                                      host_sparse_matrix<C>&             hC,
                                      Ts&&... ts)
    {
        matrix_factory.init_csc(hC, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<A>& hA,
                                 const host_dense_matrix<B>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<C>&      hC)
    {
        rocsparse_host<T, I, J, A, B, C>::cscddmm(
            trans_A,
            trans_B,
            hA.order,
            hB.order,
            hC.m,
            hC.n,
            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
            hC.nnz,
            h_alpha,
            hA,
            hA.ld,
            hB,
            hB.ld,
            h_beta,
            hC.ptr,
            hC.ind,
            hC.val,
            hC.base);
    }
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename A, typename B, typename C, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_coo, I, I, A, B, C, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo, I, I, C>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<C, I, I>& matrix_factory,
                                      host_sparse_matrix<C>&             hC,
                                      Ts&&... ts)
    {
        matrix_factory.init_coo(hC, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<A>& hA,
                                 const host_dense_matrix<B>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<C>&      hC)
    {
        rocsparse_host<T, I, I, A, B, C>::cooddmm(
            trans_A,
            trans_B,
            hA.order,
            hB.order,
            hC.m,
            hC.n,
            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
            hC.nnz,
            h_alpha,
            hA,
            hA.ld,
            hB,
            hB.ld,
            h_beta,
            hC.row_ind,
            hC.col_ind,
            hC.val,
            hC.base);
    }
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename A, typename B, typename C, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_coo_aos, I, I, A, B, C, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, C>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<C, I, I>& matrix_factory,
                                      host_sparse_matrix<C>&             hC,
                                      Ts&&... ts)
    {
        matrix_factory.init_coo_aos(hC, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<A>& hA,
                                 const host_dense_matrix<B>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<C>&      hC)
    {
        rocsparse_host<T, I, I, A, B, C>::cooaosddmm(
            trans_A,
            trans_B,
            hA.order,
            hB.order,
            hC.m,
            hC.n,
            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
            hC.nnz,
            h_alpha,
            hA,
            hA.ld,
            hB,
            hB.ld,
            h_beta,
            hC.ind,
            hC.ind + 1,
            hC.val,
            hC.base);
    }
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename A, typename B, typename C, typename T>
struct testing_sddmm_dispatch_traits<rocsparse_format_ell, I, I, A, B, C, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_ell, I, I, C>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    template <typename... Ts>
    static void sparse_initialization(rocsparse_matrix_factory<C, I, I>& matrix_factory,
                                      host_sparse_matrix<C>&             hC,
                                      Ts&&... ts)
    {
        matrix_factory.init_ell(hC, ts...);
    }

    static void host_calculation(rocsparse_operation         trans_A,
                                 rocsparse_operation         trans_B,
                                 const T*                    h_alpha,
                                 const host_dense_matrix<A>& hA,
                                 const host_dense_matrix<B>& hB,
                                 const T*                    h_beta,
                                 host_sparse_matrix<C>&      hC)
    {
        rocsparse_host<T, I, I, A, B, C>::ellddmm(
            trans_A,
            trans_B,
            hA.order,
            hB.order,
            hC.m,
            hC.n,
            ((trans_A == rocsparse_operation_none) ? hA.n : hA.m),
            hC.nnz,
            h_alpha,
            hA,
            hA.ld,
            hB,
            hB.ld,
            h_beta,
            hC.width,
            hC.ind,
            hC.val,
            hC.base);
    }
};

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename B,
          typename C,
          typename T>
struct testing_sddmm_dispatch
{
private:
    using traits = testing_sddmm_dispatch_traits<FORMAT, I, J, A, B, C, T>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

public:
    static void testing_sddmm_bad_arg(const Arguments& arg)
    {
        const T     local_alpha = 0.6;
        const T     local_beta  = 0.1;
        const void* alpha       = (const void*)&local_alpha;
        const void* beta        = (const void*)&local_beta;

        rocsparse_local_handle local_handle;

        rocsparse_handle    handle  = local_handle;
        rocsparse_operation trans_A = rocsparse_operation_none;
        rocsparse_operation trans_B = rocsparse_operation_none;
        rocsparse_sddmm_alg alg     = rocsparse_sddmm_alg_default;
        size_t              local_buffer_size;
        size_t*             buffer_size  = &local_buffer_size;
        void*               temp_buffer  = (void*)0x4;
        rocsparse_datatype  compute_type = get_datatype<T>();

#define PARAMS_BUFFER_SIZE \
    handle, trans_A, trans_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg, buffer_size

#define PARAMS \
    handle, trans_A, trans_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg, temp_buffer

        //
        // AUTOMATIC BAD ARGS.
        //
        {
            device_dense_matrix<A>  dA;
            device_dense_matrix<B>  dB;
            rocsparse_local_dnmat   local_A(dA), local_B(dB);
            device_sparse_matrix<C> dC;
            rocsparse_local_spmat   local_C(dC);

            rocsparse_dnmat_descr mat_A = local_A;
            rocsparse_dnmat_descr mat_B = local_B;
            rocsparse_spmat_descr mat_C = local_C;

            static constexpr int num_to_exclude             = 1;
            static constexpr int to_exclude[num_to_exclude] = {10};
            bad_arg_analysis(rocsparse_sddmm_buffer_size, PARAMS_BUFFER_SIZE);
            select_bad_arg_analysis(rocsparse_sddmm_preprocess, num_to_exclude, to_exclude, PARAMS);
            select_bad_arg_analysis(rocsparse_sddmm, num_to_exclude, to_exclude, PARAMS);
        }

        // conjugate transpose
        {
            device_dense_matrix<A> dA;
            device_dense_matrix<B> dB;
            rocsparse_local_dnmat  local_A(dA), local_B(dB);

            device_sparse_matrix<C> dC;
            rocsparse_local_spmat   local_C(dC);

            rocsparse_dnmat_descr mat_A = local_A;
            rocsparse_dnmat_descr mat_B = local_B;
            rocsparse_spmat_descr mat_C = local_C;

            trans_A = rocsparse_operation_conjugate_transpose;
            trans_B = rocsparse_operation_none;

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS),
                                    rocsparse_status_not_implemented);

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_not_implemented);

            trans_A = rocsparse_operation_none;
            trans_B = rocsparse_operation_conjugate_transpose;

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_preprocess(PARAMS),
                                    rocsparse_status_not_implemented);

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_not_implemented);

            trans_A = rocsparse_operation_none;
            trans_B = rocsparse_operation_none;
        }

        // buffer size
        {
            device_dense_matrix<A> dA;
            device_dense_matrix<B> dB;
            rocsparse_local_dnmat  local_A(dA), local_B(dB);

            device_sparse_matrix<C> dC(10, 10, 10, rocsparse_index_base_zero);
            rocsparse_local_spmat   local_C(dC);

            rocsparse_dnmat_descr mat_A = local_A;
            rocsparse_dnmat_descr mat_B = local_B;
            rocsparse_spmat_descr mat_C = local_C;

            alg         = rocsparse_sddmm_alg_dense;
            temp_buffer = nullptr;

            EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm(PARAMS), rocsparse_status_invalid_pointer);

            alg         = rocsparse_sddmm_alg_default;
            temp_buffer = (void*)0x4;
        }

#undef PARAMS
#undef PARAMS_BUFFER_SIZE
    }

    static void testing_sddmm(const Arguments& arg)
    {
        J                    M       = arg.M;
        J                    N       = arg.N;
        J                    K       = arg.K;
        rocsparse_operation  trans_A = arg.transA;
        rocsparse_operation  trans_B = arg.transB;
        rocsparse_index_base base    = arg.baseA;
        rocsparse_sddmm_alg  alg     = arg.sddmm_alg;
        rocsparse_datatype   ttype   = get_datatype<T>();
        rocsparse_order      order_A = arg.order;
        rocsparse_order      order_B = arg.orderB;
        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

#ifndef ROCSPARSE_WITH_ROCBLAS
        if(arg.sddmm_alg == rocsparse_sddmm_alg_dense)
        {
            std::cerr
                << "No BLAS implementation is available, skipping rocsparse_sddmm_alg_dense test."
                << std::endl;
            return;
        }
#endif

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

#define PARAMS_BUFFER_SIZE(alpha_, A_, B_, beta_, C_)                                          \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        &buffer_size
#define PARAMS(alpha_, A_, B_, beta_, C_)                                                      \
    handle, trans_A, trans_B, alpha_, (const rocsparse_dnmat_descr&)A_,                        \
        (const rocsparse_dnmat_descr&)B_, beta_, (const rocsparse_spmat_descr&)C_, ttype, alg, \
        dbuffer

        // Argument sanity check before allocating invalid memory

        // Allocate memory on device
        // Pointer mode

        // Check structures
        // Check Sddmm when structures can be created
        if(M <= 0 || N <= 0)
        {

            if(M == 0 || N == 0)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                device_sparse_matrix<C> dC;
                device_dense_matrix<A>  dA;
                device_dense_matrix<B>  dB;

                rocsparse_local_spmat mat_C(dC);
                rocsparse_local_dnmat mat_A(dA), mat_B(dB);

                size_t buffer_size;
                void*  dbuffer = nullptr;
                EXPECT_ROCSPARSE_STATUS(rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(
                                            h_alpha, mat_A, mat_B, h_beta, mat_C)),
                                        rocsparse_status_success);
                CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, 10));
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_sddmm_preprocess(PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C)),
                    rocsparse_status_success);
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_sddmm(PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C)),
                    rocsparse_status_success);
                CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
                return;
            }
            return;
        }

        // Wavefront size
        int dev;
        CHECK_HIP_ERROR(hipGetDevice(&dev));

        hipDeviceProp_t prop;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<C> hC;
        {
            bool                              type      = (prop.warpSize == 32) ? true : false;
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<C, I, J> matrix_factory(
                arg, arg.timing ? false : type, full_rank);
            traits::sparse_initialization(matrix_factory, hC, M, N, base);
        }

        device_sparse_matrix<C> dC(hC);

        const J              hA_m = (trans_A == rocsparse_operation_none) ? M : K;
        const J              hA_n = (trans_A == rocsparse_operation_none) ? K : M;
        host_dense_matrix<A> hA(hA_m, hA_n, order_A);
        rocsparse_matrix_utils::init_exact(hA);

        const J              hB_m = (trans_B == rocsparse_operation_none) ? K : N;
        const J              hB_n = (trans_B == rocsparse_operation_none) ? N : K;
        host_dense_matrix<B> hB(hB_m, hB_n, order_B);
        rocsparse_matrix_utils::init_exact(hB);

        device_dense_matrix<A> dA(hA);
        device_dense_matrix<B> dB(hB);

        rocsparse_local_spmat mat_C(dC);
        rocsparse_local_dnmat mat_A(dA), mat_B(dB);

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_sddmm_buffer_size(PARAMS_BUFFER_SIZE(h_alpha, mat_A, mat_B, h_beta, mat_C)));

        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, std::max(buffer_size, sizeof(I))));
        if(arg.unit_check)
        {
            // Pointer mode host
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_sddmm_preprocess(PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C)));

            CHECK_ROCSPARSE_ERROR(
                testing::rocsparse_sddmm(PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C)));
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("C pointer mode host", dC);
            }

            {
                host_vector<C> hC_val_copy(hC.val);
                //
                // HOST CALCULATION
                //
                traits::host_calculation(trans_A, trans_B, h_alpha, hA, hB, h_beta, hC);

                //
                // CHECK MATRICES
                //
                hC.near_check(dC);

                //
                // COPY BACK ORIGINAL VALUES.
                //
                dC.val.transfer_from(hC_val_copy);
            }

            // Pointer mode device
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sddmm_preprocess(PARAMS(d_alpha, mat_A, mat_B, d_beta, mat_C)));
                CHECK_ROCSPARSE_ERROR(
                    testing::rocsparse_sddmm(PARAMS(d_alpha, mat_A, mat_B, d_beta, mat_C)));
            }
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("C pointer mode device", dC);
            }

            hC.near_check(dC);
        }

        if(arg.timing)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_sddmm_preprocess(PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C)));

            const double gpu_time_used = rocsparse_clients::run_benchmark(
                arg, rocsparse_sddmm, PARAMS(h_alpha, mat_A, mat_B, h_beta, mat_C));

            double gflop_count = rocsparse_gflop_count<FORMAT>::sddmm(
                dC.m, dC.n, dC.nnz, K, *h_beta != static_cast<T>(0));
            double gbyte_count = rocsparse_gbyte_count<FORMAT>::template sddmm<T>(
                dC.m, dC.n, dC.nnz, K, *h_beta != static_cast<T>(0));

            double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
            double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

            display_timing_info(display_key_t::format,
                                rocsparse_format2string(FORMAT),
                                display_key_t::trans_A,
                                rocsparse_operation2string(trans_A),
                                display_key_t::trans_B,
                                rocsparse_operation2string(trans_B),
                                display_key_t::M,
                                M,
                                display_key_t::N,
                                N,
                                display_key_t::K,
                                K,
                                display_key_t::nnz,
                                dC.nnz,
                                display_key_t::alpha,
                                *h_alpha,
                                display_key_t::beta,
                                *h_beta,
                                display_key_t::algorithm,
                                rocsparse_sddmmalg2string(alg),
                                display_key_t::gflops,
                                gpu_gflops,
                                display_key_t::bandwidth,
                                gpu_gbyte,
                                display_key_t::time_ms,
                                get_gpu_time_msec(gpu_time_used));
        }

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        return;
    }
};
