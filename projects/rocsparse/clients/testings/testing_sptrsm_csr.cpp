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

#include "rocsparse_clients_objects.hpp"

struct rocsparse_local_sptrsm
{
    rocsparse_sptrsm_descr sptrsm_descr{};
    ~rocsparse_local_sptrsm()
    {
        rocsparse_destroy_sptrsm_descr(this->sptrsm_descr);
    }

    operator rocsparse_sptrsm_descr&()
    {
        return this->sptrsm_descr;
    }
    operator const rocsparse_sptrsm_descr&() const
    {
        return this->sptrsm_descr;
    }

    rocsparse_local_sptrsm(rocsparse_handle                handle,
                           const rocsparse_operation       operation_A,
                           const rocsparse_operation       operation_X,
                           const rocsparse_sptrsm_alg      alg,
                           const rocsparse_datatype        scalar_datatype,
                           const rocsparse_datatype        compute_datatype,
                           const rocsparse_analysis_policy apol)
    {
        rocsparse_create_sptrsm_descr(&this->sptrsm_descr);
        rocsparse_error p_error[1] = {nullptr};
        rocsparse_sptrsm_set_input(handle,
                                   sptrsm_descr,
                                   rocsparse_sptrsm_input_operation_A,
                                   &operation_A,
                                   sizeof(operation_A),
                                   p_error);

        rocsparse_sptrsm_set_input(handle,
                                   sptrsm_descr,
                                   rocsparse_sptrsm_input_operation_X,
                                   &operation_X,
                                   sizeof(operation_X),
                                   p_error);

        rocsparse_sptrsm_set_input(
            handle, sptrsm_descr, rocsparse_sptrsm_input_alg, &alg, sizeof(alg), p_error);

        rocsparse_sptrsm_set_input(handle,
                                   sptrsm_descr,
                                   rocsparse_sptrsm_input_scalar_datatype,
                                   &scalar_datatype,
                                   sizeof(scalar_datatype),
                                   p_error);

        rocsparse_sptrsm_set_input(handle,
                                   sptrsm_descr,
                                   rocsparse_sptrsm_input_compute_datatype,
                                   &compute_datatype,
                                   sizeof(compute_datatype),
                                   p_error);

        rocsparse_sptrsm_set_input(handle,
                                   sptrsm_descr,
                                   rocsparse_sptrsm_input_analysis_policy,
                                   &apol,
                                   sizeof(apol),
                                   p_error);
    }
};

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsm_input& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsm_output& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsm_input& p)
{
    p = (rocsparse_sptrsm_input)-1;
}

struct rocsparse_sptrsm_input_t
{
    static constexpr rocsparse_sptrsm_input all[5] = {
        rocsparse_sptrsm_input_alg,
        rocsparse_sptrsm_input_scalar_datatype,
        rocsparse_sptrsm_input_scalar_alpha,
        rocsparse_sptrsm_input_operation_A,
        rocsparse_sptrsm_input_operation_X,
    };
};

template <typename I, typename J, typename T>
void testing_sptrsm_csr_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;

    rocsparse_handle       handle       = local_handle;
    rocsparse_sptrsm_descr sptrsm_descr = (rocsparse_sptrsm_descr)0x4;

    rocsparse_spmat_descr A = (rocsparse_spmat_descr)0x4;
    rocsparse_dnmat_descr X = (rocsparse_dnmat_descr)0x4;
    rocsparse_dnmat_descr Y = (rocsparse_dnmat_descr)0x4;

    rocsparse_sptrsm_stage sptrsm_stage = rocsparse_sptrsm_stage_analysis;

    void*            buffer  = (void*)0x4;
    rocsparse_error* p_error = nullptr;

    {
        size_t* buffer_size_in_bytes = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE \
    handle, sptrsm_descr, A, X, Y, sptrsm_stage, buffer_size_in_bytes, p_error

        static constexpr int nex     = 1;
        static const int     ex[nex] = {7};
        select_bad_arg_analysis(rocsparse_sptrsm_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);

#undef PARAMS_BUFFER_SIZE
    }

    {
        const size_t buffer_size_in_bytes = 1;

#define PARAMS handle, sptrsm_descr, A, X, Y, sptrsm_stage, buffer_size_in_bytes, buffer, p_error

        static constexpr int nex     = 2;
        static const int     ex[nex] = {6, 8};
        select_bad_arg_analysis(rocsparse_sptrsm, nex, ex, PARAMS);

#undef PARAMS
    }

    {
        static constexpr int         nex                = 2;
        static const int             ex[nex]            = {4, 5};
        const rocsparse_sptrsm_input input              = rocsparse_sptrsm_input_alg;
        void*                        data               = (void*)0x4;
        size_t                       data_size_in_bytes = sizeof(input);
        select_bad_arg_analysis(rocsparse_sptrsm_set_input,
                                nex,
                                ex,
                                handle, //0
                                sptrsm_descr, //1
                                input, //2
                                data, //3
                                data_size_in_bytes, //4
                                p_error); // 5

        CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsm_descr(&sptrsm_descr));
        for(auto e : rocsparse_sptrsm_input_t::all)
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_sptrsm_set_input(handle, sptrsm_descr, e, data, 0, p_error),
                rocsparse_status_invalid_size);
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsm_descr(sptrsm_descr));
    }

    //
    // Testing with a tiny example, and tests the different wrong calls.
    //
    {
        rocsparse_clients::dense_matrix_t<T>             x(4, 5);
        rocsparse_clients::dense_matrix_t<T>             y(4, 5);
        rocsparse_clients::csr_tridiag_matrix_t<T, I, J> A(4);
        auto                                             uplo = rocsparse_fill_mode_lower;
        auto                                             diag = rocsparse_diag_type_unit;
        auto matrix_type                                      = rocsparse_matrix_type_general;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

        CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
            A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

        rocsparse_error p_error[1] = {nullptr};

        rocsparse_datatype   alpha_datatype = rocsparse_datatype_f32_r;
        host_scalar<float>   halpha(1);
        device_scalar<float> dalpha(halpha);
        for(auto analysis_policy :
            {rocsparse_analysis_policy_force, rocsparse_analysis_policy_reuse})
        {
            rocsparse_local_sptrsm sptrsm_descr(handle,
                                                arg.transA,
                                                arg.transB,
                                                arg.sptrsm_alg,
                                                alpha_datatype,
                                                get_datatype<T>(),
                                                analysis_policy);

            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                               sptrsm_descr,
                                                               A,
                                                               x,
                                                               y,
                                                               rocsparse_sptrsm_stage_analysis,
                                                               &buffer_size,
                                                               p_error));

            {
                device_dense_vector<char> buffer(buffer_size);
                CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

                //
                // Call compute before analysis.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsm(handle,
                                                         sptrsm_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsm_stage_compute,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));

                //
                // Call analysis.
                //
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                                       sptrsm_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsm_stage_analysis,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                //
                // Analysis is done, expect error if we modify the following values.
                //
                for(auto input : {rocsparse_sptrsm_input_alg,
                                  rocsparse_sptrsm_input_operation_A,
                                  rocsparse_sptrsm_input_operation_X,
                                  rocsparse_sptrsm_input_compute_datatype,
                                  rocsparse_sptrsm_input_scalar_datatype,
                                  rocsparse_sptrsm_input_scalar_alpha,
                                  rocsparse_sptrsm_input_analysis_policy})
                {
                    switch(input)
                    {
                    case rocsparse_sptrsm_input_alg:
                    case rocsparse_sptrsm_input_operation_A:
                    case rocsparse_sptrsm_input_operation_X:
                    case rocsparse_sptrsm_input_compute_datatype:
                    case rocsparse_sptrsm_input_analysis_policy:
                    {

                        EXPECT_ROCSPARSE_STATUS(
                            rocsparse_status_invalid_value,
                            rocsparse_sptrsm_set_input(
                                handle, sptrsm_descr, input, (void*)0x4, sizeof(int64_t), p_error));
                        break;
                    }
                    case rocsparse_sptrsm_input_scalar_datatype:
                    case rocsparse_sptrsm_input_scalar_alpha:
                    {
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_sptrsm_set_input(handle,
                                                       sptrsm_descr,
                                                       rocsparse_sptrsm_input_scalar_alpha,
                                                       halpha,
                                                       sizeof(halpha.data()),
                                                       p_error));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_sptrsm_set_input(handle,
                                                       sptrsm_descr,
                                                       rocsparse_sptrsm_input_scalar_alpha,
                                                       dalpha,
                                                       sizeof(dalpha.data()),
                                                       p_error));
                        break;
                    }
                    }
                }
                //
                // Call analysis twice.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsm(handle,
                                                         sptrsm_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsm_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));
            }

            {
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                                   sptrsm_descr,
                                                                   A,
                                                                   x,
                                                                   y,
                                                                   rocsparse_sptrsm_stage_compute,
                                                                   &buffer_size,
                                                                   p_error));
                device_dense_vector<char> buffer(buffer_size);
                CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_scalar_alpha,
                                               halpha,
                                               sizeof(halpha.data()),
                                               p_error));

                //
                // Call compute.
                //

                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                                       sptrsm_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsm_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                //
                // Call analysis after compute.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsm(handle,
                                                         sptrsm_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsm_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsm_set_input(handle,
                                               sptrsm_descr,
                                               rocsparse_sptrsm_input_scalar_alpha,
                                               dalpha,
                                               sizeof(dalpha.data()),
                                               p_error));

                //
                // Call compute.
                //

                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                                       sptrsm_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsm_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                //
                // Call analysis after compute.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsm(handle,
                                                         sptrsm_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsm_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));
                //
                // Call compute.
                //
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                                       sptrsm_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsm_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));
            }
        }
    }
}

template <typename I, typename J, typename T>
void testing_sptrsm_csr(const Arguments& arg)
{
    J M = arg.M;
    J N = arg.N;
    if(M != N)
    {
        return;
    }

    const J                    K               = arg.K;
    const rocsparse_operation  trans_A         = arg.transA;
    const rocsparse_operation  trans_B         = arg.transB;
    const rocsparse_index_base base            = arg.baseA;
    const rocsparse_spsm_alg   alg             = arg.spsm_alg;
    const rocsparse_diag_type  diag            = arg.diag;
    const rocsparse_fill_mode  uplo            = arg.uplo;
    const rocsparse_order      order_B         = arg.orderB;
    const rocsparse_order      order_C         = arg.orderC;
    const rocsparse_int        ld_multiplier_B = arg.ld_multiplier_B;
    const rocsparse_int        ld_multiplier_C = arg.ld_multiplier_C;

    // In the generic routines, C is always non-transposed (op(A) * C = op(B))
    const rocsparse_operation trans_C = rocsparse_operation_none;

    const host_scalar<T> halpha(arg.get_alpha<T>());

    // Index and data type
    const rocsparse_datatype ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle            handle(arg);
    rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

    // Sample matrix
    // Allocate host memory for matrix
    host_csr_matrix<T, I, J> hA;

    // Sample matrix
    matrix_factory.init_csr(hA, M, N, base);

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    const I nnz_A = hA.nnz;

    //
    // Scale values.
    //
    {
        const size_t       size = nnz_A;
        floating_data_t<T> mx   = floating_data_t<T>(0);
        for(size_t i = 0; i < size; ++i)
        {
            mx = std::max(mx, std::abs(hA.val[i]));
        }
        mx = (mx != floating_data_t<T>(0)) ? floating_data_t<T>(1.0) / mx : floating_data_t<T>(1.0);

        for(size_t i = 0; i < size; ++i)
        {
            hA.val[i] *= mx;
        }
    }

    const J B_m = (trans_B == rocsparse_operation_none) ? M : K;
    const J B_n = (trans_B == rocsparse_operation_none) ? K : M;
    const J C_m = M;
    const J C_n = K;

    const int64_t ldb = std::max(
        int64_t(1),
        ((order_B == rocsparse_order_column)
             ? ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * M)
                                                      : (int64_t(ld_multiplier_B) * K))
             : ((trans_B == rocsparse_operation_none) ? (int64_t(ld_multiplier_B) * K)
                                                      : (int64_t(ld_multiplier_B) * M))));

    const int64_t ldc
        = std::max(int64_t(1),
                   ((order_C == rocsparse_order_column) ? (int64_t(ld_multiplier_C) * M)
                                                        : (int64_t(ld_multiplier_C) * K)));

    const int64_t nrowB = (order_B == rocsparse_order_column) ? ldb : B_m;
    const int64_t ncolB = (order_B == rocsparse_order_column) ? B_n : ldb;
    const int64_t nrowC = (order_C == rocsparse_order_column) ? ldc : C_m;
    const int64_t ncolC = (order_C == rocsparse_order_column) ? C_n : ldc;

    // Allocate host memory for vectors
    host_dense_matrix<T> hB(nrowB, ncolB);

    {
        host_dense_matrix<T> htemp(B_m, B_n);
        rocsparse_matrix_utils::init(htemp);

        if(order_B == rocsparse_order_column)
        {
            for(J j = 0; j < B_n; j++)
            {
                for(J i = 0; i < B_m; i++)
                {
                    hB[i + ldb * j] = htemp[i + B_m * j];
                }
            }
        }
        else
        {
            for(J i = 0; i < B_m; i++)
            {
                for(J j = 0; j < B_n; j++)
                {
                    hB[ldb * i + j] = htemp[B_n * i + j];
                }
            }
        }
    }

    host_dense_matrix<T> hC(nrowC, ncolC);
    memset(hC, 0, sizeof(T) * nrowC * ncolC);

    // Copy B to C
    if(order_B == rocsparse_order_column)
    {
        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[i + ldb * j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[i + ldb * j];
                    }
                }
            }
        }
    }
    else
    {
        if(trans_B == rocsparse_operation_none)
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i * ldc + j] = hB[ldb * i + j];
                    }
                }
            }
            else
            {
                for(J j = 0; j < B_n; j++)
                {
                    for(J i = 0; i < B_m; i++)
                    {
                        hC[i + ldc * j] = hB[ldb * i + j];
                    }
                }
            }
        }
    }

    if(trans_B == rocsparse_operation_conjugate_transpose)
    {
        if(order_C == rocsparse_order_column)
        {
            for(J j = 0; j < C_n; j++)
            {
                for(J i = 0; i < C_m; i++)
                {
                    hC[i + ldc * j] = rocsparse_conj<T>(hC[i + ldc * j]);
                }
            }
        }
        else
        {
            for(J i = 0; i < C_m; i++)
            {
                for(J j = 0; j < C_n; j++)
                {
                    hC[ldc * i + j] = rocsparse_conj<T>(hC[ldc * i + j]);
                }
            }
        }
    }

    // Allocate device memory
    device_csr_matrix<T, I, J> dA(hA);
    device_dense_matrix<T>     dB(hB);
    device_dense_matrix<T>     dC(nrowC, ncolC);
    CHECK_HIP_ERROR(hipMemset(dC, 0, sizeof(T) * nrowC * ncolC));

    device_scalar<T> dalpha(halpha);

    // Create descriptors
    rocsparse_local_spmat A(dA);
    rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order_B);
    rocsparse_local_dnmat C(C_m, C_n, ldc, dC, ttype, order_C);
    rocsparse_error*      p_error = nullptr;

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    rocsparse_sptrsm_descr sptrsm_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsm_descr(&sptrsm_descr));

    //
    // Set algorithm.
    //
    {
        const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(
            handle, sptrsm_descr, rocsparse_sptrsm_input_alg, &alg, sizeof(alg), p_error));
    }

    //
    // Set operation A
    //
    {
        const rocsparse_operation operation_A = trans_A;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_A,
                                                         &operation_A,
                                                         sizeof(operation_A),
                                                         p_error));
    }

    //
    // Set operation B
    //
    {
        const rocsparse_operation operation_X = trans_B;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_operation_X,
                                                         &operation_X,
                                                         sizeof(operation_X),
                                                         p_error));
    }

    //
    // Set scalar datatype.
    //
    {
        const rocsparse_datatype scalar_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_datatype,
                                                         &scalar_datatype,
                                                         sizeof(scalar_datatype),
                                                         p_error));
    }

    {
        const rocsparse_datatype compute_datatype = ttype;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_compute_datatype,
                                                         &compute_datatype,
                                                         sizeof(compute_datatype),
                                                         p_error));
    }

    {
        const rocsparse_analysis_policy apol = arg.apol;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_analysis_policy,
                                                         &apol,
                                                         sizeof(apol),
                                                         p_error));
    }

    {
        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_analysis,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_analysis,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    if(arg.unit_check)
    {
        // CPU csrsm
        J analysis_pivot = -1;
        J solve_pivot    = -1;
        host_csrsm<I, J, T>(M,
                            K,
                            nnz_A,
                            trans_A,
                            trans_C,
                            halpha[0],
                            hA.ptr,
                            hA.ind,
                            hA.val,
                            hC,
                            ldc,
                            order_C,
                            diag,
                            uplo,
                            base,
                            &analysis_pivot,
                            &solve_pivot);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(const T*),
                                                         p_error));

        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_host", dC);
        }

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hC.near_check(dC);
        }

        //
        // Reset.
        //
        CHECK_HIP_ERROR(hipMemset(dC, 0, sizeof(T) * nrowC * ncolC));

        // Solve on device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         dalpha,
                                                         sizeof(const T*),
                                                         p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC_device", dC);
        }

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hC.near_check(dC);
        }
        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(const T*),
                                                         p_error));

        size_t buffer_size_in_bytes;
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));

        void* buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size_in_bytes));

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_sptrsm,
                                               handle,
                                               sptrsm_descr,
                                               A,
                                               B,
                                               C,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error);

        CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

        const double gflop_count = spsv_gflop_count(M, nnz_A, diag) * K;
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = csrsv_gbyte_count<T>(M, nnz_A) * K;
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nrhs,
                            K,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::algorithm,
                            rocsparse_spsmalg2string(alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsm_descr(sptrsm_descr));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_sptrsm_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsm_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

static void testing_sptrsm_csr_extra0(const Arguments& arg)
{

    static const bool         verbose = false;
    const rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation       trans_B = rocsparse_operation_transpose;

    //     1 0 0
    // A = 0 2 0
    //     0 0 3
    const rocsparse_int    m   = 3;
    const rocsparse_int    n   = 2;
    const rocsparse_int    nnz = 3;
    host_csr_matrix<float> hA(m, m, nnz, rocsparse_index_base_zero);
    hA.ptr[0] = 0;
    hA.ptr[1] = 1;
    hA.ptr[2] = 2;
    hA.ptr[3] = 3;
    hA.ind[0] = 0;
    hA.ind[1] = 1;
    hA.ind[2] = 2;
    hA.val[0] = 1;
    hA.val[1] = 2;
    hA.val[2] = 4;

    device_csr_matrix<float> dA(hA);
    host_dense_matrix<float> hB(m, n);
    host_dense_matrix<float> hB_T(n, m);
    host_dense_matrix<float> hC(m, n);

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            hC[hC.ld * j + i] = 777;
        }
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            hB[hB.ld * j + i] = static_cast<float>(j + 1);
        }
    }

    //     1 2
    // B = 1 2
    //     1 2
    if(verbose)
    {
        std::cout << "hB" << std::endl;
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
                fprintf(stdout, " %8.5e", hB[j * m + i]);
            fprintf(stdout, "\n");
        }
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            hB_T[n * j + i] = static_cast<float>(i + 1);
        }
    }

    if(verbose)
    {
        std::cout << "hB_T" << std::endl;
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
                fprintf(stdout, " %8.5e", hB_T[j * n + i]);
            fprintf(stdout, "\n");
        }
    }
    // Scalar alpha
    float alpha = 1.0f;
    // Offload data to device
    device_dense_matrix<float> dB(hB), dB_T(hB_T), dC(hC);
    rocsparse_handle           handle;
    rocsparse_local_spmat      matA(dA);
    rocsparse_local_dnmat      matB(dB);
    rocsparse_local_dnmat      matB_T(dB_T);
    rocsparse_local_dnmat      matC(dC);
    CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
    rocsparse_datatype compute_type = rocsparse_datatype_f32_r;
    {
        // Call spsv to get buffer size
        size_t buffer_size;
        trans_B = rocsparse_operation_transpose;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr));
        void* temp_buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc((void**)&temp_buffer, buffer_size));
        // Call spsv to perform analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_preprocess,
                                             &buffer_size,
                                             temp_buffer));
        // Call spsv to perform computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB_T,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_compute,
                                             &buffer_size,
                                             temp_buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));
        // Copy result back to host
        hC.transfer_from(dC);

        if(verbose)
        {

            std::cout << "hC with B transpose" << std::endl;
            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    fprintf(stdout, " %8.5e", hC[j * m + i]);
                }
                fprintf(stdout, "\n");
            }
        }
    }

    host_dense_matrix<float> hC2(hC);
    {
        // Call spsv to get buffer size
        size_t buffer_size;
        trans_B = rocsparse_operation_none;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr));
        void* temp_buffer;
        CHECK_HIP_ERROR(rocsparse_hipMalloc((void**)&temp_buffer, buffer_size));
        // Call spsv to perform analysis
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_preprocess,
                                             &buffer_size,
                                             temp_buffer));
        // Call spsv to perform computation
        CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                             trans_A,
                                             trans_B,
                                             &alpha,
                                             matA,
                                             matB,
                                             matC,
                                             compute_type,
                                             rocsparse_spsm_alg_default,
                                             rocsparse_spsm_stage_compute,
                                             &buffer_size,
                                             temp_buffer));
        CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));
        // Copy result back to host
        hC.transfer_from(dC);
    }
    if(verbose)
    {

        std::cout << "hC with B none" << std::endl;
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                fprintf(stdout, " %8.5e", hC[j * m + i]);
            }
            fprintf(stdout, "\n");
        }
    }
    hC.unit_check(hC2);
    // Clear rocSPARSE
    CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
}

static void
    spsm_csr_B_conjugate(const Arguments& arg, rocsparse_order order_B, rocsparse_order order_C)
{
    // This test verifies that the conjugate transpose is working correctly
    static const bool    verbose = false;
    rocsparse_operation  trans_A = rocsparse_operation_none;
    rocsparse_operation  trans_B = rocsparse_operation_conjugate_transpose;
    rocsparse_index_base base    = rocsparse_index_base_zero;

    rocsparse_spsm_alg  alg  = rocsparse_spsm_alg_default;
    rocsparse_fill_mode uplo = rocsparse_fill_mode_lower;
    rocsparse_diag_type diag = rocsparse_diag_type_non_unit;

    rocsparse_int m   = 3;
    rocsparse_int k   = 2;
    rocsparse_int ldb = (order_B == rocsparse_order_column) ? k : m;
    rocsparse_int ldc = (order_C == rocsparse_order_column) ? m : k;

    rocsparse_indextype itype        = get_indextype<rocsparse_int>();
    rocsparse_indextype jtype        = get_indextype<rocsparse_int>();
    rocsparse_datatype  compute_type = get_datatype<rocsparse_float_complex>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_float_complex halpha = rocsparse_float_complex(1, 0);

    //     1 0 0
    // A = 3 2 0
    //     4 2 -2
    rocsparse_int                        nnz_A        = 6;
    host_vector<rocsparse_int>           hcsr_row_ptr = {0, 1, 3, 6};
    host_vector<rocsparse_int>           hcsr_col_ind = {0, 0, 1, 0, 1, 2};
    host_vector<rocsparse_float_complex> hcsr_val     = {1, 3, 2, 4, 2, -2};

    // B = (1, 1)  (2, -1)  (0,  1)
    //     (1, 0)  (-1, 1)  (3, -2)
    const host_vector<rocsparse_float_complex> hB_row_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(2, -1),
                                                               rocsparse_float_complex(0, 1),
                                                               rocsparse_float_complex(1, 0),
                                                               rocsparse_float_complex(-1, 1),
                                                               rocsparse_float_complex(3, -2)};
    const host_vector<rocsparse_float_complex> hB_col_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(1, 0),
                                                               rocsparse_float_complex(2, -1),
                                                               rocsparse_float_complex(-1, 1),
                                                               rocsparse_float_complex(0, 1),
                                                               rocsparse_float_complex(3, -2)};

    // C = (1, 1)  (2, 2)
    //     (3, 3)  (4, 4)
    //     (5, 5)  (6, 6)
    const host_vector<rocsparse_float_complex> hC_row_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(2, 2),
                                                               rocsparse_float_complex(3, 3),
                                                               rocsparse_float_complex(4, 4),
                                                               rocsparse_float_complex(5, 5),
                                                               rocsparse_float_complex(6, 6)};
    const host_vector<rocsparse_float_complex> hC_col_order = {rocsparse_float_complex(1, 1),
                                                               rocsparse_float_complex(3, 3),
                                                               rocsparse_float_complex(5, 5),
                                                               rocsparse_float_complex(2, 2),
                                                               rocsparse_float_complex(4, 4),
                                                               rocsparse_float_complex(6, 6)};

    host_vector<rocsparse_float_complex> hB(k * m);
    host_vector<rocsparse_float_complex> hC(m * k);

    for(rocsparse_int i = 0; i < k * m; i++)
    {
        hB[i] = (order_B == rocsparse_order_column) ? hB_col_order[i] : hB_row_order[i];
    }

    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC[i] = (order_C == rocsparse_order_column) ? hC_col_order[i] : hC_row_order[i];
    }

    if(verbose)
    {
        std::cout << "A" << std::endl;
        for(rocsparse_int i = 0; i < m; i++)
        {
            rocsparse_int                        start = hcsr_row_ptr[i] - base;
            rocsparse_int                        end   = hcsr_row_ptr[i + 1] - base;
            std::vector<rocsparse_float_complex> temp(m, 0);
            for(rocsparse_int j = start; j < end; j++)
            {
                temp[hcsr_col_ind[j] - base] = hcsr_val[j];
            }

            for(rocsparse_int j = 0; j < m; j++)
            {
                std::cout << temp[j] << " ";
            }
            std::cout << "" << std::endl;
        }
        std::cout << "" << std::endl;

        std::cout << "B" << std::endl;
        if(order_B == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hB[i + ldb * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < k; i++)
            {
                for(rocsparse_int j = 0; j < m; j++)
                {
                    std::cout << hB[ldb * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;

        std::cout << "C" << std::endl;
        if(order_C == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[i + ldc * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[ldc * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;
    }

    device_vector<rocsparse_int>           dcsr_row_ptr(m + 1);
    device_vector<rocsparse_int>           dcsr_col_ind(nnz_A);
    device_vector<rocsparse_float_complex> dcsr_val(nnz_A);
    device_vector<rocsparse_float_complex> dB(k * m);
    device_vector<rocsparse_float_complex> dC(m * k);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_val, hcsr_val.data(), sizeof(rocsparse_float_complex) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dB, hB, sizeof(rocsparse_float_complex) * k * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dC, hC, sizeof(rocsparse_float_complex) * m * k, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(
        m, m, nnz_A, dcsr_row_ptr, dcsr_col_ind, dcsr_val, itype, jtype, base, compute_type);

    rocsparse_local_dnmat B(k, m, ldb, dB, compute_type, order_B);
    rocsparse_local_dnmat C(m, k, ldc, dC, compute_type, order_C);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    // Query SpSM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         C,
                                         compute_type,
                                         alg,
                                         rocsparse_spsm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* buffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    // Perform analysis on host
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         C,
                                         compute_type,
                                         alg,
                                         rocsparse_spsm_stage_preprocess,
                                         nullptr,
                                         buffer));

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_spsm(handle,
                                                  trans_A,
                                                  trans_B,
                                                  &halpha,
                                                  A,
                                                  B,
                                                  C,
                                                  compute_type,
                                                  alg,
                                                  rocsparse_spsm_stage_compute,
                                                  &buffer_size,
                                                  buffer));

    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy output to host
    CHECK_HIP_ERROR(
        hipMemcpy(hC, dC, sizeof(rocsparse_float_complex) * m * k, hipMemcpyDeviceToHost));

    if(verbose)
    {
        std::cout << "C" << std::endl;
        if(order_C == rocsparse_order_column)
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[i + ldc * j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        else
        {
            for(rocsparse_int i = 0; i < m; i++)
            {
                for(rocsparse_int j = 0; j < k; j++)
                {
                    std::cout << hC[ldc * i + j] << " ";
                }
                std::cout << "" << std::endl;
            }
        }
        std::cout << "" << std::endl;
    }

    // Manually computed correct solution
    // C = (1, -1)    (1, 0)
    //     (-0.5, 2)  (-2, -0.5)
    //     (1.5, 0.5) (-1.5, -1.5)
    const host_vector<rocsparse_float_complex> hC_solution_row_order
        = {rocsparse_float_complex(1, -1),
           rocsparse_float_complex(1, 0),
           rocsparse_float_complex(-0.5, 2),
           rocsparse_float_complex(-2, -0.5),
           rocsparse_float_complex(1.5, 0.5),
           rocsparse_float_complex(-1.5, -1.5)};
    const host_vector<rocsparse_float_complex> hC_solution_col_order
        = {rocsparse_float_complex(1, -1),
           rocsparse_float_complex(-0.5, 2),
           rocsparse_float_complex(1.5, 0.5),
           rocsparse_float_complex(1, 0),
           rocsparse_float_complex(-2, -0.5),
           rocsparse_float_complex(-1.5, -1.5)};

    host_dense_matrix<rocsparse_float_complex> hC_gpu(m, k);
    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC_gpu[i] = hC[i];
    }
    host_dense_matrix<rocsparse_float_complex> hC_cpu(m, k);

    for(rocsparse_int i = 0; i < m * k; i++)
    {
        hC_cpu[i] = (order_C == rocsparse_order_column) ? hC_solution_col_order[i]
                                                        : hC_solution_row_order[i];
    }

    hC_cpu.unit_check(hC_gpu);

    CHECK_HIP_ERROR(rocsparse_hipFree(buffer));
}

static void testing_sptrsm_csr_extra1(const Arguments& arg)
{
    spsm_csr_B_conjugate(arg, rocsparse_order_row, rocsparse_order_row);
    spsm_csr_B_conjugate(arg, rocsparse_order_row, rocsparse_order_column);
    spsm_csr_B_conjugate(arg, rocsparse_order_column, rocsparse_order_row);
    spsm_csr_B_conjugate(arg, rocsparse_order_column, rocsparse_order_column);
}

void testing_sptrsm_csr_extra(const Arguments& arg)
{
    testing_sptrsm_csr_extra0(arg);
    testing_sptrsm_csr_extra1(arg);
}
