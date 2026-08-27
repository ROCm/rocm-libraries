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

#include "rocsparse_clients_dnmat_descr.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "testing.hpp"

extern "C" rocsparse_status rocsparse_dnmat_transpose(rocsparse_handle            handle,
                                                      rocsparse_const_dnvec_descr alpha,
                                                      rocsparse_const_dnmat_descr X,
                                                      rocsparse_dnmat_descr       Y,
                                                      rocsparse_error*            p_error);

extern "C" rocsparse_status rocsparse_dnmat_copy_data(rocsparse_handle            handle,
                                                      rocsparse_const_dnvec_descr alpha,
                                                      rocsparse_const_dnmat_descr X,
                                                      rocsparse_dnmat_descr       Y,
                                                      rocsparse_error*            p_error);

namespace rocsparse_clients
{
    template <typename T, typename I, typename J = I>
    void sptrsm_host(rocsparse_handle                         handle,
                     int64_t                                  batch_count,
                     const T*                                 halpha,
                     int64_t                                  alpha_batch_stride,
                     rocsparse_operation                      A_op,
                     rocsparse_clients::spmat_descr<T, I, J>& A,
                     rocsparse_operation                      X_op,
                     const rocsparse_clients::dnmat_descr<T>& X,
                     rocsparse_clients::dnmat_descr<T>&       Y,
                     const rocsparse_diag_type                A_diag,
                     const rocsparse_fill_mode                A_uplo,
                     int64_t*                                 symbolic,
                     int64_t*                                 exact)
    {

        const rocsparse_format format = A.get_format();
        auto&                  hY     = Y.host();

        rocsparse_error p_error[1] = {nullptr};

        //
        // Do host calculation.
        //

        if(X_op == rocsparse_operation_none)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_copy_data(handle, nullptr, X, Y, p_error));
        }
        else
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_transpose(handle, nullptr, X, Y, p_error));
        }

        //
        // copy device to host.
        //
        Y.to_host();

        if(X_op == rocsparse_operation_conjugate_transpose)
        {
            T* data = hY.data();
            for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                const int64_t M = (hY.order == rocsparse_order_column) ? hY.m : hY.n;
                const int64_t N = (hY.order == rocsparse_order_column) ? hY.n : hY.m;
                for(int64_t j = 0; j < N; ++j)
                    for(int64_t i = 0; i < M; ++i)
                        data[j * hY.ld + i] = rocsparse_conj(data[j * hY.ld + i]);
                data += Y.get_batch_stride();
            }
        }

        switch(format)
        {
        case rocsparse_format_coo:
        {
            auto& host = A.template as<rocsparse_format_coo>().host();
            for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                I ap = -1, sp = -1;
                host_coosm<I, T>(hY.m,
                                 hY.n,
                                 host.nnz,
                                 A_op,
                                 rocsparse_operation_none,
                                 *(halpha + batch_index * alpha_batch_stride),
                                 host.row_ind,
                                 host.col_ind,
                                 host.val.data() + batch_index * A.get_stride(),
                                 hY.data() + batch_index * Y.get_batch_stride(),
                                 hY.ld,
                                 hY.order,
                                 A_diag,
                                 A_uplo,
                                 host.base,
                                 &ap,
                                 &sp);
                symbolic[batch_index] = ap;
                exact[batch_index]    = sp;
            }
            break;
        }
        case rocsparse_format_csr:
        {
            auto& host = A.template as<rocsparse_format_csr>().host();
            for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                J ap = -1, sp = -1;
                host_csrsm<I, J, T>(hY.m,
                                    hY.n,
                                    host.nnz,
                                    A_op,
                                    rocsparse_operation_none,
                                    *(halpha + batch_index * alpha_batch_stride),
                                    host.ptr,
                                    host.ind,
                                    host.val.data() + batch_index * A.get_stride(),
                                    hY.data() + batch_index * Y.get_batch_stride(),
                                    hY.ld,
                                    hY.order,
                                    A_diag,
                                    A_uplo,
                                    host.base,
                                    &ap,
                                    &sp);
                symbolic[batch_index] = ap;
                exact[batch_index]    = sp;
            }

            break;
        }

        case rocsparse_format_csc:
        {
            auto& host = A.template as<rocsparse_format_csc>().host();
            for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                J ap = -1, sp = -1;
                host_cscsm<I, J, T>(hY.m,
                                    hY.n,
                                    host.nnz,
                                    A_op,
                                    rocsparse_operation_none,
                                    *(halpha + batch_index * alpha_batch_stride),
                                    host.ptr,
                                    host.ind,
                                    host.val.data() + batch_index * A.get_stride(),
                                    hY.data() + batch_index * Y.get_batch_stride(),
                                    hY.ld,
                                    hY.order,
                                    A_diag,
                                    A_uplo,
                                    host.base,
                                    &ap,
                                    &sp);
                symbolic[batch_index] = ap;
                exact[batch_index]    = sp;
            }

            break;
        }

        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_sell:
        case rocsparse_format_bell:
        case rocsparse_format_coo_aos:
        {
            break;
        }
        }
    }

}

namespace rocsparse_clients
{
    struct sptrsm_descr
    {
    private:
        rocsparse_sptrsm_descr m_descr{};

    public:
        struct config_t
        {
            rocsparse_operation       op_A;
            rocsparse_operation       op_X;
            rocsparse_sptrsm_alg      alg;
            rocsparse_datatype        scalar_datatype;
            rocsparse_datatype        compute_datatype;
            rocsparse_analysis_policy apol;
        };

        config_t config;

        rocsparse_status set(rocsparse_handle handle)
        {
            rocsparse_create_sptrsm_descr(&this->m_descr);
            rocsparse_error p_error[1] = {nullptr};

            rocsparse_status status{};

            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_operation_A,
                                                &config.op_A,
                                                sizeof(config.op_A),
                                                p_error);
            if(status)
                return status;
            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_operation_X,
                                                &config.op_X,
                                                sizeof(config.op_X),
                                                p_error);

            if(status)
                return status;
            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_alg,
                                                &config.alg,
                                                sizeof(config.alg),
                                                p_error);

            if(status)
                return status;
            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_scalar_datatype,
                                                &config.scalar_datatype,
                                                sizeof(config.scalar_datatype),
                                                p_error);

            if(status)
                return status;
            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_compute_datatype,
                                                &config.compute_datatype,
                                                sizeof(config.compute_datatype),
                                                p_error);

            if(status)
                return status;
            status = rocsparse_sptrsm_set_input(handle,
                                                this->m_descr,
                                                rocsparse_sptrsm_input_analysis_policy,
                                                &config.apol,
                                                sizeof(config.apol),
                                                p_error);
            if(status)
                return status;
            return status;
        }

        sptrsm_descr(rocsparse_handle                handle,
                     int64_t                         batch_count,
                     const rocsparse_operation       operation_A,
                     const rocsparse_operation       operation_X,
                     const rocsparse_sptrsm_alg      alg,
                     const rocsparse_datatype        scalar_datatype,
                     const rocsparse_datatype        compute_datatype,
                     const rocsparse_analysis_policy apol)
            : config({operation_A, operation_X, alg, scalar_datatype, compute_datatype, apol})
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            const rocsparse_status status = rocsparse_create_sptrsm_descr(&this->m_descr);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
            this->set(handle);
        }

        ~sptrsm_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            std::ignore = rocsparse_destroy_sptrsm_descr(this->m_descr);
        }

        inline operator rocsparse_sptrsm_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_sptrsm_descr&() const
        {
            return this->m_descr;
        }
    };

    void sptrsm_analysis(rocsparse_handle            handle,
                         rocsparse_sptrsm_descr      sptrsm_descr,
                         rocsparse_const_spmat_descr A,
                         rocsparse_const_dnmat_descr X,
                         rocsparse_dnmat_descr       Y,
                         rocsparse_error*            p_error)
    {
        hipStream_t stream;
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           X,
                                                           Y,
                                                           rocsparse_sptrsm_stage_analysis,
                                                           &buffer_size_in_bytes,
                                                           p_error));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               X,
                                               Y,
                                               rocsparse_sptrsm_stage_analysis,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));
        //
        // We don't synchronize, this stage is supposed to be synchroneous.
        // If not, then it is probably fine because hipFree-ing the buffer will force the synchronization.
        //
    }

    void sptrsm_compute(rocsparse_handle            handle,
                        rocsparse_sptrsm_descr      sptrsm_descr,
                        rocsparse_const_spmat_descr A,
                        rocsparse_const_dnmat_descr X,
                        rocsparse_dnmat_descr       Y,
                        rocsparse_pointer_mode      pointer_mode,
                        const void*                 alpha,
                        rocsparse_error*            p_error)
    {

        hipStream_t stream;
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, pointer_mode));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         alpha,
                                                         sizeof(alpha),
                                                         p_error));

        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           X,
                                                           Y,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm(handle,
                                               sptrsm_descr,
                                               A,
                                               X,
                                               Y,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

}

template <typename I, typename J, typename T>
void testing_sptrsm_bad_arg(const Arguments& arg)
{
}

template <typename I, typename J, typename T>
void testing_sptrsm(const Arguments& arg)
{
    static constexpr bool verbose = false;

    const rocsparse_direction X_batch_layout_direction = arg.direction;
    const rocsparse_direction Y_batch_layout_direction = arg.direction;

    rocsparse_local_handle handle{};
    rocsparse_error*       p_error = nullptr;

    const rocsparse_analysis_policy test_apol             = arg.apol;
    const rocsparse_datatype        test_compute_datatype = get_datatype<T>();

    const rocsparse_sptrsm_alg test_alg         = arg.sptrsm_alg;
    const J                    test_nrhs        = arg.K;
    const int64_t              test_batch_count = (arg.batch_count_C != -1) ? arg.batch_count_C : 1;

    if(verbose)
    {
        std::cout << std::endl;
        std::cout << "-- test --" << std::endl;
        std::cout << "   test_alg              : " << rocsparse_sptrsmalg2string(test_alg)
                  << std::endl;
        std::cout << "   test_format           : " << rocsparse_format2string(arg.formatA)
                  << std::endl;
        std::cout << "   test_nrhs             : " << test_nrhs << std::endl;
        std::cout << "   test_compute_datatype : "
                  << rocsparse_datatype2string(test_compute_datatype) << std::endl;
        std::cout << "   test_batch_count      : " << test_batch_count << std::endl;
        std::cout << std::endl;
    }

    const rocsparse_operation  A_op   = arg.transA;
    const rocsparse_diag_type  A_diag = arg.diag;
    const rocsparse_fill_mode  A_uplo = arg.uplo;
    const rocsparse_index_base A_base = arg.baseA;
    const int64_t A_batch_count = (arg.batch_count_A != -1) ? arg.batch_count_A : test_batch_count;

    const rocsparse_matrix_type A_matrix_type = arg.matrix_type;
    const bool                  A_full_rank   = true;

    //
    // Define the spmat.
    //
    rocsparse_clients::spmat_descr<T, I, J> A(arg, A_batch_count, A_full_rank);
    if(false == A.is_square())
    {
        return;
    }

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &A_uplo, sizeof(A_uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &A_diag, sizeof(A_diag)));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
        A, rocsparse_spmat_matrix_type, &A_matrix_type, sizeof(A_matrix_type)));
    const int64_t M = A.get_nrows();

    if(verbose)
    {
        std::cout << "-- A --" << std::endl;
        std::cout << "   A_m               : " << M << std::endl;
        std::cout << "   A_op              : " << rocsparse_operation2string(A_op) << std::endl;
        std::cout << "   A_matrix_type     : " << rocsparse_matrixtype2string(A_matrix_type)
                  << std::endl;
        std::cout << "   A_diag            : " << rocsparse_diagtype2string(A_diag) << std::endl;
        std::cout << "   A_uplo            : " << rocsparse_fillmode2string(A_uplo) << std::endl;
        std::cout << "   A_base            : " << rocsparse_indexbase2string(A_base) << std::endl;
        std::cout << "   A_batch_count     : " << A_batch_count << std::endl;
        std::cout << "   A_full_rank       : " << ((A_full_rank) ? "yes" : "no") << std::endl;
        std::cout << std::endl;
    }

    //
    // Define X.
    //
    const rocsparse_operation X_op    = arg.transB;
    const rocsparse_order     X_order = arg.orderB;

    const J X_m = (X_op == rocsparse_operation_none) ? M : test_nrhs;

    const J X_n = (X_op == rocsparse_operation_none) ? test_nrhs : M;

    const int64_t X_batch_count = (arg.batch_count_B > 0) ? arg.batch_count_B : test_batch_count;

    const bool X_init            = true;
    const bool X_non_zero_stride = true;

    rocsparse_clients::dnmat_descr<T> X(
        X_order, X_m, X_n, X_batch_layout_direction, X_batch_count, X_non_zero_stride, X_init);
    if(verbose)
    {
        std::cout << "-- X  --" << std::endl;
        std::cout << "   X_op              : " << rocsparse_operation2string(X_op) << std::endl;
        std::cout << "   X_order           : " << rocsparse_order2string(X_order) << std::endl;
        std::cout << "   X_m               : " << X_m << std::endl;
        std::cout << "   X_n               : " << X_n << std::endl;
        std::cout << "   X_ld              : " << X.get_ld() << std::endl;
        std::cout << "   X_batch_direction : "
                  << rocsparse_direction2string(X_batch_layout_direction) << std::endl;
        std::cout << "   X_batch_count     : " << X.get_batch_count() << std::endl;
        std::cout << "   X_batch_stride    : " << X.get_batch_stride() << std::endl;
        std::cout << std::endl;
    }

    const int64_t         Y_batch_count     = test_batch_count;
    const rocsparse_order Y_order           = arg.orderC;
    const J               Y_m               = M;
    const J               Y_n               = test_nrhs;
    const bool            Y_init            = false;
    const bool            Y_non_zero_stride = true;

    rocsparse_clients::dnmat_descr<T> Y(
        Y_order, Y_m, Y_n, Y_batch_layout_direction, Y_batch_count, Y_non_zero_stride, Y_init);

    if(verbose)
    {
        std::cout << "-- Y  --" << std::endl;
        std::cout << "   Y_order           : " << rocsparse_order2string(Y_order) << std::endl;
        std::cout << "   Y_m               : " << Y_m << std::endl;
        std::cout << "   Y_n               : " << Y_n << std::endl;
        std::cout << "   Y_ld              : " << Y.get_ld() << std::endl;
        std::cout << "   Y_batch_direction : "
                  << rocsparse_direction2string(Y_batch_layout_direction) << std::endl;
        std::cout << "   Y_batch_count     : " << Y.get_batch_count() << std::endl;
        std::cout << "   Y_batch_stride    : " << Y.get_batch_stride() << std::endl;
        std::cout << std::endl;
    }

    const rocsparse_datatype alpha_datatype = get_datatype<T>();

    host_scalar<T>   halpha(arg.get_alpha<T>());
    device_scalar<T> dalpha(halpha);

    const int64_t alpha_size         = 1;
    const int64_t alpha_batch_count  = 1;
    const int64_t alpha_batch_stride = 0;

    if(verbose)
    {
        std::cout << "-- alpha  --" << std::endl;
        std::cout << "   alpha_datatype     : " << rocsparse_datatype2string(alpha_datatype)
                  << std::endl;
        std::cout << "   alpha_size         : " << alpha_size << std::endl;
        std::cout << "   alpha_batch_count  : " << alpha_batch_count << std::endl;
        std::cout << "   alpha_batch_stride : " << alpha_batch_stride << std::endl;
        std::cout << std::endl;
    }

    const auto batch_count = Y_batch_count;

    //
    // Create the descriptor.
    //
    rocsparse_clients::sptrsm_descr sptrsm_descr(handle,
                                                 test_batch_count,
                                                 A_op,
                                                 X_op,
                                                 test_alg,
                                                 alpha_datatype,
                                                 test_compute_datatype,
                                                 test_apol);

    //
    // Perform the analysis.
    //
    rocsparse_clients::sptrsm_analysis(handle, sptrsm_descr, A, X, Y, p_error);

    host_dense_vector<int64_t> host_analysis_pivot(batch_count);
    host_dense_vector<int64_t> host_solve_pivot(batch_count);

    if(arg.unit_check)
    {

        rocsparse_clients::sptrsm_host<T, I, J>(handle,
                                                batch_count,
                                                halpha,
                                                alpha_batch_stride,
                                                A_op,
                                                A,
                                                X_op,
                                                X,
                                                Y,
                                                A_diag,
                                                A_uplo,
                                                host_analysis_pivot,
                                                host_solve_pivot);

        for(auto mode : {rocsparse_pointer_mode_host, rocsparse_pointer_mode_device})
        {
            //
            // Set the mode.
            //
            void* alpha = (mode == rocsparse_pointer_mode_host) ? halpha : dalpha;
            //	  std::cout <<"halpha"<< halpha[0] << std::endl;
            //	  std::cout <<"halpha"<< halpha[0] << std::endl;
            rocsparse_clients::sptrsm_compute(handle, sptrsm_descr, A, X, Y, mode, alpha, p_error);
            //	  Y.host().print();
            //	  Y.host().print();

            //
            // Get singularity_position.
            //
            {
                device_dense_vector<int64_t> gpu_singularity_position(test_batch_count);

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsm_get_output(handle,
                                                sptrsm_descr,
                                                rocsparse_sptrsm_output_singularity_position,
                                                gpu_singularity_position,
                                                sizeof(int64_t),
                                                p_error));

                hipStream_t stream{};
                CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));

                host_solve_pivot.unit_check(gpu_singularity_position);
            }

            //
            // Check numeric results.
            //

            Y.near_check_values(host_analysis_pivot, host_solve_pivot);

            break;
        }
    }

    if(arg.timing)
    {

        hipStream_t stream;
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_set_input(handle,
                                                         sptrsm_descr,
                                                         rocsparse_sptrsm_input_scalar_alpha,
                                                         halpha,
                                                         sizeof(halpha),
                                                         p_error));

        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_sptrsm_buffer_size(handle,
                                                           sptrsm_descr,
                                                           A,
                                                           X,
                                                           Y,
                                                           rocsparse_sptrsm_stage_compute,
                                                           &buffer_size_in_bytes,
                                                           p_error));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        device_dense_vector<char> buffer(buffer_size_in_bytes);
        const double              gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_sptrsm,
                                               handle,
                                               sptrsm_descr,
                                               A,
                                               X,
                                               Y,
                                               rocsparse_sptrsm_stage_compute,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error);
        int64_t      A_nnz       = 0;
        const double gflop_count = spsv_gflop_count(M, A_nnz, A_diag) * Y_n * batch_count;
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count = csrsv_gbyte_count<T>(M, A_nnz) * Y_n * batch_count;
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::batch_count,
                            batch_count,
                            display_key_t::M,
                            M,
                            display_key_t::nnz_A,
                            A_nnz,
                            display_key_t::nrhs,
                            Y_n,
                            display_key_t::alpha,
                            halpha,
                            display_key_t::algorithm,
                            rocsparse_sptrsmalg2string(test_alg),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                             \
    template void testing_sptrsm_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsm<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

void testing_sptrsm_extra(const Arguments& arg) {}
