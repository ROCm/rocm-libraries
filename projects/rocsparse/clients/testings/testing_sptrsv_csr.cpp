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
#include "rocsparse_clients_sptrsv.hpp"

struct rocsparse_local_sptrsv
{
    struct config_t
    {
        rocsparse_operation       operation;
        rocsparse_sptrsv_alg      alg;
        rocsparse_datatype        scalar_datatype;
        rocsparse_datatype        compute_datatype;
        rocsparse_analysis_policy apol;
    };

    config_t               config;
    rocsparse_sptrsv_descr sptrsv_descr{};
    ~rocsparse_local_sptrsv()
    {
        rocsparse_destroy_sptrsv_descr(this->sptrsv_descr);
    }

    operator rocsparse_sptrsv_descr&()
    {
        return this->sptrsv_descr;
    }
    operator const rocsparse_sptrsv_descr&() const
    {
        return this->sptrsv_descr;
    }

    rocsparse_local_sptrsv(rocsparse_handle                handle,
                           const rocsparse_operation       operation,
                           const rocsparse_sptrsv_alg      alg,
                           const rocsparse_datatype        scalar_datatype,
                           const rocsparse_datatype        compute_datatype,
                           const rocsparse_analysis_policy apol)
        : config({operation, alg, scalar_datatype, compute_datatype, apol})
    {
        this->set(handle);
    }

    void set(rocsparse_handle handle)
    {
        rocsparse_create_sptrsv_descr(&this->sptrsv_descr);
        rocsparse_error p_error[1] = {nullptr};
        rocsparse_sptrsv_set_input(handle,
                                   sptrsv_descr,
                                   rocsparse_sptrsv_input_operation,
                                   &config.operation,
                                   sizeof(config.operation),
                                   p_error);

        rocsparse_sptrsv_set_input(handle,
                                   sptrsv_descr,
                                   rocsparse_sptrsv_input_alg,
                                   &config.alg,
                                   sizeof(config.alg),
                                   p_error);

        rocsparse_sptrsv_set_input(handle,
                                   sptrsv_descr,
                                   rocsparse_sptrsv_input_scalar_datatype,
                                   &config.scalar_datatype,
                                   sizeof(config.scalar_datatype),
                                   p_error);

        rocsparse_sptrsv_set_input(handle,
                                   sptrsv_descr,
                                   rocsparse_sptrsv_input_compute_datatype,
                                   &config.compute_datatype,
                                   sizeof(config.compute_datatype),
                                   p_error);

        rocsparse_sptrsv_set_input(handle,
                                   sptrsv_descr,
                                   rocsparse_sptrsv_input_analysis_policy,
                                   &config.apol,
                                   sizeof(config.apol),
                                   p_error);
    }
};

//
// Implement missing instantiations regarding enums.
//
template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsv_input& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline rocsparse_status auto_testing_bad_arg_get_status(rocsparse_sptrsv_output& p)
{
    return rocsparse_status_invalid_value;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsv_input& p)
{
    p = (rocsparse_sptrsv_input)-1;
}

template <>
inline void auto_testing_bad_arg_set_invalid(rocsparse_sptrsv_output& p)
{
    p = (rocsparse_sptrsv_output)-1;
}

//
// Convenience.
//
static constexpr rocsparse_sptrsv_input rocsparse_sptrsv_input_all[6]
    = {rocsparse_sptrsv_input_alg,
       rocsparse_sptrsv_input_scalar_datatype,
       rocsparse_sptrsv_input_scalar_alpha,
       rocsparse_sptrsv_input_compute_datatype,
       rocsparse_sptrsv_input_operation,
       rocsparse_sptrsv_input_analysis_policy};

template <typename I, typename J, typename T>
void testing_sptrsv_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;

    rocsparse_handle       handle       = local_handle;
    rocsparse_sptrsv_descr sptrsv_descr = (rocsparse_sptrsv_descr)0x4;

    rocsparse_spmat_descr A = (rocsparse_spmat_descr)0x4;
    rocsparse_dnvec_descr x = (rocsparse_dnvec_descr)0x4;
    rocsparse_dnvec_descr y = (rocsparse_dnvec_descr)0x4;

    rocsparse_sptrsv_stage sptrsv_stage = rocsparse_sptrsv_stage_analysis;

    void*            buffer  = (void*)0x4;
    rocsparse_error* p_error = nullptr;
    {
        size_t* buffer_size_in_bytes = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE \
    handle, sptrsv_descr, A, x, y, sptrsv_stage, buffer_size_in_bytes, p_error

        static constexpr int nex     = 1;
        static const int     ex[nex] = {7};
        select_bad_arg_analysis(rocsparse_sptrsv_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);

#undef PARAMS_BUFFER_SIZE
    }

    {
        const size_t buffer_size_in_bytes = 1;

#define PARAMS handle, sptrsv_descr, A, x, y, sptrsv_stage, buffer_size_in_bytes, buffer, p_error

        static constexpr int nex     = 2;
        static const int     ex[nex] = {6, 8};
        select_bad_arg_analysis(rocsparse_sptrsv, nex, ex, PARAMS);

#undef PARAMS
    }

    {
        static constexpr int         nex                = 2;
        static const int             ex[nex]            = {4, 5};
        const rocsparse_sptrsv_input input              = rocsparse_sptrsv_input_alg;
        void*                        data               = (void*)0x4;
        size_t                       data_size_in_bytes = sizeof(input);
        select_bad_arg_analysis(rocsparse_sptrsv_set_input,
                                nex,
                                ex,
                                handle, //0
                                sptrsv_descr, //1
                                input, //2
                                data, //3
                                data_size_in_bytes, //4
                                p_error); // 5
        CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsv_descr(&sptrsv_descr));

        for(auto e : rocsparse_sptrsv_input_all)
        {
            switch(e)
            {
            case rocsparse_sptrsv_input_alg:
            case rocsparse_sptrsv_input_scalar_datatype:
            case rocsparse_sptrsv_input_scalar_alpha:
            case rocsparse_sptrsv_input_compute_datatype:
            case rocsparse_sptrsv_input_operation:
            case rocsparse_sptrsv_input_analysis_policy:
            {
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_sptrsv_set_input(handle, sptrsv_descr, e, data, 0, p_error),
                    rocsparse_status_invalid_size);
                break;
            }
            }
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_sptrsv_descr(sptrsv_descr));
    }

    //
    // Now get a concrete example and continue.
    //

    {
        rocsparse_clients::dense_vector_t<T>             x(4);
        rocsparse_clients::dense_vector_t<T>             y(4);
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
            rocsparse_local_sptrsv sptrsv_descr(handle,
                                                arg.transA,
                                                arg.sptrsv_alg,
                                                alpha_datatype,
                                                get_datatype<T>(),
                                                analysis_policy);

            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(handle,
                                                               sptrsv_descr,
                                                               A,
                                                               x,
                                                               y,
                                                               rocsparse_sptrsv_stage_analysis,
                                                               &buffer_size,
                                                               p_error));

            {
                device_dense_vector<char> buffer(buffer_size);
                CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

                //
                // Call compute before analysis.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsv(handle,
                                                         sptrsv_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsv_stage_compute,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));

                //
                // Call analysis.
                //
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                                       sptrsv_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsv_stage_analysis,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                for(auto input : {rocsparse_sptrsv_input_alg,
                                  rocsparse_sptrsv_input_operation,
                                  rocsparse_sptrsv_input_compute_datatype,
                                  rocsparse_sptrsv_input_scalar_datatype,
                                  rocsparse_sptrsv_input_scalar_alpha,
                                  rocsparse_sptrsv_input_analysis_policy})
                {
                    switch(input)
                    {
                    case rocsparse_sptrsv_input_alg:
                    case rocsparse_sptrsv_input_operation:
                    case rocsparse_sptrsv_input_compute_datatype:
                    case rocsparse_sptrsv_input_analysis_policy:
                    {

                        EXPECT_ROCSPARSE_STATUS(
                            rocsparse_status_invalid_value,
                            rocsparse_sptrsv_set_input(
                                handle, sptrsv_descr, input, (void*)0x4, sizeof(int64_t), p_error));
                        break;
                    }
                    case rocsparse_sptrsv_input_scalar_datatype:
                    case rocsparse_sptrsv_input_scalar_alpha:
                    {
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_sptrsv_set_input(handle,
                                                       sptrsv_descr,
                                                       rocsparse_sptrsv_input_scalar_alpha,
                                                       halpha,
                                                       sizeof(halpha.data()),
                                                       p_error));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                        CHECK_ROCSPARSE_ERROR(
                            rocsparse_sptrsv_set_input(handle,
                                                       sptrsv_descr,
                                                       rocsparse_sptrsv_input_scalar_alpha,
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
                                        rocsparse_sptrsv(handle,
                                                         sptrsv_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsv_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));
            }

            {
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(handle,
                                                                   sptrsv_descr,
                                                                   A,
                                                                   x,
                                                                   y,
                                                                   rocsparse_sptrsv_stage_compute,
                                                                   &buffer_size,
                                                                   p_error));
                device_dense_vector<char> buffer(buffer_size);
                CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               halpha,
                                               sizeof(halpha.data()),
                                               p_error));

                //
                // Call compute.
                //

                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                                       sptrsv_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsv_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                //
                // Call analysis after compute.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsv(handle,
                                                         sptrsv_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsv_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));

                CHECK_ROCSPARSE_ERROR(
                    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sptrsv_set_input(handle,
                                               sptrsv_descr,
                                               rocsparse_sptrsv_input_scalar_alpha,
                                               dalpha,
                                               sizeof(dalpha.data()),
                                               p_error));

                //
                // Call compute.
                //

                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                                       sptrsv_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsv_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

                //
                // Call analysis after compute.
                //
                EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                        rocsparse_sptrsv(handle,
                                                         sptrsv_descr,
                                                         A,
                                                         x,
                                                         y,
                                                         rocsparse_sptrsv_stage_analysis,
                                                         buffer_size,
                                                         buffer,
                                                         p_error));
                //
                // Call compute.
                //
                CHECK_ROCSPARSE_ERROR(rocsparse_sptrsv(handle,
                                                       sptrsv_descr,
                                                       A,
                                                       x,
                                                       y,
                                                       rocsparse_sptrsv_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));
            }
        }
    }

    {
        rocsparse_sptrsv_output output             = rocsparse_sptrsv_output_zero_pivot_position;
        size_t                  data_size_in_bytes = sizeof(int64_t);
        void*                   data               = (void*)0x4;
        static constexpr int    nex                = 2;
        static const int        ex[nex]            = {4, 5};
        select_bad_arg_analysis(rocsparse_sptrsv_get_output,
                                nex,
                                ex,
                                handle,
                                sptrsv_descr,
                                output,
                                data,
                                data_size_in_bytes,
                                p_error);
    }
}

template <typename I, typename J, typename T>
void testing_sptrsv_csr_bad_arg(const Arguments& arg)
{
    testing_sptrsv_bad_arg<I, J, T>(arg);
}

template <typename I, typename J, typename T>
void testing_sptrsv_csr(const Arguments& arg)
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
    host_csr_matrix<T, I, J> hA;
    {
        rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

        matrix_factory.init_csr(hA);
    }

    const J M = hA.m;
    if(M != hA.n)
    {
        return;
    }

    //
    // Create host data.
    //
    host_scalar<T>       halpha(arg.get_alpha<T>());
    host_dense_vector<T> hx(M);
    rocsparse_init<T>(hx, M, 1, 1);

    //
    // Create device data.
    //
    device_csr_matrix<T, I, J> dA(hA);
    device_dense_vector<T>     dx(hx);
    device_dense_vector<T>     dy(M);
    device_scalar<T>           dalpha(halpha);

    //
    // Create descriptors.
    //
    rocsparse_local_spmat A(dA);
    rocsparse_local_dnvec x(dx);
    rocsparse_local_dnvec y(dy);
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_set_attribute(
        A, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)));

    rocsparse_sptrsv_descr sptrsv_descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_create_sptrsv_descr(&sptrsv_descr));

    rocsparse_error p_error[1] = {nullptr};
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
        std::cout << "inconsistent zero pivot detected during analysis " << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    if((analysis_pivot_status != rocsparse_status_zero_pivot) && (analysis_zero_pivot != -1))
    {
        std::cout << "inconsistent zero pivot detected during analysis " << std::endl;
        CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    if(arg.unit_check)
    {

        // CPU csrsv
        host_dense_vector<T> hy(M);
        J                    analysis_pivot = -1;
        J                    solve_pivot    = -1;
        host_csrsv<I, J, T>(trans_A,
                            hA.m,
                            hA.nnz,
                            *halpha,
                            hA.ptr,
                            hA.ind,
                            hA.val,
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

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }
        CHECK_HIP_ERROR(hipDeviceSynchronize());
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

        const double gbyte_count = csrsv_gbyte_count<T>(hA.m, hA.nnz);
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_sptrsv_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsv_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_sptrsv_csr_extra(const Arguments& arg) {}
