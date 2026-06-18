/*! \file */
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
#include "testing_spildlt0.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"

// Host reference for ILDLt factorization (templated on index types I, J)
template <typename T, typename I, typename J>
static void host_csrildlt0_ref(J M,
                               const I* __restrict__ csr_row_ptr,
                               const J* __restrict__ csr_col_ind,
                               T* __restrict__ csr_val,
                               floating_data_t<T>* __restrict__ diag,
                               rocsparse_index_base base,
                               int64_t* __restrict__ struct_pivot,
                               int64_t* __restrict__ numeric_pivot,
                               int64_t* __restrict__ singular_pivot,
                               double tol)
{
    *struct_pivot   = -1;
    *numeric_pivot  = -1;
    *singular_pivot = -1;

    std::vector<I> diag_offset(M, -1);
    std::vector<I> nnz_entries(M, -1);

    for(J ai = 0; ai < M; ++ai)
    {
        const I row_begin = csr_row_ptr[ai] - base;
        const I row_end   = csr_row_ptr[ai + 1] - base;
        J       j;

        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        floating_data_t<T> diag_sum = static_cast<floating_data_t<T>>(0);
        bool               has_diag = false;

        for(j = row_begin; j < row_end; ++j)
        {
            const J col_j = csr_col_ind[j] - base;

            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            if(col_j > ai)
            {
                break;
            }

            const floating_data_t<T> d_j = diag[col_j];

            if(std::abs(d_j) <= tol)
            {
                *singular_pivot = (*singular_pivot == -1)
                                      ? (col_j + base)
                                      : std::min(*singular_pivot, int64_t(col_j + base));
            }

            if(d_j == static_cast<floating_data_t<T>>(0))
            {
                *numeric_pivot = (*numeric_pivot == -1)
                                     ? (col_j + base)
                                     : std::min(*numeric_pivot, int64_t(col_j + base));
                continue;
            }

            const floating_data_t<T> inv_d_j = static_cast<floating_data_t<T>>(1) / d_j;

            const I row_begin_j = csr_row_ptr[col_j] - base;
            const I row_diag_j  = diag_offset[col_j];

            T local_sum = static_cast<T>(0);

            for(I k = row_begin_j; k < row_diag_j; ++k)
            {
                const J col_k = csr_col_ind[k] - base;
                if(nnz_entries[col_k] != -1)
                {
                    const I idx = nnz_entries[col_k];
                    // L_{ai,k} * D_k * conj(L_{col_j,k})
                    local_sum = std::fma(csr_val[idx],
                                         static_cast<T>(diag[col_k]) * rocsparse_conj(csr_val[k]),
                                         local_sum);
                }
            }

            const T val_j = (csr_val[j] - local_sum) * static_cast<T>(inv_d_j);
            csr_val[j]    = val_j;

            // Accumulate |L_{ai,col_j}|^2 * D_{col_j} (real)
            const floating_data_t<T> re_l = std::real(val_j);
            const floating_data_t<T> im_l = std::imag(val_j);
            diag_sum = std::fma(std::fma(re_l, re_l, im_l * im_l), d_j, diag_sum);
        }

        if(!has_diag)
        {
            *struct_pivot
                = (*struct_pivot == -1) ? (ai + base) : std::min(*struct_pivot, int64_t(ai + base));
            *numeric_pivot = (*numeric_pivot == -1) ? (ai + base)
                                                    : std::min(*numeric_pivot, int64_t(ai + base));
        }
        else
        {
            diag_offset[ai] = j;

            const floating_data_t<T> d_i = std::real(csr_val[j]) - diag_sum;

            if(std::abs(d_i) <= tol)
            {
                *singular_pivot = (*singular_pivot == -1)
                                      ? (ai + base)
                                      : std::min(*singular_pivot, int64_t(ai + base));
            }

            if(d_i == static_cast<floating_data_t<T>>(0))
            {
                *numeric_pivot = (*numeric_pivot == -1)
                                     ? (ai + base)
                                     : std::min(*numeric_pivot, int64_t(ai + base));
            }

            diag[ai]   = d_i;
            csr_val[j] = static_cast<T>(0); // unit diagonal stored as 0
        }

        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = -1;
        }
    }

    if(*struct_pivot != -1)
    {
        *numeric_pivot
            = (*numeric_pivot == -1) ? (*struct_pivot) : std::min(*numeric_pivot, *struct_pivot);
    }

    if(*numeric_pivot != -1)
    {
        *singular_pivot = (*singular_pivot == -1) ? (*numeric_pivot)
                                                  : std::min(*singular_pivot, *numeric_pivot);
    }
}

namespace rocsparse_clients
{
    class spildlt0_descr
    {
    private:
        rocsparse_handle         m_handle{};
        rocsparse_spildlt0_descr m_descr{};

    public:
        host_dense_vector<int64_t> m_cpu_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_exact_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_near_singularity_position{};

        host_dense_vector<int64_t> m_host_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_host_numeric_singularity_position{};

        device_dense_vector<int64_t> m_device_symbolic_singularity_position{};
        device_dense_vector<int64_t> m_device_numeric_singularity_position{};

        spildlt0_descr(rocsparse_handle handle, int64_t batch_count)
            : m_handle(handle)
            , m_cpu_symbolic_singularity_position(batch_count)
            , m_cpu_numeric_exact_singularity_position(batch_count)
            , m_cpu_numeric_near_singularity_position(batch_count)
            , m_host_symbolic_singularity_position(batch_count)
            , m_host_numeric_singularity_position(batch_count)
            , m_device_symbolic_singularity_position(batch_count)
            , m_device_numeric_singularity_position(batch_count)
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error*       p_error = nullptr;
            const rocsparse_status status
                = rocsparse_spildlt0_descr_create(handle, &this->m_descr, p_error);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }

        ~spildlt0_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error* p_error = nullptr;
            std::ignore = rocsparse_spildlt0_descr_destroy(this->m_handle, this->m_descr, p_error);
        }

        inline operator rocsparse_spildlt0_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_spildlt0_descr&() const
        {
            return this->m_descr;
        }
    };
}

template <typename I, typename J, typename T>
void testing_spildlt0_set_input_bad_arg(const Arguments& arg)
{
    rocsparse_error*               p_error = nullptr;
    rocsparse_local_handle         local_handle;
    rocsparse_spildlt0_descr       spildlt0_descr     = (rocsparse_spildlt0_descr)0x4;
    rocsparse_handle               handle             = local_handle;
    static constexpr int           nex                = 2;
    static const int               ex[nex]            = {4, 5};
    const rocsparse_spildlt0_input input              = rocsparse_spildlt0_input_alg;
    void*                          data               = (void*)0x4;
    size_t                         data_size_in_bytes = sizeof(input);
    select_bad_arg_analysis(rocsparse_spildlt0_set_input,
                            nex,
                            ex,
                            handle, //0
                            spildlt0_descr, //1
                            input, //2
                            data, //3
                            data_size_in_bytes, //4
                            p_error); // 5

    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_descr_create(handle, &spildlt0_descr, p_error));
    for(auto e : {rocsparse_spildlt0_input_alg,
                  rocsparse_spildlt0_input_analysis_policy,
                  rocsparse_spildlt0_input_compute_datatype,
                  rocsparse_spildlt0_input_boost_enable,
                  rocsparse_spildlt0_input_boost_tolerance,
                  rocsparse_spildlt0_input_boost_value,
                  rocsparse_spildlt0_input_singularity_tolerance,
                  rocsparse_spildlt0_input_diag})
    {
        switch(e)
        {
        case rocsparse_spildlt0_input_alg:
        case rocsparse_spildlt0_input_analysis_policy:
        case rocsparse_spildlt0_input_compute_datatype:
        case rocsparse_spildlt0_input_boost_enable:
        case rocsparse_spildlt0_input_boost_tolerance:
        case rocsparse_spildlt0_input_boost_value:
        case rocsparse_spildlt0_input_singularity_tolerance:
        case rocsparse_spildlt0_input_diag:
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spildlt0_set_input(handle, spildlt0_descr, e, data, 0, p_error),
                rocsparse_status_invalid_value);
            break;
        }
        }
    }

    //
    // Singularity tolerance must be double
    //
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spildlt0_set_input(handle,
                                     spildlt0_descr,
                                     rocsparse_spildlt0_input_singularity_tolerance,
                                     data,
                                     sizeof(float),
                                     p_error),
        rocsparse_status_invalid_value);

    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_descr_destroy(handle, spildlt0_descr, p_error));
}

template <typename I, typename J, typename T>
void testing_spildlt0_buffer_size_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;

    rocsparse_handle         handle         = local_handle;
    rocsparse_spildlt0_descr spildlt0_descr = (rocsparse_spildlt0_descr)0x4;

    rocsparse_spmat_descr    A              = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr    P              = (rocsparse_spmat_descr)0x4;
    rocsparse_spildlt0_stage spildlt0_stage = rocsparse_spildlt0_stage_analysis;

    rocsparse_error* p_error = nullptr;
    {
        size_t*              p_buffer_size_in_bytes = (size_t*)0x4;
        static constexpr int nex                    = 1;
        static const int     ex[nex]                = {6};
        select_bad_arg_analysis(rocsparse_spildlt0_buffer_size,
                                nex,
                                ex,
                                handle,
                                spildlt0_descr,
                                A,
                                P,
                                spildlt0_stage,
                                p_buffer_size_in_bytes,
                                p_error);
    }
}

template <typename I, typename J, typename T>
void testing_spildlt0_get_output_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle             = local_handle;
    rocsparse_spildlt0_descr  spildlt0_descr     = (rocsparse_spildlt0_descr)0x4;
    rocsparse_error*          p_error            = nullptr;
    static constexpr int      nex                = 2;
    static const int          ex[nex]            = {4, 5};
    size_t                    data_size_in_bytes = sizeof(int64_t);
    void*                     data               = (void*)0x4;
    rocsparse_spildlt0_output output             = rocsparse_spildlt0_output_singularity_position;
    select_bad_arg_analysis(rocsparse_spildlt0_get_output,
                            nex,
                            ex,
                            handle,
                            spildlt0_descr,
                            output,
                            data,
                            data_size_in_bytes,
                            p_error);
}

template <typename I, typename J, typename T>
void testing_spildlt0_analysis_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle   local_handle;
    rocsparse_handle         handle         = local_handle;
    rocsparse_spildlt0_descr spildlt0_descr = (rocsparse_spildlt0_descr)0x4;
    rocsparse_spmat_descr    A              = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr    P              = (rocsparse_spmat_descr)0x4;
    rocsparse_spildlt0_stage spildlt0_stage = rocsparse_spildlt0_stage_analysis;
    void*                    buffer         = (void*)0x4;
    rocsparse_error*         p_error        = nullptr;
    {
        const size_t buffer_size_in_bytes = 1;
#define PARAMS handle, spildlt0_descr, A, P, spildlt0_stage, buffer_size_in_bytes, buffer, p_error
        static constexpr int nex     = 2;
        static const int     ex[nex] = {5, 7};
        select_bad_arg_analysis(rocsparse_spildlt0, nex, ex, PARAMS);
#undef PARAMS
    }
}

template <typename I, typename J, typename T>
void testing_spildlt0_bad_arg(const Arguments& arg)
{
    testing_spildlt0_buffer_size_bad_arg<I, J, T>(arg);
    testing_spildlt0_analysis_bad_arg<I, J, T>(arg);
    testing_spildlt0_set_input_bad_arg<I, J, T>(arg);
    testing_spildlt0_get_output_bad_arg<I, J, T>(arg);

    rocsparse_error*       p_error = nullptr;
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    //
    // Now get a concrete example and continue.
    //
    rocsparse_clients::csr_tridiag_matrix_t<T, I, J> A(4);
    rocsparse_spmat_descr                            P = A;
    host_scalar<double>                              hsingularity_tolerance(1);
    hsingularity_tolerance[0] = 1.0e-5;
    device_scalar<double> dsingularity_tolerance(hsingularity_tolerance);

    // Allocate a diag vector on device (size = 4, same as tridiag matrix dimension)
    device_dense_vector<floating_data_t<T>> d_diag(4);

    for(auto analysis_policy : {rocsparse_analysis_policy_force, rocsparse_analysis_policy_reuse})
    {
        rocsparse_clients::spildlt0_descr spildlt0_descr(handle, 1);

        const rocsparse_spildlt0_alg alg              = rocsparse_spildlt0_alg_default;
        const rocsparse_datatype     compute_datatype = get_datatype<T>();
        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(
            handle, spildlt0_descr, rocsparse_spildlt0_input_alg, &alg, sizeof(alg), p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(handle,
                                                           spildlt0_descr,
                                                           rocsparse_spildlt0_input_analysis_policy,
                                                           &analysis_policy,
                                                           sizeof(analysis_policy),
                                                           p_error));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spildlt0_set_input(handle,
                                         spildlt0_descr,
                                         rocsparse_spildlt0_input_compute_datatype,
                                         &compute_datatype,
                                         sizeof(compute_datatype),
                                         p_error));

        // Set the diag device pointer
        void* d_diag_ptr = (void*)d_diag;
        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(handle,
                                                           spildlt0_descr,
                                                           rocsparse_spildlt0_input_diag,
                                                           &d_diag_ptr,
                                                           sizeof(d_diag_ptr),
                                                           p_error));

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_buffer_size(handle,
                                                             spildlt0_descr,
                                                             A,
                                                             P,
                                                             rocsparse_spildlt0_stage_analysis,
                                                             &buffer_size,
                                                             p_error));
        hipStream_t stream;
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));
        {
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            //
            // Call compute before analysis.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spildlt0(handle,
                                                       spildlt0_descr,
                                                       A,
                                                       P,
                                                       rocsparse_spildlt0_stage_compute,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

            //
            // Call analysis.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spildlt0_stage_analysis,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            for(auto input : {rocsparse_spildlt0_input_alg,
                              rocsparse_spildlt0_input_analysis_policy,
                              rocsparse_spildlt0_input_compute_datatype,
                              rocsparse_spildlt0_input_boost_enable,
                              rocsparse_spildlt0_input_boost_tolerance,
                              rocsparse_spildlt0_input_boost_value,
                              rocsparse_spildlt0_input_singularity_tolerance,
                              rocsparse_spildlt0_input_diag})
            {
                switch(input)
                {
                case rocsparse_spildlt0_input_alg:
                case rocsparse_spildlt0_input_compute_datatype:
                case rocsparse_spildlt0_input_analysis_policy:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spildlt0_set_input(
                            handle, spildlt0_descr, input, (void*)0x4, sizeof(int64_t), p_error));
                    break;
                }
                case rocsparse_spildlt0_input_singularity_tolerance:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spildlt0_set_input(
                            handle, spildlt0_descr, input, (void*)0x4, sizeof(float), p_error));

                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spildlt0_set_input(handle,
                                                     spildlt0_descr,
                                                     rocsparse_spildlt0_input_singularity_tolerance,
                                                     hsingularity_tolerance,
                                                     sizeof(double),
                                                     p_error));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spildlt0_set_input(handle,
                                                     spildlt0_descr,
                                                     rocsparse_spildlt0_input_singularity_tolerance,
                                                     dsingularity_tolerance,
                                                     sizeof(double),
                                                     p_error));
                    break;
                }
                case rocsparse_spildlt0_input_boost_enable:
                case rocsparse_spildlt0_input_boost_tolerance:
                case rocsparse_spildlt0_input_boost_value:
                case rocsparse_spildlt0_input_diag:
                {
                    break;
                }
                }
            }

            //
            // Call analysis twice.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spildlt0(handle,
                                                       spildlt0_descr,
                                                       A,
                                                       P,
                                                       rocsparse_spildlt0_stage_analysis,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));
        }

        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_buffer_size(handle,
                                                                 spildlt0_descr,
                                                                 A,
                                                                 P,
                                                                 rocsparse_spildlt0_stage_compute,
                                                                 &buffer_size,
                                                                 p_error));
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spildlt0_set_input(handle,
                                             spildlt0_descr,
                                             rocsparse_spildlt0_input_singularity_tolerance,
                                             hsingularity_tolerance,
                                             sizeof(double),
                                             p_error));

            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spildlt0_stage_compute,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spildlt0(handle,
                                                       spildlt0_descr,
                                                       A,
                                                       P,
                                                       rocsparse_spildlt0_stage_analysis,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spildlt0_set_input(handle,
                                             spildlt0_descr,
                                             rocsparse_spildlt0_input_singularity_tolerance,
                                             dsingularity_tolerance,
                                             sizeof(double),
                                             p_error));

            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spildlt0_stage_compute,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spildlt0(handle,
                                                       spildlt0_descr,
                                                       A,
                                                       P,
                                                       rocsparse_spildlt0_stage_analysis,
                                                       buffer_size,
                                                       buffer,
                                                       p_error));
            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spildlt0_stage_compute,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }
    }
}

template <typename I, typename J, typename T>
void testing_spildlt0(const Arguments& arg)
{
    rocsparse_error* p_error = nullptr;

    if(arg.M != arg.N)
    {
        return;
    }

    int64_t batch_count = arg.batch_count;
    if(batch_count == -1)
        batch_count = 1;

    static constexpr const bool             full_rank = true;
    rocsparse_clients::spmat_descr<T, I, J> A(arg, batch_count, full_rank);

    if(false == A.is_square())
    {
        return;
    }

    // Only CSR is supported for spildlt0
    if(A.get_format() != rocsparse_format_csr)
    {
        return;
    }

    rocsparse_local_handle handle(arg);
    hipStream_t            stream;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    const int64_t M = A.get_nrows();

    // Device diagonal output (size = M per batch)
    device_dense_vector<floating_data_t<T>> d_diag(M * batch_count);

    rocsparse_clients::spildlt0_descr spildlt0_descr(handle, batch_count);

    const rocsparse_spildlt0_alg    alg              = rocsparse_spildlt0_alg_default;
    const rocsparse_analysis_policy analysis_policy  = arg.apol;
    const rocsparse_datatype        compute_datatype = get_datatype<T>();

    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(
        handle, spildlt0_descr, rocsparse_spildlt0_input_alg, &alg, sizeof(alg), p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(handle,
                                                       spildlt0_descr,
                                                       rocsparse_spildlt0_input_analysis_policy,
                                                       &analysis_policy,
                                                       sizeof(analysis_policy),
                                                       p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(handle,
                                                       spildlt0_descr,
                                                       rocsparse_spildlt0_input_compute_datatype,
                                                       &compute_datatype,
                                                       sizeof(compute_datatype),
                                                       p_error));

    // Set the diag device pointer
    void* d_diag_ptr = (void*)d_diag;
    CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_set_input(handle,
                                                       spildlt0_descr,
                                                       rocsparse_spildlt0_input_diag,
                                                       &d_diag_ptr,
                                                       sizeof(d_diag_ptr),
                                                       p_error));

    //
    // Perform analysis.
    //
    auto& host_symbolic_pivot = spildlt0_descr.m_host_symbolic_singularity_position;

    {
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_buffer_size(handle,
                                                             spildlt0_descr,
                                                             A,
                                                             A,
                                                             rocsparse_spildlt0_stage_analysis,
                                                             &buffer_size_in_bytes,
                                                             p_error));
        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                 spildlt0_descr,
                                                 A,
                                                 A,
                                                 rocsparse_spildlt0_stage_analysis,
                                                 buffer_size_in_bytes,
                                                 buffer,
                                                 p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spildlt0_get_output(handle,
                                          spildlt0_descr,
                                          rocsparse_spildlt0_output_singularity_position,
                                          host_symbolic_pivot,
                                          sizeof(int64_t),
                                          p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    auto& device_symbolic_pivot = spildlt0_descr.m_device_symbolic_singularity_position;
    auto& host_numeric_pivot    = spildlt0_descr.m_host_numeric_singularity_position;
    auto& device_numeric_pivot  = spildlt0_descr.m_device_numeric_singularity_position;

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spildlt0_get_output(handle,
                                          spildlt0_descr,
                                          rocsparse_spildlt0_output_singularity_position,
                                          device_symbolic_pivot,
                                          sizeof(int64_t),
                                          p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        // Check pivot is the same with either host or device mode.
        host_symbolic_pivot.unit_check(device_symbolic_pivot);

        //
        // Perform compute.
        //
        {
            size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_buffer_size(handle,
                                                                 spildlt0_descr,
                                                                 A,
                                                                 A,
                                                                 rocsparse_spildlt0_stage_compute,
                                                                 &buffer_size_in_bytes,
                                                                 p_error));

            device_dense_vector<char> buffer(buffer_size_in_bytes);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     A,
                                                     rocsparse_spildlt0_stage_compute,
                                                     buffer_size_in_bytes,
                                                     buffer,
                                                     p_error));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spildlt0_get_output(handle,
                                              spildlt0_descr,
                                              rocsparse_spildlt0_output_singularity_position,
                                              host_numeric_pivot,
                                              sizeof(int64_t),
                                              p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spildlt0_get_output(handle,
                                              spildlt0_descr,
                                              rocsparse_spildlt0_output_singularity_position,
                                              device_numeric_pivot,
                                              sizeof(int64_t),
                                              p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));

            // Check numeric pivots match.
            host_numeric_pivot.unit_check(device_numeric_pivot);

            // Perform calculation on host and compare.
            {
                auto& host_csr = A.template as<rocsparse_format_csr>().host();

                // Copy current device values back to host for comparison
                host_vector<T>                  h_csr_val_gpu(host_csr.val.size());
                host_vector<floating_data_t<T>> h_diag_gpu(M * batch_count);
                CHECK_HIP_ERROR(hipMemcpy(h_csr_val_gpu,
                                          A.get_device_values(),
                                          sizeof(T) * host_csr.val.size(),
                                          hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(h_diag_gpu,
                                          d_diag,
                                          sizeof(floating_data_t<T>) * M * batch_count,
                                          hipMemcpyDeviceToHost));

                // CPU reference (run on a copy of the original host values)
                host_vector<T>                  h_csr_val_ref = host_csr.val;
                host_vector<floating_data_t<T>> h_diag_ref(M * batch_count,
                                                           static_cast<floating_data_t<T>>(0));

                const double singular_pivot_tolerance = 0.0;

                auto& cpu_symbolic_pivot = spildlt0_descr.m_cpu_symbolic_singularity_position;
                auto& cpu_numeric_near_pivot
                    = spildlt0_descr.m_cpu_numeric_near_singularity_position;
                auto& cpu_numeric_exact_pivot
                    = spildlt0_descr.m_cpu_numeric_exact_singularity_position;

                for(int64_t i = 0; i < batch_count; ++i)
                {
                    host_csrildlt0_ref<T, I, J>(static_cast<J>(host_csr.m),
                                                host_csr.ptr.data(),
                                                host_csr.ind.data(),
                                                h_csr_val_ref.data() + i * A.get_stride(),
                                                h_diag_ref.data() + i * M,
                                                host_csr.base,
                                                cpu_symbolic_pivot + i,
                                                cpu_numeric_exact_pivot + i,
                                                cpu_numeric_near_pivot + i,
                                                singular_pivot_tolerance);
                }

                for(int64_t j = 0; j < batch_count; ++j)
                {
                    if(cpu_numeric_near_pivot[j] == -1)
                    {
                        cpu_numeric_near_pivot[j] = cpu_numeric_exact_pivot[j];
                    }
                    else if(cpu_numeric_exact_pivot[j] != -1)
                    {
                        cpu_numeric_near_pivot[j]
                            = std::min(cpu_numeric_near_pivot[j], cpu_numeric_exact_pivot[j]);
                    }
                }

                cpu_symbolic_pivot.unit_check(host_symbolic_pivot);
                cpu_numeric_near_pivot.unit_check(host_numeric_pivot);

                // Check values only when no structural/numeric zero pivot was encountered (all batches).
                bool no_pivot = true;
                for(int64_t i = 0; i < batch_count; ++i)
                {
                    if(host_symbolic_pivot[i] != -1 || host_numeric_pivot[i] != -1)
                    {
                        no_pivot = false;
                        break;
                    }
                }
                if(no_pivot)
                {
                    if(ROCSPARSE_REPRODUCIBILITY)
                    {
                        rocsparse_reproducibility::save(
                            "L values", h_csr_val_gpu, "D values", h_diag_gpu);
                    }

                    // For complex types, accumulation error is larger.
                    constexpr bool is_complex = std::is_same<T, rocsparse_float_complex>::value
                                                || std::is_same<T, rocsparse_double_complex>::value;
                    const floating_data_t<T> tol_ildlt
                        = is_complex
                              ? static_cast<floating_data_t<T>>(100) * default_tolerance<T>::value
                              : default_tolerance<T>::value;

                    // Compare L values (off-diagonal CSR entries)
                    h_csr_val_ref.near_check(h_csr_val_gpu, tol_ildlt);
                    // Compare D values
                    h_diag_ref.near_check(h_diag_gpu, tol_ildlt);
                }
            }
        }
    }

    if(arg.timing)
    {
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0_buffer_size(handle,
                                                             spildlt0_descr,
                                                             A,
                                                             A,
                                                             rocsparse_spildlt0_stage_compute,
                                                             &buffer_size_in_bytes,
                                                             p_error));

        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        const int32_t n_cold_calls = 2;
        const int32_t n_sub_calls  = arg.iters_inner;
        const int32_t n_calls      = arg.iters;

        for(int32_t iter = 0; iter < n_cold_calls; ++iter)
        {
            A.reinit_values();
            CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                     spildlt0_descr,
                                                     A,
                                                     A,
                                                     rocsparse_spildlt0_stage_compute,
                                                     buffer_size_in_bytes,
                                                     buffer,
                                                     p_error));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        std::vector<double>      gpu_time(n_calls);
        rocsparse_clients::timer t(stream);
        for(int32_t iter = 0; iter < n_calls; ++iter)
        {
            gpu_time[iter] = 0;
            for(int32_t sub_iter = 0; sub_iter < n_sub_calls; ++sub_iter)
            {
                A.reinit_values();
                t.start();
                CHECK_ROCSPARSE_ERROR(rocsparse_spildlt0(handle,
                                                         spildlt0_descr,
                                                         A,
                                                         A,
                                                         rocsparse_spildlt0_stage_compute,
                                                         buffer_size_in_bytes,
                                                         buffer,
                                                         p_error));
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                const double t_microseconds = (t.stop() * 1000);
                gpu_time[iter] += t_microseconds;
            }
            gpu_time[iter] /= n_sub_calls;
        }
        std::sort(gpu_time.begin(), gpu_time.end());
        const int32_t mid = n_calls / 2;
        const double  gpu_time_used
            = (n_calls % 2 == 0) ? (gpu_time[mid] + gpu_time[mid - 1]) / 2 : gpu_time[mid];

        auto&        device      = A.template as<rocsparse_format_csr>().device();
        double       gbyte_count = csrildlt0_gbyte_count<T>(device.m, device.nnz);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::M,
                            device.m,
                            display_key_t::nnz_A,
                            device.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

void testing_spildlt0_extra(const Arguments& arg) {}

#define INSTANTIATE(I, J, T)                                               \
    template void testing_spildlt0_bad_arg<I, J, T>(const Arguments& arg); \
    template void testing_spildlt0<I, J, T>(const Arguments& arg)

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

#undef INSTANTIATE
