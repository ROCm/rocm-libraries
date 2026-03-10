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

template <typename I, typename J, typename T>
static void host_csr_lsolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            int64_t              x_inc,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            int64_t*             struct_pivot,
                            int64_t*             numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process lower triangular part
    for(J row = 0; row < M; ++row)
    {
        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_begin; l < row_end; l += prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                if(local_val == static_cast<T>(0) && local_col == row
                   && diag_type == rocsparse_diag_type_non_unit)
                {
                    // Numerical zero pivot found, avoid division by 0
                    // and store index for later use.
                    *numeric_pivot = std::min(*numeric_pivot, int64_t(row + base));
                    local_val      = static_cast<T>(1);
                }

                // Ignore all entries that are above the diagonal
                if(local_col > row)
                {
                    break;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    // If diagonal type is non unit, do division by diagonal entry
                    // This is not required for unit diagonal for obvious reasons
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    break;
                }

                // Lower triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, int64_t(row + base));
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
static void host_csr_usolve(J                    M,
                            T                    alpha,
                            const I*             csr_row_ptr,
                            const J*             csr_col_ind,
                            const T*             csr_val,
                            const T*             x,
                            int64_t              x_inc,
                            T*                   y,
                            rocsparse_diag_type  diag_type,
                            rocsparse_index_base base,
                            int64_t*             struct_pivot,
                            int64_t*             numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    std::ignore = hipGetDevice(&dev);
    std::ignore = hipGetDeviceProperties(&prop, dev);

    std::vector<T> temp(prop.warpSize);

    // Process upper triangular part
    for(J row = M - 1; row >= 0; --row)
    {

        temp.assign(prop.warpSize, static_cast<T>(0));
        temp[0] = alpha * x[x_inc * row];

        I diag      = -1;
        I row_begin = csr_row_ptr[row] - base;
        I row_end   = csr_row_ptr[row + 1] - base;

        T diag_val = static_cast<T>(0);

        for(I l = row_end - 1; l >= row_begin; l -= prop.warpSize)
        {
            for(uint32_t k = 0; k < prop.warpSize; ++k)
            {
                I j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                J local_col = csr_col_ind[j] - base;
                T local_val = csr_val[j];

                // Ignore all entries that are below the diagonal
                if(local_col < row)
                {
                    continue;
                }

                // Diagonal entry
                if(local_col == row)
                {
                    if(diag_type == rocsparse_diag_type_non_unit)
                    {
                        // Check for numerical zero
                        if(local_val == static_cast<T>(0))
                        {
                            *numeric_pivot = std::min(*numeric_pivot, int64_t(row + base));
                            local_val      = static_cast<T>(1);
                        }

                        diag     = j;
                        diag_val = static_cast<T>(1) / local_val;
                    }

                    continue;
                }

                // Upper triangular part
                temp[k] = std::fma(-local_val, y[local_col], temp[k]);
            }
        }

        for(uint32_t j = 1; j < prop.warpSize; j <<= 1)
        {
            for(uint32_t k = 0; k < prop.warpSize - j; ++k)
            {
                temp[k] += temp[k + j];
            }
        }

        if(diag_type == rocsparse_diag_type_non_unit)
        {
            if(diag == -1)
            {
                *struct_pivot = std::min(*struct_pivot, int64_t(row + base));
            }

            y[row] = temp[0] * diag_val;
        }
        else
        {
            y[row] = temp[0];
        }
    }
}

template <typename I, typename J, typename T>
void cpu_csrsv(rocsparse_operation  trans,
               J                    M,
               I                    nnz,
               T                    alpha,
               const I*             csr_row_ptr,
               const J*             csr_col_ind,
               const T*             csr_val,
               const T*             x,
               int64_t              x_inc,
               T*                   y,
               rocsparse_diag_type  diag_type,
               rocsparse_fill_mode  fill_mode,
               rocsparse_index_base base,
               int64_t*             struct_pivot,
               int64_t*             numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot  = M + 1;
    *numeric_pivot = M + 1;

    if(trans == rocsparse_operation_none)
    {
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_lsolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_usolve(M,
                            alpha,
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }
    else if(trans == rocsparse_operation_transpose
            || trans == rocsparse_operation_conjugate_transpose)
    {
        // Transpose matrix
        std::vector<I> csrt_row_ptr(M + 1);
        std::vector<J> csrt_col_ind(nnz);
        std::vector<T> csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        rocsparse_action_numeric,
                        base);

        if(trans == rocsparse_operation_conjugate_transpose)
        {
            for(size_t i = 0; i < csrt_val.size(); i++)
            {
                csrt_val[i] = rocsparse_conj(csrt_val[i]);
            }
        }

        if(fill_mode == rocsparse_fill_mode_lower)
        {
            host_csr_usolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
        else
        {
            host_csr_lsolve(M,
                            alpha,
                            csrt_row_ptr.data(),
                            csrt_col_ind.data(),
                            csrt_val.data(),
                            x,
                            x_inc,
                            y,
                            diag_type,
                            base,
                            struct_pivot,
                            numeric_pivot);
        }
    }

    *numeric_pivot = std::min(*numeric_pivot, *struct_pivot);

    *struct_pivot  = (*struct_pivot == M + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == M + 1) ? -1 : *numeric_pivot;
}

template <typename I, typename T>
void cpu_coosv(rocsparse_operation  trans,
               I                    M,
               int64_t              nnz,
               T                    alpha,
               const I*             coo_row_ind,
               const I*             coo_col_ind,
               const T*             coo_val,
               const T*             x,
               T*                   y,
               rocsparse_diag_type  diag_type,
               rocsparse_fill_mode  fill_mode,
               rocsparse_index_base base,
               int64_t*             struct_pivot,
               int64_t*             numeric_pivot)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        std::vector<int32_t> csr_row_ptr(M + 1);

        host_coo_to_csr<int32_t, I>(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        cpu_csrsv<int32_t, I>(trans,
                              M,
                              nnz,
                              alpha,
                              csr_row_ptr.data(),
                              coo_col_ind,
                              coo_val,
                              x,
                              (int64_t)1,
                              y,
                              diag_type,
                              fill_mode,
                              base,
                              struct_pivot,
                              numeric_pivot);
    }
    else
    {
        std::vector<int64_t> csr_row_ptr(M + 1);

        host_coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr.data(), base);

        cpu_csrsv(trans,
                  M,
                  nnz,
                  alpha,
                  csr_row_ptr.data(),
                  coo_col_ind,
                  coo_val,
                  x,
                  (int64_t)1,
                  y,
                  diag_type,
                  fill_mode,
                  base,
                  struct_pivot,
                  numeric_pivot);
    }
}

namespace rocsparse_clients
{

    class sptrsv_descr
    {
    private:
        rocsparse_handle       m_handle{};
        rocsparse_sptrsv_descr m_descr{};

    public:
        struct config_t
        {
            rocsparse_operation       operation;
            rocsparse_sptrsv_alg      alg;
            rocsparse_datatype        scalar_datatype;
            rocsparse_datatype        compute_datatype;
            rocsparse_analysis_policy apol;
        };

        config_t config;

        void set(rocsparse_handle handle)
        {
            rocsparse_create_sptrsv_descr(&this->m_descr);
            rocsparse_error p_error[1] = {nullptr};
            rocsparse_sptrsv_set_input(handle,
                                       this->m_descr,
                                       rocsparse_sptrsv_input_operation,
                                       &config.operation,
                                       sizeof(config.operation),
                                       p_error);

            rocsparse_sptrsv_set_input(handle,
                                       this->m_descr,
                                       rocsparse_sptrsv_input_alg,
                                       &config.alg,
                                       sizeof(config.alg),
                                       p_error);

            rocsparse_sptrsv_set_input(handle,
                                       this->m_descr,
                                       rocsparse_sptrsv_input_scalar_datatype,
                                       &config.scalar_datatype,
                                       sizeof(config.scalar_datatype),
                                       p_error);

            rocsparse_sptrsv_set_input(handle,
                                       this->m_descr,
                                       rocsparse_sptrsv_input_compute_datatype,
                                       &config.compute_datatype,
                                       sizeof(config.compute_datatype),
                                       p_error);

            rocsparse_sptrsv_set_input(handle,
                                       this->m_descr,
                                       rocsparse_sptrsv_input_analysis_policy,
                                       &config.apol,
                                       sizeof(config.apol),
                                       p_error);
        }

        sptrsv_descr(rocsparse_handle                handle,
                     int64_t                         batch_count,
                     const rocsparse_operation       operation,
                     const rocsparse_sptrsv_alg      alg,
                     const rocsparse_datatype        scalar_datatype,
                     const rocsparse_datatype        compute_datatype,
                     const rocsparse_analysis_policy apol)
            : m_handle(handle)
            , config({operation, alg, scalar_datatype, compute_datatype, apol})
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error*       p_error = nullptr;
            const rocsparse_status status
                = rocsparse_sptrsv_descr_create(handle, &this->m_descr, p_error);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
            this->set(handle);
        }

        ~sptrsv_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error* p_error = nullptr;
            std::ignore = rocsparse_sptrsv_descr_destroy(this->m_handle, this->m_descr, p_error);
        }

        inline operator rocsparse_sptrsv_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_sptrsv_descr&() const
        {
            return this->m_descr;
        }
    };

    template <typename T, typename I, typename J = I>
    void sptrsv_host(int64_t                                  batch_count,
                     rocsparse_clients::sptrsv_descr&         sptrsv_descr,
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
        const rocsparse_format format = A.get_format();
        switch(format)
        {
        case rocsparse_format_coo:
        {
            auto& host = A.template as<rocsparse_format_coo>().host();

            for(int64_t i = 0; i < batch_count; ++i)
            {
                const T* p    = host.val.data() + i * A.get_stride();
                const T* p_hx = x.host().data() + i * x.get_stride();
                T*       p_hy = y.host().data() + i * y.get_stride();

                cpu_coosv<I, T>(operation,
                                host.m,
                                host.nnz,
                                *halpha,
                                host.row_ind,
                                host.col_ind,
                                p,
                                p_hx,
                                p_hy,
                                diag,
                                uplo,
                                host.base,
                                symbolic + i,
                                exact + i);
            }
            break;
        }
        case rocsparse_format_csr:
        {
            auto& host = A.template as<rocsparse_format_csr>().host();
            for(int64_t i = 0; i < batch_count; ++i)
            {
                const T* p    = host.val.data() + i * A.get_stride();
                const T* p_hx = x.host().data() + i * x.get_stride();
                T*       p_hy = y.host().data() + i * y.get_stride();
                cpu_csrsv<I, J, T>(operation,
                                   host.m,
                                   host.nnz,
                                   *halpha,
                                   host.ptr,
                                   host.ind,
                                   p,
                                   p_hx,
                                   (int64_t)1,
                                   p_hy,
                                   diag,
                                   uplo,
                                   host.base,
                                   symbolic + i,
                                   exact + i);
            }

            break;
        }

        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_sell:
        case rocsparse_format_bell:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        {
            break;
        }
        }
    }
}

template <typename I, typename J, typename T>
void testing_sptrsv_bad_arg(const Arguments& arg)
{
}

template <typename I, typename J, typename T>
void testing_sptrsv(const Arguments& arg)
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
    rocsparse_clients::sptrsv_descr sptrsv_descr(
        handle, batch_count, operation, alg, ttype, ttype, apol);

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

        rocsparse_clients::sptrsv_host<T, I, J>(batch_count,
                                                sptrsv_descr,
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

        int64_t                A_m         = A.get_nrows();
        int64_t                A_nnz       = 0;
        int64_t                A_bdim      = 1;
        double                 gbyte_count = 0;
        const rocsparse_format format      = A.get_format();
        switch(format)
        {

        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
        case rocsparse_format_coo_aos:
        {
            break;
        }

        case rocsparse_format_coo:
        {
            auto& device = A.template as<rocsparse_format_coo>().device();
            gbyte_count  = coosv_gbyte_count<T>(device.m, device.nnz);
            A_nnz        = device.nnz;
            break;
        }

        case rocsparse_format_csr:
        {
            auto& device = A.template as<rocsparse_format_csr>().device();
            gbyte_count  = csrsv_gbyte_count<T>(device.m, device.nnz);
            A_nnz        = device.nnz;
            break;
        }

        case rocsparse_format_bsr:
        {
            auto& device = A.template as<rocsparse_format_bsr>().device();
            gbyte_count
                = csrsv_gbyte_count<T>(device.mb * device.row_block_dim,
                                       device.nnzb * device.row_block_dim * device.col_block_dim);
            A_nnz  = device.nnzb * device.row_block_dim * device.col_block_dim;
            A_bdim = device.row_block_dim;
            break;
        }
        }

        const double gflop_count = spsv_gflop_count(A_m, A_nnz, diag);
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::M,
                            A_m,
                            display_key_t::nnz_A,
                            A_nnz,
                            display_key_t::bdim_A,
                            A_bdim,
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
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                             \
    template void testing_sptrsv_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sptrsv<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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

void testing_sptrsv_extra(const Arguments& arg) {}
