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
#include "testing_spic0.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"

namespace rocsparse_clients
{
    class spic0_descr
    {
    private:
        rocsparse_handle      m_handle{};
        rocsparse_spic0_descr m_descr{};

    public:
        host_dense_vector<int64_t> m_cpu_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_exact_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_near_singularity_position{};

        host_dense_vector<int64_t> m_host_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_host_numeric_singularity_position{};

        device_dense_vector<int64_t> m_device_symbolic_singularity_position{};
        device_dense_vector<int64_t> m_device_numeric_singularity_position{};

        spic0_descr(rocsparse_handle handle, int64_t batch_count)
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
                = rocsparse_spic0_descr_create(handle, &this->m_descr, p_error);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }

        ~spic0_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error* p_error = nullptr;
            std::ignore = rocsparse_spic0_descr_destroy(this->m_handle, this->m_descr, p_error);
        }

        inline operator rocsparse_spic0_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_spic0_descr&() const
        {
            return this->m_descr;
        }
    };
}

template <typename T, typename I, typename J>
void host_csric0(J                    M,
                 const I*             csr_row_ptr,
                 const J*             csr_col_ind,
                 T*                   csr_val,
                 rocsparse_index_base base,
                 int64_t*             struct_pivot,
                 int64_t*             numeric_pivot,
                 int64_t*             singular_pivot,
                 double               tol)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivot
    *struct_pivot   = -1;
    *numeric_pivot  = -1;
    *singular_pivot = -1;

    // pointer of upper part of each row
    std::vector<I> diag_offset(M);
    std::vector<I> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(J ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        I row_begin = csr_row_ptr[ai] - base;
        I row_end   = csr_row_ptr[ai + 1] - base;
        J j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        T sum = static_cast<T>(0);

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            J col_j = csr_col_ind[j] - base;
            T val_j = csr_val[j];

            // Mark diagonal and skip row
            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            // Skip upper triangular
            if(col_j > ai)
            {
                break;
            }

            I row_begin_j = csr_row_ptr[col_j] - base;
            I row_diag_j  = diag_offset[col_j];

            T local_sum = static_cast<T>(0);
            T diag_val  = csr_val[row_diag_j];
            T inv_diag  = static_cast<T>(0);

            // Check for numeric negative
            if((std::real(diag_val) <= tol) && (std::imag(diag_val) == 0))
            {
                // Numerically negative diagonal
                *singular_pivot = (*singular_pivot == -1)
                                      ? (col_j + base)
                                      : std::min(*singular_pivot, int64_t(col_j + base));
            }

            // Check for numeric zero
            if(diag_val == static_cast<T>(0))
            {
                // Numerically zero diagonal
                *numeric_pivot = (*numeric_pivot == -1)
                                     ? (col_j + base)
                                     : std::min(*numeric_pivot, int64_t(col_j + base));
            }
            else
            {

                inv_diag = static_cast<T>(1) / diag_val;
            }

            // loop over upper offset pointer and do linear combination for nnz entry
            for(I k = row_begin_j; k < row_diag_j; ++k)
            {
                J col_k = csr_col_ind[k] - base;

                // if nnz at this position do linear combination
                if(nnz_entries[col_k] != 0)
                {
                    I idx     = nnz_entries[col_k];
                    local_sum = std::fma(csr_val[k], rocsparse_conj(csr_val[idx]), local_sum);
                }
            }

            val_j = (val_j - local_sum) * inv_diag;
            sum   = std::fma(val_j, rocsparse_conj(val_j), sum);

            csr_val[j] = val_j;
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot
                = (*struct_pivot == -1) ? (ai + base) : std::min(*struct_pivot, int64_t(ai + base));
            *numeric_pivot = (*numeric_pivot == -1) ? (ai + base)
                                                    : std::min(*numeric_pivot, int64_t(ai + base));
        }
        else
        {
            // Store diagonal offset
            diag_offset[ai] = j;

            // Process diagonal entry
            T diag_entry = csr_val[j] - sum;
            csr_val[j]   = std::sqrt(std::abs(diag_entry));
            auto tolXtol = tol * tol;
            if((std::real(diag_entry) <= tolXtol) && (std::imag(diag_entry) == 0))
            {
                *singular_pivot = (*singular_pivot == -1)
                                      ? (ai + base)
                                      : std::min(*singular_pivot, int64_t(ai + base));
            }

            // check for zero diagonal
            if(diag_entry == static_cast<T>(0))
            {
                *numeric_pivot = (*numeric_pivot == -1)
                                     ? (ai + base)
                                     : std::min(*numeric_pivot, int64_t(ai + base));
            }
        }

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
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

template <typename T, typename I, typename J>
void host_bsric0(rocsparse_direction  direction,
                 J                    Mb,
                 J                    block_dim,
                 const I*             bsr_row_ptr,
                 const J*             bsr_col_ind,
                 T*                   bsr_val,
                 rocsparse_index_base base,
                 int64_t*             struct_pivot,
                 int64_t*             numeric_pivot)

{

    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    J M = Mb * block_dim;

    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<I> diag_block_offset(Mb);
    std::vector<I> diag_offset(M, -1);
    std::vector<I> nnz_entries(M, -1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(J i = 0; i < Mb; i++)
    {
        I row_begin = bsr_row_ptr[i] - base;
        I row_end   = bsr_row_ptr[i + 1] - base;

        for(I j = row_begin; j < row_end; j++)
        {
            if(bsr_col_ind[j] - base == i)
            {
                diag_block_offset[i] = j;
                break;
            }
        }
    }

    for(J i = 0; i < M; i++)
    {
        J local_row = i % block_dim;

        I row_begin = bsr_row_ptr[i / block_dim] - base;
        I row_end   = bsr_row_ptr[i / block_dim + 1] - base;

        for(I j = row_begin; j < row_end; j++)
        {
            J block_col_j = bsr_col_ind[j] - base;

            for(J k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * local_row + k;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k]
                        = block_dim * block_dim * j + block_dim * k + local_row;
                }
            }
        }

        T sum            = static_cast<T>(0);
        I diag_val_index = -1;

        bool has_diag         = false;
        bool break_outer_loop = false;

        for(I j = row_begin; j < row_end; j++)
        {
            J block_col_j = bsr_col_ind[j] - base;
            for(J k = 0; k < block_dim; k++)
            {
                J col_j = block_dim * block_col_j + k;

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    diag_val_index = block_dim * block_dim * j + block_dim * k + k;

                    has_diag         = true;
                    break_outer_loop = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break_outer_loop = true;
                    break;
                }

                T val_j = static_cast<T>(0);
                if(direction == rocsparse_direction_row)
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * local_row + k];
                }
                else
                {
                    val_j = bsr_val[block_dim * block_dim * j + block_dim * k + local_row];
                }

                J local_row_j = col_j % block_dim;

                I row_begin_j = bsr_row_ptr[col_j / block_dim] - base;
                I row_end_j   = diag_block_offset[col_j / block_dim];
                I row_diag_j  = diag_offset[col_j];

                T local_sum = static_cast<T>(0);
                T inv_diag  = row_diag_j != -1 ? bsr_val[row_diag_j] : static_cast<T>(0);

                // Check for numeric zero
                if(inv_diag == static_cast<T>(0))
                {
                    // Numerically non-invertible block diagonal
                    if(*numeric_pivot == -1)
                    {
                        *numeric_pivot = block_col_j + base;
                    }

                    *numeric_pivot = std::min(*numeric_pivot, int64_t(block_col_j + base));
                    inv_diag       = static_cast<T>(1);
                }

                inv_diag = static_cast<T>(1) / inv_diag;

                // loop over upper offset pointer and do linear combination for nnz entry
                for(I l = row_begin_j; l < row_end_j + 1; l++)
                {
                    J block_col_l = bsr_col_ind[l] - base;

                    for(J m = 0; m < block_dim; m++)
                    {
                        I idx = nnz_entries[block_dim * block_col_l + m];

                        if(idx != -1 && block_dim * block_col_l + m < col_j)
                        {
                            if(direction == rocsparse_direction_row)
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * local_row_j + m],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                            else
                            {
                                local_sum = std::fma(bsr_val[block_dim * block_dim * l
                                                             + block_dim * m + local_row_j],
                                                     rocsparse_conj(bsr_val[idx]),
                                                     local_sum);
                            }
                        }
                    }
                }
                val_j = (val_j - local_sum) * inv_diag;
                sum   = std::fma(val_j, rocsparse_conj(val_j), sum);
                if(direction == rocsparse_direction_row)
                {
                    bsr_val[block_dim * block_dim * j + block_dim * local_row + k] = val_j;
                }
                else
                {
                    bsr_val[block_dim * block_dim * j + block_dim * k + local_row] = val_j;
                }
            }

            if(break_outer_loop)
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural missing block diagonal
            if(*struct_pivot == -1)
            {
                *struct_pivot = i / block_dim + base;
            }
        }

        // Process diagonal entry
        if(has_diag)
        {
            T diag_entry            = std::sqrt(std::abs(bsr_val[diag_val_index] - sum));
            bsr_val[diag_val_index] = diag_entry;
            if(diag_entry == static_cast<T>(0))
            {
                // Numerically non-invertible block diagonal
                if(*numeric_pivot == -1)
                {
                    *numeric_pivot = i / block_dim + base;
                }

                *numeric_pivot = std::min(*numeric_pivot, int64_t(i / block_dim + base));
            }

            // Store diagonal offset
            diag_offset[i] = diag_val_index;
        }

        for(I j = row_begin; j < row_end; j++)
        {
            J block_col_j = bsr_col_ind[j] - base;

            for(J k = 0; k < block_dim; k++)
            {
                if(direction == rocsparse_direction_row)
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
                else
                {
                    nnz_entries[block_dim * block_col_j + k] = -1;
                }
            }
        }
    }

    // Solve pivot gives the first numerical or structural non-invertible block
    if(numeric_pivot[0] == -1)
    {
        numeric_pivot[0] = struct_pivot[0];
    }
    else if(struct_pivot[0] != -1)
    {
        numeric_pivot[0] = std::min(numeric_pivot[0], struct_pivot[0]);
    }
}

template <typename I, typename J, typename T>
void testing_spic0_set_input_bad_arg(const Arguments& arg)
{

    rocsparse_error*            p_error = nullptr;
    rocsparse_local_handle      local_handle;
    rocsparse_spic0_descr       spic0_descr        = (rocsparse_spic0_descr)0x4;
    rocsparse_handle            handle             = local_handle;
    static constexpr int        nex                = 2;
    static const int            ex[nex]            = {4, 5};
    const rocsparse_spic0_input input              = rocsparse_spic0_input_alg;
    void*                       data               = (void*)0x4;
    size_t                      data_size_in_bytes = sizeof(input);
    select_bad_arg_analysis(rocsparse_spic0_set_input,
                            nex,
                            ex,
                            handle, //0
                            spic0_descr, //1
                            input, //2
                            data, //3
                            data_size_in_bytes, //4
                            p_error); // 5

    CHECK_ROCSPARSE_ERROR(rocsparse_spic0_descr_create(handle, &spic0_descr, p_error));
    for(auto e : {rocsparse_spic0_input_alg,
                  rocsparse_spic0_input_analysis_policy,
                  rocsparse_spic0_input_compute_datatype,
                  rocsparse_spic0_input_boost_enable,
                  rocsparse_spic0_input_boost_tolerance,
                  rocsparse_spic0_input_boost_value,
                  rocsparse_spic0_input_singularity_tolerance})
    {
        switch(e)
        {
        case rocsparse_spic0_input_alg:
        case rocsparse_spic0_input_analysis_policy:
        case rocsparse_spic0_input_compute_datatype:
        case rocsparse_spic0_input_boost_enable:
        case rocsparse_spic0_input_boost_tolerance:
        case rocsparse_spic0_input_boost_value:
        case rocsparse_spic0_input_singularity_tolerance:
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spic0_set_input(handle, spic0_descr, e, data, 0, p_error),
                rocsparse_status_invalid_value);
            break;
        }
        }
    }

    //
    // Singularity force to be double
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_spic0_set_input(handle,
                                                      spic0_descr,
                                                      rocsparse_spic0_input_singularity_tolerance,
                                                      data,
                                                      sizeof(float),
                                                      p_error),
                            rocsparse_status_invalid_value);

    CHECK_ROCSPARSE_ERROR(rocsparse_spic0_descr_destroy(handle, spic0_descr, p_error));
}

template <typename I, typename J, typename T>
void testing_spic0_buffer_size_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;

    rocsparse_handle      handle      = local_handle;
    rocsparse_spic0_descr spic0_descr = (rocsparse_spic0_descr)0x4;

    rocsparse_spmat_descr A           = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr P           = (rocsparse_spmat_descr)0x4;
    rocsparse_spic0_stage spic0_stage = rocsparse_spic0_stage_analysis;

    rocsparse_error* p_error = nullptr;
    {
        size_t*              p_buffer_size_in_bytes = (size_t*)0x4;
        static constexpr int nex                    = 1;
        static const int     ex[nex]                = {6};
        select_bad_arg_analysis(rocsparse_spic0_buffer_size,
                                nex,
                                ex,
                                handle,
                                spic0_descr,
                                A,
                                P,
                                spic0_stage,
                                p_buffer_size_in_bytes,
                                p_error);
    }
}

template <typename I, typename J, typename T>
void testing_spic0_get_output_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle             = local_handle;
    rocsparse_spic0_descr  spic0_descr        = (rocsparse_spic0_descr)0x4;
    rocsparse_error*       p_error            = nullptr;
    static constexpr int   nex                = 2;
    static const int       ex[nex]            = {4, 5};
    size_t                 data_size_in_bytes = sizeof(int64_t);
    void*                  data               = (void*)0x4;
    rocsparse_spic0_output output             = rocsparse_spic0_output_singularity_position;
    select_bad_arg_analysis(rocsparse_spic0_get_output,
                            nex,
                            ex,
                            handle,
                            spic0_descr,
                            output,
                            data,
                            data_size_in_bytes,
                            p_error);
}

template <typename I, typename J, typename T>
void testing_spic0_analysis_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle      = local_handle;
    rocsparse_spic0_descr  spic0_descr = (rocsparse_spic0_descr)0x4;
    rocsparse_spmat_descr  A           = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr  P           = (rocsparse_spmat_descr)0x4;
    rocsparse_spic0_stage  spic0_stage = rocsparse_spic0_stage_analysis;
    void*                  buffer      = (void*)0x4;
    rocsparse_error*       p_error     = nullptr;
    {
        const size_t buffer_size_in_bytes = 1;
#define PARAMS handle, spic0_descr, A, P, spic0_stage, buffer_size_in_bytes, buffer, p_error
        static constexpr int nex     = 2;
        static const int     ex[nex] = {5, 7};
        select_bad_arg_analysis(rocsparse_spic0, nex, ex, PARAMS);
#undef PARAMS
    }
}

template <typename I, typename J, typename T>
void testing_spic0_bad_arg(const Arguments& arg)
{

    testing_spic0_buffer_size_bad_arg<I, J, T>(arg);
    testing_spic0_analysis_bad_arg<I, J, T>(arg);
    testing_spic0_set_input_bad_arg<I, J, T>(arg);
    testing_spic0_get_output_bad_arg<I, J, T>(arg);

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

    for(auto analysis_policy : {rocsparse_analysis_policy_force, rocsparse_analysis_policy_reuse})
    {
        rocsparse_clients::spic0_descr spic0_descr(handle, 1);

        const rocsparse_spic0_alg alg              = rocsparse_spic0_alg_default;
        const rocsparse_datatype  compute_datatype = get_datatype<T>();
        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(
            handle, spic0_descr, rocsparse_spic0_input_alg, &alg, sizeof(alg), p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(handle,
                                                        spic0_descr,
                                                        rocsparse_spic0_input_analysis_policy,
                                                        &analysis_policy,
                                                        sizeof(analysis_policy),
                                                        p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(handle,
                                                        spic0_descr,
                                                        rocsparse_spic0_input_compute_datatype,
                                                        &compute_datatype,
                                                        sizeof(compute_datatype),
                                                        p_error));

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_buffer_size(
            handle, spic0_descr, A, P, rocsparse_spic0_stage_analysis, &buffer_size, p_error));
        hipStream_t stream;
        CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));
        {
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            //
            // Call compute before analysis.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spic0(handle,
                                                    spic0_descr,
                                                    A,
                                                    P,
                                                    rocsparse_spic0_stage_compute,
                                                    buffer_size,
                                                    buffer,
                                                    p_error));

            //
            // Call analysis.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  P,
                                                  rocsparse_spic0_stage_analysis,
                                                  buffer_size,
                                                  buffer,
                                                  p_error));

            for(auto input : {rocsparse_spic0_input_alg,
                              rocsparse_spic0_input_analysis_policy,
                              rocsparse_spic0_input_compute_datatype,
                              rocsparse_spic0_input_boost_enable,
                              rocsparse_spic0_input_boost_tolerance,
                              rocsparse_spic0_input_boost_value,
                              rocsparse_spic0_input_singularity_tolerance})
            {
                switch(input)
                {
                case rocsparse_spic0_input_alg:
                case rocsparse_spic0_input_compute_datatype:
                case rocsparse_spic0_input_analysis_policy:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spic0_set_input(
                            handle, spic0_descr, input, (void*)0x4, sizeof(int64_t), p_error));
                    break;
                }
                case rocsparse_spic0_input_singularity_tolerance:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spic0_set_input(
                            handle, spic0_descr, input, (void*)0x4, sizeof(float), p_error));

                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spic0_set_input(handle,
                                                  spic0_descr,
                                                  rocsparse_spic0_input_singularity_tolerance,
                                                  hsingularity_tolerance,
                                                  sizeof(double),
                                                  p_error));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spic0_set_input(handle,
                                                  spic0_descr,
                                                  rocsparse_spic0_input_singularity_tolerance,
                                                  dsingularity_tolerance,
                                                  sizeof(double),
                                                  p_error));
                    break;
                }
                case rocsparse_spic0_input_boost_enable:
                case rocsparse_spic0_input_boost_tolerance:
                case rocsparse_spic0_input_boost_value:
                {
                    break;
                }
                }
            }

            //
            // Call analysis twice.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spic0(handle,
                                                    spic0_descr,
                                                    A,
                                                    P,
                                                    rocsparse_spic0_stage_analysis,
                                                    buffer_size,
                                                    buffer,
                                                    p_error));
        }

        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0_buffer_size(
                handle, spic0_descr, A, P, rocsparse_spic0_stage_compute, &buffer_size, p_error));
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spic0_set_input(handle,
                                          spic0_descr,
                                          rocsparse_spic0_input_singularity_tolerance,
                                          hsingularity_tolerance,
                                          sizeof(double),
                                          p_error));

            //
            // Call compute.
            //

            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  P,
                                                  rocsparse_spic0_stage_compute,
                                                  buffer_size,
                                                  buffer,
                                                  p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spic0(handle,
                                                    spic0_descr,
                                                    A,
                                                    P,
                                                    rocsparse_spic0_stage_analysis,
                                                    buffer_size,
                                                    buffer,
                                                    p_error));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spic0_set_input(handle,
                                          spic0_descr,
                                          rocsparse_spic0_input_singularity_tolerance,
                                          dsingularity_tolerance,
                                          sizeof(double),
                                          p_error));

            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  P,
                                                  rocsparse_spic0_stage_compute,
                                                  buffer_size,
                                                  buffer,
                                                  p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spic0(handle,
                                                    spic0_descr,
                                                    A,
                                                    P,
                                                    rocsparse_spic0_stage_analysis,
                                                    buffer_size,
                                                    buffer,
                                                    p_error));
            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  P,
                                                  rocsparse_spic0_stage_compute,
                                                  buffer_size,
                                                  buffer,
                                                  p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }
    }
}

template <typename T, typename I, typename J = I>
void rocsparse_clients_spic0_host(rocsparse_clients::spic0_descr&          spic0_descr,
                                  rocsparse_clients::spmat_descr<T, I, J>& A,
                                  const double*                            singular_pivot_tolerance)
{
    auto&         cpu_symbolic_pivot      = spic0_descr.m_cpu_symbolic_singularity_position;
    auto&         cpu_numeric_near_pivot  = spic0_descr.m_cpu_numeric_near_singularity_position;
    auto&         cpu_numeric_exact_pivot = spic0_descr.m_cpu_numeric_exact_singularity_position;
    const int64_t batch_count             = A.get_batch_count();
    const rocsparse_format format         = A.get_format();
    switch(format)
    {
    case rocsparse_format_csr:
    {
        auto& host = A.template as<rocsparse_format_csr>().host();
        for(int64_t i = 0; i < batch_count; ++i)
        {
            T* p = host.val.data() + i * A.get_stride();
            host_csric0<T, I, J>(host.m,
                                 host.ptr,
                                 host.ind,
                                 p,
                                 host.base,
                                 cpu_symbolic_pivot + i,
                                 cpu_numeric_exact_pivot + i,
                                 cpu_numeric_near_pivot + i,
                                 singular_pivot_tolerance[0]);
        }

        for(int64_t j = 0; j < batch_count; ++j)
        {
            if(cpu_numeric_near_pivot[j] == -1)
            {
                cpu_numeric_near_pivot[j] = cpu_numeric_exact_pivot[j];
            }
            else
            {
                if(cpu_numeric_exact_pivot[j] != -1)
                {
                    cpu_numeric_near_pivot[j]
                        = std::min(cpu_numeric_near_pivot[j], cpu_numeric_exact_pivot[j]);
                }
            }
        }

        break;
    }

    case rocsparse_format_bsr:
    {
        auto& host = A.template as<rocsparse_format_bsr>().host();
        for(int64_t i = 0; i < batch_count; ++i)
        {
            T* p = host.val.data() + i * A.get_stride();
            host_bsric0<T, I, J>(host.block_direction,
                                 host.mb,
                                 host.row_block_dim,
                                 host.ptr,
                                 host.ind,
                                 p,
                                 host.base,
                                 cpu_symbolic_pivot + i,
                                 cpu_numeric_exact_pivot + i);
        }

        for(int64_t j = 0; j < batch_count; ++j)
        {
            cpu_numeric_near_pivot[j] = cpu_numeric_exact_pivot[j];
        }
        //
        // Set numeric_near_pivot to -1.
        //
        break;
    }

    case rocsparse_format_ell:
    case rocsparse_format_sell:
    case rocsparse_format_bell:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_csc:
    {
        break;
    }
    }
}

template <typename I, typename J, typename T>
void testing_spic0(const Arguments& arg)
{
    rocsparse_error* p_error = nullptr;
    //
    // Import matrix.
    //
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
    rocsparse_local_handle handle(arg);
    hipStream_t            stream;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    const double                   numeric_pivot_tolerance = 0.0000;
    rocsparse_clients::spic0_descr spic0_descr(handle, batch_count);

    const rocsparse_spic0_alg       alg              = rocsparse_spic0_alg_default;
    const rocsparse_analysis_policy analysis_policy  = arg.apol;
    const rocsparse_datatype        compute_datatype = get_datatype<T>();

    CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(
        handle, spic0_descr, rocsparse_spic0_input_alg, &alg, sizeof(alg), p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(handle,
                                                    spic0_descr,
                                                    rocsparse_spic0_input_analysis_policy,
                                                    &analysis_policy,
                                                    sizeof(analysis_policy),
                                                    p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spic0_set_input(handle,
                                                    spic0_descr,
                                                    rocsparse_spic0_input_compute_datatype,
                                                    &compute_datatype,
                                                    sizeof(compute_datatype),
                                                    p_error));

    //
    // Perform analysis.
    //

    auto& host_symbolic_pivot = spic0_descr.m_host_symbolic_singularity_position;

    {
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_buffer_size(handle,
                                                          spic0_descr,
                                                          A,
                                                          A,
                                                          rocsparse_spic0_stage_analysis,
                                                          &buffer_size_in_bytes,
                                                          p_error));
        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                              spic0_descr,
                                              A,
                                              A,
                                              rocsparse_spic0_stage_analysis,
                                              buffer_size_in_bytes,
                                              buffer,
                                              p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spic0_get_output(handle,
                                       spic0_descr,
                                       rocsparse_spic0_output_singularity_position,
                                       host_symbolic_pivot,
                                       sizeof(int64_t),
                                       p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    auto& device_symbolic_pivot = spic0_descr.m_device_symbolic_singularity_position;
    auto& host_numeric_pivot    = spic0_descr.m_host_numeric_singularity_position;
    auto& device_numeric_pivot  = spic0_descr.m_device_numeric_singularity_position;
    if(arg.unit_check)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spic0_get_output(handle,
                                       spic0_descr,
                                       rocsparse_spic0_output_singularity_position,
                                       device_symbolic_pivot,
                                       sizeof(int64_t),
                                       p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        //
        // Check pivot is the same with either host or device mode.
        //
        host_symbolic_pivot.unit_check(device_symbolic_pivot);

        //
        // Check consistency of the status and the symbolic resulting pivot.
        //
        //
        // Perform calculation.
        //
        {
            size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0_buffer_size(handle,
                                                              spic0_descr,
                                                              A,
                                                              A,
                                                              rocsparse_spic0_stage_compute,
                                                              &buffer_size_in_bytes,
                                                              p_error));

            device_dense_vector<char> buffer(buffer_size_in_bytes);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  A,
                                                  rocsparse_spic0_stage_compute,
                                                  buffer_size_in_bytes,
                                                  buffer,
                                                  p_error));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spic0_get_output(handle,
                                           spic0_descr,
                                           rocsparse_spic0_output_singularity_position,
                                           host_numeric_pivot,
                                           sizeof(int64_t),
                                           p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spic0_get_output(handle,
                                           spic0_descr,
                                           rocsparse_spic0_output_singularity_position,
                                           device_numeric_pivot,
                                           sizeof(int64_t),
                                           p_error));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));

            //
            // Check numeric pivots match.
            //
            host_numeric_pivot.unit_check(device_numeric_pivot);

            //
            // Perform calculation on host.
            //
            auto& cpu_numeric_near_pivot = spic0_descr.m_cpu_numeric_near_singularity_position;
            auto& cpu_symbolic_pivot     = spic0_descr.m_cpu_symbolic_singularity_position;

            rocsparse_clients_spic0_host(spic0_descr, A, &numeric_pivot_tolerance);
            cpu_symbolic_pivot.unit_check(host_symbolic_pivot);
            cpu_numeric_near_pivot.unit_check(host_numeric_pivot);

            //
            // Check values only if analysis stage and compute stage
            // didn't encounter a zero pivot.
            //
            A.near_check_values(host_symbolic_pivot, host_numeric_pivot);
        }
    }

    if(arg.timing)
    {
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spic0_buffer_size(handle,
                                                          spic0_descr,
                                                          A,
                                                          A,
                                                          rocsparse_spic0_stage_compute,
                                                          &buffer_size_in_bytes,
                                                          p_error));

        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        const int32_t n_cold_calls = 2;
        const int32_t n_sub_calls  = arg.iters_inner;
        const int32_t n_calls      = arg.iters;

        hipStream_t stream;
        rocsparse_get_stream(handle, &stream);

        for(int32_t iter = 0; iter < n_cold_calls; ++iter)
        {
            A.reinit_values();
            CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                  spic0_descr,
                                                  A,
                                                  A,
                                                  rocsparse_spic0_stage_compute,
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
                CHECK_ROCSPARSE_ERROR(rocsparse_spic0(handle,
                                                      spic0_descr,
                                                      A,
                                                      A,
                                                      rocsparse_spic0_stage_compute,
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
        const rocsparse_format format = A.get_format();

        switch(format)
        {
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
        case rocsparse_format_coo_aos:
        case rocsparse_format_coo:
        {
            break;
        }
        case rocsparse_format_csr:
        {
            auto&        device      = A.template as<rocsparse_format_csr>().device();
            double       gbyte_count = csric0_gbyte_count<T>(device.m, device.nnz);
            const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
            display_timing_info(display_key_t::M,
                                device.m,
                                display_key_t::nnz_A,
                                device.nnz,
                                display_key_t::bandwidth,
                                gpu_gbyte,
                                display_key_t::time_ms,
                                get_gpu_time_msec(gpu_time_used));
            break;
        }
        case rocsparse_format_bsr:
        {
            auto&  device = A.template as<rocsparse_format_bsr>().device();
            double gbyte_count
                = bsric0_gbyte_count<T>(device.mb, device.row_block_dim, device.nnzb);
            const double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);
            display_timing_info(display_key_t::M,
                                device.mb,
                                display_key_t::nnzb,
                                device.nnzb,
                                display_key_t::bdim,
                                device.row_block_dim,
                                display_key_t::bandwidth,
                                gpu_gbyte,
                                display_key_t::time_ms,
                                get_gpu_time_msec(gpu_time_used));

            break;
        }
        }
    }
}

void testing_spic0_extra(const Arguments& arg) {}

#define INSTANTIATE(I, J, T)                                            \
    template void testing_spic0_bad_arg<I, J, T>(const Arguments& arg); \
    template void testing_spic0<I, J, T>(const Arguments& arg)

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
