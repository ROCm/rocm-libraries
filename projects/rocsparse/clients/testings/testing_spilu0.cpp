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

#include "testing_spilu0.hpp"
#include "rocsparse_clients_objects.hpp"
#include "rocsparse_clients_spmat_descr.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"

namespace rocsparse_clients
{
    class spilu0_descr
    {
    private:
        rocsparse_handle       m_handle{};
        rocsparse_spilu0_descr m_descr{};

    public:
        struct config
        {
            rocsparse_spilu0_alg      alg{};
            rocsparse_analysis_policy analysis_policy{};
            rocsparse_datatype        compute_datatype{};
            rocsparse_pointer_mode    singular_pivot_tolerance_pointer_mode{};
            const void*               singular_pivot_tolerance{};
            const int32_t             boost_enable{};
            rocsparse_pointer_mode    boost_val_pointer_mode{};
            const void*               boost_val{};
            rocsparse_pointer_mode    boost_tolerance_pointer_mode{};
            const void*               boost_tolerance{};
        };

    protected:
        config* m_config;

    public:
        const config* get_config() const
        {
            return this->m_config;
        }

        host_dense_vector<int64_t> m_cpu_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_exact_singularity_position{};
        host_dense_vector<int64_t> m_cpu_numeric_near_singularity_position{};

        host_dense_vector<int64_t> m_host_symbolic_singularity_position{};
        host_dense_vector<int64_t> m_host_numeric_singularity_position{};

        device_dense_vector<int64_t> m_device_symbolic_singularity_position{};
        device_dense_vector<int64_t> m_device_numeric_singularity_position{};

        spilu0_descr(rocsparse_handle handle, int64_t batch_count)
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
                = rocsparse_spilu0_descr_create(handle, &this->m_descr, p_error);
            if(status != rocsparse_status_success)
            {
                throw(status);
            }
        }

        ~spilu0_descr()
        {
            ROCSPARSE_CLIENTS_ROUTINE_TRACE;
            rocsparse_error* p_error = nullptr;
            std::ignore = rocsparse_spilu0_descr_destroy(this->m_handle, this->m_descr, p_error);
        }

        inline operator rocsparse_spilu0_descr&()
        {
            return this->m_descr;
        }

        inline operator const rocsparse_spilu0_descr&() const
        {
            return this->m_descr;
        }
    };

}

template <typename T, typename I, typename J>
void host_csrilu0(J                         M,
                  const I*                  csr_row_ptr,
                  const J*                  csr_col_ind,
                  T*                        csr_val,
                  rocsparse_index_base      base,
                  int64_t*                  struct_pivot,
                  int64_t*                  numeric_pivot,
                  int64_t*                  singular_pivot,
                  const double*             singular_tol,
                  int32_t                   boost_enable,
                  const floating_data_t<T>* boost_tol,
                  const T*                  boost_val)
{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    if((struct_pivot == nullptr) || (numeric_pivot == nullptr) || (singular_pivot == nullptr))
    {
        std::cerr << "host_csrilu0: invalid parameters, failed condition: ((struct_pivot != "
                     "nullptr) && (numeric_pivot != nullptr) && (singular_pivot != nullptr))."
                  << std::endl;
        throw(1);
    }

    // Initialize pivot
    *struct_pivot   = -1;
    *numeric_pivot  = -1;
    *singular_pivot = -1;

    // pointer of upper part of each row
    std::vector<I> diag_offset(M, -1);
    std::vector<I> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(J ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        I row_begin = csr_row_ptr[ai] - base;
        I row_end   = csr_row_ptr[ai + 1] - base;
        I j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        bool has_diag = false;
        I    diag_pos = -1;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(csr_col_ind[j] - base < ai)
            {

                J col_j  = csr_col_ind[j] - base;
                I diag_j = diag_offset[col_j];
                if(diag_j < 0)
                    continue;

                T diag_val = csr_val[diag_j];

                if(boost_enable)
                {
                    diag_val = (boost_tol[0] >= std::abs(diag_val)) ? boost_val[0] : diag_val;
                    csr_val[diag_j] = diag_val;
                }
                else
                {

                    // Check for numeric singular pivot
                    if(std::abs(diag_val) <= singular_tol[0])
                    {
                        *singular_pivot = (*singular_pivot == -1)
                                              ? (col_j + base)
                                              : std::min(*singular_pivot, int64_t(col_j + base));
                    }

                    // Check for numeric zero pivot
                    if(diag_val == static_cast<T>(0))
                    {
                        *numeric_pivot = (*numeric_pivot == -1)
                                             ? (col_j + base)
                                             : std::min(*numeric_pivot, int64_t(col_j + base));
                        continue;
                    }
                }

                {
                    // multiplication factor

                    csr_val[j] = csr_val[j] / diag_val;

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(I k = diag_j + 1; k < csr_row_ptr[col_j + 1] - base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[csr_col_ind[k] - base] != 0)
                        {
                            I idx        = nnz_entries[csr_col_ind[k] - base];
                            csr_val[idx] = std::fma(-csr_val[j], csr_val[k], csr_val[idx]);
                        }
                    }
                }
            }
            else if(csr_col_ind[j] - base == ai)
            {
                has_diag = true;
                diag_pos = j;
                break;
            }
            else
            {
                break;
            }
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
            // set diagonal pointer to diagonal element
            diag_offset[ai] = diag_pos;
            if(boost_enable)
            {
                if(std::abs(csr_val[diag_pos]) <= boost_tol[0])
                {
                    csr_val[diag_pos] = boost_val[0];
                }
            }
            else
            {
                const I    diag_pos = diag_offset[ai];
                const bool is_diag  = (diag_pos >= 0) && (csr_col_ind[diag_pos] == (ai + base));

                const bool is_singular_diag
                    = is_diag && (std::abs(csr_val[diag_pos]) <= singular_tol[0]);
                const bool is_zero_diag = is_diag && (csr_val[diag_pos] == static_cast<T>(0));

                // check for singular diagonal
                if(is_singular_diag)
                {
                    *singular_pivot = (*singular_pivot == -1)
                                          ? (ai + base)
                                          : std::min(*singular_pivot, int64_t(ai + base));
                }

                // check for zero diagonal
                if(is_zero_diag)
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
void host_bsrilu0(rocsparse_direction       dir,
                  J                         mb,
                  const I*                  bsr_row_ptr,
                  const J*                  bsr_col_ind,
                  T*                        bsr_val,
                  J                         bsr_dim,
                  rocsparse_index_base      base,
                  int64_t*                  struct_pivot,
                  int64_t*                  numeric_pivot,
                  int32_t                   boost_enable,
                  const floating_data_t<T>* boost_tol,
                  const T*                  boost_val)

{
    ROCSPARSE_CLIENTS_ROUTINE_TRACE;

    // Initialize pivots
    *struct_pivot  = mb + 1;
    *numeric_pivot = mb + 1;

    // Temporary vector to hold diagonal offset to access diagonal BSR block
    std::vector<I> diag_offset(mb);
    std::vector<I> nnz_entries(mb, -1);

    if(mb > 0)
    {
        // First diagonal block is index 0
        diag_offset[0] = 0;
    }

    // Loop over all BSR rows
    for(J i = 0; i < mb; ++i)
    {
        // Flag whether we have a diagonal block or not
        bool has_diag = false;

        // BSR column entry and exit point
        I row_begin = bsr_row_ptr[i] - base;
        I row_end   = bsr_row_ptr[i + 1] - base;

        I j;

        // Set up entry points for linear combination
        for(j = row_begin; j < row_end; ++j)
        {
            J col_j            = bsr_col_ind[j] - base;
            nnz_entries[col_j] = j;
        }

        // Process lower diagonal BSR blocks (diagonal BSR block is excluded)
        for(j = row_begin; j < row_end; ++j)
        {
            // Column index of current BSR block
            J bsr_col = bsr_col_ind[j] - base;

            // If this is a diagonal block, set diagonal flag to true and skip
            // all upcoming blocks as we exceed the lower matrix part
            if(bsr_col == i)
            {
                has_diag = true;
                break;
            }

            // Skip all upper matrix blocks
            if(bsr_col > i)
            {
                break;
            }

            // Process all lower matrix BSR blocks

            // Obtain corresponding row entry and exit point that corresponds with the
            // current BSR column. Actually, we skip all lower matrix column indices,
            // therefore starting with the diagonal entry.
            I diag_j    = diag_offset[bsr_col];
            I row_end_j = bsr_row_ptr[bsr_col + 1] - base;

            // Loop through all rows within the BSR block
            for(J bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(diag_j, bi, bi, dir)];

                // Process all rows within the BSR block
                for(J bk = 0; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through columns of bk-th row and do linear combination
                    for(J bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(diag_j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }

            // Loop over upper offset pointer and do linear combination for nnz entry
            for(I k = diag_j + 1; k < row_end_j; ++k)
            {
                J bsr_col_k = bsr_col_ind[k] - base;

                if(nnz_entries[bsr_col_k] != -1)
                {
                    J m = nnz_entries[bsr_col_k];

                    // Loop through all rows within the BSR block
                    for(J bi = 0; bi < bsr_dim; ++bi)
                    {
                        // Loop through columns of bi-th row and do linear combination
                        for(J bj = 0; bj < bsr_dim; ++bj)
                        {
                            T sum = static_cast<T>(0);

                            for(J bk = 0; bk < bsr_dim; ++bk)
                            {
                                sum = std::fma(bsr_val[BSR_IND(j, bi, bk, dir)],
                                               bsr_val[BSR_IND(k, bk, bj, dir)],
                                               sum);
                            }

                            bsr_val[BSR_IND(m, bi, bj, dir)] -= sum;
                        }
                    }
                }
            }
        }

        // Check for structural pivot
        if(!has_diag)
        {
            *struct_pivot = std::min(*struct_pivot, int64_t(i + base));
            break;
        }

        // Process diagonal
        if(bsr_col_ind[j] - base == i)
        {
            // Loop through all rows within the BSR block
            for(J bi = 0; bi < bsr_dim; ++bi)
            {
                T diag = bsr_val[BSR_IND(j, bi, bi, dir)];

                if(boost_enable)
                {
                    diag = (boost_tol[0] >= std::abs(diag)) ? boost_val[0] : diag;
                    bsr_val[BSR_IND(j, bi, bi, dir)] = diag;
                }
                else
                {
                    // Check for numeric pivot
                    if(diag == static_cast<T>(0))
                    {
                        *numeric_pivot = std::min(*numeric_pivot, int64_t(bsr_col_ind[j]));
                        continue;
                    }
                }

                // Process all rows within the BSR block after bi-th row
                for(J bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    T val = bsr_val[BSR_IND(j, bk, bi, dir)];

                    // Multiplication factor
                    bsr_val[BSR_IND(j, bk, bi, dir)] = val /= diag;

                    // Loop through remaining columns of bk-th row and do linear combination
                    for(J bj = bi + 1; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-val,
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Store diagonal BSR block entry point
        I row_diag = diag_offset[i] = j;

        // Process upper diagonal BSR blocks
        for(j = row_diag + 1; j < row_end; ++j)
        {
            // Loop through all rows within the BSR block
            for(J bi = 0; bi < bsr_dim; ++bi)
            {
                // Process all rows within the BSR block after bi-th row
                for(J bk = bi + 1; bk < bsr_dim; ++bk)
                {
                    // Loop through columns of bk-th row and do linear combination
                    for(J bj = 0; bj < bsr_dim; ++bj)
                    {
                        bsr_val[BSR_IND(j, bk, bj, dir)]
                            = std::fma(-bsr_val[BSR_IND(row_diag, bk, bi, dir)],
                                       bsr_val[BSR_IND(j, bi, bj, dir)],
                                       bsr_val[BSR_IND(j, bk, bj, dir)]);
                    }
                }
            }
        }

        // Reset entry points
        for(j = row_begin; j < row_end; ++j)
        {
            J col_j            = bsr_col_ind[j] - base;
            nnz_entries[col_j] = -1;
        }
    }

    *struct_pivot  = (*struct_pivot == mb + 1) ? -1 : *struct_pivot;
    *numeric_pivot = (*numeric_pivot == mb + 1) ? -1 : *numeric_pivot;

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
void testing_spilu0_set_input_bad_arg(const Arguments& arg)
{
    rocsparse_error*             p_error = nullptr;
    rocsparse_local_handle       local_handle;
    rocsparse_spilu0_descr       spilu0_descr       = (rocsparse_spilu0_descr)0x4;
    rocsparse_handle             handle             = local_handle;
    static constexpr int         nex                = 2;
    static const int             ex[nex]            = {4, 5};
    const rocsparse_spilu0_input input              = rocsparse_spilu0_input_alg;
    void*                        data               = (void*)0x4;
    size_t                       data_size_in_bytes = sizeof(input);
    select_bad_arg_analysis(rocsparse_spilu0_set_input,
                            nex,
                            ex,
                            handle, //0
                            spilu0_descr, //1
                            input, //2
                            data, //3
                            data_size_in_bytes, //4
                            p_error); // 5

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_descr_create(handle, &spilu0_descr, p_error));
    for(auto e : {rocsparse_spilu0_input_alg,
                  rocsparse_spilu0_input_analysis_policy,
                  rocsparse_spilu0_input_compute_datatype,
                  rocsparse_spilu0_input_boost_enable,
                  rocsparse_spilu0_input_boost_tolerance,
                  rocsparse_spilu0_input_boost_value,
                  rocsparse_spilu0_input_singularity_tolerance})
    {
        switch(e)
        {
        case rocsparse_spilu0_input_alg:
        case rocsparse_spilu0_input_analysis_policy:
        case rocsparse_spilu0_input_compute_datatype:
        case rocsparse_spilu0_input_boost_enable:
        case rocsparse_spilu0_input_boost_tolerance:
        case rocsparse_spilu0_input_boost_value:
        case rocsparse_spilu0_input_singularity_tolerance:
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spilu0_set_input(handle, spilu0_descr, e, data, 0, p_error),
                rocsparse_status_invalid_value);
            break;
        }
        }
    }

    //
    // Singularity force to be double
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_spilu0_set_input(handle,
                                                       spilu0_descr,
                                                       rocsparse_spilu0_input_singularity_tolerance,
                                                       data,
                                                       sizeof(float),
                                                       p_error),
                            rocsparse_status_invalid_value);

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_descr_destroy(handle, spilu0_descr, p_error));
}

template <typename I, typename J, typename T>
void testing_spilu0_buffer_size_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;

    rocsparse_handle       handle       = local_handle;
    rocsparse_spilu0_descr spilu0_descr = (rocsparse_spilu0_descr)0x4;

    rocsparse_spmat_descr  A            = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr  P            = (rocsparse_spmat_descr)0x4;
    rocsparse_spilu0_stage spilu0_stage = rocsparse_spilu0_stage_analysis;

    rocsparse_error* p_error = nullptr;
    {
        size_t* p_buffer_size_in_bytes = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE handle, spilu0_descr, A, P, spilu0_stage, p_buffer_size_in_bytes, p_error

        static constexpr int nex     = 1;
        static const int     ex[nex] = {6};
        select_bad_arg_analysis(rocsparse_spilu0_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);

#undef PARAMS_BUFFER_SIZE
    }
}

template <typename I, typename J, typename T>
void testing_spilu0_get_output_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle  local_handle;
    rocsparse_handle        handle             = local_handle;
    rocsparse_spilu0_descr  spilu0_descr       = (rocsparse_spilu0_descr)0x4;
    rocsparse_error*        p_error            = nullptr;
    static constexpr int    nex                = 2;
    static const int        ex[nex]            = {4, 5};
    size_t                  data_size_in_bytes = sizeof(int64_t);
    void*                   data               = (void*)0x4;
    rocsparse_spilu0_output output             = rocsparse_spilu0_output_singularity_position;
    select_bad_arg_analysis(rocsparse_spilu0_get_output,
                            nex,
                            ex,
                            handle,
                            spilu0_descr,
                            output,
                            data,
                            data_size_in_bytes,
                            p_error);
}

template <typename I, typename J, typename T>
void testing_spilu0_analysis_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle       = local_handle;
    rocsparse_spilu0_descr spilu0_descr = (rocsparse_spilu0_descr)0x4;
    rocsparse_spmat_descr  A            = (rocsparse_spmat_descr)0x4;
    rocsparse_spmat_descr  P            = (rocsparse_spmat_descr)0x4;
    rocsparse_spilu0_stage spilu0_stage = rocsparse_spilu0_stage_analysis;
    void*                  buffer       = (void*)0x4;
    rocsparse_error*       p_error      = nullptr;
    {
        const size_t buffer_size_in_bytes = 1;
#define PARAMS handle, spilu0_descr, A, P, spilu0_stage, buffer_size_in_bytes, buffer, p_error
        static constexpr int nex     = 2;
        static const int     ex[nex] = {5, 7};
        select_bad_arg_analysis(rocsparse_spilu0, nex, ex, PARAMS);
#undef PARAMS
    }
}

template <typename I, typename J, typename T>
void testing_spilu0_bad_arg(const Arguments& arg)
{
    testing_spilu0_buffer_size_bad_arg<I, J, T>(arg);
    testing_spilu0_analysis_bad_arg<I, J, T>(arg);
    testing_spilu0_set_input_bad_arg<I, J, T>(arg);
    testing_spilu0_get_output_bad_arg<I, J, T>(arg);

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
        rocsparse_clients::spilu0_descr spilu0_descr(handle, 1);

        const rocsparse_spilu0_alg alg              = rocsparse_spilu0_alg_default;
        const rocsparse_datatype   compute_datatype = get_datatype<T>();
        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(
            handle, spilu0_descr, rocsparse_spilu0_input_alg, &alg, sizeof(alg), p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                         spilu0_descr,
                                                         rocsparse_spilu0_input_analysis_policy,
                                                         &analysis_policy,
                                                         sizeof(analysis_policy),
                                                         p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                         spilu0_descr,
                                                         rocsparse_spilu0_input_compute_datatype,
                                                         &compute_datatype,
                                                         sizeof(compute_datatype),
                                                         p_error));

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_buffer_size(
            handle, spilu0_descr, A, P, rocsparse_spilu0_stage_analysis, &buffer_size, p_error));

        {
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            //
            // Call compute before analysis.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spilu0(handle,
                                                     spilu0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spilu0_stage_compute,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            //
            // Call analysis.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   P,
                                                   rocsparse_spilu0_stage_analysis,
                                                   buffer_size,
                                                   buffer,
                                                   p_error));

            for(auto input : {rocsparse_spilu0_input_alg,
                              rocsparse_spilu0_input_analysis_policy,
                              rocsparse_spilu0_input_compute_datatype,
                              rocsparse_spilu0_input_boost_enable,
                              rocsparse_spilu0_input_boost_tolerance,
                              rocsparse_spilu0_input_boost_value,
                              rocsparse_spilu0_input_singularity_tolerance})
            {
                switch(input)
                {
                case rocsparse_spilu0_input_alg:
                case rocsparse_spilu0_input_compute_datatype:
                case rocsparse_spilu0_input_analysis_policy:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spilu0_set_input(
                            handle, spilu0_descr, input, (void*)0x4, sizeof(int64_t), p_error));
                    break;
                }
                case rocsparse_spilu0_input_singularity_tolerance:
                {
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_status_invalid_value,
                        rocsparse_spilu0_set_input(
                            handle, spilu0_descr, input, (void*)0x4, sizeof(float), p_error));

                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spilu0_set_input(handle,
                                                   spilu0_descr,
                                                   rocsparse_spilu0_input_singularity_tolerance,
                                                   hsingularity_tolerance,
                                                   sizeof(double),
                                                   p_error));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
                    CHECK_ROCSPARSE_ERROR(
                        rocsparse_spilu0_set_input(handle,
                                                   spilu0_descr,
                                                   rocsparse_spilu0_input_singularity_tolerance,
                                                   dsingularity_tolerance,
                                                   sizeof(double),
                                                   p_error));
                    break;
                }
                case rocsparse_spilu0_input_boost_enable:
                case rocsparse_spilu0_input_boost_tolerance:
                case rocsparse_spilu0_input_boost_value:
                {
                    break;
                }
                }
            }

            //
            // Call analysis twice.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spilu0(handle,
                                                     spilu0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spilu0_stage_analysis,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));
        }

        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_buffer_size(
                handle, spilu0_descr, A, P, rocsparse_spilu0_stage_compute, &buffer_size, p_error));
            device_dense_vector<char> buffer(buffer_size);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spilu0_set_input(handle,
                                           spilu0_descr,
                                           rocsparse_spilu0_input_singularity_tolerance,
                                           hsingularity_tolerance,
                                           sizeof(double),
                                           p_error));

            //
            // Call compute.
            //

            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   P,
                                                   rocsparse_spilu0_stage_compute,
                                                   buffer_size,
                                                   buffer,
                                                   p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spilu0(handle,
                                                     spilu0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spilu0_stage_analysis,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spilu0_set_input(handle,
                                           spilu0_descr,
                                           rocsparse_spilu0_input_singularity_tolerance,
                                           dsingularity_tolerance,
                                           sizeof(double),
                                           p_error));

            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   P,
                                                   rocsparse_spilu0_stage_compute,
                                                   buffer_size,
                                                   buffer,
                                                   p_error));

            //
            // Call analysis after compute.
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_status_invalid_value,
                                    rocsparse_spilu0(handle,
                                                     spilu0_descr,
                                                     A,
                                                     P,
                                                     rocsparse_spilu0_stage_analysis,
                                                     buffer_size,
                                                     buffer,
                                                     p_error));
            //
            // Call compute.
            //
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   P,
                                                   rocsparse_spilu0_stage_compute,
                                                   buffer_size,
                                                   buffer,
                                                   p_error));
        }
    }
}

template <typename T, typename I, typename J = I>
void rocsparse_clients_spilu0_host(rocsparse_clients::spilu0_descr&         spilu0_descr,
                                   rocsparse_clients::spmat_descr<T, I, J>& A,
                                   const double*             singular_pivot_tolerance,
                                   int32_t                   boost_enable,
                                   const floating_data_t<T>* boost_tolerance,
                                   const T*                  boost_value)
{

    auto& cpu_symbolic_pivot           = spilu0_descr.m_cpu_symbolic_singularity_position;
    auto& cpu_numeric_near_pivot       = spilu0_descr.m_cpu_numeric_near_singularity_position;
    auto& cpu_numeric_exact_pivot      = spilu0_descr.m_cpu_numeric_exact_singularity_position;
    const rocsparse_format format      = A.get_format();
    const int64_t          batch_count = A.get_batch_count();
    switch(format)
    {
    case rocsparse_format_csr:
    {
        auto& host = A.template as<rocsparse_format_csr>().host();
        for(int64_t i = 0; i < batch_count; ++i)
        {
            T* p = host.val.data() + i * A.get_stride();
            host_csrilu0<T, I, J>(host.m,
                                  host.ptr,
                                  host.ind,
                                  p,
                                  host.base,
                                  cpu_symbolic_pivot + i,
                                  cpu_numeric_exact_pivot + i,
                                  cpu_numeric_near_pivot + i,
                                  singular_pivot_tolerance,
                                  boost_enable,
                                  boost_tolerance,
                                  boost_value);
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

            host_bsrilu0<T, I, J>(host.block_direction,
                                  host.mb,
                                  host.ptr,
                                  host.ind,
                                  p,
                                  host.row_block_dim,
                                  host.base,
                                  cpu_symbolic_pivot + i,
                                  cpu_numeric_exact_pivot + i,
                                  boost_enable,
                                  boost_tolerance,
                                  boost_value);
            for(int64_t j = 0; j < batch_count; ++j)
            {
                cpu_numeric_near_pivot[j] = cpu_numeric_exact_pivot[j];
            }
        }
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
void testing_spilu0(const Arguments& arg_)
{
    Arguments arg(arg_);

    rocsparse_error* p_error = nullptr;
    //
    // Import matrix.
    //
    if(arg.M != arg.N)
    {
        return;
    }

    static constexpr const bool full_rank   = true;
    int64_t                     batch_count = arg.batch_count;
    if(batch_count == -1)
        batch_count = 1;

    rocsparse_local_handle handle(arg);

    hipStream_t stream;
    CHECK_ROCSPARSE_ERROR(rocsparse_get_stream(handle, &stream));

    const double                    h_singular_pivot_tolerance[1] = {0.001};
    const rocsparse_spilu0_alg      alg                           = rocsparse_spilu0_alg_default;
    const rocsparse_analysis_policy analysis_policy               = arg.apol;
    const rocsparse_datatype        compute_datatype              = get_datatype<T>();
    const int32_t                   boost_enable                  = arg.numericboost;
    const T                         h_boost_value[1]              = {arg.get_boostval<T>()};
    const floating_data_t<T> h_boost_tolerance[1] = {static_cast<floating_data_t<T>>(arg.boosttol)};

    rocsparse_clients::spmat_descr<T, I, J> A(arg, batch_count, full_rank);

    if(false == A.is_square())
    {
        return;
    }

    rocsparse_clients::spilu0_descr spilu0_descr(handle, batch_count);

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(
        handle, spilu0_descr, rocsparse_spilu0_input_alg, &alg, sizeof(alg), p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_analysis_policy,
                                                     &analysis_policy,
                                                     sizeof(analysis_policy),
                                                     p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_compute_datatype,
                                                     &compute_datatype,
                                                     sizeof(compute_datatype),
                                                     p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_boost_enable,
                                                     &boost_enable,
                                                     sizeof(compute_datatype),
                                                     p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    const T h_boost_val = arg.get_boostval<T>();
    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_boost_value,
                                                     &h_boost_val,
                                                     sizeof(&h_boost_val),
                                                     p_error));

    const double h_boost_tol = static_cast<double>(arg.boosttol);
    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_boost_tolerance,
                                                     &h_boost_tol,
                                                     sizeof(h_boost_tol),
                                                     p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_set_input(handle,
                                                     spilu0_descr,
                                                     rocsparse_spilu0_input_singularity_tolerance,
                                                     &h_singular_pivot_tolerance,
                                                     sizeof(double),
                                                     p_error));

    auto& device_symbolic_pivot = spilu0_descr.m_device_symbolic_singularity_position;
    auto& host_symbolic_pivot   = spilu0_descr.m_host_symbolic_singularity_position;

    {
        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_buffer_size(handle,
                                                           spilu0_descr,
                                                           A,
                                                           A,
                                                           rocsparse_spilu0_stage_analysis,
                                                           &buffer_size_in_bytes,
                                                           p_error));
        device_dense_vector<char> buffer(buffer_size_in_bytes);
        CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));

        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                               spilu0_descr,
                                               A,
                                               A,
                                               rocsparse_spilu0_stage_analysis,
                                               buffer_size_in_bytes,
                                               buffer,
                                               p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spilu0_get_output(handle,
                                        spilu0_descr,
                                        rocsparse_spilu0_output_singularity_position,
                                        host_symbolic_pivot,
                                        sizeof(int64_t),
                                        p_error));

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    auto& device_numeric_pivot = spilu0_descr.m_device_numeric_singularity_position;
    auto& host_numeric_pivot   = spilu0_descr.m_host_numeric_singularity_position;

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        CHECK_ROCSPARSE_ERROR(
            rocsparse_spilu0_get_output(handle,
                                        spilu0_descr,
                                        rocsparse_spilu0_output_singularity_position,
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
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_buffer_size(handle,
                                                               spilu0_descr,
                                                               A,
                                                               A,
                                                               rocsparse_spilu0_stage_compute,
                                                               &buffer_size_in_bytes,
                                                               p_error));

            device_dense_vector<char> buffer(buffer_size_in_bytes);
            CHECK_HIP_ERROR(hipMemset(buffer, 255 - 1, buffer_size_in_bytes));
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   A,
                                                   rocsparse_spilu0_stage_compute,
                                                   buffer_size_in_bytes,
                                                   buffer,
                                                   p_error));

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spilu0_get_output(handle,
                                            spilu0_descr,
                                            rocsparse_spilu0_output_singularity_position,
                                            host_numeric_pivot,
                                            sizeof(int64_t),
                                            p_error));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_spilu0_get_output(handle,
                                            spilu0_descr,
                                            rocsparse_spilu0_output_singularity_position,
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

            rocsparse_clients_spilu0_host<T, I, J>(spilu0_descr,
                                                   A,
                                                   h_singular_pivot_tolerance,
                                                   boost_enable,
                                                   h_boost_tolerance,
                                                   h_boost_value);

            auto& cpu_symbolic_pivot     = spilu0_descr.m_cpu_symbolic_singularity_position;
            auto& cpu_numeric_near_pivot = spilu0_descr.m_cpu_numeric_near_singularity_position;

            cpu_symbolic_pivot.unit_check(host_symbolic_pivot);
            cpu_numeric_near_pivot.unit_check(host_numeric_pivot);
            A.near_check_values(host_symbolic_pivot, host_numeric_pivot);
        }
    }

    if(arg.timing)
    {

        size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();
        CHECK_ROCSPARSE_ERROR(rocsparse_spilu0_buffer_size(handle,
                                                           spilu0_descr,
                                                           A,
                                                           A,
                                                           rocsparse_spilu0_stage_compute,
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
            CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                   spilu0_descr,
                                                   A,
                                                   A,
                                                   rocsparse_spilu0_stage_compute,
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

                CHECK_ROCSPARSE_ERROR(rocsparse_spilu0(handle,
                                                       spilu0_descr,
                                                       A,
                                                       A,
                                                       rocsparse_spilu0_stage_compute,
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
            auto& device = A.template as<rocsparse_format_csr>().device();

            double       gbyte_count = csrilu0_gbyte_count<T>(device.m, device.nnz);
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
                = bsrilu0_gbyte_count<T>(device.mb, device.row_block_dim, device.nnzb);
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

void testing_spilu0_extra(const Arguments& arg) {}

#define INSTANTIATE(I, J, T)                                             \
    template void testing_spilu0_bad_arg<I, J, T>(const Arguments& arg); \
    template void testing_spilu0<I, J, T>(const Arguments& arg)

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
