/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_enum.hpp"
#include "testing.hpp"

//
// Split such that L is unit and U is non-unit.
//
template <typename I, typename J>
rocsparse_status rocsparse_host_csritsv_ptr_end(rocsparse_fill_mode fill_mode_,
                                                rocsparse_diag_type diag_type_,
                                                J                   m_,
                                                I                   nnz_,
                                                const I* __restrict__ ptr_,
                                                I* __restrict__ ptr_end_,
                                                const J* __restrict__ ind_,
                                                rocsparse_index_base base_,
                                                rocsparse_int*       zero_pivot)
{
    zero_pivot[0] = -1;
    switch(fill_mode_)
    {
    case rocsparse_fill_mode_lower:
    {
        switch(diag_type_)
        {
        case rocsparse_diag_type_unit:
        {
            for(J i = 0; i < m_; ++i)
            {
                ptr_end_[i] = ptr_[i + 1];
                for(I k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
                {
                    const J j = ind_[k] - base_;
                    if(j >= i)
                    {
                        ptr_end_[i] = k + base_;
                        break;
                    }
                }
            }
            break;
        }
        case rocsparse_diag_type_non_unit:
        {
            zero_pivot[0]         = std::numeric_limits<rocsparse_int>::max();
            J count_symbolic_diag = 0;
            for(J i = 0; i < m_; ++i)
            {

                ptr_end_[i] = ptr_[i + 1];
                bool mark   = false;
                for(I k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
                {
                    const J j = ind_[k] - base_;
                    if(j == i)
                    {
                        mark        = true;
                        ptr_end_[i] = k + 1 + base_;
                        break;
                    }
                }

                if(!mark)
                {
                    zero_pivot[0] = std::min(zero_pivot[0], i + base_);
                    ++count_symbolic_diag;
                }
            }
            if(zero_pivot[0] == std::numeric_limits<rocsparse_int>::max())
            {
                zero_pivot[0] = -1;
            }

            if(count_symbolic_diag > 0)
            {
                return rocsparse_status_success;
            }
            break;
        }
        }
        break;
    }
    case rocsparse_fill_mode_upper:
    {
        switch(diag_type_)
        {
        case rocsparse_diag_type_unit:
        {
            for(J i = 0; i < m_; ++i)
            {
                ptr_end_[i] = ptr_[i + 1];
                for(I k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
                {
                    const J j = ind_[k] - base_;
                    if(j > i)
                    {
                        ptr_end_[i] = k + base_;
                        break;
                    }
                }
            }
            break;
        }
        case rocsparse_diag_type_non_unit:
        {
            zero_pivot[0]         = std::numeric_limits<rocsparse_int>::max();
            J count_symbolic_diag = 0;
            for(J i = 0; i < m_; ++i)
            {
                bool mark   = false;
                ptr_end_[i] = ptr_[i + 1];
                for(I k = ptr_[i] - base_; k < ptr_[i + 1] - base_; ++k)
                {
                    const J j = ind_[k] - base_;
                    if(j == i)
                    {
                        ptr_end_[i] = k + base_;
                        mark        = true;
                        break;
                    }
                }
                if(!mark)
                {
                    zero_pivot[0] = std::min(zero_pivot[0], i + base_);
                    ;
                    ++count_symbolic_diag;
                }
            }
            if(zero_pivot[0] == std::numeric_limits<rocsparse_int>::max())
            {
                zero_pivot[0] = -1;
            }

            if(count_symbolic_diag > 0)
            {
                return rocsparse_status_success;
            }
            break;
        }
        }
        break;
    }
    }
    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_host_csritsv_buffer_size_template(rocsparse_handle          handle,
                                                             rocsparse_operation       trans,
                                                             J                         m,
                                                             I                         nnz,
                                                             const rocsparse_mat_descr descr,
                                                             const T*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             rocsparse_mat_info        info,
                                                             size_t*                   buffer_size)
{
    buffer_size[0]                      = ((sizeof(I) * m - 1) / 256 + 1) * 256 + sizeof(T) * m;
    const rocsparse_diag_type diag_type = rocsparse_get_mat_diag_type(descr);
    if(diag_type == rocsparse_diag_type_non_unit)
    {
        buffer_size[0] += sizeof(T) * m;
    }
    return rocsparse_status_success;
}

//
// Non optimal host implementation.
//
template <typename I, typename J, typename T>
rocsparse_status rocsparse_host_csritsv_solve(rocsparse_handle          handle,
                                              rocsparse_int*            host_nmaxiter,
                                              const floating_data_t<T>* host_tol,
                                              floating_data_t<T>*       host_history,
                                              rocsparse_operation       trans,
                                              J                         m,
                                              I                         nnz,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const T*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
                                              const T*                  x,
                                              T*                        y,
                                              void*                     temp_buffer,
                                              rocsparse_int*            zero_pivot)
{
    zero_pivot[0]                         = -1;
    static constexpr bool       verbose   = false;
    const rocsparse_fill_mode   fill_mode = rocsparse_get_mat_fill_mode(descr);
    const rocsparse_diag_type   diag_type = rocsparse_get_mat_diag_type(descr);
    const rocsparse_matrix_type mat_type  = rocsparse_get_mat_type(descr);
    if(verbose)
    {
        std::cout << "diag_type_" << diag_type << std::endl;
        std::cout << "fill_mode_" << fill_mode << std::endl;
    }

    if(m == 0 || nnz == 0)
    {
        if(nnz == 0 && diag_type == rocsparse_diag_type_unit)
        {
            //
            // copy and scal.
            //
            for(J i = 0; i < m; ++i)
                y[i] = alpha[0] * x[i];
            host_nmaxiter[0] = 1;
        }
        return rocsparse_status_success;
    }

    const I*             ptr_end  = nullptr;
    T*                   y_p      = nullptr;
    T*                   inv_diag = nullptr;
    rocsparse_index_base base     = rocsparse_get_mat_index_base(descr);

    if(mat_type == rocsparse_matrix_type_general)
    {
        ptr_end = (I*)temp_buffer;

        rocsparse_status status = rocsparse_host_csritsv_ptr_end(fill_mode,
                                                                 diag_type,
                                                                 m,
                                                                 nnz,
                                                                 csr_row_ptr,
                                                                 (I*)temp_buffer,
                                                                 csr_col_ind,
                                                                 base,
                                                                 zero_pivot);

        if(status != rocsparse_status_success)
        {
            return status;
        }

        temp_buffer = (void*)(((char*)temp_buffer) + ((sizeof(I) * m - 1) / 256 + 1) * 256);
        y_p         = (T*)temp_buffer;
        temp_buffer = (void*)(y_p + m);
    }
    else if(mat_type == rocsparse_matrix_type_triangular)
    {
        y_p         = (T*)temp_buffer;
        temp_buffer = (void*)(y_p + m);
        if(fill_mode == rocsparse_fill_mode_lower)
        {
            ptr_end = csr_row_ptr + 1;
        }
        else
        {
            ptr_end = csr_row_ptr;
        }

        switch(diag_type)
        {
        case rocsparse_diag_type_non_unit:
        {
            if(fill_mode == rocsparse_fill_mode_lower)
            {
                for(J i = 0; i < m; ++i)
                {
                    const J j = csr_col_ind[csr_row_ptr[i + 1] - base - 1] - base;
                    if(i != j)
                    {
                        zero_pivot[0] = i + base;
                        break;
                    }
                }
            }
            else
            {
                for(J i = 0; i < m; ++i)
                {
                    const J j = csr_col_ind[csr_row_ptr[i] - base] - base;
                    if(i != j)
                    {
                        zero_pivot[0] = i + base;
                        break;
                    }
                }
            }
            break;
        }
        case rocsparse_diag_type_unit:
        {
            break;
        }
        }
    }

    if(zero_pivot[0] != -1)
    {
        return rocsparse_status_success;
    }

    const I* b = nullptr;
    const I* e = nullptr;
    const I* d = nullptr;
    switch(fill_mode)
    {
    case rocsparse_fill_mode_lower:
    {
        b = csr_row_ptr;
        e = ptr_end;
        break;
    }

    case rocsparse_fill_mode_upper:
    {
        b = ptr_end;
        e = csr_row_ptr + 1;
        break;
    }
    }

    switch(diag_type)
    {
    case rocsparse_diag_type_non_unit:
    {
        d = ptr_end;
        break;
    }
    case rocsparse_diag_type_unit:
    {
        break;
    }
    }

    switch(diag_type)
    {
    case rocsparse_diag_type_unit:
    {
        break;
    }
    case rocsparse_diag_type_non_unit:
    {
        inv_diag    = (T*)temp_buffer;
        temp_buffer = (void*)(inv_diag + m);

        for(J i = 0; i < m; ++i)
        {
            I k = (fill_mode == rocsparse_fill_mode_upper) ? (d[i] - base) : (d[i] - base - 1);
            if(csr_val[k] == static_cast<T>(0))
            {
                zero_pivot[0] = i + base;
                return rocsparse_status_success;
            }
            if(trans == rocsparse_operation_conjugate_transpose)
            {
                inv_diag[i] = static_cast<T>(1) / rocsparse_conj(csr_val[k]);
            }
            else
            {
                inv_diag[i] = static_cast<T>(1) / csr_val[k];
            }
        }
        break;
    }
    }

    const floating_data_t<T> nrm0 = static_cast<floating_data_t<T>>(1);
    //
    // Iterative Loop.
    //
    for(J iter = 0; iter < host_nmaxiter[0]; ++iter)
    {
        //
        // Copy y to y_p.
        //
        for(J i = 0; i < m; ++i)
        {
            y_p[i] = y[i];
        }

        floating_data_t<T> mx_residual = static_cast<floating_data_t<T>>(0);
        floating_data_t<T> mx          = static_cast<floating_data_t<T>>(0);
        //
        // Compute y = alpha
        //
        switch(trans)
        {
        case rocsparse_operation_none:
        {
            switch(diag_type)
            {
            case rocsparse_diag_type_non_unit:
            {

                for(J i = 0; i < m; ++i)
                {

                    T sum = static_cast<T>(0);

                    if((e[i] > b[i] + 1))
                    {

                        for(I k = b[i] - base; k < e[i] - base; ++k)
                        {
                            sum += csr_val[k] * y_p[csr_col_ind[k] - base];
                        }
                        const T h   = inv_diag[i] * (alpha[0] * x[i] - sum);
                        mx          = std::max(mx, std::abs(h));
                        mx_residual = std::max(mx_residual, std::abs(alpha[0] * x[i] - sum));
                        y[i]        = y_p[i] + h;
                    }
                    else
                    {
                        y[i]        = inv_diag[i] * alpha[0] * x[i];
                        mx          = std::max(mx, std::abs(y[i] - y_p[i]));
                        mx_residual = std::max(mx_residual,
                                               std::abs(alpha[0] * x[i] - y_p[i] / inv_diag[i]));
                    }
                }
                break;
            }
            case rocsparse_diag_type_unit:
            {
                for(J i = 0; i < m; ++i)
                {
                    T sum = static_cast<T>(0);
                    for(I k = b[i] - base; k < e[i] - base; ++k)
                        sum += csr_val[k] * y_p[csr_col_ind[k] - base];
                    y[i]        = alpha[0] * x[i] - sum;
                    const T h   = y[i] - y_p[i];
                    mx          = std::max(mx, std::abs(h));
                    mx_residual = mx;
                }
                break;
            }
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            for(J i = 0; i < m; ++i)
            {
                y[i] = static_cast<T>(0);
            }
            for(J i = 0; i < m; ++i)
            {
                // row i, column csr_col_ind[k]
                // row csr_col_ind[k]
                for(I k = b[i] - base; k < e[i] - base; ++k)
                {
                    const J j = csr_col_ind[k] - base;
                    const T a = (trans == rocsparse_operation_conjugate_transpose)
                                    ? rocsparse_conj(csr_val[k])
                                    : csr_val[k];
                    y[j] += a * y_p[i];
                }
            }
            switch(diag_type)
            {
            case rocsparse_diag_type_non_unit:
            {
                for(J i = 0; i < m; ++i)
                {
                    mx_residual = std::max(mx, std::abs(alpha[0] * x[i] - y[i]));
                    const T h   = inv_diag[i] * (alpha[0] * x[i] - y[i]);
                    mx          = std::max(mx, std::abs(h));
                    y[i]        = h + y_p[i];
                }
                break;
            }
            case rocsparse_diag_type_unit:
            {
                for(J i = 0; i < m; ++i)
                {
                    y[i]        = (alpha[0] * x[i] - y[i]);
                    const T h   = y[i] - y_p[i];
                    mx          = std::max(mx, std::abs(h));
                    mx_residual = mx;
                }
                break;
            }
            }
            break;
        }
        }

        //
        // y_k+1 = yk + (alpha * x - (id + T) * yk )
        //
        if(verbose)
        {
            std::cout << "iter " << iter << ", mx " << mx / nrm0 << ", mx_residual "
                      << mx_residual / nrm0 << std::endl;
        }

        if(host_history)
        {
            host_history[iter] = mx;
        }

        if(host_tol && (mx_residual <= host_tol[0]))
        {
            host_nmaxiter[0] = iter + 1;
            break;
        }
    }

    return rocsparse_status_success;
}

template <typename T>
void testing_csritsv_bad_arg(const Arguments& arg)
{

    static constexpr bool verbose = false;
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    const T h_alpha = static_cast<T>(1);

    rocsparse_handle          handle            = local_handle;
    rocsparse_int*            host_nmaxiter     = (rocsparse_int*)0x4;
    rocsparse_int             host_nfreeiter    = 2;
    const floating_data_t<T>* host_tol          = (const floating_data_t<T>*)0x4;
    floating_data_t<T>*       host_history      = (floating_data_t<T>*)0x4;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             m                 = 32;
    rocsparse_int             nnz               = 32;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  csr_val           = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_mat_info        info              = local_info;
    const T*                  x                 = (const T*)0x4;
    T*                        y                 = (T*)0x4;
    rocsparse_solve_policy    solve             = rocsparse_solve_policy_auto;
    rocsparse_solve_policy    policy            = rocsparse_solve_policy_auto;
    rocsparse_analysis_policy analysis          = rocsparse_analysis_policy_force;
    void*                     temp_buffer       = (void*)0x4;
    size_t*                   buffer_size       = (size_t*)0x4;

#define PARAMS_BUFFER_SIZE \
    handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size

#define PARAMS_ANALYSIS                                                                     \
    handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, analysis, solve, \
        temp_buffer
#define PARAMS_SOLVE                                                                        \
    handle, host_nmaxiter, host_tol, host_history, trans, m, nnz, alpha_device_host, descr, \
        csr_val, csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer

#define PARAMSX_SOLVE                                                                    \
    handle, host_nmaxiter, host_nfreeiter, host_tol, host_history, trans, m, nnz,        \
        alpha_device_host, descr, csr_val, csr_row_ptr, csr_col_ind, info, x, y, policy, \
        temp_buffer

    //
    //
    //
    if(verbose)
    {
        std::cout << "bad_arg_analysis(rocsparse_csritsv_buffer_size<T>, PARAMS_BUFFER_SIZE)"
                  << std::endl;
    }
    bad_arg_analysis(rocsparse_csritsv_buffer_size<T>, PARAMS_BUFFER_SIZE);

    //
    //
    //
    if(verbose)
    {
        std::cout << "bad_arg_analysis(rocsparse_csritsv_analysis<T>, PARAMS_ANALYSIS)"
                  << std::endl;
    }
    bad_arg_analysis(rocsparse_csritsv_analysis<T>, PARAMS_ANALYSIS);

    //
    //
    //
    if(verbose)
    {
        std::cout << "bad_arg_analysis(rocsparse_csritsv_solve<T>, PARAMS_SOLVE)" << std::endl;
    }

    //
    //
    //
    {
        static constexpr int nargs_to_exclude                  = 2;
        static constexpr int args_to_exclude[nargs_to_exclude] = {2, 3};

        select_bad_arg_analysis(
            rocsparse_csritsv_solve<T>, nargs_to_exclude, args_to_exclude, PARAMS_SOLVE);
    }

    //
    //
    //
    {
        static constexpr int nargs_to_exclude                  = 2;
        static constexpr int args_to_exclude[nargs_to_exclude] = {3, 4};
        select_bad_arg_analysis(
            rocsparse_csritsv_solve_ex<T>, nargs_to_exclude, args_to_exclude, PARAMSX_SOLVE);
    }

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general
           && matrix_type != rocsparse_matrix_type_triangular)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve<T>(PARAMS_SOLVE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE),
                                    rocsparse_status_not_implemented);
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));

    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve<T>(PARAMS_SOLVE),
                            rocsparse_status_requires_sorted_storage);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE),
                            rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE
#undef PARAMSX_SOLVE

    // Test rocsparse_csritsv_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(nullptr, descr, info, &position),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, nullptr, &position),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csritsv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_clear(nullptr, descr, info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_clear(handle, nullptr, info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_clear(handle, descr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csritsv(const Arguments& arg)
{
    static const bool verbose = false;

    //
    // Set nmaxiter.
    //
    rocsparse_int host_nmaxiter[1] = {arg.nmaxiter};
    rocsparse_int host_nfreeiter   = arg.nfreeiter;

    //
    // Tolerance for the iterative method.
    //
    floating_data_t<T> tol_iterative = static_cast<floating_data_t<T>>(1.0e-6);
    if(std::is_same<floating_data_t<T>, double>{})
        tol_iterative = static_cast<floating_data_t<T>>(1.0e-14);
    floating_data_t<T> host_tol[1] = {tol_iterative};

    //
    // Tolerance for the comparison.
    //
    floating_data_t<T> tol_compare = static_cast<floating_data_t<T>>(1.0e-5);
    if(std::is_same<floating_data_t<T>, double>{})
        tol_compare = static_cast<floating_data_t<T>>(1.0e-10);

    floating_data_t<T> tol_compare_with_direct = static_cast<floating_data_t<T>>(1.0e-3);
    if(std::is_same<floating_data_t<T>, double>{})
        tol_compare_with_direct = static_cast<floating_data_t<T>>(1.0e-10);

    //
    // Convergence history.
    //
    host_vector<floating_data_t<T>> host_history(host_nmaxiter[0]);

    //
    // Grab parametrization.
    //
    rocsparse_int             M     = arg.M;
    rocsparse_int             N     = arg.N;
    rocsparse_operation       trans = arg.transA;
    rocsparse_diag_type       diag  = arg.diag;
    rocsparse_fill_mode       uplo  = arg.uplo;
    rocsparse_analysis_policy apol  = arg.apol;
    rocsparse_solve_policy    spol  = arg.spol;
    rocsparse_index_base      base  = arg.baseA;

    //
    // Create host alpha.
    //
    host_scalar<T> h_alpha(arg.get_alpha<T>());

    //
    // Create rocsparse handle
    //
    rocsparse_local_handle handle;

    //
    // Create matrix info.
    //
    rocsparse_local_mat_info info;

    //
    // Create matrix descriptor
    //
    rocsparse_local_mat_descr descr;

    //
    // Configure descriptor.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, arg.matrix_type));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    //
    // Define routine parameters.
    //
#define PARAMS_BUFFER_SIZE(A_) \
    handle, trans, A_.m, A_.nnz, descr, A_.val, A_.ptr, A_.ind, info, &buffer_size

#define PARAMS_ANALYSIS(A_) \
    handle, trans, A_.m, A_.nnz, descr, A_.val, A_.ptr, A_.ind, info, apol, spol, dbuffer

#define PARAMS_SOLVE(alpha_, A_, x_, y_)                                                       \
    handle, host_nmaxiter, host_tol, host_history, trans, A_.m, A_.nnz, alpha_, descr, A_.val, \
        A_.ptr, A_.ind, info, x_, y_, spol, dbuffer

#define PARAMSX_SOLVE(alpha_, A_, x_, y_)                                                       \
    handle, host_nmaxiter, host_nfreeiter, host_tol, host_history, trans, A_.m, A_.nnz, alpha_, \
        descr, A_.val, A_.ptr, A_.ind, info, x_, y_, spol, dbuffer

    rocsparse_int host_zero_pivot;

#define HOST_PARAMS_SOLVE(alpha_, A_, x_, y_)                                                  \
    handle, host_nmaxiter, host_tol, host_history, trans, A_.m, A_.nnz, alpha_, descr, A_.val, \
        A_.ptr, A_.ind, x_, y_, hbuffer, &host_zero_pivot

    //
    // Non-squared matrices are not supported
    //
    if(M != N)
    {
        if(verbose)
        {
            std::cerr
                << "// rocSPARSE.WARNING clients testing_csritsv, skipping non-squared matrices."
                << std::endl;
        }
        return;
    }

    if(M == 0)
    {
        size_t        buffer_size;
        rocsparse_int pivot;

        device_vector<T>     dx, dy, dbuffer;
        device_csr_matrix<T> dA;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE(h_alpha, dA, dx, dy)),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, &pivot),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_clear(handle, descr, info),
                                rocsparse_status_success);

        return;
    }

    //
    // Declare matrix A.
    //
    host_csr_matrix<T> hA;

    if(diag == rocsparse_diag_type_unit && rocsparse_matrix_type_triangular == arg.matrix_type)
    {
        host_csr_matrix<T>          hB;
        static constexpr bool       to_int    = false;
        static constexpr bool       full_rank = true;
        rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);
        matrix_factory.init_csr(hB, M, N);

        //
        // Let's remove the diagonal elements...
        //
        int ndiag = 0;
        for(int i = 0; i < hB.m; ++i)
        {
            for(int k = hB.ptr[i] - hB.base; k < hB.ptr[i + 1] - hB.base; ++k)
            {
                const int j = hB.ind[k] - hB.base;
                if(i == j)
                {
                    ++ndiag;
                }
            }
        }

        hA.define(hB.m, hB.n, hB.nnz - ndiag, hB.base);
        if(ndiag > 0)
        {
            hA.ptr[0] = hA.base;
            for(int i = 0; i < hB.m; ++i)
            {
                bool find = false;
                for(int k = hB.ptr[i] - hB.base; k < hB.ptr[i + 1] - hB.base; ++k)
                {
                    const int j = hB.ind[k] - hB.base;
                    if(i == j)
                    {
                        find = true;
                        break;
                    }
                }
                if(find)
                {
                    hA.ptr[i + 1] = hB.ptr[i + 1] - hB.ptr[i] - 1;
                }
                else
                {
                    hA.ptr[i + 1] = hB.ptr[i + 1] - hB.ptr[i];
                }
            }

            for(int i = 1; i <= hA.m; ++i)
                hA.ptr[i] += hA.ptr[i - 1];

            for(int i = 0; i < hB.m; ++i)
            {
                int s = 0;
                for(int k = hB.ptr[i] - hB.base; k < hB.ptr[i + 1] - hB.base; ++k)
                {
                    if((hB.ind[k] - hB.base) != i)
                    {
                        hA.ind[hA.ptr[i] - hA.base + s] = hB.ind[k] - hB.base + hA.base;
                        hA.val[hA.ptr[i] - hA.base + s] = hB.val[k];
                        ++s;
                    }
                }
            }
        }
        else
        {
            hA.transfer_from(hB);
        }
    }
    else
    {
        static constexpr bool       to_int    = false;
        static constexpr bool       full_rank = true;
        rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);
        matrix_factory.init_csr(hA, M, N);
    }

    //
    // Again, since we import or generate the matrix, Non-squared matrices are not supported.
    //
    if(M != N)
    {
        if(verbose)
        {
            std::cerr
                << "// rocSPARSE.WARNING clients testing_csritsv, skipping non-squared matrices."
                << std::endl;
        }
        return;
    }

    //
    // Let's normalize ... it helps on horrible matrices.
    //
    floating_data_t<T> mx = 0;
    for(rocsparse_int i = 0; i < hA.nnz; ++i)
        mx = std::max(mx, std::abs(hA.val[i]));
    if(mx > 0)
    {
        for(rocsparse_int i = 0; i < hA.nnz; ++i)
            hA.val[i] /= mx;
    }

    //
    // Declare and initialize host X.
    //
    host_dense_matrix<T> hx(M, 1);
    rocsparse_matrix_utils::init(hx);

    //
    // Define and transfer A and X from host to device.
    //
    device_csr_matrix<T>   dA(hA);
    device_dense_matrix<T> dx(hx);

    //
    // Define Y on device.
    //
    device_dense_matrix<T> dy(M, 1);

    host_scalar<rocsparse_int> h_analysis_pivot, h_solve_pivot;
    if(verbose)
    {
        std::cout << "M   : " << M << std::endl;
        std::cout << "N   : " << N << std::endl;
        std::cout << "NNZ : " << hA.nnz << std::endl;
    }
    //
    // Buffer for calculation on device.
    //
    void* dbuffer;
    {
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)));
        if(verbose)
        {
            std::cout << "csritsv_buffer_size " << buffer_size << std::endl;
        }
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    }

    if(arg.unit_check)
    {
        if(verbose)
        {
            std::cout << "testing_csritsv unit_check pointer mode host ... " << std::endl;
        }

        if(verbose)
        {
            std::cout << " - compute host iterative" << std::endl;
        }

        //
        // Buffer for calculation on host.
        //
        void* hbuffer;

        {
            size_t buffer_size;
            CHECK_ROCSPARSE_ERROR(
                (rocsparse_host_csritsv_buffer_size_template<rocsparse_int,
                                                             rocsparse_int,
                                                             T>)(PARAMS_BUFFER_SIZE(hA)));
            if(verbose)
            {
                std::cout << "host_csritsv_buffer_size " << buffer_size << std::endl;
            }
            hbuffer = malloc(buffer_size);
        }

        //
        // Calculate on host.
        //
        host_dense_matrix<T> hy_iterative(M, 1);
        for(rocsparse_int i = 0; i < M; ++i)
        {
            hy_iterative[i] = static_cast<T>(0);
        }

        //
        // Compute the iterative method on host.
        //
        CHECK_ROCSPARSE_ERROR(
            (rocsparse_host_csritsv_solve<rocsparse_int, rocsparse_int, T>)(HOST_PARAMS_SOLVE(
                h_alpha, hA, hx, hy_iterative)));
        const bool host_iterative_convergence = (host_nmaxiter[0] < arg.nmaxiter);
        if(false == host_iterative_convergence)
        {
            //
            // it didn't converge, for some reasons ...
            //
            if(verbose)
            {
                std::cerr << "host csritsv didn't converge. " << std::endl;
            }
        }
        //
        // Calculate the direct method on host.
        //
        if(verbose)
        {
            std::cout << " - compute host direct" << std::endl;
        }
        host_dense_matrix<T> hy_direct(M, 1);
        host_csrsv<rocsparse_int, rocsparse_int, T>(trans,
                                                    hA.m,
                                                    hA.nnz,
                                                    *h_alpha,
                                                    hA.ptr,
                                                    hA.ind,
                                                    hA.val,
                                                    hx,
                                                    (int64_t)1,
                                                    hy_direct,
                                                    diag,
                                                    uplo,
                                                    base,
                                                    h_analysis_pivot,
                                                    h_solve_pivot);

        if(verbose)
        {
            std::cout << "h_analysis_pivot " << *h_analysis_pivot << std::endl;
            std::cout << "h_solve_pivot " << *h_solve_pivot << std::endl;
        }

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            if(host_iterative_convergence)
            {
                if(verbose)
                {
                    std::cout << " - compare host iterative and direct solutions" << std::endl;
                }

                //
                // Compare direct solution and iterative solution calculated on host
                // In other words, make sure what we calculate with the iterative method (even on host) makes sense.
                //
                hy_direct.near_check(hy_iterative, tol_compare_with_direct);
            }
        }
        else
        {
            if(verbose)
            {
                std::cout << " - zero pivot detected with direct method" << std::endl;
            }
        }

        if(verbose)
        {
            std::cout << " - compute device iterative" << std::endl;
        }
        //
        // Now we calculate on gpu
        //
        host_scalar<rocsparse_int> analysis_no_pivot(-1);
        host_scalar<rocsparse_int> analysis_pivot;
        host_scalar<rocsparse_int> solve_pivot;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        if(verbose)
        {
            std::cout << " - compute device iterative, check if default zero pivot is -1"
                      << std::endl;
        }
        //
        // CHECK IF DEFAULT ZERO PIVOT IS -1
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, analysis_pivot),
                                rocsparse_status_success);
        analysis_no_pivot.unit_check(analysis_pivot);

        if(verbose)
        {
            std::cout << " - compute device iterative, calling solve before analysis must fail"
                      << std::endl;
        }
        //
        // Call before analysis
        //
        host_nmaxiter[0] = arg.nmaxiter;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                rocsparse_status_invalid_pointer);
        if(verbose)
        {
            std::cout << " - compute device iterative, call analysis" << std::endl;
        }

        // Call it twice, for code coverage.
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        if(verbose)
        {
            std::cout
                << " - compute device iterative, call analysis twice (second call for coverage)"
                << std::endl;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        {
            auto st = rocsparse_csritsv_zero_pivot(handle, descr, info, analysis_pivot);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (*analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                            : rocsparse_status_success);
        }

        if(verbose)
        {
            std::cout << " - compute device iterative, call solve" << std::endl;
        }

        CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * M));
        host_nmaxiter[0] = arg.nmaxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
        {
            auto st = rocsparse_csritsv_zero_pivot(handle, descr, info, solve_pivot);
            EXPECT_ROCSPARSE_STATUS(
                st, (*solve_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success);
        }

        const bool device_iterative_convergence = (host_nmaxiter[0] < arg.nmaxiter);

        h_analysis_pivot.unit_check(analysis_pivot);
        h_solve_pivot.unit_check(solve_pivot);

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            if(verbose)
            {
                std::cout
                    << " - compute device iterative, compare solutions between host and device."
                    << std::endl;
            }
            if(host_iterative_convergence)
            {
                hy_iterative.near_check(dy, tol_compare);
            }
        }

        host_nmaxiter[0] = arg.nmaxiter;
        CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * M));
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE(h_alpha, dA, dx, dy)));
        {
            auto st = rocsparse_csritsv_zero_pivot(handle, descr, info, solve_pivot);
            CHECK_HIP_ERROR(hipDeviceSynchronize());

            EXPECT_ROCSPARSE_STATUS(
                st, (*solve_pivot != -1) ? rocsparse_status_zero_pivot : rocsparse_status_success);
        }

        const bool device_iterative_convergence_ex = (host_nmaxiter[0] < arg.nmaxiter);
        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            if(verbose)
            {
                std::cout
                    << " - compute device iterative_ex, compare solutions between host and device."
                    << std::endl;
            }
            if(host_iterative_convergence)
            {
                hy_iterative.near_check(dy, tol_compare);
            }
        }

        if(device_iterative_convergence != host_iterative_convergence)
        {
            if(verbose)
            {
                if(trans == rocsparse_operation_none)
                {
                    std::cout << " - WARNING host and device iterative convergence differs on "
                                 "NonTranspose case ... it happens rarely but it does."
                              << std::endl;
                }
                else
                {
                    std::cout
                        << " - WARNING host and device iterative convergence differs, it happens "
                           "with transpose cases since csrmv uses atomics."
                        << std::endl;
                }
            }
        }

        if(device_iterative_convergence_ex != host_iterative_convergence)
        {
            if(verbose)
            {
                if(trans == rocsparse_operation_none)
                {
                    std::cout << " - WARNING host and device iterative_ex convergence differs on "
                                 "NonTranspose case ... it happens rarely but it does."
                              << std::endl;
                }
                else
                {
                    std::cout << " - WARNING host and device iterative_ex convergence differs, it "
                                 "happens "
                                 "with transpose cases since csrmv uses atomics."
                              << std::endl;
                }
            }
        }

        if(verbose)
        {
            std::cout << " - compute device iterative, check pivot." << std::endl;
            std::cout << "h_analysis_pivot " << *h_analysis_pivot << std::endl;
            std::cout << "h_solve_pivot " << *h_solve_pivot << std::endl;
        }

        //
        // RESET MAT INFO.
        //
        info.reset();
        if(verbose)
        {
            std::cout << "testing_csritsv unit_check pointer mode device  ... " << std::endl;
        }
        if(verbose)
        {
            std::cout << " - compute device iterative" << std::endl;
        }

        device_scalar<rocsparse_int> d_analysis_pivot;
        device_scalar<rocsparse_int> d_solve_pivot;
        device_scalar<T>             d_alpha(h_alpha);

        //
        // POINTER MODE DEVICE
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        //
        // CHECK IF DEFAULT ZERO PIVOT IS -1
        //
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, d_analysis_pivot),
                                rocsparse_status_success);
        analysis_no_pivot.unit_check(d_analysis_pivot);

        //
        // Call before analysis
        //
        host_nmaxiter[0] = arg.nmaxiter;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                rocsparse_status_invalid_pointer);

        //
        // Call it twice.
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, d_analysis_pivot),
                                (*h_analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * M));
        host_nmaxiter[0] = arg.nmaxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_solve<T>(PARAMS_SOLVE(d_alpha, dA, dx, dy)));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, d_solve_pivot),
                                (*h_solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                       : rocsparse_status_success);
        h_analysis_pivot.unit_check(d_analysis_pivot);
        h_solve_pivot.unit_check(d_solve_pivot);

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y", dy);
            }
            if(host_iterative_convergence)
            {
                hy_iterative.near_check(dy, tol_compare);
            }
        }
        else
        {
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save(
                    "analysis_pivot", d_analysis_pivot, "solve_pivot", d_solve_pivot);
            }
        }

        CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * M));
        host_nmaxiter[0] = arg.nmaxiter;
        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE(d_alpha, dA, dx, dy)));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csritsv_zero_pivot(handle, descr, info, d_solve_pivot),
                                (*h_solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                       : rocsparse_status_success);

        h_analysis_pivot.unit_check(d_analysis_pivot);
        h_solve_pivot.unit_check(d_solve_pivot);

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            if(host_iterative_convergence)
            {
                hy_iterative.near_check(dy, tol_compare);
            }
        }

        free(hbuffer);
        if(verbose)
        {
            std::cout << "testing_csritsv unit_check done. " << std::endl;
        }
    }

    if(arg.timing)
    {

        host_nfreeiter = arg.nfreeiter;

        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csritsv_zero_pivot(handle, descr, info, h_analysis_pivot));
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE(h_alpha, dA, dx, dy)));
            CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_zero_pivot(handle, descr, info, h_solve_pivot));
            CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_clear(handle, descr, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = 0;
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * M));
            host_nmaxiter[0]             = arg.nmaxiter;
            double gpu_solve_time_used_1 = get_time_us();

            CHECK_ROCSPARSE_ERROR(
                rocsparse_csritsv_solve_ex<T>(PARAMSX_SOLVE(h_alpha, dA, dx, dy)));
            gpu_solve_time_used_1 = (get_time_us() - gpu_solve_time_used_1);
            gpu_solve_time_used += gpu_solve_time_used_1;
        }
        gpu_solve_time_used /= number_hot_calls;
        double gflop_count = csrsv_gflop_count(M, dA.nnz, diag);
        double gbyte_count = csrsv_gbyte_count<T>(M, dA.nnz);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::nfreeiter,
                            host_nfreeiter,
                            display_key_t::nmaxiter,
                            host_nmaxiter[0],
                            display_key_t::pivot,
                            std::min(*h_analysis_pivot, *h_solve_pivot),
                            display_key_t::trans,
                            rocsparse_operation2string(trans),
                            display_key_t::diag_type,
                            rocsparse_diagtype2string(diag),
                            display_key_t::fill_mode,
                            rocsparse_fillmode2string(uplo),
                            display_key_t::analysis_policy,
                            rocsparse_analysis2string(apol),
                            display_key_t::solve_policy,
                            rocsparse_solve2string(spol),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::analysis_time_ms,
                            get_gpu_time_msec(gpu_analysis_time_used),
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Clear csritsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csritsv_clear(handle, descr, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csritsv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csritsv<TYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csritsv_extra(const Arguments& arg) {}
