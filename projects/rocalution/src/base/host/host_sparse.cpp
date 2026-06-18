/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "host_sparse.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include "rocalution/utils/types.hpp"

#include <algorithm>
#include <limits>

namespace rocalution
{
    template <typename I, typename J, typename T>
    bool host_csritsv_buffer_size(host_sparse_operation   trans,
                                  J                       m,
                                  I                       nnz,
                                  host_sparse_fill_mode   fill_mode,
                                  host_sparse_diag_type   diag_type,
                                  host_sparse_matrix_type mat_type,
                                  const T*                csr_val,
                                  const I*                csr_row_ptr,
                                  const J*                csr_col_ind,
                                  size_t*                 buffer_size)
    {
        buffer_size[0] = ((sizeof(I) * m - 1) / 256 + 1) * 256 + sizeof(T) * m;
        if(diag_type == host_sparse_diag_type_non_unit)
        {
            buffer_size[0] += sizeof(T) * m;
        }
        return true;
    }

    template <typename I, typename J>
    static bool host_csritsv_ptr_end(host_sparse_fill_mode fill_mode_,
                                     host_sparse_diag_type diag_type_,
                                     J                     m_,
                                     I                     nnz_,
                                     const I* __restrict__ ptr_,
                                     I* __restrict__ ptr_end_,
                                     const J* __restrict__ ind_,
                                     J* zero_pivot)
    {
        zero_pivot[0] = -1;
        switch(fill_mode_)
        {
        case host_sparse_fill_mode_lower:
        {
            switch(diag_type_)
            {
            case host_sparse_diag_type_unit:
            {
                for(J i = 0; i < m_; ++i)
                {
                    ptr_end_[i] = ptr_[i + 1];
                    for(I k = ptr_[i]; k < ptr_[i + 1]; ++k)
                    {
                        const J j = ind_[k];
                        if(j >= i)
                        {
                            ptr_end_[i] = k;
                            break;
                        }
                    }
                }
                break;
            }
            case host_sparse_diag_type_non_unit:
            {
                zero_pivot[0]         = std::numeric_limits<J>::max();
                J count_symbolic_diag = 0;
                for(J i = 0; i < m_; ++i)
                {

                    ptr_end_[i] = ptr_[i + 1];
                    bool mark   = false;
                    for(I k = ptr_[i]; k < ptr_[i + 1]; ++k)
                    {
                        const J j = ind_[k];
                        if(j == i)
                        {
                            mark        = true;
                            ptr_end_[i] = k + 1;
                            break;
                        }
                    }

                    if(!mark)
                    {
                        zero_pivot[0] = std::min(zero_pivot[0], i);
                        ++count_symbolic_diag;
                    }
                }
                if(zero_pivot[0] == std::numeric_limits<J>::max())
                {
                    zero_pivot[0] = -1;
                }

                if(count_symbolic_diag > 0)
                {
                    return true;
                }
                break;
            }
            }
            break;
        }
        case host_sparse_fill_mode_upper:
        {
            switch(diag_type_)
            {
            case host_sparse_diag_type_unit:
            {
                for(J i = 0; i < m_; ++i)
                {
                    ptr_end_[i] = ptr_[i + 1];
                    for(I k = ptr_[i]; k < ptr_[i + 1]; ++k)
                    {
                        const J j = ind_[k];
                        if(j > i)
                        {
                            ptr_end_[i] = k;
                            break;
                        }
                    }
                }
                break;
            }
            case host_sparse_diag_type_non_unit:
            {
                zero_pivot[0]         = std::numeric_limits<J>::max();
                J count_symbolic_diag = 0;
                for(J i = 0; i < m_; ++i)
                {
                    bool mark   = false;
                    ptr_end_[i] = ptr_[i + 1];
                    for(I k = ptr_[i]; k < ptr_[i + 1]; ++k)
                    {
                        const J j = ind_[k];
                        if(j == i)
                        {
                            ptr_end_[i] = k;
                            mark        = true;
                            break;
                        }
                    }
                    if(!mark)
                    {
                        zero_pivot[0] = std::min(zero_pivot[0], i);
                        ;
                        ++count_symbolic_diag;
                    }
                }
                if(zero_pivot[0] == std::numeric_limits<J>::max())
                {
                    zero_pivot[0] = -1;
                }

                if(count_symbolic_diag > 0)
                {
                    return true;
                }
                break;
            }
            }
            break;
        }
        }
        return true;
    }

    template <typename I, typename J, typename T>
    bool host_csritsv_solve(int*                       host_nmaxiter,
                            const numeric_traits_t<T>* host_tol,
                            numeric_traits_t<T>*       host_history,
                            host_sparse_operation      trans,
                            J                          m,
                            I                          nnz,
                            const T*                   alpha,
                            host_sparse_fill_mode      fill_mode,
                            host_sparse_diag_type      diag_type,
                            host_sparse_matrix_type    mat_type,
                            const T*                   csr_val,
                            const I*                   csr_row_ptr,
                            const J*                   csr_col_ind,
                            const T*                   x,
                            T*                         y,
                            void*                      temp_buffer,
                            J*                         zero_pivot)
    {
        zero_pivot[0]                 = -1;
        static constexpr bool verbose = false;
        if(verbose)
        {
            std::cout << "diag_type_" << diag_type << std::endl;
            std::cout << "fill_mode_" << fill_mode << std::endl;
        }

        if(m == 0 || nnz == 0)
        {
            if(nnz == 0 && diag_type == host_sparse_diag_type_unit)
            {
                //
                // copy and scal.
                //
                for(J i = 0; i < m; ++i)
                    y[i] = alpha[0] * x[i];
                host_nmaxiter[0] = 1;
            }
            return true;
        }

        const I* ptr_end  = nullptr;
        T*       y_p      = nullptr;
        T*       inv_diag = nullptr;

        if(mat_type == host_sparse_matrix_type_general)
        {
            ptr_end = (I*)temp_buffer;

            bool status = host_csritsv_ptr_end(fill_mode,
                                               diag_type,
                                               m,
                                               nnz,
                                               csr_row_ptr,
                                               (I*)temp_buffer,
                                               csr_col_ind,
                                               zero_pivot);

            if(!status)
            {
                return status;
            }

            temp_buffer = (void*)(((char*)temp_buffer) + ((sizeof(I) * m - 1) / 256 + 1) * 256);
            y_p         = (T*)temp_buffer;
            temp_buffer = (void*)(y_p + m);
        }
        else if(mat_type == host_sparse_matrix_type_triangular)
        {
            y_p         = (T*)temp_buffer;
            temp_buffer = (void*)(y_p + m);
            if(fill_mode == host_sparse_fill_mode_lower)
            {
                ptr_end = csr_row_ptr + 1;
            }
            else
            {
                ptr_end = csr_row_ptr;
            }

            switch(diag_type)
            {
            case host_sparse_diag_type_non_unit:
            {
                if(fill_mode == host_sparse_fill_mode_lower)
                {
                    for(J i = 0; i < m; ++i)
                    {
                        const J j = csr_col_ind[csr_row_ptr[i + 1] - 1];
                        if(i != j)
                        {
                            zero_pivot[0] = i;
                            break;
                        }
                    }
                }
                else
                {
                    for(J i = 0; i < m; ++i)
                    {
                        const J j = csr_col_ind[csr_row_ptr[i]];
                        if(i != j)
                        {
                            zero_pivot[0] = i;
                            break;
                        }
                    }
                }
                break;
            }
            case host_sparse_diag_type_unit:
            {
                break;
            }
            }
        }

        if(zero_pivot[0] != -1)
        {
            return true;
        }

        const I* b = nullptr;
        const I* e = nullptr;
        const I* d = nullptr;
        switch(fill_mode)
        {
        case host_sparse_fill_mode_lower:
        {
            b = csr_row_ptr;
            e = ptr_end;
            break;
        }

        case host_sparse_fill_mode_upper:
        {
            b = ptr_end;
            e = csr_row_ptr + 1;
            break;
        }
        }

        switch(diag_type)
        {
        case host_sparse_diag_type_non_unit:
        {
            d = ptr_end;
            break;
        }
        case host_sparse_diag_type_unit:
        {
            break;
        }
        }

        switch(diag_type)
        {
        case host_sparse_diag_type_unit:
        {
            break;
        }
        case host_sparse_diag_type_non_unit:
        {
            inv_diag    = (T*)temp_buffer;
            temp_buffer = (void*)(inv_diag + m);

            for(J i = 0; i < m; ++i)
            {
                I k = (fill_mode == host_sparse_fill_mode_upper) ? (d[i]) : (d[i] - 1);
                if(csr_val[k] == static_cast<T>(0))
                {
                    zero_pivot[0] = i;
                    return true;
                }
                if(trans == host_sparse_operation_conjugate_transpose)
                {
                    inv_diag[i] = static_cast<T>(1) / rocalution_conj(csr_val[k]);
                }
                else
                {
                    inv_diag[i] = static_cast<T>(1) / csr_val[k];
                }
            }
            break;
        }
        }

        const numeric_traits_t<T> nrm0 = static_cast<numeric_traits_t<T>>(1);
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

            numeric_traits_t<T> mx_residual = static_cast<numeric_traits_t<T>>(0);
            numeric_traits_t<T> mx          = static_cast<numeric_traits_t<T>>(0);
            //
            // Compute y = alpha
            //
            switch(trans)
            {
            case host_sparse_operation_none:
            {
                switch(diag_type)
                {
                case host_sparse_diag_type_non_unit:
                {

                    for(J i = 0; i < m; ++i)
                    {

                        T sum = static_cast<T>(0);

                        if((e[i] > b[i] + 1))
                        {

                            for(I k = b[i]; k < e[i]; ++k)
                            {
                                sum += csr_val[k] * y_p[csr_col_ind[k]];
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
                            mx_residual = std::max(
                                mx_residual, std::abs(alpha[0] * x[i] - y_p[i] / inv_diag[i]));
                        }
                    }
                    break;
                }
                case host_sparse_diag_type_unit:
                {
                    for(J i = 0; i < m; ++i)
                    {
                        T sum = static_cast<T>(0);
                        for(I k = b[i]; k < e[i]; ++k)
                            sum += csr_val[k] * y_p[csr_col_ind[k]];
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
            case host_sparse_operation_transpose:
            case host_sparse_operation_conjugate_transpose:
            {
                for(J i = 0; i < m; ++i)
                {
                    y[i] = static_cast<T>(0);
                }
                for(J i = 0; i < m; ++i)
                {
                    // row i, column csr_col_ind[k]
                    // row csr_col_ind[k]
                    for(I k = b[i]; k < e[i]; ++k)
                    {
                        const J j = csr_col_ind[k];
                        const T a = (trans == host_sparse_operation_conjugate_transpose)
                                        ? rocalution_conj(csr_val[k])
                                        : csr_val[k];
                        y[j] += a * y_p[i];
                    }
                }
                switch(diag_type)
                {
                case host_sparse_diag_type_non_unit:
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
                case host_sparse_diag_type_unit:
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

        return true;
    }

#define INSTANTIATE_T(TTYPE)                                                                         \
    template bool host_csritsv_buffer_size<PtrType, int, TTYPE>(host_sparse_operation   trans,       \
                                                                int                     m,           \
                                                                PtrType                 nnz,         \
                                                                host_sparse_fill_mode   fill_mode,   \
                                                                host_sparse_diag_type   diag_type,   \
                                                                host_sparse_matrix_type mat_type,    \
                                                                const TTYPE*            csr_val,     \
                                                                const PtrType*          csr_row_ptr, \
                                                                const int*              csr_col_ind, \
                                                                size_t* buffer_size);                \
                                                                                                     \
    template bool host_csritsv_solve<PtrType, int, TTYPE>(int* host_nmaxiter,                        \
                                                          const numeric_traits_t<TTYPE>* host_tol,   \
                                                          numeric_traits_t<TTYPE>* host_history,     \
                                                          host_sparse_operation    trans,            \
                                                          int                      m,                \
                                                          PtrType                  nnz,              \
                                                          const TTYPE*             alpha,            \
                                                          host_sparse_fill_mode    fill_mode,        \
                                                          host_sparse_diag_type    diag_type,        \
                                                          host_sparse_matrix_type  mat_type,         \
                                                          const TTYPE*             csr_val,          \
                                                          const PtrType*           csr_row_ptr,      \
                                                          const int*               csr_col_ind,      \
                                                          const TTYPE*             x,                \
                                                          TTYPE*                   y,                \
                                                          void*                    temp_buffer,      \
                                                          int*                     zero_pivot)

    INSTANTIATE_T(float);
    INSTANTIATE_T(double);
#ifdef SUPPORT_COMPLEX
    INSTANTIATE_T(std::complex<float>);
    INSTANTIATE_T(std::complex<double>);
#endif
#undef INSTANTIATE_T
}
