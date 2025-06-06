/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
#include "rocsparse_csxsldu.hpp"
#include "rocsparse_utility.hpp"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csxsldu_buffer_size_template(rocsparse_handle     handle_,
                                                         rocsparse_direction  dir_,
                                                         J                    m_,
                                                         J                    n_,
                                                         I                    nnz_,
                                                         const I*             ptr_,
                                                         const J*             ind_,
                                                         const T*             val_,
                                                         rocsparse_index_base base_,
                                                         rocsparse_diag_type  ldiag_,
                                                         rocsparse_direction  ldir_,
                                                         rocsparse_diag_type  udiag_,
                                                         rocsparse_direction  udir_,
                                                         size_t*              buffer_size_)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(ldir_ != dir_ && udir_ != dir_)
    {
        //
        // L and U are of the other direction from A.
        //
        buffer_size_[0] = ((((dir_ == rocsparse_direction_row) ? m_ : n_) + 1) * 2) * sizeof(I);
    }
    else if(ldir_ == dir_ && udir_ == dir_)
    {
        //
        // L and U are of the same direction from A.
        //
        buffer_size_[0] = 0;
    }
    else
    {
        buffer_size_[0] = (((dir_ == rocsparse_direction_row) ? m_ : n_) + 1) * sizeof(I);
    }
    return rocsparse_status_success;
}

#define INSTANTIATE(T)                                                    \
    template rocsparse_status rocsparse::csxsldu_buffer_size_template<T>( \
        rocsparse_handle     handle_,                                     \
        rocsparse_direction  dir_,                                        \
        rocsparse_int        m_,                                          \
        rocsparse_int        n_,                                          \
        rocsparse_int        nnz_,                                        \
        const rocsparse_int* ptr_,                                        \
        const rocsparse_int* ind_,                                        \
        const T*             val_,                                        \
        rocsparse_index_base base_,                                       \
        rocsparse_diag_type  ldiag_,                                      \
        rocsparse_direction  ldir_,                                       \
        rocsparse_diag_type  udiag_,                                      \
        rocsparse_direction  udir_,                                       \
        size_t*              buffer_size_)
INSTANTIATE(rocsparse_int);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#define C_IMPL(NAME, TYPE)                                                                \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle_,                        \
                                     rocsparse_direction  dir_,                           \
                                     rocsparse_int        m_,                             \
                                     rocsparse_int        n_,                             \
                                     rocsparse_int        nnz_,                           \
                                     const rocsparse_int* ptr_,                           \
                                     const rocsparse_int* ind_,                           \
                                     const TYPE*          val_,                           \
                                     rocsparse_index_base base_,                          \
                                     rocsparse_diag_type  ldiag_,                         \
                                     rocsparse_direction  ldir_,                          \
                                     rocsparse_diag_type  udiag_,                         \
                                     rocsparse_direction  udir_,                          \
                                     size_t*              buffer_size_)                   \
    try                                                                                   \
    {                                                                                     \
        ROCSPARSE_ROUTINE_TRACE;                                                          \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csxsldu_buffer_size_template(handle_,        \
                                                                          dir_,           \
                                                                          m_,             \
                                                                          n_,             \
                                                                          nnz_,           \
                                                                          ptr_,           \
                                                                          ind_,           \
                                                                          val_,           \
                                                                          base_,          \
                                                                          ldiag_,         \
                                                                          ldir_,          \
                                                                          udiag_,         \
                                                                          udir_,          \
                                                                          buffer_size_)); \
        return rocsparse_status_success;                                                  \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        RETURN_ROCSPARSE_EXCEPTION();                                                     \
    }

C_IMPL(rocsparse_scsxsldu_buffer_size, float);
C_IMPL(rocsparse_dcsxsldu_buffer_size, double);
C_IMPL(rocsparse_ccsxsldu_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsxsldu_buffer_size, rocsparse_double_complex);

#undef C_IMPL
