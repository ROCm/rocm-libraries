/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_trm_info.hpp"
#include <memory>

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status trm_analysis(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  J                         m,
                                  I                         nnz,
                                  const rocsparse_mat_descr descr,
                                  const T*                  csr_val,
                                  const I*                  csr_row_ptr,
                                  const J*                  csr_col_ind,
                                  rocsparse::trm_info_t*    info,
                                  rocsparse::pivot_info_t*  pivot_info,
                                  void*                     temp_buffer);

    rocsparse_status gtrm_analysis(rocsparse_handle          handle,
                                   rocsparse_operation       trans,
                                   int64_t                   m,
                                   int64_t                   nnz,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_datatype        csr_val_datatype,
                                   const void*               csr_val,
                                   rocsparse_indextype       csr_row_ptr_indextype,
                                   const void*               csr_row_ptr,
                                   rocsparse_indextype       csr_col_ind_indextype,
                                   const void*               csr_col_ind,
                                   rocsparse::trm_info_t*    info,
                                   rocsparse::pivot_info_t*  pivot_info,
                                   void*                     temp_buffer);
}

namespace rocsparse
{

    struct trm_data_t : rocsparse::pivot_info_t
    {
    protected:
        rocsparse::trm_info_t* m_data[4]{};
        static int storage_index(rocsparse_operation operation, rocsparse_fill_mode fill_mode)
        {
            return ((operation != rocsparse_operation_none) ? 1 : 0) * 2 + fill_mode;
        }

    public:
        trm_data_t() = default;
        ~trm_data_t();
        void uncouple(const trm_data_t* that);
        void copy(const trm_data_t* that, hipStream_t stream);

        rocsparse::trm_info_t* get(rocsparse_operation operation, rocsparse_fill_mode fill_mode);

        void set(rocsparse_operation    operation,
                 rocsparse_fill_mode    fill_mode,
                 rocsparse::trm_info_t* trm_info);

        rocsparse_indextype get_indextype_J() const;

        rocsparse::trm_info_t* create(rocsparse_handle          handle,
                                      rocsparse_operation       trans,
                                      int64_t                   batch_count,
                                      int64_t                   m,
                                      int64_t                   nnz,
                                      const rocsparse_mat_descr descr,
                                      rocsparse_datatype        csr_val_datatype,
                                      const void*               csr_val,
                                      int64_t                   csr_val_stride,
                                      rocsparse_indextype       csr_row_ptr_indextype,
                                      const void*               csr_row_ptr,
                                      int64_t                   csr_row_ptr_stride,
                                      rocsparse_indextype       csr_col_ind_indextype,
                                      const void*               csr_col_ind,
                                      int64_t                   csr_col_ind_stride,
                                      void*                     temp_buffer)
        {
            rocsparse::trm_info_t* trm_info = new rocsparse::trm_info_t();
            // Perform analysis

            THROW_IF_ROCSPARSE_ERROR(rocsparse::gtrm_analysis(handle,
                                                              trans,
                                                              m,
                                                              nnz,
                                                              descr,
                                                              csr_val_datatype,
                                                              csr_val,
                                                              csr_row_ptr_indextype,
                                                              csr_row_ptr,
                                                              csr_col_ind_indextype,
                                                              csr_col_ind,
                                                              trm_info,
                                                              this,
                                                              temp_buffer));
            return trm_info;
        }

        rocsparse::trm_info_t* create(rocsparse_handle          handle,
                                      rocsparse_operation       trans,
                                      int64_t                   m,
                                      int64_t                   nnz,
                                      const rocsparse_mat_descr descr,
                                      rocsparse_datatype        csr_val_datatype,
                                      const void*               csr_val,
                                      rocsparse_indextype       csr_row_ptr_indextype,
                                      const void*               csr_row_ptr,
                                      rocsparse_indextype       csr_col_ind_indextype,
                                      const void*               csr_col_ind,
                                      void*                     temp_buffer)
        {
            rocsparse::trm_info_t* trm_info = new rocsparse::trm_info_t();
            // Perform analysis

            THROW_IF_ROCSPARSE_ERROR(rocsparse::gtrm_analysis(handle,
                                                              trans,
                                                              m,
                                                              nnz,
                                                              descr,
                                                              csr_val_datatype,
                                                              csr_val,
                                                              csr_row_ptr_indextype,
                                                              csr_row_ptr,
                                                              csr_col_ind_indextype,
                                                              csr_col_ind,
                                                              trm_info,
                                                              this,
                                                              temp_buffer));
            return trm_info;
        }

        template <typename I, typename J, typename T>
        rocsparse::trm_info_t* create(rocsparse_handle          handle,
                                      rocsparse_operation       trans,
                                      J                         m,
                                      I                         nnz,
                                      const rocsparse_mat_descr descr,
                                      const T*                  csr_val,
                                      const I*                  csr_row_ptr,
                                      const J*                  csr_col_ind,
                                      void*                     temp_buffer)
        {
            return create(handle,
                          trans,
                          m,
                          nnz,
                          descr,
                          rocsparse::get_datatype<T>(),
                          csr_val,
                          rocsparse::get_indextype<I>(),
                          csr_row_ptr,
                          rocsparse::get_indextype<J>(),
                          csr_col_ind,
                          temp_buffer);
        }

        template <typename... ARGS>
        rocsparse_status
            recreate(rocsparse_operation operation, rocsparse_fill_mode fill_mode, ARGS... arg)
        {
            const int index = this->storage_index(operation, fill_mode);
            if(this->m_data[index] != nullptr)
            {
                delete this->m_data[index];
            }
            this->m_data[index] = nullptr;
            this->m_data[index] = this->create(arg...);
            return rocsparse_status_success;
        }

        template <typename I, typename J, typename T>
        rocsparse_status recreate(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  J                         m,
                                  I                         nnz,
                                  const rocsparse_mat_descr descr,
                                  const T*                  csr_val,
                                  const I*                  csr_row_ptr,
                                  const J*                  csr_col_ind,
                                  void*                     temp_buffer)
        {
            return this->recreate(trans,
                                  descr->fill_mode,
                                  handle,
                                  trans,
                                  m,
                                  nnz,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  temp_buffer);
        }

        rocsparse_status recreate(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  int64_t                   m,
                                  int64_t                   nnz,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_datatype        csr_val_datatype,
                                  const void*               csr_val,
                                  rocsparse_indextype       csr_row_ptr_indextype,
                                  const void*               csr_row_ptr,
                                  rocsparse_indextype       csr_col_ind_indextype,
                                  const void*               csr_col_ind,
                                  void*                     temp_buffer)
        {
            return this->recreate(trans,
                                  descr->fill_mode,
                                  handle,
                                  trans,
                                  m,
                                  nnz,
                                  descr,
                                  csr_val_datatype,
                                  csr_val,
                                  csr_row_ptr_indextype,
                                  csr_row_ptr,
                                  csr_col_ind_indextype,
                                  csr_col_ind,
                                  temp_buffer);
        }
    };

}
