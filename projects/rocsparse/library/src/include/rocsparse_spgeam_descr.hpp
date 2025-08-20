/*! \file */
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

#pragma once

#include "rocsparse-types.h"

/********************************************************************************
 * \brief rocsparse_spgeam_descr is a structure holding the rocsparse spgeam
 * descriptor. It must be initialized using rocsparse_create_spgeam_descr().
 * It should be destroyed at the end using rocsparse_destroy_spgeam_descr().
 *******************************************************************************/
struct _rocsparse_spgeam_descr
{
    // C matrix row pointer data
    void*               csr_row_ptr_C{};
    int64_t             nnz_C{};
    int64_t             m{};
    rocsparse_indextype indextype{rocsparse_indextype_i32};

    // rocprim buffer data
    bool   rocprim_alloc{};
    size_t rocprim_size{};
    void*  rocprim_buffer{};

protected:
    rocsparse_spgeam_stage stage;
    rocsparse_spgeam_alg   alg;
    rocsparse_datatype     scalar_datatype;
    rocsparse_datatype     compute_datatype;
    rocsparse_operation    op_A;
    rocsparse_operation    op_B;
    const void*            scalar_A;
    const void*            scalar_B;

    float m_local_host_alpha_value[4];
    float m_local_host_beta_value[4];

public:
    ~_rocsparse_spgeam_descr() = default;

    _rocsparse_spgeam_descr()
        : stage((rocsparse_spgeam_stage)-1)
        , alg((rocsparse_spgeam_alg)-1)
        , scalar_datatype((rocsparse_datatype)-1)
        , compute_datatype((rocsparse_datatype)-1)
        , op_A((rocsparse_operation)-1)
        , op_B((rocsparse_operation)-1)
        , scalar_A(nullptr)
        , scalar_B(nullptr)
    {
    }

    void* get_local_host_alpha()
    {
        return &this->m_local_host_alpha_value[0];
    }
    void* get_local_host_beta()
    {
        return &this->m_local_host_beta_value[0];
    }

    rocsparse_spgeam_stage get_stage() const
    {
        return this->stage;
    }
    rocsparse_spgeam_alg get_alg() const
    {
        return this->alg;
    }
    rocsparse_operation get_operation_A() const
    {
        return this->op_A;
    }
    rocsparse_operation get_operation_B() const
    {
        return this->op_B;
    }

    const void* get_scalar_A() const
    {
        return this->scalar_A;
    }

    const void* get_scalar_B() const
    {
        return this->scalar_B;
    }

    rocsparse_datatype get_scalar_datatype() const
    {
        return this->scalar_datatype;
    }
    rocsparse_datatype get_compute_datatype() const
    {
        return this->compute_datatype;
    }

    void set_stage(rocsparse_spgeam_stage value)
    {
        this->stage = value;
    }
    void set_alg(rocsparse_spgeam_alg value)
    {
        this->alg = value;
    }

    void set_operation_A(rocsparse_operation value)
    {
        this->op_A = value;
    }
    void set_operation_B(rocsparse_operation value)
    {
        this->op_B = value;
    }

    void set_scalar_A(const void* value)
    {
        this->scalar_A = value;
    }
    void set_scalar_B(const void* value)
    {
        this->scalar_B = value;
    }

    void set_scalar_datatype(rocsparse_datatype value)
    {
        this->scalar_datatype = value;
    }
    void set_compute_datatype(rocsparse_datatype value)
    {
        this->compute_datatype = value;
    }

    rocsparse_status csrgeam_allocate_descr_memory(
        rocsparse_handle handle, int64_t m, int64_t n, int64_t nnz_A, int64_t nnz_B);

    rocsparse_status csrgeam_copy_row_pointer(rocsparse_handle          handle,
                                              int64_t                   m,
                                              int64_t                   n,
                                              const rocsparse_mat_descr descr_C,
                                              rocsparse_indextype       csr_row_ptr_C_indextype,
                                              void*                     csr_row_ptr_C,
                                              int64_t*                  nnz_C);
};
