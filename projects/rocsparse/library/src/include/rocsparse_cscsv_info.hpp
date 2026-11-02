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

#pragma once

#include "rocsparse_control.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_trm_data_t.hpp"

// Analysis of the CSC column triangular solve, stored under rocsparse_operation_none.
struct _rocsparse_cscsv_info : rocsparse::trm_data_t
{
protected:
    rocsparse::singular_info_t m_singularity_numeric_exact{};
    // Number of unknowns each unknown depends on.
    void* m_dependency_count{nullptr};

public:
    _rocsparse_cscsv_info()                             = default;
    _rocsparse_cscsv_info(const _rocsparse_cscsv_info&) = delete;
    _rocsparse_cscsv_info& operator=(const _rocsparse_cscsv_info&) = delete;

    ~_rocsparse_cscsv_info()
    {
        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_dependency_count));
    }

    const void* get_dependency_count() const
    {
        return this->m_dependency_count;
    }

    void* get_dependency_count()
    {
        return this->m_dependency_count;
    }

    void** get_ref_dependency_count()
    {
        return &this->m_dependency_count;
    }

    rocsparse::singular_info_t* get_singularity_numeric_exact()
    {
        return &this->m_singularity_numeric_exact;
    }

    void create_singularity_numeric_exact(int64_t             batch_count,
                                          rocsparse_indextype indextype,
                                          hipStream_t         stream)
    {
        THROW_IF_ROCSPARSE_ERROR(this->m_singularity_numeric_exact.create_singular_pivot_async(
            batch_count, indextype, stream));
    }
};
typedef _rocsparse_cscsv_info* rocsparse_cscsv_info;
