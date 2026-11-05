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
#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_trm_data_t.hpp"

#include <memory>

// Analysis of the CSC triangular solve. The CSC solve without transpose stores its analysis
// here under rocsparse_operation_none; the other operations use the transposed CSR analysis.
struct _rocsparse_cscsv_info : rocsparse::trm_data_t
{
protected:
    rocsparse::singular_info_t m_singularity_numeric_exact{};
    // Number of unknowns each unknown depends on, per fill mode.
    void* m_dependency_count[2]{};
    // Transposed CSR analysis, shared with the matrix info with rocsparse_analysis_policy_reuse.
    std::shared_ptr<_rocsparse_csrsv_info> m_csrsv_info;

public:
    _rocsparse_cscsv_info()                             = default;
    _rocsparse_cscsv_info(const _rocsparse_cscsv_info&) = delete;
    _rocsparse_cscsv_info& operator=(const _rocsparse_cscsv_info&) = delete;

    ~_rocsparse_cscsv_info()
    {
        for(auto& p : this->m_dependency_count)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipFree(p));
        }
    }

    // Copies the CSC analysis of that; the transposed analysis is linked by the caller.
    void copy(const _rocsparse_cscsv_info* that, hipStream_t stream)
    {
        this->trm_data_t::copy(that, stream);
        for(int f = 0; f < 2; ++f)
        {
            const rocsparse_fill_mode fill_mode
                = (f == 0) ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
            const rocsparse::trm_info_t* trm = that->get(rocsparse_operation_none, fill_mode);
            if(that->m_dependency_count[f] != nullptr && trm != nullptr)
            {
                const size_t size
                    = rocsparse::indextype_sizeof(trm->get_index_indextype()) * trm->get_m();
                WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_dependency_count[f]));
                THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&this->m_dependency_count[f], size));
                THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->m_dependency_count[f],
                                                       that->m_dependency_count[f],
                                                       size,
                                                       hipMemcpyDeviceToDevice));
            }
        }
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

    const void* get_dependency_count(rocsparse_fill_mode fill_mode) const
    {
        return this->m_dependency_count[fill_mode == rocsparse_fill_mode_upper];
    }

    void* get_dependency_count(rocsparse_fill_mode fill_mode)
    {
        return this->m_dependency_count[fill_mode == rocsparse_fill_mode_upper];
    }

    void** get_ref_dependency_count(rocsparse_fill_mode fill_mode)
    {
        return &this->m_dependency_count[fill_mode == rocsparse_fill_mode_upper];
    }

    rocsparse_csrsv_info get_csrsv_info()
    {
        return this->m_csrsv_info.get();
    }

    void set_csrsv_info(rocsparse_csrsv_info value)
    {
        this->m_csrsv_info = std::shared_ptr<_rocsparse_csrsv_info>(value);
    }

    void set_shared_csrsv_info(std::shared_ptr<_rocsparse_csrsv_info> value)
    {
        this->m_csrsv_info = value;
    }

    // Pivots of the analysis that was performed, CSC or transposed.
    rocsparse::pivot_info_t* get_symbolic_pivot()
    {
        return (this->m_csrsv_info != nullptr)
                   ? static_cast<rocsparse::pivot_info_t*>(this->m_csrsv_info.get())
                   : static_cast<rocsparse::pivot_info_t*>(this);
    }

    rocsparse::singular_info_t* get_exact_pivot()
    {
        return (this->m_csrsv_info != nullptr) ? this->m_csrsv_info->get_singularity_numeric_exact()
                                               : &this->m_singularity_numeric_exact;
    }
};
typedef _rocsparse_cscsv_info* rocsparse_cscsv_info;
