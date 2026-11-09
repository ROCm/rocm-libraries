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
    // Tasks of the solve per fill mode: column and first entry of each task.
    int64_t  m_num_tasks[2]{};
    uint32_t m_num_lanes[2]{};
    void*    m_task_col[2]{};
    void*    m_task_begin[2]{};
    // Transposed CSR analysis, shared with the matrix info with rocsparse_analysis_policy_reuse.
    std::shared_ptr<_rocsparse_csrsv_info> m_csrsv_info;

public:
    _rocsparse_cscsv_info()                             = default;
    _rocsparse_cscsv_info(const _rocsparse_cscsv_info&) = delete;
    _rocsparse_cscsv_info& operator=(const _rocsparse_cscsv_info&) = delete;

    ~_rocsparse_cscsv_info()
    {
        bool allocated = false;
        for(int f = 0; f < 2; ++f)
        {
            allocated |= (this->m_dependency_count[f] != nullptr)
                         || (this->m_task_col[f] != nullptr) || (this->m_task_begin[f] != nullptr);
        }
        if(allocated)
        {
            // hipFree does not wait for pending work on hipMallocAsync memory since HIP 7.0.
            WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());
        }
        for(int f = 0; f < 2; ++f)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_dependency_count[f]));
            WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_task_col[f]));
            WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_task_begin[f]));
        }
    }

    // Copies the CSC analysis of that; the transposed analysis is linked by the caller.
    void copy(const _rocsparse_cscsv_info* that, hipStream_t stream)
    {
        this->trm_data_t::copy(that, stream);
        for(int f = 0; f < 2; ++f)
        {
            this->m_num_tasks[f] = that->m_num_tasks[f];
            this->m_num_lanes[f] = that->m_num_lanes[f];
            const rocsparse_fill_mode fill_mode
                = (f == 0) ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
            const rocsparse::trm_info_t* trm = that->get(rocsparse_operation_none, fill_mode);
            if(trm == nullptr)
            {
                continue;
            }
            const size_t sizeof_I = rocsparse::indextype_sizeof(trm->get_offset_indextype());
            const size_t sizeof_J = rocsparse::indextype_sizeof(trm->get_index_indextype());
            copy_array(
                &this->m_dependency_count[f], that->m_dependency_count[f], sizeof_J * trm->get_m());
            copy_array(&this->m_task_col[f], that->m_task_col[f], sizeof_J * that->m_num_tasks[f]);
            copy_array(
                &this->m_task_begin[f], that->m_task_begin[f], sizeof_I * that->m_num_tasks[f]);
        }
    }

    static void copy_array(void** dest, const void* src, size_t size)
    {
        if(*dest != nullptr)
        {
            WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());
            WARNING_IF_HIP_ERROR(rocsparse_hipFree(*dest));
            *dest = nullptr;
        }
        if(src != nullptr && size > 0)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(dest, size));
            THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(*dest, src, size, hipMemcpyDeviceToDevice));
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

    int64_t get_num_tasks(rocsparse_fill_mode fill_mode) const
    {
        return this->m_num_tasks[fill_mode == rocsparse_fill_mode_upper];
    }

    uint32_t get_num_lanes(rocsparse_fill_mode fill_mode) const
    {
        return this->m_num_lanes[fill_mode == rocsparse_fill_mode_upper];
    }

    void set_tasks(rocsparse_fill_mode fill_mode, int64_t num_tasks, uint32_t num_lanes)
    {
        this->m_num_tasks[fill_mode == rocsparse_fill_mode_upper] = num_tasks;
        this->m_num_lanes[fill_mode == rocsparse_fill_mode_upper] = num_lanes;
    }

    const void* get_task_col(rocsparse_fill_mode fill_mode) const
    {
        return this->m_task_col[fill_mode == rocsparse_fill_mode_upper];
    }

    const void* get_task_begin(rocsparse_fill_mode fill_mode) const
    {
        return this->m_task_begin[fill_mode == rocsparse_fill_mode_upper];
    }

    void** get_ref_task_col(rocsparse_fill_mode fill_mode)
    {
        return &this->m_task_col[fill_mode == rocsparse_fill_mode_upper];
    }

    void** get_ref_task_begin(rocsparse_fill_mode fill_mode)
    {
        return &this->m_task_begin[fill_mode == rocsparse_fill_mode_upper];
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
