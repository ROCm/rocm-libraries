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

#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_trm_data_t.hpp"

#include <memory>

// Analysis of the CSC triangular solve. The CSC solve without transpose stores its analysis
// here under rocsparse_operation_none; the other operations use the transposed CSR analysis.
struct _rocsparse_cscsv_info : rocsparse::trm_data_t
{
protected:
    rocsparse::singular_info_t m_singularity_numeric_exact{};
    // Per fill mode, indexed by fill_index: number of unknowns each unknown depends on, and the
    // tasks of the solve (column and first entry of each task).
    void*    m_dependency_count[2]{};
    int64_t  m_num_tasks[2]{};
    uint32_t m_num_lanes[2]{};
    void*    m_task_col[2]{};
    void*    m_task_begin[2]{};
    // Transposed CSR analysis, shared with the matrix info with rocsparse_analysis_policy_reuse.
    std::shared_ptr<_rocsparse_csrsv_info> m_csrsv_info;
    // Whether the last analysis is the CSC one rather than the transposed CSR one.
    bool m_csc_analysis{};

    static_assert(rocsparse_fill_mode_lower == 0 && rocsparse_fill_mode_upper == 1,
                  "the fill mode indexes the per fill mode arrays");
    static int fill_index(rocsparse_fill_mode fill_mode)
    {
        return static_cast<int>(fill_mode);
    }

public:
    _rocsparse_cscsv_info()                             = default;
    _rocsparse_cscsv_info(const _rocsparse_cscsv_info&) = delete;
    _rocsparse_cscsv_info& operator=(const _rocsparse_cscsv_info&) = delete;
    ~_rocsparse_cscsv_info();

    // Copies the CSC analysis of that; the transposed analysis is linked by the caller.
    void copy(const _rocsparse_cscsv_info* that, hipStream_t stream);

    // Replaces the dependency counts and the task arrays of a fill mode by arrays of the given
    // sizes, allocated on the stream.
    rocsparse_status
        allocate_dependency_count(rocsparse_fill_mode fill_mode, size_t size, hipStream_t stream);
    rocsparse_status allocate_tasks(rocsparse_fill_mode fill_mode,
                                    int64_t             num_tasks,
                                    uint32_t            num_lanes,
                                    size_t              sizeof_col,
                                    size_t              sizeof_begin,
                                    hipStream_t         stream);

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

    void* get_dependency_count(rocsparse_fill_mode fill_mode) const
    {
        return this->m_dependency_count[fill_index(fill_mode)];
    }

    int64_t get_num_tasks(rocsparse_fill_mode fill_mode) const
    {
        return this->m_num_tasks[fill_index(fill_mode)];
    }

    uint32_t get_num_lanes(rocsparse_fill_mode fill_mode) const
    {
        return this->m_num_lanes[fill_index(fill_mode)];
    }

    void* get_task_col(rocsparse_fill_mode fill_mode) const
    {
        return this->m_task_col[fill_index(fill_mode)];
    }

    void* get_task_begin(rocsparse_fill_mode fill_mode) const
    {
        return this->m_task_begin[fill_index(fill_mode)];
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

    void set_csc_analysis(bool value)
    {
        this->m_csc_analysis = value;
    }

    // Whether the last analysis is the transposed CSR one.
    bool uses_csrsv_info() const
    {
        return !this->m_csc_analysis && this->m_csrsv_info != nullptr;
    }

    // Pivots of the analysis that was performed, CSC or transposed.
    rocsparse::pivot_info_t*    get_symbolic_pivot();
    rocsparse::singular_info_t* get_exact_pivot();
};
typedef _rocsparse_cscsv_info* rocsparse_cscsv_info;
