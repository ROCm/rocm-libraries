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

#include "rocsparse_cscsv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Replaces *p by an array of size bytes, allocated on the stream.
    static rocsparse_status reallocate_async(void** p, size_t size, hipStream_t stream)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*p, stream));
        *p = nullptr;
        if(size > 0)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(p, size, stream));
        }
        return rocsparse_status_success;
    }

    // Replaces *dest by a copy of the size bytes of src, on the stream.
    static rocsparse_status
        duplicate_async(void** dest, const void* src, size_t size, hipStream_t stream)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::reallocate_async(dest, size, stream));
        if(size > 0)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(*dest, src, size, hipMemcpyDeviceToDevice, stream));
        }
        return rocsparse_status_success;
    }
}

_rocsparse_cscsv_info::~_rocsparse_cscsv_info()
{
    // hipFree does not wait for pending work on hipMallocAsync memory since HIP 7.0.
    WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());
    for(auto fill_mode : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_dependency_count[fill_mode]));
        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_task_col[fill_mode]));
        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_task_begin[fill_mode]));
    }
}

void _rocsparse_cscsv_info::copy(const _rocsparse_cscsv_info* that, hipStream_t stream)
{
    this->trm_data_t::copy(that, stream);
    this->m_csc_analysis = that->m_csc_analysis;
    for(auto fill_mode : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        const rocsparse::trm_info_t* trm = that->get(rocsparse_operation_none, fill_mode);

        const size_t m         = (trm != nullptr) ? trm->get_m() : 0;
        const size_t num_tasks = (trm != nullptr) ? that->m_num_tasks[fill_mode] : 0;
        const size_t sizeof_I
            = (trm != nullptr) ? rocsparse::indextype_sizeof(trm->get_offset_indextype()) : 0;
        const size_t sizeof_J
            = (trm != nullptr) ? rocsparse::indextype_sizeof(trm->get_index_indextype()) : 0;

        this->m_num_tasks[fill_mode] = that->m_num_tasks[fill_mode];
        this->m_num_lanes[fill_mode] = that->m_num_lanes[fill_mode];
        THROW_IF_ROCSPARSE_ERROR(rocsparse::duplicate_async(&this->m_dependency_count[fill_mode],
                                                            that->m_dependency_count[fill_mode],
                                                            sizeof_J * m,
                                                            stream));
        THROW_IF_ROCSPARSE_ERROR(rocsparse::duplicate_async(&this->m_task_col[fill_mode],
                                                            that->m_task_col[fill_mode],
                                                            sizeof_J * num_tasks,
                                                            stream));
        THROW_IF_ROCSPARSE_ERROR(rocsparse::duplicate_async(&this->m_task_begin[fill_mode],
                                                            that->m_task_begin[fill_mode],
                                                            sizeof_I * num_tasks,
                                                            stream));
    }
    THROW_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
}

rocsparse_status _rocsparse_cscsv_info::allocate_dependency_count(rocsparse_fill_mode fill_mode,
                                                                  size_t              size,
                                                                  hipStream_t         stream)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::reallocate_async(&this->m_dependency_count[fill_mode], size, stream));
    return rocsparse_status_success;
}

rocsparse_status _rocsparse_cscsv_info::allocate_tasks(rocsparse_fill_mode fill_mode,
                                                       int64_t             num_tasks,
                                                       uint32_t            num_lanes,
                                                       size_t              sizeof_col,
                                                       size_t              sizeof_begin,
                                                       hipStream_t         stream)
{
    this->m_num_tasks[fill_mode] = num_tasks;
    this->m_num_lanes[fill_mode] = num_lanes;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::reallocate_async(&this->m_task_col[fill_mode], sizeof_col * num_tasks, stream));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::reallocate_async(
        &this->m_task_begin[fill_mode], sizeof_begin * num_tasks, stream));
    return rocsparse_status_success;
}

rocsparse::pivot_info_t* _rocsparse_cscsv_info::get_symbolic_pivot()
{
    return this->uses_csrsv_info() ? static_cast<rocsparse::pivot_info_t*>(this->m_csrsv_info.get())
                                   : static_cast<rocsparse::pivot_info_t*>(this);
}

rocsparse::singular_info_t* _rocsparse_cscsv_info::get_exact_pivot()
{
    return this->uses_csrsv_info() ? this->m_csrsv_info->get_singularity_numeric_exact()
                                   : &this->m_singularity_numeric_exact;
}
