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

#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

/********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
void _rocsparse_csritsv_info::copy(const _rocsparse_csritsv_info* that, hipStream_t stream)
{
    ROCSPARSE_ROUTINE_TRACE;
    //
    // this == nullptr
    //
    if(that == nullptr || this == that)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }
    this->is_submatrix      = that->is_submatrix;
    this->ptr_end_size      = that->ptr_end_size;
    this->ptr_end_indextype = that->ptr_end_indextype;
    this->ptr_end           = that->ptr_end;
    this->rocsparse::pivot_info_t::copy_pivot_info_async(that, stream);
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
}

/********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
_rocsparse_csritsv_info::~_rocsparse_csritsv_info()
{
    ROCSPARSE_ROUTINE_TRACE;
    if(this->ptr_end != nullptr && this->is_submatrix)
    {
        // Due to the changes in the hipFree introduced in HIP 7.0
        // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
        // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
        // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
        // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
        WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->ptr_end));
        this->ptr_end = nullptr;
    }
    if(this->m_csrmv_info)
    {
        delete this->m_csrmv_info;
    }
}
