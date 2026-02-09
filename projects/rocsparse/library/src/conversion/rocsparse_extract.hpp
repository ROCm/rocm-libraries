/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/generic/rocsparse_extract.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

struct _rocsparse_extract_descr
{
protected:
    rocsparse_extract_alg   m_alg{};
    rocsparse_extract_stage m_stage{};
    rocsparse_direction     m_direction{};
    size_t                  m_stage_analysis_buffer_size{};
    size_t                  m_stage_compute_buffer_size{};

public:
    int64_t* m_device_nnz{};
    virtual ~_rocsparse_extract_descr()
    {
        // Due to the changes in the hipFree introduced in HIP 7.0
        // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
        // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
        // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
        // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
        WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

        std::ignore = rocsparse_hipFree(this->m_device_nnz);
    }

    ///
    /// @brief Get which algorithm is used.
    /// @return The algorithm to use.
    ///
    rocsparse_extract_alg alg() const
    {
        return this->m_alg;
    }

    ///
    /// @brief Set which algorithm to use.
    /// @param[in] value The algorithm to use.
    ///
    void alg(rocsparse_extract_alg value)
    {
        this->m_alg = value;
    }

    ///
    /// @brief Get which stage is used.
    /// @return The stage to use.
    ///
    rocsparse_extract_stage stage() const
    {
        return this->m_stage;
    }

    ///
    /// @brief Set which stage to use.
    /// @param[in] value The stage to use.
    ///
    void stage(rocsparse_extract_stage value)
    {
        this->m_stage = value;
    }

    ///
    /// @brief Constructor
    /// @param[in] value The algorithm to use.
    /// @param[in] source The source sparse matrix descriptor.
    /// @param[in] target The target sparse matrix descriptor.
    ///
    _rocsparse_extract_descr(rocsparse_extract_alg       alg,
                             rocsparse_const_spmat_descr source,
                             rocsparse_const_spmat_descr target)
        : m_alg(alg)
    {
        this->m_stage = rocsparse_extract_stage_analysis;
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&this->m_device_nnz, sizeof(int64_t)));
    }

    ///
    /// @brief Get the buffer size for a given stage to run.
    /// @param[in] handle The rocsparse handle.
    /// @param[in] source The source sparse matrix descriptor.
    /// @param[in] target The target sparse matrix descriptor.
    /// @param[in] stage  The stage to use.
    /// @param[out] buffer_size_in_bytes The calculated buffer size in bytes.
    /// @return rocsparse_status_success if the operation succesfull, the appropriate enumeration value otherwise.
    ///
    virtual rocsparse_status buffer_size(rocsparse_handle            handle,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_spmat_descr       target,
                                         rocsparse_extract_stage     stage,
                                         uint64_t* __restrict__ buffer_size_in_bytes)
        = 0;

    ///
    /// @brief Run a given stage.
    /// @param[in] handle The rocsparse handle.
    /// @param[in] source The source sparse matrix descriptor.
    /// @param[in] target The target sparse matrix descriptor.
    /// @param[in] stage  The stage to use.
    /// @param[in] buffer_size_in_bytes The calculated buffer size in bytes.
    /// @param[in] buffer The calculated buffer size in bytes.
    /// @return rocsparse_status_success if the operation succesfull, the appropriate enumeration value otherwise.
    ///
    virtual rocsparse_status run(rocsparse_handle            handle,
                                 rocsparse_const_spmat_descr source,
                                 rocsparse_spmat_descr       target,
                                 rocsparse_extract_stage     stage,
                                 uint64_t                    buffer_size_in_bytes,
                                 void* __restrict__ buffer)
        = 0;
};
