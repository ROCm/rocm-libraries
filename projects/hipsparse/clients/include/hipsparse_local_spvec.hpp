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
#ifndef HIPSPARSE_LOCAL_SPVEC_HPP
#define HIPSPARSE_LOCAL_SPVEC_HPP

#include <hipsparse/hipsparse.h>

#include "arg_check.hpp"

/*! \brief  Sparse vector descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalSpVec
{
    hipsparseSpVecDescr_t descr{};

public:
    hipsparseLocalSpVec(int64_t              size,
                        int64_t              nnz,
                        void*                indices,
                        void*                values,
                        hipsparseIndexType_t idxType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        const hipsparseStatus_t status
            = hipsparseCreateSpVec(&descr, size, nnz, indices, values, idxType, idxBase, valueType);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseCreateSpVec failed: " << hipsparseGetErrorName(status)
                      << std::endl;
            throw(status);
        }
    }

    ~hipsparseLocalSpVec()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroySpVec(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                verify_hipsparse_status_success(status, "ERROR: hipsparseLocalSpVec destructor");
            }
        }
    }

    hipsparseLocalSpVec(const hipsparseLocalSpVec&) = delete;
    hipsparseLocalSpVec& operator=(const hipsparseLocalSpVec&) = delete;

    operator hipsparseSpVecDescr_t() const
    {
        return descr;
    }
};

// Exact match for the wrapper, so this beats the implicit conversion to
// hipsparseSpVecDescr_t. A leftover hipsparseDestroySpVec(wrapper) is a compile
// error instead of destroying the descriptor twice.
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalSpVec&) = delete;

#endif // HIPSPARSE_LOCAL_SPVEC_HPP
