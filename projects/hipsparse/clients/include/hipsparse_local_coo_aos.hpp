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
#ifndef HIPSPARSE_LOCAL_COO_AOS_HPP
#define HIPSPARSE_LOCAL_COO_AOS_HPP

#include <hipsparse/hipsparse.h>

#include "arg_check.hpp"

/*! \brief  COO (AoS) matrix descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalCooAoS
{
    hipsparseSpMatDescr_t descr{};

public:
    hipsparseLocalCooAoS(int64_t              rows,
                         int64_t              cols,
                         int64_t              nnz,
                         void*                cooInd,
                         void*                cooValues,
                         hipsparseIndexType_t cooIdxType,
                         hipsparseIndexBase_t idxBase,
                         hipDataType          valueType)
    {
        const hipsparseStatus_t status = hipsparseCreateCooAoS(
            &descr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseCreateCooAoS failed: " << hipsparseGetErrorName(status)
                      << std::endl;
            throw(status);
        }
    }

    ~hipsparseLocalCooAoS()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroySpMat(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                verify_hipsparse_status_success(status, "ERROR: hipsparseLocalCooAoS destructor");
            }
        }
    }

    hipsparseLocalCooAoS(const hipsparseLocalCooAoS&) = delete;
    hipsparseLocalCooAoS& operator=(const hipsparseLocalCooAoS&) = delete;

    operator hipsparseSpMatDescr_t&()
    {
        return descr;
    }

    operator const hipsparseSpMatDescr_t&() const
    {
        return descr;
    }
};

// Exact match for the wrapper, so this beats the implicit conversion to
// hipsparseSpMatDescr_t. A leftover hipsparseDestroySpMat(wrapper) is a compile
// error instead of destroying the descriptor twice.
hipsparseStatus_t hipsparseDestroySpMat(const hipsparseLocalCooAoS&) = delete;

#endif // HIPSPARSE_LOCAL_COO_AOS_HPP
