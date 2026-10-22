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
#ifndef HIPSPARSE_LOCAL_BSR_HPP
#define HIPSPARSE_LOCAL_BSR_HPP

#include <hipsparse/hipsparse.h>

#include "arg_check.hpp"

/*! \brief  BSR matrix descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalBsr
{
    hipsparseSpMatDescr_t descr{};

public:
    hipsparseLocalBsr(int64_t              mb,
                      int64_t              nb,
                      int64_t              nnzb,
                      int64_t              rowBlockDim,
                      int64_t              colBlockDim,
                      void*                bsrRowPtr,
                      void*                bsrColInd,
                      void*                bsrValues,
                      hipsparseIndexType_t bsrRowPtrType,
                      hipsparseIndexType_t bsrColIndType,
                      hipsparseIndexBase_t idxBase,
                      hipDataType          valueType,
                      hipsparseOrder_t     order)
    {
        const hipsparseStatus_t status = hipsparseCreateBsr(&descr,
                                                            mb,
                                                            nb,
                                                            nnzb,
                                                            rowBlockDim,
                                                            colBlockDim,
                                                            bsrRowPtr,
                                                            bsrColInd,
                                                            bsrValues,
                                                            bsrRowPtrType,
                                                            bsrColIndType,
                                                            idxBase,
                                                            valueType,
                                                            order);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "hipsparseCreateBsr failed: " << hipsparseGetErrorName(status)
                      << std::endl;
            throw(status);
        }
    }

    ~hipsparseLocalBsr()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroySpMat(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                verify_hipsparse_status_success(status, "ERROR: hipsparseLocalBsr destructor");
            }
        }
    }

    hipsparseLocalBsr(const hipsparseLocalBsr&) = delete;
    hipsparseLocalBsr& operator=(const hipsparseLocalBsr&) = delete;

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
hipsparseStatus_t hipsparseDestroySpMat(const hipsparseLocalBsr&) = delete;

#endif // HIPSPARSE_LOCAL_BSR_HPP
