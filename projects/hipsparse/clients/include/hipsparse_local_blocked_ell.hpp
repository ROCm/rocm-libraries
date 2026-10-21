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
#ifndef HIPSPARSE_LOCAL_BLOCKED_ELL_HPP
#define HIPSPARSE_LOCAL_BLOCKED_ELL_HPP

#include <hipsparse/hipsparse.h>

/*! \brief  blocked ELL matrix descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalBlockedEll
{
    hipsparseSpMatDescr_t descr{};

public:
    hipsparseLocalBlockedEll(int64_t              rows,
                             int64_t              cols,
                             int64_t              ellBlockSize,
                             int64_t              ellCols,
                             void*                ellColInd,
                             void*                ellValue,
                             hipsparseIndexType_t ellIdxType,
                             hipsparseIndexBase_t idxBase,
                             hipDataType          valueType)
    {
        const hipsparseStatus_t status = hipsparseCreateBlockedEll(&descr,
                                                                   rows,
                                                                   cols,
                                                                   ellBlockSize,
                                                                   ellCols,
                                                                   ellColInd,
                                                                   ellValue,
                                                                   ellIdxType,
                                                                   idxBase,
                                                                   valueType);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            if(descr != nullptr)
            {
                hipsparseDestroySpMat(descr);
                descr = nullptr;
            }
            throw(status);
        }
    }

    ~hipsparseLocalBlockedEll()
    {
        if(descr != nullptr)
        {
            hipsparseDestroySpMat(descr);
        }
    }

    hipsparseLocalBlockedEll(const hipsparseLocalBlockedEll&)            = delete;
    hipsparseLocalBlockedEll& operator=(const hipsparseLocalBlockedEll&) = delete;

    operator hipsparseSpMatDescr_t&()
    {
        return descr;
    }

    operator const hipsparseSpMatDescr_t&() const
    {
        return descr;
    }
};

#endif // HIPSPARSE_LOCAL_BLOCKED_ELL_HPP
