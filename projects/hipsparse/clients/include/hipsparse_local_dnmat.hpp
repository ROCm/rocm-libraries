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
#ifndef HIPSPARSE_LOCAL_DNMAT_HPP
#define HIPSPARSE_LOCAL_DNMAT_HPP

#include <hipsparse/hipsparse.h>

#include "utility.hpp"

/*! \brief  Dense matrix descriptor which is destroyed when it goes out of scope. */
class hipsparseLocalDnMat
{
    hipsparseDnMatDescr_t descr{};

public:
    hipsparseLocalDnMat(int64_t          rows,
                        int64_t          cols,
                        int64_t          ld,
                        void*            values,
                        hipDataType      valueType,
                        hipsparseOrder_t order)
    {
        const hipsparseStatus_t status
            = hipsparseCreateDnMat(&descr, rows, cols, ld, values, valueType, order);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseCreateDnMat", status);
        }
    }

    ~hipsparseLocalDnMat()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroyDnMat(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseDestroyDnMat", status);
            }
        }
    }

    hipsparseLocalDnMat(const hipsparseLocalDnMat&) = delete;
    hipsparseLocalDnMat& operator=(const hipsparseLocalDnMat&) = delete;

    operator hipsparseDnMatDescr_t() const
    {
        return descr;
    }
};

// These beat the implicit conversion to void*. The handle, matrix descriptor,
// HYB, color, sparse-vector, dense-vector, and dense-matrix typedefs are all
// void*, so a leftover destroy of the wrong kind would otherwise compile and
// free this descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalDnMat&)          = delete;
hipsparseStatus_t hipsparseDestroyMatDescr(const hipsparseLocalDnMat&)  = delete;
hipsparseStatus_t hipsparseDestroyHybMat(const hipsparseLocalDnMat&)    = delete;
hipsparseStatus_t hipsparseDestroyColorInfo(const hipsparseLocalDnMat&) = delete;
hipsparseStatus_t hipsparseDestroySpVec(const hipsparseLocalDnMat&)     = delete;
hipsparseStatus_t hipsparseDestroyDnVec(const hipsparseLocalDnMat&)     = delete;
hipsparseStatus_t hipsparseDestroyDnMat(const hipsparseLocalDnMat&)     = delete;

#endif // HIPSPARSE_LOCAL_DNMAT_HPP
