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
#ifndef HIPSPARSE_LOCAL_SPMAT_HPP
#define HIPSPARSE_LOCAL_SPMAT_HPP

#include "utility.hpp"

#include <hipsparse/hipsparse.h>
#include <type_traits>

namespace hipsparse_matrix_format
{
    struct Csr
    {
    };
    struct Coo
    {
    };
    struct CooAoS
    {
    };
    struct Csc
    {
    };
    struct Bsr
    {
    };
    struct BlockedEll
    {
    };
    struct SlicedEll
    {
    };
}

// Drops a constructor unless this specialization was created for RequiredFormat.
template <typename Actual, typename RequiredFormat>
using hipsparse_local_spmat_ctor_enable
    = std::enable_if_t<std::is_same<Actual, RequiredFormat>::value, int>;

// Same availability as hipsparseDestroySpMat. The format aliases below name each
// specialization; only the matching constructor is callable.
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
template <typename Format>
class hipsparseLocalSpMat
{
    hipsparseSpMatDescr_t descr{};

    void check_status(const char* call, hipsparseStatus_t status)
    {
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error(call, status);
        }
    }

public:
    template <typename Required                                                         = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::Csr> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              nnz,
                        void*                csrRowOffsets,
                        void*                csrColInd,
                        void*                csrValues,
                        hipsparseIndexType_t csrRowOffsetsType,
                        hipsparseIndexType_t csrColIndType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status("hipsparseCreateCsr",
                     hipsparseCreateCsr(&descr,
                                        rows,
                                        cols,
                                        nnz,
                                        csrRowOffsets,
                                        csrColInd,
                                        csrValues,
                                        csrRowOffsetsType,
                                        csrColIndType,
                                        idxBase,
                                        valueType));
    }

    template <typename Required                                                         = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::Coo> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              nnz,
                        void*                cooRowInd,
                        void*                cooColInd,
                        void*                cooValues,
                        hipsparseIndexType_t cooIdxType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status("hipsparseCreateCoo",
                     hipsparseCreateCoo(&descr,
                                        rows,
                                        cols,
                                        nnz,
                                        cooRowInd,
                                        cooColInd,
                                        cooValues,
                                        cooIdxType,
                                        idxBase,
                                        valueType));
    }

#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
    template <typename Required                                                            = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::CooAoS> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              nnz,
                        void*                cooInd,
                        void*                cooValues,
                        hipsparseIndexType_t cooIdxType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status(
            "hipsparseCreateCooAoS",
            hipsparseCreateCooAoS(
                &descr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType));
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
    template <typename Required                                                         = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::Csc> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              nnz,
                        void*                cscColOffsets,
                        void*                cscRowInd,
                        void*                cscValues,
                        hipsparseIndexType_t cscColOffsetsType,
                        hipsparseIndexType_t cscRowIndType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status("hipsparseCreateCsc",
                     hipsparseCreateCsc(&descr,
                                        rows,
                                        cols,
                                        nnz,
                                        cscColOffsets,
                                        cscRowInd,
                                        cscValues,
                                        cscColOffsetsType,
                                        cscRowIndType,
                                        idxBase,
                                        valueType));
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
    template <typename Required = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::BlockedEll> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              ellBlockSize,
                        int64_t              ellCols,
                        void*                ellColInd,
                        void*                ellValue,
                        hipsparseIndexType_t ellIdxType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status("hipsparseCreateBlockedEll",
                     hipsparseCreateBlockedEll(&descr,
                                               rows,
                                               cols,
                                               ellBlockSize,
                                               ellCols,
                                               ellColInd,
                                               ellValue,
                                               ellIdxType,
                                               idxBase,
                                               valueType));
    }
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)
    template <typename Required                                                         = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::Bsr> = 0>
    hipsparseLocalSpMat(int64_t              mb,
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
        check_status("hipsparseCreateBsr",
                     hipsparseCreateBsr(&descr,
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
                                        order));
    }

    template <typename Required = Format,
              hipsparse_local_spmat_ctor_enable<Required, hipsparse_matrix_format::SlicedEll> = 0>
    hipsparseLocalSpMat(int64_t              rows,
                        int64_t              cols,
                        int64_t              nnz,
                        int64_t              sellValuesSize,
                        int64_t              sliceSize,
                        void*                sellSliceOffsets,
                        void*                sellColInd,
                        void*                sellValues,
                        hipsparseIndexType_t sellSliceOffsetsType,
                        hipsparseIndexType_t sellColIndType,
                        hipsparseIndexBase_t idxBase,
                        hipDataType          valueType)
    {
        check_status("hipsparseCreateSlicedEll",
                     hipsparseCreateSlicedEll(&descr,
                                              rows,
                                              cols,
                                              nnz,
                                              sellValuesSize,
                                              sliceSize,
                                              sellSliceOffsets,
                                              sellColInd,
                                              sellValues,
                                              sellSliceOffsetsType,
                                              sellColIndType,
                                              idxBase,
                                              valueType));
    }
#endif

    ~hipsparseLocalSpMat()
    {
        if(descr != nullptr)
        {
            const hipsparseStatus_t status = hipsparseDestroySpMat(descr);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseDestroySpMat", status);
            }
        }
    }

    hipsparseLocalSpMat(const hipsparseLocalSpMat&) = delete;
    hipsparseLocalSpMat& operator=(const hipsparseLocalSpMat&) = delete;

    operator hipsparseSpMatDescr_t&()
    {
        return descr;
    }

    operator const hipsparseSpMatDescr_t&() const
    {
        return descr;
    }
};

// Exact match for every format alias, so this beats the implicit conversion to
// hipsparseSpMatDescr_t. A leftover hipsparseDestroySpMat(wrapper) is a compile
// error instead of destroying the descriptor twice.
template <typename Format>
hipsparseStatus_t hipsparseDestroySpMat(const hipsparseLocalSpMat<Format>&) = delete;

/*! \brief  CSR matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalCsr = hipsparseLocalSpMat<hipsparse_matrix_format::Csr>;
/*! \brief  COO matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalCoo = hipsparseLocalSpMat<hipsparse_matrix_format::Coo>;

#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
/*! \brief  COO (AoS) matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalCooAoS = hipsparseLocalSpMat<hipsparse_matrix_format::CooAoS>;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
/*! \brief  CSC matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalCsc = hipsparseLocalSpMat<hipsparse_matrix_format::Csc>;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
/*! \brief  blocked ELL matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalBlockedEll = hipsparseLocalSpMat<hipsparse_matrix_format::BlockedEll>;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)
/*! \brief  BSR matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalBsr = hipsparseLocalSpMat<hipsparse_matrix_format::Bsr>;
/*! \brief  sliced ELL matrix descriptor which is destroyed when it goes out of scope. */
using hipsparseLocalSlicedEll = hipsparseLocalSpMat<hipsparse_matrix_format::SlicedEll>;
#endif
#endif

#endif // HIPSPARSE_LOCAL_SPMAT_HPP
