/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "utility.h"

hipsparseSpMVDescr_st::Entry::Entry(Entry&& other) noexcept
    : operation(other.operation)
    , alg(other.alg)
    , scalar_datatype(other.scalar_datatype)
    , compute_datatype(other.compute_datatype)
    , spmv_descr(other.spmv_descr)
    , buffer_size_stage_analysis(other.buffer_size_stage_analysis)
    , buffer_size_stage_compute(other.buffer_size_stage_compute)
    , compute_buffer(other.compute_buffer)
    , is_buffer_size_called(other.is_buffer_size_called)
    , is_stage_analysis_called(other.is_stage_analysis_called)
    , is_implicit_stage_analysis_called(other.is_implicit_stage_analysis_called)
    , is_stage_compute_subsequent(other.is_stage_compute_subsequent)
{
    // Transfer ownership of the owned resources; the moved-from object's
    // destructor must not double-free them.
    other.spmv_descr     = nullptr;
    other.compute_buffer = nullptr;
}

hipsparseSpMVDescr_st::Entry::~Entry()
{
    // Destructors cannot propagate status codes, so the rocsparse / hip
    // return values are intentionally discarded here. Entries are only
    // destroyed at sparse-matrix-descriptor tear-down time (via
    // m_entries.clear() in the implicit destructor), so failures
    // here would only mean a leaked rocsparse descriptor or hip buffer
    // (already on the tear-down path).
    if(this->spmv_descr != nullptr)
    {
        (void)rocsparse_destroy_spmv_descr(this->spmv_descr);
        this->spmv_descr = nullptr;
    }

    if(this->compute_buffer != nullptr)
    {
        (void)hipFree(this->compute_buffer);
        this->compute_buffer = nullptr;
    }
}

hipsparseSpMVDescr_st::Entry* hipsparseSpMVDescr_st::find_entry(rocsparse_operation operation,
                                                                rocsparse_spmv_alg  alg,
                                                                rocsparse_datatype  scalar_datatype,
                                                                rocsparse_datatype compute_datatype)
{
    for(auto& entry : this->m_entries)
    {
        if(entry.operation == operation && entry.alg == alg
           && entry.scalar_datatype == scalar_datatype
           && entry.compute_datatype == compute_datatype)
        {
            return &entry;
        }
    }
    return nullptr;
}

hipsparseStatus_t hipsparseSpMVDescr_st::add_entry(hipsparseHandle_t   handle,
                                                   rocsparse_operation operation,
                                                   rocsparse_spmv_alg  alg,
                                                   rocsparse_datatype  scalar_datatype,
                                                   rocsparse_datatype  compute_datatype,
                                                   Entry**             out_entry)
{
    // Build the entry as a local first. If any of the rocsparse calls below
    // fails the local entry's destructor frees the partially-initialized
    // rocsparse_spmv_descr, so push_back is only reached on full success.
    Entry entry;
    entry.operation        = operation;
    entry.alg              = alg;
    entry.scalar_datatype  = scalar_datatype;
    entry.compute_datatype = compute_datatype;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&entry.spmv_descr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       entry.spmv_descr,
                                                       rocsparse_spmv_input_alg,
                                                       &entry.alg,
                                                       sizeof(entry.alg),
                                                       nullptr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       entry.spmv_descr,
                                                       rocsparse_spmv_input_operation,
                                                       &entry.operation,
                                                       sizeof(entry.operation),
                                                       nullptr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       entry.spmv_descr,
                                                       rocsparse_spmv_input_scalar_datatype,
                                                       &entry.scalar_datatype,
                                                       sizeof(entry.scalar_datatype),
                                                       nullptr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       entry.spmv_descr,
                                                       rocsparse_spmv_input_compute_datatype,
                                                       &entry.compute_datatype,
                                                       sizeof(entry.compute_datatype),
                                                       nullptr));

    // Move the fully-built entry into the cache. Entry's move constructor is
    // noexcept, so push_back is strong-exception-safe and any vector
    // reallocation relocates existing entries by moving their owned
    // rocsparse descriptors / compute buffers (never double-freed). Take
    // the address only after push_back so that *out_entry refers to the
    // final stored location, not the local 'entry' that is about to die.
    this->m_entries.push_back(std::move(entry));
    *out_entry = &this->m_entries.back();

    return HIPSPARSE_STATUS_SUCCESS;
}

// hipsparseSpMatDescr_st
rocsparse_spmat_descr hipsparseSpMatDescr_st::get_spmat_descr()
{
    return this->m_spmat_descr;
}

rocsparse_const_spmat_descr hipsparseSpMatDescr_st::get_const_spmat_descr() const
{
    return this->m_spmat_descr;
}

void hipsparseSpMatDescr_st::set_spmat_descr(rocsparse_spmat_descr value)
{
    this->m_spmat_descr = value;
}

hipsparseSpMVDescr_st* hipsparseSpMatDescr_st::get_hip_spmv_descr()
{
    return &this->m_hip_spmv_descr;
}

hipsparseSpMVDescr_st* hipsparseSpMatDescr_st::get_hip_spmv_descr() const
{
    return &this->m_hip_spmv_descr;
}

rocsparse_spmat_descr* hipsparseSpMatDescr_st::get_spmat_descr_reference()
{
    return &this->m_spmat_descr;
}

rocsparse_const_spmat_descr* hipsparseSpMatDescr_st::get_const_spmat_descr_reference() const
{
    return (rocsparse_const_spmat_descr*)&this->m_spmat_descr;
}

//
// Cast hipsparseSpMatDescr_st to rocsparse_spmat_descr.
//
rocsparse_const_spmat_descr to_rocsparse_const_spmat_descr(const hipsparseConstSpMatDescr_t source)
{
    return (source != nullptr) ? source->get_const_spmat_descr() : nullptr;
}

rocsparse_spmat_descr to_rocsparse_spmat_descr(const hipsparseSpMatDescr_t source)
{
    return (source != nullptr) ? source->get_spmat_descr() : nullptr;
}

/* Generic API */
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_spvec_descr((rocsparse_spvec_descr*)spVecDescr,
                                     size,
                                     nnz,
                                     indices,
                                     values,
                                     hipsparse::hipIndexTypeToHCCIndexType(idxType),
                                     hipsparse::hipBaseToHCCBase(idxBase),
                                     hipsparse::hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateConstSpVec(hipsparseConstSpVecDescr_t* spVecDescr,
                                            int64_t                     size,
                                            int64_t                     nnz,
                                            const void*                 indices,
                                            const void*                 values,
                                            hipsparseIndexType_t        idxType,
                                            hipsparseIndexBase_t        idxBase,
                                            hipDataType                 valueType)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_spvec_descr((rocsparse_const_spvec_descr*)spVecDescr,
                                           size,
                                           nnz,
                                           indices,
                                           values,
                                           hipsparse::hipIndexTypeToHCCIndexType(idxType),
                                           hipsparse::hipBaseToHCCBase(idxBase),
                                           hipsparse::hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseDestroySpVec(hipsparseConstSpVecDescr_t spVecDescr)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_spvec_descr((rocsparse_const_spvec_descr)spVecDescr));
}

hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spvec_get((const rocsparse_spvec_descr)spVecDescr,
                                                  size,
                                                  nnz,
                                                  indices,
                                                  values,
                                                  idxType != nullptr ? &hcc_index_type : nullptr,
                                                  idxBase != nullptr ? &hcc_index_base : nullptr,
                                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstSpVecGet(hipsparseConstSpVecDescr_t spVecDescr,
                                         int64_t*                   size,
                                         int64_t*                   nnz,
                                         const void**               indices,
                                         const void**               values,
                                         hipsparseIndexType_t*      idxType,
                                         hipsparseIndexBase_t*      idxBase,
                                         hipDataType*               valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_spvec_get((rocsparse_const_spvec_descr)spVecDescr,
                                  size,
                                  nnz,
                                  indices,
                                  values,
                                  idxType != nullptr ? &hcc_index_type : nullptr,
                                  idxBase != nullptr ? &hcc_index_base : nullptr,
                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseConstSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*            idxBase)
{
    rocsparse_index_base hcc_index_base;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_spvec_get_index_base((const rocsparse_const_spvec_descr)spVecDescr,
                                       idxBase != nullptr ? &hcc_index_base : nullptr));

    *idxBase = hipsparse::HCCBaseToHIPBase(hcc_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spvec_get_values((const rocsparse_spvec_descr)spVecDescr, values));
}

hipsparseStatus_t hipsparseConstSpVecGetValues(hipsparseConstSpVecDescr_t spVecDescr,
                                               const void**               values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_const_spvec_get_values((rocsparse_const_spvec_descr)spVecDescr, values));
}

hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spvec_set_values((rocsparse_spvec_descr)spVecDescr, values));
}

hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cooRowInd,
                                     void*                  cooColInd,
                                     void*                  cooValues,
                                     hipsparseIndexType_t   cooIdxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_coo_descr(spMatDescr[0]->get_spmat_descr_reference(),
                                   rows,
                                   cols,
                                   nnz,
                                   cooRowInd,
                                   cooColInd,
                                   cooValues,
                                   hipsparse::hipIndexTypeToHCCIndexType(cooIdxType),
                                   hipsparse::hipBaseToHCCBase(idxBase),
                                   hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateConstCoo(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 cooRowInd,
                                          const void*                 cooColInd,
                                          const void*                 cooValues,
                                          hipsparseIndexType_t        cooIdxType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_coo_descr(spMatDescr[0]->get_const_spmat_descr_reference(),
                                         rows,
                                         cols,
                                         nnz,
                                         cooRowInd,
                                         cooColInd,
                                         cooValues,
                                         hipsparse::hipIndexTypeToHCCIndexType(cooIdxType),
                                         hipsparse::hipBaseToHCCBase(idxBase),
                                         hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateBlockedEll(hipsparseSpMatDescr_t* spMatDescr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                ellBlockSize,
                                            int64_t                ellCols,
                                            void*                  ellColInd,
                                            void*                  ellValue,
                                            hipsparseIndexType_t   ellIdxType,
                                            hipsparseIndexBase_t   idxBase,
                                            hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_bell_descr(spMatDescr[0]->get_spmat_descr_reference(),
                                    rows,
                                    cols,
                                    rocsparse_direction_column,
                                    ellBlockSize,
                                    ellCols,
                                    ellColInd,
                                    ellValue,
                                    hipsparse::hipIndexTypeToHCCIndexType(ellIdxType),
                                    hipsparse::hipBaseToHCCBase(idxBase),
                                    hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateConstBlockedEll(hipsparseConstSpMatDescr_t* spMatDescr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 int64_t                     ellBlockSize,
                                                 int64_t                     ellCols,
                                                 const void*                 ellColInd,
                                                 const void*                 ellValue,
                                                 hipsparseIndexType_t        ellIdxType,
                                                 hipsparseIndexBase_t        idxBase,
                                                 hipDataType                 valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_bell_descr(spMatDescr[0]->get_const_spmat_descr_reference(),
                                          rows,
                                          cols,
                                          rocsparse_direction_column,
                                          ellBlockSize,
                                          ellCols,
                                          ellColInd,
                                          ellValue,
                                          hipsparse::hipIndexTypeToHCCIndexType(ellIdxType),
                                          hipsparse::hipBaseToHCCBase(idxBase),
                                          hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateSlicedEll(hipsparseSpMatDescr_t* spMatDescr,
                                           int64_t                rows,
                                           int64_t                cols,
                                           int64_t                nnz,
                                           int64_t                sellValuesSize,
                                           int64_t                sliceSize,
                                           void*                  sellSliceOffsets,
                                           void*                  sellColInd,
                                           void*                  sellValues,
                                           hipsparseIndexType_t   sellSliceOffsetsType,
                                           hipsparseIndexType_t   sellColIndType,
                                           hipsparseIndexBase_t   idxBase,
                                           hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    spMatDescr[0] = new hipsparseSpMatDescr_st();
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_sell_descr(spMatDescr[0]->get_spmat_descr_reference(),
                                    rows,
                                    cols,
                                    nnz,
                                    sliceSize,
                                    sellValuesSize,
                                    sellSliceOffsets,
                                    sellColInd,
                                    sellValues,
                                    hipsparse::hipIndexTypeToHCCIndexType(sellSliceOffsetsType),
                                    hipsparse::hipIndexTypeToHCCIndexType(sellColIndType),
                                    hipsparse::hipBaseToHCCBase(idxBase),
                                    hipsparse::hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateConstSlicedEll(hipsparseConstSpMatDescr_t* spMatDescr,
                                                int64_t                     rows,
                                                int64_t                     cols,
                                                int64_t                     nnz,
                                                int64_t                     sellValuesSize,
                                                int64_t                     sliceSize,
                                                const void*                 sellSliceOffsets,
                                                const void*                 sellColInd,
                                                const void*                 sellValues,
                                                hipsparseIndexType_t        sellSliceOffsetsType,
                                                hipsparseIndexType_t        sellColIndType,
                                                hipsparseIndexBase_t        idxBase,
                                                hipDataType                 valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_create_const_sell_descr(
        spMatDescr[0]->get_const_spmat_descr_reference(),
        rows,
        cols,
        nnz,
        sliceSize,
        sellValuesSize,
        sellSliceOffsets,
        sellColInd,
        sellValues,
        hipsparse::hipIndexTypeToHCCIndexType(sellSliceOffsetsType),
        hipsparse::hipIndexTypeToHCCIndexType(sellColIndType),
        hipsparse::hipBaseToHCCBase(idxBase),
        hipsparse::hipDataTypeToHCCDataType(valueType)));
}

#ifdef HIPSPARSE_WITH_SPMV_BSR
hipsparseStatus_t hipsparseCreateBsr(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                mb,
                                     int64_t                nb,
                                     int64_t                nnzb,
                                     int64_t                rowBlockDim,
                                     int64_t                colBlockDim,
                                     void*                  bsrRowPtr,
                                     void*                  bsrColInd,
                                     void*                  bsrValues,
                                     hipsparseIndexType_t   bsrRowPtrType,
                                     hipsparseIndexType_t   bsrColIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType,
                                     hipsparseOrder_t       order)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(rowBlockDim != colBlockDim)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_create_bsr_descr(
        spMatDescr[0]->get_spmat_descr_reference(),
        mb,
        nb,
        nnzb,
        (order == HIPSPARSE_ORDER_ROW)
            ? hipsparse::hipDirectionToHCCDirection(HIPSPARSE_DIRECTION_ROW)
            : hipsparse::hipDirectionToHCCDirection(HIPSPARSE_DIRECTION_COLUMN),
        rowBlockDim,
        bsrRowPtr,
        bsrColInd,
        bsrValues,
        hipsparse::hipIndexTypeToHCCIndexType(bsrRowPtrType),
        hipsparse::hipIndexTypeToHCCIndexType(bsrColIndType),
        hipsparse::hipBaseToHCCBase(idxBase),
        hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateConstBsr(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     mb,
                                          int64_t                     nb,
                                          int64_t                     nnzb,
                                          int64_t                     rowBlockDim,
                                          int64_t                     colBlockDim,
                                          const void*                 bsrRowPtr,
                                          const void*                 bsrColInd,
                                          const void*                 bsrValues,
                                          hipsparseIndexType_t        bsrRowPtrType,
                                          hipsparseIndexType_t        bsrColIndType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType,
                                          hipsparseOrder_t            order)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(rowBlockDim != colBlockDim)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status
        = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_create_const_bsr_descr(
            spMatDescr[0]->get_const_spmat_descr_reference(),
            mb,
            nb,
            nnzb,
            (order == HIPSPARSE_ORDER_ROW)
                ? hipsparse::hipDirectionToHCCDirection(HIPSPARSE_DIRECTION_ROW)
                : hipsparse::hipDirectionToHCCDirection(HIPSPARSE_DIRECTION_COLUMN),
            rowBlockDim,
            bsrRowPtr,
            bsrColInd,
            bsrValues,
            hipsparse::hipIndexTypeToHCCIndexType(bsrRowPtrType),
            hipsparse::hipIndexTypeToHCCIndexType(bsrColIndType),
            hipsparse::hipBaseToHCCBase(idxBase),
            hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}
#endif /* HIPSPARSE_WITH_SPMV_BSR */

hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                nnz,
                                        void*                  cooInd,
                                        void*                  cooValues,
                                        hipsparseIndexType_t   cooIdxType,
                                        hipsparseIndexBase_t   idxBase,
                                        hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_coo_aos_descr(spMatDescr[0]->get_spmat_descr_reference(),
                                       rows,
                                       cols,
                                       nnz,
                                       cooInd,
                                       cooValues,
                                       hipsparse::hipIndexTypeToHCCIndexType(cooIdxType),
                                       hipsparse::hipBaseToHCCBase(idxBase),
                                       hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csr_descr_SWDEV_453599(rocsparse_spmat_descr* descr,
                                                         int64_t                rows,
                                                         int64_t                cols,
                                                         int64_t                nnz,
                                                         void*                  csrRowPtr,
                                                         void*                  csrColInd,
                                                         void*                  csrVal,
                                                         rocsparse_indextype    row_ptr_type,
                                                         rocsparse_indextype    col_ind_type,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_datatype     data_type);
#ifdef __cplusplus
}
#endif

hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  csrRowOffsets,
                                     void*                  csrColInd,
                                     void*                  csrValues,
                                     hipsparseIndexType_t   csrRowOffsetsType,
                                     hipsparseIndexType_t   csrColIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status
        = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_create_csr_descr_SWDEV_453599(
            spMatDescr[0]->get_spmat_descr_reference(),
            rows,
            cols,
            nnz,
            csrRowOffsets,
            csrColInd,
            csrValues,
            hipsparse::hipIndexTypeToHCCIndexType(csrRowOffsetsType),
            hipsparse::hipIndexTypeToHCCIndexType(csrColIndType),
            hipsparse::hipBaseToHCCBase(idxBase),
            hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateConstCsr(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 csrRowOffsets,
                                          const void*                 csrColInd,
                                          const void*                 csrValues,
                                          hipsparseIndexType_t        csrRowOffsetsType,
                                          hipsparseIndexType_t        csrColIndType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_csr_descr(spMatDescr[0]->get_const_spmat_descr_reference(),
                                         rows,
                                         cols,
                                         nnz,
                                         csrRowOffsets,
                                         csrColInd,
                                         csrValues,
                                         hipsparse::hipIndexTypeToHCCIndexType(csrRowOffsetsType),
                                         hipsparse::hipIndexTypeToHCCIndexType(csrColIndType),
                                         hipsparse::hipBaseToHCCBase(idxBase),
                                         hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cscColOffsets,
                                     void*                  cscRowInd,
                                     void*                  cscValues,
                                     hipsparseIndexType_t   cscColOffsetsType,
                                     hipsparseIndexType_t   cscRowIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_csc_descr(spMatDescr[0]->get_spmat_descr_reference(),
                                   rows,
                                   cols,
                                   nnz,
                                   cscColOffsets,
                                   cscRowInd,
                                   cscValues,
                                   hipsparse::hipIndexTypeToHCCIndexType(cscColOffsetsType),
                                   hipsparse::hipIndexTypeToHCCIndexType(cscRowIndType),
                                   hipsparse::hipBaseToHCCBase(idxBase),
                                   hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseCreateConstCsc(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 cscColOffsets,
                                          const void*                 cscRowInd,
                                          const void*                 cscValues,
                                          hipsparseIndexType_t        cscColOffsetsType,
                                          hipsparseIndexType_t        cscRowIndType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType)
{
    if(spMatDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr[0] = new hipsparseSpMatDescr_st();

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_csc_descr(spMatDescr[0]->get_const_spmat_descr_reference(),
                                         rows,
                                         cols,
                                         nnz,
                                         cscColOffsets,
                                         cscRowInd,
                                         cscValues,
                                         hipsparse::hipIndexTypeToHCCIndexType(cscColOffsetsType),
                                         hipsparse::hipIndexTypeToHCCIndexType(cscRowIndType),
                                         hipsparse::hipBaseToHCCBase(idxBase),
                                         hipsparse::hipDataTypeToHCCDataType(valueType)));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        delete spMatDescr[0];
    }

    return status;
}

hipsparseStatus_t hipsparseDestroySpMat(hipsparseConstSpMatDescr_t spMatDescr)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_destroy_spmat_descr(to_rocsparse_const_spmat_descr(spMatDescr)));
    delete spMatDescr;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseBlockedEllGet(const hipsparseSpMatDescr_t spMatDescr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    ellBlockSize,
                                         int64_t*                    ellCols,
                                         void**                      ellColInd,
                                         void**                      ellValue,
                                         hipsparseIndexType_t*       ellIdxType,
                                         hipsparseIndexBase_t*       idxBase,
                                         hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;
    rocsparse_direction  hcc_block_direction;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_bell_get(to_rocsparse_spmat_descr(spMatDescr),
                                                 rows,
                                                 cols,
                                                 &hcc_block_direction,
                                                 ellBlockSize,
                                                 ellCols,
                                                 ellColInd,
                                                 ellValue,
                                                 ellIdxType != nullptr ? &hcc_index_type : nullptr,
                                                 idxBase != nullptr ? &hcc_index_base : nullptr,
                                                 valueType != nullptr ? &hcc_data_type : nullptr));

    *ellIdxType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase    = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType  = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstBlockedEllGet(hipsparseConstSpMatDescr_t spMatDescr,
                                              int64_t*                   rows,
                                              int64_t*                   cols,
                                              int64_t*                   ellBlockSize,
                                              int64_t*                   ellCols,
                                              const void**               ellColInd,
                                              const void**               ellValue,
                                              hipsparseIndexType_t*      ellIdxType,
                                              hipsparseIndexBase_t*      idxBase,
                                              hipDataType*               valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;
    rocsparse_direction  hcc_block_direction;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_bell_get(to_rocsparse_const_spmat_descr(spMatDescr),
                                 rows,
                                 cols,
                                 &hcc_block_direction,
                                 ellBlockSize,
                                 ellCols,
                                 ellColInd,
                                 ellValue,
                                 ellIdxType != nullptr ? &hcc_index_type : nullptr,
                                 idxBase != nullptr ? &hcc_index_base : nullptr,
                                 valueType != nullptr ? &hcc_data_type : nullptr));

    *ellIdxType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase    = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType  = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cooRowInd,
                                  void**                      cooColInd,
                                  void**                      cooValues,
                                  hipsparseIndexType_t*       idxType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coo_get(to_rocsparse_spmat_descr(spMatDescr),
                                                rows,
                                                cols,
                                                nnz,
                                                cooRowInd,
                                                cooColInd,
                                                cooValues,
                                                idxType != nullptr ? &hcc_index_type : nullptr,
                                                idxBase != nullptr ? &hcc_index_base : nullptr,
                                                valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstCooGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               cooRowInd,
                                       const void**               cooColInd,
                                       const void**               cooValues,
                                       hipsparseIndexType_t*      idxType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_coo_get(to_rocsparse_const_spmat_descr(spMatDescr),
                                rows,
                                cols,
                                nnz,
                                cooRowInd,
                                cooColInd,
                                cooValues,
                                idxType != nullptr ? &hcc_index_type : nullptr,
                                idxBase != nullptr ? &hcc_index_base : nullptr,
                                valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    nnz,
                                     void**                      cooInd,
                                     void**                      cooValues,
                                     hipsparseIndexType_t*       idxType,
                                     hipsparseIndexBase_t*       idxBase,
                                     hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo_aos_get(to_rocsparse_spmat_descr(spMatDescr),
                              rows,
                              cols,
                              nnz,
                              cooInd,
                              cooValues,
                              idxType != nullptr ? &hcc_index_type : nullptr,
                              idxBase != nullptr ? &hcc_index_base : nullptr,
                              valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = hipsparse::HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCsrGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      csrRowOffsets,
                                  void**                      csrColInd,
                                  void**                      csrValues,
                                  hipsparseIndexType_t*       csrRowOffsetsType,
                                  hipsparseIndexType_t*       csrColIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    rocsparse_indextype  hcc_row_index_type;
    rocsparse_indextype  hcc_col_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr_get(to_rocsparse_spmat_descr(spMatDescr),
                          rows,
                          cols,
                          nnz,
                          csrRowOffsets,
                          csrColInd,
                          csrValues,
                          csrRowOffsetsType != nullptr ? &hcc_row_index_type : nullptr,
                          csrColIndType != nullptr ? &hcc_col_index_type : nullptr,
                          idxBase != nullptr ? &hcc_index_base : nullptr,
                          valueType != nullptr ? &hcc_data_type : nullptr));

    *csrRowOffsetsType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_row_index_type);
    *csrColIndType     = hipsparse::HCCIndexTypeToHIPIndexType(hcc_col_index_type);
    *idxBase           = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType         = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstCsrGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               csrRowOffsets,
                                       const void**               csrColInd,
                                       const void**               csrValues,
                                       hipsparseIndexType_t*      csrRowOffsetsType,
                                       hipsparseIndexType_t*      csrColIndType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType)
{
    rocsparse_indextype  hcc_row_index_type;
    rocsparse_indextype  hcc_col_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_csr_get(to_rocsparse_const_spmat_descr(spMatDescr),
                                rows,
                                cols,
                                nnz,
                                csrRowOffsets,
                                csrColInd,
                                csrValues,
                                csrRowOffsetsType != nullptr ? &hcc_row_index_type : nullptr,
                                csrColIndType != nullptr ? &hcc_col_index_type : nullptr,
                                idxBase != nullptr ? &hcc_index_base : nullptr,
                                valueType != nullptr ? &hcc_data_type : nullptr));

    *csrRowOffsetsType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_row_index_type);
    *csrColIndType     = hipsparse::HCCIndexTypeToHIPIndexType(hcc_col_index_type);
    *idxBase           = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType         = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues)
{

    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_csr_set_pointers(
        to_rocsparse_spmat_descr(spMatDescr), csrRowOffsets, csrColInd, csrValues));
}

hipsparseStatus_t hipsparseCscGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cscColOffsets,
                                  void**                      cscRowInd,
                                  void**                      cscValues,
                                  hipsparseIndexType_t*       cscColOffsetsType,
                                  hipsparseIndexType_t*       cscRowIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    rocsparse_indextype  hcc_col_index_type;
    rocsparse_indextype  hcc_row_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csc_get(to_rocsparse_spmat_descr(spMatDescr),
                          rows,
                          cols,
                          nnz,
                          cscColOffsets,
                          cscRowInd,
                          cscValues,
                          cscColOffsetsType != nullptr ? &hcc_col_index_type : nullptr,
                          cscRowIndType != nullptr ? &hcc_row_index_type : nullptr,
                          idxBase != nullptr ? &hcc_index_base : nullptr,
                          valueType != nullptr ? &hcc_data_type : nullptr));

    *cscColOffsetsType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_col_index_type);
    *cscRowIndType     = hipsparse::HCCIndexTypeToHIPIndexType(hcc_row_index_type);
    *idxBase           = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType         = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstCscGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               cscColOffsets,
                                       const void**               cscRowInd,
                                       const void**               cscValues,
                                       hipsparseIndexType_t*      cscColOffsetsType,
                                       hipsparseIndexType_t*      cscRowIndType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType)
{
    rocsparse_indextype  hcc_col_index_type;
    rocsparse_indextype  hcc_row_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_csc_get(to_rocsparse_const_spmat_descr(spMatDescr),
                                rows,
                                cols,
                                nnz,
                                cscColOffsets,
                                cscRowInd,
                                cscValues,
                                cscColOffsetsType != nullptr ? &hcc_col_index_type : nullptr,
                                cscRowIndType != nullptr ? &hcc_row_index_type : nullptr,
                                idxBase != nullptr ? &hcc_index_base : nullptr,
                                valueType != nullptr ? &hcc_data_type : nullptr));

    *cscColOffsetsType = hipsparse::HCCIndexTypeToHIPIndexType(hcc_col_index_type);
    *cscRowIndType     = hipsparse::HCCIndexTypeToHIPIndexType(hcc_row_index_type);
    *idxBase           = hipsparse::HCCBaseToHIPBase(hcc_index_base);
    *valueType         = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCscSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cscColOffsets,
                                          void*                 cscRowInd,
                                          void*                 cscValues)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_csc_set_pointers(
        to_rocsparse_spmat_descr(spMatDescr), cscColOffsets, cscRowInd, cscValues));
}

hipsparseStatus_t hipsparseCooSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cooRowInd,
                                          void*                 cooColInd,
                                          void*                 cooValues)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_coo_set_pointers(
        to_rocsparse_spmat_descr(spMatDescr), cooRowInd, cooColInd, cooValues));
}

hipsparseStatus_t hipsparseSpMatGetSize(hipsparseConstSpMatDescr_t spMatDescr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_size(to_rocsparse_const_spmat_descr(spMatDescr), rows, cols, nnz));
}

hipsparseStatus_t hipsparseSpMatGetFormat(hipsparseConstSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*         format)
{
    rocsparse_format hcc_format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(
        to_rocsparse_const_spmat_descr(spMatDescr), format != nullptr ? &hcc_format : nullptr));

    *format = hipsparse::HCCFormatToHIPFormat(hcc_format);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMatGetIndexBase(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*      idxBase)
{
    rocsparse_index_base hcc_index_base;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_spmat_get_index_base(to_rocsparse_const_spmat_descr(spMatDescr),
                                       idxBase != nullptr ? &hcc_index_base : nullptr));

    *idxBase = hipsparse::HCCBaseToHIPBase(hcc_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_values(to_rocsparse_spmat_descr(spMatDescr), values));
}

hipsparseStatus_t hipsparseConstSpMatGetValues(hipsparseConstSpMatDescr_t spMatDescr,
                                               const void**               values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_const_spmat_get_values(to_rocsparse_const_spmat_descr(spMatDescr), values));
}

hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_set_values(to_rocsparse_spmat_descr(spMatDescr), values));
}

hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseConstSpMatDescr_t spMatDescr,
                                                int*                       batchCount)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_strided_batch(to_rocsparse_const_spmat_descr(spMatDescr), batchCount));
}

hipsparseStatus_t hipsparseSpMatSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_set_strided_batch(to_rocsparse_spmat_descr(spMatDescr), batchCount));
}

hipsparseStatus_t hipsparseCooSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               batchStride)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_coo_set_strided_batch(
        to_rocsparse_spmat_descr(spMatDescr), batchCount, batchStride));
}

hipsparseStatus_t hipsparseCsrSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               offsetsBatchStride,
                                              int64_t               columnsValuesBatchStride)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_csr_set_strided_batch(to_rocsparse_spmat_descr(spMatDescr),
                                        batchCount,
                                        offsetsBatchStride,
                                        columnsValuesBatchStride));
}

hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseSpMatAttribute_t  attribute,
                                             void*                      data,
                                             size_t                     dataSize)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_attribute(to_rocsparse_const_spmat_descr(spMatDescr),
                                      (rocsparse_spmat_attribute)attribute,
                                      data,
                                      dataSize));
}

hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             const void*               data,
                                             size_t                    dataSize)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_set_attribute(to_rocsparse_spmat_descr(spMatDescr),
                                      (rocsparse_spmat_attribute)attribute,
                                      data,
                                      dataSize));
}

hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_dnvec_descr((rocsparse_dnvec_descr*)dnVecDescr,
                                     size,
                                     values,
                                     hipsparse::hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateConstDnVec(hipsparseConstDnVecDescr_t* dnVecDescr,
                                            int64_t                     size,
                                            const void*                 values,
                                            hipDataType                 valueType)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_dnvec_descr((rocsparse_const_dnvec_descr*)dnVecDescr,
                                           size,
                                           values,
                                           hipsparse::hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseDestroyDnVec(hipsparseConstDnVecDescr_t dnVecDescr)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_dnvec_descr((rocsparse_const_dnvec_descr)dnVecDescr));
}

hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType)
{
    rocsparse_datatype hcc_data_type;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnvec_get((const rocsparse_dnvec_descr)dnVecDescr,
                                                  size,
                                                  values,
                                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstDnVecGet(hipsparseConstDnVecDescr_t dnVecDescr,
                                         int64_t*                   size,
                                         const void**               values,
                                         hipDataType*               valueType)
{
    rocsparse_datatype hcc_data_type;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_dnvec_get((const rocsparse_const_dnvec_descr)dnVecDescr,
                                  size,
                                  values,
                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dnvec_get_values((const rocsparse_dnvec_descr)dnVecDescr, values));
}

hipsparseStatus_t hipsparseConstDnVecGetValues(hipsparseConstDnVecDescr_t dnVecDescr,
                                               const void**               values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_const_dnvec_get_values((const rocsparse_const_dnvec_descr)dnVecDescr, values));
}

hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dnvec_set_values((rocsparse_dnvec_descr)dnVecDescr, values));
}

hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_dnmat_descr((rocsparse_dnmat_descr*)dnMatDescr,
                                     rows,
                                     cols,
                                     ld,
                                     values,
                                     hipsparse::hipDataTypeToHCCDataType(valueType),
                                     hipsparse::hipOrderToHCCOrder(order)));
}

hipsparseStatus_t hipsparseCreateConstDnMat(hipsparseConstDnMatDescr_t* dnMatDescr,
                                            int64_t                     rows,
                                            int64_t                     cols,
                                            int64_t                     ld,
                                            const void*                 values,
                                            hipDataType                 valueType,
                                            hipsparseOrder_t            order)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_create_const_dnmat_descr((rocsparse_const_dnmat_descr*)dnMatDescr,
                                           rows,
                                           cols,
                                           ld,
                                           values,
                                           hipsparse::hipDataTypeToHCCDataType(valueType),
                                           hipsparse::hipOrderToHCCOrder(order)));
}

hipsparseStatus_t hipsparseDestroyDnMat(hipsparseConstDnMatDescr_t dnMatDescr)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_dnmat_descr((rocsparse_const_dnmat_descr)dnMatDescr));
}

hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order)
{
    rocsparse_datatype hcc_data_type;
    rocsparse_order    hcc_order;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnmat_get((const rocsparse_dnmat_descr)dnMatDescr,
                                                  rows,
                                                  cols,
                                                  ld,
                                                  values,
                                                  valueType != nullptr ? &hcc_data_type : nullptr,
                                                  order != nullptr ? &hcc_order : nullptr));

    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);
    *order     = hipsparse::HCCOrderToHIPOrder(hcc_order);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseConstDnMatGet(hipsparseConstDnMatDescr_t dnMatDescr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         hipDataType*               valueType,
                                         hipsparseOrder_t*          order)
{
    rocsparse_datatype hcc_data_type;
    rocsparse_order    hcc_order;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_const_dnmat_get((rocsparse_const_dnmat_descr)dnMatDescr,
                                  rows,
                                  cols,
                                  ld,
                                  values,
                                  valueType != nullptr ? &hcc_data_type : nullptr,
                                  order != nullptr ? &hcc_order : nullptr));

    *valueType = hipsparse::HCCDataTypeToHIPDataType(hcc_data_type);
    *order     = hipsparse::HCCOrderToHIPOrder(hcc_order);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dnmat_get_values((const rocsparse_dnmat_descr)dnMatDescr, values));
}

hipsparseStatus_t hipsparseConstDnMatGetValues(hipsparseConstDnMatDescr_t dnMatDescr,
                                               const void**               values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_const_dnmat_get_values((rocsparse_const_dnmat_descr)dnMatDescr, values));
}

hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dnmat_set_values((rocsparse_dnmat_descr)dnMatDescr, values));
}

hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseConstDnMatDescr_t dnMatDescr,
                                                int*                       batchCount,
                                                int64_t*                   batchStride)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dnmat_get_strided_batch(
        (rocsparse_const_dnmat_descr)dnMatDescr, batchCount, batchStride));
}

hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int                   batchCount,
                                                int64_t               batchStride)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dnmat_set_strided_batch(
        (const rocsparse_dnmat_descr)dnMatDescr, batchCount, batchStride));
}
