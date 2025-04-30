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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "utility.h"

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateSpVec((cusparseSpVecDescr_t*)spVecDescr,
                            size,
                            nnz,
                            indices,
                            values,
                            hipsparse::hipIndexTypeToCudaIndexType(idxType),
                            hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                            hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseCreateConstSpVec(hipsparseConstSpVecDescr_t* spVecDescr,
                                            int64_t                     size,
                                            int64_t                     nnz,
                                            const void*                 indices,
                                            const void*                 values,
                                            hipsparseIndexType_t        idxType,
                                            hipsparseIndexBase_t        idxBase,
                                            hipDataType                 valueType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstSpVec((cusparseConstSpVecDescr_t*)spVecDescr,
                                 size,
                                 nnz,
                                 indices,
                                 values,
                                 hipsparse::hipIndexTypeToCudaIndexType(idxType),
                                 hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                                 hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDestroySpVec(hipsparseConstSpVecDescr_t spVecDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroySpVec((cusparseConstSpVecDescr_t)spVecDescr));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroySpVec((cusparseSpVecDescr_t)spVecDescr));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType)
{
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpVecGet((const cusparseSpVecDescr_t)spVecDescr,
                                              size,
                                              nnz,
                                              indices,
                                              values,
                                              idxType != nullptr ? &cuda_index_type : nullptr,
                                              idxBase != nullptr ? &cuda_index_base : nullptr,
                                              valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstSpVecGet(hipsparseConstSpVecDescr_t spVecDescr,
                                         int64_t*                   size,
                                         int64_t*                   nnz,
                                         const void**               indices,
                                         const void**               values,
                                         hipsparseIndexType_t*      idxType,
                                         hipsparseIndexBase_t*      idxBase,
                                         hipDataType*               valueType)
{
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseConstSpVecGet((const cusparseConstSpVecDescr_t)spVecDescr,
                              size,
                              nnz,
                              indices,
                              values,
                              idxType != nullptr ? &cuda_index_type : nullptr,
                              idxBase != nullptr ? &cuda_index_base : nullptr,
                              valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpVecGetIndexBase(hipsparseConstSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*      idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpVecGetIndexBase(
        (const cusparseConstSpVecDescr_t)spVecDescr, (cusparseIndexBase_t*)idxBase));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpVecGetIndexBase(
        (const cusparseSpVecDescr_t)spVecDescr, (cusparseIndexBase_t*)idxBase));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpVecGetValues((const cusparseSpVecDescr_t)spVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstSpVecGetValues(hipsparseConstSpVecDescr_t spVecDescr,
                                               const void**               values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseConstSpVecGetValues((const cusparseConstSpVecDescr_t)spVecDescr, values));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpVecSetValues((const cusparseSpVecDescr_t)spVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCoo((cusparseSpMatDescr_t*)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          cooRowInd,
                          cooColInd,
                          cooValues,
                          hipsparse::hipIndexTypeToCudaIndexType(cooIdxType),
                          hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                          hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstCoo((cusparseConstSpMatDescr_t*)spMatDescr,
                               rows,
                               cols,
                               nnz,
                               cooRowInd,
                               cooColInd,
                               cooValues,
                               hipsparse::hipIndexTypeToCudaIndexType(cooIdxType),
                               hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                               hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCooAoS((cusparseSpMatDescr_t*)spMatDescr,
                             rows,
                             cols,
                             nnz,
                             cooInd,
                             cooValues,
                             hipsparse::hipIndexTypeToCudaIndexType(cooIdxType),
                             hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                             hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsr((cusparseSpMatDescr_t*)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          csrRowOffsets,
                          csrColInd,
                          csrValues,
                          hipsparse::hipIndexTypeToCudaIndexType(csrRowOffsetsType),
                          hipsparse::hipIndexTypeToCudaIndexType(csrColIndType),
                          hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                          hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12001)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstCsr((cusparseConstSpMatDescr_t*)spMatDescr,
                               rows,
                               cols,
                               nnz,
                               csrRowOffsets,
                               csrColInd,
                               csrValues,
                               hipsparse::hipIndexTypeToCudaIndexType(csrRowOffsetsType),
                               hipsparse::hipIndexTypeToCudaIndexType(csrColIndType),
                               hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                               hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 11020)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsc((cusparseSpMatDescr_t*)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          cscColOffsets,
                          cscRowInd,
                          cscValues,
                          hipsparse::hipIndexTypeToCudaIndexType(cscColOffsetsType),
                          hipsparse::hipIndexTypeToCudaIndexType(cscRowIndType),
                          hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                          hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstCsc((cusparseConstSpMatDescr_t*)spMatDescr,
                               rows,
                               cols,
                               nnz,
                               cscColOffsets,
                               cscRowInd,
                               cscValues,
                               hipsparse::hipIndexTypeToCudaIndexType(cscColOffsetsType),
                               hipsparse::hipIndexTypeToCudaIndexType(cscRowIndType),
                               hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                               hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 11021)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateBlockedEll((cusparseSpMatDescr_t*)spMatDescr,
                                 rows,
                                 cols,
                                 ellBlockSize,
                                 ellCols,
                                 ellColInd,
                                 ellValue,
                                 hipsparse::hipIndexTypeToCudaIndexType(ellIdxType),
                                 hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                                 hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstBlockedEll((cusparseConstSpMatDescr_t*)spMatDescr,
                                      rows,
                                      cols,
                                      ellBlockSize,
                                      ellCols,
                                      ellColInd,
                                      ellValue,
                                      hipsparse::hipIndexTypeToCudaIndexType(ellIdxType),
                                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                                      hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDestroySpMat(hipsparseConstSpMatDescr_t spMatDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroySpMat((cusparseConstSpMatDescr_t)spMatDescr));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroySpMat((cusparseSpMatDescr_t)spMatDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
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
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseCooGet((const cusparseSpMatDescr_t)spMatDescr,
                                            rows,
                                            cols,
                                            nnz,
                                            cooRowInd,
                                            cooColInd,
                                            cooValues,
                                            idxType != nullptr ? &cuda_index_type : nullptr,
                                            idxBase != nullptr ? &cuda_index_base : nullptr,
                                            valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
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
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseConstCooGet((const cusparseConstSpMatDescr_t)spMatDescr,
                                                 rows,
                                                 cols,
                                                 nnz,
                                                 cooRowInd,
                                                 cooColInd,
                                                 cooValues,
                                                 idxType != nullptr ? &cuda_index_type : nullptr,
                                                 idxBase != nullptr ? &cuda_index_base : nullptr,
                                                 valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
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
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseCooAoSGet((const cusparseSpMatDescr_t)spMatDescr,
                                               rows,
                                               cols,
                                               nnz,
                                               cooInd,
                                               cooValues,
                                               idxType != nullptr ? &cuda_index_type : nullptr,
                                               idxBase != nullptr ? &cuda_index_base : nullptr,
                                               valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
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
    cusparseIndexType_t cuda_row_index_type;
    cusparseIndexType_t cuda_col_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseCsrGet((const cusparseSpMatDescr_t)spMatDescr,
                       rows,
                       cols,
                       nnz,
                       csrRowOffsets,
                       csrColInd,
                       csrValues,
                       csrRowOffsetsType != nullptr ? &cuda_row_index_type : nullptr,
                       csrColIndType != nullptr ? &cuda_col_index_type : nullptr,
                       idxBase != nullptr ? &cuda_index_base : nullptr,
                       valueType != nullptr ? &cuda_data_type : nullptr));

    *csrRowOffsetsType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_row_index_type);
    *csrColIndType     = hipsparse::CudaIndexTypeToHIPIndexType(cuda_col_index_type);
    *idxBase           = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType         = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
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
    cusparseIndexType_t cuda_row_index_type;
    cusparseIndexType_t cuda_col_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseConstCsrGet((const cusparseConstSpMatDescr_t)spMatDescr,
                            rows,
                            cols,
                            nnz,
                            csrRowOffsets,
                            csrColInd,
                            csrValues,
                            csrRowOffsetsType != nullptr ? &cuda_row_index_type : nullptr,
                            csrColIndType != nullptr ? &cuda_col_index_type : nullptr,
                            idxBase != nullptr ? &cuda_index_base : nullptr,
                            valueType != nullptr ? &cuda_data_type : nullptr));

    *csrRowOffsetsType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_row_index_type);
    *csrColIndType     = hipsparse::CudaIndexTypeToHIPIndexType(cuda_col_index_type);
    *idxBase           = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType         = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12001)
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
    cusparseIndexType_t cuda_col_index_type;
    cusparseIndexType_t cuda_row_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseCscGet((const cusparseSpMatDescr_t)spMatDescr,
                       rows,
                       cols,
                       nnz,
                       cscColOffsets,
                       cscRowInd,
                       cscValues,
                       cscColOffsetsType != nullptr ? &cuda_col_index_type : nullptr,
                       cscRowIndType != nullptr ? &cuda_row_index_type : nullptr,
                       idxBase != nullptr ? &cuda_index_base : nullptr,
                       valueType != nullptr ? &cuda_data_type : nullptr));

    *cscColOffsetsType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_col_index_type);
    *cscRowIndType     = hipsparse::CudaIndexTypeToHIPIndexType(cuda_row_index_type);
    *idxBase           = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType         = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12001)
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
    cusparseIndexType_t cuda_row_index_type;
    cusparseIndexType_t cuda_col_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseConstCscGet((const cusparseConstSpMatDescr_t)spMatDescr,
                            rows,
                            cols,
                            nnz,
                            cscColOffsets,
                            cscRowInd,
                            cscValues,
                            cscColOffsetsType != nullptr ? &cuda_row_index_type : nullptr,
                            cscRowIndType != nullptr ? &cuda_col_index_type : nullptr,
                            idxBase != nullptr ? &cuda_index_base : nullptr,
                            valueType != nullptr ? &cuda_data_type : nullptr));

    *cscColOffsetsType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_row_index_type);
    *cscRowIndType     = hipsparse::CudaIndexTypeToHIPIndexType(cuda_col_index_type);
    *idxBase           = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType         = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 11021)
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
    // As of cusparse 11.4.1, this routine does not actually exist as a symbol in the cusparse
    // library (the documentation indicates that it should exist starting at cusparse 11.2.1).
#if(CUDART_VERSION >= 11070)
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseBlockedEllGet((cusparseSpMatDescr_t)spMatDescr,
                              rows,
                              cols,
                              ellBlockSize,
                              ellCols,
                              ellColInd,
                              ellValue,
                              ellIdxType != nullptr ? &cuda_index_type : nullptr,
                              idxBase != nullptr ? &cuda_index_base : nullptr,
                              valueType != nullptr ? &cuda_data_type : nullptr));

    *ellIdxType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase    = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType  = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
#else
    return HIPSPARSE_STATUS_NOT_SUPPORTED;
#endif
}
#endif

#if(CUDART_VERSION >= 12000)
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
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseConstBlockedEllGet((cusparseConstSpMatDescr_t)spMatDescr,
                                   rows,
                                   cols,
                                   ellBlockSize,
                                   ellCols,
                                   ellColInd,
                                   ellValue,
                                   ellIdxType != nullptr ? &cuda_index_type : nullptr,
                                   idxBase != nullptr ? &cuda_index_base : nullptr,
                                   valueType != nullptr ? &cuda_data_type : nullptr));

    *ellIdxType = hipsparse::CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase    = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType  = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCsrSetPointers(
        (cusparseSpMatDescr_t)spMatDescr, csrRowOffsets, csrColInd, csrValues));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseConstSpMatDescr_t spMatDescr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetSize((cusparseConstSpMatDescr_t)spMatDescr, rows, cols, nnz));
}
#elif(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                        int64_t*              rows,
                                        int64_t*              cols,
                                        int64_t*              nnz)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetSize((cusparseSpMatDescr_t)spMatDescr, rows, cols, nnz));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMatGetFormat(hipsparseConstSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*         format)
{
    cusparseFormat_t cuda_format;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpMatGetFormat((const cusparseConstSpMatDescr_t)spMatDescr,
                                                    format != nullptr ? &cuda_format : nullptr));

    *format = hipsparse::CudaFormatToHIPFormat(cuda_format);

    return HIPSPARSE_STATUS_SUCCESS;
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*          format)
{
    cusparseFormat_t cuda_format;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpMatGetFormat((const cusparseSpMatDescr_t)spMatDescr,
                                                    format != nullptr ? &cuda_format : nullptr));

    *format = hipsparse::CudaFormatToHIPFormat(cuda_format);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMatGetIndexBase(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*      idxBase)
{
    cusparseIndexBase_t cuda_index_base;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseSpMatGetIndexBase((const cusparseConstSpMatDescr_t)spMatDescr,
                                  idxBase != nullptr ? &cuda_index_base : nullptr));

    *idxBase = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    cusparseIndexBase_t cuda_index_base;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpMatGetIndexBase(
        (const cusparseSpMatDescr_t)spMatDescr, idxBase != nullptr ? &cuda_index_base : nullptr));

    *idxBase = hipsparse::CudaIndexBaseToHIPIndexBase(cuda_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetValues((cusparseSpMatDescr_t)spMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstSpMatGetValues(hipsparseConstSpMatDescr_t spMatDescr,
                                               const void**               values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseConstSpMatGetValues((cusparseConstSpMatDescr_t)spMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatSetValues((cusparseSpMatDescr_t)spMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseConstSpMatDescr_t spMatDescr,
                                                int*                       batchCount)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetStridedBatch((cusparseConstSpMatDescr_t)spMatDescr, batchCount));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int* batchCount)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount));
}
#endif

#if(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
hipsparseStatus_t hipsparseSpMatSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatSetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseCooSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               batchStride)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCooSetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount, batchStride));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseCsrSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               offsetsBatchStride,
                                              int64_t               columnsValuesBatchStride)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCsrSetStridedBatch((cusparseSpMatDescr_t)spMatDescr,
                                   batchCount,
                                   offsetsBatchStride,
                                   columnsValuesBatchStride));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseSpMatAttribute_t  attribute,
                                             void*                      data,
                                             size_t                     dataSize)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetAttribute((cusparseConstSpMatDescr_t)spMatDescr,
                                  (cusparseSpMatAttribute_t)attribute,
                                  data,
                                  dataSize));
}
#elif(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             void*                     data,
                                             size_t                    dataSize)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpMatGetAttribute(
        (cusparseSpMatDescr_t)spMatDescr, (cusparseSpMatAttribute_t)attribute, data, dataSize));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             const void*               data,
                                             size_t                    dataSize)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatSetAttribute((cusparseSpMatDescr_t)spMatDescr,
                                  (cusparseSpMatAttribute_t)attribute,
                                  const_cast<void*>(data),
                                  dataSize));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateDnVec((cusparseDnVecDescr_t*)dnVecDescr,
                            size,
                            values,
                            hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseCreateConstDnVec(hipsparseConstDnVecDescr_t* dnVecDescr,
                                            int64_t                     size,
                                            const void*                 values,
                                            hipDataType                 valueType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstDnVec((cusparseConstDnVecDescr_t*)dnVecDescr,
                                 size,
                                 values,
                                 hipsparse::hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseConstDnVecDescr_t dnVecDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyDnVec((cusparseConstDnVecDescr_t)dnVecDescr));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyDnVec((cusparseDnVecDescr_t)dnVecDescr));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType)
{
    cudaDataType cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseDnVecGet((const cusparseDnVecDescr_t)dnVecDescr,
                                              size,
                                              values,
                                              valueType != nullptr ? &cuda_data_type : nullptr));

    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstDnVecGet(hipsparseConstDnVecDescr_t dnVecDescr,
                                         int64_t*                   size,
                                         const void**               values,
                                         hipDataType*               valueType)
{
    cudaDataType cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseConstDnVecGet((const cusparseConstDnVecDescr_t)dnVecDescr,
                              size,
                              values,
                              valueType != nullptr ? &cuda_data_type : nullptr));

    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnVecGetValues((const cusparseDnVecDescr_t)dnVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 12001)
hipsparseStatus_t hipsparseConstDnVecGetValues(hipsparseConstDnVecDescr_t dnVecDescr,
                                               const void**               values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseConstDnVecGetValues((const cusparseConstDnVecDescr_t)dnVecDescr, values));
}
#endif

#if(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnVecSetValues((cusparseDnVecDescr_t)dnVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateDnMat((cusparseDnMatDescr_t*)dnMatDescr,
                            rows,
                            cols,
                            ld,
                            values,
                            hipsparse::hipDataTypeToCudaDataType(valueType),
                            hipsparse::hipOrderToCudaOrder(order)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseCreateConstDnMat(hipsparseConstDnMatDescr_t* dnMatDescr,
                                            int64_t                     rows,
                                            int64_t                     cols,
                                            int64_t                     ld,
                                            const void*                 values,
                                            hipDataType                 valueType,
                                            hipsparseOrder_t            order)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCreateConstDnMat((cusparseConstDnMatDescr_t*)dnMatDescr,
                                 rows,
                                 cols,
                                 ld,
                                 values,
                                 hipsparse::hipDataTypeToCudaDataType(valueType),
                                 hipsparse::hipOrderToCudaOrder(order)));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseConstDnMatDescr_t dnMatDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyDnMat((cusparseConstDnMatDescr_t)dnMatDescr));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyDnMat((cusparseDnMatDescr_t)dnMatDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order)
{
    cudaDataType    cuda_data_type;
    cusparseOrder_t cusparse_order;
    RETURN_IF_CUSPARSE_ERROR(cusparseDnMatGet((const cusparseDnMatDescr_t)dnMatDescr,
                                              rows,
                                              cols,
                                              ld,
                                              values,
                                              valueType != nullptr ? &cuda_data_type : nullptr,
                                              order != nullptr ? &cusparse_order : nullptr));

    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);
    *order     = hipsparse::CudaOrderToHIPOrder(cusparse_order);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstDnMatGet(hipsparseConstDnMatDescr_t dnMatDescr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         hipDataType*               valueType,
                                         hipsparseOrder_t*          order)
{
    cudaDataType    cuda_data_type;
    cusparseOrder_t cusparse_order;
    RETURN_IF_CUSPARSE_ERROR(cusparseConstDnMatGet((const cusparseConstDnMatDescr_t)dnMatDescr,
                                                   rows,
                                                   cols,
                                                   ld,
                                                   values,
                                                   valueType != nullptr ? &cuda_data_type : nullptr,
                                                   order != nullptr ? &cusparse_order : nullptr));

    *valueType = hipsparse::CudaDataTypeToHIPDataType(cuda_data_type);
    *order     = hipsparse::CudaOrderToHIPOrder(cusparse_order);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatGetValues((const cusparseDnMatDescr_t)dnMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseConstDnMatGetValues(hipsparseConstDnMatDescr_t dnMatDescr,
                                               const void**               values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseConstDnMatGetValues((const cusparseConstDnMatDescr_t)dnMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatSetValues((cusparseDnMatDescr_t)dnMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseConstDnMatDescr_t dnMatDescr,
                                                int*                       batchCount,
                                                int64_t*                   batchStride)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDnMatGetStridedBatch(
        (cusparseConstDnMatDescr_t)dnMatDescr, batchCount, batchStride));
}
#elif(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int*                  batchCount,
                                                int64_t*              batchStride)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatGetStridedBatch((cusparseDnMatDescr_t)dnMatDescr, batchCount, batchStride));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int                   batchCount,
                                                int64_t               batchStride)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatSetStridedBatch((cusparseDnMatDescr_t)dnMatDescr, batchCount, batchStride));
}
#endif
