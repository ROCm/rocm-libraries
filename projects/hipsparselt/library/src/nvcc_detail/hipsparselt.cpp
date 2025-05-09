/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "exceptions.hpp"
#include <hipsparselt/hipsparselt.h>

#include <cusparseLt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_CUSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                   \
        cusparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != CUSPARSE_STATUS_SUCCESS)             \
        {                                                               \
            return hipCUSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK);  \
        }                                                               \
    }

hipsparseStatus_t hipCUSPARSEStatusToHIPStatus(cusparseStatus_t cuStatus)
{

#if(CUDART_VERSION >= 11003)
    switch(cuStatus)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return HIPSPARSE_STATUS_SUCCESS;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return HIPSPARSE_STATUS_MAPPING_ERROR;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return HIPSPARSE_STATUS_EXECUTION_FAILED;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    case CUSPARSE_STATUS_ZERO_PIVOT:
        return HIPSPARSE_STATUS_ZERO_PIVOT;
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
    default:
        throw "Non existent cusparseStatus_t";
    }
#elif(CUDART_VERSION >= 10010)
    switch(cuStatus)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return HIPSPARSE_STATUS_SUCCESS;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return HIPSPARSE_STATUS_MAPPING_ERROR;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return HIPSPARSE_STATUS_EXECUTION_FAILED;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    case CUSPARSE_STATUS_ZERO_PIVOT:
        return HIPSPARSE_STATUS_ZERO_PIVOT;
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    default:
        throw "Non existent cusparseStatus_t";
    }
#else
#error "CUDART_VERSION is not supported"
#endif
}

cudaDataType HIPDatatypeToCuSparseLtDatatype(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return CUDA_R_32F;

    case HIP_R_16BF:
        return CUDA_R_16BF;

    case HIP_R_16F:
        return CUDA_R_16F;

    case HIP_R_8I:
        return CUDA_R_8I;

    case HIP_R_8F_E4M3:
        return CUDA_R_8F_E4M3;

    case HIP_R_8F_E5M2:
        return CUDA_R_8F_E5M2;

    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipDataType CuSparseLtDatatypeToHIPDatatype(cudaDataType type)
{
    switch(type)
    {
    case CUDA_R_32F:
        return HIP_R_32F;

    case CUDA_R_16BF:
        return HIP_R_16BF;

    case CUDA_R_16F:
        return HIP_R_16F;

    case CUDA_R_8I:
        return HIP_R_8I;

    case CUDA_R_8F_E4M3:
        return HIP_R_8F_E4M3;

    case CUDA_R_8F_E5M2:
        return HIP_R_8F_E5M2;

    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseComputeType HIPComputetypeToCuSparseComputetype(hipsparseLtComputetype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_COMPUTE_16F:
        return CUSPARSE_COMPUTE_16F;
    case HIPSPARSELT_COMPUTE_32I:
        return CUSPARSE_COMPUTE_32I;
    case HIPSPARSELT_COMPUTE_32F:
        return CUSPARSE_COMPUTE_32F;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtComputetype_t CuSparseLtComputetypeToHIPComputetype(cusparseComputeType type)
{
    switch(type)
    {
    case CUSPARSE_COMPUTE_16F:
        return HIPSPARSELT_COMPUTE_16F;
    case CUSPARSE_COMPUTE_32I:
        return HIPSPARSELT_COMPUTE_32I;
    case CUSPARSE_COMPUTE_32F:
        return HIPSPARSELT_COMPUTE_32F;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseOperation_t hipOperationToCudaOperation(hipsparseOperation_t op)
{
    switch(op)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return CUSPARSE_OPERATION_TRANSPOSE;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw "Non existent hipsparseOperation_t";
    }
}

hipsparseOperation_t CudaOperationToHIPOperation(cusparseOperation_t op)
{
    switch(op)
    {
    case CUSPARSE_OPERATION_NON_TRANSPOSE:
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case CUSPARSE_OPERATION_TRANSPOSE:
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw "Non existent cusparseOperation_t";
    }
}

cusparseOrder_t hipOrderToCudaOrder(hipsparseOrder_t op)
{
    switch(op)
    {
    case HIPSPARSE_ORDER_ROW:
        return CUSPARSE_ORDER_ROW;
    case HIPSPARSE_ORDER_COL:
        return CUSPARSE_ORDER_COL;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseOrder_t CudaOrderToHIPOrder(cusparseOrder_t op)
{
    switch(op)
    {
    case CUSPARSE_ORDER_ROW:
        return HIPSPARSE_ORDER_ROW;
    case CUSPARSE_ORDER_COL:
        return HIPSPARSE_ORDER_COL;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtSparsity_t HIPSparsityToCuSparseLtSparsity(hipsparseLtSparsity_t sparsity)
{
    switch(sparsity)
    {
    case HIPSPARSELT_SPARSITY_50_PERCENT:
        return CUSPARSELT_SPARSITY_50_PERCENT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtSparsity_t CuSparseLtSparsityToHIPSparsity(cusparseLtSparsity_t sparsity)
{
    switch(sparsity)
    {
    case CUSPARSELT_SPARSITY_50_PERCENT:
        return HIPSPARSELT_SPARSITY_50_PERCENT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtMatmulDescAttribute_t
    HIPMatmulDescAttributeToCuSparseLtMatmulDescAttribute(hipsparseLtMatmulDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU:
        return CUSPARSELT_MATMUL_ACTIVATION_RELU;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
        return CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
        return CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD;
    case HIPSPARSELT_MATMUL_ACTIVATION_GELU:
        return CUSPARSELT_MATMUL_ACTIVATION_GELU;
    case HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
        return CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING;
    case HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
        return CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING;
    case HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING:
        return CUSPARSELT_MATMUL_BETA_VECTOR_SCALING;
    case HIPSPARSELT_MATMUL_BIAS_STRIDE:
        return CUSPARSELT_MATMUL_BIAS_STRIDE;
    case HIPSPARSELT_MATMUL_BIAS_POINTER:
        return CUSPARSELT_MATMUL_BIAS_POINTER;
    case HIPSPARSELT_MATMUL_SPARSE_MAT_POINTER:
        return CUSPARSELT_MATMUL_SPARSE_MAT_POINTER;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulDescAttribute_t
    CuSparseLtMatmulDescAttributeToHIPMatmulDescAttribute(cusparseLtMatmulDescAttribute_t attr)
{
    switch(attr)
    {
    case CUSPARSELT_MATMUL_ACTIVATION_RELU:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU;
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND;
    case CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD;
    case CUSPARSELT_MATMUL_ACTIVATION_GELU:
        return HIPSPARSELT_MATMUL_ACTIVATION_GELU;
    case CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
        return HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING;
    case CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
        return HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING;
    case CUSPARSELT_MATMUL_BETA_VECTOR_SCALING:
        return HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING;
    case CUSPARSELT_MATMUL_BIAS_STRIDE:
        return HIPSPARSELT_MATMUL_BIAS_STRIDE;
    case CUSPARSELT_MATMUL_BIAS_POINTER:
        return HIPSPARSELT_MATMUL_BIAS_POINTER;
    case CUSPARSELT_MATMUL_SPARSE_MAT_POINTER:
        return HIPSPARSELT_MATMUL_SPARSE_MAT_POINTER;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatDescAttribute_t
    CuSparseLtMatDescAttributeToHIPMatDescAttribute(cusparseLtMatDescAttribute_t attr)
{
    switch(attr)
    {
    case CUSPARSELT_MAT_NUM_BATCHES:
        return HIPSPARSELT_MAT_NUM_BATCHES;
    case CUSPARSELT_MAT_BATCH_STRIDE:
        return HIPSPARSELT_MAT_BATCH_STRIDE;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtMatDescAttribute_t
    HIPMatDescAttributeToCuSparseLtMatDescAttribute(hipsparseLtMatDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MAT_NUM_BATCHES:
        return CUSPARSELT_MAT_NUM_BATCHES;
    case HIPSPARSELT_MAT_BATCH_STRIDE:
        return CUSPARSELT_MAT_BATCH_STRIDE;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtMatmulAlg_t HIPMatmulAlgToCuSparseLtMatmulAlg(hipsparseLtMatmulAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_DEFAULT:
        return CUSPARSELT_MATMUL_ALG_DEFAULT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulAlg_t CuSparseLtMatmulAlgToHIPMatmulAlg(cusparseLtMatmulAlg_t alg)
{
    switch(alg)
    {
    case CUSPARSELT_MATMUL_ALG_DEFAULT:
        return HIPSPARSELT_MATMUL_ALG_DEFAULT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtMatmulAlgAttribute_t
    HIPMatmulAlgAttributeToCuSparseLtAlgAttribute(hipsparseLtMatmulAlgAttribute_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_CONFIG_ID:
        return CUSPARSELT_MATMUL_ALG_CONFIG_ID;
    case HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
        return CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
    case HIPSPARSELT_MATMUL_SEARCH_ITERATIONS:
        return CUSPARSELT_MATMUL_SEARCH_ITERATIONS;
    case HIPSPARSELT_MATMUL_SPLIT_K:
        return CUSPARSELT_MATMUL_SPLIT_K;
    case HIPSPARSELT_MATMUL_SPLIT_K_MODE:
        return CUSPARSELT_MATMUL_SPLIT_K_MODE;
    case HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS:
        return CUSPARSELT_MATMUL_SPLIT_K_BUFFERS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulAlgAttribute_t
    CuSparseLtAlgAttributeToHIPMatmulAlgAttribute(cusparseLtMatmulAlgAttribute_t alg)
{
    switch(alg)
    {
    case CUSPARSELT_MATMUL_ALG_CONFIG_ID:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_ID;
    case CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
    case CUSPARSELT_MATMUL_SEARCH_ITERATIONS:
        return HIPSPARSELT_MATMUL_SEARCH_ITERATIONS;
    case CUSPARSELT_MATMUL_SPLIT_K:
        return HIPSPARSELT_MATMUL_SPLIT_K;
    case CUSPARSELT_MATMUL_SPLIT_K_MODE:
        return HIPSPARSELT_MATMUL_SPLIT_K_MODE;
    case CUSPARSELT_MATMUL_SPLIT_K_BUFFERS:
        return HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtPruneAlg_t HIPPruneAlgToCuSparseLtPruneAlg(hipsparseLtPruneAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_PRUNE_SPMMA_TILE:
        return CUSPARSELT_PRUNE_SPMMA_TILE;
    case HIPSPARSELT_PRUNE_SPMMA_STRIP:
        return CUSPARSELT_PRUNE_SPMMA_STRIP;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtPruneAlg_t CuSparseLtPruneAlgToHIPPruneAlg(cusparseLtPruneAlg_t alg)
{
    switch(alg)
    {
    case CUSPARSELT_PRUNE_SPMMA_TILE:
        return HIPSPARSELT_PRUNE_SPMMA_TILE;
    case CUSPARSELT_PRUNE_SPMMA_STRIP:
        return HIPSPARSELT_PRUNE_SPMMA_STRIP;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

cusparseLtSplitKMode_t HIPSplitKModeToCuSparseLtSplitKMode(hipsparseLtSplitKMode_t mode)
{
    switch(mode)
    {
    case HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL:
        return CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
    case HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS:
        return CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtSplitKMode_t CuSparseLtSplitKModeToHIPSplitKMode(cusparseLtSplitKMode_t mode)
{
    switch(mode)
    {
    case CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL:
        return HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
    case CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS:
        return HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

libraryPropertyType HIPLibraryPropertyTypeToCuLibraryPoropertyType(hipLibraryPropertyType property)
{
    switch(property)
    {
    case HIP_LIBRARY_MAJOR_VERSION:
        return MAJOR_VERSION;
    case HIP_LIBRARY_MINOR_VERSION:
        return MINOR_VERSION;
    case HIP_LIBRARY_PATCH_LEVEL:
        return PATCH_LEVEL;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipLibraryPropertyType CuLibraryPropertyTypeToHIPLibraryPoropertyType(libraryPropertyType property)
{
    switch(property)
    {
    case MAJOR_VERSION:
        return HIP_LIBRARY_MAJOR_VERSION;
    case MINOR_VERSION:
        return HIP_LIBRARY_MINOR_VERSION;
    case PATCH_LEVEL:
        return HIP_LIBRARY_PATCH_LEVEL;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle)
{
    char* log_env;
    if((log_env = getenv("HIPSPARSELT_LOG_LEVEL")) != NULL)
    {
        setenv("CUSPARSELT_LOG_LEVEL", log_env, 0);
    }
    if((log_env = getenv("HIPSPARSELT_LOG_MASK")) != NULL)
    {
        int mask = strtol(log_env, nullptr, 0);
        char mask_str[11];
        snprintf(mask_str, 11, "%d",mask);
        setenv("CUSPARSELT_LOG_MASK", mask_str, 0);
    }
    if((log_env = getenv("HIPSPARSELT_LOG_FILE")) != NULL)
    {
        setenv("CUSPARSELT_LOG_FILE", log_env, 0);
    }

    return hipCUSPARSEStatusToHIPStatus(cusparseLtInit((cusparseLtHandle_t*)handle));
}

hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtDestroy((const cusparseLtHandle_t*)handle));
}

hipsparseStatus_t hipsparseLtGetVersion(const hipsparseLtHandle_t* handle, int* version)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtGetVersion((const cusparseLtHandle_t*)handle, version));
}

hipsparseStatus_t hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int* value)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtGetProperty(HIPLibraryPropertyTypeToCuLibraryPoropertyType(propertyType), value));
}

/* matrix descriptor */
// dense matrix
hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                 hipsparseLtMatDescriptor_t* matDescr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 int64_t                     ld,
                                                 uint32_t                    alignment,
                                                 hipDataType                 valueType,
                                                 hipsparseOrder_t            order)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtDenseDescriptorInit((const cusparseLtHandle_t*)handle,
                                      (cusparseLtMatDescriptor_t*)matDescr,
                                      rows,
                                      cols,
                                      ld,
                                      alignment,
                                      HIPDatatypeToCuSparseLtDatatype(valueType),
                                      hipOrderToCudaOrder(order)));
}

// structured matrix
hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                      hipsparseLtMatDescriptor_t* matDescr,
                                                      int64_t                     rows,
                                                      int64_t                     cols,
                                                      int64_t                     ld,
                                                      uint32_t                    alignment,
                                                      hipDataType                 valueType,
                                                      hipsparseOrder_t            order,
                                                      hipsparseLtSparsity_t       sparsity)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtStructuredDescriptorInit((const cusparseLtHandle_t*)handle,
                                           (cusparseLtMatDescriptor_t*)matDescr,
                                           rows,
                                           cols,
                                           ld,
                                           alignment,
                                           HIPDatatypeToCuSparseLtDatatype(valueType),
                                           hipOrderToCudaOrder(order),
                                           HIPSparsityToCuSparseLtSparsity(sparsity)));
}

hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatDescriptorDestroy((const cusparseLtMatDescriptor_t*)matDescr));
}

hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                 hipsparseLtMatDescriptor_t*   matmulDescr,
                                                 hipsparseLtMatDescAttribute_t matAttribute,
                                                 const void*                   data,
                                                 size_t                        dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatDescSetAttribute((const cusparseLtHandle_t*)handle,
                                      (cusparseLtMatDescriptor_t*)matmulDescr,
                                      HIPMatDescAttributeToCuSparseLtMatDescAttribute(matAttribute),
                                      data,
                                      dataSize));
}

hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                 const hipsparseLtMatDescriptor_t* matmulDescr,
                                                 hipsparseLtMatDescAttribute_t     matAttribute,
                                                 void*                             data,
                                                 size_t                            dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatDescGetAttribute((const cusparseLtHandle_t*)handle,
                                      (const cusparseLtMatDescriptor_t*)matmulDescr,
                                      HIPMatDescAttributeToCuSparseLtMatDescAttribute(matAttribute),
                                      data,
                                      dataSize));
}

/* matmul descriptor */
hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        handle,
                                                  hipsparseLtMatmulDescriptor_t*    matmulDescr,
                                                  hipsparseOperation_t              opA,
                                                  hipsparseOperation_t              opB,
                                                  const hipsparseLtMatDescriptor_t* matA,
                                                  const hipsparseLtMatDescriptor_t* matB,
                                                  const hipsparseLtMatDescriptor_t* matC,
                                                  const hipsparseLtMatDescriptor_t* matD,
                                                  hipsparseLtComputetype_t          computeType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulDescriptorInit((const cusparseLtHandle_t*)handle,
                                       (cusparseLtMatmulDescriptor_t*)matmulDescr,
                                       hipOperationToCudaOperation(opA),
                                       hipOperationToCudaOperation(opB),
                                       (const cusparseLtMatDescriptor_t*)matA,
                                       (const cusparseLtMatDescriptor_t*)matB,
                                       (const cusparseLtMatDescriptor_t*)matC,
                                       (const cusparseLtMatDescriptor_t*)matD,
                                       HIPComputetypeToCuSparseComputetype(computeType)));
}

hipsparseStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtMatmulDescSetAttribute(
        (const cusparseLtHandle_t*)handle,
        (cusparseLtMatmulDescriptor_t*)matmulDescr,
        HIPMatmulDescAttributeToCuSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
}

hipsparseStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtMatmulDescGetAttribute(
        (const cusparseLtHandle_t*)handle,
        (const cusparseLtMatmulDescriptor_t*)matmulDescr,
        HIPMatmulDescAttributeToCuSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
}

/* algorithm selection */
hipsparseStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulAlgSelectionInit((const cusparseLtHandle_t*)handle,
                                         (cusparseLtMatmulAlgSelection_t*)algSelection,
                                         (const cusparseLtMatmulDescriptor_t*)matmulDescr,
                                         HIPMatmulAlgToCuSparseLtMatmulAlg(alg)));
}

hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                   hipsparseLtMatmulAlgSelection_t* algSelection,
                                                   hipsparseLtMatmulAlgAttribute_t  attribute,
                                                   const void*                      data,
                                                   size_t                           dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulAlgSetAttribute((const cusparseLtHandle_t*)handle,
                                        (cusparseLtMatmulAlgSelection_t*)algSelection,
                                        HIPMatmulAlgAttributeToCuSparseLtAlgAttribute(attribute),
                                        data,
                                        dataSize));
}

hipsparseStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulAlgGetAttribute((const cusparseLtHandle_t*)handle,
                                        (const cusparseLtMatmulAlgSelection_t*)algSelection,
                                        HIPMatmulAlgAttributeToCuSparseLtAlgAttribute(attribute),
                                        data,
                                        dataSize));
}

/* matmul plan */
hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                const hipsparseLtMatmulPlan_t* plan,
                                                size_t*                        workspaceSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtMatmulGetWorkspace(
        (const cusparseLtHandle_t*)handle, (const cusparseLtMatmulPlan_t*)plan, workspaceSize));
}

hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                            hipsparseLtMatmulPlan_t*               plan,
                                            const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                            const hipsparseLtMatmulAlgSelection_t* algSelection)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulPlanInit((const cusparseLtHandle_t*)handle,
                                 (cusparseLtMatmulPlan_t*)plan,
                                 (const cusparseLtMatmulDescriptor_t*)matmulDescr,
                                 (const cusparseLtMatmulAlgSelection_t*)algSelection));
}

hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtMatmulPlanDestroy((const cusparseLtMatmulPlan_t*)plan));
}

/* matmul execution */
hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                    const hipsparseLtMatmulPlan_t* plan,
                                    const void*                    alpha,
                                    const void*                    d_A,
                                    const void*                    d_B,
                                    const void*                    beta,
                                    const void*                    d_C,
                                    void*                          d_D,
                                    void*                          workspace,
                                    hipStream_t*                   streams,
                                    int32_t                        numStreams)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtMatmul((const cusparseLtHandle_t*)handle,
                                                         (const cusparseLtMatmulPlan_t*)plan,
                                                         alpha,
                                                         d_A,
                                                         d_B,
                                                         beta,
                                                         d_C,
                                                         d_D,
                                                         workspace,
                                                         streams,
                                                         numStreams));
}

hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* handle,
                                          hipsparseLtMatmulPlan_t*   plan,
                                          const void*                alpha,
                                          const void*                d_A,
                                          const void*                d_B,
                                          const void*                beta,
                                          const void*                d_C,
                                          void*                      d_D,
                                          void*                      workspace,
                                          hipStream_t*               streams,
                                          int32_t                    numStreams)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtMatmulSearch((const cusparseLtHandle_t*)handle,
                                                               (cusparseLtMatmulPlan_t*)plan,
                                                               alpha,
                                                               d_A,
                                                               d_B,
                                                               beta,
                                                               d_C,
                                                               d_D,
                                                               workspace,
                                                               streams,
                                                               numStreams));
}

/* helper */
// prune
hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                        const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                        const void*                          d_in,
                                        void*                                d_out,
                                        hipsparseLtPruneAlg_t                pruneAlg,
                                        hipStream_t                          stream)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMAPrune((const cusparseLtHandle_t*)handle,
                             (const cusparseLtMatmulDescriptor_t*)matmulDescr,
                             d_in,
                             d_out,
                             HIPPruneAlgToCuSparseLtPruneAlg(pruneAlg),
                             stream));
}

hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                             const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                             const void*                          d_in,
                                             int*                                 valid,
                                             hipStream_t                          stream)

{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMAPruneCheck((const cusparseLtHandle_t*)handle,
                                  (const cusparseLtMatmulDescriptor_t*)matmulDescr,
                                  d_in,
                                  valid,
                                  stream));
}

hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                         const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                         int                               isSparseA,
                                         hipsparseOperation_t              op,
                                         const void*                       d_in,
                                         void*                             d_out,
                                         hipsparseLtPruneAlg_t             pruneAlg,
                                         hipStream_t                       stream)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMAPrune2((const cusparseLtHandle_t*)handle,
                              (const cusparseLtMatDescriptor_t*)sparseMatDescr,
                              isSparseA,
                              hipOperationToCudaOperation(op),
                              d_in,
                              d_out,
                              HIPPruneAlgToCuSparseLtPruneAlg(pruneAlg),
                              stream));
}

hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseOperation_t              op,
                                              const void*                       d_in,
                                              int*                              d_valid,
                                              hipStream_t                       stream)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMAPruneCheck2((const cusparseLtHandle_t*)handle,
                                   (const cusparseLtMatDescriptor_t*)sparseMatDescr,
                                   isSparseA,
                                   hipOperationToCudaOperation(op),
                                   d_in,
                                   d_valid,
                                   stream));
}

// compression
hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                 const hipsparseLtMatmulPlan_t* plan,
                                                 size_t*                        compressedSize,
                                                 size_t*                        compressBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMACompressedSize((const cusparseLtHandle_t*)handle,
                                      (const cusparseLtMatmulPlan_t*)plan,
                                      compressedSize,
                                      compressBufferSize));
}

hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                           const hipsparseLtMatmulPlan_t* plan,
                                           const void*                    d_dense,
                                           void*                          d_compressed,
                                           void*                          d_compressBuffer,
                                           hipStream_t                    stream)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseLtSpMMACompress((const cusparseLtHandle_t*)handle,
                                                                (const cusparseLtMatmulPlan_t*)plan,
                                                                d_dense,
                                                                d_compressed,
                                                                d_compressBuffer,
                                                                stream));
}

hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                                  const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                  size_t*                           compressedSize,
                                                  size_t* compressBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMACompressedSize2((const cusparseLtHandle_t*)handle,
                                       (const cusparseLtMatDescriptor_t*)sparseMatDescr,
                                       compressedSize,
                                       compressBufferSize));
}

hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                            const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                            int                               isSparseA,
                                            hipsparseOperation_t              op,
                                            const void*                       d_dense,
                                            void*                             d_compressed,
                                            void*                             d_compressBuffer,
                                            hipStream_t                       stream)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseLtSpMMACompress2((const cusparseLtHandle_t*)handle,
                                 (const cusparseLtMatDescriptor_t*)sparseMatDescr,
                                 isSparseA,
                                 hipOperationToCudaOperation(op),
                                 d_dense,
                                 d_compressed,
                                 d_compressBuffer,
                                 stream));
}

void hipsparseLtInitialize() {}

hipsparseStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char* rev)
try
{
    // Get hipSPARSE revision
    if(rev == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(hipsparseltVersionTweak);

    memcpy(rev, v, sizeof(v));

    return HIPSPARSE_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtGetArchName(char** archName)
try
{
    *archName = (char*)malloc(5);
    snprintf(*archName, 5, "cuda\0");
    return HIPSPARSE_STATUS_SUCCESS;
}
catch(...)
{
    if(archName != nullptr)
    {
        free(*archName);
        *archName = nullptr;
    }
    return exception_to_hipsparselt_status();
}

#ifdef __cplusplus
}
#endif
