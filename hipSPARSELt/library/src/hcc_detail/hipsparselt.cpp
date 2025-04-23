/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
#include "hipsparselt_ostream.hpp"
#include "utility.hpp"
#include <hip/hip_runtime_api.h>
#include <hipsparselt/hipsparselt.h>
#include <rocsparselt.h>
#include <stdio.h>
#include <stdlib.h>
#include "Debug.hpp"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPSPARSELT_ERROR(INPUT_STATUS_FOR_CHECK)              \
    {                                                                    \
        hipsparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPSPARSE_STATUS_SUCCESS)             \
        {                                                                \
            return TMP_STATUS_FOR_CHECK;                                 \
        }                                                                \
    }

#define RETURN_IF_ROCSPARSELT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                     \
        rocsparselt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocsparselt_status_success)            \
        {                                                                 \
            return RocSparseLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                                 \
    }

hipsparseStatus_t hipErrorToHIPSPARSEStatus(hipError_t status)
{
    switch(status)
    {
    case hipSuccess:
        return HIPSPARSE_STATUS_SUCCESS;
    case hipErrorMemoryAllocation:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case hipErrorLaunchOutOfResources:
        return HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
    case hipErrorInvalidDevicePointer:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case hipErrorInvalidValue:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorNoDevice:
    case hipErrorUnknown:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    default:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }
}

hipsparseStatus_t RocSparseLtStatusToHIPStatus(rocsparselt_status_ status)
{
    switch(status)
    {
    case rocsparselt_status_success:
        return HIPSPARSE_STATUS_SUCCESS;
    case rocsparselt_status_invalid_handle:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparselt_status_not_implemented:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    case rocsparselt_status_invalid_pointer:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparselt_status_invalid_size:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparselt_status_memory_error:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case rocsparselt_status_internal_error:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case rocsparselt_status_invalid_value:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparselt_status_arch_mismatch:
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_compute_type_ HIPComputetypeToRocSparseLtComputetype(hipsparseLtComputetype_t type)
{
    switch(type)
    {
    case HIPSPARSELT_COMPUTE_32F:
        return rocsparselt_compute_f32;
    case HIPSPARSELT_COMPUTE_32I:
        return rocsparselt_compute_i32;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtComputetype_t RocSparseLtComputetypeToHIPComputetype(rocsparselt_compute_type_ type)
{
    switch(type)
    {
    case rocsparselt_compute_f32:
        return HIPSPARSELT_COMPUTE_32F;

    case rocsparselt_compute_i32:
        return HIPSPARSELT_COMPUTE_32I;
    }
    throw HIPSPARSE_STATUS_NOT_SUPPORTED;
}

rocsparselt_operation_ HIPOperationToHCCOperation(hipsparseOperation_t op)
{
    switch(op)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return rocsparselt_operation_none;
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return rocsparselt_operation_transpose;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return rocsparselt_operation_conjugate_transpose;
    default:
        throw HIPSPARSE_STATUS_INVALID_VALUE;
    }
}

hipsparseOperation_t HCCOperationToHIPOperation(rocsparselt_operation_ op)
{
    switch(op)
    {
    case rocsparselt_operation_none:
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case rocsparselt_operation_transpose:
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case rocsparselt_operation_conjugate_transpose:
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw HIPSPARSE_STATUS_INVALID_VALUE;
    }
}

rocsparselt_order_ HIPOrderToHCCOrder(hipsparseOrder_t op)
{
    switch(op)
    {
    case HIPSPARSE_ORDER_ROW:
        return rocsparselt_order_row;
    case HIPSPARSE_ORDER_COL:
        return rocsparselt_order_column;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseOrder_t HCCOrderToHIPOrder(rocsparselt_order_ op)
{
    switch(op)
    {
    case rocsparselt_order_row:
        return HIPSPARSE_ORDER_ROW;
    case rocsparselt_order_column:
        return HIPSPARSE_ORDER_COL;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_sparsity_ HIPSparsityToRocSparseLtSparsity(hipsparseLtSparsity_t sparsity)
{
    switch(sparsity)
    {
    case HIPSPARSELT_SPARSITY_50_PERCENT:
        return rocsparselt_sparsity_50_percent;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtSparsity_t RocSparseLtSparsityToHIPSparsity(rocsparselt_sparsity_ sparsity)
{
    switch(sparsity)
    {
    case rocsparselt_sparsity_50_percent:
        return HIPSPARSELT_SPARSITY_50_PERCENT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_matmul_descr_attribute_
    HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(hipsparseLtMatmulDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU:
        return rocsparselt_matmul_activation_relu;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
        return rocsparselt_matmul_activation_relu_upperbound;
    case HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
        return rocsparselt_matmul_activation_relu_threshold;
    case HIPSPARSELT_MATMUL_ACTIVATION_GELU:
        return rocsparselt_matmul_activation_gelu;
    case HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
        return rocsparselt_matmul_activation_gelu_scaling;
    case HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING:
        return rocsparselt_matmul_alpha_vector_scaling;
    case HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING:
        return rocsparselt_matmul_beta_vector_scaling;
    case HIPSPARSELT_MATMUL_BIAS_STRIDE:
        return rocsparselt_matmul_bias_stride;
    case HIPSPARSELT_MATMUL_BIAS_POINTER:
        return rocsparselt_matmul_bias_pointer;
    case HIPSPARSELT_MATMUL_ACTIVATION_ABS:
        return rocsparselt_matmul_activation_abs;
    case HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU:
        return rocsparselt_matmul_activation_leakyrelu;
    case HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA:
        return rocsparselt_matmul_activation_leakyrelu_alpha;
    case HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID:
        return rocsparselt_matmul_activation_sigmoid;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH:
        return rocsparselt_matmul_activation_tanh;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA:
        return rocsparselt_matmul_activation_tanh_alpha;
    case HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA:
        return rocsparselt_matmul_activation_tanh_beta;
    case HIPSPARSELT_MATMUL_BIAS_TYPE:
        return rocsparselt_matmul_bias_type;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulDescAttribute_t
    RocSparseLtMatmulDescAttributeToHIPMatmulDescAttribute(rocsparselt_matmul_descr_attribute_ attr)
{
    switch(attr)
    {
    case rocsparselt_matmul_activation_relu:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU;
    case rocsparselt_matmul_activation_relu_upperbound:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND;
    case rocsparselt_matmul_activation_relu_threshold:
        return HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD;
    case rocsparselt_matmul_activation_gelu:
        return HIPSPARSELT_MATMUL_ACTIVATION_GELU;
    case rocsparselt_matmul_activation_gelu_scaling:
        return HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING;
    case rocsparselt_matmul_alpha_vector_scaling:
        return HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING;
    case rocsparselt_matmul_beta_vector_scaling:
        return HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING;
    case rocsparselt_matmul_bias_stride:
        return HIPSPARSELT_MATMUL_BIAS_STRIDE;
    case rocsparselt_matmul_bias_pointer:
        return HIPSPARSELT_MATMUL_BIAS_POINTER;
    case rocsparselt_matmul_activation_abs:
        return HIPSPARSELT_MATMUL_ACTIVATION_ABS;
    case rocsparselt_matmul_activation_leakyrelu:
        return HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU;
    case rocsparselt_matmul_activation_leakyrelu_alpha:
        return HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA;
    case rocsparselt_matmul_activation_sigmoid:
        return HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID;
    case rocsparselt_matmul_activation_tanh:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH;
    case rocsparselt_matmul_activation_tanh_alpha:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA;
    case rocsparselt_matmul_activation_tanh_beta:
        return HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA;
    case rocsparselt_matmul_bias_type:
        return HIPSPARSELT_MATMUL_BIAS_TYPE;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatDescAttribute_t
    RocSparseLtMatDescAttributeToHIPMatDescAttribute(rocsparselt_mat_descr_attribute_ attr)
{
    switch(attr)
    {
    case rocsparselt_mat_num_batches:
        return HIPSPARSELT_MAT_NUM_BATCHES;
    case rocsparselt_mat_batch_stride:
        return HIPSPARSELT_MAT_BATCH_STRIDE;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_mat_descr_attribute_
    HIPMatDescAttributeToRocSparseLtMatDescAttribute(hipsparseLtMatDescAttribute_t attr)
{
    switch(attr)
    {
    case HIPSPARSELT_MAT_NUM_BATCHES:
        return rocsparselt_mat_num_batches;
    case HIPSPARSELT_MAT_BATCH_STRIDE:
        return rocsparselt_mat_batch_stride;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_matmul_alg_ HIPMatmulAlgToRocSparseLtMatmulAlg(hipsparseLtMatmulAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_DEFAULT:
        return rocsparselt_matmul_alg_default;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulAlg_t RocSparseLtMatmulAlgToHIPMatmulAlg(rocsparselt_matmul_alg_ alg)
{
    switch(alg)
    {
    case rocsparselt_matmul_alg_default:
        return HIPSPARSELT_MATMUL_ALG_DEFAULT;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_matmul_alg_attribute_
    HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(hipsparseLtMatmulAlgAttribute_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_MATMUL_ALG_CONFIG_ID:
        return rocsparselt_matmul_alg_config_id;
    case HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID:
        return rocsparselt_matmul_alg_config_max_id;
    case HIPSPARSELT_MATMUL_SEARCH_ITERATIONS:
        return rocsparselt_matmul_search_iterations;
    case HIPSPARSELT_MATMUL_SPLIT_K:
        return rocsparselt_matmul_split_k;
    case HIPSPARSELT_MATMUL_SPLIT_K_MODE:
        return rocsparselt_matmul_split_k_mode;
    case HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS:
        return rocsparselt_matmul_split_k_buffers;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtMatmulAlgAttribute_t
    RocSparseLtAlgAttributeToHIPMatmulAlgAttribute(rocsparselt_matmul_alg_attribute_ alg)
{
    switch(alg)
    {
    case rocsparselt_matmul_alg_config_id:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_ID;
    case rocsparselt_matmul_alg_config_max_id:
        return HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
    case rocsparselt_matmul_search_iterations:
        return HIPSPARSELT_MATMUL_SEARCH_ITERATIONS;
    case rocsparselt_matmul_split_k:
        return HIPSPARSELT_MATMUL_SPLIT_K;
    case rocsparselt_matmul_split_k_mode:
        return HIPSPARSELT_MATMUL_SPLIT_K_MODE;
    case rocsparselt_matmul_split_k_buffers:
        return HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_prune_alg_ HIPPruneAlgToRocSparseLtPruneAlg(hipsparseLtPruneAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSELT_PRUNE_SPMMA_TILE:
        return rocsparselt_prune_smfmac_tile;
    case HIPSPARSELT_PRUNE_SPMMA_STRIP:
        return rocsparselt_prune_smfmac_strip;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtPruneAlg_t RocSparseLtPruneAlgToHIPPruneAlg(rocsparselt_prune_alg_ alg)
{
    switch(alg)
    {
    case rocsparselt_prune_smfmac_tile:
        return HIPSPARSELT_PRUNE_SPMMA_TILE;
    case rocsparselt_prune_smfmac_strip:
        return HIPSPARSELT_PRUNE_SPMMA_STRIP;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

rocsparselt_split_k_mode HIPSplitKModeToRocSparseLtSplitKMode(hipsparseLtSplitKMode_t mode)
{
    switch(mode)
    {
    case HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL:
        return rocsparselt_splik_k_mode_one_kernel;
    case HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS:
        return rocsparselt_split_k_mode_two_kernels;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseLtSplitKMode_t RocSparseLtSplitKModeToHIPSplitKMode(rocsparselt_split_k_mode mode)
{
    switch(mode)
    {
    case rocsparselt_splik_k_mode_one_kernel:
        return HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL;
    case rocsparselt_split_k_mode_two_kernels:
        return HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
    default:
        throw HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
}

hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle)
try
{
    // Check if handle is valid
    rocsparselt::Debug::Instance().markerStart("hipsparseLtInit");
    if(handle == nullptr)
    {
        rocsparselt::Debug::Instance().markerStop();
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    int               deviceId;
    hipError_t        err;
    hipsparseStatus_t retval = HIPSPARSE_STATUS_SUCCESS;

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocSparseLtStatusToHIPStatus(rocsparselt_init((rocsparselt_handle*)handle));
    }
    rocsparselt::Debug::Instance().markerStop();
    return retval;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle)
try
{
    return RocSparseLtStatusToHIPStatus(rocsparselt_destroy((const rocsparselt_handle*)handle));
}
catch(...)
{
    return exception_to_hipsparselt_status();
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
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtDenseDescriptorInit");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_dense_descr_init((const rocsparselt_handle*)handle,
                                     (rocsparselt_mat_descr*)matDescr,
                                     rows,
                                     cols,
                                     ld,
                                     alignment,
                                     valueType,
                                     HIPOrderToHCCOrder(order)));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
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
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtStructuredDescriptorInit");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_structured_descr_init((const rocsparselt_handle*)handle,
                                          (rocsparselt_mat_descr*)matDescr,
                                          rows,
                                          cols,
                                          ld,
                                          alignment,
                                          valueType,
                                          HIPOrderToHCCOrder(order),
                                          HIPSparsityToRocSparseLtSparsity(sparsity)));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatDescriptorDestroy");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_mat_descr_destroy((const rocsparselt_mat_descr*)matDescr));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                 hipsparseLtMatDescriptor_t*   matmulDescr,
                                                 hipsparseLtMatDescAttribute_t matAttribute,
                                                 const void*                   data,
                                                 size_t                        dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatDescSetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_mat_descr_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_mat_descr*)matmulDescr,
        HIPMatDescAttributeToRocSparseLtMatDescAttribute(matAttribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                 const hipsparseLtMatDescriptor_t* matmulDescr,
                                                 hipsparseLtMatDescAttribute_t     matAttribute,
                                                 void*                             data,
                                                 size_t                            dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatDescGetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_mat_descr_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_mat_descr*)matmulDescr,
        HIPMatDescAttributeToRocSparseLtMatDescAttribute(matAttribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
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
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulDescriptorInit");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_descr_init((const rocsparselt_handle*)handle,
                                      (rocsparselt_matmul_descr*)matmulDescr,
                                      HIPOperationToHCCOperation(opA),
                                      HIPOperationToHCCOperation(opB),
                                      (const rocsparselt_mat_descr*)matA,
                                      (const rocsparselt_mat_descr*)matB,
                                      (const rocsparselt_mat_descr*)matC,
                                      (const rocsparselt_mat_descr*)matD,
                                      HIPComputetypeToRocSparseLtComputetype(computeType)));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulDescSetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_descr_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_matmul_descr*)matmulDescr,
        HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulDescGetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_descr_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_matmul_descr*)matmulDescr,
        HIPMatmulDescAttributeToRocSparseLtMatmulDescAttribute(matmulAttribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* algorithm selection */
hipsparseStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulAlgSelectionInit");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_alg_selection_init((const rocsparselt_handle*)handle,
                                              (rocsparselt_matmul_alg_selection*)algSelection,
                                              (const rocsparselt_matmul_descr*)matmulDescr,
                                              HIPMatmulAlgToRocSparseLtMatmulAlg(alg)));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                   hipsparseLtMatmulAlgSelection_t* algSelection,
                                                   hipsparseLtMatmulAlgAttribute_t  attribute,
                                                   const void*                      data,
                                                   size_t                           dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulAlgSetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_alg_set_attribute(
        (const rocsparselt_handle*)handle,
        (rocsparselt_matmul_alg_selection*)algSelection,
        HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(attribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulAlgGetAttribute");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_alg_get_attribute(
        (const rocsparselt_handle*)handle,
        (const rocsparselt_matmul_alg_selection*)algSelection,
        HIPMatmulAlgAttributeToRocSparseLtAlgAttribute(attribute),
        data,
        dataSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* matmul plan */
hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                const hipsparseLtMatmulPlan_t* plan,
                                                size_t*                        workspaceSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulGetWorkspace");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_get_workspace(
        (const rocsparselt_handle*)handle, (const rocsparselt_matmul_plan*)plan, workspaceSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                            hipsparseLtMatmulPlan_t*               plan,
                                            const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                            const hipsparseLtMatmulAlgSelection_t* algSelection)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulPlanInit");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_plan_init((const rocsparselt_handle*)handle,
                                     (rocsparselt_matmul_plan*)plan,
                                     (const rocsparselt_matmul_descr*)matmulDescr,
                                     (const rocsparselt_matmul_alg_selection*)algSelection));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulPlanDestroy");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_matmul_plan_destroy((const rocsparselt_matmul_plan*)plan));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
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
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmul");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul((const rocsparselt_handle*)handle,
                                                           (const rocsparselt_matmul_plan*)plan,
                                                           alpha,
                                                           d_A,
                                                           d_B,
                                                           beta,
                                                           d_C,
                                                           d_D,
                                                           workspace,
                                                           streams,
                                                           numStreams));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
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
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtMatmulSearch");
    auto status = RocSparseLtStatusToHIPStatus(rocsparselt_matmul_search((const rocsparselt_handle*)handle,
                                                                  (rocsparselt_matmul_plan*)plan,
                                                                  alpha,
                                                                  d_A,
                                                                  d_B,
                                                                  beta,
                                                                  d_C,
                                                                  d_D,
                                                                  workspace,
                                                                  streams,
                                                                  numStreams));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

/* helper */
// prune
hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                        const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                        const void*                          d_in,
                                        void*                                d_out,
                                        hipsparseLtPruneAlg_t                pruneAlg,
                                        hipStream_t                          stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMAPrune");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune((const rocsparselt_handle*)handle,
                                 (const rocsparselt_matmul_descr*)matmulDescr,
                                 d_in,
                                 d_out,
                                 HIPPruneAlgToRocSparseLtPruneAlg(pruneAlg),
                                 stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                             const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                             const void*                          d_in,
                                             int*                                 valid,
                                             hipStream_t                          stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMAPruneCheck");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune_check((const rocsparselt_handle*)handle,
                                       (const rocsparselt_matmul_descr*)matmulDescr,
                                       d_in,
                                       valid,
                                       stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                         const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                         int                               isSparseA,
                                         hipsparseOperation_t              op,
                                         const void*                       d_in,
                                         void*                             d_out,
                                         hipsparseLtPruneAlg_t             pruneAlg,
                                         hipStream_t                       stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMAPrune2");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune2((const rocsparselt_handle*)handle,
                                  (const rocsparselt_mat_descr*)sparseMatDescr,
                                  isSparseA,
                                  HIPOperationToHCCOperation(op),
                                  d_in,
                                  d_out,
                                  HIPPruneAlgToRocSparseLtPruneAlg(pruneAlg),
                                  stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseOperation_t              op,
                                              const void*                       d_in,
                                              int*                              d_valid,
                                              hipStream_t                       stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMAPruneCheck2");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_prune_check2((const rocsparselt_handle*)handle,
                                        (const rocsparselt_mat_descr*)sparseMatDescr,
                                        isSparseA,
                                        HIPOperationToHCCOperation(op),
                                        d_in,
                                        d_valid,
                                        stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

// compression
hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                 const hipsparseLtMatmulPlan_t* plan,
                                                 size_t*                        compressedSize,
                                                 size_t*                        compressBufferSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMACompressedSizes");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compressed_size((const rocsparselt_handle*)handle,
                                           (const rocsparselt_matmul_plan*)plan,
                                           compressedSize,
                                           compressBufferSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                           const hipsparseLtMatmulPlan_t* plan,
                                           const void*                    d_dense,
                                           void*                          d_compressed,
                                           void*                          d_compressBuffer,
                                           hipStream_t                    stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMACompress");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compress((const rocsparselt_handle*)handle,
                                    (const rocsparselt_matmul_plan*)plan,
                                    d_dense,
                                    d_compressed,
                                    d_compressBuffer,
                                    stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                                  const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                  size_t*                           compressedSize,
                                                  size_t* compressBufferSize)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMACompressedSize2");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compressed_size2((const rocsparselt_handle*)handle,
                                            (const rocsparselt_mat_descr*)sparseMatDescr,
                                            compressedSize,
                                            compressBufferSize));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                            const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                            int                               isSparseA,
                                            hipsparseOperation_t              op,
                                            const void*                       d_dense,
                                            void*                             d_compressed,
                                            void*                             d_compressBuffer,
                                            hipStream_t                       stream)
try
{
    rocsparselt::Debug::Instance().markerStart("hipsparseLtSpMMACompress2");
    auto status = RocSparseLtStatusToHIPStatus(
        rocsparselt_smfmac_compress2((const rocsparselt_handle*)handle,
                                     (const rocsparselt_mat_descr*)sparseMatDescr,
                                     isSparseA,
                                     HIPOperationToHCCOperation(op),
                                     d_dense,
                                     d_compressed,
                                     d_compressBuffer,
                                     stream));
    rocsparselt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

void hipsparseLtInitialize()
{
    rocsparselt_initialize();
}

hipsparseStatus_t hipsparseLtGetVersion(const hipsparseLtHandle_t* handle, int* version)
try
{
    *version = hipsparseltVersionMajor * 100000 + hipsparseltVersionMinor * 100
               + hipsparseltVersionPatch;

    return HIPSPARSE_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

hipsparseStatus_t hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int* value)
try
{
    switch(propertyType)
    {
    case HIP_LIBRARY_MAJOR_VERSION:
        *value = hipsparseltVersionMajor;
        break;
    case HIP_LIBRARY_MINOR_VERSION:
        *value = hipsparseltVersionMinor;
        break;
    case HIP_LIBRARY_PATCH_LEVEL:
        *value = hipsparseltVersionPatch;
        break;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipsparselt_status();
}

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
    *archName        = nullptr;
    std::string arch = "cuda";
    *archName        = (char*)malloc(arch.size() * sizeof(char));
    strncpy(*archName, arch.c_str(), arch.size());
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
