/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "internal_publishKernels.h"
#if _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#include <smmintrin.h>
#include <x86intrin.h>
#endif

struct TensorMulScalarLocalData {
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    float scalarValue;
    size_t tensorSize;
    vx_enum inputTensorType;
    vx_enum outputTensorType;
};

static vx_status VX_CALLBACK refreshTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorMulScalarLocalData *data) {
    vx_status status = VX_SUCCESS;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        return status;
    }
    return status;
}

static vx_status VX_CALLBACK validateTensorMulScalar(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #2 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);

    // Validate for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));

    // Validate for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_type;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    vx_status status = VX_SUCCESS;
    TensorMulScalarLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshTensorMulScalar(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        if (data->inputTensorType == vx_type_e::VX_TYPE_FLOAT32 && data->outputTensorType == vx_type_e::VX_TYPE_FLOAT32) {
            __m256 pMul = _mm256_set1_ps(data->scalarValue);
            float scalarValue = data->scalarValue;
            float *srcPtrTemp = static_cast<float *>(data->pSrc);
            float *dstPtrTemp = static_cast<float *>(data->pDst);
            uint bufferLength = data->tensorSize / sizeof(float);
            uint alignedLength = bufferLength & ~7;
            uint vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 8) {
                __m256 pSrc = _mm256_loadu_ps(srcPtrTemp);
                __m256 pDst = _mm256_mul_ps(pSrc, pMul);
                _mm256_storeu_ps(dstPtrTemp, pDst);
                srcPtrTemp += 8;
                dstPtrTemp += 8;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                *dstPtrTemp++ = *srcPtrTemp++ * scalarValue;
        } else {
            return VX_ERROR_NOT_SUPPORTED;
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    TensorMulScalarLocalData *data = new TensorMulScalarLocalData;
    if (data) {
        memset(data, 0, sizeof(TensorMulScalarLocalData));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[2], &data->scalarValue));
        vx_size num_of_dims;
        size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(vx_size)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, tensor_dims, sizeof(vx_size) * num_of_dims));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->inputTensorType, sizeof(data->inputTensorType)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data->outputTensorType, sizeof(data->outputTensorType)));
        // Calculate the total tensor size
        data->tensorSize = 1;
        for (int i = 0; i < num_of_dims; i++)
            data->tensorSize *= tensor_dims[i];
        if (data->inputTensorType == vx_type_e::VX_TYPE_FLOAT32 && data->outputTensorType == vx_type_e::VX_TYPE_FLOAT32)
            data->tensorSize *= sizeof(float);
        else
            return VX_ERROR_NOT_SUPPORTED;
        STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
        return VX_SUCCESS;
    } else {
        return VX_FAILURE;
    }
}

static vx_status VX_CALLBACK uninitializeTensorMulScalar(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    TensorMulScalarLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32 &supported_target_affinity) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status TensorMulScalar_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorMulScalar",
                                       VX_KERNEL_RPP_TENSORMULSCALAR,
                                       processTensorMulScalar,
                                       4,
                                       validateTensorMulScalar,
                                       initializeTensorMulScalar,
                                       uninitializeTensorMulScalar);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
