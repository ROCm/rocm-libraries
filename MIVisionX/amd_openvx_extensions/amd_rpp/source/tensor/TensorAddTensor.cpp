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

struct TensorAddTensorLocalData {
    Rpp32u deviceType;
    RppPtr_t pSrc1;
    RppPtr_t pSrc2;
    RppPtr_t pDst;
    RpptROI *pSrcRoi;
    RpptROI *pDstRoi;
    size_t inputTensorDims1[RPP_MAX_TENSOR_DIMS];
    size_t inputTensorDims2[RPP_MAX_TENSOR_DIMS];
    vx_enum inTensorType;
    vx_enum outTensorType;
};

static vx_status VX_CALLBACK refreshTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorAddTensorLocalData *data) {
    vx_status status = VX_SUCCESS;
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        void *roi_tensor_ptr_src, *roi_tensor_ptr_dst;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc1, sizeof(data->pSrc1)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pSrc2, sizeof(data->pSrc2)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
        data->pDstRoi = static_cast<RpptROI *>(roi_tensor_ptr_dst);
        data->pSrcRoi = static_cast<RpptROI *>(roi_tensor_ptr_src);
        for (uint i = 0; i < data->inputTensorDims1[0]; i++) {
            data->pDstRoi[i].xywhROI.roiWidth = data->pSrcRoi[i].xywhROI.roiWidth;
            data->pDstRoi[i].xywhROI.roiHeight = data->pSrcRoi[i].xywhROI.roiHeight;
        }
        return status;
    }
    return status;
}

static vx_status VX_CALLBACK validateTensorAddTensor(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);

    // Validate for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));

    // Validate for output parameters
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_type;

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &tensor_type, sizeof(tensor_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    vx_status status = VX_SUCCESS;
    TensorAddTensorLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        refreshTensorAddTensor(node, parameters, num, data);
        // Currently supports adding audio tensors and distribution node tensors - (BS, samples, 1) and (BS, 1) - will be extended to include all tensors of supported layouts
        if (data->inTensorType == vx_type_e::VX_TYPE_FLOAT32 && data->outTensorType == vx_type_e::VX_TYPE_FLOAT32) {
            size_t nStride = data->inputTensorDims1[1] * data->inputTensorDims1[2];
            for (uint i = 0; i < data->inputTensorDims1[0]; i++) {
                float *src1Temp = static_cast<float *>(data->pSrc1) + i * nStride;
                float *src2Temp = static_cast<float *>(data->pSrc2) + i;
                float *dstTemp = static_cast<float *>(data->pDst) + i * nStride;
                uint height = data->pSrcRoi[i].xywhROI.roiHeight;
                uint width = data->pSrcRoi[i].xywhROI.roiWidth;
                uint alignedWidth = width & ~7;
                float additionFactor = src2Temp[0];
                __m256 pSrc2 = _mm256_set1_ps(additionFactor);
                for (uint row = 0; row < height; row++) {
                    float *srcPtr1Row = src1Temp + row * data->inputTensorDims1[1];
                    float *dstPtrRow = dstTemp + row * data->inputTensorDims1[1];
                    uint vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedWidth; vectorLoopCount += 8) {
                        __m256 pSrc1 = _mm256_loadu_ps(srcPtr1Row);
                        __m256 pDst = _mm256_add_ps(pSrc1, pSrc2);
                        _mm256_storeu_ps(dstPtrRow, pDst);
                        srcPtr1Row += 8;
                        dstPtrRow += 8;
                    }
                    for (; vectorLoopCount < width; vectorLoopCount++)
                        *dstPtrRow++ = (*srcPtr1Row++) + additionFactor;
                }
            }
        } else {
            return VX_ERROR_NOT_SUPPORTED;
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    TensorAddTensorLocalData *data = new TensorAddTensorLocalData;
    if (data) {
        memset(data, 0, sizeof(TensorAddTensorLocalData));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        vx_size num_of_dims1, num_of_dims2;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims1, sizeof(vx_size)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims1, sizeof(vx_size) * num_of_dims1));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data->inTensorType, sizeof(data->inTensorType)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims2, sizeof(vx_size)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->inputTensorDims2, sizeof(vx_size) * num_of_dims2));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data->inTensorType, sizeof(data->inTensorType)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &data->outTensorType, sizeof(data->outTensorType)));

        STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
        return VX_SUCCESS;
    } else {
        return VX_FAILURE;
    }    
}

static vx_status VX_CALLBACK uninitializeTensorAddTensor(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    TensorAddTensorLocalData *data;
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

vx_status TensorAddTensor_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorAddTensor",
                                       VX_KERNEL_RPP_TENSORADDTENSOR,
                                       processTensorAddTensor,
                                       6,
                                       validateTensorAddTensor,
                                       initializeTensorAddTensor,
                                       uninitializeTensorAddTensor);
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}
