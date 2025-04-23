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

struct SequenceRearrangeLocalData {
    vxRppHandle *handle;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    vx_uint32 deviceType;
    vx_uint32 newSequenceLength;
    vx_uint32 sequenceLength;
    vx_uint32 *pNewOrder;
    vxTensorLayout layout;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
#if ENABLE_OPENCL
    cl_mem pClSrc;
    cl_mem pClDst;
#endif
};

static vx_status VX_CALLBACK refreshSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num, SequenceRearrangeLocalData *data) {
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->newSequenceLength, sizeof(vx_uint32), data->pNewOrder, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->pClSrc, sizeof(data->pClSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->pClDst, sizeof(data->pClDst)));
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->pDst, sizeof(data->pDst)));
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
    }
    return status;
}

static vx_status VX_CALLBACK validateSequenceRearrange(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_INT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #3 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Parameter: #4 type=%d (must be size)\n", scalar_type);

    // Check for input tensor
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims != 5) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: SequenceRearrange: tensor: #0 dimensions=%lu (must be equal to 5)\n", num_tensor_dims);

    // Check for output tensor
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_dtype;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims != 5) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: SequenceRearrange: tensor: #1 dimensions=%lu (must be equal to 5)\n", num_tensor_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_dtype, sizeof(tensor_dtype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));

    return status;
}

static vx_status VX_CALLBACK processSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SequenceRearrangeLocalData *data = NULL;
    vx_status status = VX_SUCCESS;

    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshSequenceRearrange(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
        cl_command_queue handle = data->handle->cmdq;
        for (unsigned sequence_cnt = 0; sequence_cnt < data->pSrcDesc->n; sequence_cnt++) {
            unsigned src_sequence_start_address = sequence_cnt * data->pSrcDesc->strides.nStride * data->sequenceLength;
            unsigned dst_sequence_start_address = sequence_cnt * data->pDstDesc->strides.nStride * data->newSequenceLength;
            for (unsigned dst_index = 0; dst_index < data->newSequenceLength; dst_index++) {
                unsigned src_index = data->pNewOrder[dst_index];
                if (src_index > data->sequenceLength)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequenceLength - 1);
                auto dst_offset = dst_sequence_start_address + (dst_index * data->pSrcDesc->strides.nStride);
                auto src_offset = src_sequence_start_address + (src_index * data->pDstDesc->strides.nStride);
                if (clEnqueueCopyBuffer(handle, data->pClSrc, data->pClDst, src_offset, dst_offset, data->pSrcDesc->strides.nStride, 0, NULL, NULL) != CL_SUCCESS)
                        return VX_FAILURE;
            }
        }
#elif ENABLE_HIP
        for (unsigned sequence_cnt = 0; sequence_cnt < data->pSrcDesc->n; sequence_cnt++) {
            unsigned src_sequence_start_address = sequence_cnt * data->pSrcDesc->strides.nStride * data->sequenceLength;
            unsigned dst_sequence_start_address = sequence_cnt * data->pDstDesc->strides.nStride * data->newSequenceLength;
            for (unsigned dst_index = 0; dst_index < (data->newSequenceLength); dst_index++) {
                unsigned src_index = data->pNewOrder[dst_index];
                if (src_index > data->sequenceLength)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequenceLength - 1);
                auto dst_address = static_cast<unsigned char *>(data->pDst) + dst_sequence_start_address + (dst_index * data->pSrcDesc->strides.nStride);
                auto src_address = static_cast<unsigned char *>(data->pSrc) + src_sequence_start_address + (src_index * data->pDstDesc->strides.nStride);
                hipError_t status = hipMemcpyDtoD(dst_address, src_address, data->pSrcDesc->strides.nStride);
                    if (status != hipSuccess)
                        return VX_FAILURE;  
            }
        }
#endif
    } else if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        for (unsigned sequence_cnt = 0; sequence_cnt < data->pSrcDesc->n; sequence_cnt++) {
            unsigned src_sequence_start_address = sequence_cnt * data->pSrcDesc->strides.nStride * data->sequenceLength;
            unsigned dst_sequence_start_address = sequence_cnt * data->pDstDesc->strides.nStride * data->newSequenceLength;
            for (unsigned dst_index = 0; dst_index < (data->newSequenceLength); dst_index++) {
                unsigned src_index = data->pNewOrder[dst_index];
                if (src_index > data->sequenceLength)
                    ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequenceLength - 1);
                auto dst_address = static_cast<unsigned char *>(data->pDst) + dst_sequence_start_address + (dst_index * data->pSrcDesc->strides.nStride);
                auto src_address = static_cast<unsigned char *>(data->pSrc) + src_sequence_start_address + (src_index * data->pDstDesc->strides.nStride);
                memcpy(dst_address, src_address, data->pSrcDesc->strides.nStride);
            }
        }
    }
    return status;
}

static vx_status VX_CALLBACK initializeSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SequenceRearrangeLocalData *data = new SequenceRearrangeLocalData;
    memset(data, 0, sizeof(SequenceRearrangeLocalData));
    
    int layout;
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &layout, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->layout = static_cast<vxTensorLayout>(layout);

    vx_size in_num_of_dims, out_num_of_dims;
    size_t in_tensor_dims[RPP_MAX_TENSOR_DIMS], out_tensor_dims[RPP_MAX_TENSOR_DIMS];

    // Querying for input tensor 
    data->pSrcDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &in_num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, in_tensor_dims, sizeof(vx_size) * in_num_of_dims));
    data->pSrcDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pSrcDesc, data->layout, in_tensor_dims);

    // Querying for output tensor
    data->pDstDesc = new RpptDesc;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &out_num_of_dims, sizeof(vx_size)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, out_tensor_dims, sizeof(vx_size) * out_num_of_dims));
    data->pDstDesc->offsetInBytes = 0;
    fillDescriptionPtrfromDims(data->pDstDesc, data->layout, out_tensor_dims);
    
    data->pSrcDesc->n = in_tensor_dims[0];
    data->sequenceLength = in_tensor_dims[1];

    data->pDstDesc->n = out_tensor_dims[0];
    data->newSequenceLength = out_tensor_dims[1];
    data->pNewOrder = new vx_uint32[data->newSequenceLength];
    refreshSequenceRearrange(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSequenceRearrange(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    SequenceRearrangeLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    delete[] data->pNewOrder;
    delete data->pSrcDesc;
    delete data->pDstDesc;
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

// hardcode the affinity to  CPU for OpenCL backend to avoid VerifyGraph failure since there is no codegen callback for amd_rpp nodes
#if ENABLE_OPENCL
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU;
#endif
    return VX_SUCCESS;
}

vx_status SequenceRearrange_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.SequenceRearrange",
                                       VX_KERNEL_RPP_SEQUENCEREARRANGE,
                                       processSequenceRearrange,
                                       5,
                                       validateSequenceRearrange,
                                       initializeSequenceRearrange,
                                       uninitializeSequenceRearrange);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
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
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
        exit:	vxRemoveKernel(kernel); return VX_FAILURE;
    }
    return status;
}
