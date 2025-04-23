/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

struct SequenceRearrangebatchPDLocalData
{
    vxRppHandle *handle;
    RppiSize dimensions;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    Rpp32u device_type;
    vx_uint32 new_sequence_length;
    vx_uint32 sequence_length;
    vx_uint32 sequence_count;
    vx_uint32 *new_order;
#if ENABLE_OPENCL
    cl_mem cl_pSrc;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK validateSequenceRearrangebatchPD(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #4 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #5 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    vx_parameter param = vxGetParameterByIndex(node, 1);
    vx_image image;
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
    STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_RGB)
        status = VX_ERROR_INVALID_VALUE;
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &df_image, sizeof(df_image)));
    vx_uint32 height, width;
    STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    vxReleaseImage(&image);
    return status;
}

static vx_status VX_CALLBACK processSequenceRearrangebatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SequenceRearrangebatchPDLocalData *data = NULL;
    vx_status return_status = VX_SUCCESS;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    vx_df_image df_image = VX_DF_IMAGE_VIRT;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
#if ENABLE_OPENCL
        cl_command_queue handle = data->handle->cmdq;
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc, sizeof(data->cl_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pDst, sizeof(data->cl_pDst)));
        unsigned size = data->dimensions.height * data->dimensions.width;
        if (df_image == VX_DF_IMAGE_U8)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count));
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_offset = dst_sequence_start_address + (dst_index * elem_size);
                    auto src_offset = src_sequence_start_address + (src_index * elem_size);
                    if (clEnqueueCopyBuffer(handle, data->cl_pSrc, data->cl_pDst, src_offset, dst_offset, elem_size, 0, NULL, NULL) != CL_SUCCESS)
                        return VX_FAILURE;
                }
            }
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count)) * 3;
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_offset = dst_sequence_start_address + (dst_index * elem_size);
                    auto src_offset = src_sequence_start_address + (src_index * elem_size);
                    if (clEnqueueCopyBuffer(handle, data->cl_pSrc, data->cl_pDst, src_offset, dst_offset, elem_size, 0, NULL, NULL) != CL_SUCCESS)
                        return VX_FAILURE;
                }
            }
        }
        return_status = VX_SUCCESS;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pSrc, sizeof(data->hip_pSrc)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pDst, sizeof(data->hip_pDst)));
        unsigned size = data->dimensions.height * data->dimensions.width;
        if (df_image == VX_DF_IMAGE_U8)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count));
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_address = (unsigned char *)data->hip_pDst + dst_sequence_start_address + (dst_index * elem_size);
                    auto src_address = (unsigned char *)data->hip_pSrc + src_sequence_start_address + (src_index * elem_size);
                    hipError_t status = hipMemcpyDtoD(dst_address, src_address, elem_size);
                    if (status != hipSuccess)
                        return VX_FAILURE;
                }
            }
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count)) * 3;
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_address = (unsigned char *)data->hip_pDst + dst_sequence_start_address + (dst_index * elem_size);
                    auto src_address = (unsigned char *)data->hip_pSrc + src_sequence_start_address + (src_index * elem_size);
                    hipError_t status = hipMemcpyDtoD(dst_address, src_address, elem_size);
                    if (status != hipSuccess)
                        return VX_FAILURE;
                }
            }
        }
        return_status = VX_SUCCESS;
#endif
    }
    else if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pSrc, sizeof(vx_uint8)));
        STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, &data->pDst, sizeof(vx_uint8)));
        unsigned size = data->dimensions.height * data->dimensions.width;
        if (df_image == VX_DF_IMAGE_U8)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count));
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_address = (unsigned char *)data->pDst + dst_sequence_start_address + (dst_index * elem_size);
                    auto src_address = (unsigned char *)data->pSrc + src_sequence_start_address + (src_index * elem_size);
                    memcpy(dst_address, src_address, elem_size);
                }
            }
        }
        else if (df_image == VX_DF_IMAGE_RGB)
        {
            unsigned elem_size = (size / (data->sequence_length * data->sequence_count)) * 3;
            for (int sequence_cnt = 0; sequence_cnt < data->sequence_count; sequence_cnt++)
            {
                unsigned src_sequence_start_address = sequence_cnt * elem_size * data->sequence_length;
                unsigned dst_sequence_start_address = sequence_cnt * elem_size * data->new_sequence_length;
                for (unsigned dst_index = 0; dst_index < (data->new_sequence_length); dst_index++)
                {
                    unsigned src_index = data->new_order[dst_index];
                    if (src_index > data->sequence_length)
                        ERRMSG(VX_ERROR_INVALID_VALUE, "invalid new order value=%d (must be between 0-%d)\n", src_index, data->sequence_length - 1);
                    auto dst_address = (unsigned char *)data->pDst + dst_sequence_start_address + (dst_index * elem_size);
                    auto src_address = (unsigned char *)data->pSrc + src_sequence_start_address + (src_index * elem_size);
                    memcpy(dst_address, src_address, elem_size);
                }
            }
        }
        return_status = VX_SUCCESS;
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeSequenceRearrangebatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SequenceRearrangebatchPDLocalData *data = new SequenceRearrangebatchPDLocalData;
    memset(data, 0, sizeof(*data));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &data->dimensions.height, sizeof(data->dimensions.height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &data->dimensions.width, sizeof(data->dimensions.width)));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[3], &data->new_sequence_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[4], &data->sequence_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->sequence_count, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[6], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->new_order = (vx_uint32 *)malloc(sizeof(vx_uint32) * data->new_sequence_length);
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, data->new_sequence_length, sizeof(vx_uint32), data->new_order, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#if ENABLE_OPENCL
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_OPENCL_BUFFER, &data->cl_pSrc, sizeof(data->cl_pSrc)));
#elif ENABLE_HIP
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &data->hip_pSrc, sizeof(data->hip_pSrc)));
#else
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HOST_BUFFER, data->pSrc, sizeof(data->pSrc)));
#endif
    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSequenceRearrangebatchPD(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    return VX_SUCCESS;
}

vx_status SequenceRearrangebatchPD_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.SequenceRearrangebatchPD",
                                       VX_KERNEL_RPP_SEQUENCEREARRANGEBATCHPD,
                                       processSequenceRearrangebatchPD,
                                       7,
                                       validateSequenceRearrangebatchPD,
                                       initializeSequenceRearrangebatchPD,
                                       uninitializeSequenceRearrangebatchPD);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    if (kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
        exit:	vxRemoveKernel(kernel); return VX_FAILURE;
    }
    return status;
}
