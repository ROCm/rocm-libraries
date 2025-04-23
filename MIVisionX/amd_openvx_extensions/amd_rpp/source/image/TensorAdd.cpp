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

struct TensorAddLocalData
{
    vxRppHandle *handle;
    Rpp32u device_type;
    Rpp8u *pSrc1;
    Rpp8u *pSrc2;
    Rpp8u *pDst;
    Rpp32u tensorDimensions;
    Rpp32u *tensorDimensionsValue;
#if ENABLE_OPENCL
    cl_mem cl_pSrc1;
    cl_mem cl_pSrc2;
    cl_mem cl_pDst;
#elif ENABLE_HIP
    void *hip_pSrc1;
    void *hip_pSrc2;
    void *hip_pDst;
#endif
};

static vx_status VX_CALLBACK refreshTensorAdd(vx_node node, const vx_reference *parameters, vx_uint32 num, TensorAddLocalData *data)
{
    vx_status status = VX_SUCCESS;
    size_t arr_size;
    vx_status copy_status;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->pSrc1 = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[0], 0, arr_size, sizeof(Rpp8u), data->pSrc1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->pSrc2 = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
    data->pDst = (Rpp8u *)malloc(sizeof(Rpp8u) * arr_size);
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[1], 0, arr_size, sizeof(Rpp8u), data->pSrc2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[3], &data->tensorDimensions));
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[4], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    data->tensorDimensionsValue = (Rpp32u *)malloc(sizeof(Rpp32u) * arr_size);
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[4], 0, arr_size, sizeof(Rpp32u), data->tensorDimensionsValue, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        cl_context theContext;
        cl_command_queue theQueue;
        theQueue = data->handle->cmdq;
        clGetCommandQueueInfo(theQueue,
                              CL_QUEUE_CONTEXT,
                              sizeof(cl_context), &theContext, NULL);
        cl_int err;
        STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
        size_t bytes = arr_size * sizeof(Rpp8u);
        err = clEnqueueWriteBuffer(theQueue, data->cl_pSrc1, CL_TRUE, 0,
                                   bytes, data->pSrc1, 0, NULL, NULL);
        if (err)
            return VX_FAILURE;
        err = clEnqueueWriteBuffer(theQueue, data->cl_pSrc2, CL_TRUE, 0,
                                   bytes, data->pSrc2, 0, NULL, NULL);
        if (err)
            return VX_FAILURE;
#elif ENABLE_HIP
        STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
        size_t bytes = arr_size * sizeof(Rpp8u);
        hipError_t err;
        err = hipMemcpy(data->hip_pSrc1, data->pSrc1, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            return VX_FAILURE;
        err = hipMemcpy(data->hip_pSrc2, data->pSrc2, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
            return VX_FAILURE;
#endif
    }

    return status;
}

static vx_status VX_CALLBACK validateTensorAdd(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;
    size_t arr_size;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #3 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &scalar_type, sizeof(scalar_type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[2], VX_ARRAY_ITEMTYPE, &scalar_type, sizeof(scalar_type)));
    return status;
}

static vx_status VX_CALLBACK processTensorAdd(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    TensorAddLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    size_t arr_size;
    if (data->device_type == AGO_TARGET_AFFINITY_GPU)
    {
#if ENABLE_OPENCL
        refreshTensorAdd(node, parameters, num, data);
        rpp_status = rppi_tensor_add_u8_gpu(static_cast<void *>(data->cl_pSrc1), static_cast<void *>(data->cl_pSrc2), static_cast<void *>(data->cl_pDst), data->tensorDimensions, data->tensorDimensionsValue, data->handle->rppHandle);
        cl_command_queue theQueue;
        theQueue = data->handle->cmdq;
        cl_int err;
        STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
        size_t bytes = arr_size * sizeof(Rpp8u);
        clEnqueueReadBuffer(theQueue, data->cl_pDst, CL_TRUE, 0, bytes, data->pDst, 0, NULL, NULL);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#elif ENABLE_HIP
        refreshTensorAdd(node, parameters, num, data);
        rpp_status = rppi_tensor_add_u8_gpu(static_cast<void *>(data->hip_pSrc1), static_cast<void *>(data->hip_pSrc2), static_cast<void *>(data->hip_pDst), data->tensorDimensions, data->tensorDimensionsValue, data->handle->rppHandle);
        hipError_t err;
        STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
        size_t bytes = arr_size * sizeof(Rpp8u);
        err = hipMemcpy(data->pDst, data->hip_pDst, bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess)
            return VX_FAILURE;
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#endif
    }
    if (data->device_type == AGO_TARGET_AFFINITY_CPU)
    {
        refreshTensorAdd(node, parameters, num, data);
        rpp_status = rppi_tensor_add_u8_host(data->pSrc1, data->pSrc2, data->pDst, data->tensorDimensions, data->tensorDimensionsValue, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
    }
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[2], 0, arr_size, sizeof(Rpp8u), data->pDst, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    return return_status;
}

static vx_status VX_CALLBACK initializeTensorAdd(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddLocalData *data = new TensorAddLocalData;
    memset(data, 0, sizeof(*data));
#if ENABLE_OPENCL
    cl_context theContext;     // theContext
    cl_command_queue theQueue; // command theQueue
    theQueue = data->handle->cmdq;
    clGetCommandQueueInfo(theQueue,
                          CL_QUEUE_CONTEXT,
                          sizeof(cl_context), &theContext, NULL);
    size_t arr_size;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    size_t bytes = arr_size * sizeof(Rpp8u);
    data->cl_pSrc1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    data->cl_pSrc2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    data->cl_pDst = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
#elif ENABLE_HIP
    size_t arr_size;
    STATUS_ERROR_CHECK(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
    size_t bytes = arr_size * sizeof(Rpp8u);
    hipError_t status;
    status = hipMalloc(&data->hip_pSrc1, bytes);
    if (status != hipSuccess)
        return VX_FAILURE;
    status = hipMalloc(&data->hip_pSrc2, bytes);
    if (status != hipSuccess)
        return VX_FAILURE;
    status = hipMalloc(&data->hip_pDst, bytes);
    if (status != hipSuccess)
        return VX_FAILURE;
#endif
    STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[5], &data->device_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    refreshTensorAdd(node, parameters, num, data);
    STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, 1, data->device_type));

    STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorAdd(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorAddLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->device_type));
    delete (data);
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hubrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
                                                  vx_uint32 &supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
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

vx_status TensorAdd_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.TensorAdd",
                                       VX_KERNEL_RPP_TENSORADD,
                                       processTensorAdd,
                                       6,
                                       validateTensorAdd,
                                       initializeTensorAdd,
                                       uninitializeTensorAdd);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_OPENCL || ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    if (kernel)
    {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS)
    {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
