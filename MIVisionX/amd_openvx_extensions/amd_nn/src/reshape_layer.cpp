/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "kernels.h"

#include <stdio.h>
#include <sys/stat.h>
struct ReshapeLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    void *input_mem;
    void *output_mem;
    vx_bool aliased;
    vx_size memsizeInBytes;
};


static vx_status VX_CALLBACK validateReshapeLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input and output tensor dimensions
    vx_size num_dims;
    vx_enum type, out_type;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: reshape: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: reshape: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: reshape: #1 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: reshape: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // check if the input and output are of the same size in memory
    if ( (output_dims[0]*output_dims[1]*output_dims[2]*output_dims[3]) != (input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3]) || (out_type != type))
         return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: reshape: output_dims[%ldx%ldx%ldx%ld] input_dims[%ldx%ldx%ldx%ld]\n", output_dims[3], output_dims[2], output_dims[1], output_dims[0], input_dims[3], input_dims[2], input_dims[1], input_dims[0]);

    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    //alias output to input tensor for zero copy
    vxAliasTensor((vx_tensor)parameters[0], 0, (vx_tensor)parameters[1]);

    return VX_SUCCESS;
}


static vx_status VX_CALLBACK processReshapeLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Reshape_Layer)
    ReshapeLayerLocalData * data= NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    if (data->aliased == vx_false_e) {
#if ENABLE_OPENCL
        ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, (cl_mem)data->input_mem, (cl_mem)data->output_mem, 0, 0, data->memsizeInBytes, 0, NULL, NULL));
#elif ENABLE_HIP
        hipError_t errcode_ret = hipMemcpyDtoD(data->output_mem, data->input_mem, data->memsizeInBytes);
        if (errcode_ret != hipSuccess) {
            return VX_FAILURE;
        }
#endif
#if ENABLE_DEBUG_PRINT_DIMS
        std::cout << "Reshape Layer: not using aliased buffer "<< std::endl;
#endif
    } else {
#if ENABLE_DEBUG_PRINT_DIMS
        std::cout << "Reshape Layer: using aliased buffer "<< std::endl;
#endif
    }
PROFILER_STOP(VX_NN, Reshape_Layer)
    return VX_SUCCESS;
}


static vx_status VX_CALLBACK initializeReshapeLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_size dims[4];
    vx_enum type;
    ReshapeLayerLocalData * data = new ReshapeLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    // check if the input and output tensors are aliased
    data->aliased = vxIsTensorAliased((vx_tensor)parameters[0], 0, (vx_tensor)parameters[1]);
#if defined(AMD_FP16_SUPPORT)
    vx_size dataTypeSize = (type == VX_TYPE_FLOAT32) ? sizeof(vx_float32) : sizeof(vx_float16);
#else
    // TBD: Add vx_float16 support for all nodes using uint16
    vx_size dataTypeSize = (type == VX_TYPE_FLOAT32) ? sizeof(vx_float32) : sizeof(vx_uint16);
#endif

    data->memsizeInBytes = dims[0]*dims[1]*dims[2]*dims[3]* dataTypeSize;
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeReshapeLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ReshapeLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}


//! \brief The kernel publisher.
vx_status publishReshapeLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.reshape_layer", VX_KERNEL_RESHAPE_LAYER, processReshapeLayer, 2, validateReshapeLayer, initializeReshapeLayer, uninitializeReshapeLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxReshapeLayer(vx_graph graph, vx_tensor input, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_RESHAPE_LAYER, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
