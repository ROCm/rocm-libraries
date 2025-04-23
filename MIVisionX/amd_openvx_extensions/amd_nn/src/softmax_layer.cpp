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

struct SoftmaxLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    float alpha;
    float beta;
    miopenDataType_t data_type;          // data_type for the kernel
    miopenTensorDescriptor_t input_desc;
    void *input_mem;
    miopenTensorDescriptor_t output_desc;
    void *output_mem;
    int dim_in;
    int dim_out;
    vx_int32 axis;
};

static vx_status VX_CALLBACK validateSoftmaxLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dimensions
    vx_enum type, out_type;
    vx_size num_dims;
    vx_size input_dims[4] = { 1, 1, 1, 1 }, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: #0 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: softmax: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], num_dims*sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: #1 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: softmax: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims*sizeof(vx_size)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2] ||
        output_dims[1] != input_dims[1] || output_dims[0] != input_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: dims input[%ld,%ld,%ld,%ld] != output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    // output tensor configuration
    out_type = type;        // has to be same as input
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims*sizeof(vx_size)));

    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_INT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: softmax: #3 type=%d (must be VX_TYPE_INT32)\n", type);
        vx_int32 axis = 1;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processSoftmaxLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Softmax_Layer)
    SoftmaxLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;


#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    ERROR_CHECK_STATUS(miopenSoftmaxForward(miopenHandle, &data->alpha, data->input_desc, data->input_mem, &data->beta, data->output_desc, data->output_mem));


    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("softmax_%04d.bin", (vx_tensor)parameters[1]);
    #endif  

PROFILER_STOP(VX_NN, Softmax_Layer)

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeSoftmaxLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SoftmaxLayerLocalData * data = new SoftmaxLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //Parameters input and output.
    vx_enum out_type;
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 }, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], num_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    data->data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;

    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output_desc));

    data->axis = 1;
    if(parameters[2])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    if(data->axis == 1)
    {
        ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input_desc, data->data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0]));
        ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output_desc, data->data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));
    }
    else if(data->axis == 2)
    {
        ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input_desc, data->data_type, input_dims[3]*input_dims[2], input_dims[1], input_dims[0], 1));
        ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output_desc, data->data_type, output_dims[3]*output_dims[2], output_dims[1], output_dims[0],1));
    }

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    data->alpha = 1;
    data->beta = 0;

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "softmax input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << "output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSoftmaxLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SoftmaxLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishSoftmaxLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.softmax_layer", VX_KERNEL_SOFTMAX_LAYER, processSoftmaxLayer, 3, validateSoftmaxLayer, initializeSoftmaxLayer, uninitializeSoftmaxLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxSoftmaxLayer(vx_graph graph, vx_tensor inputs, vx_tensor outputs)
{
    vx_reference params[] = {
        (vx_reference)inputs,
        (vx_reference)outputs
    };
    return createNode(graph, VX_KERNEL_SOFTMAX_LAYER, params, sizeof(params)/sizeof(params[0]));
}
