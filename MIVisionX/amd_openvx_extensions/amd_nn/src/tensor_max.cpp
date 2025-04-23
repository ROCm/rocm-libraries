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

struct TensorMaxLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenTensorOp_t operation;
    float alpha1;
    float alpha2;
    float beta;
    miopenTensorDescriptor_t input1;
    void *input1_mem;
    miopenTensorDescriptor_t input2;
    void *input2_mem;
    miopenTensorDescriptor_t output;
    void *output_mem;

};

static vx_status VX_CALLBACK validateTensorMax(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_enum type, out_type;
    // check tensor dimensions
    vx_size num_dims;
    vx_size input1_dims[4],input2_dims[4] = { 1, 1, 0, 0 }, output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: max: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: max: #0 tensor type=%d (not float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: max: #1 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: max: #1 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &input2_dims[4-num_dims], num_dims * sizeof(vx_size)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: max: #2 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: max: #2 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if (output_dims[3] != input1_dims[3] || output_dims[2] != input1_dims[2] ||
        output_dims[1] != input1_dims[1] || output_dims[0] != input1_dims[0] ||
        output_dims[2] != input2_dims[2] || type != out_type ||
        !((             1 == input2_dims[3] &&              1 == input2_dims[1] &&              1 == input2_dims[0]) ||
          (output_dims[3] == input2_dims[3] && output_dims[1] == input2_dims[1] && output_dims[0] == input2_dims[0])))
    {
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: max: dims input1[%ld,%ld,%ld,%ld] input2[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3],
                    input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    // output tensor configuration
    out_type = type;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processTensorMax(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    TensorMaxLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    //miopen elementwise max call.
    ERROR_CHECK_MIOPEN_STATUS(miopenOpTensor(miopenHandle, data->operation, &data->alpha1, data->input1, data->input1_mem, &data->alpha2, data->input2, data->input2_mem, &data->beta, data->output, data->output_mem));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeTensorMax(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorMaxLocalData * data = new TensorMaxLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //initialize input and output tensor descriptors.
    vx_enum type;
    miopenDataType_t data_type;          // data_type for the kernel
    vx_size input1_dims[4], num_dims, input2_dims[4] = { 1, 1, 0, 0 }, output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &input2_dims[4-num_dims], num_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    data_type = (type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input1));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input2));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input1, data_type, input1_dims[3], input1_dims[2], input1_dims[1], input1_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input2, data_type, input2_dims[3], input2_dims[2], input2_dims[1], input2_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output, data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));

    //scaling parameters.
    data->alpha1 = 1;
    data->alpha2 = 1;
    data->beta = 0;
    data->operation = miopenTensorOpMax;

#if ENABLE_OPENCL
    //input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    //input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "tensor_max input1 " << input1_dims[3] << " " << input1_dims[2] << " " << input1_dims[1] << " " << input1_dims[0] << " ";
    std::cout << "tensor_max input2 " << input2_dims[3] << " " << input2_dims[2] << " " << input2_dims[1] << " " << input2_dims[0] << " ";
    std::cout << "tensor_max output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTensorMax(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    TensorMaxLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input1));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input2));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishTensorMax(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.tensor_max", VX_KERNEL_TENSOR_MAX_AMD, processTensorMax, 3, validateTensorMax, initializeTensorMax, uninitializeTensorMax);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorMaxNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)output
        };
        node = createNode(graph, VX_KERNEL_TENSOR_MAX_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}

