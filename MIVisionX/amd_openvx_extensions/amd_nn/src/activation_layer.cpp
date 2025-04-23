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
struct ActivationLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenActivationMode_t mode;
    miopenDataType_t data_type;          // data_type for the kernel
    double activAlpha;
    double activBeta;
    double activPower;
    miopenTensorDescriptor_t inputDescriptor;
    miopenTensorDescriptor_t outputDescriptor;
    miopenActivationDescriptor_t activationDesc;
    void* input_mem;
    void* output_mem;
};

static vx_status VX_CALLBACK validateActivationLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: activation: #1 scalar type=%d (not enum)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: activation: #2 scalar type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: activation: #3 scalar type=%d (not float)\n", type);

    // check tensor dimensions
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: activation: #0 num_dims=%ld (must be 4)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: activation: #0 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: activation: #4 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: activation: #4 tensor type=%d (not float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2] ||
        output_dims[1] != input_dims[1] || output_dims[0] != input_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: activation: dims input[%ld,%ld,%ld,%ld] != output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    // output tensor configuration
    out_type = type;        // has to be same as input
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processActivationLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Activation_Layer)
    ActivationLayerLocalData * data= NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#else
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    float alpha = 1.0f, beta = 0.0f;
    //miopen activation forward call.
    ERROR_CHECK_MIOPEN_STATUS((miopenActivationForward(miopenHandle, data->activationDesc, &alpha, data->inputDescriptor, data->input_mem, &beta, data->outputDescriptor, data->output_mem)));

    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("activation_%04d.bin", (vx_tensor)parameters[4]);
    #endif  
PROFILER_STOP(VX_NN, Activation_Layer)
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeActivationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ActivationLayerLocalData * data = new ActivationLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //initializing input and output Descriptors.
    vx_size input_dims[4], output_dims[4];
    vx_enum out_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    data->data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->inputDescriptor)));
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->outputDescriptor)));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->inputDescriptor, data->data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0])));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->outputDescriptor, data->data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0])));

    //activation Function Type
    vx_int32 activationMode;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &activationMode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    data->activAlpha = 1.0f;
    data->activPower = 1.0f;
    vx_float32 neg_slope = 0.0f;

    if (activationMode == VX_NN_ACTIVATION_RELU) {
        data->mode = miopenActivationRELU;
    }
    else if (activationMode == VX_NN_ACTIVATION_ABS) {
        data->mode = miopenActivationABS;
    }
    else if (activationMode == VX_NN_ACTIVATION_LOGISTIC) {
        data->mode = miopenActivationLOGISTIC;
    }
    else if (activationMode == VX_NN_ACTIVATION_HYPERBOLIC_TAN) {
        data->mode = miopenActivationTANH;
    }
    else if (activationMode == VX_NN_ACTIVATION_SOFTRELU) {
        data->mode = miopenActivationSOFTRELU;
    }
    else if (activationMode == VX_NN_ACTIVATION_LEAKY_RELU) {
        data->mode = miopenActivationLEAKYRELU;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &neg_slope, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        data->activAlpha = neg_slope;
    }

    //activation Descriptor.
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateActivationDescriptor(&data->activationDesc)));
    ERROR_CHECK_MIOPEN_STATUS((miopenSetActivationDescriptor(data->activationDesc, data->mode, data->activAlpha, data->activBeta, data->activPower)));

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "activation param active_alpha: " << data->activAlpha << "active_beta: " << data->activBeta << "activationmode: " << activationMode << std::endl;
    std::cout << "activation input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << "output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeActivationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ActivationLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyActivationDescriptor(data->activationDesc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->inputDescriptor));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->outputDescriptor));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishActivationLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.activation_layer", VX_KERNEL_ACTIVATION_LAYER, processActivationLayer, 5, validateActivationLayer, initializeActivationLayer, uninitializeActivationLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable GPU buffer access since the kernel_f callback uses GPU buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxActivationLayer(vx_graph graph, vx_tensor inputs, vx_enum function, vx_float32 a, vx_float32 b, vx_tensor outputs)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_function = vxCreateScalarWithSize(context, VX_TYPE_ENUM, &function, sizeof(function));
        vx_scalar s_a = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &a, sizeof(a));
        vx_scalar s_b = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &b, sizeof(b));
        if (vxGetStatus((vx_reference)s_function) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_a) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_b) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)s_function,
                (vx_reference)s_a,
                (vx_reference)s_b,
                (vx_reference)outputs
            };
            node = createNode(graph, VX_KERNEL_ACTIVATION_LAYER, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_function);
            vxReleaseScalar(&s_a);
            vxReleaseScalar(&s_b);
        }
    }
    return node;
}
