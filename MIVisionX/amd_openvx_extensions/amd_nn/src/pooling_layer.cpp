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

struct PoolingLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenPoolingDescriptor_t pool_desc;
    float alpha;
    float beta;
    miopenTensorDescriptor_t input_desc;
    miopenTensorDescriptor_t output_desc;
    miopenDataType_t data_type;          // data_type for the kernel
    void *input_mem;
    void *output_mem;
    void *pooling_workspace;
    size_t pooling_workspace_size;
    miopenPoolingMode_t mode;
    vx_enum pad_border_mode;
    miopenActivationMode_t activation_mode;
    double activation_alpha;
    double activation_beta;
    double activation_power;
    miopenActivationDescriptor_t activation_desc;
};

static vx_status VX_CALLBACK validatePoolingLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #1 type=%d (must be enum)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_SIZE) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #2 type=%d (must be size)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_SIZE) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #3 type=%d (must be size)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_SIZE) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #4 type=%d (must be size)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_SIZE) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #5 type=%d (must be size)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #6 type=%d (must be enum)\n", type);

    // check tensor dimensions
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: POOL: #0 num_dims=%ld (must be 4)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: POOL: #7 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #7 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: POOL: dims input[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    out_type = type;        // has to be same as input

    vx_enum pad_border_mode = 0;
    if(parameters[8]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[8], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #8 type=%d (must be enum)\n", type);
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &pad_border_mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(pad_border_mode != 0 && pad_border_mode != 1) {
            return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #8 pad_border_mode=%d (must be 0 or 1)\n", pad_border_mode);
        }
    }
    if(parameters[9]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[9], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_INT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #8 type=%d (must be VX_TYPE_INT32)\n", type);
        vx_int32 activation_mode = 0;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &activation_mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if(activation_mode != 0 && activation_mode != 1) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: POOL: #5 activation_mode=%d (must be 0 or 1)\n", activation_mode);
    }

    // output tensor configuration
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processPoolingLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Pooling_Layer)
    PoolingLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    ERROR_CHECK_MIOPEN_STATUS(miopenPoolingForward(miopenHandle, data->pool_desc, &data->alpha, data->input_desc, data->input_mem, &data->beta, data->output_desc, data->output_mem, false, nullptr, 0));

    // activation (in-place in output_mem)
    if(parameters[9]) {
        float alpha = 1.0f, beta = 0.0f;
        ERROR_CHECK_MIOPEN_STATUS(miopenActivationForward(data->handle->miopen_handle, data->activation_desc, &alpha, data->output_desc, data->output_mem, &beta, data->output_desc, data->output_mem));
    }


    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("pooling_%04d.bin", (vx_tensor)parameters[7]);
    #endif

PROFILER_STOP(VX_NN, Pooling_Layer)

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializePoolingLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    PoolingLayerLocalData * data = new PoolingLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //Deducing the pooling type.
    vx_nn_pooling_type_e modeType;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &modeType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (modeType == VX_NN_POOLING_MAX) {
        data->mode = miopenPoolingMax;
    }
    else if (modeType == VX_NN_POOLING_AVG) {
        data->mode = miopenPoolingAverage;
    }
    vx_enum pad_border_mode = 0;
    if(parameters[8]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[8], &pad_border_mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    data->pad_border_mode = pad_border_mode;

    //Set Descriptors.
    vx_size kernel_w;
    vx_size kernel_h;
    vx_size pad_w;
    vx_size pad_h;
    vx_size stride_w;
    vx_size stride_h;
    vx_size input_dims[4], output_dims[4];
    vx_enum out_type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &kernel_w, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &kernel_h, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &pad_w, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &pad_h, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    stride_w = (output_dims[0] > 1) ? ((input_dims[0] + 2 * pad_w - kernel_w + ((output_dims[0] - 1) / 2)) / (output_dims[0] - 1)) : 1;
    stride_h = (output_dims[1] > 1) ? ((input_dims[1] + 2 * pad_h - kernel_h + ((output_dims[1] - 1) / 2)) / (output_dims[1] - 1)) : 1;
    data->data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;

    ERROR_CHECK_MIOPEN_STATUS(miopenCreatePoolingDescriptor(&data->pool_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet2dPoolingDescriptor(data->pool_desc, data->mode, kernel_h, kernel_w, pad_h, pad_w, stride_h , stride_w));
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->input_desc)));
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->output_desc)));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->input_desc, data->data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0])));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->output_desc, data->data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0])));

#if ENABLE_OPENCL
    //Declare Memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    //Declare Memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[7], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    data->alpha = 1;
    data->beta = 0;

    // activation descriptor
    vx_int32 activation_mode = 0;
    if(parameters[9]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[9], &activation_mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    data->activation_mode = miopenActivationPASTHRU;
    if(activation_mode == 1) {
        data->activation_mode = miopenActivationRELU;
        data->activation_alpha = 1.0;
        data->activation_beta = 0.0;
        data->activation_power = 1.0;
    }
    if(data->activation_mode == miopenActivationRELU) {
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateActivationDescriptor(&data->activation_desc));
        ERROR_CHECK_MIOPEN_STATUS(miopenSetActivationDescriptor(data->activation_desc, data->activation_mode, data->activation_alpha, data->activation_beta, data->activation_power));
    }

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "pooling input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << "kernel " << kernel_h << " " << kernel_w << " ";
    std::cout << "stride " << stride_h << " " << stride_w << " " << "pad " << pad_h << " " << pad_w;
    std::cout << " output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializePoolingLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    PoolingLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyPoolingDescriptor(data->pool_desc));
    if(data->activation_mode != miopenActivationPASTHRU) {
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyActivationDescriptor(data->activation_desc));
    }
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishPoolingLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.pooling_layer", VX_KERNEL_POOLING_LAYER, processPoolingLayer, 10, validatePoolingLayer, initializePoolingLayer, uninitializePoolingLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxPoolingLayer(vx_graph graph, vx_tensor inputs, vx_enum pooling_type,
                                                vx_size pooling_size_x,
                                                vx_size pooling_size_y,
                                                vx_size pooling_padding_x,
                                                vx_size pooling_padding_y,
                                                vx_enum rounding,
                                                vx_tensor outputs)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_pooling_type = vxCreateScalarWithSize(context, VX_TYPE_ENUM, &pooling_type, sizeof(pooling_type));
        vx_scalar s_pooling_size_x = vxCreateScalarWithSize(context, VX_TYPE_SIZE, &pooling_size_x, sizeof(pooling_size_x));
        vx_scalar s_pooling_size_y = vxCreateScalarWithSize(context, VX_TYPE_SIZE, &pooling_size_y, sizeof(pooling_size_y));
        vx_scalar s_pooling_padding_x = vxCreateScalarWithSize(context, VX_TYPE_SIZE, &pooling_padding_x, sizeof(pooling_padding_x));
        vx_scalar s_pooling_padding_y = vxCreateScalarWithSize(context, VX_TYPE_SIZE, &pooling_padding_y, sizeof(pooling_padding_y));
        vx_scalar s_rounding = vxCreateScalarWithSize(context, VX_TYPE_ENUM, &rounding, sizeof(rounding));
        if(vxGetStatus((vx_reference)s_pooling_type) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_pooling_size_x) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_pooling_size_y) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_pooling_padding_x) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_pooling_padding_y) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_rounding) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)s_pooling_type,
                (vx_reference)s_pooling_size_x,
                (vx_reference)s_pooling_size_y,
                (vx_reference)s_pooling_padding_x,
                (vx_reference)s_pooling_padding_y,
                (vx_reference)s_rounding,
                (vx_reference)outputs
            };
            node = createNode(graph, VX_KERNEL_POOLING_LAYER, params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&s_pooling_type);
            vxReleaseScalar(&s_pooling_size_x);
            vxReleaseScalar(&s_pooling_size_y);
            vxReleaseScalar(&s_pooling_padding_x);
            vxReleaseScalar(&s_pooling_padding_y);
            vxReleaseScalar(&s_rounding);
        }
    }
    return node;
}
