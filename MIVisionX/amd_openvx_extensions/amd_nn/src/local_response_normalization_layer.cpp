/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

struct LocalResponseNormalizationLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenLRNMode_t mode;
    miopenLRNDescriptor_t lrnDesc;
    unsigned int normN;
    double normAlpha;
    double normBeta;
    double normBias;
    miopenTensorDescriptor_t input_desc;
    void *input_mem;
    miopenTensorDescriptor_t output_desc;
    void *output_mem;
    void *workspace;
    size_t workspace_size;
};

static vx_status VX_CALLBACK validateLocalResponseNormalizationLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_ENUM) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #1 type=%d (must be enum)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_SIZE) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #2 type=%d (must be size)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #3 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #4 type=%d (must be float)\n", type);
    if(parameters[6]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #6 type=%d (must be float)\n", type);
    }

    // check tensor dimensions
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: LRN: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: LRN: #5 num_dims=%ld (must be 4)\n", num_dims);
    if((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: LRN: #5 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: LRN: dims input[%ld,%ld,%ld,%ld] output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    // output tensor configuration
    out_type = type;        // should be same as input
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processLocalResponseNormalizationLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Local_Response_Normalization_Layer)
    LocalResponseNormalizationLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    float alpha = 1.0f, beta = 0.0f;
    //Apply Local Response Normalization forward.
    ERROR_CHECK_MIOPEN_STATUS(miopenLRNForward(miopenHandle, data->lrnDesc, &alpha, data->input_desc, data->input_mem, &beta, data->output_desc, data->output_mem, false, nullptr));

    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("local_response_normalization_%04d.bin", (vx_tensor)parameters[5]);
    #endif
PROFILER_STOP(VX_NN, Local_Response_Normalization_Layer)
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeLocalResponseNormalizationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    LocalResponseNormalizationLayerLocalData * data = new LocalResponseNormalizationLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));
    miopenDataType_t data_type;          // data_type for the kernel

    vx_size input_dims[4], output_dims[4];
    vx_enum out_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;

    vx_nn_norm_type_e type;
    vx_float32 alpha = 0, beta = 0, bias = 1;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->normN, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &beta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(parameters[6]){
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &bias, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    data->mode = miopenLRNCrossChannel;
    if (type == VX_NN_NORMALIZATION_SAME_MAP) {
        data->mode = miopenLRNWithinChannel;
    }
    else if (type == VX_NN_NORMALIZATION_ACROSS_MAPS) {
        data->mode = miopenLRNCrossChannel;
    }

    data->normAlpha = alpha;
    data->normBeta  = beta;
    data->normBias  = bias;

    //Input and Output descriptors.
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->input_desc)));
    ERROR_CHECK_MIOPEN_STATUS((miopenCreateTensorDescriptor(&data->output_desc)));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->input_desc, data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0])));
    ERROR_CHECK_MIOPEN_STATUS((miopenSet4dTensorDescriptor(data->output_desc, data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0])));

    //LRN Descriptor.
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateLRNDescriptor(&data->lrnDesc));
    ERROR_CHECK_MIOPEN_STATUS(miopenSetLRNDescriptor(data->lrnDesc, data->mode, data->normN, data->normAlpha, data->normBeta, data->normBias));

#if ENABLE_OPENCL
    //Input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    //Input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "lrn input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << "LRN Mode : " << data->mode << std::endl;
    std::cout << "Alpha " << data->normAlpha << " Beta " << data->normBeta << " N " << data->normN << " K " << data->normBias << std::endl;
    std::cout << "output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeLocalResponseNormalizationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    LocalResponseNormalizationLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyLRNDescriptor(data->lrnDesc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishLocalResponseNormalizationLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.local_reponse_normalization_layer", VX_KERNEL_LOCAL_RESPONSE_NORMALIZATION_LAYER, processLocalResponseNormalizationLayer, 7, validateLocalResponseNormalizationLayer, initializeLocalResponseNormalizationLayer, uninitializeLocalResponseNormalizationLayer);
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
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxLocalResponseNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_enum type,
                                                      vx_size normalization_size,
                                                      vx_float32 alpha,
                                                      vx_float32 beta,
                                                      vx_float32 bias,
                                                      vx_tensor outputs)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_type = vxCreateScalarWithSize(context, VX_TYPE_ENUM, &type, sizeof(type));
        vx_scalar s_normalization_size = vxCreateScalarWithSize(context, VX_TYPE_SIZE, &normalization_size, sizeof(normalization_size));
        vx_scalar s_alpha = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &alpha, sizeof(alpha));
        vx_scalar s_beta = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &beta, sizeof(beta));
        vx_scalar s_bias = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &bias, sizeof(bias));
        if(vxGetStatus((vx_reference)s_type) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_normalization_size) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_alpha) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_beta) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_bias) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)s_type,
                (vx_reference)s_normalization_size,
                (vx_reference)s_alpha,
                (vx_reference)s_beta,
                (vx_reference)s_bias,
                (vx_reference)outputs
            };
            node = createNode(graph, VX_KERNEL_LOCAL_RESPONSE_NORMALIZATION_LAYER, params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&s_type);
            vxReleaseScalar(&s_normalization_size);
            vxReleaseScalar(&s_alpha);
            vxReleaseScalar(&s_beta);
            vxReleaseScalar(&s_bias);
        }
    }
    return node;
}
