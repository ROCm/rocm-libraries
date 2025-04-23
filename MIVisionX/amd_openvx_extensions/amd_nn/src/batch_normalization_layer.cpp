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

struct BatchNormLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenTensorDescriptor_t input_desc;
    void *input_mem;
    miopenTensorDescriptor_t output_desc;
    miopenDataType_t data_type;          // data_type for the kernel
    void *output_mem;
    void *workspace;
    size_t workspace_size;
    float alpha, beta, eps;
    miopenTensorDescriptor_t bnScaleBiasMeanVarDesc;
    void *bnScale, *bnBias, *bnMean, *bnVariance;
};

static vx_status VX_CALLBACK validateBatchNormalizationLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dimensions
    vx_enum type, in_type, out_type;
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_type, sizeof(in_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #0 num_dims=%ld (must be 4)\n", num_dims);
    if ((in_type != VX_TYPE_FLOAT32) && (in_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #0 type=%d (must be float)\n", in_type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #6 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #6 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2] ||
        output_dims[1] != input_dims[1] || output_dims[0] != input_dims[0] || out_type != in_type)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: dims input[%ld,%ld,%ld,%ld] type[%d] != output[%ld,%ld,%ld,%ld] type[%d]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3], in_type,
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3], out_type);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1 && num_dims != 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #1 num_dims=%ld (must be 1 or 2)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32)&& (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #1 type=%d (must be float)\n", type);
    vx_size mean_dims[2] = { 0, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, mean_dims, num_dims * sizeof(vx_size)));
    if (mean_dims[0] != input_dims[2]) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: mean[0](%ld) != input[2](%ld)\n", mean_dims[0], input_dims[2]);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1 && num_dims != 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #2 num_dims=%ld (must be 1 or 2)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32)&& (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #2 type=%d (must be float)\n", type);
    vx_size variance_dims[2] = { 0, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, variance_dims, num_dims * sizeof(vx_size)));
    if (variance_dims[0] != input_dims[2]) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: variance[0](%ld) != input[2](%ld)\n", variance_dims[0], input_dims[2]);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 1 && num_dims != 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #3 num_dims=%ld (must be 1 or 2)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32)&& (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #3 type=%d (must be float)\n", type);
    vx_size scale_dims[2] = { 0, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, scale_dims, num_dims * sizeof(vx_size)));
    if (scale_dims[0] != input_dims[2]) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: scale[0](%ld) != input[2](%ld)\n", scale_dims[0], input_dims[2]);

    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 1 && num_dims != 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: #4 num_dims=%ld (must be 1 or 2)\n", num_dims);
        if ((type != VX_TYPE_FLOAT32)&& (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: batch_norm: #4 type=%d (must be float)\n", type);
        vx_size bias_dims[2] = { 0, 1 };
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, bias_dims, num_dims * sizeof(vx_size)));
        if (bias_dims[0] != input_dims[2]) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: batch_norm: bias[0](%ld) != input[2](%ld)\n", bias_dims[0], input_dims[2]);
    }

    // output tensor configuration.
    out_type = in_type;        // should be same as input type
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processBatchNormalizationLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Batch_Normalization_Layer)
    BatchNormLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
#endif

    // miopen batch norm inference.
    ERROR_CHECK_MIOPEN_STATUS(miopenBatchNormalizationForwardInference(miopenHandle, miopenBNSpatial, &data->alpha, &data->beta, data->input_desc, data->input_mem,
        data->output_desc, data->output_mem, data->bnScaleBiasMeanVarDesc, data->bnScale, data->bnBias, data->bnMean, data->bnVariance, data->eps));

    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("bn_%04d.bin", (vx_tensor)parameters[6]);
    #endif  

PROFILER_STOP(VX_NN, Batch_Normalization_Layer)
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeBatchNormalizationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    BatchNormLayerLocalData * data = new BatchNormLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    // initialize input and output tensor descriptors.
    vx_size input_dims[4], output_dims[4];
    vx_enum out_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    data->data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->bnScaleBiasMeanVarDesc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input_desc, data->data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->bnScaleBiasMeanVarDesc, data->data_type, 1, input_dims[2], 1, 1));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output_desc, data->data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenDeriveBNTensorDescriptor(data->bnScaleBiasMeanVarDesc, data->input_desc, miopenBNSpatial));

    data->alpha = 1; data->beta = 0;

#if ENABLE_OPENCL
    // input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->bnMean, sizeof(data->bnMean)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->bnVariance, sizeof(data->bnVariance)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->bnScale, sizeof(data->bnScale)));
    if(parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->bnBias, sizeof(data->bnBias)));
    }
    else {
        vx_context   vxContext = vxGetContext((vx_reference)node);
        cl_context context;
        ERROR_CHECK_STATUS(vxQueryContext(vxContext, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &context, sizeof(context)));
        cl_float pattern = 0; cl_int err = 0;
        data->bnBias = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*input_dims[2], NULL, &err);

        if (err) return VX_FAILURE;
        if (data->data_type == miopenFloat)
            err = clEnqueueFillBuffer(data->handle->cmdq, (cl_mem)data->bnBias, &pattern, sizeof(cl_float), 0, input_dims[2], 0, NULL, NULL);
        else
            err = clEnqueueFillBuffer(data->handle->cmdq, (cl_mem)data->bnBias, &pattern, sizeof(cl_half), 0, input_dims[2], 0, NULL, NULL);

        if (err) return VX_FAILURE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
#elif ENABLE_HIP
    // input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->bnMean, sizeof(data->bnMean)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->bnVariance, sizeof(data->bnVariance)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HIP, &data->bnScale, sizeof(data->bnScale)));
    if(parameters[4]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->bnBias, sizeof(data->bnBias)));
    } else {
        vx_context vxContext = vxGetContext((vx_reference)node);
        int hip_device = -1;
        ERROR_CHECK_STATUS(vxQueryContext(vxContext, VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &hip_device, sizeof(hip_device)));
        if (hip_device < 0) {
            return VX_FAILURE;
        }
        hipError_t errcode_ret = hipSuccess;
        errcode_ret = hipMalloc(&data->bnBias, sizeof(float)*input_dims[2]);
        if (errcode_ret != hipSuccess) {
            return VX_FAILURE;
        }

        errcode_ret = hipMemset(data->bnBias, 0, input_dims[2]);
        if (errcode_ret != hipSuccess) {
            return VX_FAILURE;
        }
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));

#endif

    data->eps = 0.00001;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &data->eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "batch_normalization input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << " output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeBatchNormalizationLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    BatchNormLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->bnScaleBiasMeanVarDesc));
        if(!parameters[4]){
            if(data->bnBias) {
#if ENABLE_OPENCL
                cl_int err = clReleaseMemObject((cl_mem)data->bnBias);
                if (err) return VX_FAILURE;
#elif ENABLE_HIP
            hipError_t errcode_ret = hipFree(data->bnBias);
            if (errcode_ret != hipSuccess) {
                return VX_FAILURE;
            }
#endif
            }
        }
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishBatchNormalizationLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.batch_normalization_layer", VX_KERNEL_BATCH_NORMALISATION_LAYER_AMD, processBatchNormalizationLayer, 4, validateBatchNormalizationLayer, initializeBatchNormalizationLayer, uninitializeBatchNormalizationLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxBatchNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_tensor mean, vx_tensor variance, vx_tensor scale, vx_tensor bias, vx_float32 eps, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_eps = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &eps, sizeof(eps));
        if(vxGetStatus((vx_reference)s_eps) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)mean,
                (vx_reference)variance,
                (vx_reference)scale,
                (vx_reference)bias,
                (vx_reference)s_eps,
                (vx_reference)output
            };
            node = createNode(graph, VX_KERNEL_BATCH_NORMALISATION_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_eps);
        }
    }
    return node;
}
