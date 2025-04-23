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

enum {
    NONE,                       //No bias and no activation present.
    BIAS_ONLY_SEPERATE,         // only bias is present and can't fuse.
    BIAS_ONLY_FUSED,                  //only bias is present and can fuse.
    ACTIVATION_ONLY_SEPERATE,            // only activation is present and can't fuse.
    ACTIVATION_ONLY_FUSED,            // only activation is present and can fuse
    BIAS_ACTIVATION_SEPERATE,   // both bias and activation are executed seperately.
    BIAS_ACTIVATION_FUSED       //both bias and activation are fused.
};

struct ConvolutionLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    float conv_alpha;
    float conv_beta;
    float bias_alpha, bias_beta;
    miopenDataType_t data_type;          // data_type for the kernel
    miopenTensorDescriptor_t input_desc;
    void *input_mem;
    miopenTensorDescriptor_t weight_desc;
    void *weight_mem;
    miopenConvolutionDescriptor_t conv_desc;
    miopenConvFwdAlgorithm_t algo;
    miopenTensorDescriptor_t output_desc;
    void *output_mem;
    void *workspace;
    size_t workspace_size;
    miopenTensorDescriptor_t bias_desc;
    void *bias_mem;
    miopenActivationMode_t activation_mode;
    float activation_alpha;
    float activation_beta;
    float activation_power;
    miopenActivationDescriptor_t activation_desc;
    vx_int32 bias_activ_mode;
    vx_float32 leaky_alpha;
    vx_int32 groupCount;
    vx_bool fusion_possible;
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t activOp;
    miopenOperatorArgs_t fusionArgs;
};

static vx_status VX_CALLBACK validateConvolutionLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check scalar type
    vx_enum in_type, type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_NN_CONVOLUTION_PARAMS) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #3 type=%d (must be CONV_PARAMS)\n", type);
    if(parameters[5]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #5 type=%d (must be VX_TYPE_FLOAT32)\n", type);
        vx_float32 leaky_alpha = 1.0f;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &leaky_alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    if(parameters[6]) {
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &type, sizeof(type)));
        if(type != VX_TYPE_INT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #6 type=%d (must be VX_TYPE_INT32)\n", type);
        vx_int32 groupCount = 1;
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &groupCount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    // check tensor dimensions
    vx_size num_dims;
    vx_size input_dims[4], weights_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_type, sizeof(in_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: #0 num_dims=%ld (must be 4)\n", num_dims);
    if((in_type != VX_TYPE_FLOAT32) && (in_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #0 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: #1 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, weights_dims, sizeof(weights_dims)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if(num_dims != 1 && num_dims != 2) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: #2 num_dims=%ld (must be 1 or 2)\n", num_dims);
        if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #2 type=%d (must be float/float16)\n", type);
        vx_size bias_dims[2] = { 0, 1 };
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, bias_dims, num_dims*sizeof(bias_dims[0])));
        if(bias_dims[0] != weights_dims[3] || bias_dims[1] != 1) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: bias[%ldx%ld] weights[%ldx%ldx%ldx%ld]\n", bias_dims[1], bias_dims[0], weights_dims[3], weights_dims[2], weights_dims[1], weights_dims[0]);
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: #4 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: conv: #4 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if(output_dims[3] != input_dims[3] || output_dims[2] != weights_dims[3])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: conv: input[%ldx%ldx%ldx%ld] weights[%ldx%ldx%ldx%ld] output[%ldx%ldx%ldx%ld]\n",
            input_dims[3], input_dims[2], input_dims[1], input_dims[0],
            weights_dims[3], weights_dims[2], weights_dims[1], weights_dims[0],
            output_dims[3], output_dims[2], output_dims[1], output_dims[0]);

    // output tensor configuration
    type = in_type;     // should be same as input type
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processConvolutionLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
PROFILER_START(VX_NN, Convolution_Layer)
    ConvolutionLayerLocalData * data= NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->weight_mem, sizeof(data->weight_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->bias_mem, sizeof(data->bias_mem)));
    }
#elif ENABLE_HIP
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->weight_mem, sizeof(data->weight_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->bias_mem, sizeof(data->bias_mem)));
    }
#endif

    if (data->fusion_possible == true)
    {
        // Set the Args
        ERROR_CHECK_MIOPEN_STATUS(miopenExecuteFusionPlan(data->handle->miopen_handle, data->fusePlanDesc, data->input_desc, data->input_mem, data->output_desc, data->output_mem, data->fusionArgs));
        //ERROR_CHECK_STATUS(clFinish(data->handle->cmdq));       // this is required to fix the sync issue in fusion

    }else
    {
        //ConvolutionForward.
        ERROR_CHECK_MIOPEN_STATUS(miopenConvolutionForward(data->handle->miopen_handle, &data->conv_alpha, data->input_desc, data->input_mem,
                                                           data->weight_desc,data->weight_mem,data->conv_desc,data->algo,&data->conv_beta, data->output_desc, data->output_mem, data->workspace, data->workspace_size));

        //Convolution Forward Bias if bias_activ mode is BIAS_ONLY or BIAS_ACTIVATION_FUSED or BIAS_ACTIVATION_SEPERATE.
        if(data->bias_activ_mode == BIAS_ONLY_SEPERATE || data->bias_activ_mode == BIAS_ACTIVATION_SEPERATE) {
            ERROR_CHECK_MIOPEN_STATUS(miopenConvolutionForwardBias(data->handle->miopen_handle, &data->bias_alpha, data->bias_desc, data->bias_mem,
                                                               &data->bias_beta, data->output_desc, data->output_mem));
        }

        // activation (in-place in output_mem)
        if (data->bias_activ_mode == ACTIVATION_ONLY_SEPERATE || data->bias_activ_mode == BIAS_ACTIVATION_SEPERATE) {
            float alpha = 1.0f, beta = 0.0f;
            ERROR_CHECK_MIOPEN_STATUS(miopenActivationForward(data->handle->miopen_handle, data->activation_desc, &alpha, data->output_desc, data->output_mem,
                                                              &beta, data->output_desc, data->output_mem));
        }
    }  

    /*DUMP LAYER BUFFER*/
    #if ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
        //dump the output layer
        nn_layer_test_dumpBuffer("conv_%04d.bin", (vx_tensor)parameters[4]);
    #endif  
PROFILER_STOP(VX_NN, Convolution_Layer)
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeConvolutionLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ConvolutionLayerLocalData * data = new ConvolutionLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //convolution params.
    vx_nn_convolution_params_t params;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &params, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    vx_size pad_h, pad_w;
    vx_size dilation_w, dilation_h;
    vx_enum downscale_size_rounding, overflow_policy, rounding_policy;

    pad_h = params.padding_y; pad_w = params.padding_x;
    downscale_size_rounding = params.down_scale_size_rounding;
    overflow_policy = params.overflow_policy;
    rounding_policy = params.rounding_policy;
    dilation_h = params.dilation_y + 1;
    dilation_w = params.dilation_x + 1;
    
    data->groupCount = 1;
    if(parameters[6])
    {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &data->groupCount, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    if(data->groupCount < 1)
    {
        data->groupCount = 1;
    }
    miopenConvolutionMode_t mode;
    if(data->groupCount == 1)
    {
        mode = miopenConvolution;
    }
    else
    {
        mode = miopenGroupConv;
    }

    // override default cbr_mode by NN_MIOPEN_CBR_MODE environment variable.
    char textBuffer[1024];
    vx_int32 nn_cbr_mode = getEnvironmentVariable("NN_MIOPEN_CBR_MODE", textBuffer, sizeof(textBuffer));
    if (nn_cbr_mode < 0) nn_cbr_mode = 1; // default cbr_mode

    // initialize the bias activ mode
    data->conv_alpha = 1.0; data->conv_beta = 0.0;
    data->bias_alpha = 1.0; data->bias_beta = 0.0;
    data->leaky_alpha = 0.0;

    vx_size input_dims[4], weights_dims[4], output_dims[4], bias_dims[2] = { 0, 1 };
    vx_enum out_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, weights_dims, sizeof(weights_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    data->data_type = (out_type == VX_TYPE_FLOAT32)? miopenFloat:miopenHalf;
    if(parameters[2]) {
        vx_size num_dims;
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(vx_size)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, bias_dims, num_dims * sizeof(vx_size)));
    }  
    
    if(input_dims[2] != (weights_dims[2] * data->groupCount))
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "initialize: conv: input[%ldx%ldx%ldx%ld] weights[%ldx%ldx%ldx%ld] output[%ldx%ldx%ldx%ld]\n",
            input_dims[3], input_dims[2], input_dims[1], input_dims[0],
            weights_dims[3], weights_dims[2], weights_dims[1], weights_dims[0],
            output_dims[3], output_dims[2], output_dims[1], output_dims[0]);

    vx_size stride_h, stride_w;
    vx_size kernel_h, kernel_w;

    kernel_h = weights_dims[1];
    kernel_w = weights_dims[0];
    stride_w = (output_dims[0] > 1) ? ((input_dims[0] + 2 * pad_w - kernel_w - (kernel_w - 1) * (dilation_w - 1) + ((output_dims[0] - 1) / 2)) / (output_dims[0] - 1)) : 1;
    stride_h = (output_dims[1] > 1) ? ((input_dims[1] + 2 * pad_h - kernel_h - (kernel_h - 1) * (dilation_h - 1) + ((output_dims[1] - 1) / 2)) / (output_dims[1] - 1)) : 1;

    data->bias_activ_mode = NONE;
    data->fusion_possible = nn_cbr_mode && (stride_w == 1) && (stride_h == 1) && (dilation_w == 1) && (dilation_h == 1) && (pad_w <=1) && (pad_h <=1);   // MIOpen only support stride 1 for fusion
    data->fusion_possible &= (data->data_type == miopenFloat) && (kernel_h > 1) && (kernel_w > 1); // only supported for FP32 and there is a crash with 1x1 kernels
    if (parameters[2]) {
        data->bias_activ_mode = data->fusion_possible? BIAS_ONLY_FUSED : BIAS_ONLY_SEPERATE;
    }
    if (parameters[5]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &data->leaky_alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        if (data->leaky_alpha >= 0 && data->leaky_alpha <= 1) {
            if(data->bias_activ_mode == BIAS_ONLY_FUSED) {
                data->bias_activ_mode = BIAS_ACTIVATION_FUSED;
                //data->bias_beta = data->leaky_alpha - 3;              // do we need this hack??
            }
            else if(data->bias_activ_mode == BIAS_ONLY_SEPERATE) {
                data->bias_activ_mode = BIAS_ACTIVATION_SEPERATE;
            }
            else {
                data->bias_activ_mode = data->fusion_possible? ACTIVATION_ONLY_FUSED : ACTIVATION_ONLY_SEPERATE;
            }
        }
    }

    //input, weight and output descriptors.
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->weight_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input_desc, data->data_type, input_dims[3], input_dims[2], input_dims[1], input_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->weight_desc, data->data_type, weights_dims[3], weights_dims[2], weights_dims[1], weights_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output_desc, data->data_type, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));
    if(parameters[2]) {
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->bias_desc));
        ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->bias_desc, data->data_type, 1, bias_dims[0], 1, 1));
    }

    //Convolution Descriptor.
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateConvolutionDescriptor(&data->conv_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenInitConvolutionDescriptor(data->conv_desc, mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w));

    //Grouped Convolution
    ERROR_CHECK_MIOPEN_STATUS(miopenSetConvolutionGroupCount(data->conv_desc, data->groupCount));

#if ENABLE_OPENCL
    //Memory Declaration.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->weight_mem, sizeof(data->weight_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->bias_mem, sizeof(data->bias_mem)));
    }
#elif ENABLE_HIP
    //Memory Declaration.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &data->weight_mem, sizeof(data->weight_mem)));
    if(parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &data->bias_mem, sizeof(data->bias_mem)));
    }
#endif

    if (/*data->bias_activ_mode == BIAS_ONLY_FUSED || data->bias_activ_mode == ACTIVATION_ONLY_FUSED ||*/ data->bias_activ_mode == BIAS_ACTIVATION_FUSED) {
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateFusionPlan(&data->fusePlanDesc, miopenVerticalFusion, data->input_desc));
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateOperatorArgs(&data->fusionArgs));
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateOpConvForward(data->fusePlanDesc, &data->convoOp, data->conv_desc, data->weight_desc));
        miopenSetOpArgsConvForward(data->fusionArgs, data->convoOp, &data->conv_alpha, &data->conv_beta, data->weight_mem);
        if (data->bias_activ_mode == BIAS_ONLY_FUSED || data->bias_activ_mode == BIAS_ACTIVATION_FUSED)
        {
            ERROR_CHECK_MIOPEN_STATUS(miopenCreateOpBiasForward(data->fusePlanDesc, &data->biasOp, data->bias_desc));        // add bias to fusion plan
            miopenSetOpArgsBiasForward(data->fusionArgs, data->biasOp, &data->bias_alpha, &data->bias_beta, data->bias_mem);
        }
        if (data->bias_activ_mode == ACTIVATION_ONLY_FUSED || data->bias_activ_mode == BIAS_ACTIVATION_FUSED) {
            if (data->leaky_alpha == 0.0) {
                ERROR_CHECK_MIOPEN_STATUS(miopenCreateOpActivationForward(data->fusePlanDesc, &data->activOp, miopenActivationRELU));
            }
            else {
                ERROR_CHECK_MIOPEN_STATUS(miopenCreateOpActivationForward(data->fusePlanDesc, &data->activOp, miopenActivationLEAKYRELU));
            }
            data->activation_alpha = data->leaky_alpha;
            data->activation_beta = 1.0;
            data->activation_power = 1.0;
            miopenSetOpArgsActivForward(data->fusionArgs, data->activOp, &data->conv_alpha, &data->conv_beta, data->activation_alpha, data->activation_beta, data->activation_power);
        }

        // compile fusion plan
        auto status = miopenCompileFusionPlan(data->handle->miopen_handle, data->fusePlanDesc);
        if (status != miopenStatusSuccess){
          data->fusion_possible = false;
#if ENABLE_DEBUG_PRINT_DIMS
          std::cout << "miopenCompileFusionPlan returned failure running without fused kernels: " << data->bias_activ_mode << std::endl;
#endif
        }
    }else
    {
        if (data->bias_activ_mode == BIAS_ONLY_FUSED) data->bias_activ_mode = BIAS_ONLY_SEPERATE;
        if (data->bias_activ_mode == ACTIVATION_ONLY_FUSED) data->bias_activ_mode = ACTIVATION_ONLY_SEPERATE;
        data->fusion_possible = false;
    }

    if (data->fusion_possible != true)
    {
        //initialize activation parameters if bias_activ_mode is ACTIVATION_ONLY or BIAS_ACTIVATION_SEPERATE.
        data->activation_mode = miopenActivationPASTHRU;
        if (data->bias_activ_mode == ACTIVATION_ONLY_SEPERATE || data->bias_activ_mode == BIAS_ACTIVATION_SEPERATE) {
            data->activation_mode = data->leaky_alpha? miopenActivationLEAKYRELU : miopenActivationRELU;
            data->activation_alpha = data->leaky_alpha;
            data->activation_beta = 0.0;
            data->activation_power = 1.0;
            ERROR_CHECK_MIOPEN_STATUS(miopenCreateActivationDescriptor(&data->activation_desc));
            ERROR_CHECK_MIOPEN_STATUS(miopenSetActivationDescriptor(data->activation_desc, data->activation_mode, data->activation_alpha, data->activation_beta, data->activation_power));
        }
        //Workspace Size.
        ERROR_CHECK_MIOPEN_STATUS(miopenConvolutionForwardGetWorkSpaceSize(data->handle->miopen_handle, data->weight_desc, data->input_desc, data->conv_desc, data->output_desc, &data->workspace_size ));
        if (data->workspace_size > 0) {
            vx_context   vxContext = vxGetContext((vx_reference)node);
#if ENABLE_OPENCL
            cl_context context;
            ERROR_CHECK_STATUS(vxQueryContext(vxContext, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &context, sizeof(context)));
            data->workspace_size = (data->workspace_size + 3) & ~3;
            data->workspace = clCreateBuffer(context, CL_MEM_READ_WRITE, data->workspace_size, NULL, NULL);
            if (!data->workspace) {
                return VX_FAILURE;
            }
            cl_float pattern = 0;
            cl_int err;
            if (data->data_type == miopenFloat)
                err = clEnqueueFillBuffer(data->handle->cmdq, (cl_mem)data->workspace, &pattern, sizeof(cl_float), 0, data->workspace_size, 0, NULL, NULL);
            else
                err = clEnqueueFillBuffer(data->handle->cmdq, (cl_mem)data->workspace, &pattern, sizeof(cl_half), 0, data->workspace_size, 0, NULL, NULL);
            if(err) return VX_FAILURE;
#elif ENABLE_HIP
            int hip_device = -1;
            ERROR_CHECK_STATUS(vxQueryContext(vxContext, VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE, &hip_device, sizeof(hip_device)));
            if (hip_device < 0) {
                return VX_FAILURE;
            }
            hipError_t errcode_ret = hipSuccess;
            data->workspace_size = (data->workspace_size + 3) & ~3;
            errcode_ret = hipMalloc(&data->workspace, data->workspace_size);
            if (errcode_ret != hipSuccess) {
                return VX_FAILURE;
            }

            errcode_ret = hipMemset(data->workspace, 0, data->workspace_size);
            if (errcode_ret != hipSuccess) {
                return VX_FAILURE;
            }
#endif
        }
        //Finding best Convolution Algorithm.
        miopenConvAlgoPerf_t perf;
        int algo_count;
        ERROR_CHECK_MIOPEN_STATUS(miopenFindConvolutionForwardAlgorithm(data->handle->miopen_handle, data->input_desc, data->input_mem, data->weight_desc, data->weight_mem,
                                                                        data->conv_desc, data->output_desc, data->output_mem, 1, &algo_count, &perf, data->workspace, data->workspace_size, data->handle->exhaustiveSearch));
        data->algo = perf.fwd_algo;
    }

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "conv input " << input_dims[0] << " " << input_dims[1] << " " << input_dims[2] << " " << input_dims[3] << " ";
    std::cout << "conv_alpha : " << data->conv_alpha << " " << "conv_beta : " << data->conv_beta << " ";
    std::cout << "bias_alpha : " << data->bias_alpha << " " << "bias_beta : " << data->bias_beta << " ";
    std::cout << "Leaky_alpha : " << data->leaky_alpha << " ";
    std::cout << "fusion_possible : " << data->fusion_possible << " " << "fusion_mode: " << data->bias_activ_mode << " ";
    if (data->bias_activ_mode > BIAS_ONLY_FUSED) {
            std::cout << "activation alpha : " << data->activation_alpha << " activation beta:" << data->activation_beta << " ";
    }
    std::cout << " Mode : " << data->activation_mode << " ";
    std::cout << "weights " << weights_dims[0] << " " << weights_dims[1] << " "<< weights_dims[2] <<" " <<  weights_dims[3] << " ";
    std::cout << "bias " << bias_dims[0] << " ";
    std::cout << "stride " << stride_h << " " << stride_w << " " << "pad " << pad_h << " " << pad_w;
    std::cout << " output " << output_dims[0] << " " << output_dims[1] << " " << output_dims[2] << " " << output_dims[3] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeConvolutionLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ConvolutionLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
#if ENABLE_OPENCL
    if(data && data->workspace && clReleaseMemObject((cl_mem)data->workspace) != 0) return VX_FAILURE;
#elif ENABLE_HIP
    if (data->workspace) {
        hipError_t errcode_ret = hipFree(data->workspace);
        if (errcode_ret != hipSuccess) {
            return VX_FAILURE;
        }
    }
#endif
    if (data->fusePlanDesc) miopenDestroyFusionPlan(data->fusePlanDesc);
    if (data->fusionArgs) miopenDestroyOperatorArgs(data->fusionArgs);
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyConvolutionDescriptor(data->conv_desc));
    if (data->activation_desc) {
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyActivationDescriptor(data->activation_desc));
    }
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->weight_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->bias_desc));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishConvolutionLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.convolution_layer", VX_KERNEL_CONVOLUTION_LAYER, processConvolutionLayer, 7, validateConvolutionLayer, initializeConvolutionLayer, uninitializeConvolutionLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxConvolutionLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases,
                                                    const vx_nn_convolution_params_t *convolution_params, vx_size size_of_convolution_params, vx_tensor outputs)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar conv_params = vxCreateScalarWithSize(context, VX_TYPE_NN_CONVOLUTION_PARAMS, convolution_params, size_of_convolution_params);
        if(vxGetStatus((vx_reference)conv_params) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)weights,
                (vx_reference)biases,
                (vx_reference)conv_params,
                (vx_reference)outputs
            };
            node = createNode(graph, VX_KERNEL_CONVOLUTION_LAYER, params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&conv_params);
        }
    }
    return node;
}
