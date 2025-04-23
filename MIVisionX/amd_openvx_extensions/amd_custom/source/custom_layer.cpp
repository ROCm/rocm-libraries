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
#include "custom_api.h"

struct CustomLayerLocalData {
    CustomFunctionType function;
    customHandle custom_handle;
    unsigned char *pCustomParameterArray;
    customTensorDesc input_desc, output_desc;
    void* input_mem;
    void* output_mem;
    unsigned int cpu_num_threads;
    void * hipstream;
    customBackend backend;
};

inline void Set4dTensorDesc(customTensorDesc &desc, int data_type, vx_size (&dims)[4],  vx_size (&strides)[4]) {
    desc.data_type = (customDataType)data_type;
    desc.dims[0] = (unsigned int)dims[0], desc.dims[1] = (unsigned int)dims[1];
    desc.dims[2] = (unsigned int)dims[2], desc.dims[3] = (unsigned int)dims[3];
    desc.strides[0] = (unsigned int)strides[0], desc.strides[1] = (unsigned int)strides[1];
    desc.strides[2] = (unsigned int)strides[2], desc.strides[3] = (unsigned int)strides[3];
}

static vx_status VX_CALLBACK validateCustomLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    // check scalar type
    vx_enum type, in_type, out_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_UINT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: custom: #1 scalar type=%d (not uint32)\n", type);
    if (parameters[2]) {
      ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
      if (type != VX_TYPE_UINT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: custom: #2 scalar type=%d (not uint32)\n", type);
    }
    if (parameters[3]) {
      vx_size itemsize = 0;
      ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
      if(type != VX_TYPE_CHAR) return VX_ERROR_INVALID_TYPE;
      vx_size capacity;
      ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity)));
      if(capacity != 256) return VX_ERROR_INVALID_DIMENSION;
      ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMSIZE, &itemsize, sizeof(itemsize)));
      if(itemsize != sizeof(VX_TYPE_CHAR)) return VX_ERROR_INVALID_TYPE;
    }

    // check tensor dimensions
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_type, sizeof(in_type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: custom: #0 num_dims=%ld (must be 4)\n", num_dims);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: custom: #4 num_dims=%ld (must be 4)\n", num_dims);

    // output tensor configuration
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processCustomLayer(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    CustomLayerLocalData * data= NULL;
    vx_map_id map_id, map_id_1;
    vx_size istride[4], ostride[4];

    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->backend == customBackend::GPU) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &data->input_mem, sizeof(data->input_mem)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HIP, &data->output_mem, sizeof(data->output_mem)));
    } else{
        ERROR_CHECK_STATUS(vxMapTensorPatch((vx_tensor)parameters[0], 4, NULL, NULL, &map_id, istride, (void **)&data->input_mem, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ERROR_CHECK_STATUS(vxMapTensorPatch((vx_tensor)parameters[4], 4, NULL, NULL, &map_id_1, ostride, (void **)&data->output_mem, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    //custom node execute call.
    ERROR_CHECK_CUSTOM_STATUS(CustomExecute(data->custom_handle, data->input_mem, data->input_desc, data->output_mem, data->output_desc));
    if (data->backend != customBackend::GPU) { 
        ERROR_CHECK_STATUS(vxUnmapTensorPatch((vx_tensor)parameters[0], map_id));
        ERROR_CHECK_STATUS(vxUnmapTensorPatch((vx_tensor)parameters[4], map_id_1));
    }

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeCustomLayer(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    CustomLayerLocalData * data = new CustomLayerLocalData;
    memset(data, 0, sizeof(*data));

    //custom Function Type
    vx_int32 customFuntion;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &customFuntion, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (parameters[3]) {
      // Get Custom parameters array
      size_t arr_size;
      ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_size, sizeof(arr_size)));
      data->pCustomParameterArray = new unsigned char[arr_size];
      ERROR_CHECK_STATUS(vxCopyArrayRange((vx_array)parameters[3], 0, arr_size, 1, data->pCustomParameterArray, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    // create input and output desc
    vx_size input_dims[4], output_dims[4];
    vx_size input_strides[4], output_strides[4];
    vx_enum in_data_type, out_data_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &in_data_type, sizeof(in_data_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &out_data_type, sizeof(out_data_type)));
    data->backend = customBackend::CPU;   //default based on compiler flag

    if (parameters[2])
      ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->backend, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
#if ENABLE_HIP
    if (data->backend == customBackend::GPU) { 
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_GPU, input_strides, sizeof(input_strides)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_STRIDE_GPU, output_strides, sizeof(output_strides)));
        ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &data->hipstream, sizeof(data->hipstream)));
    }
    else {
        data->backend = customBackend::CPU;   //default for CPU
        input_strides[0] = sizeof(in_data_type);
        input_strides[1] = input_strides[0] * input_dims[0];
        input_strides[2] = input_strides[1] * input_dims[1];
        input_strides[3] = input_strides[2] * input_dims[2];
        output_strides[0] = sizeof(out_data_type);
        output_strides[1] = output_strides[0] * output_dims[0];
        output_strides[2] = output_strides[1] * output_dims[1];
        output_strides[3] = output_strides[2] * output_dims[2];
    }
#else
    data->backend = customBackend::CPU;   //default for CPU
    input_strides[0] = sizeof(in_data_type);
    input_strides[1] = input_strides[0] * input_dims[0];
    input_strides[2] = input_strides[1] * input_dims[1];
    input_strides[3] = input_strides[2] * input_dims[2];
    output_strides[0] = sizeof(out_data_type);
    output_strides[1] = output_strides[0] * output_dims[0];
    output_strides[2] = output_strides[1] * output_dims[1];
    output_strides[3] = output_strides[2] * output_dims[2];
#endif
    Set4dTensorDesc(data->input_desc, in_data_type, input_dims,  input_strides);
    Set4dTensorDesc(data->output_desc, out_data_type, output_dims,  output_strides);
    if (parameters[2])
      ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->backend, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // Create custom lib and run setup
    data->custom_handle = CustomCreate((CustomFunctionType)customFuntion);
    ERROR_CHECK_CUSTOM_STATUS(CustomSetup(data->custom_handle, data->input_desc, data->output_desc, 
                              (customBackend)data->backend, (customStream)data->hipstream));

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeCustomLayer(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    CustomLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
      ERROR_CHECK_CUSTOM_STATUS(CustomShutdown(data->custom_handle));
      if (data->pCustomParameterArray)
          delete data->pCustomParameterArray;
      delete data;
    }

    return VX_SUCCESS;
}

vx_status publishCustomLayer(vx_context context) {
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.custom_extension.custom_layer", VX_KERNEL_CUSTOM_LAYER, processCustomLayer, 5, validateCustomLayer, initializeCustomLayer, uninitializeCustomLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable GPU buffer access since the kernel_f callback uses GPU buffers instead of host accessible buffers
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#endif

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxCustomLayer(vx_graph graph, vx_tensor inputs, vx_uint32 function, vx_uint32 custom_backend, vx_array custom_parameters, vx_tensor outputs) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
      // todo register custom enum for using here and support in runvx
        vx_scalar s_function = vxCreateScalarWithSize(context, VX_TYPE_UINT32, &function, sizeof(function));
        vx_scalar s_backend = vxCreateScalarWithSize(context, VX_TYPE_UINT32, &custom_backend, sizeof(custom_backend));
        if (vxGetStatus((vx_reference)s_function) == VX_SUCCESS && vxGetStatus((vx_reference)s_backend) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)inputs,
                (vx_reference)s_function,
                (vx_reference)s_backend,
                (vx_reference)custom_parameters,
                (vx_reference)outputs
            };
            node = createCustomNode(graph, "com.amd.custom_extension.custom_layer", params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_function);
        }
    }
    return node;
}
