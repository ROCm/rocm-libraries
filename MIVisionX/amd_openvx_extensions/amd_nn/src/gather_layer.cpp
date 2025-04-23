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

#include <kernels.h>

static vx_status VX_CALLBACK validateGatherLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, type2, out_type;
    vx_size num_dims, num_dims2, out_num_dims;
    vx_size input_dims[4], input_dims2[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims2, sizeof(num_dims2)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type2, sizeof(type2)));
    if ((type2 != VX_TYPE_INT32) && (type2 != VX_TYPE_INT64)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims2, sizeof(input_dims2)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // the output num_dim should equal to (input_dim + input_dim2 - 1)
    if (num_dims + num_dims2 - 1 != out_num_dims) {
        printf("validate: gather: The [rank(output tensor)] should equal to [rank(input tensor) + rank(indices tensor) - 1)]\n");
        printf("validate: gather: %d != %d + %d - 1\n", (int)out_num_dims, (int)num_dims, (int)num_dims2);
        return VX_ERROR_INVALID_DIMENSION;
    }
    
    vx_uint32 axis;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if (axis < 0 || axis > 3) {
        printf("validate: gather: Axis value should be 0~2\n");
        printf("validate: gather: Axis = %d\n", axis);
        return VX_ERROR_INVALID_PARAMETERS;
    } 
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));

    return VX_SUCCESS;

}

static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,
    vx_uint32& supported_target_affinity
)
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

#if ENABLE_OPENCL
static vx_status VX_CALLBACK opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
)
{
    //get input params
    vx_size num_of_dims, num_of_dims2;
    vx_enum type;
    vx_uint32 axis;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    vx_size input_dims[num_of_dims];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));    

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims2, sizeof(num_of_dims)));
    vx_size input_dims2[num_of_dims2];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims2, sizeof(input_dims2)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    
    strcpy(opencl_kernel_function_name, "gather_layer");

    //reverse input dims w,h,c,n- > n,c,h,w
    int start = 0, end = num_of_dims-1;
    while (start < end) {
        int temp = input_dims[start];
        input_dims[start] = input_dims[end];
        input_dims[end] = temp;
        start++;
        end--;
    }

    opencl_work_dim = 3;
    opencl_global_work[0] = 1;
    opencl_global_work[1] = 1;
    opencl_global_work[2] = 1;
    
    for (int i=0; i<num_of_dims; i++) {
        if (i < axis) {
            opencl_global_work[2] *= input_dims[i];
        }
        else if (i > axis) {
            opencl_global_work[0] *= input_dims[i];
        }
    }
    
    for (int i=0; i<num_of_dims2; i++) {
        opencl_global_work[1] *= input_dims2[i];
    }

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims) {
        char item[8192];
        if (type == VX_TYPE_FLOAT32) {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ind, uint ind_offset, uint4 ind_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint axis) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   int indices = *(__global int*)&ind[ind_offset + y*ind_stride.s0];\n"
                "   float value;\n"
                "   uint offset;\n"
                "   if (axis == 0) {\n"
                "       value = *(__global float*)&in[in_offset + x*in_stride.s0 + indices*in_stride.s1 + c*in_stride.s2];\n"
                "       offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   }\n"
                "   else if (axis == 1) {\n"
                "       value = *(__global float*)&in[in_offset + indices*in_stride.s0 + c*in_stride.s1];\n"
                "       offset = out_offset + y*out_stride.s0 + c*out_stride.s1;\n"
                "   }\n"
                "   else if (axis == 2) {\n"
                "       value = *(__global float*)&in[in_offset + c*in_stride.s0];\n"
                "       offset = out_offset + c*out_stride.s0;\n"
                "   }\n"
                "   out += offset;\n"
                "   *(__global float *)&out[0] = value;\n"
                "}\n", opencl_kernel_function_name);
        }
        else {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * ind, uint ind_offset, uint4 ind_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint axis) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   int indices = *(__global int*)&ind[ind_offset + y*ind_stride.s0];\n"
                "   half value;\n"
                "   uint offset;\n"
                "   if (axis == 0) {\n"
                "       value = *(__global half*)&in[in_offset + x*in_stride.s0 + indices*in_stride.s1 + c*in_stride.s2];\n"
                "       offset = out_offset + x*out_stride.s0 + y*out_stride.s1 + c*out_stride.s2;\n"
                "   }\n"
                "   else if (axis == 1) {\n"
                "       value = *(__global half*)&in[in_offset + indices*in_stride.s0 + c*in_stride.s1];\n"
                "       offset = out_offset + y*out_stride.s0 + c*out_stride.s1;\n"
                "   }\n"
                "   else if (axis == 2) {\n"
                "       value = *(__global half*)&in[in_offset + c*in_stride.s0];\n"
                "       offset = out_offset + c*out_stride.s0;\n"
                "   }\n"
                "   out += offset;\n"
                "   *(__global half *)&out[0] = value;\n"
                "}\n", opencl_kernel_function_name);
        }
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) 
{
#if ENABLE_HIP
    //get input params
    vx_size num_of_dims, num_of_dims2;
    vx_enum type;
    vx_uint32 axis;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    vx_size input_dims[num_of_dims];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims2, sizeof(num_of_dims)));
    vx_size input_dims2[num_of_dims2];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims2, sizeof(input_dims2)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    //reverse input dims w,h,c,n- > n,c,h,w
    std::reverse(input_dims, input_dims + num_of_dims);

    dim3 globalThreads = dim3(1);
    for (int i = 0; i < num_of_dims; i++) {
        if (i < axis) {
            globalThreads.z *= input_dims[i];
        }
        else if (i > axis) {
            globalThreads.x *= input_dims[i];
        }
    }
    for (int i = 0; i < num_of_dims2; i++) {
        globalThreads.y *= input_dims2[i];
    }

    vx_size temp[4] = {0};
    uint4 input_stride, ind_stride, output_stride;
    vx_size in_offset, ind_offset, output_offset;
    unsigned char *input_mem = NULL;
    unsigned char *ind_mem = NULL;
    unsigned char *output_mem = NULL;
    hipStream_t hip_stream;

    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &hip_stream, sizeof(hip_stream)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_OFFSET_GPU, &in_offset, sizeof(in_offset)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &ind_mem, sizeof(ind_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_OFFSET_GPU, &ind_offset, sizeof(ind_offset)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_OFFSET_GPU, &output_offset, sizeof(output_offset)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    input_stride.x = temp[0];
    input_stride.y = temp[1];
    input_stride.z = temp[2];
    input_stride.w = temp[3];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    ind_stride.x = temp[0];
    ind_stride.y = temp[1];
    ind_stride.z = temp[2];
    ind_stride.w = temp[3];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    output_stride.x = temp[0];
    output_stride.y = temp[1];
    output_stride.z = temp[2];
    output_stride.w = temp[3];

    if (HipExec_Gather_layer(hip_stream, globalThreads, dim3(1), type, input_mem, in_offset, input_stride,
        ind_mem, ind_offset, ind_stride, output_mem, output_offset, output_stride, axis)) {
        return VX_FAILURE;
    }

    return VX_SUCCESS;

#elif ENABLE_OPENCL
    return VX_ERROR_NOT_IMPLEMENTED;
#endif
}

//! \brief The kernel publisher.
vx_status publishGatherLayer(vx_context context) 
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.gather_layer", VX_KERNEL_GATHER_LAYER_AMD, host_kernel, 4, validateGatherLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    
    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxGatherLayer(vx_graph graph, vx_tensor input, vx_tensor indices, vx_tensor output, vx_int32 axis) 
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar s_axis = vxCreateScalarWithSize(context, VX_TYPE_INT32, &axis, sizeof(axis));
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS && vxGetStatus((vx_reference)s_axis) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) indices,
            (vx_reference) output,
            (vx_reference) s_axis,
        };
        node = createNode(graph, VX_KERNEL_GATHER_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
        vxReleaseScalar(&s_axis);
    }

    return node;
}

