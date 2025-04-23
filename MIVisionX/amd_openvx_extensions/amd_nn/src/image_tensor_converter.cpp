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

static vx_status VX_CALLBACK validateImageToTensorKernel(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input configuration
    vx_uint32 width, height;
    vx_df_image format;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if(format != VX_DF_IMAGE_RGB && format != VX_DF_IMAGE_U8)
        return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: img2tensor: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&format);
    vx_enum scalar_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: img2tensor: #2 type=%d (must be float)\n", scalar_type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: img2tensor: #3 type=%d (must be float)\n", scalar_type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_BOOL) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: img2tensor: #4 type=%d (must be bool)\n", scalar_type);

    // check output dimensions
    vx_enum type;
    vx_size num_dims, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: img2tensor: #1 type=%d (must be float/float16)\n", type);
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: img2tensor: #1 num_dims=%ld (must be 4)\n", num_dims);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    if ((output_dims[2] != 3 && output_dims[2] != 1) || output_dims[0] != (size_t)width || (output_dims[1] * output_dims[3]) != (size_t)height)
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: img2tensor: output_dims[%ldx%ldx%ldx%ld] width=%d height=%d\n", output_dims[3], output_dims[2], output_dims[1], output_dims[0], width, height);

    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

#if ENABLE_OPENCL
//! \brief The OpenCL code generator callback.
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
    // get configuration
    vx_uint32 width, height, N;
    vx_df_image format;
    vx_size num_dims, output_dims[4] = { 1, 1, 1, 1 };
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    height = (vx_uint32)output_dims[1];
    N = (vx_uint32)output_dims[3];

    // generate OpenCL C code and compute global work
    strcpy(opencl_kernel_function_name, "image_to_tensor");
    if(format == VX_DF_IMAGE_RGB) {
        opencl_work_dim = 3;
        opencl_local_work[0] = 8;
        opencl_local_work[1] = 8;
        opencl_local_work[2] = 1;
        opencl_global_work[0] = (width  + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
        opencl_global_work[1] = (height + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);
        opencl_global_work[2] = N;

        char item[8192];
        if (type == VX_TYPE_FLOAT32){
        snprintf(item, sizeof(item),
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
            "void %s(uint i0_width, uint i0_height, __global uchar * i0_buf, uint i0_stride, uint i0_offset, __global uchar * o0_buf, uint o0_offset, uint4 o0_stride, float ka, float kb, uint reverse_channel_order)\n"
            "{\n"
            "    uint x = get_global_id(0);\n"
            "    uint y = get_global_id(1);\n"
            "    uint n = get_global_id(2);\n"
            "    if(x < %d && y < %d) {\n"
            "        uint ioffset = i0_offset + (y + n * %d) * i0_stride + x * 3;\n"
            "        uint2 rgb2 = vload2(0, (__global uint *)&i0_buf[ioffset & ~3]);\n"
            "        uint rgb = amd_bytealign(rgb2.s1, rgb2.s0, ioffset & 3);\n"
            "        float r = ka * amd_unpack0(rgb) + kb;\n"
            "        float g = ka * amd_unpack1(rgb) + kb;\n"
            "        float b = ka * amd_unpack2(rgb) + kb;\n"
            "        o0_buf += o0_offset + n * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0;\n"
            "        *(__global float *)&o0_buf[               0] = reverse_channel_order ? b : r;\n"
            "        *(__global float *)&o0_buf[    o0_stride.s2] =                             g;\n"
            "        *(__global float *)&o0_buf[2 * o0_stride.s2] = reverse_channel_order ? r : b;\n"
            "    }\n"
            "}\n"
            , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        } else {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
                "void %s(uint i0_width, uint i0_height, __global uchar * i0_buf, uint i0_stride, uint i0_offset, __global uchar * o0_buf, uint o0_offset, uint4 o0_stride, float ka, float kb, uint reverse_channel_order)\n"
                "{\n"
                "    uint x = get_global_id(0);\n"
                "    uint y = get_global_id(1);\n"
                "    uint n = get_global_id(2);\n"
                "    if(x < %d && y < %d) {\n"
                "        uint ioffset = i0_offset + (y + n * %d) * i0_stride + x * 3;\n"
                "        uint2 rgb2 = vload2(0, (__global uint *)&i0_buf[ioffset & ~3]);\n"
                "        uint rgb = amd_bytealign(rgb2.s1, rgb2.s0, ioffset & 3);\n"
                "        half r = (half)ka * amd_unpack0(rgb) + (half)kb;\n"
                "        half g = (half)ka * amd_unpack1(rgb) + (half)kb;\n"
                "        half b = (half)ka * amd_unpack2(rgb) + (half)kb;\n"
                "        o0_buf += o0_offset + n * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0;\n"
                "        *(__global half *)&o0_buf[               0] = reverse_channel_order ? b : r;\n"
                "        *(__global half *)&o0_buf[    o0_stride.s2] =                             g;\n"
                "        *(__global half *)&o0_buf[2 * o0_stride.s2] = reverse_channel_order ? r : b;\n"
                "    }\n"
                "}\n"
                , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);

        }
        opencl_kernel_code = item;
    }
    else if(format == VX_DF_IMAGE_U8) {
        opencl_work_dim = 3;
        opencl_local_work[0] = 8;
        opencl_local_work[1] = 8;
        opencl_local_work[2] = 1;
        opencl_global_work[0] = ((width+3)/4 + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
        opencl_global_work[1] = (height + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);
        opencl_global_work[2] = N;

        char item[8192];
        if (type == VX_TYPE_FLOAT32){
        snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
                "void %s(uint i0_width, uint i0_height, __global uchar * i0_buf, uint i0_stride, uint i0_offset, __global uchar * o0_buf, uint o0_offset, uint4 o0_stride, float a, float b, uint reverse_channel_order)\n"
                "{\n"
                "    uint x = get_global_id(0) * 4;\n"
                "    uint y = get_global_id(1);\n"
                "    uint n = get_global_id(2);\n"
                "    if(x < %d && y < %d) {\n"
                "        uint u4 = *(__global uint *)&i0_buf[i0_offset + (y + n * %d) * i0_stride + x];\n"
                "        float p0 = a * amd_unpack0(u4) + b;\n"
                "        float p1 = a * amd_unpack1(u4) + b;\n"
                "        float p2 = a * amd_unpack2(u4) + b;\n"
                "        float p3 = a * amd_unpack3(u4) + b;\n"
                "        *(__global float4 *)&o0_buf[o0_offset + n * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0] = (float4)(p0 , p1, p2, p3);\n"
                "    }\n"
                "}\n"
            , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        } else {
            snprintf(item, sizeof(item),
                    "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                    "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
                    "void %s(uint i0_width, uint i0_height, __global uchar * i0_buf, uint i0_stride, uint i0_offset, __global uchar * o0_buf, uint o0_offset, uint4 o0_stride, float a, float b, uint reverse_channel_order)\n"
                    "{\n"
                    "    uint x = get_global_id(0) * 4;\n"
                    "    uint y = get_global_id(1);\n"
                    "    uint n = get_global_id(2);\n"
                    "    if(x < %d && y < %d) {\n"
                    "        uint u4 = *(__global uint *)&i0_buf[i0_offset + (y + n * %d) * i0_stride + x];\n"
                    "        half p0 = (half)a * amd_unpack0(u4) + (half)b;\n"
                    "        half p1 = (half)a * amd_unpack1(u4) + (half)b;\n"
                    "        half p2 = (half)a * amd_unpack2(u4) + (half)b;\n"
                    "        half p3 = (half)a * amd_unpack3(u4) + (half)b;\n"
                    "        *(__global half4 *)&o0_buf[o0_offset + n * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0] = (half4)(p0 , p1, p2, p3);\n"
                    "    }\n"
                    "}\n"
                , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name, width, height, height);
        }
        opencl_kernel_code = item;
    }

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "KERNEL image_to_tensor output " << width << " " << height << " " << N << std::endl;
#endif

    return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
#if ENABLE_HIP
    // get configuration
    vx_uint32 width, height, N;
    vx_df_image format;
    vx_size num_dims, output_dims[4] = { 1, 1, 1, 1 };
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0]) * num_dims));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    height = (vx_uint32)output_dims[1];
    N = (vx_uint32)output_dims[3];

    vx_size temp[4] = {0};
    uint4 output_stride;
    vx_size output_offset;
    vx_uint32 input_offset, input_stride;
    unsigned char *input_mem = NULL;
    unsigned char *output_mem = NULL;
    float sc1, sc2;
    uint reverse_channel_order;
    hipStream_t hip_stream;

    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &hip_stream, sizeof(hip_stream)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET, &input_offset, sizeof(input_offset)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE, &input_stride, sizeof(input_stride)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_OFFSET_GPU, &output_offset, sizeof(output_offset)));
    ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[2], &sc1));
    ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[3], &sc2));
    ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[4], &reverse_channel_order));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    output_stride.x = temp[0];
    output_stride.y = temp[1];
    output_stride.z = temp[2];
    output_stride.w = temp[3];

    if (HipExec_image_to_tensor_layer(hip_stream, format, type, width, height, N, input_mem, input_offset,
        input_stride, output_mem, output_offset, output_stride, sc1, sc2, reverse_channel_order)) {
        return VX_FAILURE;
    }

    return VX_SUCCESS;

#elif ENABLE_OPENCL
    return VX_ERROR_NOT_IMPLEMENTED;
#endif
}

//! \brief The kernel publisher.
vx_status publishImageToTensorConvert(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.convert_image_to_tensor", VX_KERNEL_CONVERT_IMAGE_TO_TENSOR_AMD, host_kernel, 5, validateImageToTensorKernel, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxConvertImageToTensorNode(vx_graph graph, vx_image input, vx_tensor output, vx_float32 a, vx_float32 b, vx_bool reverse_channel_order)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_a = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &a, sizeof(a));
        vx_scalar s_b = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &b, sizeof(b));
        vx_scalar s_order = vxCreateScalarWithSize(context, VX_TYPE_BOOL, &reverse_channel_order, sizeof(reverse_channel_order));
        if(vxGetStatus((vx_reference)s_order) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)input,
                (vx_reference)output,
                (vx_reference)s_a,
                (vx_reference)s_b,
                (vx_reference)s_order
            };
            node = createNode(graph, VX_KERNEL_CONVERT_IMAGE_TO_TENSOR_AMD, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_a);
            vxReleaseScalar(&s_b);
            vxReleaseScalar(&s_order);
        }
    }
    return node;
}
