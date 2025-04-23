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

#include "internal_publishKernels.h"

static vx_status VX_CALLBACK validateImageToTensorKernel(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input configuration
    vx_uint32 width, height;
    vx_df_image format;
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if(format != VX_DF_IMAGE_RGB && format != VX_DF_IMAGE_U8)
        return printf("validate: img2tensor: #0 format=%4.4s (must be RGB2 or U008)\n", (char *)&format);
    vx_enum scalar_type;
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return printf("validate: img2tensor: #2 type=%d (must be float)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_FLOAT32) return printf("validate: img2tensor: #3 type=%d (must be float)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_BOOL) return printf("validate: img2tensor: #4 type=%d (must be bool)\n", scalar_type);

    // check output dimensions
    vx_enum type;
    vx_size num_dims, output_dims[4] = { 1, 1, 1, 1 };
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return printf("validate: img2tensor: #1 type=%d (must be float/float16)\n", type);
    if (num_dims != 4) return printf("validate: img2tensor: #1 num_dims=%ld (must be 4)\n", num_dims);
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    if ((output_dims[2] != 3 && output_dims[2] != 1) || output_dims[0] != (size_t)width || (output_dims[1] * output_dims[3]) != (size_t)height)
        return printf("validate: img2tensor: output_dims[%ldx%ldx%ldx%ld] width=%d height=%d\n", output_dims[3], output_dims[2], output_dims[1], output_dims[0], width, height);

    // set output tensor configuration
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

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
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
    STATUS_ERROR_CHECK(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims[0])*num_dims));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
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

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status WINML_ConvertImageToTensor_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddUserKernel(context, 
		"com.winml.convert_image_to_tensor", 
		VX_KERNEL_WINML_CONVERT_IMAGE_TO_TENSOR,
		host_kernel, 
		5, 
		validateImageToTensorKernel, 
		nullptr, 
		nullptr);
	
	if (kernel)
	{
		amd_kernel_query_target_support_f query_target_support_f = query_target_support;
		amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
		STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
		STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

		// set kernel parameters.
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

		// finalize and release kernel object.
		PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
		PARAM_ERROR_CHECK(vxReleaseKernel(&kernel));
	}

	if (status != VX_SUCCESS)
	{
	exit:   vxRemoveKernel(kernel);  std::cerr << "ERROR: vxAddParameterToKernel() failed for convert_image_to_tensor" << std::endl; return VX_FAILURE;
	}

    return status;
}


