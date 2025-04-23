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

static vx_status VX_CALLBACK validateKernel(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input configuration
    vx_enum type, format;
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if (num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validation: argmax: #0 num_dims=%ld (num_dims must be 2 or 4)\n", num_dims);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], sizeof(input_dims[0])*num_dims));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validation: argmax: #0 type=%d (not float32/float16)\n", type);

    // check output object type and set configuration
    ERROR_CHECK_STATUS(vxQueryReference(parameters[1], VX_REFERENCE_TYPE, &type, sizeof(type)));
    if (type == VX_TYPE_IMAGE) {
        vx_df_image format;
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
        if(format == VX_DF_IMAGE_U8 && input_dims[2] > 256)
            return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: argmax: #1 img U008 with input_dims[2](=%ld) > 256\n", input_dims[2]);
        if(format == VX_DF_IMAGE_VIRT)
            format = (input_dims[2] < 256) ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_U16;
        vx_uint32 width = (vx_uint32)input_dims[0];
        vx_uint32 height = (vx_uint32)(input_dims[1]*input_dims[3]);
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_HEIGHT, &height, sizeof(height)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
    }
    else if (type == VX_TYPE_TENSOR) {
        vx_size output_num_dims, output_dims[4] = { 1, 1, 1, 1 };
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &output_num_dims, sizeof(output_num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-output_num_dims], sizeof(output_dims[0])*output_num_dims));
        if (output_dims[2] != 1 && output_dims[2] != 2) // top_k must be 1 or 2
            return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validation: argmax: #1 top_k=%ld (must be 1 or 2)\n", output_dims[2]);
        if(type == VX_TYPE_UINT8 && input_dims[2] > 256)
            return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: argmax: #1 tensor U8 with input_dims[2](=%ld) > 256\n", input_dims[2]);
        if(type != VX_TYPE_UINT8 && type != VX_TYPE_UINT16 && type != VX_TYPE_INT16 && type != VX_TYPE_INT64)
            return ERRMSG(VX_ERROR_INVALID_FORMAT, "validate: argmax: #1 tensor output type=%d (must be U8/U16/I16/I64)\n", type);
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &output_num_dims, sizeof(output_num_dims)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims[4-output_num_dims], sizeof(output_dims[0])*output_num_dims));
    }
    else
        return ERRMSG(VX_ERROR_INVALID_PARAMETERS, "validate: argmax: output object type=%d must be image or tensor\n", type);

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
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 }, top_k = 1;
    vx_enum output_obj_type, output_data_type = VX_TYPE_UINT16;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], sizeof(input_dims[0])*num_dims));
    ERROR_CHECK_STATUS(vxQueryReference(parameters[1], VX_REFERENCE_TYPE, &output_obj_type, sizeof(output_obj_type)));
    if(output_obj_type == VX_TYPE_IMAGE) {
        vx_df_image format;
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
        if(format == VX_DF_IMAGE_U8)
            output_data_type = VX_TYPE_UINT8;
        else if(format == VX_DF_IMAGE_U16)
            output_data_type = VX_TYPE_UINT16;
    }
    else {
        vx_size num_dims_output, output_dims[4] = { 1, 1, 1, 1 };
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims_output, sizeof(num_dims_output)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims_output], sizeof(output_dims[0])*num_dims_output));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_data_type, sizeof(output_data_type)));
        top_k = output_dims[2];
    }
    size_t N = input_dims[3];

    // compute global work
    bool input_width_multiple_of_4 = (input_dims[0] & 3) ? false : true;
    opencl_work_dim = 3;
    opencl_local_work[0] = 8;
    opencl_local_work[1] = 8;
    opencl_local_work[2] = 1;
    opencl_global_work[0] = ((input_width_multiple_of_4 ? ((input_dims[0] + 3) / 4) : input_dims[0]) + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
    opencl_global_work[1] = (input_dims[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);
    opencl_global_work[2] = N;

    // generate OpenCL C code
    strcpy(opencl_kernel_function_name, "argmax");
    char item[8192];
    snprintf(item, sizeof(item),
        "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
        "__kernel __attribute__((reqd_work_group_size(%ld, %ld, 1)))\n" // opencl_local_work[0] opencl_local_work[1]
        "void %s(__global uchar * i0_buf, uint i0_offset, uint4 i0_stride, %s)\n"
        "{\n"
        "    uint x = get_global_id(0) * %d;\n"
        "    uint y = get_global_id(1);\n"
        "    uint z = get_global_id(2);\n"
        "    if(x < %ld && y < %ld) {\n"
        "        i0_buf += i0_offset + z * i0_stride.s3 + y * i0_stride.s1 + x * i0_stride.s0;\n"
        "        %s cmax;\n"
        , opencl_local_work[0], opencl_local_work[1], opencl_kernel_function_name
        , (output_obj_type == VX_TYPE_IMAGE) ?
            "uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset" :
            "__global uchar * o0_buf, uint o0_offset, uint4 o0_stride"
        , input_width_multiple_of_4 ? 4 : 1, input_dims[0], input_dims[1], input_width_multiple_of_4 ? "uint4" : "uint");
    opencl_kernel_code = item;
    if(top_k == 2) {
        if(input_width_multiple_of_4) {
            snprintf(item, sizeof(item),
                "        uint4 cmax1;\n"
                "        float4 f, fmax, fmax1;\n"
                "        fmax = *(__global float4 *)i0_buf;\n"
                "        i0_buf += i0_stride.s2; f = *(__global float4 *)i0_buf;\n"
                "        cmax1.s0 = (f.s0 > fmax.s0) ? 0 : 1;\n"
                "         cmax.s0 = (f.s0 > fmax.s0) ? 1 : 0;\n"
                "        fmax1.s0 = (f.s0 > fmax.s0) ? fmax.s0 :    f.s0;\n"
                "         fmax.s0 = (f.s0 > fmax.s0) ?    f.s0 : fmax.s0;\n"
                "        cmax1.s1 = (f.s1 > fmax.s1) ? 0 : 1;\n"
                "         cmax.s1 = (f.s1 > fmax.s1) ? 1 : 0;\n"
                "        fmax1.s1 = (f.s1 > fmax.s1) ? fmax.s1 :    f.s1;\n"
                "         fmax.s1 = (f.s1 > fmax.s1) ?    f.s1 : fmax.s1;\n"
                "        cmax1.s2 = (f.s2 > fmax.s2) ? 0 : 1;\n"
                "         cmax.s2 = (f.s2 > fmax.s2) ? 1 : 0;\n"
                "        fmax1.s2 = (f.s2 > fmax.s2) ? fmax.s2 :    f.s2;\n"
                "         fmax.s2 = (f.s2 > fmax.s2) ?    f.s2 : fmax.s2;\n"
                "        cmax1.s3 = (f.s3 > fmax.s3) ? 0 : 1;\n"
                "         cmax.s3 = (f.s3 > fmax.s3) ? 1 : 0;\n"
                "        fmax1.s3 = (f.s3 > fmax.s3) ? fmax.s3 :    f.s3;\n"
                "         fmax.s3 = (f.s3 > fmax.s3) ?    f.s3 : fmax.s3;\n"
                "        for(uint c = 2; c < %ld; c++) {\n"
                "            i0_buf += i0_stride.s2; f = *(__global float4 *)i0_buf;\n"
                "            cmax1.s0 = (f.s0 > fmax.s0) ? cmax.s0 : ((f.s0 > fmax1.s0) ? c    : cmax1.s0);\n"
                "            fmax1.s0 = (f.s0 > fmax.s0) ? fmax.s0 : ((f.s0 > fmax1.s0) ? f.s0 : fmax1.s0);\n"
                "            cmax.s0  = (f.s0 > fmax.s0) ? c    : cmax.s0;\n"
                "            fmax.s0  = (f.s0 > fmax.s0) ? f.s0 : fmax.s0;\n"
                "            cmax1.s1 = (f.s1 > fmax.s1) ? cmax.s1 : ((f.s1 > fmax1.s1) ? c    : cmax1.s1);\n"
                "            fmax1.s1 = (f.s1 > fmax.s1) ? fmax.s1 : ((f.s1 > fmax1.s1) ? f.s1 : fmax1.s1);\n"
                "            cmax.s1  = (f.s1 > fmax.s1) ? c    : cmax.s1;\n"
                "            fmax.s1  = (f.s1 > fmax.s1) ? f.s1 : fmax.s1;\n"
                "            cmax1.s2 = (f.s2 > fmax.s2) ? cmax.s2 : ((f.s2 > fmax1.s2) ? c    : cmax1.s2);\n"
                "            fmax1.s2 = (f.s2 > fmax.s2) ? fmax.s2 : ((f.s2 > fmax1.s2) ? f.s2 : fmax1.s2);\n"
                "            cmax.s2  = (f.s2 > fmax.s2) ? c    : cmax.s2;\n"
                "            fmax.s2  = (f.s2 > fmax.s2) ? f.s2 : fmax.s2;\n"
                "            cmax1.s3 = (f.s3 > fmax.s3) ? cmax.s3 : ((f.s3 > fmax1.s3) ? c    : cmax1.s3);\n"
                "            fmax1.s3 = (f.s3 > fmax.s3) ? fmax.s3 : ((f.s3 > fmax1.s3) ? f.s3 : fmax1.s3);\n"
                "            cmax.s3  = (f.s3 > fmax.s3) ? c    : cmax.s3;\n"
                "            fmax.s3  = (f.s3 > fmax.s3) ? f.s3 : fmax.s3;\n"
                "        }\n"
                , input_dims[2]);
            opencl_kernel_code += item;
        }
        else { // width not multiple of 4
            snprintf(item, sizeof(item),
                "        uint cmax1;\n"
                "        float f, fmax, fmax1;\n"
                "        fmax = *(__global float *)i0_buf;\n"
                "        i0_buf += i0_stride.s2; f = *(__global float *)i0_buf;\n"
                "        cmax1.s0 = (f > fmax) ? 0 : 1;\n"
                "         cmax.s0 = (f > fmax) ? 1 : 0;\n"
                "        fmax1.s0 = (f > fmax) ? fmax :    f;\n"
                "         fmax.s0 = (f > fmax) ?    f : fmax;\n"
                "        for(uint c = 2; c < %ld; c++) {\n"
                "            i0_buf += i0_stride.s2; f = *(__global float *)i0_buf;\n"
                "            cmax1 = (f > fmax) ? cmax : ((f > fmax1) ? c : cmax1);\n"
                "            fmax1 = (f > fmax) ? fmax : ((f > fmax1) ? f : fmax1);\n"
                "            cmax  = (f > fmax) ? c : cmax;\n"
                "            fmax  = (f > fmax) ? f : fmax;\n"
                "        }\n"
                , input_dims[2]);
            opencl_kernel_code += item;
        }
    }
    else if (top_k == 1) {
        if(input_width_multiple_of_4) {
            snprintf(item, sizeof(item),
                "        cmax = (uint4)0;\n"
                "        float4 fmax = *(__global float4 *)i0_buf;\n"
                "        for(uint c = 1; c < %ld; c++) {\n"
                "            i0_buf += i0_stride.s2;\n"
                "            float4 f = *(__global float4 *)i0_buf;\n"
                "            cmax.s0 = (f.s0 > fmax.s0) ? c    : cmax.s0;\n"
                "            fmax.s0 = (f.s0 > fmax.s0) ? f.s0 : fmax.s0;\n"
                "            cmax.s1 = (f.s1 > fmax.s1) ? c    : cmax.s1;\n"
                "            fmax.s1 = (f.s1 > fmax.s1) ? f.s1 : fmax.s1;\n"
                "            cmax.s2 = (f.s2 > fmax.s2) ? c    : cmax.s2;\n"
                "            fmax.s2 = (f.s2 > fmax.s2) ? f.s2 : fmax.s2;\n"
                "            cmax.s3 = (f.s3 > fmax.s3) ? c    : cmax.s3;\n"
                "            fmax.s3 = (f.s3 > fmax.s3) ? f.s3 : fmax.s3;\n"
                "        }\n"
                , input_dims[2]);
            opencl_kernel_code += item;
        }
        else { // width not multiple of 4
            snprintf(item, sizeof(item),
                "        cmax = (uint)0;\n"
                "        float fmax = *(__global float *)i0_buf;\n"
                "        for(uint c = 1; c < %ld; c++) {\n"
                "            i0_buf += i0_stride.s2;\n"
                "            float f = *(__global float *)i0_buf;\n"
                "            cmax = (f > fmax) ? c : cmax;\n"
                "            fmax = (f > fmax) ? f : fmax;\n"
                "        }\n"
                , input_dims[2]);
            opencl_kernel_code += item;
        }
    }
    if(output_data_type == VX_TYPE_UINT8) {
        if(output_obj_type == VX_TYPE_IMAGE) {
            snprintf(item, sizeof(item), "        o0_buf += o0_offset + (z * %ld + y) * o0_stride + x;\n" , input_dims[1]);
            opencl_kernel_code += item;
        }
        else {
            opencl_kernel_code +=
                "        o0_buf += o0_offset + z * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0;\n";
        }
        if(input_width_multiple_of_4) {
            opencl_kernel_code +=
                "        uint imax = cmax.s0 + (cmax.s1 << 8) + (cmax.s2 << 16) + (cmax.s3 << 24);\n"
                "        *(__global uint *)o0_buf = imax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        uint imax1 = cmax1.s0 + (cmax1.s1 << 8) + (cmax1.s2 << 16) + (cmax1.s3 << 24);\n"
                    "        *(__global uint *)&o0_buf[o0_stride.s2] = imax1;\n";
            }
        }
        else {
            opencl_kernel_code +=
                "        uint imax = cmax;\n"
                "        *(__global uchar *)o0_buf = (uchar)imax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        uint imax1 = cmax1;\n"
                    "        *(__global uchar *)&o0_buf[o0_stride.s2] = (uchar)imax1;\n";
            }
        }
    }
    else if(output_data_type == VX_TYPE_UINT16) {
        if(output_obj_type == VX_TYPE_IMAGE) {
            snprintf(item, sizeof(item), "        o0_buf += o0_offset + (z * %ld + y) * o0_stride + x * 2;\n" , input_dims[1]);
            opencl_kernel_code += item;
        }
        else {
            opencl_kernel_code +=
                "        o0_buf += o0_offset + z * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0;\n";
        }
        if(input_width_multiple_of_4) {
            opencl_kernel_code +=
                "        uint2 imax;\n"
                "        imax.s0 = cmax.s0 + (cmax.s1 << 16);\n"
                "        imax.s1 = cmax.s2 + (cmax.s3 << 16);\n"
                "        *(__global uint2 *)o0_buf = imax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        uint2 imax1;\n"
                    "        imax1.s0 = cmax1.s0 + (cmax1.s1 << 16);\n"
                    "        imax1.s1 = cmax1.s2 + (cmax1.s3 << 16);\n"
                    "        *(__global uint2 *)&o0_buf[o0_stride.s2] = imax1;\n";
            }
        }
        else {
            opencl_kernel_code +=
                "        *(__global ushort *)o0_buf = (ushort)cmax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        *(__global ushort *)&o0_buf[o0_stride.s2] = cmax1;\n";
            }
        }
    }
    else if(output_data_type == VX_TYPE_INT64) {
        if(output_obj_type == VX_TYPE_IMAGE) {
            snprintf(item, sizeof(item), "        o0_buf += o0_offset + (z * %ld + y) * o0_stride + x * 2;\n" , input_dims[1]);
            opencl_kernel_code += item;
        }
        else {
            opencl_kernel_code +=
                "        o0_buf += o0_offset + z * o0_stride.s3 + y * o0_stride.s1 + x * o0_stride.s0;\n";
        }
        if(input_width_multiple_of_4) {
            opencl_kernel_code +=
                "        uint2 imax;\n"
                "        imax.s0 = cmax.s0 + (cmax.s1 << 64);\n"
                "        imax.s1 = cmax.s2 + (cmax.s3 << 64);\n"
                "        *(__global uint2 *)o0_buf = imax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        uint2 imax1;\n"
                    "        imax1.s0 = cmax1.s0 + (cmax1.s1 << 64);\n"
                    "        imax1.s1 = cmax1.s2 + (cmax1.s3 << 64);\n"
                    "        *(__global uint2 *)&o0_buf[o0_stride.s2] = imax1;\n";
            }
        }
        else {
            opencl_kernel_code +=
                "        *(__global ushort *)o0_buf = (ushort)cmax;\n";
            if(top_k == 2) {
                opencl_kernel_code +=
                    "        *(__global ushort *)&o0_buf[o0_stride.s2] = cmax1;\n";
            }
        }
    }
    opencl_kernel_code +=
        "    }\n"
        "}\n";

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "KERNEL argmax_layer output " << input_dims[0] << "x" << input_dims[1] << " " << std::endl;
#endif
    return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
#if ENABLE_HIP
    vx_size num_dims, top_k = 1;
    vx_enum output_obj_type, output_data_type = VX_TYPE_UINT16;
    vx_size temp[4] = {1, 1, 1, 1};
    uint4 input_dims = (uint4)1, input_stride = (uint4)1, output_stride = (uint4)1;
    uint output_image_stride;
    vx_size in_offset, output_offset;
    unsigned char *input_mem = NULL;
    unsigned char *output_mem = NULL;
    hipStream_t hip_stream;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &temp[4 - num_dims], sizeof(temp[0]) * num_dims));
    input_dims.x = temp[0];
    input_dims.y = temp[1];
    input_dims.z = temp[2];
    input_dims.w = temp[3];
    temp[0] = temp[1] = temp[2] = temp[3] = 1;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_OFFSET_GPU, &in_offset, sizeof(in_offset)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    input_stride.x = temp[0];
    input_stride.y = temp[1];
    input_stride.z = temp[2];
    input_stride.w = temp[3];
    temp[0] = temp[1] = temp[2] = temp[3] = 1;

    ERROR_CHECK_STATUS(vxQueryReference(parameters[1], VX_REFERENCE_TYPE, &output_obj_type, sizeof(output_obj_type)));
    if (output_obj_type == VX_TYPE_IMAGE) {
        vx_df_image format;
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_HIP_BUFFER, &output_mem, sizeof(output_mem)));
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_OFFSET, &output_offset, sizeof(output_offset)));
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_ATTRIBUTE_AMD_GPU_BUFFER_STRIDE, &output_image_stride, sizeof(output_image_stride)));
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format, sizeof(format)));
        if(format == VX_DF_IMAGE_U8)
            output_data_type = VX_TYPE_UINT8;
        else if(format == VX_DF_IMAGE_U16)
            output_data_type = VX_TYPE_UINT16;
    } else {
        vx_size num_dims_output, output_dims[4] = { 1, 1, 1, 1 };
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims_output, sizeof(num_dims_output)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims_output], sizeof(output_dims[0])*num_dims_output));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_data_type, sizeof(output_data_type)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_OFFSET_GPU, &output_offset, sizeof(output_offset)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
        output_stride.x = temp[0];
        output_stride.y = temp[1];
        output_stride.z = temp[2];
        output_stride.w = temp[3];
        top_k = output_dims[2];
    }

    bool input_width_multiple_of_4 = (input_dims.x & 3) ? false : true;
    dim3 localThreads = dim3(8, 8, 1);
    dim3 globalThreads = dim3(((input_width_multiple_of_4 ? ((input_dims.x + 3) / 4) : input_dims.x) + localThreads.x - 1) & ~(localThreads.x - 1),
                               (input_dims.y + localThreads.y - 1) & ~(localThreads.y - 1),
                                input_dims.w);

    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &hip_stream, sizeof(hip_stream)));

    if (HipExec_Argmax_layer(hip_stream, globalThreads, localThreads, input_mem, in_offset, input_stride, input_dims, output_mem, output_offset,
        output_stride, output_image_stride, output_data_type, top_k, output_obj_type)) {
        return VX_FAILURE;
    }

    return VX_SUCCESS;

#elif ENABLE_OPENCL
    return VX_ERROR_NOT_IMPLEMENTED;
#endif
}

//! \brief The kernel publisher.
vx_status publishArgmaxLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.argmax_layer", VX_KERNEL_ARGMAX_LAYER_AMD, host_kernel, 2, validateKernel, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_REFERENCE, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxArgmaxLayer(vx_graph graph, vx_tensor input, vx_reference output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output
        };
        node = createNode(graph, VX_KERNEL_ARGMAX_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
