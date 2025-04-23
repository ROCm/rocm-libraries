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

void lut_U8U8_codegen(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global uchar * in, uint in_offset, uint4 in_stride, __read_only image1d_t lut, __global uchar * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += in_offset;\n"
            "  out += out_offset;\n"
            "  if(id < %d) {\n" // work_size
            "    out[id] = (uchar)(read_imagef(lut, in[id]).s0 * 255.0f);\n"
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size);
    opencl_code = item;
}

void lut_U8U8_codegen_packed(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global uint * in, uint in_offset, uint4 in_stride, __read_only image1d_t lut, __global uint * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += (in_offset >> 2);\n"
            "  out += (out_offset >> 2);\n"
            "  float4 f;\n"
            "  if(id < %d) {\n" // work_size
            "    f.s0 = read_imagef(lut, (int)( in[id]        & 255)).s0 * 255.0f;\n"
            "    f.s1 = read_imagef(lut, (int)((in[id] >> 8)  & 255)).s0 * 255.0f;\n"
            "    f.s2 = read_imagef(lut, (int)((in[id] >> 16) & 255)).s0 * 255.0f;\n"
            "    f.s3 = read_imagef(lut, (int)((in[id] >> 24) & 255)).s0 * 255.0f;\n"
            "    out[id] = amd_pack(f);\n"
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size);
    opencl_code = item;
}

void lut_S16U8_codegen(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size, int max_idx)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global short * lut, uint lut_count, uint lut_offset, __global short * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += in_offset;\n"
            "  out += (out_offset >> 1);\n"
            "  lut += lut_offset;\n"
            "  if(id < %d) {\n" // work_size
            "    out[id] = lut[min((int)in[id], %d)];\n"    // max_idx
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size, max_idx);
    opencl_code = item;
}

void lut_S16U8_codegen_packed(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size, int max_idx)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global uint * in, uint in_offset, uint4 in_stride, __global short * lut, uint lut_count, uint lut_offset, __global uint2 * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += (in_offset >> 2);\n"
            "  out += (out_offset >> 3);\n"
            "  lut += lut_offset;\n"
            "  if(id < %d) {\n" // work_size
            "    uint2 res;\n"
            "    res.s0  = lut[min((int)(in[id]      ) & 255, %d)] & 65535;\n"    // max_idx
            "    res.s0 |= lut[min((int)(in[id] >> 8 ) & 255, %d)] << 16;\n"      // max_idx
            "    res.s1  = lut[min((int)(in[id] >> 16) & 255, %d)] & 65535;\n"    // max_idx
            "    res.s1 |= lut[min((int)(in[id] >> 24) & 255, %d)] << 16;\n"      // max_idx
            "    out[id] = res;\n"
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size, max_idx, max_idx, max_idx, max_idx);
    opencl_code = item;
}


void lut_S16S16_codegen(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size, int min_idx, int max_idx)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global short * in, uint in_offset, uint4 in_stride, __global short * lut, uint lut_count, uint lut_offset, __global short * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += (in_offset >> 1);\n"
            "  out += (out_offset >> 1);\n"
            "  lut += lut_offset;\n"
            "  if(id < %d) {\n" // work_size
            "    int idx = min(max((int)in[id], %d), %d);\n"    // min_idx, max_idx
            "    out[id] = lut[idx];\n"
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size, min_idx, max_idx);
    opencl_code = item;
}

void lut_S16S16_codegen_packed(std::string& opencl_code, char * kern_name, vx_size local_wg_size, vx_uint32 work_size, int min_idx, int max_idx)
{
    char item[8192];
    snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global uint * in, uint in_offset, uint4 in_stride, __global short * lut, uint lut_count, uint lut_offset, __global uint * out, uint out_offset, uint4 out_stride)\n" // opencl_kernel_function_name
            "{\n"
            "  size_t id = get_global_id(0);\n"
            "  in  += (in_offset >> 2);\n"
            "  out += (out_offset >> 2);\n"
            "  lut += lut_offset;\n"
            "  if(id < %d) {\n" // work_size
            "    uint res;\n"
            "    res  = lut[min(max((int)( in[id]        & 65535), %d), %d)];\n"   // min_idx, max_idx
            "    res |= lut[min(max((int)((in[id] >> 16) & 65535), %d), %d)] << 16;\n"   // min_idx, max_idx
            "    out[id] = res;\n"
            "  }\n"
            "}\n"
            , (int)local_wg_size, kern_name, (int) work_size, min_idx, max_idx, min_idx, max_idx);
    opencl_code = item;
}

static vx_status VX_CALLBACK validateTensorTableLookup(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_enum lut_type, input_type, output_type;
    vx_size input_ndims = 0, output_ndims = 0;
    vx_size input_dims[4], output_dims[4];
    vx_uint8 input_fixedpt_pos = 0, output_fixedpt_pos = 0;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &input_ndims, sizeof(input_ndims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_FIXED_POINT_POSITION, &input_fixedpt_pos, sizeof(input_fixedpt_pos)));
    if((input_type != VX_TYPE_UINT8) && (input_type != VX_TYPE_INT16)) return VX_ERROR_INVALID_TYPE;

    ERROR_CHECK_STATUS(vxQueryLUT((vx_lut)parameters[1], VX_LUT_TYPE, &lut_type, sizeof(lut_type)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &output_ndims, sizeof(output_ndims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_FIXED_POINT_POSITION, &output_fixedpt_pos, sizeof(output_fixedpt_pos)));

    if(input_ndims != output_ndims) return VX_ERROR_INVALID_DIMENSION;
    if((input_ndims >= 1) && (input_dims[0] != output_dims[0])) return VX_ERROR_INVALID_DIMENSION;
    if((input_ndims >= 2) && (input_dims[1] != output_dims[1])) return VX_ERROR_INVALID_DIMENSION;
    if((input_ndims >= 3) && (input_dims[2] != output_dims[2])) return VX_ERROR_INVALID_DIMENSION;
    if((input_ndims >= 4) && (input_dims[3] != output_dims[3])) return VX_ERROR_INVALID_DIMENSION;
    if(lut_type != output_type) return VX_ERROR_INVALID_TYPE;
    if((input_type == VX_TYPE_INT16) && (output_type == VX_TYPE_UINT8)) return VX_ERROR_INVALID_TYPE;

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &output_ndims, sizeof(output_ndims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK tensorTableLookup_query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
)
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK tensorTableLookup_opencl_codegen(
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
    vx_size dims[4], num_of_dims = 0, lut_count = 0;
    vx_enum input_type, output_type;
    vx_uint32 lut_offs = 0;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));

    ERROR_CHECK_STATUS(vxQueryLUT((vx_lut)parameters[1], VX_LUT_OFFSET, &lut_offs, sizeof(lut_offs)));
    ERROR_CHECK_STATUS(vxQueryLUT((vx_lut)parameters[1], VX_LUT_COUNT, &lut_count, sizeof(lut_count)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));

    strcpy(opencl_kernel_function_name, "tensor_table_lookup");
    vx_uint32 work_size = 1;
    for(int i = 0; i < num_of_dims; i++) {
        work_size *= dims[i];
    }

    bool use_packed_kern = false;

    if(output_type == VX_TYPE_UINT8) {
        use_packed_kern = ((work_size & 3) == 0) ? true : false;
        if(use_packed_kern) work_size >>= 2;        // four pixels processed in a work item
    }
    else {
        if(input_type == VX_TYPE_UINT8) {
            use_packed_kern = ((work_size & 3) == 0) ? true : false;
            if(use_packed_kern) work_size >>= 2;    // four pixels processed in a work item
        }
        else {
            use_packed_kern = ((work_size & 1) == 0) ? true : false;
            if(use_packed_kern) work_size >>= 1;    // two pixels processed in a work item
        }
    }

    opencl_work_dim = 1;
    opencl_local_work[0] = 128;
    opencl_global_work[0] = (work_size + (opencl_local_work[0] - 1)) & ~(opencl_local_work[0] - 1);

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if(output_type == VX_TYPE_UINT8) {
        if(use_packed_kern)
            lut_U8U8_codegen_packed(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size);
        else
            lut_U8U8_codegen(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size);
    }
    else {
        int min_idx = -1 * (int)lut_offs;
        int max_idx = (int) (lut_count - lut_offs - 1);
        if(input_type == VX_TYPE_UINT8) {
            if(use_packed_kern)
                lut_S16U8_codegen_packed(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size, max_idx);
            else
                lut_S16U8_codegen(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size, max_idx);
        }
        else {
            if(use_packed_kern)
                lut_S16S16_codegen_packed(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size, min_idx, max_idx);
            else
                lut_S16S16_codegen(opencl_kernel_code, opencl_kernel_function_name, opencl_local_work[0], work_size, min_idx, max_idx);
        }
    }

    return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK tensorTableLookup_host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishTensorTableLookup(vx_context context) {

    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.openvx.tensor_table_lookup", VX_KERNEL_TENSOR_TABLE_LOOKUP, tensorTableLookup_host_kernel, 3, validateTensorTableLookup, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = tensorTableLookup_query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = tensorTableLookup_opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_LUT, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorTableLookupNode(vx_graph graph, vx_tensor input1, vx_lut lut, vx_tensor output)
{
    vx_node node = NULL;
    vx_reference params[] = {
        (vx_reference) input1,
        (vx_reference) lut,
        (vx_reference) output
    };
    node = createNode(graph, VX_KERNEL_TENSOR_TABLE_LOOKUP, params, sizeof(params) / sizeof(params[0]));
    return node;
}
