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

static vx_status VX_CALLBACK validateCropAndResizeLayer(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[]) {
    vx_enum type, out_type;
    vx_size num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    vx_int32 x_coord, y_coord, width, height, mode, scaleFactor;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &x_coord, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &y_coord, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[6], &scaleFactor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    if (x_coord < 0 || y_coord < 0 || x_coord > input_dims[0] || y_coord > input_dims[1]) {
        printf("Crop coordinates out of bound\n");
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if (x_coord + width > input_dims[0] || y_coord + height > input_dims[1]) {
        printf("Crop width/height out of bound\n");
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if (scaleFactor <= 0) {
        printf("The scale factor has to be a positive integer\n");
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if (mode != 0 && mode != 1) {
        printf("Mode should be either 0 or 1\n");
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if (output_dims[0] != width*scaleFactor || output_dims[1] != height*scaleFactor) {
        printf("Output tensor's width/height should match the crop width/height multiplied by the scale factor\n");
        return VX_ERROR_INVALID_PARAMETERS;
    }

    if (out_type != type) return VX_ERROR_INVALID_TYPE;
    if (output_dims[2] != input_dims[2] || output_dims[3] != input_dims[3]) return VX_ERROR_INVALID_DIMENSION;
    
    out_type = type;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));
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
    //get tensor dimensions
    vx_size input_dims[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum type;
    vx_uint32 x_coord, y_coord, width, height, mode;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &x_coord, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &y_coord, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[7], &mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    
    strcpy(opencl_kernel_function_name, "crop_and_resize_layer");

    vx_uint32 input_dim_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];

    opencl_work_dim = 3;

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;
    
    if (num_of_dims == 4) {
        char item[8192];
        if (mode == 0) {
            opencl_global_work[0] = width;
            opencl_global_work[1] = height;
            opencl_global_work[2] = input_dims[2];

            if (type == VX_TYPE_FLOAT32) {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint x_coord, uint y_coord, uint width, uint height, uint scaleFactor, uint mode) \n"
                "{ \n"
                "   uint x = get_global_id(0) + %d;\n"
                "   uint y = get_global_id(1) + %d;\n"
                "   uint c = get_global_id(2);\n"
                "   float value = *(__global float*)&in[in_offset + x*in_stride.s0 + y*in_stride.s1 + c*in_stride.s2];\n"
                "   out += out_offset + get_global_id(0)*out_stride.s0*scaleFactor + get_global_id(1)*out_stride.s1*scaleFactor + get_global_id(2)*out_stride.s2;\n"
                "   for (uint s0 = 0; s0 < scaleFactor; s0++) {\n"
                "       for (uint s1 = 0; s1 < scaleFactor; s1++) {\n"
                "           *(__global float *)&out[s0*out_stride.s0 + s1*out_stride.s1] = value;\n"
                "       }\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, (int)x_coord, (int)y_coord);
            }
            else {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint x_coord, uint y_coord, uint width, uint height, uint scaleFactor, uint mode) \n"
                "{ \n"
                "   uint x = get_global_id(0) + %d;\n"
                "   uint y = get_global_id(1) + %d;\n"
                "   uint c = get_global_id(2);\n"
                "   half value = *(__global half*)&in[in_offset + x*in_stride.s0 + y*in_stride.s1 + c*in_stride.s2];\n"
                "   out += out_offset + get_global_id(0)*out_stride.s0*scaleFactor + get_global_id(1)*out_stride.s1*scaleFactor + get_global_id(2)*out_stride.s2;\n"
                "   for (uint s0 = 0; s0 < scaleFactor; s0++) {\n"
                "       for (uint s1 = 0; s1 < scaleFactor; s1++) {\n"
                "           *(__global half *)&out[s0*out_stride.s0 + s1*out_stride.s1] = value;\n"
                "       }\n"
                "   }\n"
                "}\n", opencl_kernel_function_name, (int)x_coord, (int)y_coord);
            }
        }
        else {
            opencl_global_work[0] = output_dims[0];
            opencl_global_work[1] = output_dims[1];
            opencl_global_work[2] = input_dims[2];

            if (type == VX_TYPE_FLOAT32) {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint x_coord, uint y_coord, uint width, uint height, uint scaleFactor, uint mode) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                
                "   uint px = (int)(x / scaleFactor);\n"
                "   uint py = (int)(y / scaleFactor);\n"

                "   uint nx = px + %d;\n"
                "   uint ny = py + %d;\n"

                "   float fx1 = (float)x / (float)scaleFactor - (float)px;\n"
                "   float fx2 = 1 - fx1;\n"
                "   float fy1 = (float)y / (float)scaleFactor - (float)py;\n"
                "   float fy2 = 1 - fy1;\n"

                "   float w1 = fx2 * fy2;\n"
                "   float w2 = fx1 * fy2;\n"
                "   float w3 = fx2 * fy1;\n"
                "   float w4 = fx1 * fy1;\n"

                "   float value1 = *(__global float*)&in[in_offset + nx*in_stride.s0 + ny*in_stride.s1 + c*in_stride.s2];\n"
                "   float value2 = *(__global float*)&in[in_offset + (nx+1)*in_stride.s0 + ny*in_stride.s1 + c*in_stride.s2];\n"
                "   float value3 = *(__global float*)&in[in_offset + nx*in_stride.s0 + (ny+1)*in_stride.s1 + c*in_stride.s2];\n"
                "   float value4 = *(__global float*)&in[in_offset + (nx+1)*in_stride.s0 + (ny+1)*in_stride.s1 + c*in_stride.s2];\n"

                "   out += out_offset + get_global_id(0)*out_stride.s0 + get_global_id(1)*out_stride.s1 + get_global_id(2)*out_stride.s2;\n"
                "   *(__global float *)&out[0] = w1*value1 + w2*value2 + w3*value3 + w4*value4;\n"
                "}\n", opencl_kernel_function_name, (int)x_coord, (int)y_coord);
            }
            else {
            snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, __global uchar * out, uint out_offset, uint4 out_stride, uint x_coord, uint y_coord, uint width, uint height, uint scaleFactor, uint mode) \n"
                "{ \n"
                "   uint x = get_global_id(0);\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                
                "   uint px = (int)(x / scaleFactor);\n"
                "   uint py = (int)(y / scaleFactor);\n"

                "   uint nx = px + %d;\n"
                "   uint ny = py + %d;\n"

                "   half fx1 = (half)x / (half)scaleFactor - (half)px;\n"
                "   half fx2 = 1 - fx1;\n"
                "   half fy1 = (half)y / (half)scaleFactor - (half)py;\n"
                "   half fy2 = 1 - fy1;\n"

                "   half w1 = fx2 * fy2;\n"
                "   half w2 = fx1 * fy2;\n"
                "   half w3 = fx2 * fy1;\n"
                "   half w4 = fx1 * fy1;\n"

                "   half value1 = *(__global halft*)&in[in_offset + nx*in_stride.s0 + ny*in_stride.s1 + c*in_stride.s2];\n"
                "   half value2 = *(__global half*)&in[in_offset + (nx+1)*in_stride.s0 + ny*in_stride.s1 + c*in_stride.s2];\n"
                "   half value3 = *(__global half*)&in[in_offset + nx*in_stride.s0 + (ny+1)*in_stride.s1 + c*in_stride.s2];\n"
                "   half value4 = *(__global half*)&in[in_offset + (nx+1)*in_stride.s0 + (ny+1)*in_stride.s1 + c*in_stride.s2];\n"

                "   out += out_offset + get_global_id(0)*out_stride.s0 + get_global_id(1)*out_stride.s1 + get_global_id(2)*out_stride.s2;\n"
                "   *(__global half *)&out[0] = w1*value1 + w2*value2 + w3*value3 + w4*value4;\n"
                "}\n", opencl_kernel_function_name, (int)x_coord, (int)y_coord);
            }
        }
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}


static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    return VX_ERROR_NOT_IMPLEMENTED;
}

vx_status publishCropAndResizeLayer(vx_context context) {
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.crop_and_resize_layer", VX_KERNEL_CROP_AND_RESIZE_LAYER_AMD, host_kernel, 8, validateCropAndResizeLayer, NULL, NULL);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS; 
}

VX_API_ENTRY vx_node VX_API_CALL vxCropAndResizeLayer(vx_graph graph, vx_tensor input, vx_tensor output, vx_scalar x_coord, vx_scalar y_coord, vx_scalar width, vx_scalar height, vx_scalar scaleFactor, vx_scalar mode) 
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) input,
            (vx_reference) output,
            (vx_reference) x_coord,
            (vx_reference) y_coord,
            (vx_reference) width,
            (vx_reference) height,
            (vx_reference) scaleFactor,
            (vx_reference) mode
        };
        node = createNode(graph, VX_KERNEL_CROP_AND_RESIZE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
