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

void slice_codegen_batchsz1(std::string& opencl_code, vx_size work_items, vx_size input_dims[4], int num_outputs, vx_size op_size_per_batch[8])
{
    vx_size op_buffer_offset[8];
    for(int i = 0; i < num_outputs; i++) {
        op_buffer_offset[i] = 0;
        for(int j = 0; j < i; j++) {
            op_buffer_offset[i] += op_size_per_batch[j];
        }
    }

    char item[8192];
    snprintf(item, sizeof(item),
        "{\n"
        "  size_t id = get_global_id(0);\n"
        "  if(id < %ld)\n"  // work_items
        "  {\n"
        "    in += in_offset >> 2;\n\n"
        , work_items);
    opencl_code += item;

    snprintf(item, sizeof(item),
        "    if(id < %ld)\n"   // op_size_per_batch[0]
        "    {\n"
        "      out0 = out0 + (out0_offset >> 2);\n"
        "      out0[id] = in[id];\n"
        "    }\n"
        , op_size_per_batch[0]);
    opencl_code += item;

    for(int i = 1; i < num_outputs; i++) {
        snprintf(item, sizeof(item),
            "    else if((id >= %ld) && (id < %ld))\n"  // op_buffer_offset[i], op_buffer_offset[i] + op_size_per_batch[i]
            "    {\n"
            "      out%d = out%d + (out%d_offset >> 2);\n"    // i, i, i
            "      out%d[id - %ld] = in[id];\n"    // i, ip_buffer_offset[i]
            "    }\n"
            , op_buffer_offset[i], op_buffer_offset[i] + op_size_per_batch[i], i, i, i, i, op_buffer_offset[i]);
        opencl_code += item;
    }
    opencl_code +=
            "  }\n"
            "}\n";
}

void slice_codegen_batchszN(std::string& opencl_code, vx_size work_items, vx_size input_dims[4], int num_outputs, vx_size op_size_per_batch[8])
{
    vx_size op_buffer_offset[8];
    for(int i = 0; i < num_outputs; i++) {
        op_buffer_offset[i] = 0;
        for(int j = 0; j < i; j++) {
            op_buffer_offset[i] += op_size_per_batch[j];
        }
    }

    char item[8192];
    snprintf(item, sizeof(item),
        "{\n"
        "  size_t id = get_global_id(0);\n"
        "  if(id < %ld)\n"  // work_items
        "  {\n"
        "    size_t batch_id = id / %ld;     // in_c*in_h*in_w\n"  // input_dims[2] * input_dims[1] * input_dims[0]
        "    size_t id_within_batch = id - batch_id * %ld;\n\n"    // input_dims[2] * input_dims[1] * input_dims[0]
        "    in += in_offset >> 2;\n\n"
        , work_items, input_dims[2] * input_dims[1] * input_dims[0], input_dims[2] * input_dims[1] * input_dims[0]);
    opencl_code += item;

    snprintf(item, sizeof(item),
        "    if(id_within_batch < %ld)\n"   // op_size_per_batch[0]
        "    {\n"
        "      out0 = out0 + (out0_offset >> 2) + (batch_id * %ld);\n"   // op_size_per_batch[0]
        "      out0[id_within_batch] = in[id];\n"
        "    }\n"
        , op_size_per_batch[0], op_size_per_batch[0]);
    opencl_code += item;

    for(int i = 1; i < num_outputs; i++) {
        snprintf(item, sizeof(item),
            "    else if((id_within_batch >= %ld) && (id_within_batch < %ld))\n"  // op_buffer_offset[i], op_buffer_offset[i] + op_size_per_batch[i]
            "    {\n"
            "      out%d = out%d + (out%d_offset >> 2) + (batch_id * %ld);\n"    // i, i, i, op_size_per_batch[i]
            "      out%d[id_within_batch - %ld] = in[id];\n"    // i, op_buffer_offset[i]
            "    }\n"
            , op_buffer_offset[i], op_buffer_offset[i] + op_size_per_batch[i], i, i, i, op_size_per_batch[i], i, op_buffer_offset[i]);
        opencl_code += item;
    }
    opencl_code +=
            "  }\n"
            "}\n";
}

static vx_status VX_CALLBACK validateSliceLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    //check tensor dims.
    vx_enum in_type, type;
    vx_size num_dims;
    vx_size input_dims[4], outputn_dims[4], num_channels = 0;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #0 num_dims=%ld (must be 4)\n", num_dims);
    if ((in_type != VX_TYPE_FLOAT32) && (in_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: concat: #1 type=%d (must be float/float16)\n", in_type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #1 num_dims=%ld (must be 4)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: slice: #1 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));
    if (outputn_dims[3] != input_dims[3] || outputn_dims[1] != input_dims[1] || outputn_dims[0] != input_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #1 dims input[%ld,%ld,%ld,%ld] != outputn[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    outputn_dims[0], outputn_dims[1], outputn_dims[2], outputn_dims[3]);
    num_channels = outputn_dims[2];
    //output tensor configuration
    type = in_type;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #2 num_dims=%ld (must be 4)\n", num_dims);
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: slice: #2 type=%d (must be float/float16)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));
    if (outputn_dims[3] != input_dims[3] || outputn_dims[1] != input_dims[1] || outputn_dims[0] != input_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #2 dims input[%ld,%ld,%ld,%ld] != outputn[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    outputn_dims[0], outputn_dims[1], outputn_dims[2], outputn_dims[3]);
    num_channels += outputn_dims[2];
    //output tensor configuration
    type = in_type;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));

    int i = 3;
    while(parameters[i] && (i < 9)) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        if (num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #%d num_dims=%ld (must be 4)\n", i, num_dims);
        if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: slice: #%d type=%d (must be float/float16)\n", i, type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));
        if (outputn_dims[3] != input_dims[3] || outputn_dims[1] != input_dims[1] || outputn_dims[0] != input_dims[0])
            return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: #%d dims input[%ld,%ld,%ld,%ld] != outputn[%ld,%ld,%ld,%ld]\n", i,
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    outputn_dims[0], outputn_dims[1], outputn_dims[2], outputn_dims[3]);
        num_channels += outputn_dims[2];

        //output tensor configuration
        type = in_type;
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[i], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[i], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[i], VX_TENSOR_DIMS, outputn_dims, sizeof(outputn_dims)));

        i++;
    }

    if(num_channels != input_dims[2]) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: slice: num_channels=%ld != input_dims[2]=%ld\n", num_channels, input_dims[2]);

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
    //get tensor dimensions
    vx_size input_dims[4];
    vx_size op_size_per_batch[8], batch_size = 0, ip_batch_stride = 0;
    vx_enum type;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    batch_size = input_dims[3];
    ip_batch_stride = input_dims[2] * input_dims[1] * input_dims[0];
#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "slice input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << std::endl;
#endif

    int i = 1;
    while(parameters[i] && (i < 9)) {
        vx_size output_dims[4];
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
        op_size_per_batch[i-1] = output_dims[2] * output_dims[1] * output_dims[0];
#if ENABLE_DEBUG_PRINT_DIMS
        std::cout << "slice output " << i << " " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

        i++;
    }
    int num_outputs = i - 1;

    strcpy(opencl_kernel_function_name, "slice_layer");
    vx_size work_items = input_dims[3] * input_dims[2] * input_dims[1] * input_dims[0];
    opencl_work_dim = 1;
    opencl_local_work[0] = 128;
    opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    char item[8192];
    if (type == VX_TYPE_FLOAT32){
        snprintf(item, sizeof(item),
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global float * in, uint in_offset, uint4 in_stride" // opencl_kernel_function_name
            , (int)opencl_local_work[0], opencl_kernel_function_name);
    }else
    {
        snprintf(item, sizeof(item),
            "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
            "__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n"    // opencl_local_work[0]
            "void %s(__global half * in, uint in_offset, uint4 in_stride" // opencl_kernel_function_name
            , (int)opencl_local_work[0], opencl_kernel_function_name);
    }
    opencl_kernel_code = item;

    for(int i = 0; i < num_outputs; i++) {
        if (type == VX_TYPE_FLOAT32){
        snprintf(item, sizeof(item),
            ",\n"
            "                  __global float * out%d, uint out%d_offset, uint4 out%d_stride"  // i, i, i
            , i, i, i);
        }else
        {
            snprintf(item, sizeof(item),
                ",\n"
                "                  __global half * out%d, uint out%d_offset, uint4 out%d_stride"  // i, i, i
                , i, i, i);
        }
        opencl_kernel_code += item;
    }
    opencl_kernel_code += ")\n";

    if(input_dims[3] == 1) {
        slice_codegen_batchsz1(opencl_kernel_code, work_items, input_dims, num_outputs, op_size_per_batch);
    }
    else {
        slice_codegen_batchszN(opencl_kernel_code, work_items, input_dims, num_outputs, op_size_per_batch);
    }

    return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishSliceLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.slice_layer", VX_KERNEL_SLICE_LAYER_AMD, host_kernel, 9, validateSliceLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxSliceLayer(vx_graph graph, vx_tensor input, vx_tensor output1, vx_tensor output2, vx_tensor output3, vx_tensor output4, vx_tensor output5, vx_tensor output6, vx_tensor output7, vx_tensor output8)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output1,
            (vx_reference)output2,
            (vx_reference)output3,
            (vx_reference)output4,
            (vx_reference)output5,
            (vx_reference)output6,
            (vx_reference)output7,
            (vx_reference)output8
        };
        node = createNode(graph, VX_KERNEL_SLICE_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
