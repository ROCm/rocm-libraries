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

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 && num_dims != 2) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_FLOAT32) && (type!= VX_TYPE_INT32) && (type!= VX_TYPE_INT64)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: cast: #1 input tensor data type=%d not supprted yet\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_int32 output_data_type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[1], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &output_data_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(output_data_type < 0 || output_data_type > 13) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: cast: #2 scalar type=%d ('to' must be between 0-13)\n", output_data_type);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4 && num_dims != 2) return VX_ERROR_INVALID_DIMENSION;
    if ((type != VX_TYPE_INT64) && (type!= VX_TYPE_INT32) && (type!= VX_TYPE_FLOAT32)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: cast: #3 output tensor data type=%d not supprted yet\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

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
	vx_enum input_type, output_type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_int32 output_data_type;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &output_data_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

#if ENABLE_DEBUG_PRINT_DIMS
    if (num_dims == 4) { 
    	std::cout << "cast_layer input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    	std::cout << "cast_layer output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
    }
    else if(num_dims == 2) {
    	std::cout << "cast_layer input " << " " << input_dims[1] << " " << input_dims[0] << " ";
    	std::cout << "cast_layer output " << " " << output_dims[1] << " " << output_dims[0] << std::endl;
    }
    
#endif

    vx_size tot_count_elements = input_dims[0]*input_dims[1];
    bool input_element_count_multiple_of_4 = (tot_count_elements & 3) ? false : true;

	strcpy(opencl_kernel_function_name, "cast_layer");

    if (num_dims == 2) { 
        opencl_work_dim = 2;
        opencl_global_work[0] = input_dims[0];
        opencl_global_work[1] = input_dims[1];
    } 
    else if (num_dims == 4) {
        opencl_work_dim = 3;
        opencl_global_work[0] = input_dims[0];
        opencl_global_work[1] = input_dims[1];
        opencl_global_work[2] = input_dims[2] * input_dims[3];
    }
    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_dims == 2 || num_dims == 4) {
        char item[8192];
        snprintf(item, sizeof(item),
                "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
                "__kernel void %s(__global uchar * in, uint in_offset, uint4 in_stride, const int output_data_type, __global uchar * out, uint out_offset, uint4 out_stride) \n"
                "{ \n"
                , opencl_kernel_function_name);
        opencl_kernel_code = item;
        if (num_dims == 2) {
                snprintf(item, sizeof(item),
                "    uint x = get_global_id(0) * %d;\n"
		        "    uint y = get_global_id(1);\n"
		        "    in += in_offset + y * in_stride.s1 + x * in_stride.s0;\n"
		        "    out += out_offset + y * out_stride.s1 + x * out_stride.s0;\n"
                , input_element_count_multiple_of_4 ? 4 : 1);
            opencl_kernel_code += item;
        }
        else if (num_dims == 4){
            snprintf(item, sizeof(item),
                "   uint x = get_global_id(0) * %d;\n"
                "   uint y = get_global_id(1);\n"
                "   uint c = get_global_id(2);\n"
                "   in += in_offset + c * in_stride.s2 + y * in_stride.s1 + x * in_stride.s0;\n"
                "   out += out_offset + c * out_stride.s2 + y * out_stride.s1 + x * out_stride.s0;\n"
                , input_element_count_multiple_of_4 ? 4 : 1);
            opencl_kernel_code += item;
        }
        if(input_element_count_multiple_of_4) {
        	if(input_type == VX_TYPE_FLOAT32) {
        		if(output_type == VX_TYPE_INT32) {
        			snprintf(item, sizeof(item),
                	"	float4 f = *(__global float4 *)in; \n"
                	"	int4 i = convert_int4(f);  \n"
                	"	*(__global int4 *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
        		else if(output_type == VX_TYPE_INT64) {
        			snprintf(item, sizeof(item),
                	"	float4 f = *(__global float4 *)in; \n"
                	"	long4 i = convert_long4(f);  \n"
                	"	*(__global long4 *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
                else if(output_type == VX_TYPE_FLOAT32) {
                    snprintf(item, sizeof(item),
                    "   float4 i = *(__global float4 *)in; \n"
                    "   *(__global float4 *)&out[0] = i; \n"
                    );
                    opencl_kernel_code += item;
                }   
        	}
        	else if(input_type == VX_TYPE_INT32) {
        		if(output_type == VX_TYPE_INT64) {
        			snprintf(item, sizeof(item),
                	"	int4 f = *(__global int4 *)in; \n"
                	"	long4 i = convert_long4(f);  \n"
                	"	*(__global long4 *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
        	}
        	else if(input_type == VX_TYPE_INT64) {
        		if(output_type == VX_TYPE_INT32) {
        			snprintf(item, sizeof(item),
        			"	long4 f = *(__global long4 *)in; \n"
                	"	int4 i = convert_int4(f); \n"
                	"	*(__global int4 *)&out[0] = i; \n"
                	);
        			opencl_kernel_code += item;
        		}
        	}	
        }
        else {
        	if(input_type == VX_TYPE_FLOAT32) {
        		if(output_type == VX_TYPE_INT32) {
        			snprintf(item, sizeof(item),
                	"	float f = *(__global float *)in; \n"
                	"	int i = convert_int(f); \n"
                	"	*(__global int *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
        		else if(output_type == VX_TYPE_INT64) {
        			snprintf(item, sizeof(item),
                	"	float f = *(__global float *)in; \n"
                	"	long i = convert_long(f); \n"
                	"	*(__global long *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
                else if(output_type == VX_TYPE_FLOAT32) {
                    snprintf(item, sizeof(item),
                    "   float i = *(__global float *)in; \n"
                    "   *(__global float *)&out[0] = i; \n"
                    );
                    opencl_kernel_code += item;
                }
        	}
        	else if(input_type == VX_TYPE_INT32) {
        		if(output_type == VX_TYPE_INT64) {
        			snprintf(item, sizeof(item),
                	"	int f = *(__global int *)in; \n"
                	"	long i = convert_long(f); \n"
                	"	*(__global long *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
        	}
        	else if(input_type == VX_TYPE_INT64) {
        		if(output_type == VX_TYPE_INT32) {        			
        			snprintf(item, sizeof(item),
                	"	long f = *(__global long *)in; \n"
                	"	int i = convert_int(f); \n"
                	"	*(__global int *)&out[0] = i; \n"
        			);
        			opencl_kernel_code += item;
        		}
        	}
		}
		opencl_kernel_code +=
        "}\n";
	}
	return VX_SUCCESS;
}
#endif

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
#if ENABLE_HIP
    vx_enum input_type, output_type;
    vx_size num_dims;
    vx_size input_dims[4] = {1, 1, 1, 1};
    vx_size output_dims[4] = {1, 1, 1, 1};

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_type, sizeof(input_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_int32 output_data_type;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[1], &output_data_type, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &output_type, sizeof(output_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    dim3 globalThreads = dim3(1);
    globalThreads.x = input_dims[0];
    globalThreads.y = input_dims[1];
    globalThreads.z = input_dims[2] * input_dims[3];

    vx_size temp[4] = {0};
    uint4 input_stride, output_stride;
    vx_size in_offset, output_offset;
    unsigned char *input_mem = NULL;
    unsigned char *output_mem = NULL;
    hipStream_t hip_stream;

    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &hip_stream, sizeof(hip_stream)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HIP, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_OFFSET_GPU, &in_offset, sizeof(in_offset)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_OFFSET_GPU, &output_offset, sizeof(output_offset)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    input_stride.x = temp[0];
    input_stride.y = temp[1];
    input_stride.z = temp[2];
    input_stride.w = temp[3];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_STRIDE_GPU, temp, sizeof(temp)));
    output_stride.x = temp[0];
    output_stride.y = temp[1];
    output_stride.z = temp[2];
    output_stride.w = temp[3];

    if (HipExec_Cast_layer(hip_stream, globalThreads, dim3(1), input_type, output_type, input_mem, in_offset, input_stride,
            output_mem, output_offset, output_stride)) {
        return VX_FAILURE;
    }

    return VX_SUCCESS;

#elif ENABLE_OPENCL
    return VX_ERROR_NOT_IMPLEMENTED;
#endif
}

//! \brief The kernel publisher.
vx_status publishCastLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.cast_layer", VX_KERNEL_CAST_LAYER_AMD, host_kernel, 3, validate, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));

#if ENABLE_OPENCL
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
#endif

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxCastLayer(vx_graph graph, vx_tensor input, vx_int32 output_data_type, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
    	vx_scalar s_output_data_type = vxCreateScalarWithSize(context, VX_TYPE_INT32, &output_data_type, sizeof(output_data_type));
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)s_output_data_type,
            (vx_reference)output,
        };
        node = createNode(graph, VX_KERNEL_CAST_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
