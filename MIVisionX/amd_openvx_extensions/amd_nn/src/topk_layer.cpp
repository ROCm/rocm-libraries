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

typedef struct TopKLayerLocalData {
	float *x_tensor_buffer;
	int64_t k_tensor_buffer[1];
	vx_int32 axis;
	vx_int32 largest;
	vx_int32 sorted;
}TopKLayerLocalData;

TopKLayerLocalData *data = NULL;

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
	// check tensor dims.
	vx_enum type;
	vx_size num_dims;
	vx_size input_dims_1[4], input_dims_2[4], output_dims_1[4], output_dims_2[4];

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
	if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #1 input tensor data type=%d (must be float32) (other types not supported yet)\n", type);
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
	if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #2 input tensor data type=%d (must be int64)\n", type);
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_2, sizeof(input_dims_2)));

	vx_int32 axis;
	ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
	if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	if(axis < -4 || axis > 3) 
		return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #3 scalar type=%d ('axis' must be between [-r,r-1])(r = rank of input)\n", axis);

	vx_int32 largest;
	ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
	if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &largest, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	if(largest != 0 && largest != 1) 
		return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #4 scalar type=%d ('largest' must be either 0/1)\n", largest);

	vx_int32 sorted;
	ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
	if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &sorted, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	if(sorted != 0 && sorted != 1) 
		return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: TopK: #5 scalar type=%d ('sorted' must be either 0/1)\n", sorted);

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
	if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #6 output tensor data 'values' type=%d (must be float32)\n", type);
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims_1, sizeof(output_dims_1)));

	// output tensor configuration
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DIMS, output_dims_1, sizeof(output_dims_1)));

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
	if (type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: TopK: #7 output tensor data 'indices' type=%d (must be int64)\n", type);
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));

	// output tensor configuration
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[6], VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));

	return VX_SUCCESS;
}

static vx_status VX_CALLBACK processTopKLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//get tensor dimensions
	vx_size input_dims_0[4], input_dims_1[4], output_dims_0[4], output_dims_1[4];
	vx_size num_of_dims;

	vx_enum usage = VX_READ_ONLY;
	vx_status status;
	vx_map_id map_id;
	vx_size stride[4];

	//query and copy inputs
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
	
	float * ptr_input_0;
	vx_size count_input_dims_0 = input_dims_0[0]*input_dims_0[1]*input_dims_0[2]*input_dims_0[3];

	ERROR_CHECK_STATUS(vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr_input_0, usage, VX_MEMORY_TYPE_HOST));

	memcpy(data->x_tensor_buffer, ptr_input_0, (count_input_dims_0*sizeof(float)));

	ERROR_CHECK_STATUS(vxUnmapTensorPatch((vx_tensor)parameters[0], map_id));

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims_1, sizeof(input_dims_1)));
	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
	
	int64_t * ptr_input_1;
	vx_size count_input_dims_1 = input_dims_1[0]*input_dims_1[1]*input_dims_1[2]*input_dims_1[3];

	ERROR_CHECK_STATUS(vxMapTensorPatch((vx_tensor)parameters[1], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr_input_1, usage, VX_MEMORY_TYPE_HOST));

	memcpy(&data->k_tensor_buffer, ptr_input_1, (count_input_dims_1*sizeof(int64_t)));

	ERROR_CHECK_STATUS(vxUnmapTensorPatch((vx_tensor)parameters[1], map_id));

	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &data->largest, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &data->sorted, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

	//output vectors
	std::vector<float> values;
	std::vector<int64_t> indices;

	//opposite of onnx. Maybe have to remove for model. Works with GDF testing
	data->axis = 3 - data->axis;
	
	//accessing the tensor from last dimension
	if(data->axis > 3)
	{
		int count = 0, dims_idx = 0;
		//get real dimensions. Ignoring the appended 1s
		while(dims_idx <= 3 && input_dims_0[dims_idx] == 1)
		{
			count++;
			dims_idx++;
		}

		count = 4 - count;
		data->axis = data->axis - count;
	}

	//vector to sort indices
	std::vector<size_t> idx(input_dims_0[data->axis]);  
	//temporary moving ptr
	float *x_tensor_temp = data->x_tensor_buffer;    
	
	for(int i = 0; i < count_input_dims_0; i += input_dims_0[data->axis])
	{
		std::iota(idx.begin(), idx.end(), 0); 
		if (data->largest)
		{   
			std::sort(idx.begin(), idx.end(), [&x_tensor_temp](const size_t &a, const size_t &b)
												{ return x_tensor_temp[a] > x_tensor_temp[b];});   
		}
		else
		{ 
			std::sort(idx.begin(), idx.end(), [&x_tensor_temp](const size_t &a, const size_t &b)
											   { return x_tensor_temp[a] < x_tensor_temp[b];});  
		}
		
		//keep only top k elements
		int keep_elements = (input_dims_0[data->axis] < data->k_tensor_buffer[0]) ? input_dims_0[data->axis]:data->k_tensor_buffer[0];
		for (int j = 0; j < keep_elements; j++)
		{
			values.push_back(x_tensor_temp[idx[j]]);
			indices.push_back(idx[j]);
		}
		x_tensor_temp += input_dims_0[data->axis];
	}

	float* values_ptr = &values[0];
	int64_t* indices_ptr = &indices[0];

	//finding size of topk output and assigning stride
	output_dims_0[3] = 1; 
	output_dims_0[2] = 1;
	output_dims_0[1] = 1; 
	output_dims_0[0] = values.size(); //total count of values. Acts as a 1D array

	output_dims_1[3] = 1; 
	output_dims_1[2] = 1;
	output_dims_1[1] = 1; 
	output_dims_1[0] = indices.size(); //total count of indices. Acts as a 1D array    

	vx_size stride_output_0[4] = {sizeof(float), output_dims_0[0]*sizeof(float), output_dims_0[0]*output_dims_0[1]*sizeof(float), output_dims_0[0]*output_dims_0[1]*output_dims_0[2]*sizeof(float) };
	vx_size stride_output_1[4] = {sizeof(int64_t), output_dims_1[0]*sizeof(int64_t), output_dims_1[0]*output_dims_1[1]*sizeof(int64_t), output_dims_1[0]*output_dims_1[1]*output_dims_1[2]*sizeof(int64_t) };

	ERROR_CHECK_STATUS(vxCopyTensorPatch((vx_tensor)parameters[5], 4, nullptr, nullptr, stride_output_0, values_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxCopyTensorPatch((vx_tensor)parameters[6], 4, nullptr, nullptr, stride_output_1, indices_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

	return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeTopK(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	data = new TopKLayerLocalData;
	memset(data, 0, sizeof(*data));

	vx_size input_dims_0[4];
	vx_size count_input_dims_0 = input_dims_0[0]*input_dims_0[1]*input_dims_0[2]*input_dims_0[3];

	ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims_0, sizeof(input_dims_0)));

	data->x_tensor_buffer = (float*)malloc(count_input_dims_0*sizeof(float));

	ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

	return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeTopK(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
	if(data)
	{
		delete data;
	}

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status publishTopKLayer(vx_context context)
{
	vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.topk_layer", VX_KERNEL_TOPK_LAYER_AMD, processTopKLayer, 7, validate, initializeTopK, uninitializeTopK);
	ERROR_CHECK_OBJECT(kernel);

	//set kernel parameters.
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));    
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

	//finalize and release kernel object.
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxTopKLayer(vx_graph graph, vx_tensor x_tensor, vx_tensor k_tensor, vx_int32 axis, vx_int32 largest, vx_int32 sorted, 
											vx_tensor values, vx_tensor indices)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_scalar s_axis = vxCreateScalarWithSize(context, VX_TYPE_INT32, &axis, sizeof(axis));
		vx_scalar s_largest = vxCreateScalarWithSize(context, VX_TYPE_INT32, &largest, sizeof(largest));
		vx_scalar s_sorted = vxCreateScalarWithSize(context, VX_TYPE_INT32, &sorted, sizeof(sorted));
		if (vxGetStatus((vx_reference)s_axis) == VX_SUCCESS &&
				vxGetStatus((vx_reference)s_largest) == VX_SUCCESS &&
				vxGetStatus((vx_reference)s_sorted) == VX_SUCCESS)
		{
			vx_reference params[] = {
				(vx_reference)x_tensor,
				(vx_reference)k_tensor,
				(vx_reference)s_axis,
				(vx_reference)s_largest,
				(vx_reference)s_sorted,
				(vx_reference)values,
				(vx_reference)indices,
			};
			node = createNode(graph, VX_KERNEL_TOPK_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
			vxReleaseScalar(&s_axis);
			vxReleaseScalar(&s_largest);
			vxReleaseScalar(&s_sorted);
		}
	}
	return node;
}
