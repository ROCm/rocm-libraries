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

typedef struct ReduceMinLocalData {
    float *input_data;
    vx_int32 keepdims;
    int *axes;
}ReduceMinLocalData;

ReduceMinLocalData *data_reduce = NULL;

static vx_status VX_CALLBACK validate(vx_node node, const vx_reference *parameters, vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dims.
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4],  output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    vx_size axes_cap = 0;
    vx_size itemsize = 0;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_CAPACITY, &axes_cap, sizeof(axes_cap)));
    if(axes_cap < 0 || axes_cap > 4) return VX_ERROR_INVALID_DIMENSION;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMSIZE, &itemsize, sizeof(itemsize)));
    if(itemsize != sizeof(int)) return VX_ERROR_INVALID_TYPE;

    vx_int32 keepdims;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if(type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &keepdims, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(keepdims != 0 && keepdims != 1) return VX_ERROR_INVALID_VALUE;


    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    // output tensor configuration
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processReduceMin(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    //get tensor dimensions
    vx_size input_dims[4], output_dims[4];
    vx_size num_of_dims;

    vx_enum usage = VX_READ_ONLY;
    vx_status status;
    vx_map_id map_id;
    vx_size stride[4];

    //query and copy inputs
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    
    float * ptr_input;
    vx_size count_input_dims = input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3];
    ERROR_CHECK_STATUS(vxMapTensorPatch((vx_tensor)parameters[0], num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr_input, usage, VX_MEMORY_TYPE_HOST));
    memcpy(data_reduce->input_data, ptr_input, (count_input_dims*sizeof(float)));
    ERROR_CHECK_STATUS(vxUnmapTensorPatch((vx_tensor)parameters[0], map_id));

    int* ptr_axes;
    vx_size axes_numitems;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_NUMITEMS, &axes_numitems, sizeof(axes_numitems)));

    vx_size stride_axes[axes_numitems];
    ERROR_CHECK_STATUS(vxMapArrayRange((vx_array)parameters[1], 0, axes_numitems, &map_id, stride_axes, (void **)&ptr_axes, usage, VX_MEMORY_TYPE_HOST, 0));
    memcpy(data_reduce->axes, ptr_axes, (axes_numitems*sizeof(int)));
    ERROR_CHECK_STATUS(vxUnmapArrayRange((vx_array)parameters[1], map_id));

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data_reduce->keepdims, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    std::vector<float> reduced;
    //WRITE ALGORITHM
    if(axes_numitems == 1)
    {
        if(data_reduce->axes[0] < 0)
        {
            int count = 0, dims_idx = 0;
            //get real dimensions. Ignoring the appended 1s
            while(dims_idx <= 3 && input_dims[dims_idx] == 1)
            {
                count++;
                dims_idx++;
            }

            count = 3 - count;
            data_reduce->axes[0] = data_reduce->axes[0] + count;
        }
    
        if(data_reduce->axes[0] == 4) //corresponds to axes=None
        {
            reduced.push_back(*std::min_element(data_reduce->input_data, data_reduce->input_data+count_input_dims));
        }
        else if(data_reduce->axes[0] == 0 || data_reduce->axes[0] == -4) //corresponds to axis = 0/-4
        {
            std::vector<float> temp;
            for(int j = 0; j < (input_dims[1]*input_dims[2]*input_dims[3]); j++)
            {
                temp.clear();
                for(int i = 0; i < count_input_dims; i+= (input_dims[1]*input_dims[2]*input_dims[3]))
                {
                    temp.push_back(data_reduce->input_data[j+i]);
                }
                reduced.push_back(*std::min_element(temp.begin(), temp.end()));
            }
        }
        else if(data_reduce->axes[0] == 1 || data_reduce->axes[0] == -3) //corresponds to axis = 1/-3
        {   
            ///vector to of required elements
            std::vector<float> temp;
            for(int k = 0; k < count_input_dims; k+=(input_dims[1]*input_dims[2]*input_dims[3]))
            {
                for(int j = 0; j < input_dims[2]*input_dims[3]; j++)
                {
                    temp.clear();
                    for(int i = 0; i < input_dims[1]; i++)
                    {
                        int idx = k + j + i*input_dims[2]*input_dims[3];
                        temp.push_back(data_reduce->input_data[idx]);
                    }
                    reduced.push_back(*std::min_element(temp.begin(), temp.end()));
                }
            }
        }
        else if(data_reduce->axes[0] == 2 || data_reduce->axes[0] == 3 || data_reduce->axes[0] == -2 || data_reduce->axes[0] == -1) //corresponds to axis = 2/-2; 3\-1
        {   
            //vector to of required elements
            std::vector<float> temp;
            for(int j = 0; j < count_input_dims; j += (input_dims[2]*input_dims[3]))
            {
                temp.clear();
                for(int i = 0; i < (input_dims[2]*input_dims[3]); i++)
                {
                    temp.push_back(data_reduce->input_data[j+i]); 
                }
                reduced.push_back(*std::min_element(temp.begin(), temp.end()));
            }
        }
    }
    else
        return VX_ERROR_NOT_SUPPORTED;

    float *reduced_ptr = &reduced[0];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    vx_size stride_output[4] = {sizeof(float), output_dims[0]*sizeof(float), output_dims[0]*output_dims[1]*sizeof(float), output_dims[0]*output_dims[1]*output_dims[2]*sizeof(float)};
    ERROR_CHECK_STATUS(vxCopyTensorPatch((vx_tensor)parameters[3], 4, nullptr, nullptr, stride_output, reduced_ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeReduceMin(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    data_reduce = new ReduceMinLocalData;
    memset(data_reduce, 0, sizeof(*data_reduce));

    //allocate memory for input data
    vx_size input_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    vx_size count_input_dims = input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3];
    data_reduce->input_data = (float*)malloc(count_input_dims*sizeof(float));

    //allocate memeory for axes 
    vx_size axes_numitems;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_NUMITEMS, &axes_numitems, sizeof(axes_numitems)));
    data_reduce->axes = (int*)malloc(axes_numitems*sizeof(int));
    
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data_reduce, sizeof(data_reduce)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeReduceMin(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data_reduce, sizeof(data_reduce)));
    if(data_reduce)
    {
        free(data_reduce->input_data);
        free(data_reduce->axes);
        delete data_reduce;
    }

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status publishReduceMinLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.reduce_min_layer", VX_KERNEL_REDUCE_MIN_LAYER_AMD, processReduceMin, 4, 
                                        validate, initializeReduceMin, uninitializeReduceMin);
    ERROR_CHECK_OBJECT(kernel);

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxReduceMinLayer(vx_graph graph, vx_tensor data, vx_array axes, vx_int32 keepdims, vx_tensor reduced)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_keepdims = vxCreateScalarWithSize(context, VX_TYPE_INT32, &keepdims, sizeof(keepdims));
        if(vxGetStatus((vx_reference)s_keepdims) == VX_SUCCESS)
        {
            vx_reference params[] = {
                (vx_reference)data,
                (vx_reference)axes,
                (vx_reference)s_keepdims,
                (vx_reference)reduced,
            };
            node = createNode(graph, VX_KERNEL_REDUCE_MIN_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
            vxReleaseScalar(&s_keepdims);
        }
    }
    return node;
}
