/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "vx_amd_migraphx.h"
#include "kernels.h"
#include <sstream>

struct migraphXLocalData {
    migraphx::program prog;
    migraphx::program_parameters prog_params;
};

//! \brief The kernel execution.
static vx_status VX_CALLBACK amd_migraphx_node_kernel(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    migraphXLocalData *data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data != NULL) {
        data->prog.eval(data->prog_params);
    }

    return VX_SUCCESS;
}

//! \brief The kernel initializer.
static vx_status VX_CALLBACK amd_migraphx_node_initialize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    migraphXLocalData *data = new migraphXLocalData;
    unsigned char *input_mem = NULL;
    unsigned char *output_mem = NULL;
    char path[VX_MAX_STRING_BUFFER_SIZE_AMD];
    vx_bool fp16q = false;
    vx_bool int8q = false;
    vx_size in_num_dims, out_num_dims;
    vx_size input_dims[4], output_dims[4];

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[0], path, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HIP, &input_mem, sizeof(input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &in_num_dims, sizeof(in_num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HIP, &output_mem, sizeof(output_mem)));
    if (parameters[3]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &fp16q, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }
    if (parameters[4]) {
        ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &int8q, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    }

    std::string fs = path;
    std::string ext = "";
    std::size_t found = fs.find_last_of(".");
    if (found != std::string::npos) {
        ext = fs.substr(found + 1);
    }

    if (ext.compare("onnx") == 0) {
        migraphx::onnx_options onnx_opts;

        //to get the name of the input param to set batch size
        data->prog = parse_onnx(path, onnx_opts);
        auto param_shapes = data->prog.get_parameter_shapes();
        auto input = param_shapes.names().back();

        //set the name and batch size dimensions for parsing
        std::string param_name, system_out;
        param_name =  std::string(input);

        std::vector<std::size_t> input_dims_vector;
        input_dims_vector.assign(input_dims, input_dims + in_num_dims);
        onnx_opts.set_input_parameter_shape(param_name, input_dims_vector);
        onnx_opts.set_default_dim_value((unsigned int)input_dims[0]);  //set batch size
        
        //parse the onnx file
        data->prog = parse_onnx(path, onnx_opts);
        migraphx::target targ = migraphx::target("gpu");
        if (fp16q) {
            migraphx::quantize_fp16(data->prog);
        } else if (int8q) {
            migraphx::quantize_int8(data->prog, targ, migraphx::quantize_int8_options());
        }
        migraphx::compile_options comp_opts;
        comp_opts.set_fast_math();
        data->prog.compile(targ, comp_opts);
    } else if (ext.compare("mxr") == 0) {
        migraphx::file_options options;
        options.set_file_format("msgpack");
        data->prog = migraphx::load(path, options);
    } else {
        migraphx::file_options options;
        options.set_file_format("json");
        data->prog = migraphx::load(path, options);
    }

    auto param_shapes = data->prog.get_parameter_shapes();
    auto input = param_shapes.names().back();
    auto output = param_shapes.names().front();
    std::vector<size_t> inputDims = param_shapes[input].lengths();
    std::vector<size_t> outputDims = param_shapes[output].lengths();

    if (in_num_dims != inputDims.size()) {
        delete data;
        return ERRMSG(VX_ERROR_INVALID_VALUE, "the input dimension is %zu (should be %zu)\n", in_num_dims, inputDims.size());
    }

    std::stringstream tensorDims;
    std::stringstream expectedTensorDims;
    std::stringstream expectedTensorDimsInv;
    bool isDimWrong = false;
    for (auto i = 0; i < inputDims.size(); i++) {
        tensorDims << input_dims[i];
        expectedTensorDims << inputDims[i];
        if ((input_dims[i] != inputDims[i]) && (input_dims[i] != inputDims[in_num_dims - 1 - i])) {
            isDimWrong = true;
            break;
        }
    }
    tensorDims.str("");
    expectedTensorDims.str("");
    expectedTensorDimsInv.str("");
    if (isDimWrong) {
        tensorDims << " ";
        expectedTensorDims << " ";
        expectedTensorDimsInv << " ";
        for (auto i = 0; i < inputDims.size(); i++) {
            tensorDims << input_dims[i] << " ";
            expectedTensorDims << inputDims[i] << " ";
            expectedTensorDimsInv << inputDims[in_num_dims - 1 - i] << " ";
        }
        delete data;
        return ERRMSG(VX_ERROR_INVALID_VALUE, "the input tensor dimension passed to the node is [%s] which is worng. It must be either [%s] or [%s]. \n",
            tensorDims.str().c_str(), expectedTensorDims.str().c_str(), expectedTensorDimsInv.str().c_str());
    }

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    if (out_num_dims != outputDims.size()) {
        delete data;
        return ERRMSG(VX_ERROR_INVALID_VALUE, "the output dimension is %zu (should be %zu)\n", out_num_dims, outputDims.size());
    }

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    tensorDims.str("");
    expectedTensorDims.str("");
    expectedTensorDimsInv.str("");
    isDimWrong = false;
    for (auto i = 0; i < outputDims.size(); i++) {
        if ((output_dims[i] != outputDims[i]) && (output_dims[i] != outputDims[out_num_dims - 1 - i])) {
            isDimWrong = true;
            break;
        }
    }

    if (isDimWrong) {
        tensorDims << " ";
        expectedTensorDims << " ";
        expectedTensorDimsInv << " ";
        for (auto i = 0; i < outputDims.size(); i++) {
            tensorDims << output_dims[i] << " ";
            expectedTensorDims << outputDims[i] << " ";
            expectedTensorDimsInv << outputDims[out_num_dims - 1 - i] << " ";
        }
        delete data;
        return ERRMSG(VX_ERROR_INVALID_VALUE, "the output tensor dimension passed to the node is [%s] which is wrong. It must be either [%s] or [%s]. \n",
            tensorDims.str().c_str(), expectedTensorDims.str().c_str(), expectedTensorDimsInv.str().c_str());
    }

    data->prog_params.add(input, migraphx::argument(param_shapes[input], input_mem));
    data->prog_params.add(output, migraphx::argument(param_shapes[output], output_mem));

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

//! \brief The kernel deinitializer.
static vx_status VX_CALLBACK amd_migraphx_node_deinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    migraphXLocalData *data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        delete data;
    }

    return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK amd_migraphx_node_validate(vx_node node, const vx_reference parameters[], vx_uint32 num,
    vx_meta_format metas[]) {
    vx_enum type, out_type;
    vx_size in_num_dims, out_num_dims;
    vx_size output_dims[4];
    char path[VX_MAX_STRING_BUFFER_SIZE_AMD];
    std::string fs;
    std::string ext = "";

    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[0], path, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    fs = path;
    if (fs.compare("") == 0) {
        return ERRMSG(VX_ERROR_INVALID_VALUE, "the input path is empty %d (please pass a valid path to a onnx file)\n", 0);
    }

    std::size_t found = fs.find_last_of(".");
    if (found != std::string::npos) {
        ext = fs.substr(found + 1);
    }

    if ((ext.compare("onnx") != 0) && ext.compare("mxr") && ext.compare("json")) {
        return ERRMSG(VX_ERROR_INVALID_FORMAT, "the file extension for input file \
        is = .%s (only .onnx, .mxr. ,and .json files are supported!)\n", ext.c_str());
    }

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &in_num_dims, sizeof(in_num_dims)));
    if (in_num_dims > 4) {
        return VX_ERROR_INVALID_TYPE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16) && (type != VX_TYPE_INT8)) {
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    if (out_num_dims > 4) {
        return VX_ERROR_INVALID_TYPE;
    }
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16) && (type != VX_TYPE_INT8)) {
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &out_num_dims, sizeof(out_num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, &output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status amd_vx_migraphx_node_publish(vx_context context) {
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.amd_migraphx_node", AMDOVX_KERNEL_AMD_MIGRAPHX,
                            amd_migraphx_node_kernel, 5, amd_migraphx_node_validate,
                            amd_migraphx_node_initialize, amd_migraphx_node_deinitialize);
    ERROR_CHECK_OBJECT(kernel);

    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE,
        &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // input file (e.g., onnx)
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // input tensor
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED)); // output tensor
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL)); // fp16
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL)); // int8

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL amdMIGraphXnode(vx_graph graph, const vx_char *path, vx_tensor input, vx_tensor output,
 vx_bool fp16q, vx_bool int8q) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_path = vxCreateScalar(context, VX_TYPE_STRING_AMD, path);
        vx_scalar s_fp16q = vxCreateScalar(context, VX_TYPE_BOOL, &fp16q);
        vx_scalar s_int8q = vxCreateScalar(context, VX_TYPE_BOOL, &int8q);
        if (vxGetStatus((vx_reference)s_path) == VX_SUCCESS &&
            vxGetStatus((vx_reference)s_fp16q) == VX_SUCCESS &&
            vxGetStatus((vx_reference)s_int8q) == VX_SUCCESS) {
            vx_reference params[] = {
                (vx_reference)s_path,
                (vx_reference)input,
                (vx_reference)output,
                (vx_reference)s_fp16q,
                (vx_reference)s_int8q,
            };
            node = createMIGraphXNode(graph, "com.amd.amd_migraphx_node", params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&s_path);
            vxReleaseScalar(&s_fp16q);
            vxReleaseScalar(&s_int8q);
        }
    }
    return node;
}