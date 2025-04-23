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

////////////////////////////////////////////////////////////////////////////
// utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        return NULL;
    }
    vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
    if(vxGetStatus((vx_reference)kernel) == VX_SUCCESS) {
        node = vxCreateGenericNode(graph, kernel);
        if (node) {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++) {
                if (params[p]) {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS) {
                        char kernelName[VX_MAX_KERNEL_NAME];
                        vxQueryKernel(kernel, VX_KERNEL_NAME, kernelName, VX_MAX_KERNEL_NAME);
                        vxAddLogEntry((vx_reference)graph, status, "createNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to create node with kernel enum %d\n", kernelEnum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to retrieve kernel enum %d\n", kernelEnum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

vx_reference getNodeParameterByIndex(vx_node node, vx_uint32 index)
{
    vx_reference ref = NULL;
    vx_parameter param = vxGetParameterByIndex(node, index);
    if(vxGetStatus((vx_reference)param) == VX_SUCCESS) {
        vxQueryParameter(param, VX_PARAMETER_REF, &ref, sizeof(ref));
        vxReleaseParameter(&param);
    }
    return ref;
}

int getEnvironmentVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
    char text[512] = { 0 };
    DWORD len = GetEnvironmentVariableA(name, text, (DWORD)sizeof(text));
    if ( len > 1) {
        value[len-1] = '\0';
        if(isdigit(value[0]) != 0)
            return atoi(value);
        else return 1;
    }
#else
    const char * text = getenv(name);
    if (text) {
        strncpy(value, text, valueSize);
        value[strlen(text)+1] = '\0';
        if(isdigit(value[0]) != 0)
            return atoi(value);
        else return 1;
    }
#endif
    return -1;
}

vx_status createGraphHandle(vx_node node, NeuralNetworkCommonHandle ** pHandle)
{
    NeuralNetworkCommonHandle * handle = NULL;
    ERROR_CHECK_STATUS(vxGetModuleHandle(node, OPENVX_KHR_NN, (void **)&handle));
    if(handle) {
        handle->count++;
    }
    else {
        handle = new NeuralNetworkCommonHandle;
        memset(handle, 0, sizeof(*handle));
        const char * searchEnvName = "NN_MIOPEN_SEARCH";
        char textBuffer[1024];
        int isEnvSet = getEnvironmentVariable(searchEnvName, textBuffer, sizeof(textBuffer));
        if (isEnvSet > 0)
            handle->exhaustiveSearch = true;

        handle->count = 1;

#if ENABLE_OPENCL
        ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &handle->cmdq, sizeof(handle->cmdq)));
#elif ENABLE_HIP
        ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &handle->cmdq, sizeof(handle->cmdq)));
#endif

        //create miopen_handle from cmdq
        ERROR_CHECK_MIOPEN_STATUS(miopenCreateWithStream(&handle->miopen_handle, handle->cmdq));

        ERROR_CHECK_STATUS(vxSetModuleHandle(node, OPENVX_KHR_NN, handle));
    }
    *pHandle = handle;
    return VX_SUCCESS;
}

vx_status releaseGraphHandle(vx_node node, NeuralNetworkCommonHandle * handle)
{
    handle->count--;
    if(handle->count == 0) {
        //TBD: release miopen_handle
        delete handle;
        ERROR_CHECK_STATUS(vxSetModuleHandle(node, OPENVX_KHR_NN, NULL));
    }
    return VX_SUCCESS;
}


void nn_layer_test_dumpBuffer(const char * fileNameFormat, vx_tensor tensor)
{
    //get dump location and file name
    char dump_location[512] = "NN_BufferDump/";
    char textBuffer[512];
    if (getEnvironmentVariable("NN_LAYER_DUMP_LOCATION", textBuffer, sizeof(textBuffer)) > 0 ) 
    {
        snprintf(dump_location, sizeof(dump_location), "%s", textBuffer);
    }
    #if _WIN32
        CreateDirectory(dump_location, NULL);
    #else
        struct stat st = {0};
        if (stat(dump_location, &st) == -1) { mkdir(dump_location, 0700); }
    #endif
    char fileName[1024];
    static int dumpBufferCount = 0; 
    dumpBufferCount++;
    snprintf(fileName, sizeof(fileName), strcat(dump_location, fileNameFormat), dumpBufferCount);
    FILE * fp = fopen(fileName, "wb");

    //map tensor to pointer
    vx_size tensor_dims[4];
    vx_status status;
    status = vxQueryTensor((vx_tensor)tensor, VX_TENSOR_DIMS, tensor_dims, sizeof(tensor_dims));
    if(status)
    {
        std::cerr << "ERROR: vxQueryTensor() failed for layer dump tensor (" << status << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    vx_map_id map_id;
    vx_size stride[4];
    float * ptr;
    vx_enum usage = VX_READ_ONLY;
    
    vx_size count_tensor = tensor_dims[0]*tensor_dims[1]*tensor_dims[2]*tensor_dims[3];
    status = vxMapTensorPatch(tensor, 4, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST);
    if(status)
    {
        std::cerr << "ERROR: vxMapTensorPatch() failed for layer dump tensor (" << status << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    //write values to file
    if(!fp) printf("Could not open file %s\n", fileName);
    else
    {
        printf("OK: Writing file %s into BufferDump folder with %lu bytes\n", fileName, count_tensor*4);
        fwrite((void **)&ptr, sizeof(float), count_tensor, fp);
    }
    fclose(fp);
    status = vxUnmapTensorPatch(tensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for layer dump tensor (" << status << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    PROFILER_INITIALIZE();
#if ENABLE_OPENCL
    // set command-queue properties to be CL_QUEUE_PROFILING_ENABLE needed by MIOpen (default)
    const char * searchEnvName = "NN_MIOPEN_CL_QUEUE_PROPERTIES";
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
#if _WIN32
    char text[64] = { 0 };
    if (GetEnvironmentVariableA(searchEnvName, text, (DWORD)sizeof(text)) > 0) {
        properties = atoi(text);
    }
#else
    const char * text = getenv(searchEnvName);
    if (text) {
        properties = atoi(text);
    }
#endif
     ERROR_CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_CL_QUEUE_PROPERTIES, &properties, sizeof(properties)));
#endif

    // register kernels
    ERROR_CHECK_STATUS(publishConvolutionLayer(context));
    ERROR_CHECK_STATUS(publishFullyConnectedLayer(context));
    ERROR_CHECK_STATUS(publishPoolingLayer(context));
    ERROR_CHECK_STATUS(publishSoftmaxLayer(context));
    ERROR_CHECK_STATUS(publishNormalizationLayer(context));
    ERROR_CHECK_STATUS(publishLocalResponseNormalizationLayer(context));
    ERROR_CHECK_STATUS(publishActivationLayer(context));
    ERROR_CHECK_STATUS(publishROIPoolingLayer(context));
    ERROR_CHECK_STATUS(publishDeconvolutionLayer(context));
    ERROR_CHECK_STATUS(publishBatchNormalizationLayer(context));
    ERROR_CHECK_STATUS(publishArgmaxLayer(context));
    ERROR_CHECK_STATUS(publishConcatLayer(context));
    ERROR_CHECK_STATUS(publishSliceLayer(context));
    ERROR_CHECK_STATUS(publishImageToTensorConvert(context));
    ERROR_CHECK_STATUS(publishTensorToImageConvert(context));
    ERROR_CHECK_STATUS(publishTensorAdd(context));
    ERROR_CHECK_STATUS(publishTensorSubtraction(context));
    ERROR_CHECK_STATUS(publishTensorMultiply(context));
    ERROR_CHECK_STATUS(publishScaleLayer(context));
    ERROR_CHECK_STATUS(publishUpsampleNearest(context));
    ERROR_CHECK_STATUS(publishTensorTableLookup(context));
    ERROR_CHECK_STATUS(publishTensorMatrixMultiply(context));
    ERROR_CHECK_STATUS(publishReshapeLayer(context));
    ERROR_CHECK_STATUS(publishPermuteLayer(context));
    ERROR_CHECK_STATUS(publishPriorBoxLayer(context));
    ERROR_CHECK_STATUS(publishCropLayer(context));
    ERROR_CHECK_STATUS(publishCropAndResizeLayer(context));
    ERROR_CHECK_STATUS(publishTensorMin(context));
    ERROR_CHECK_STATUS(publishTensorMax(context));
    ERROR_CHECK_STATUS(publishCastLayer(context));
    ERROR_CHECK_STATUS(publishDetectionOutputLayer(context));
    ERROR_CHECK_STATUS(publishTensorExp(context));
    ERROR_CHECK_STATUS(publishTensorLog(context));
    ERROR_CHECK_STATUS(publishNMSLayer(context));
    ERROR_CHECK_STATUS(publishGatherLayer(context));
    ERROR_CHECK_STATUS(publishTopKLayer(context));
    ERROR_CHECK_STATUS(publishReduceMinLayer(context));
    ERROR_CHECK_STATUS(publishTileLayer(context));
    ERROR_CHECK_STATUS(publishTensorCompare(context));

    // register drama rules
    AgoNodeMergeRule softmax_rule = {
        {
            { VX_KERNEL_SOFTMAX_LAYER, { 1, 2 | AGO_MERGE_RULE_SOLITARY_FLAG } },
            { VX_KERNEL_ARGMAX_LAYER_AMD, { 2 | AGO_MERGE_RULE_SOLITARY_FLAG, 3 } },
        },
        {
            { VX_KERNEL_ARGMAX_LAYER_AMD, { 1, 3 } },
        }
    };
    ERROR_CHECK_STATUS(vxSetContextAttribute(context, VX_CONTEXT_ATTRIBUTE_AMD_SET_MERGE_RULE, &softmax_rule, sizeof(softmax_rule)));

    return VX_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////
//! \brief The module entry point for unpublishing kernel.
SHARED_PUBLIC vx_status VX_API_CALL vxUnpublishKernels(vx_context context)
{
    PROFILER_SHUTDOWN();
    return VX_SUCCESS;
}
