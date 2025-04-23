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

#ifndef __KERNELS_H__
#define __KERNELS_H__

//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
// STITCH_API_ENTRY - export API symbols
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif

//////////////////////////////////////////////////////////////////////
// common header files
#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>
#include <miopen/miopen.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#if __APPLE__
#include <opencl.h>
#else
#if ENABLE_OPENCL
#include <CL/cl.h>
#elif ENABLE_HIP
#include "../nn_hip/nn_hip_host_decls.h"
#endif
#endif
#if _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <stdlib.h>
#include <ctype.h>
#endif

// Visual Profiler (enabled by setting PROFILER_MODE=1 in profiler.h)
#include "profiler.h"

//////////////////////////////////////////////////////////////////////
//! \brief The macro for error checking from OpenVX status.
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS){ vxAddLogEntry(NULL, status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error checking from OpenVX object.
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
//! \brief The macro for error message and return error code
#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status

#ifndef ERROR_CHECK_MIOPEN_STATUS
#define ERROR_CHECK_MIOPEN_STATUS(call) if(call) { \
    std::cerr << "ERROR: fatal error occured at " __FILE__ << "#" << __LINE__ << std::endl; \
    exit(1); \
    }
#endif

// Debug Print Dims : disabled unless enabled explicitly by setting DEBUG_PRINT_DIMS=1
#ifndef ENABLE_DEBUG_PRINT_DIMS
#define ENABLE_DEBUG_PRINT_DIMS 0
#endif

// Debug Dump Layer outputs : disabled unless enabled explicitly by setting ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS=1
#ifndef ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS
#define ENABLE_DEBUG_DUMP_NN_LAYER_BUFFERS 0
#endif
//////////////////////////////////////////////////////////////////////
//! user kernels
enum nn_additional_library
{
    NN_EXTENSION_LIBRARY = 1,
};
enum user_kernel_e
{
    VX_KERNEL_BATCH_NORMALISATION_LAYER_AMD  = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x001,
    VX_KERNEL_ARGMAX_LAYER_AMD               = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x002,
    VX_KERNEL_CONVERT_IMAGE_TO_TENSOR_AMD    = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x003,
    VX_KERNEL_CONVERT_TENSOR_TO_IMAGE_AMD    = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x004,
    VX_KERNEL_CONCAT_LAYER_AMD               = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x006,
    VX_KERNEL_SLICE_LAYER_AMD                = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x007,
    VX_KERNEL_SCALE_LAYER_AMD                = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x008,
    VX_KERNEL_UPSAMPLE_NEAREST_LAYER_AMD     = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x009,
    VX_KERNEL_RESHAPE_LAYER                  = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00a,
    VX_KERNEL_PERMUTE_LAYER_AMD              = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00b,
    VX_KERNEL_PRIOR_BOX_LAYER_AMD            = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00c,
    VX_KERNEL_CROP_LAYER_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00d,
    VX_KERNEL_CROP_AND_RESIZE_LAYER_AMD      = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00e,
    VX_KERNEL_DETECTION_OUTPUT_LAYER_AMD     = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x00f,
    VX_KERNEL_TENSOR_MIN_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x010,
    VX_KERNEL_TENSOR_MAX_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x011,
    VX_KERNEL_TENSOR_EXP_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x012,
    VX_KERNEL_TENSOR_LOG_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x013,
    VX_KERNEL_CAST_LAYER_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x014,
    VX_KERNEL_NMS_LAYER_AMD                  = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x015,
    VX_KERNEL_GATHER_LAYER_AMD               = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x016,
    VX_KERNEL_TOPK_LAYER_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x017,
    VX_KERNEL_REDUCE_MIN_LAYER_AMD           = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x018,
    VX_KERNEL_TILE_LAYER_AMD                 = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x019,
    VX_KERNEL_TENSOR_COMPARE_AMD            = VX_KERNEL_BASE(VX_ID_AMD, NN_EXTENSION_LIBRARY) + 0x01a,
};

//////////////////////////////////////////////////////////////////////
//! \brief Common data shared across all nodes in a graph
struct NeuralNetworkCommonHandle {
    int count;
    miopenHandle_t  miopen_handle;
#if ENABLE_OPENCL
    cl_command_queue cmdq;
#elif ENABLE_HIP
    hipStream_t cmdq;
#endif
    bool exhaustiveSearch;
};

//////////////////////////////////////////////////////////////////////
//! \brief The utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num);
vx_reference getNodeParameterByIndex(vx_node node, vx_uint32 index);
vx_status createGraphHandle(vx_node node, NeuralNetworkCommonHandle ** pHandle);
vx_status releaseGraphHandle(vx_node node, NeuralNetworkCommonHandle * handle);
int getEnvironmentVariable(const char* name, char * value, size_t valueSize);
void nn_layer_test_dumpBuffer(const char * fileNameFormat, vx_tensor tensor);

//////////////////////////////////////////////////////////////////////
//! \brief The kernel publish functions
vx_status publishConvolutionLayer(vx_context context);
vx_status publishFullyConnectedLayer(vx_context context);
vx_status publishPoolingLayer(vx_context context);
vx_status publishSoftmaxLayer(vx_context context);
vx_status publishNormalizationLayer(vx_context context);
vx_status publishLocalResponseNormalizationLayer(vx_context context);
vx_status publishActivationLayer(vx_context context);
vx_status publishROIPoolingLayer(vx_context context);
vx_status publishDeconvolutionLayer(vx_context context);
vx_status publishBatchNormalizationLayer(vx_context context);
vx_status publishArgmaxLayer(vx_context context);
vx_status publishConcatLayer(vx_context context);
vx_status publishSliceLayer(vx_context context);
vx_status publishImageToTensorConvert(vx_context context);
vx_status publishTensorToImageConvert(vx_context context);
vx_status publishTensorAdd(vx_context context);
vx_status publishTensorSubtraction(vx_context context);
vx_status publishTensorMultiply(vx_context context);
vx_status publishScaleLayer(vx_context context);
vx_status publishUpsampleNearest(vx_context context);
vx_status publishTensorTableLookup(vx_context context);
vx_status publishTensorMatrixMultiply(vx_context context);
vx_status publishReshapeLayer(vx_context context);
vx_status publishPermuteLayer(vx_context context);
vx_status publishPriorBoxLayer(vx_context context);
vx_status publishCropLayer(vx_context context);
vx_status publishCropAndResizeLayer(vx_context context);
vx_status publishTensorMin(vx_context context);
vx_status publishTensorMax(vx_context context);
vx_status publishCastLayer(vx_context context);
vx_status publishTensorExp(vx_context context);
vx_status publishTensorLog(vx_context context);
vx_status publishDetectionOutputLayer(vx_context context);
vx_status publishNMSLayer(vx_context context);
vx_status publishGatherLayer(vx_context context);
vx_status publishTopKLayer(vx_context context);
vx_status publishReduceMinLayer(vx_context context);
vx_status publishTileLayer(vx_context context);
vx_status publishTensorCompare(vx_context context);

//////////////////////////////////////////////////////////////////////
//! \brief The module entry point for publishing/unpublishing kernels
SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);
SHARED_PUBLIC vx_status VX_API_CALL vxUnpublishKernels(vx_context context);

#endif //__KERNELS_H__
