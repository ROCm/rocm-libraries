/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "internal_publishKernels.h"

struct ResampleLocalData {
    vxRppHandle *handle;
    Rpp32u deviceType;
    RppPtr_t pSrc;
    RppPtr_t pDst;
    float quality;
    Rpp32s *pSrcRoi;
    RpptDescPtr pSrcDesc;
    RpptDescPtr pDstDesc;
    Rpp32f *pInRateTensor;
    Rpp32f *pOutRateTensor;
#if RPP_AUDIO
    RpptResamplingWindow window;
#endif
    size_t inputTensorDims[RPP_MAX_TENSOR_DIMS];
    size_t outputTensorDims[RPP_MAX_TENSOR_DIMS];
};

inline float sinc(float x) {
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * (1.0f / 6)) : std::sin(x) / x;
}

inline double hann(double x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

// initialization function used for filling the values in Resampling window (RpptResamplingWindow)
// using the coeffs and lobes value this function generates a LUT (look up table) which is further used in Resample audio augmentation
#if RPP_AUDIO
inline void windowed_sinc(RpptResamplingWindow &window, int32_t coeffs, int32_t lobes) {
    float scale = 2.0f * lobes / (coeffs - 1);
    float scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookupSize = window.lookup.size();
    int32_t center = (coeffs - 1) * 0.5f;
    for (int32_t i = 0; i < coeffs; i++) {
        float x = (i - center) * scale;
        float y = (i - center) * scale_envelope;
        float w = sinc(x) * hann(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1.0f / scale;
    window.pCenter = _mm_set1_ps(window.center);
    window.pScale = _mm_set1_ps(window.scale);
}
#endif

void update_destination_roi(ResampleLocalData *data, RpptROI *src_roi, RpptROI *dst_roi) {
    float scale_ratio;
    for (uint32_t i = 0; i < data->pSrcDesc->n; i++) {
        scale_ratio = (data->pInRateTensor[i] != 0) ? (data->pOutRateTensor[i] / static_cast<float>(data->pInRateTensor[i])) : 0;
        dst_roi[i].xywhROI.roiWidth = static_cast<int32_t>(std::ceil(scale_ratio * src_roi[i].xywhROI.roiWidth));
        dst_roi[i].xywhROI.roiHeight = src_roi[i].xywhROI.roiHeight;
    }
}

static vx_status VX_CALLBACK refreshResample(vx_node node, const vx_reference *parameters, vx_uint32 num, ResampleLocalData *data) {
    vx_status status = VX_SUCCESS;
    int32_t nDim = 2;  // Num dimensions for audio tensor
    STATUS_ERROR_CHECK(vxCopyArrayRange((vx_array)parameters[5], 0, data->pSrcDesc->n, sizeof(float), data->pInRateTensor, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_HOST, &data->pOutRateTensor, sizeof(data->pOutRateTensor)));
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
        void *roi_tensor_ptr_src, *roi_tensor_ptr_dst;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_HOST, &data->pSrc, sizeof(data->pSrc)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_HOST, &data->pDst, sizeof(data->pDst)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_src, sizeof(roi_tensor_ptr_src)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_HOST, &roi_tensor_ptr_dst, sizeof(roi_tensor_ptr_dst)));
        if (!data->pSrcRoi) {
            data->pSrcRoi = new Rpp32s[data->pSrcDesc->n * nDim];
        }
        RpptROI *src_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_src);
        RpptROI *dst_roi = reinterpret_cast<RpptROI *>(roi_tensor_ptr_dst);
        update_destination_roi(data, src_roi, dst_roi);
        for (uint32_t i = 0; i < data->pSrcDesc->n; i++) {
            data->pSrcRoi[i * nDim] = src_roi[i].xywhROI.roiWidth;
            data->pSrcRoi[i * nDim + 1] = src_roi[i].xywhROI.roiHeight;
        }
        return status;
    }
    return status;
}

static vx_status VX_CALLBACK validateResample(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]) {
    vx_status status = VX_SUCCESS;
    vx_enum scalar_type;

    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[6], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_FLOAT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #6 type=%d (must be size)\n", scalar_type);
    STATUS_ERROR_CHECK(vxQueryScalar((vx_scalar)parameters[7], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if (scalar_type != VX_TYPE_UINT32)
        return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: Paramter: #7 type=%d (must be size)\n", scalar_type);

    // Validate for input parameters
    size_t num_tensor_dims;
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate Resample: tensor #0 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    // Validate for output parameters
    vx_tensor output;
    vx_uint8 tensor_fixed_point_position;
    size_t tensor_dims[RPP_MAX_TENSOR_DIMS];
    vx_enum tensor_datatype;

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    if (num_tensor_dims < 3) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate Resample: tensor #1 dimensions=%lu (must be greater than or equal to 3)\n", num_tensor_dims);

    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_tensor_dims, sizeof(num_tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &tensor_dims, sizeof(tensor_dims)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &tensor_datatype, sizeof(tensor_datatype)));
    STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &tensor_fixed_point_position, sizeof(tensor_fixed_point_position)));
    return status;
}

static vx_status VX_CALLBACK processResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    RppStatus rpp_status = RPP_SUCCESS;
    vx_status return_status = VX_SUCCESS;
    ResampleLocalData *data = NULL;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    refreshResample(node, parameters, num, data);
    if (data->deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
        return VX_ERROR_NOT_IMPLEMENTED;
#endif
    }
    if (data->deviceType == AGO_TARGET_AFFINITY_CPU) {
#if RPP_AUDIO
        rpp_status = rppt_resample_host(data->pSrc, data->pSrcDesc, data->pDst, data->pDstDesc,
                                        data->pInRateTensor, data->pOutRateTensor, data->pSrcRoi, data->window, data->handle->rppHandle);
        return_status = (rpp_status == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
#else
        return_status = VX_ERROR_NOT_SUPPORTED;
#endif
    }
    return return_status;
}

static vx_status VX_CALLBACK initializeResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResampleLocalData *data = new ResampleLocalData;
    if (data) {
        memset(data, 0, sizeof(ResampleLocalData));
        vx_enum input_tensor_dtype, output_tensor_dtype;
        STATUS_ERROR_CHECK(vxReadScalarValue((vx_scalar)parameters[6], &data->quality));
        STATUS_ERROR_CHECK(vxCopyScalar((vx_scalar)parameters[7], &data->deviceType, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        // Querying for input tensor
        data->pSrcDesc = new RpptDesc;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &data->pSrcDesc->numDims, sizeof(data->pSrcDesc->numDims)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &data->inputTensorDims, sizeof(vx_size) * data->pSrcDesc->numDims));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &input_tensor_dtype, sizeof(input_tensor_dtype)));
        data->pSrcDesc->dataType = getRpptDataType(input_tensor_dtype);
        data->pSrcDesc->offsetInBytes = 0;
        fillAudioDescriptionPtrFromDims(data->pSrcDesc, data->inputTensorDims);

        // Querying for output tensor
        data->pDstDesc = new RpptDesc;
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &data->pDstDesc->numDims, sizeof(data->pDstDesc->numDims)));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &data->outputTensorDims, sizeof(vx_size) * data->pDstDesc->numDims));
        STATUS_ERROR_CHECK(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &output_tensor_dtype, sizeof(output_tensor_dtype)));
        data->pDstDesc->dataType = getRpptDataType(output_tensor_dtype);
        data->pDstDesc->offsetInBytes = 0;
        fillAudioDescriptionPtrFromDims(data->pDstDesc, data->outputTensorDims);

        data->pInRateTensor = new float[data->pSrcDesc->n];
        int32_t lobes = std::round(0.007 * data->quality * data->quality - 0.09 * data->quality + 3);
        int32_t lookupSize = lobes * 64 + 1;
#if RPP_AUDIO
        windowed_sinc(data->window, lookupSize, lobes);
#endif
        refreshResample(node, parameters, num, data);
        STATUS_ERROR_CHECK(createRPPHandle(node, &data->handle, data->pSrcDesc->n, data->deviceType));
        STATUS_ERROR_CHECK(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
        return VX_SUCCESS;
    } else {
        return VX_FAILURE;
    }
}

static vx_status VX_CALLBACK uninitializeResample(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ResampleLocalData *data;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data->pInRateTensor) delete[] data->pInRateTensor;
    if (data->pSrcRoi) delete[] data->pSrcRoi;
    if (data->pSrcDesc) delete data->pSrcDesc;
    if (data->pDstDesc) delete data->pDstDesc;
    STATUS_ERROR_CHECK(releaseRPPHandle(node, data->handle, data->deviceType));
    if (data) delete data;
    return VX_SUCCESS;
}

//! \brief The kernel target support callback.
// TODO::currently the node is setting the same affinity as context. This needs to change when we have hybrid modes in the same graph
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
                                                  vx_bool use_opencl_1_2,
                                                  vx_uint32 &supported_target_affinity) {
    vx_context context = vxGetContext((vx_reference)graph);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    else
        supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

    return VX_SUCCESS;
}

vx_status Resample_Register(vx_context context) {
    vx_status status = VX_SUCCESS;
    // Add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.rpp.Resample",
                                       VX_KERNEL_RPP_RESAMPLE,
                                       processResample,
                                       8,
                                       validateResample,
                                       initializeResample,
                                       uninitializeResample);
    ERROR_CHECK_OBJECT(kernel);
    AgoTargetAffinityInfo affinity;
    vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
#if ENABLE_HIP
    vx_bool enableBufferAccess = vx_true_e;
    if (affinity.device_type == AGO_TARGET_AFFINITY_GPU)
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_GPU_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
#else
    vx_bool enableBufferAccess = vx_false_e;
#endif
    amd_kernel_query_target_support_f query_target_support_f = query_target_support;

    if (kernel) {
        STATUS_ERROR_CHECK(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }
    if (status != VX_SUCCESS) {
    exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }
    return status;
}
