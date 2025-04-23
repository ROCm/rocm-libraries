/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "pipeline/commons.h"
#include "pipeline/context.h"
#include "rocal_api.h"
#if ENABLE_OPENCL
#include "CL/cl.h"
#endif

RocalStatus ROCAL_API_CALL
rocalToTensor(RocalContext p_context, void* out_ptr, RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
              float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
              bool reverse_channels, RocalOutputMemType output_mem_type, int max_roi_height, int max_roi_width) {
    auto context = static_cast<Context*>(p_context);
    try {
        if (tensor_format != ROCAL_NHWC && tensor_format != ROCAL_NCHW)
            THROW("Supported only for NHWC and NCHW tensor layout")

        if (tensor_output_type != ROCAL_FP32 && tensor_output_type != ROCAL_FP16)
            THROW("Supported only for FP32 and FP16 tensor data types")

        auto tensor_layout = (tensor_format == ROCAL_NHWC) ? RocalTensorlayout::NHWC : RocalTensorlayout::NCHW;
        auto tensor_output_data_type = (tensor_output_type == ROCAL_FP32) ? RocalTensorDataType::FP32 : RocalTensorDataType::FP16;
        context->master_graph->to_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                                         offset0, offset1, offset2, reverse_channels, tensor_output_data_type, output_mem_type, max_roi_height, max_roi_width);
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalCopyToOutput(
    RocalContext p_context,
    unsigned char* out_ptr,
    size_t out_size) {
    auto context = static_cast<Context*>(p_context);
    try {
        context->master_graph->copy_output(out_ptr, out_size);
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

void
    ROCAL_API_CALL
    rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalTensor>& output_tensors) {
    if (!p_context)
        THROW("Invalid rocal context passed to rocalSetOutputs")
    auto context = static_cast<Context*>(p_context);
    for (auto& it : output_tensors) {
        auto tensor = static_cast<Tensor*>(it);
        context->master_graph->set_output(tensor);
    }
}

RocalStatus ROCAL_API_CALL
rocalExternalSourceFeedInput(
    RocalContext p_context,
    const std::vector<std::string>& input_images_names,
    bool is_labels,
    const std::vector<unsigned char*>& input_buffer,
    const std::vector<ROIxywh>& roi_xywh,
    unsigned int max_width,
    unsigned int max_height,
    unsigned int channels,
    RocalExternalSourceMode mode,
    RocalTensorLayout layout,
    bool eos) {
    auto context = static_cast<Context*>(p_context);
    try {
        ExternalSourceFileMode external_file_mode = static_cast<ExternalSourceFileMode>(mode);
        RocalTensorlayout format = static_cast<RocalTensorlayout>(layout);
        context->master_graph->feed_external_input(input_images_names, is_labels, input_buffer,
                                                   roi_xywh, max_width, max_height, channels,
                                                   external_file_mode, format, eos);
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalTensorList ROCAL_API_CALL
rocalGetOutputTensors(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        return context->master_graph->get_output_tensors();
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return nullptr;
    }
    return nullptr;
}
