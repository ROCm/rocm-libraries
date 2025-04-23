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

#include "rocal_api.h"

#include <exception>
#include <string>

#include "pipeline/commons.h"
#include "pipeline/context.h"

RocalStatus ROCAL_API_CALL
rocalRelease(RocalContext p_context) {
    // Deleting context is required to call the destructor of all the member objects
    auto context = static_cast<Context*>(p_context);
    delete context;
    return ROCAL_OK;
}

RocalContext ROCAL_API_CALL
rocalCreate(
    size_t batch_size,
    RocalProcessMode affinity,
    int gpu_id,
    size_t cpu_thread_count,
    size_t prefetch_queue_depth,
    RocalTensorOutputType output_tensor_data_type) {
    RocalContext context = nullptr;
    try {
        auto translate_process_mode = [](RocalProcessMode process_mode) {
            switch (process_mode) {
                case ROCAL_PROCESS_GPU:
                    return RocalAffinity::GPU;
                case ROCAL_PROCESS_CPU:
                    return RocalAffinity::CPU;
                default:
                    THROW("Unkown Rocal data type")
            }
        };
        auto translate_output_data_type = [](RocalTensorOutputType data_type) {
            switch (data_type) {
                case ROCAL_FP32:
                    return RocalTensorDataType::FP32;
                case ROCAL_FP16:
                    return RocalTensorDataType::FP16;
                case ROCAL_UINT8:
                    return RocalTensorDataType::UINT8;
                default:
                    THROW("Unkown Rocal data type")
            }
        };
        if (gpu_id < 0)
            ERR(STR("Negative GPU device ID passed to context creation. Setting GPU device ID to 0"));
        context = new Context(batch_size, translate_process_mode(affinity), std::max(gpu_id, 0), cpu_thread_count, prefetch_queue_depth, translate_output_data_type(output_tensor_data_type));
        // Reset seed in case it's being randomized during context creation
    } catch (const std::exception& e) {
        ERR(STR("Failed to init the Rocal context, ") + STR(e.what()))
    }
    return context;
}

RocalStatus ROCAL_API_CALL
rocalRun(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        auto ret = context->master_graph->run();
        if (ret != MasterGraph::Status::OK)
            return ROCAL_RUNTIME_ERROR;
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalVerify(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        context->master_graph->build();
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
