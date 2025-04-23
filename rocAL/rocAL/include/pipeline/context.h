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

#pragma once

#include "pipeline/commons.h"
#include "pipeline/graph.h"
#include "pipeline/master_graph.h"

struct Context {
    explicit Context(size_t batch_size, RocalAffinity affinity, int gpu_id, size_t cpu_thread_count, size_t prefetch_queue_depth, RocalTensorDataType output_tensor_type) : affinity(affinity),
                                                                                                                                                                            _user_batch_size(batch_size) {
        LOG("Processing on " + STR(((affinity == RocalAffinity::CPU) ? " CPU" : " GPU")))
        master_graph = std::make_shared<MasterGraph>(batch_size, affinity, cpu_thread_count, gpu_id, prefetch_queue_depth, output_tensor_type);
    }
    ~Context() {
        clear_errors();
    };
    std::shared_ptr<MasterGraph> master_graph;

    RocalAffinity affinity;
    bool no_error() { return error.empty(); }
    const char* error_msg() { return error.c_str(); }
    void capture_error(const std::string& err_msg) { error = err_msg; }
    Timing timing() {
        return master_graph->timing();
    }
    size_t user_batch_size() { return _user_batch_size; }

   private:
    void clear_errors() { error = ""; }
    std::string error;
    size_t _user_batch_size;
};
