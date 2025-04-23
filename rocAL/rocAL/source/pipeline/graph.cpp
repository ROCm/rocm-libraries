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

#include <iostream>
#include <vx_ext_amd.h>

#include "pipeline/graph.h"
#include "pipeline/commons.h"

AgoTargetAffinityInfo
get_ago_affinity_info(
    RocalAffinity rocal_affinity,
    int cpu_id,
    int gpu_id) {
    AgoTargetAffinityInfo affinity;
    switch (rocal_affinity) {
        case RocalAffinity::GPU:
            affinity.device_type = AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = (gpu_id >= 0 && gpu_id <= 9) ? gpu_id : 0;
            affinity.group = affinity.reserved = 0;
            break;
        case RocalAffinity::CPU:
            affinity.device_type = AGO_TARGET_AFFINITY_CPU;
            affinity.device_info = (cpu_id >= 0 && cpu_id <= 9) ? cpu_id : 0;
            affinity.group = affinity.reserved = 0;
            break;
        default:
            throw std::invalid_argument("Unsupported affinity");
    }
    return affinity;
}

Graph::Graph(vx_context context, RocalAffinity affinity, int cpu_id, size_t cpu_num_threads, int gpu_id) : _context(context),
                                                                                                           _graph(nullptr),
                                                                                                           _affinity(affinity),
                                                                                                           _gpu_id(gpu_id),
                                                                                                           _cpu_id(cpu_id) {
    try {
        vx_status status;
        auto vx_affinity = get_ago_affinity_info(_affinity, _cpu_id, _gpu_id);
        vx_uint32 _cpu_num_threads = cpu_num_threads;

        _graph = vxCreateGraph(_context);

        if ((status = vxGetStatus((vx_reference)_graph)) != VX_SUCCESS)
            THROW("vxCreateGraph failed " + TOSTR(status))

        // Setting attribute to run on CPU or GPU
        if ((status = vxSetGraphAttribute(_graph,
                                          VX_GRAPH_ATTRIBUTE_AMD_AFFINITY,
                                          &vx_affinity,
                                          sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetGraphAttribute failed " + TOSTR(status))

        // Setting attribute to run on CPU or GPU
        if ((status = vxSetGraphAttribute(_graph,
                                          VX_GRAPH_ATTRIBUTE_AMD_CPU_NUM_THREADS,
                                          &_cpu_num_threads,
                                          sizeof(_cpu_num_threads))) != VX_SUCCESS)
            THROW("vxSetGraphAttribute failed " + TOSTR(status))

    } catch (const std::exception& e) {
        release();
        throw;
    }
}

Graph::Status
Graph::verify() {
    vx_status status;
    if ((status = vxVerifyGraph(_graph)) != VX_SUCCESS)
        THROW("vxVerifyGraph failed " + TOSTR(status))

    return Status::OK;
}

Graph::Status
Graph::process() {
    vx_status status;
    if ((status = vxProcessGraph(_graph)) != VX_SUCCESS)
        THROW("ERROR: vxProcessGraph failed " + TOSTR(status))

    return Status::OK;
}

Graph::Status
Graph::release() {
    vx_status status = VX_SUCCESS;

    if (_graph && (status = vxReleaseGraph(&_graph)) != VX_SUCCESS)
        LOG("Failed to call vxReleaseGraph " + TOSTR(status))

    return Status::OK;
}
