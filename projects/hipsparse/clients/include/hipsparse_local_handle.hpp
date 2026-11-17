/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef HIPSPARSE_LOCAL_HANDLE_HPP
#define HIPSPARSE_LOCAL_HANDLE_HPP

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include "hipsparse_arguments.hpp"
#include "utility.hpp"

/*! \brief  local handle which is automatically created and destroyed  */
class hipsparseLocalHandle_t
{
    hipsparseHandle_t handle{};

public:
    hipsparseLocalHandle_t()
        : graph_testing(false)
    {
        const hipsparseStatus_t status = hipsparseCreate(&this->handle);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseCreate", status);
        }
    }
    hipsparseLocalHandle_t(const Arguments& arg)
        : graph_testing(arg.graph_test)
    {
        const hipsparseStatus_t status = hipsparseCreate(&this->handle);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error("hipsparseCreate", status);
        }
    }
    ~hipsparseLocalHandle_t()
    {
        // An exception from the test can unwind past END_GRAPH_CAPTURE(). Finish the
        // capture here so it does not stay active for later tests. Do not throw.
        release_graph_capture();

        const hipsparseStatus_t status = hipsparseDestroy(this->handle);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            log_hipsparse_error("hipsparseDestroy", status);
        }
    }

    hipsparseLocalHandle_t(const hipsparseLocalHandle_t&) = delete;
    hipsparseLocalHandle_t& operator=(const hipsparseLocalHandle_t&) = delete;

    // Allow hipsparseLocalHandle_t to be used anywhere hipsparseHandle_t is expected
    operator hipsparseHandle_t&()
    {
        return this->handle;
    }
    operator const hipsparseHandle_t&() const
    {
        return this->handle;
    }

    void hipsparseStreamBeginCapture()
    {
// As of ROCm 7.1, the HIP graph APIs require nvcc unnecessarily as they are gated by
// __CUDACC__ in the nvidia_hip_runtime_api.h header. Since we can compile hipSPARSE
// with g++/clang, these graph APIs cannot be called from hipSPARSE.
#if(!defined(CUDART_VERSION))
        if(!(this->graph_testing))
        {
            return;
        }

        if(this->capture_started)
        {
            throw std::runtime_error(
                "hipsparseStreamBeginCapture called while a capture is already active");
        }

        throw_on_hip_error("hipStreamCreate", hipStreamCreate(&this->graph_stream));
        this->graph_stream_created = true;
        throw_on_hipsparse_error("hipsparseGetStream",
                                 hipsparseGetStream(*this, &this->old_stream));
        throw_on_hipsparse_error("hipsparseSetStream",
                                 hipsparseSetStream(*this, this->graph_stream));
        this->handle_on_graph_stream = true;

        // BEGIN GRAPH CAPTURE
        throw_on_hip_error("hipStreamBeginCapture",
                           hipStreamBeginCapture(this->graph_stream, hipStreamCaptureModeGlobal));
        this->capture_started = true;
#endif
    }

    void hipsparseStreamEndCapture(int runs = 1)
    {
        (void)runs;

// As of ROCm 7.1, the HIP graph APIs require nvcc unnecessarily as they are gated by
// __CUDACC__ in the nvidia_hip_runtime_api.h header. Since we can compile hipSPARSE
// with g++/clang, these graph APIs cannot be called from hipSPARSE.
#if(!defined(CUDART_VERSION))
        if(!(this->graph_testing))
        {
            return;
        }

        if(!this->capture_started)
        {
            throw std::runtime_error("hipsparseStreamEndCapture called without an active capture");
        }

        // END GRAPH CAPTURE
        throw_on_hip_error("hipStreamEndCapture",
                           hipStreamEndCapture(this->graph_stream, &this->graph));
        this->capture_started = false;

        throw_on_hip_error("hipGraphInstantiate",
                           hipGraphInstantiate(&this->instance, this->graph, nullptr, nullptr, 0));
        throw_on_hip_error("hipGraphDestroy", hipGraphDestroy(this->graph));
        this->graph = nullptr;

        throw_on_hip_error("hipGraphLaunch", hipGraphLaunch(this->instance, this->graph_stream));
        throw_on_hip_error("hipStreamSynchronize", hipStreamSynchronize(this->graph_stream));
        throw_on_hip_error("hipGraphExecDestroy", hipGraphExecDestroy(this->instance));
        this->instance = nullptr;

        throw_on_hipsparse_error("hipsparseSetStream", hipsparseSetStream(*this, this->old_stream));
        this->handle_on_graph_stream = false;
        throw_on_hip_error("hipStreamDestroy", hipStreamDestroy(this->graph_stream));
        this->graph_stream         = nullptr;
        this->graph_stream_created = false;
#endif
    }

    hipStream_t get_stream()
    {
        hipStream_t stream{};
        throw_on_hipsparse_error("hipsparseGetStream", hipsparseGetStream(*this, &stream));
        return stream;
    }

private:
    static void throw_on_hip_error(const char* call, hipError_t status)
    {
        if(status != hipSuccess)
        {
            throw_hip_error(call, status);
        }
    }

    static void throw_on_hipsparse_error(const char* call, hipsparseStatus_t status)
    {
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw_hipsparse_error(call, status);
        }
    }

    // Drop any graph left by a failed begin or end capture, restore the previous
    // stream, and destroy the capture stream. hipStream_t{} is the default stream,
    // so the capture stream is destroyed only when this object created it.
    void release_graph_capture()
    {
// As of ROCm 7.1, the HIP graph APIs require nvcc unnecessarily as they are gated by
// __CUDACC__ in the nvidia_hip_runtime_api.h header. Since we can compile hipSPARSE
// with g++/clang, these graph APIs cannot be called from hipSPARSE.
#if(!defined(CUDART_VERSION))
        if(this->instance != nullptr)
        {
            const hipError_t status = hipGraphExecDestroy(this->instance);
            if(status != hipSuccess)
            {
                log_hip_error("hipGraphExecDestroy", status);
            }
            this->instance = nullptr;
        }

        if(this->graph != nullptr)
        {
            const hipError_t status = hipGraphDestroy(this->graph);
            if(status != hipSuccess)
            {
                log_hip_error("hipGraphDestroy", status);
            }
            this->graph = nullptr;
        }

        if(this->capture_started)
        {
            hipGraph_t       graph{};
            const hipError_t status = hipStreamEndCapture(this->graph_stream, &graph);
            if(status != hipSuccess)
            {
                log_hip_error("hipStreamEndCapture", status);
            }
            if(graph != nullptr)
            {
                const hipError_t status = hipGraphDestroy(graph);
                if(status != hipSuccess)
                {
                    log_hip_error("hipGraphDestroy", status);
                }
            }
            this->capture_started = false;
        }

        if(this->handle_on_graph_stream)
        {
            const hipsparseStatus_t status = hipsparseSetStream(*this, this->old_stream);
            if(status != HIPSPARSE_STATUS_SUCCESS)
            {
                log_hipsparse_error("hipsparseSetStream", status);
            }
            this->handle_on_graph_stream = false;
        }

        if(this->graph_stream_created)
        {
            const hipError_t status = hipStreamDestroy(this->graph_stream);
            if(status != hipSuccess)
            {
                log_hip_error("hipStreamDestroy", status);
            }
            this->graph_stream         = nullptr;
            this->graph_stream_created = false;
        }
#endif
    }

    hipStream_t graph_stream{};
    hipStream_t old_stream{};
// As of ROCm 7.1, the HIP graph APIs require nvcc unnecessarily as they are gated by
// __CUDACC__ in the nvidia_hip_runtime_api.h header. Since we can compile hipSPARSE
// with g++/clang, these graph APIs cannot be called from hipSPARSE.
#if(!defined(CUDART_VERSION))
    hipGraph_t     graph{};
    hipGraphExec_t instance{};
#endif
    bool capture_started{};
    bool graph_stream_created{};
    bool handle_on_graph_stream{};
    bool graph_testing{};
};

// Exact match for the wrapper, so this beats the implicit conversion to
// hipsparseHandle_t. A leftover hipsparseDestroy(wrapper) is a compile error
// instead of destroying the descriptor twice.
hipsparseStatus_t hipsparseDestroy(const hipsparseLocalHandle_t&) = delete;

#endif // HIPSPARSE_LOCAL_HANDLE_HPP