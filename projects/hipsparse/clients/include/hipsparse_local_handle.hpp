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
        : capture_started(false)
        , graph_testing(false)
    {
        const hipsparseStatus_t status = hipsparseCreate(&this->handle);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw(status);
        }
    }
    hipsparseLocalHandle_t(const Arguments& arg)
        : capture_started(false)
        , graph_testing(arg.graph_test)
    {
        const hipsparseStatus_t status = hipsparseCreate(&this->handle);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            throw(status);
        }
    }
    ~hipsparseLocalHandle_t()
    {
        hipsparseDestroy(this->handle);
    }

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

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, false);
#endif

        CHECK_HIP_ERROR(hipStreamCreate(&this->graph_stream));
        CHECK_HIPSPARSE_ERROR(hipsparseGetStream(*this, &this->old_stream));
        CHECK_HIPSPARSE_ERROR(hipsparseSetStream(*this, this->graph_stream));

        // BEGIN GRAPH CAPTURE
        CHECK_HIP_ERROR(hipStreamBeginCapture(this->graph_stream, hipStreamCaptureModeGlobal));

        capture_started = true;
#endif
    }

    void hipsparseStreamEndCapture(int runs = 1)
    {
// As of ROCm 7.1, the HIP graph APIs require nvcc unnecessarily as they are gated by
// __CUDACC__ in the nvidia_hip_runtime_api.h header. Since we can compile hipSPARSE
// with g++/clang, these graph APIs cannot be called from hipSPARSE.
#if(!defined(CUDART_VERSION))
        if(!(this->graph_testing))
        {
            return;
        }

#ifdef GOOGLE_TEST
        ASSERT_EQ(capture_started, true);
#endif

        hipGraph_t     graph;
        hipGraphExec_t instance;

        // END GRAPH CAPTURE
        CHECK_HIP_ERROR(hipStreamEndCapture(this->graph_stream, &graph));
        CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        CHECK_HIP_ERROR(hipGraphDestroy(graph));
        CHECK_HIP_ERROR(hipGraphLaunch(instance, this->graph_stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(this->graph_stream));
        CHECK_HIP_ERROR(hipGraphExecDestroy(instance));

        CHECK_HIPSPARSE_ERROR(hipsparseSetStream(*this, this->old_stream));
        CHECK_HIP_ERROR(hipStreamDestroy(this->graph_stream));
        this->graph_stream = nullptr;

        capture_started = false;
#endif
    }

    hipStream_t get_stream()
    {
        hipStream_t stream;
        hipsparseGetStream(*this, &stream);
        return stream;
    }

private:
    hipStream_t graph_stream;
    hipStream_t old_stream;
    bool        capture_started;
    bool        graph_testing;
};

#endif // HIPSPARSE_LOCAL_HANDLE_HPP