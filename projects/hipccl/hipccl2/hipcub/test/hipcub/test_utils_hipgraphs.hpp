// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_TEST_TEST_UTILS_HIPGRAPHS_HPP_
#define HIPCUB_TEST_TEST_UTILS_HIPGRAPHS_HPP_

#include <hip/hip_runtime.h>

#include "test_utils.hpp"

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

// Helper functions for testing with hipGraph stream capture.
// Note: graphs will not work on the default stream.
namespace test_utils
{

inline hipGraph_t createGraphHelper(hipStream_t& stream, const bool beginCapture = true)
{
    // Create a new graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Optionally begin stream capture
    if(beginCapture)
    {
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    }

    return graph;
}

inline void cleanupGraphHelper(hipGraph_t& graph, hipGraphExec_t& instance)
{
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(instance));
}

inline hipGraphExec_t endCaptureGraphHelper(hipGraph_t&  graph,
                                            hipStream_t& stream,
                                            const bool   launchGraph = false,
                                            const bool   sync        = false)
{
    // End the capture
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Instantiate the graph
    hipGraphExec_t instance;
    HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // Optionally launch the graph
    if(launchGraph)
        HIP_CHECK(hipGraphLaunch(instance, stream));

    // Optionally synchronize the stream when we're done
    if(sync)
    {
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    return instance;
}

} // end namespace test_utils

#undef HIP_CHECK

#endif //HIPCUB_TEST_TEST_UTILS_HIPGRAPHS_HPP_
