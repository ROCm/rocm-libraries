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
    class GraphHelper{
        private:
            hipGraph_t graph;
            hipGraphExec_t graph_instance;
        public:

            inline void startStreamCapture(hipStream_t & stream)
            {
                HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            }

            inline void endStreamCapture(hipStream_t & stream)
            {
                HIP_CHECK(hipStreamEndCapture(stream, &graph));
            }

            inline void createAndLaunchGraph(hipStream_t & stream, const bool launchGraph=true, const bool sync=true)
            {
                // End current capture
                endStreamCapture(stream);

                // Create the graph instance
                HIP_CHECK(hipGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));

                // Optionally launch the graph
                if (launchGraph)
                    HIP_CHECK(hipGraphLaunch(graph_instance, stream));

                // Optionally synchronize the stream when we're done
                if (sync)
                    HIP_CHECK(hipStreamSynchronize(stream));
            }

            inline void cleanupGraphHelper()
            {
                HIP_CHECK(hipGraphDestroy(this->graph));
                HIP_CHECK(hipGraphExecDestroy(this->graph_instance));
            }

            inline void resetGraphHelper(hipStream_t& stream, const bool beginCapture=true)
            {
                // Destroy the old graph and instance
                cleanupGraphHelper();

                // Re-start capture
                if(beginCapture)
                    startStreamCapture(stream);
            }

            inline void launchGraphHelper(hipStream_t& stream,const bool sync=false)
            {
                HIP_CHECK(hipGraphLaunch(this->graph_instance, stream));

                // Optionally sync after the launch
                if (sync)
                    HIP_CHECK(hipStreamSynchronize(stream));
            }
    };
} // end namespace test_utils

#undef HIP_CHECK

#endif //HIPCUB_TEST_TEST_UTILS_HIPGRAPHS_HPP_
