// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// required test headers
#include "../common_test_header.hpp"

// required rocprim headers
#include "../rocprim/test_seed.hpp"
#include "../rocprim/test_utils.hpp"

// Basic test functions that can be used to check if HIP API functions
// work inside graphs.
// To test an API call, you can:
// - call it inside the graph stream capture zone of testStreamCapture()
// - add the corresponding type of graph node in the testManualConstruction() function
//
// HIP API functions that do not currently work:
// - hipMallocAsync

// Simple test kernel that increments a value using a single thread.
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void increment(int* data)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (!gid)
        data[gid]++;
}

// Another simple kernel that can be used to test atomics inside a graph.
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void atomicIncrement(int* data)
{
       atomicAdd(data, 1);
}

void testStreamCapture()
{
    // The default stream does not support HipGraph stream capture, so create our own.
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate a counter variable on the device and set it to 0.
    // We will use this to record the number of times the graph is launched.
    int* d_data = nullptr;
    int h_data = 0;

    // Create a new graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Note: currently, calls to hipMallocAsync do not work inside the stream capture section
    HIP_CHECK(hipMallocAsync(&d_data, sizeof(int), stream));

    // ** Begin stream capture **
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Transfer the host value
    HIP_CHECK(hipMemcpyAsync(d_data, &h_data, sizeof(int), hipMemcpyHostToDevice, stream));

    // Launch kernel
    hipLaunchKernelGGL(increment, dim3(1), dim3(1), 0, stream, d_data);

    // Transfer result back to host
    HIP_CHECK(hipMemcpyAsync(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost, stream));

    // ** End stream capture **
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Instantiate the graph
    hipGraphExec_t instance;
    HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // Launch it
    const int num_launches = 3;
    for (int i = 0; i < num_launches; i++)
    {
        HIP_CHECK(hipGraphLaunch(instance, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Counter value should match the number of graph launches
    ASSERT_EQ(h_data, num_launches);

    // Clean up
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(instance));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipStreamDestroy(stream));
}

void testManualConstruction()
{
    // The default stream does not support HipGraph stream capture, so create our own.
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate a counter variable on the device and set it to 0.
    // We will use this to record the number of times the graph is launched.
    int* d_data = nullptr;
    int h_data = 0;
    HIP_CHECK(hipMallocAsync(&d_data, sizeof(int), stream));

    // Create a new graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Transfer the counter value from host to device using a graph memcpy node
    hipGraphNode_t hostToDevMemcpyNode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&hostToDevMemcpyNode, graph, nullptr, 0, d_data, &h_data, sizeof(int), hipMemcpyHostToDevice));

    // Launch the kernel
    hipGraphNode_t kernelNode;
    void* kernelArgs[1] = {(void*) &d_data};
    hipKernelNodeParams kernelNodeParams{};
    kernelNodeParams.func = (void*) increment;
    kernelNodeParams.gridDim = dim3(1);
    kernelNodeParams.blockDim = dim3(1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void**) (kernelArgs);
    kernelNodeParams.extra = nullptr;

    // Add the kernel node to the graph, listing the memcpyNode as a dependency
    HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, &hostToDevMemcpyNode, 1, &kernelNodeParams));

    // Transfer result back to the device
    hipGraphNode_t devToHostMemcpyNode;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&devToHostMemcpyNode, graph, &kernelNode, 1, &h_data, d_data, sizeof(int), hipMemcpyDeviceToHost));

    // Instantiate the graph
    hipGraphExec_t instance;
    HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // Launch it
    const int num_launches = 3;
    for (int i = 0; i < num_launches; i++)
    {
        HIP_CHECK(hipGraphLaunch(instance, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // The counter value should match the number of times we launched the graph
    ASSERT_EQ(h_data, num_launches);

    // Clean up
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(instance));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipStreamDestroy(stream));
}

void testStreamCaptureWithAtomics()
{
    // The default stream does not support HipGraph stream capture, so create our own.
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate a counter variable on the device.
    // We will have each thread atomically increment it.
    int* d_data = nullptr;
    int h_data = 0;
       const int num_blocks = 2;
       const int num_threads = 33;

    // Create a new graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Note: currently, calls to hipMallocAsync do not work inside the stream capture section
    HIP_CHECK(hipMallocAsync(&d_data, sizeof(int), stream));

    // ** Begin stream capture **
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Transfer the host value
    HIP_CHECK(hipMemcpyAsync(d_data, &h_data, sizeof(int), hipMemcpyHostToDevice, stream));

    // Launch kernel
    hipLaunchKernelGGL(atomicIncrement, dim3(num_blocks), dim3(num_threads), 0, stream, d_data);

    // Transfer result back to host
    HIP_CHECK(hipMemcpyAsync(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost, stream));

    // ** End stream capture **
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Instantiate the graph
    hipGraphExec_t instance;
    HIP_CHECK(hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // Launch it
    const int num_launches = 3;
    for (int i = 0; i < num_launches; i++)
    {
        HIP_CHECK(hipGraphLaunch(instance, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Counter value should match the number of graph launches multiplied by
    // the number of threads that were launched.
    ASSERT_EQ(h_data, num_launches * num_blocks * num_threads);

    // Clean up
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(instance));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(TestHipGraphBasic, CaptureFromStream)
{
    testStreamCapture();
}

TEST(TestHipGraphBasic, ManualConstruction)
{
    testManualConstruction();
}

TEST(TestHipGraphBasic, StreamCaptureAtomics)
{
    testStreamCaptureWithAtomics();
}
