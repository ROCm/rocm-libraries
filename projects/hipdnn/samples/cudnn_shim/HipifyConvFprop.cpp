// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   samples/cpp/convolution/fprop.cpp
//     Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: frontend/THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

// A cuDNN FE convolution-fprop program after hipify. Each line changed from the
// cuDNN original is preceded by "// cuDNN: <original line>"; everything else is unchanged
// cuDNN FE source. Needs a GPU; without one it prints "SKIPPED: No GPU devices
// available." and exits 0.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// cuDNN: #include <cuda_runtime.h>
#include <hip/hip_runtime.h>

// cuDNN: #include <cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
// cuDNN: (none; added for the shim)
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

namespace
{
// cuDNN: void checkRuntime(cudaError_t status, const char* step)
void checkRuntime(hipError_t status, const char* step)
{
    // cuDNN: if(status != cudaSuccess)
    if(status != hipSuccess)
    {
        // cuDNN: throw std::runtime_error(std::string(step) + ": " + cudaGetErrorString(status));
        throw std::runtime_error(std::string(step) + ": " + hipGetErrorString(status));
    }
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        if(bytes > 0)
        {
            // cuDNN: checkRuntime(cudaMalloc(&_ptr, bytes), "alloc");
            checkRuntime(hipMalloc(&_ptr, bytes), "alloc");
        }
    }

    ~DeviceBuffer()
    {
        // cuDNN: static_cast<void>(cudaFree(_ptr));
        static_cast<void>(hipFree(_ptr));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _ptr;
    }

    void upload(const std::vector<float>& host) const
    {
        const size_t bytes = host.size() * sizeof(float);
        // cuDNN: checkRuntime(cudaMemcpy(_ptr, host.data(), bytes, cudaMemcpyHostToDevice), "H2D");
        checkRuntime(hipMemcpy(_ptr, host.data(), bytes, hipMemcpyHostToDevice), "H2D");
    }

    void download(std::vector<float>& host) const
    {
        const size_t bytes = host.size() * sizeof(float);
        // cuDNN: checkRuntime(cudaMemcpy(host.data(), _ptr, bytes, cudaMemcpyDeviceToHost), "D2H");
        checkRuntime(hipMemcpy(host.data(), _ptr, bytes, hipMemcpyDeviceToHost), "D2H");
    }

private:
    void* _ptr = nullptr;
};

int fail(const char* step, const cudnn_frontend::error_t& error)
{
    std::cerr << step << " failed: " << error.get_message() << '\n';
    return 1;
}

// Small integers that depend on every index, so channel, filter, or layout mix-ups change the
// output; |y| <= c * 3 * 2 keeps every partial sum exact in FP32.
float inputValue(int64_t in, int64_t ic, int64_t ih, int64_t iw)
{
    return static_cast<float>(((in + (2 * ic) + (3 * ih) + (5 * iw)) % 7) - 3);
}

float filterValue(int64_t ik, int64_t ic)
{
    return static_cast<float>((((3 * ik) + ic) % 5) - 2);
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        static_cast<void>(argc);
        static_cast<void>(argv);

        int deviceCount = 0;
        // cuDNN: if(cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        {
            std::cout << "SKIPPED: No GPU devices available.\n";
            return 0;
        }

        namespace fe = cudnn_frontend;

        const int64_t n = 2;
        const int64_t c = 8;
        const int64_t h = 8;
        const int64_t w = 8;
        const int64_t k = 4;
        const int64_t r = 1;
        const int64_t s = 1;

        const auto handlePtr = create_cudnn_handle();
        if(handlePtr == nullptr)
        {
            std::cerr << "create_cudnn_handle failed\n";
            return 1;
        }
        auto handle = *handlePtr;

        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

        auto x = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("image")
                                  .set_dim({n, c, h, w})
                                  .set_stride({c * h * w, h * w, w, 1}));

        auto filter = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("filter")
                                       .set_dim({k, c, r, s})
                                       .set_stride({c * r * s, r * s, s, 1}));

        auto y = graph.conv_fprop(
            x,
            filter,
            fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation(
                {1, 1}));
        y->set_output(true);

        if(auto error = graph.validate(); error.is_bad())
        {
            return fail("validate", error);
        }
        if(auto error = graph.build_operation_graph(handle); error.is_bad())
        {
            return fail("build_operation_graph", error);
        }
        // hipDNN accepts mode A but logs a warning and uses its fallback engine selection.
        if(auto error = graph.create_execution_plans({fe::HeurMode_t::A}); error.is_bad())
        {
            return fail("create_execution_plans", error);
        }
        if(auto error = graph.check_support(); error.is_bad())
        {
            return fail("check_support", error);
        }
        if(auto error = graph.build_plans(); error.is_bad())
        {
            return fail("build_plans", error);
        }

        int64_t workspaceSize = 0;
        if(auto error = graph.get_workspace_size(workspaceSize); error.is_bad())
        {
            return fail("get_workspace_size", error);
        }

        std::vector<float> xHost(static_cast<size_t>(n * c * h * w));
        for(size_t i = 0; i < xHost.size(); ++i)
        {
            const auto idx = static_cast<int64_t>(i);
            xHost[i] = inputValue(idx / (c * h * w), (idx / (h * w)) % c, (idx / w) % h, idx % w);
        }
        std::vector<float> filterHost(static_cast<size_t>(k * c * r * s));
        for(size_t i = 0; i < filterHost.size(); ++i)
        {
            const auto idx = static_cast<int64_t>(i);
            filterHost[i] = filterValue(idx / c, idx % c);
        }
        std::vector<float> yHost(static_cast<size_t>(n * k * h * w), 0.0f);

        const DeviceBuffer xDevice(xHost.size() * sizeof(float));
        const DeviceBuffer filterDevice(filterHost.size() * sizeof(float));
        const DeviceBuffer yDevice(yHost.size() * sizeof(float));
        const DeviceBuffer workspace(static_cast<size_t>(workspaceSize));

        xDevice.upload(xHost);
        filterDevice.upload(filterHost);

        std::unordered_map<int64_t, void*> variantPack = {{x->get_uid(), xDevice.get()},
                                                          {filter->get_uid(), filterDevice.get()},
                                                          {y->get_uid(), yDevice.get()}};

        if(auto error = graph.execute(handle, variantPack, workspace.get()); error.is_bad())
        {
            return fail("execute", error);
        }

        // cuDNN: checkRuntime(cudaDeviceSynchronize(), "synchronize");
        checkRuntime(hipDeviceSynchronize(), "synchronize");
        yDevice.download(yHost);

        // Index the output through the strides the graph inferred for y.
        const auto& yStride = y->get_stride();
        for(int64_t idx = 0; idx < n * k * h * w; ++idx)
        {
            const int64_t in = idx / (k * h * w);
            const int64_t ik = (idx / (h * w)) % k;
            const int64_t ih = (idx / w) % h;
            const int64_t iw = idx % w;

            float expected = 0.0f;
            for(int64_t ic = 0; ic < c; ++ic)
            {
                expected += inputValue(in, ic, ih, iw) * filterValue(ik, ic);
            }

            const auto offset = static_cast<size_t>((in * yStride[0]) + (ik * yStride[1])
                                                    + (ih * yStride[2]) + (iw * yStride[3]));
            if(yHost[offset] != expected)
            {
                std::cerr << "Mismatch at y[" << in << "," << ik << "," << ih << "," << iw
                          << "]: got " << yHost[offset] << ", expected " << expected << '\n';
                return 1;
            }
        }

        std::cout << "cuDNN-shim hipified conv_fprop: all " << yHost.size() << " outputs match\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << '\n';
        return 1;
    }
    catch(...)
    {
        std::cerr << "Unhandled unknown exception\n";
        return 1;
    }
}
