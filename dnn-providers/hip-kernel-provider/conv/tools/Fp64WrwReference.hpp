// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hip_kernel_provider::conv_validation
{
// Independent logical K,C,T,R,S reference geometry, not the production K3 ABI.
// NOLINTBEGIN(modernize-avoid-c-arrays)
struct ReferenceGeometry
{
    int64_t n, c, k, groups;
    int64_t input[3], output[3], filter[3];
    int64_t stride[3], dilation[3], padding[3];
    int64_t xs[5], ys[5];
    int64_t weights;
    int flip;
};
// NOLINTEND(modernize-avoid-c-arrays)

inline constexpr const char* REFERENCE_SOURCE = R"hip(
struct ReferenceGeometry
{
    long long n, c, k, groups;
    long long input[3], output[3], filter[3];
    long long stride[3], dilation[3], padding[3];
    long long xs[5], ys[5];
    long long weights;
    int flip;
};
extern "C" __global__ void fp64_wrw_reference(
    const float* x, const float* dy, double* result, ReferenceGeometry a)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= a.weights) return;
    long long rest = i;
    long long tap[3];
    for(int j = 2; j >= 0; --j)
    {
        tap[j] = rest % a.filter[j];
        rest /= a.filter[j];
        if(a.flip) tap[j] = a.filter[j] - 1 - tap[j];
    }
    const long long cg = a.c / a.groups, kg = a.k / a.groups;
    const long long c = rest % cg, k = rest / cg, group = k / kg;
    double sum = 0.0, correction = 0.0;
    for(long long n = 0; n < a.n; ++n)
    for(long long d = 0; d < a.output[0]; ++d)
    {
        const long long id = d * a.stride[0] + tap[0] * a.dilation[0] - a.padding[0];
        if(id < 0 || id >= a.input[0]) continue;
        for(long long h = 0; h < a.output[1]; ++h)
        {
            const long long ih = h * a.stride[1] + tap[1] * a.dilation[1] - a.padding[1];
            if(ih < 0 || ih >= a.input[1]) continue;
            for(long long w = 0; w < a.output[2]; ++w)
            {
                const long long iw = w * a.stride[2] + tap[2] * a.dilation[2] - a.padding[2];
                if(iw < 0 || iw >= a.input[2]) continue;
                const double product = static_cast<double>(x[n * a.xs[0]
                    + (group * cg + c) * a.xs[1] + id * a.xs[2] + ih * a.xs[3] + iw * a.xs[4]])
                    * static_cast<double>(dy[n * a.ys[0] + k * a.ys[1]
                        + d * a.ys[2] + h * a.ys[3] + w * a.ys[4]]);
                const double next = sum + product;
                const double absoluteSum = sum < 0.0 ? -sum : sum;
                const double absoluteProduct = product < 0.0 ? -product : product;
                correction += absoluteSum >= absoluteProduct
                    ? (sum - next) + product : (product - next) + sum;
                sum = next;
            }
        }
    }
    result[i] = sum + correction;
}
)hip";

inline void checkHip(hipError_t error, const char* operation)
{
    if(error != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
    }
}

class Fp64WrwReference
{
public:
    Fp64WrwReference()
    {
        int device = 0;
        hipDeviceProp_t properties{};
        checkHip(hipGetDevice(&device), "hipGetDevice");
        checkHip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
        hiprtcProgram program = nullptr;
        const auto created = hiprtcCreateProgram(
            &program, REFERENCE_SOURCE, "fp64_wrw_reference.cpp", 0, nullptr, nullptr);
        if(created != HIPRTC_SUCCESS)
        {
            throw std::runtime_error(hiprtcGetErrorString(created));
        }
        const std::string architecture
            = std::string("--gpu-architecture=") + properties.gcnArchName;
        std::array<const char*, 3> options = {architecture.c_str(), "-std=c++17", "-O3"};
        const auto compiled = hiprtcCompileProgram(program, 3, options.data());
        if(compiled != HIPRTC_SUCCESS)
        {
            size_t size = 0;
            hiprtcGetProgramLogSize(program, &size);
            std::string log(size, '\0');
            hiprtcGetProgramLog(program, log.data());
            hiprtcDestroyProgram(&program);
            throw std::runtime_error("FP64 reference compilation: " + log);
        }
        size_t size = 0;
        const auto sized = hiprtcGetCodeSize(program, &size);
        if(sized != HIPRTC_SUCCESS)
        {
            hiprtcDestroyProgram(&program);
            throw std::runtime_error(hiprtcGetErrorString(sized));
        }
        std::vector<char> binary(size);
        const auto copied = hiprtcGetCode(program, binary.data());
        hiprtcDestroyProgram(&program);
        if(copied != HIPRTC_SUCCESS)
        {
            throw std::runtime_error(hiprtcGetErrorString(copied));
        }
        checkHip(hipModuleLoadData(&_module, binary.data()), "hipModuleLoadData");
        const auto loaded = hipModuleGetFunction(&_function, _module, "fp64_wrw_reference");
        if(loaded != hipSuccess)
        {
            const auto cleanup = hipModuleUnload(_module);
            if(cleanup != hipSuccess)
            {
                std::cerr << "FP64 reference cleanup: " << hipGetErrorString(cleanup) << '\n';
            }
            _module = nullptr;
            checkHip(loaded, "hipModuleGetFunction");
        }
    }
    Fp64WrwReference(const Fp64WrwReference&) = delete;
    Fp64WrwReference& operator=(const Fp64WrwReference&) = delete;
    ~Fp64WrwReference()
    {
        if(_module != nullptr)
        {
            const auto cleanup = hipModuleUnload(_module);
            if(cleanup != hipSuccess)
            {
                std::cerr << "FP64 reference cleanup: " << hipGetErrorString(cleanup) << '\n';
            }
        }
    }
    std::vector<double> execute(const float* x,
                                const float* dy,
                                ReferenceGeometry geometry,
                                hipStream_t stream) const
    {
        if(geometry.weights <= 0
           || static_cast<uint64_t>(geometry.weights) > SIZE_MAX / sizeof(double)
           || (geometry.weights + 255) / 256 > UINT32_MAX)
        {
            throw std::runtime_error("FP64 reference output/grid exceeds supported bounds");
        }
        const auto count = static_cast<size_t>(geometry.weights);
        const hipdnn_data_sdk::utilities::Workspace<> deviceResult(count * sizeof(double));
        void* result = deviceResult.get();
        std::array<void*, 4> arguments = {static_cast<void*>(&x),
                                          static_cast<void*>(&dy),
                                          static_cast<void*>(&result),
                                          static_cast<void*>(&geometry)};
        checkHip(hipModuleLaunchKernel(_function,
                                       static_cast<unsigned>((count + 255) / 256),
                                       1,
                                       1,
                                       256,
                                       1,
                                       1,
                                       0,
                                       stream,
                                       arguments.data(),
                                       nullptr),
                 "fp64_wrw_reference launch");
        checkHip(hipStreamSynchronize(stream), "FP64 reference synchronization");
        std::vector<double> host(count);
        checkHip(hipMemcpy(host.data(), result, count * sizeof(double), hipMemcpyDeviceToHost),
                 "FP64 reference readback");
        return host;
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};
}
