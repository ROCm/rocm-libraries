// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../utils/Helpers.hpp"

namespace v8_sample
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

struct Summary
{
    int ran = 0;
    int skipped = 0;
};

inline Summary& summary()
{
    static Summary s_summary;
    return s_summary;
}

class CudnnHandle
{
public:
    CudnnHandle()
    {
        const cudnnStatus_t status = cudnnCreate(&_handle);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("cudnnCreate failed: ")
                                     + cudnnGetErrorString(status));
        }
    }

    ~CudnnHandle()
    {
        static_cast<void>(cudnnDestroy(_handle));
    }

    CudnnHandle(const CudnnHandle&) = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;
    CudnnHandle(CudnnHandle&&) = delete;
    CudnnHandle& operator=(CudnnHandle&&) = delete;

    cudnnHandle_t get() const
    {
        return _handle;
    }

private:
    cudnnHandle_t _handle = nullptr;
};

template <typename T>
constexpr cudnnDataType_t cudnnTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return CUDNN_DATA_FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return CUDNN_DATA_HALF;
    }
    else
    {
        static_assert(std::is_same_v<T, bfloat16>, "cudnnTypeOf: unsupported type");
        return CUDNN_DATA_BFLOAT16;
    }
}

constexpr int64_t K_ALIGNMENT = 16;

inline fe::Tensor makeTensor(int64_t uid,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides,
                             cudnnDataType_t type,
                             bool isVirtual = false)
{
    return fe::TensorBuilder()
        .setDataType(type)
        .setDim(static_cast<int64_t>(dims.size()), dims.data())
        .setStride(static_cast<int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(K_ALIGNMENT)
        .setVirtual(isVirtual)
        .build();
}

inline fe::ConvDesc makeConvDesc(const std::vector<int64_t>& pad,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& dilation)
{
    const auto spatialDims = static_cast<int64_t>(pad.size());
    return fe::ConvDescBuilder()
        .setComputeType(CUDNN_DATA_FLOAT)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setSpatialDimCount(spatialDims)
        .setPrePadding(spatialDims, pad.data())
        .setPostPadding(spatialDims, pad.data())
        .setSpatialStride(spatialDims, stride.data())
        .setDilation(spatialDims, dilation.data())
        .build();
}

inline std::optional<fe::ExecutionPlan> firstPlan(cudnnHandle_t handle, fe::OperationGraph& graph)
{
    fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                          .setOperationGraph(graph)
                                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                          .build();
    const int64_t count = heuristics.getEngineConfigCount();
    if(count == 0)
    {
        return std::nullopt;
    }
    auto& configs = heuristics.getEngineConfig(count);
    return fe::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(configs[0], graph.getTag())
        .build();
}

inline void execute(cudnnHandle_t handle,
                    fe::ExecutionPlan& plan,
                    const std::vector<int64_t>& uids,
                    std::vector<void*> ptrs,
                    void* workspace)
{
    const fe::VariantPack pack
        = fe::VariantPackBuilder()
              .setWorkspacePointer(workspace)
              .setDataPointers(static_cast<int64_t>(ptrs.size()), ptrs.data())
              .setUids(static_cast<int64_t>(uids.size()), uids.data())
              .build();
    const cudnnStatus_t status
        = cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc());
    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("cudnnBackendExecute failed: ")
                                 + cudnnGetErrorString(status));
    }
    HIP_CHECK(hipDeviceSynchronize());
}

inline bool skipVariant(const std::string& what)
{
    std::cout << "Skipping variant: " << what
              << ": no engine supports this graph on this device.\n";
    ++summary().skipped;
    return true;
}

inline void ranVariant()
{
    ++summary().ran;
}

inline int finish(bool allPassed)
{
    const Summary& result = summary();
    std::cout << "RESULT ran=" << result.ran << " skipped=" << result.skipped << '\n';
    if(!allPassed)
    {
        return 1;
    }
    if(result.ran == 0)
    {
        std::cout << "SKIPPED: no applicable engine for any variant\n";
    }
    return 0;
}

template <typename F>
int guardedMain(F&& body)
{
    try
    {
        return body();
    }
    // getCudnnStatus() is non-const, so the exception is caught by non-const reference.
    catch(fe::cudnnException& e)
    {
        std::cerr << "cuDNN shim error (" << cudnnGetErrorString(e.getCudnnStatus())
                  << "): " << e.what() << '\n';
        return 1;
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

} // namespace v8_sample
