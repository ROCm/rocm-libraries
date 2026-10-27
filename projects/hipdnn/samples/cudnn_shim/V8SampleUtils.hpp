// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>

#include "../utils/Helpers.hpp"

namespace v8_sample
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace sdk = hipdnn_data_sdk::utilities;

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

struct Options
{
    Config config;
    // The run fails when fewer variants executed, so losing every engine is not a skip.
    int requireRan = 0;
    std::vector<std::string> switches;

    bool has(std::string_view name) const
    {
        return std::find(switches.begin(), switches.end(), name) != switches.end();
    }
};

// parseCommandLineArgs rejects unknown arguments, so the v8-only ones are consumed first.
inline Options
    parseOptions(int argc, char** argv, std::initializer_list<std::string_view> sampleSwitches = {})
{
    Options options;
    std::vector<char*> args;
    args.reserve(static_cast<size_t>(argc));
    for(int i = 0; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if(i > 0 && arg == "--require-ran")
        {
            if(i + 1 >= argc)
            {
                throw std::invalid_argument("--require-ran requires a value");
            }
            options.requireRan = parseInteger<int>(argv[++i], "--require-ran");
            if(options.requireRan < 0)
            {
                throw std::invalid_argument("--require-ran must not be negative");
            }
            continue;
        }
        if(i > 0
           && std::find(sampleSwitches.begin(), sampleSwitches.end(), arg) != sampleSwitches.end())
        {
            options.switches.emplace_back(arg);
            continue;
        }
        args.push_back(argv[i]);
    }
    options.config = parseCommandLineArgs(static_cast<int>(args.size()), args.data());
    if(options.config.engineId != -1 || !options.config.engineName.empty())
    {
        throw std::invalid_argument("--engine-id and --engine-name are not supported by the cuDNN "
                                    "v8 samples, which run the heuristics' first engine config");
    }
    return options;
}

inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter> createHandle()
{
    auto handle = create_cudnn_handle();
    if(!handle)
    {
        throw std::runtime_error("create_cudnn_handle failed");
    }
    return handle;
}

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
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return CUDNN_DATA_INT32;
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

inline fe::Tensor makeTensor(int64_t uid,
                             const std::vector<int64_t>& dims,
                             const TensorLayout& layout,
                             cudnnDataType_t type,
                             bool isVirtual = false)
{
    return makeTensor(uid, dims, sdk::generateStrides(dims, layout.strideOrder), type, isVirtual);
}

// A by-value scalar: its variant-pack entry is a host pointer.
inline fe::Tensor makeScalar(int64_t uid)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    return fe::TensorBuilder()
        .setDataType(CUDNN_DATA_FLOAT)
        .setDim(static_cast<int64_t>(dims.size()), dims.data())
        .setStride(static_cast<int64_t>(dims.size()), dims.data())
        .setId(uid)
        .setAlignment(K_ALIGNMENT)
        .setByValue(true)
        .build();
}

struct ConvProblem
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t k;
    int64_t r;
    int64_t s;
    std::vector<int64_t> stride;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> dilation;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

    std::vector<int64_t> xDims() const
    {
        return {n, c, h, w};
    }

    std::vector<int64_t> wDims() const
    {
        return {k, c, r, s};
    }

    // No v8 shape inference: the caller supplies y's dims.
    std::vector<int64_t> yDims() const
    {
        const int64_t outH
            = (h + prePadding[0] + postPadding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1;
        const int64_t outW
            = (w + prePadding[1] + postPadding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1;
        return {n, k, outH, outW};
    }
};

// defaults = {n, c, h, w, k}; the filter is 3x3, stride 1, padding 1, dilation 1 unless overridden.
inline ConvProblem problemFrom(const Config& config, const std::array<int64_t, 5>& defaults)
{
    const auto pick = [](const std::vector<int64_t>& values, size_t index, int64_t fallback) {
        return values.size() > index ? values[index] : fallback;
    };
    const std::vector<int64_t> padding = {pick(config.padding, 0, 1), pick(config.padding, 1, 1)};
    return ConvProblem{pick(config.dims, 0, defaults[0]),
                       pick(config.dims, 1, defaults[1]),
                       pick(config.dims, 2, defaults[2]),
                       pick(config.dims, 3, defaults[3]),
                       pick(config.filter, 0, defaults[4]),
                       pick(config.filter, 1, 3),
                       pick(config.filter, 2, 3),
                       {pick(config.stride, 0, 1), pick(config.stride, 1, 1)},
                       padding,
                       padding,
                       {pick(config.dilation, 0, 1), pick(config.dilation, 1, 1)}};
}

// Convolutions PyTorch can emit beyond symmetric cross-correlation: true convolution and
// asymmetric padding.
inline std::vector<std::pair<std::string, ConvProblem>> edgeConvProblems(const ConvProblem& base)
{
    ConvProblem trueConvolution = base;
    trueConvolution.mode = CUDNN_CONVOLUTION;
    ConvProblem asymmetric = base;
    asymmetric.prePadding = {1, 0};
    asymmetric.postPadding = {0, 1};
    return {{"CUDNN_CONVOLUTION", trueConvolution}, {"asymmetric padding", asymmetric}};
}

inline bool selectsFp32Nchw(const Config& config)
{
    return (config.dtype.empty() || config.dtype == "fp32")
           && (config.layout.empty() || config.layout == "nchw");
}

inline fe::ConvDesc makeConvDesc(const ConvProblem& problem)
{
    const auto spatialDims = static_cast<int64_t>(problem.stride.size());
    return fe::ConvDescBuilder()
        .setComputeType(CUDNN_DATA_FLOAT)
        .setMathMode(problem.mode)
        .setSpatialDimCount(spatialDims)
        .setPrePadding(spatialDims, problem.prePadding.data())
        .setPostPadding(spatialDims, problem.postPadding.data())
        .setSpatialStride(spatialDims, problem.stride.data())
        .setDilation(spatialDims, problem.dilation.data())
        .build();
}

// Every direction names its tensors x, w and y as in the forward pass, the way PyTorch builds
// them; the descriptor type decides which one is the gradient.
inline fe::Operation makeConvOp(cudnnBackendDescriptorType_t direction,
                                const ConvProblem& problem,
                                const fe::Tensor& x,
                                const fe::Tensor& w,
                                const fe::Tensor& y)
{
    return fe::OperationBuilder(direction)
        .setxDesc(x)
        .setwDesc(w)
        .setyDesc(y)
        .setcDesc(makeConvDesc(problem))
        .setAlpha(1.0f)
        .setBeta(0.0f)
        .build();
}

inline fe::Operation makePointwiseOp(cudnnPointwiseMode_t mode,
                                     cudnnDataType_t computeType,
                                     const fe::Tensor& x,
                                     const fe::Tensor* b,
                                     const fe::Tensor& y)
{
    fe::OperationBuilder builder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(fe::PointWiseDescBuilder().setMode(mode).setComputeType(computeType).build())
        .setxDesc(x)
        .setyDesc(y);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

inline fe::OperationGraph makeGraph(cudnnHandle_t handle, std::vector<const fe::Operation*> ops)
{
    return fe::OperationGraphBuilder()
        .setHandle(handle)
        .setOperationGraph(static_cast<int64_t>(ops.size()), ops.data())
        .build();
}

// The reference for CUDNN_CONVOLUTION: true convolution is cross-correlation with the filter
// flipped in every spatial dimension.
template <typename T>
sdk::Tensor<T> flipSpatial(const sdk::Tensor<T>& source)
{
    sdk::Tensor<T> flipped(source.dims(), source.strides());
    const std::vector<int64_t>& dims = source.dims();
    sdk::iterateAlongDimensions(dims, [&](const std::vector<int64_t>& index) {
        std::vector<int64_t> mirrored = index;
        for(size_t dim = 2; dim < dims.size(); ++dim)
        {
            mirrored[dim] = dims[dim] - 1 - index[dim];
        }
        flipped.setHostValue(source.getHostValue(index), mirrored);
    });
    return flipped;
}

inline fe::EngineConfigList instantConfigs(fe::OperationGraph& graph)
{
    fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                          .setOperationGraph(graph)
                                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                          .build();
    const int64_t count = heuristics.getEngineConfigCount();
    if(count == 0)
    {
        return {};
    }
    return heuristics.getEngineConfig(count);
}

inline std::optional<fe::ExecutionPlan> firstPlan(cudnnHandle_t handle, fe::OperationGraph& graph)
{
    const fe::EngineConfigList configs = instantConfigs(graph);
    if(configs.empty())
    {
        return std::nullopt;
    }
    return fe::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(configs[0], graph.getTag())
        .build();
}

// Runs the plan on a workspace of the size it asks for, then waits for the device.
inline void execute(cudnnHandle_t handle,
                    fe::ExecutionPlan& plan,
                    const std::vector<int64_t>& uids,
                    std::vector<void*> ptrs)
{
    const sdk::Workspace workspace(static_cast<size_t>(plan.getWorkspaceSize()));
    const fe::VariantPack pack
        = fe::VariantPackBuilder()
              .setWorkspacePointer(workspace.get())
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

template <typename T>
bool matches(const std::string& name,
             sdk::Tensor<T>& reference,
             sdk::Tensor<T>& actual,
             float absoluteTolerance,
             float relativeTolerance)
{
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T> validator(absoluteTolerance,
                                                                            relativeTolerance);
    return hipdnn_test_sdk::utilities::validateAndReport<T>(
        std::cout, name, validator, reference, actual, absoluteTolerance, relativeTolerance);
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

inline int finish(const Options& options, bool allPassed)
{
    const Summary& result = summary();
    std::cout << "RESULT ran=" << result.ran << " skipped=" << result.skipped << '\n';
    if(!allPassed)
    {
        return 1;
    }
    if(result.ran < options.requireRan)
    {
        std::cout << "FAILED: " << result.ran << " variant(s) ran, --require-ran expects at least "
                  << options.requireRan << '\n';
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
    // getCudnnStatus() is non-const, so the exception is caught by non-const reference. The
    // handlers use fprintf because a throwing stream insertion would escape main.
    catch(fe::cudnnException& e)
    {
        std::fprintf(stderr,
                     "cuDNN shim error (%s): %s\n",
                     cudnnGetErrorString(e.getCudnnStatus()),
                     e.what());
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
    catch(...)
    {
        std::fprintf(stderr, "Unhandled unknown exception\n");
        return 1;
    }
}

} // namespace v8_sample
