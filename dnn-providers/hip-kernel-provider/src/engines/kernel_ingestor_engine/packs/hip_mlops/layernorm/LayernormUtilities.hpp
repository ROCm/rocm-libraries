// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "compilation/KernelCompileOptions.hpp"

namespace hip_kernel_provider::layernorm
{

enum class Direction
{
    FORWARD,
    BACKWARD
};

/// One launch of a layernorm kernel: the entry point of the compiled source and its 1D
/// geometry.
struct LayernormLaunch
{
    const char* entryPoint;
    unsigned int blockSize;
    unsigned int gridSize;
};

/// What a layernorm plan compiles and launches: the compile options that bake the graph into
/// the source, and its launches in order, all entry points of one program. Built by
/// buildLayernormFwdKernel()/buildLayernormBwdKernel(), which the kernel ingestor's HipMlops
/// packs call.
struct LayernormKernelBuild
{
    compilation::KernelCompileOptions options;
    std::vector<LayernormLaunch> launches;
};

/// Whether @p localSize is a workgroup the layernorm kernels can run: their tree reductions
/// halve it down to one lane, so a power of two, of at most 1024 threads.
inline bool isLocalSizeSupported(int64_t localSize)
{
    return localSize >= 1 && localSize <= 1024 && (localSize & (localSize - 1)) == 0;
}

/// The performance choice of the layernorm forward launch: the workgroup that normalizes each
/// row (HIP_PLUGIN_LAYERNORM_LOCAL_SIZE). The default is LayernormFwdParams::LOCAL_SIZE.
struct LayernormFwdConfig
{
    int64_t localSize = 1024;
};

inline bool operator==(const LayernormFwdConfig& lhs, const LayernormFwdConfig& rhs)
{
    return lhs.localSize == rhs.localSize;
}

inline bool operator!=(const LayernormFwdConfig& lhs, const LayernormFwdConfig& rhs)
{
    return !(lhs == rhs);
}

/// The performance choice of the layernorm backward launches: whether the scale/bias
/// reduction splits across workgroups (LayernormBwdScaleBiasParallel then
/// LayernormBwdScaleBiasReduceSum) or runs in one pass (LayernormBwdScaleBias), and the
/// workgroup every launch takes (HIP_PLUGIN_LAYERNORM_LOCAL_SIZE; the default is
/// LayernormBwdParams::MAX_LOCAL_SIZE). The parallel size derives from the problem and the
/// device, sized for MAX_LOCAL_SIZE whatever the local size, so the partial sums and the
/// workspace do not depend on it.
struct LayernormBwdConfig
{
    bool parallel = false;
    int64_t localSize = 1024;
};

inline bool operator==(const LayernormBwdConfig& lhs, const LayernormBwdConfig& rhs)
{
    return lhs.parallel == rhs.parallel && lhs.localSize == rhs.localSize;
}

inline bool operator!=(const LayernormBwdConfig& lhs, const LayernormBwdConfig& rhs)
{
    return !(lhs == rhs);
}

class ProblemDescription
{
public:
    ProblemDescription(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr,
        std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr,
        Direction direction);

    Direction direction() const
    {
        return _direction;
    }
    size_t normalizedDim() const
    {
        return _normalizedDim;
    }
    int64_t outerSize() const
    {
        return _outerSize;
    }
    int64_t innerSize() const
    {
        return _innerSize;
    }
    int64_t stride() const
    {
        return _stride;
    }

private:
    Direction _direction;
    size_t _normalizedDim;
    int64_t _outerSize{1};
    int64_t _innerSize{1};
    int64_t _stride{1};
};

size_t getMinNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr);

size_t getMinNormalizedDimFromAffine(
    int64_t ioTensorId,
    int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMaxNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr);

size_t getMaxNormalizedDimFromAffine(
    int64_t ioTensorId,
    int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMinNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr);

size_t getMinNormalizedDimFromStat(
    int64_t ioTensorId,
    int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMaxNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr);

size_t getMaxNormalizedDimFromStat(
    int64_t ioTensorId,
    int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t guessNormalizedDim(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> affineAttr,
    std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr);

} // namespace hip_kernel_provider::layernorm
