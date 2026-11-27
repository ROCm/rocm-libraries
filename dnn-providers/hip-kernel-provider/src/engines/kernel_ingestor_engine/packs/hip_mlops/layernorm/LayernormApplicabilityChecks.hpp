// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/kernel_ingestor_engine/packs/hip_mlops/ApplicabilityChecks.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormUtilities.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::layernorm
{
using NamedTensorId = std::pair<const char*, int64_t>;

class LayernormValidator : public IValidator
{
private:
public:
    LayernormValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    // --- Validation Utilities ---

    void validateNormalizedDim(const std::vector<int64_t>& ioTensorIds,
                               const std::vector<int64_t>& affineTensorIds,
                               const std::vector<int64_t>& statTensorIds);

    // --- Component Validators ---

    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds,
                                       const std::vector<int64_t>& epsilonTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds);

    // Affine tensors align to x's trailing dims, stat tensors to x's leading dims.
    // Each must be a device buffer, not pass-by-value, packed with no two indices sharing an
    // element and, over its dims of size > 1, ordered in memory as x is.
    void checkTensorMemoryOrderSupported(int64_t xTensorId,
                                         const std::vector<NamedTensorId>& affineTensors,
                                         const std::vector<NamedTensorId>& statTensors);

    // With normalizedDim taken from scale and mean as the kernels take it, each affine tensor
    // (leading 1s added up to x's rank) must be 1 before normalizedDim and equal x from it on,
    // and each stat tensor (trailing 1s added) must equal x before normalizedDim and be 1 from
    // it on. The kernels index affine tensors over x's normalized extent and stat tensors over
    // the rest.
    void checkOperandDimsMatchNormalizedDims(int64_t xTensorId,
                                             const std::vector<NamedTensorId>& affineTensors,
                                             std::optional<int64_t> meanTensorId,
                                             const std::vector<NamedTensorId>& statTensors);

    // The backward kernel writes dx with x's data type.
    void checkDxDataTypeSupported(int64_t xTensorId, int64_t dxTensorId);

    // --- High-level Configuration Validators ---

    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr);

    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& lnAttr);
};

class LayernormFwdParams;
class LayernormBwdParams;

// The launch of the Layernorm forward kernel for lnAttr, outerSize * stride workgroups, is not
// empty and stays below the 2^32 threads HIP accepts. The kernel ingestor's
// hipkernel:LayernormFwd declines a graph that fails it, so it accepts no graph its build then
// rejects.
//
// Throws HipdnnPluginException with BAD_PARAM naming the launch that does not fit.
void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

// The backward's graph-wide launch check, as for the forward (hipkernel:LayernormBwd): the main
// kernel, which every config launches alike, and the one-pass scale/bias reduction. The
// parallel reduction needs innerSize <= 4 * localSize * multiProcessorCount and sizes its grid
// to that bound, so on any real device its launches stay far below 2^32 threads; its grid
// depends on the device, so isBwdConfigApplicable() checks it per config.
//
// Throws HipdnnPluginException with BAD_PARAM naming the launch that does not fit.
void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& lnAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

// Whether the forward launch of config can run params: a supported local size
// (isLocalSizeSupported()) whose launch (layernormFwdLaunchShapes()) stays below the 2^32
// threads HIP accepts. checkLaunchShapesFitThreadLimit() checks the default's; a narrower
// workgroup launches fewer threads.
bool isFwdConfigApplicable(const LayernormFwdParams& params, const LayernormFwdConfig& config);

// Whether the backward launches of config can run params on a device with
// multiProcessorCount CUs: both scale/bias reductions handle any shape (at a parallel size of
// 1 one thread reduces each column over every row), so a config applies when its local size
// is supported (isLocalSizeSupported()) and each of its launches (layernormBwdLaunchShapes())
// stays below the 2^32 threads HIP accepts.
bool isBwdConfigApplicable(const LayernormBwdParams& params,
                           int multiProcessorCount,
                           const LayernormBwdConfig& config);

// Layernorm Type Configuration ---

// hip-kernel-provider layernorm requirements (based on underlying kernel constraints):
// - IO/Affine/Stat tensors: same type (FLOAT, HALF or BFLOAT16)
// - Epsilon tensors: FLOAT only
struct TensorTypes
{
    hipdnn_flatbuffers_sdk::data_objects::DataType io;
    hipdnn_flatbuffers_sdk::data_objects::DataType affine;
    hipdnn_flatbuffers_sdk::data_objects::DataType stat;
    hipdnn_flatbuffers_sdk::data_objects::DataType epsilon;
};

namespace type_configs
{
using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

inline constexpr TensorTypes FLOAT = {DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::FLOAT};
inline constexpr TensorTypes HALF = {DT::HALF, DT::HALF, DT::HALF, DT::FLOAT};
inline constexpr TensorTypes BFLOAT16 = {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16, DT::FLOAT};

inline constexpr std::array<TensorTypes, 3> VALID = {FLOAT, HALF, BFLOAT16};

std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedIoTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedAffineTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedStatTypes();
std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType> getAllowedEpsilonTypes();

} // namespace type_configs

} // namespace hip_kernel_provider::layernorm
