// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"

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

    // With normalizedDim taken from scale and mean as the plans take it, each affine tensor
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

// Whether the backward launches can run problem with config: both scale/bias reductions
// handle any shape (at a parallel size of 1 one thread reduces each column over every row),
// so every config applies.
inline bool isBwdConfigApplicable([[maybe_unused]] const ProblemDescription& problem,
                                  [[maybe_unused]] const LayernormBwdConfig& config)
{
    return true;
}

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
