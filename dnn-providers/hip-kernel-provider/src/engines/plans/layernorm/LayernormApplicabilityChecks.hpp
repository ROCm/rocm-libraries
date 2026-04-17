// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/ApplicabilityChecks.hpp"

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::layernorm
{

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

    void checkTensorLayoutsAndDimsSupported() override;

    void checkTensorIDLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds);

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::vector<int64_t>& affineTensorIds,
                                       const std::vector<int64_t>& statTensorIds,
                                       const std::vector<int64_t>& epsilonTensorIds);

    void checkTensorShapesSupported(const std::vector<int64_t>& ioTensorIds,
                                    const std::vector<int64_t>& affineTensorIds,
                                    const std::vector<int64_t>& statTensorIds);

    // --- High-level Configuration Validators ---

    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& lnAttr);
};

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
