// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/BatchnormApplicabilityChecks.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

// --- Configuration Data Structs ---
struct TensorConfig
{
    int64_t uid;
    std::string name;
    hipdnn_data_sdk::data_objects::DataType dataType;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::string description;
    bool isVirtual = false;
    bool isPassByValue = false;
    double passedValue = 0.0;
};

// --- Tensor Role Classification ---
enum class TensorRole
{
    IO, // Input/output tensors (x, y, dx, dy)
    AFFINE, // Scale/bias parameters
    STAT, // Mean/variance statistics
    SCALAR // Pass-by-value scalars (epsilon, momentum)
};

class TensorConfigBuilder
{
public:
    TensorConfigBuilder(int64_t uid, const std::string& name, TensorRole role)
        : _config{uid,
                  name,
                  hipdnn_data_sdk::data_objects::DataType::FLOAT,
                  {},
                  {},
                  "",
                  false,
                  false,
                  0.0}
        , _role(role)
    {
    }

    TensorConfigBuilder& withDataType(hipdnn_data_sdk::data_objects::DataType dt)
    {
        _config.dataType = dt;
        return *this;
    }

    TensorConfigBuilder& withDims(std::vector<int64_t> dims)
    {
        _config.dims = std::move(dims);
        return *this;
    }

    TensorConfigBuilder& withStrides(std::vector<int64_t> strides)
    {
        _config.strides = std::move(strides);
        return *this;
    }

    TensorConfigBuilder& withDescription(const std::string& desc)
    {
        _config.description = desc;
        return *this;
    }

    TensorConfigBuilder& asVirtual()
    {
        _config.isVirtual = true;
        return *this;
    }

    TensorConfigBuilder& asScalar(double value)
    {
        _config.isPassByValue = true;
        _config.passedValue = value;
        _config.dims = {1};
        _config.strides = {1};
        return *this;
    }

    TensorConfig build() const
    {
        return _config;
    }

private:
    TensorConfig _config;
    [[maybe_unused]] TensorRole _role;
};

// --- Canonical Tensor Layout Database ---
namespace canonical_layouts
{

// Test-specific: naming helper (uses TensorLayout.name!)
inline std::string generateName(const std::vector<int64_t>& dims,
                                const hipdnn_data_sdk::utilities::TensorLayout& layout)
{
    std::string name = layout.name; // Reuse production name!
    for(auto dim : dims)
    {
        name += "_" + std::to_string(dim);
    }
    return name;
}

// Shape collections (layout-independent)
namespace shapes
{
// 4D shapes for inference/backward (larger spatial dimensions)
inline const std::vector<std::vector<int64_t>> INFERENCE_4D = {
    {1, 3, 56, 56},
    {1, 3, 112, 112},
    {1, 3, 224, 224},
    {2, 3, 224, 224},
    {1, 16, 224, 224},
};

// 4D shapes for training (sufficient spatial: B×S > 1)
inline const std::vector<std::vector<int64_t>> TRAINING_4D = {
    {1, 3, 14, 14},
    {2, 3, 14, 14},
    {1, 3, 28, 28},
};

// 5D shapes for inference/backward
inline const std::vector<std::vector<int64_t>> INFERENCE_5D = {
    {1, 3, 16, 224, 224},
    {1, 4, 16, 224, 224},
    {1, 3, 8, 112, 112},
};

// 5D shapes for training (sufficient spatial: B×S > 1)
inline const std::vector<std::vector<int64_t>> TRAINING_5D = {
    {1, 4, 16, 14, 14},
    {1, 3, 16, 14, 14},
};

// Edge case shapes for specific test scenarios
inline const std::vector<int64_t> DEGENERATE_4D = {1, 1, 1, 1};
inline const std::vector<int64_t> INSUFFICIENT_SPATIAL_4D = {1, 3, 1, 1};
inline const std::vector<int64_t> DIFFERENT_CHANNELS_4D = {1, 5, 224, 224};

} // namespace shapes

// Test-specific: iteration arrays (use struct pointers or references)
inline const std::vector<const hipdnn_data_sdk::utilities::TensorLayout*> LAYOUTS_4D
    = {&hipdnn_data_sdk::utilities::TensorLayout::NCHW,
       &hipdnn_data_sdk::utilities::TensorLayout::NHWC};

inline const std::vector<const hipdnn_data_sdk::utilities::TensorLayout*> LAYOUTS_5D
    = {&hipdnn_data_sdk::utilities::TensorLayout::NCDHW,
       &hipdnn_data_sdk::utilities::TensorLayout::NDHWC};

namespace type_configs
{
using DT = hipdnn_data_sdk::data_objects::DataType;
// All invalid type configurations that should be REJECTED
// Documents what type combinations are NOT supported
inline const std::vector<hip_kernel_plugin::BnTensorTypes> INVALID_ALL = {
    // Invalid IO type (IO must be FLOAT/HALF/BFLOAT16)
    {DT::UINT8, DT::FLOAT, DT::FLOAT, DT::FLOAT}, // IO=UINT8 (unsupported)

    // Invalid affine types (scale/bias must be FLOAT)
    {DT::HALF, DT::HALF, DT::FLOAT, DT::FLOAT}, // IO=HALF, Affine=HALF
    {DT::BFLOAT16, DT::HALF, DT::FLOAT, DT::FLOAT}, // IO=BFLOAT16, Affine=HALF
    {DT::BFLOAT16, DT::BFLOAT16, DT::FLOAT, DT::FLOAT}, // IO=BFLOAT16, Affine=BFLOAT16

    // Invalid stat types (mean/variance must be FLOAT)
    {DT::HALF, DT::FLOAT, DT::HALF, DT::FLOAT}, // IO=HALF, Stat=HALF
    {DT::BFLOAT16, DT::FLOAT, DT::HALF, DT::FLOAT}, // IO=BFLOAT16, Stat=HALF
    {DT::BFLOAT16, DT::FLOAT, DT::BFLOAT16, DT::FLOAT}, // IO=BFLOAT16, Stat=BFLOAT16

    // Both affine and stat invalid (must be FLOAT)
    {DT::HALF, DT::HALF, DT::HALF, DT::FLOAT}, // All HALF
    {DT::BFLOAT16, DT::BFLOAT16, DT::BFLOAT16, DT::FLOAT}, // All BFLOAT16
    {DT::BFLOAT16, DT::HALF, DT::HALF, DT::FLOAT}, // Mixed invalid
};

// Invalid configs for fused operations (intermediate tensor data type errors)
inline const std::vector<hip_kernel_plugin::BnTensorTypes> INVALID_FUSED_ALL = []() {
    auto configs = INVALID_ALL;
    configs.push_back({DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::HALF});
    configs.push_back({DT::HALF, DT::FLOAT, DT::FLOAT, DT::HALF});
    configs.push_back({DT::FLOAT, DT::FLOAT, DT::FLOAT, DT::BFLOAT16});
    return configs;
}();

} // namespace type_configs

} // namespace canonical_layouts

// --- Helper Functions ---

std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*> buildTensorMap(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>&
        tensorOffsets);

std::pair<flatbuffers::FlatBufferBuilder,
          std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>>
    buildTensorMapFromConfigs(const std::vector<TensorConfig>& configs);

// --- Factory Helpers ---
TensorConfig createIoTensor(int64_t uid,
                            const std::string& name,
                            hipdnn_data_sdk::data_objects::DataType dt,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            bool isVirtual = false);
