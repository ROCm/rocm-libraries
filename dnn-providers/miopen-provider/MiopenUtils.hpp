// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <miopen/miopen.h>

#include "MiopenTensor.hpp"

#define LOG_ON_MIOPEN_FAILURE(status)                                                           \
    do                                                                                          \
    {                                                                                           \
        if((status) != miopenStatusSuccess)                                                     \
        {                                                                                       \
            HIPDNN_PLUGIN_LOG_ERROR("MIOpen error occurred: " << miopenGetErrorString(status)); \
        }                                                                                       \
    } while(0)

#define THROW_ON_MIOPEN_FAILURE(status)                                                 \
    do                                                                                  \
    {                                                                                   \
        if((status) != miopenStatusSuccess)                                             \
        {                                                                               \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                             \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "MIOpen error occurred: " + std::string(miopenGetErrorString(status))); \
        }                                                                               \
    } while(0)

/// @brief RAII guard for setting MIOpen tuning policy on a handle.
///
/// This class saves the current tuning policy, sets a new one based on the
/// benchmarking flag, and restores the original policy upon destruction.
/// This ensures that tuning policy changes don't leak to subsequent operations.
///
/// When benchmarking is enabled, uses miopenTuningPolicySearch (3) rather than
/// miopenTuningPolicySearchDbUpdate (4). The difference:
/// - Search (3): Uses cached results if available, searches only on cache miss
/// - SearchDbUpdate (4): Always re-searches, ignoring cached results
///
/// Search (3) matches PyTorch's benchmark=true behavior: first run searches and
/// caches, subsequent runs with same config use the cached optimal kernel.
/// This is more efficient for production use.

class ScopedTuningPolicy
{
public:
    /// @brief Construct and set the tuning policy on the handle.
    /// @param handle The MIOpen handle to set the policy on.
    /// @param benchmarkingEnabled If true, sets policy to miopenTuningPolicySearch.
    ///                           If false, sets policy to miopenTuningPolicyNone.
    ScopedTuningPolicy(miopenHandle_t handle, bool benchmarkingEnabled)
        : _handle(handle)
    {
        // Save original policy
        auto status = miopenGetTuningPolicy(_handle, &_originalPolicy);
        if(status != miopenStatusSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to get tuning policy: " << miopenGetErrorString(status));
            _originalPolicy = miopenTuningPolicyNone; // Fallback
        }

        // Set new policy: Search (3) for benchmarking, None (1) otherwise
        auto policy = benchmarkingEnabled ? miopenTuningPolicySearch : miopenTuningPolicyNone;
        status = miopenSetTuningPolicy(_handle, policy);
        if(status != miopenStatusSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to set tuning policy: " << miopenGetErrorString(status));
        }
        else
        {
            HIPDNN_PLUGIN_LOG_INFO("Tuning policy set to "
                                   << static_cast<int>(policy)
                                   << " (benchmarking=" << benchmarkingEnabled << ")");
        }
    }

    /// @brief Destructor restores tuning policy to original value.
    ~ScopedTuningPolicy()
    {
        auto status = miopenSetTuningPolicy(_handle, _originalPolicy);
        if(status != miopenStatusSuccess)
        {
            HIPDNN_PLUGIN_LOG_ERROR(
                "Failed to restore tuning policy: " << miopenGetErrorString(status));
        }
        else
        {
            HIPDNN_PLUGIN_LOG_INFO("Tuning policy restored to "
                                   << static_cast<int>(_originalPolicy));
        }
    }

    // Non-copyable
    ScopedTuningPolicy(const ScopedTuningPolicy&) = delete;
    ScopedTuningPolicy& operator=(const ScopedTuningPolicy&) = delete;

    // Non-movable
    ScopedTuningPolicy(ScopedTuningPolicy&&) = delete;
    ScopedTuningPolicy& operator=(ScopedTuningPolicy&&) = delete;

private:
    miopenHandle_t _handle;
    miopenTuningPolicy_t _originalPolicy{miopenTuningPolicyNone};
};

#define HIPDNN_PREPEND_MESSAGE_ON_THROW(statement, message)                                 \
    do                                                                                      \
    {                                                                                       \
        try                                                                                 \
        {                                                                                   \
            statement;                                                                      \
        }                                                                                   \
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)                        \
        {                                                                                   \
            throw hipdnn_plugin_sdk::HipdnnPluginException(error.getStatus(),               \
                                                           (message) + error.getMessage()); \
        }                                                                                   \
    } while(0)

namespace miopen_plugin::miopen_utils
{

struct ActivationParams
{
    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
};

ActivationParams mapPointwiseModeToMiopenActivation(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

miopenDataType_t
    tensorDataTypeToMiopenDataType(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType);

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

MiopenTensor createTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

/// @brief Creates a MiopenTensor with automatic 3D→4D padding for batchnorm.
///
/// If the tensor has 3 dimensions, it will be padded to 4D before creating
/// the MIOpen descriptor. 4D and 5D tensors are passed through unchanged.
///
/// @param tensorMap Map of tensor UIDs to TensorAttributes
/// @param uid The tensor UID to look up
/// @return MiopenTensor with potentially padded descriptor
MiopenTensor createBatchnormTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

size_t getSpatialDimCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attr);

/// @brief Returns the GPU architecture string (e.g., "gfx942") for the
/// device backing the given HIP stream. Strips any feature suffix such
/// as ":xnack+".
/// @throws hipdnn_plugin_sdk::HipdnnPluginException on HIP failure.
std::string getDeviceArch(hipStream_t stream);

using hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue;
using hipdnn_flatbuffers_sdk::utilities::extractValueFromTensorValue;

} // namespace miopen_plugin::miopen_utils
