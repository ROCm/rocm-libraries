// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

#include "MiopenApi.hpp"
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

namespace miopen_plugin::miopen_utils
{

/// @brief True when MIOPEN_FIND_ENFORCE names an enforcing action as MIOpen parses it:
/// DB_UPDATE, SEARCH, SEARCH_DB_UPDATE or DB_CLEAN (case-insensitive), or a number 2-5 as
/// std::stoul reads it. Unset, NONE, 1 and values MIOpen cannot use mean "nothing enforced".
bool envFindEnforceIsSet();

}

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
/// If MIOPEN_FIND_ENFORCE names an enforcing action, the guard sets miopenTuningPolicyNone
/// so MIOpen applies the environment value instead, as it does for PyTorch.
///
/// Batchnorm and fused convolution plans use this guard. Plain convolution plans do
/// not: they request tuning through Find's exhaustiveSearch argument, as PyTorch does,
/// so an MIOPEN_FIND_ENFORCE environment setting still applies to them.

class ScopedTuningPolicy
{
public:
    /// @brief Construct and set the tuning policy on the handle.
    /// @param handle The MIOpen handle to set the policy on.
    /// @param benchmarkingEnabled If true and MIOPEN_FIND_ENFORCE enforces nothing, sets policy
    ///                           to miopenTuningPolicySearch. Otherwise sets miopenTuningPolicyNone.
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

        // Search (3) for benchmarking, None (1) otherwise. None defers to MIOPEN_FIND_ENFORCE.
        auto policy = benchmarkingEnabled && !miopen_plugin::miopen_utils::envFindEnforceIsSet()
                          ? miopenTuningPolicySearch
                          : miopenTuningPolicyNone;
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

/// @brief Creates a MiopenTensor with automatic 3D→4D padding.
///
/// MIOpen has no 1D operation API. A 3D (NCL or NLC) tensor is padded to 4D by
/// appending a trailing size-1 dimension, which does not change the data layout.
/// 4D and 5D tensors are passed through unchanged.
///
/// @param tensorMap Map of tensor UIDs to TensorAttributes
/// @param uid The tensor UID to look up
/// @return MiopenTensor with potentially padded descriptor
MiopenTensor createPaddedTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

size_t getSpatialDimCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attr);

/// @brief Smallest spatial dimension count MIOpen accepts for a convolution.
inline constexpr size_t MIOPEN_MIN_CONV_SPATIAL_DIM_COUNT = 2;

/// @brief Returns the spatial dimension count to give MIOpen for a convolution.
///
/// A 1D convolution runs as a 2D convolution with a trailing size-1 spatial
/// dimension. Zero padding, unit stride and unit dilation over that dimension
/// leave the result unchanged.
constexpr size_t paddedConvSpatialDimCount(size_t spatialDimCount)
{
    return spatialDimCount < MIOPEN_MIN_CONV_SPATIAL_DIM_COUNT ? MIOPEN_MIN_CONV_SPATIAL_DIM_COUNT
                                                               : spatialDimCount;
}

/// @brief Matches PyTorch's torch.backends.cudnn.benchmark=True for convolution Find.
///
/// When benchmarking is enabled, sets the descriptor's find mode to
/// miopenConvolutionFindModeNormal. The default (DYNAMIC_HYBRID) returns an installed
/// find-db entry without measuring it and, on a miss, considers only dynamic solvers.
/// Normal mode skips the installed find-db and, on a user find-db miss, measures every
/// applicable solver. An existing user find-db entry is reused unmeasured, including one
/// written by a non-benchmarking run; PyTorch behaves the same. Callers also pass
/// exhaustiveSearch=true to Find so that solvers without a perf-db entry are tuned.
/// When benchmarking is disabled the descriptor keeps MIOpen's default find mode.
void setBenchmarkingFindMode(miopenConvolutionDescriptor_t convDescriptor,
                             bool benchmarkingEnabled);

using hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue;
using hipdnn_flatbuffers_sdk::utilities::extractValueFromTensorValue;

} // namespace miopen_plugin::miopen_utils
