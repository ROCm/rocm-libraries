// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/hip_mlops_engine/plans/ApplicabilityChecks.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

namespace hip_kernel_provider::resample
{

class ResampleValidator : public IValidator
{
public:
    explicit ResampleValidator(
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMapLocal)
        : IValidator(tensorMapLocal) {};

    void checkTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr);

    void checkBwdTensorConfigSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& resampleBwdAttr);

private:
    void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) override;

    void checkTensorDataTypesSupported(const std::vector<int64_t>& ioTensorIds,
                                       const std::optional<int64_t>& indexTensorId = std::nullopt);

    void checkTensorShapesSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr);

    void checkBwdTensorShapesSupported(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& resampleBwdAttr);
};

/// The launch of the ResampleFwd plan for @p resampleAttr, one thread per y element, is not
/// empty and stays below the 2^32 threads HIP accepts. The HIP MLOps engine's builder and the
/// kernel ingestor's hipkernel:ResampleFwd both decline a graph that fails it, so neither
/// accepts a graph its build then rejects.
///
/// @throws HipdnnPluginException with BAD_PARAM naming the launch that does not fit.
void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

/// The launch of the ResampleBwd plan for @p resampleBwdAttr, one thread per dx element, is
/// not empty and stays below the 2^32 threads HIP accepts (as for the forward).
///
/// @throws HipdnnPluginException with BAD_PARAM naming the launch that does not fit.
void checkLaunchShapesFitThreadLimit(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& resampleBwdAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider::resample
