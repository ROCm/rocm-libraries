// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/resample/ResamplePlanUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{
using namespace compilation;

namespace resample
{

class ResampleFwdParams
{
public:
    ResampleFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType);

    ResampleFwdParams(const ResampleFwdParams&) = delete;
    ResampleFwdParams& operator=(const ResampleFwdParams&) = delete;

    ResampleFwdParams(ResampleFwdParams&&) = default;
    ResampleFwdParams& operator=(ResampleFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* index() const;
    const std::vector<int64_t>& prePadding() const;
    const std::vector<int64_t>& postPadding() const;
    const std::vector<int64_t>& stride() const;
    const std::vector<int64_t>& window() const;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode() const;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode() const;
    bool generateIndex() const;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _index;
    std::vector<int64_t> _prePadding;
    std::vector<int64_t> _postPadding;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode _mode;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode _paddingMode;
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType;
};

/// The compile options and grid of the ResampleFwd kernel for @p params on @p gcnArchName, in
/// blocks of @p config's size.
///
/// @throws HipdnnPluginException with BAD_PARAM when the shapes disagree with the resample
///         parameters or the output is too large for one launch.
ResampleKernelBuild buildResampleFwdKernel(const ResampleFwdParams& params,
                                           const std::string& gcnArchName,
                                           const ResampleConfig& config);

/// Whether the launch of @p config can run @p params: a supported block
/// (isBlockSizeSupported()) whose grid stays below the 2^32 threads HIP accepts.
/// checkLaunchShapesFitThreadLimit() checks the default's.
bool isResampleFwdConfigApplicable(const ResampleFwdParams& params, const ResampleConfig& config);

/// Every kernel launch of the ResampleFwd kernel for @p numElements output elements, in blocks of
/// @p blockSize threads.
std::vector<KernelLaunchShape> resampleFwdLaunchShapes(int64_t numElements, int64_t blockSize);

} // namespace resample
} // namespace hip_kernel_provider
