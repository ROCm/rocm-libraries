// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{
using namespace compilation;

namespace resample
{

class ResampleBwdParams
{
public:
    ResampleBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType);

    ResampleBwdParams(const ResampleBwdParams&) = delete;
    ResampleBwdParams& operator=(const ResampleBwdParams&) = delete;

    ResampleBwdParams(ResampleBwdParams&&) = default;
    ResampleBwdParams& operator=(ResampleBwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* index() const;
    const std::vector<int64_t>& prePadding() const;
    const std::vector<int64_t>& postPadding() const;
    const std::vector<int64_t>& stride() const;
    const std::vector<int64_t>& window() const;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode() const;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode() const;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _index;
    std::vector<int64_t> _prePadding;
    std::vector<int64_t> _postPadding;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode _resampleMode;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode _paddingMode;
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType;
};

/// The compile options and grid of the ResampleBwd kernel for @p params on @p gcnArchName, in
/// blocks of @p config's size.
///
/// @throws HipdnnPluginException with BAD_PARAM when the shapes disagree with the resample
///         parameters or the output is too large for one launch.
ResampleKernelBuild buildResampleBwdKernel(const ResampleBwdParams& params,
                                           const std::string& gcnArchName,
                                           const ResampleConfig& config);

/// Whether the launch of @p config can run @p params: a supported block
/// (isBlockSizeSupported()) whose grid stays below the 2^32 threads HIP accepts.
/// checkLaunchShapesFitThreadLimit() checks the default's.
bool isResampleBwdConfigApplicable(const ResampleBwdParams& params, const ResampleConfig& config);

class ResampleBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit ResampleBwdPlan(ResampleBwdParams&& params);

    ResampleBwdPlan(const ResampleBwdPlan&) = delete;
    ResampleBwdPlan& operator=(const ResampleBwdPlan&) = delete;

    ResampleBwdPlan(ResampleBwdPlan&&) = default;
    ResampleBwdPlan& operator=(ResampleBwdPlan&&) = delete;

    size_t getWorkspaceSize(const Handle& handle) const override;

    // Every kernel launch of the plan for the given number of dx elements, in blocks of
    // blockSize threads.
    static std::vector<KernelLaunchShape> launchShapes(int64_t numElements, int64_t blockSize);

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ResampleBwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace resample
} // namespace hip_kernel_provider
