// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace hip_kernel_provider
{

using namespace compilation;

namespace layernorm
{

class LayernormFwdParams
{
public:
    LayernormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    LayernormFwdParams(const LayernormFwdParams&) = delete;
    LayernormFwdParams& operator=(const LayernormFwdParams&) = delete;

    LayernormFwdParams(LayernormFwdParams&&) = default;
    LayernormFwdParams& operator=(LayernormFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;
    double epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    static constexpr int64_t LOCAL_SIZE = 1024;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
};

/// The compile options and single launch of the LayernormFwd kernel for @p params on
/// @p gcnArchName.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 4D or 5D.
LayernormKernelBuild buildLayernormFwdKernel(const LayernormFwdParams& params,
                                             const std::string& gcnArchName);

class LayernormFwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit LayernormFwdPlan(LayernormFwdParams&& params);

    LayernormFwdPlan(const LayernormFwdPlan&) = delete;
    LayernormFwdPlan& operator=(const LayernormFwdPlan&) = delete;

    LayernormFwdPlan(LayernormFwdPlan&&) = default;
    LayernormFwdPlan& operator=(LayernormFwdPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const Handle& handle) const override;

    static std::vector<KernelLaunchShape> launchShapes(int64_t outerSize, int64_t stride);

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    LayernormFwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace layernorm

} // namespace hip_kernel_provider
