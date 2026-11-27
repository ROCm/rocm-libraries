// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "core/Handle.hpp"
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

class LayernormBwdParams
{
public:
    LayernormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    LayernormBwdParams(const LayernormBwdParams&) = delete;
    LayernormBwdParams& operator=(const LayernormBwdParams&) = delete;

    LayernormBwdParams(LayernormBwdParams&&) = default;
    LayernormBwdParams& operator=(LayernormBwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;
    std::optional<double> epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers) const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dscale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dbias() const;

    static const int64_t MAX_LOCAL_SIZE = 1024;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _epsilon;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dbias;
};

/// Whether the launch heuristics split the scale/bias reduction of @p params across
/// workgroups on a device with @p multiProcessorCount CUs: when the inner size alone does not
/// exceed 4 workgroups of MAX_LOCAL_SIZE per CU but the whole tensor does.
LayernormBwdConfig defaultLayernormBwdConfig(const LayernormBwdParams& params,
                                             int multiProcessorCount);

/// The compile options and launches of the LayernormBwd program for @p params on
/// @p gcnArchName, on a device with @p multiProcessorCount CUs: LayernormBwd, then
/// LayernormBwdScaleBias, or LayernormBwdScaleBiasParallel and LayernormBwdScaleBiasReduceSum
/// when @p config splits the reduction across workgroups.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 4D or 5D, or a grid
///         that does not fit one launch.
LayernormKernelBuild buildLayernormBwdKernel(const LayernormBwdParams& params,
                                             const std::string& gcnArchName,
                                             int multiProcessorCount,
                                             const LayernormBwdConfig& config);

/// The workspace the LayernormBwd launches of @p config need on a device with
/// @p multiProcessorCount CUs: the parallel reduction's partial sums, and mean and inverse
/// variance when the graph does not supply them. Pass the config the build takes.
size_t layernormBwdWorkspaceBytes(const LayernormBwdParams& params,
                                  int multiProcessorCount,
                                  const LayernormBwdConfig& config);

class LayernormBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit LayernormBwdPlan(LayernormBwdParams&& params);

    LayernormBwdPlan(const LayernormBwdPlan&) = delete;
    LayernormBwdPlan& operator=(const LayernormBwdPlan&) = delete;

    LayernormBwdPlan(LayernormBwdPlan&&) = default;
    LayernormBwdPlan& operator=(LayernormBwdPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

    // Launches in execution order: the main kernel, then the scale/bias kernels of the chosen mode.
    static std::vector<KernelLaunchShape> launchShapes(int64_t outerSize,
                                                       int64_t innerSize,
                                                       int64_t stride,
                                                       int64_t localSize,
                                                       bool isParallel,
                                                       int64_t parallelSize);

private:
    LayernormBwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::vector<std::unique_ptr<IRunnableKernel>> _runnableKernels;
    bool _isParallel;
};

} // namespace layernorm

} // namespace hip_kernel_provider
