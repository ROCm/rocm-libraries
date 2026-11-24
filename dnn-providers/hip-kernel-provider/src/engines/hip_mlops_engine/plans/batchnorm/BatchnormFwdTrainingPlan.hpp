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
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormKernelCompileOptions.hpp"

#include <memory>
#include <string>

namespace hip_kernel_provider
{

using namespace compilation;

namespace batchnorm
{

class BatchnormFwdTrainingParams
{
public:
    BatchnormFwdTrainingParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(const BatchnormFwdTrainingParams&) = delete;
    BatchnormFwdTrainingParams& operator=(const BatchnormFwdTrainingParams&) = delete;

    BatchnormFwdTrainingParams(BatchnormFwdTrainingParams&&) = default;
    BatchnormFwdTrainingParams& operator=(BatchnormFwdTrainingParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    double epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    bool hasSaveMeanVariance() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;

    bool hasRunningStats() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningVariance() const;
    double momentumValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers) const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningVariance() const;

    const std::optional<core::utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;

    // Save mean/variance
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance = nullptr;

    // Running statistics
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _prevRunningMean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _prevRunningVariance = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _nextRunningMean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _nextRunningVariance = nullptr;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _momentum;
    bool _hasRunningStats{false};

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _activationOut;
};

/// The compile options, variant and launches of the BatchNormFwdTrainSpatial program for
/// @p params on @p gcnArchName: one launch, or for variant 2 the mean/variance, final
/// mean/variance and normalization launches. The launch heuristics size for a device of
/// @p multiProcessorCount CUs of @p warpSize lanes.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormKernelBuild buildBatchnormFwdTrainingKernel(const BatchnormFwdTrainingParams& params,
                                                     const std::string& gcnArchName,
                                                     int multiProcessorCount,
                                                     int warpSize);

class BatchnormFwdTrainingPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams);

    BatchnormFwdTrainingPlan(const BatchnormFwdTrainingPlan&) = delete;
    BatchnormFwdTrainingPlan& operator=(const BatchnormFwdTrainingPlan&) = delete;

    BatchnormFwdTrainingPlan(BatchnormFwdTrainingPlan&&) = default;
    BatchnormFwdTrainingPlan& operator=(BatchnormFwdTrainingPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdTrainingParams _trainingParams;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::vector<std::unique_ptr<IRunnableKernel>> _runnableKernels;

    // Kernel launch parameters computed during compile()
    int _kernelVariant = -1;
};

} // namespace batchnorm

} // namespace hip_kernel_provider
