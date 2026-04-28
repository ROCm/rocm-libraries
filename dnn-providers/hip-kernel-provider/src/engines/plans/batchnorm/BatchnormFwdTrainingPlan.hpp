// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

#include <memory>

namespace hip_kernel_provider
{

class IKernelCompiler;

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

    BatchnormFwdTrainingParams(const BatchnormFwdTrainingParams&) = delete;
    BatchnormFwdTrainingParams& operator=(const BatchnormFwdTrainingParams&) = delete;

    BatchnormFwdTrainingParams(BatchnormFwdTrainingParams&&) = default;
    BatchnormFwdTrainingParams& operator=(BatchnormFwdTrainingParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    double epsilonValue() const;

    bool hasSaveMeanVariance() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;

    bool hasRunningStats() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningVariance() const;
    double momentumValue() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningVariance() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    double _epsilonValue;

    // Save mean/variance
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance = nullptr;

    // Running statistics
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _prevRunningMean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _prevRunningVariance = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _nextRunningMean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _nextRunningVariance = nullptr;
    std::optional<double> _momentumValue;
    bool _hasRunningStats{false};
};

class BatchnormFwdTrainingPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams);

    BatchnormFwdTrainingPlan(const BatchnormFwdTrainingPlan&) = delete;
    BatchnormFwdTrainingPlan& operator=(const BatchnormFwdTrainingPlan&) = delete;

    BatchnormFwdTrainingPlan(BatchnormFwdTrainingPlan&&) = default;
    BatchnormFwdTrainingPlan& operator=(BatchnormFwdTrainingPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
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
    float _invInNhw;
};

} // namespace batchnorm

} // namespace hip_kernel_provider
