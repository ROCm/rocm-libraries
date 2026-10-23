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

#include <memory>

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

    static size_t getReqdWorkItemCount(const hipDeviceProp_t& deviceProperties, size_t localSize);

    static bool isParallel(const hipDeviceProp_t& deviceProperties,
                           size_t localSize,
                           size_t innerSize,
                           size_t outerSize);

    static size_t getParallelSize(const hipDeviceProp_t& deviceProperties,
                                  size_t localSize,
                                  size_t innerSize,
                                  size_t outerSize);

private:
    LayernormBwdParams _params;
    int64_t _innerSize, _outerSize, _stride, _localSize;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::vector<std::unique_ptr<IRunnableKernel>> _runnableKernels;
    bool _isParallel;
};

} // namespace layernorm

} // namespace hip_kernel_provider
