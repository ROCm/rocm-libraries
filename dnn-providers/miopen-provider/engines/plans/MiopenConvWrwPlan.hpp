// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <miopen/miopen.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

class ConvWrwParams
{
public:
    ConvWrwParams(
        const hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        bool deterministicEnabled = false);

    ConvWrwParams(const ConvWrwParams&) = delete;
    ConvWrwParams& operator=(const ConvWrwParams&) = delete;

    ConvWrwParams(ConvWrwParams&&) = default;
    ConvWrwParams& operator=(ConvWrwParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& dw() const;
    const MiopenTensor& dy() const;
    const MiopenConvDescriptor& conv() const;

    bool validTensors() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _dw;
    MiopenTensor _dy;
    MiopenConvDescriptor _conv;
    bool _tensorsValid;
};

class ConvWrwPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    ConvWrwPlan(const HipdnnMiopenHandle& handle,
                ConvWrwParams&& params,
                const HipdnnMiopenSettings& executionSettings);
    ~ConvWrwPlan() override = default;

    ConvWrwPlan(const ConvWrwPlan&) = delete;
    ConvWrwPlan& operator=(const ConvWrwPlan&) = delete;

    ConvWrwPlan(ConvWrwPlan&& other) = delete;
    ConvWrwPlan& operator=(ConvWrwPlan&& other) = delete;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvWrwParams _params;
    mutable std::mutex _algorithmMutex;
    mutable std::optional<miopenConvBwdWeightsAlgorithm_t> _algorithm;
    mutable size_t _workspaceSize = 0;
    HipdnnMiopenSettings _executionSettings;
};

} // namespace miopen_plugin
