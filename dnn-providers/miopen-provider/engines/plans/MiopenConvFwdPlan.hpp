// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <miopen/miopen.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

class ConvFwdParams
{
public:
    ConvFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        bool deterministicEnabled = false);

    ConvFwdParams(const ConvFwdParams&) = delete;
    ConvFwdParams& operator=(const ConvFwdParams&) = delete;

    ConvFwdParams(ConvFwdParams&&) = default;
    ConvFwdParams& operator=(ConvFwdParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& w() const;
    const MiopenTensor& y() const;
    const MiopenConvDescriptor& conv() const;

    size_t spatialDimCount() const;
    bool validTensors() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _w;
    MiopenTensor _y;
    MiopenConvDescriptor _conv;
    bool _tensorsValid;
};

class ConvFwdPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    ConvFwdPlan(const HipdnnMiopenHandle& handle,
                ConvFwdParams&& params,
                const HipdnnMiopenSettings& executionSettings);
    ~ConvFwdPlan() override = default;

    ConvFwdPlan(const ConvFwdPlan&) = delete;
    ConvFwdPlan& operator=(const ConvFwdPlan&) = delete;

    ConvFwdPlan(ConvFwdPlan&& other) = delete;
    ConvFwdPlan& operator=(ConvFwdPlan&& other) = delete;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvFwdParams _params;
    mutable std::mutex _algorithmMutex;
    mutable std::optional<miopenConvFwdAlgorithm_t> _algorithm;
    mutable size_t _workspaceSize = 0;
    HipdnnMiopenSettings _executionSettings;
};

} // namespace miopen_plugin
