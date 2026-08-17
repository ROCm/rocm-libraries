// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <miopen/miopen.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenConvDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

class ConvFwdBiasActivParams
{
public:
    ConvFwdBiasActivParams(
        const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        bool deterministicEnabled = false);
    ConvFwdBiasActivParams(const ConvFwdBiasActivParams&) = delete;
    ConvFwdBiasActivParams& operator=(const ConvFwdBiasActivParams&) = delete;

    ConvFwdBiasActivParams(ConvFwdBiasActivParams&&) = default;
    ConvFwdBiasActivParams& operator=(ConvFwdBiasActivParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& w() const;
    const MiopenConvDescriptor& conv() const;
    const std::optional<MiopenTensor>& bias() const;
    const miopen_utils::ActivationParams& activParams() const;
    const MiopenTensor& y() const;

private:
    size_t _spatialDimCount;
    MiopenTensor _x;
    MiopenTensor _w;
    MiopenConvDescriptor _conv;
    std::optional<MiopenTensor> _bias;
    miopen_utils::ActivationParams _activParams;
    MiopenTensor _y;
};

class ConvFwdBiasActivPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    ConvFwdBiasActivPlan(const HipdnnMiopenHandle& handle,
                         ConvFwdBiasActivParams&& params,
                         const HipdnnMiopenSettings& executionSettings,
                         bool compile = true,
                         bool getWsSize = true);
    ~ConvFwdBiasActivPlan() override = default;

    ConvFwdBiasActivPlan(const ConvFwdBiasActivPlan&) = delete;
    ConvFwdBiasActivPlan& operator=(const ConvFwdBiasActivPlan&) = delete;

    ConvFwdBiasActivPlan(ConvFwdBiasActivPlan&& other) = default;
    ConvFwdBiasActivPlan& operator=(ConvFwdBiasActivPlan&& other) = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ConvFwdBiasActivParams _params;
    hipdnn_data_sdk::utilities::ScopedResource<miopenFusionPlanDescriptor_t> _fusePlanDesc;
    size_t _workspaceSize = 0;
    HipdnnMiopenSettings _executionSettings;
};

} // namespace miopen_plugin
