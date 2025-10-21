// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvFwdBiasActivPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

ConvFwdBiasActivParams::ConvFwdBiasActivParams(
        const hipdnn_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
        const hipdnn_sdk::data_objects::PointwiseAttributes* biasAttr,
        const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid())))
    , _x(miopen_utils::createTensor(tensorMap, convAttr.x_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, convAttr.w_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, activAttr.out_0_tensor_uid()))
{
    const auto& attrX = miopen_utils::findTensorAttributes(tensorMap, _x.uid());
    const auto& attrW = miopen_utils::findTensorAttributes(tensorMap, _w.uid());

    const auto xDims = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto wDims = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_sdk::utilities::calculateGroupCount(xDims, wDims);

    _conv = MiopenConvDescriptor(_spatialDimCount, convAttr, static_cast<int>(groupCount));

    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr missing in_1_tensor_uid");
        }

        if(biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid())
        {
            _bias = miopen_utils::createTensor(tensorMap, biasAttr->in_1_tensor_uid().value());
        }
        else if(biasAttr->in_1_tensor_uid().value() == convAttr.y_tensor_uid())
        {
            _bias = miopen_utils::createTensor(tensorMap, biasAttr->in_0_tensor_uid());
        }
        else
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr tensor UIDs do not match convAttr y_tensor_uid");
        }
    }

    using PointwiseMode = hipdnn_sdk::data_objects::PointwiseMode;
    switch(activAttr.operation())
    {
    case PointwiseMode::ABS:
        _activMode = miopenActivationABS;
        break;
    case PointwiseMode::ELU_FWD:
        _activMode = miopenActivationELU;
        _activAlpha = static_cast<double>(activAttr.elu_alpha().value_or(1.0f));
        break;
    case PointwiseMode::RELU_FWD:
        if(activAttr.relu_upper_clip().has_value())
        {
            if(activAttr.relu_lower_clip().has_value())
            {
                _activMode = miopenActivationCLAMP;
                _activAlpha = static_cast<double>(activAttr.relu_lower_clip().value());
                _activBeta = static_cast<double>(activAttr.relu_upper_clip().value());
            }
            else
            {
                _activMode = miopenActivationCLIPPEDRELU;
                _activAlpha = static_cast<double>(activAttr.relu_upper_clip().value());
            }
        }
        else if(activAttr.relu_lower_clip_slope().has_value())
        {
            _activMode = miopenActivationLEAKYRELU;
            _activAlpha = static_cast<double>(activAttr.relu_lower_clip_slope().value());
        }
        else
        {
            _activMode = miopenActivationRELU;
        }
        break;
    case PointwiseMode::SIGMOID_FWD:
        _activMode = miopenActivationLOGISTIC;
        break;
    case PointwiseMode::SOFTPLUS_FWD:
        if(activAttr.softplus_beta().has_value() && activAttr.softplus_beta().value() != 1.0f)
        {
            throw hipdnn_plugin::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "ConvFwdBiasActivParams: softplus_beta other than 1.0 is not supported by MIOpen");
        }
        _activMode = miopenActivationSOFTRELU;
        break;
    case PointwiseMode::TANH_FWD:
        _activMode = miopenActivationTANH;
        _activAlpha = 1.0;
        _activBeta = 1.0;
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvFwdBiasActivParams: Unsupported activation mode in ConvFwdBiasActivParams");
    }
}

const MiopenTensor& ConvFwdBiasActivParams::x() const
{
    return _x;
}

const MiopenTensor& ConvFwdBiasActivParams::w() const
{
    return _w;
}

const MiopenConvDescriptor& ConvFwdBiasActivParams::conv() const
{
    return _conv;
}

const std::optional<MiopenTensor>& ConvFwdBiasActivParams::bias() const
{
    return _bias;
}

miopenActivationMode_t ConvFwdBiasActivParams::activMode() const
{
    return _activMode;
}

double ConvFwdBiasActivParams::activAlpha() const
{
    return _activAlpha;
}

double ConvFwdBiasActivParams::activBeta() const
{
    return _activBeta;
}

double ConvFwdBiasActivParams::activGamma() const
{
    return _activGamma;
}

const MiopenTensor& ConvFwdBiasActivParams::y() const
{
    return _y;
}

ConvFwdBiasActivPlan::ConvFwdBiasActivPlan(const HipdnnEnginePluginHandle& handle, ConvFwdBiasActivParams&& params, bool compile, bool getWsSize)
    : _params(std::move(params))
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    THROW_ON_MIOPEN_FAILURE(miopenCreateFusionPlan(&fusePlanDesc,
                                                   miopenVerticalFusion,
                                                   _params.x().tensorDescriptor()));
    _fusePlanDesc = hipdnn_sdk::utilities::ScopedResource<miopenFusionPlanDescriptor_t>(
        fusePlanDesc, [](miopenFusionPlanDescriptor_t desc) {
            auto status = miopenDestroyFusionPlan(desc);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroyFusionPlan failed in ConvFwdBiasActivPlan destructor");
            }
        });

    miopenFusionOpDescriptor_t convOp;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOpConvForward(fusePlanDesc,
                                                      &convOp,
                                                      _params.conv().convDescriptor(),
                                                      _params.w().tensorDescriptor()));

    if(_params.bias().has_value())
    {
        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenCreateOpBiasForward(
            fusePlanDesc, &biasOp, _params.bias().value().tensorDescriptor()));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOpActivationForward(fusePlanDesc,
                                                            &activOp,
                                                            _params.activMode()));
    
    if(compile)
    {
        THROW_ON_MIOPEN_FAILURE(miopenCompileFusionPlan(handle.miopenHandle,
                                                        fusePlanDesc));
    }

    if(getWsSize)
    {
        THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetWorkSpaceSize(handle.miopenHandle,
                                                                 fusePlanDesc,
                                                                 &_workspaceSize,
                                                                 static_cast<miopenConvFwdAlgorithm_t>(-1))); // Algo is not used in MIOpen
    }
}

size_t ConvFwdBiasActivPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void ConvFwdBiasActivPlan::execute(const HipdnnEnginePluginHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    miopenOperatorArgs_t fusionArgs;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOperatorArgs(&fusionArgs));
    auto fusionArgsRes = hipdnn_sdk::utilities::ScopedResource<miopenOperatorArgs_t>(
        fusionArgs, [](miopenOperatorArgs_t args) {
            auto status = miopenDestroyOperatorArgs(args);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroyOperatorArgs failed in ConvFwdBiasActivPlan destructor");
            }
        });

    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);

    int opIdx = 0;
    miopenFusionOpDescriptor_t convoOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &convoOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsConvForward(fusionArgs,
                                                       convoOp,
                                                       nullptr, // Default value for alpha is 1.0f
                                                       nullptr, // Default value for beta is 0.0f
                                                       wBuffer.ptr));

    if(_params.bias().has_value())
    {
        auto biasBuffer
            = miopen_utils::findDeviceBuffer(_params.bias().value().uid(), deviceBuffers, numDeviceBuffers);

        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &biasOp));
        THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsBiasForward(fusionArgs,
                                                           biasOp,
                                                           nullptr, // alpha is not used in MIOpen
                                                           nullptr, // beta is not used in MIOpen
                                                           biasBuffer.ptr));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx, &activOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsActivForward(fusionArgs,
                                                        activOp,
                                                        nullptr, // alpha is not used in MIOpen
                                                        nullptr, // beta is not used in MIOpen
                                                        _params.activAlpha(),
                                                        _params.activBeta(),
                                                        _params.activGamma()));

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);

    THROW_ON_MIOPEN_FAILURE(miopenExecuteFusionPlan_v2(handle.miopenHandle,
                                                       _fusePlanDesc.get(),
                                                       _params.x().tensorDescriptor(),
                                                       xBuffer.ptr,
                                                       _params.y().tensorDescriptor(),
                                                       yBuffer.ptr,
                                                       fusionArgs,
                                                       workspace,
                                                       workspaceSize));
}

} // namespace miopen_legacy_plugin
