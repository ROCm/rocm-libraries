// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_PointWiseDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// Lowering facts are valid once finalized; record() is null until then.

class ConvolutionDescriptor final : public Descriptor
{
public:
    ConvolutionDescriptor()
        : Descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::ConvolutionRecord>& record() const noexcept
    {
        return _record;
    }

    DataType_t computeType() const noexcept
    {
        return _record->computeType();
    }

    ConvolutionMode_t mode() const noexcept
    {
        return _record->mode();
    }

    std::int64_t spatialDims() const noexcept
    {
        return _record->spatialDims();
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _record->prePaddings();
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _record->postPaddings();
    }

    const std::vector<std::int64_t>& dilations() const noexcept
    {
        return _record->dilations();
    }

    const std::vector<std::int64_t>& filterStrides() const noexcept
    {
        return _record->filterStrides();
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ConvolutionLabels s_labels{
            attributeLabel(CUDNN_ATTR_CONVOLUTION_COMP_TYPE),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_CONV_MODE),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_DILATIONS),
            attributeLabel(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES),
        };
        v8::ConvolutionInputs inputs;
        inputs.computeType = scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_CONVOLUTION_COMP_TYPE);
        inputs.mode = scalarAttribute<cudnnConvolutionMode_t>(CUDNN_ATTR_CONVOLUTION_CONV_MODE);
        inputs.spatialDims = scalarAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS);
        inputs.prePaddings = arrayAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS);
        inputs.postPaddings = arrayAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS);
        inputs.dilations = arrayAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_DILATIONS);
        inputs.filterStrides = arrayAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES);
        _record = v8::makeConvRecord(std::move(inputs), s_labels);
    }

private:
    std::shared_ptr<const v8::ConvolutionRecord> _record;
};

class PointwiseDescriptor final : public Descriptor
{
public:
    PointwiseDescriptor()
        : Descriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::PointwiseRecord>& record() const noexcept
    {
        return _record;
    }

    PointwiseMode_t mode() const noexcept
    {
        return _record->mode();
    }

    DataType_t computeType() const noexcept
    {
        return _record->computeType();
    }

    std::optional<double> reluLowerClip() const noexcept
    {
        return _record->reluLowerClip();
    }

    std::optional<double> reluUpperClip() const noexcept
    {
        return _record->reluUpperClip();
    }

    std::optional<double> reluLowerClipSlope() const noexcept
    {
        return _record->reluLowerClipSlope();
    }

    std::optional<double> eluAlpha() const noexcept
    {
        return _record->eluAlpha();
    }

    std::optional<double> softplusBeta() const noexcept
    {
        return _record->softplusBeta();
    }

    std::optional<double> swishBeta() const noexcept
    {
        return _record->swishBeta();
    }

    std::optional<std::int64_t> axis() const noexcept
    {
        return _record->axis();
    }

protected:
    void finalizeImpl() override
    {
        static const v8::PointwiseLabels s_labels{
            attributeLabel(CUDNN_ATTR_POINTWISE_MODE),
            attributeLabel(CUDNN_ATTR_POINTWISE_MATH_PREC),
            attributeLabel(CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE),
            attributeLabel(CUDNN_ATTR_POINTWISE_ELU_ALPHA),
            attributeLabel(CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA),
            attributeLabel(CUDNN_ATTR_POINTWISE_SWISH_BETA),
        };
        v8::PointwiseInputs inputs;
        inputs.mode = scalarAttribute<cudnnPointwiseMode_t>(CUDNN_ATTR_POINTWISE_MODE);
        inputs.computeType = scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_POINTWISE_MATH_PREC);
        inputs.reluLowerClip = realAttribute(CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP);
        inputs.reluUpperClip = realAttribute(CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP);
        inputs.reluLowerClipSlope = realAttribute(CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE);
        inputs.eluAlpha = realAttribute(CUDNN_ATTR_POINTWISE_ELU_ALPHA);
        inputs.softplusBeta = realAttribute(CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA);
        inputs.swishBeta = realAttribute(CUDNN_ATTR_POINTWISE_SWISH_BETA);
        inputs.axis = scalarAttribute<std::int64_t>(CUDNN_ATTR_POINTWISE_AXIS);
        _record = v8::makePointwiseRecord(inputs, s_labels);
    }

private:
    std::shared_ptr<const v8::PointwiseRecord> _record;
};

class MatmulDescriptor final : public Descriptor
{
public:
    MatmulDescriptor()
        : Descriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::MatmulRecord>& record() const noexcept
    {
        return _record;
    }

    DataType_t computeType() const noexcept
    {
        return _record->computeType();
    }

protected:
    void finalizeImpl() override
    {
        static const v8::MatmulLabels s_labels{
            attributeLabel(CUDNN_ATTR_MATMUL_COMP_TYPE),
            attributeLabel(CUDNN_ATTR_MATMUL_PADDING_VALUE),
        };
        _record
            = v8::makeMatmulRecord({scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_MATMUL_COMP_TYPE),
                                    hasAttribute(CUDNN_ATTR_MATMUL_PADDING_VALUE)},
                                   s_labels);
    }

private:
    std::shared_ptr<const v8::MatmulRecord> _record;
};

class ReductionDescriptor final : public Descriptor
{
public:
    ReductionDescriptor()
        : Descriptor(CUDNN_BACKEND_REDUCTION_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::ReductionRecord>& record() const noexcept
    {
        return _record;
    }

    ReductionMode_t mode() const noexcept
    {
        return _record->mode();
    }

    DataType_t computeType() const noexcept
    {
        return _record->computeType();
    }

    bool isDeterministic() const noexcept
    {
        return _record->isDeterministic();
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ReductionLabels s_labels{
            attributeLabel(CUDNN_ATTR_REDUCTION_OPERATOR),
            attributeLabel(CUDNN_ATTR_REDUCTION_COMP_TYPE),
        };
        _record = v8::makeReductionRecord(
            {scalarAttribute<cudnnReduceTensorOp_t>(CUDNN_ATTR_REDUCTION_OPERATOR),
             scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_REDUCTION_COMP_TYPE),
             scalarAttribute<bool>(CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC).value_or(false)},
            s_labels);
    }

private:
    std::shared_ptr<const v8::ReductionRecord> _record;
};

class ResampleDescriptor final : public Descriptor
{
public:
    ResampleDescriptor()
        : Descriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::ResampleRecord>& record() const noexcept
    {
        return _record;
    }

    ResampleMode_t mode() const noexcept
    {
        return _record->mode();
    }

    DataType_t computeType() const noexcept
    {
        return _record->computeType();
    }

    PaddingMode_t paddingMode() const noexcept
    {
        return _record->paddingMode();
    }

    const std::vector<std::int64_t>& windowDims() const noexcept
    {
        return _record->windowDims();
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _record->prePaddings();
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _record->postPaddings();
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _record->strides();
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ResampleLabels s_labels{
            attributeLabel(CUDNN_ATTR_RESAMPLE_MODE),
            attributeLabel(CUDNN_ATTR_RESAMPLE_COMP_TYPE),
            attributeLabel(CUDNN_ATTR_RESAMPLE_PADDING_MODE),
            attributeLabel(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS),
            attributeLabel(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS),
            attributeLabel(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS),
            attributeLabel(CUDNN_ATTR_RESAMPLE_POST_PADDINGS),
            attributeLabel(CUDNN_ATTR_RESAMPLE_STRIDES),
        };
        // SET already rejected zero denominators.
        v8::ResampleInputs inputs;
        inputs.mode = scalarAttribute<cudnnResampleMode_t>(CUDNN_ATTR_RESAMPLE_MODE);
        inputs.computeType = scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_RESAMPLE_COMP_TYPE);
        inputs.paddingMode = scalarAttribute<cudnnPaddingMode_t>(CUDNN_ATTR_RESAMPLE_PADDING_MODE);
        inputs.spatialDims = scalarAttribute<std::int64_t>(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS);
        inputs.windowDims = arrayAttribute<cudnnFraction_t>(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS);
        inputs.prePaddings = arrayAttribute<cudnnFraction_t>(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS);
        inputs.postPaddings = arrayAttribute<cudnnFraction_t>(CUDNN_ATTR_RESAMPLE_POST_PADDINGS);
        inputs.strides = arrayAttribute<cudnnFraction_t>(CUDNN_ATTR_RESAMPLE_STRIDES);
        _record = v8::makeResampleRecord(inputs, s_labels);
    }

private:
    std::shared_ptr<const v8::ResampleRecord> _record;
};

// The RNG operation is not lowered (it reports NOT_SUPPORTED), so its
// parameters stay in the attribute store for GET only.
class RngDescriptor final : public Descriptor
{
public:
    RngDescriptor()
        : Descriptor(CUDNN_BACKEND_RNG_DESCRIPTOR)
    {
    }

    const std::shared_ptr<const v8::RngRecord>& record() const noexcept
    {
        return _record;
    }

protected:
    void finalizeImpl() override
    {
        _record = v8::makeRngRecord(
            scalarAttribute<cudnnRngDistribution_t>(CUDNN_ATTR_RNG_DISTRIBUTION),
            attributeLabel(CUDNN_ATTR_RNG_DISTRIBUTION));
    }

private:
    std::shared_ptr<const v8::RngRecord> _record;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
