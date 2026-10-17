// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_PointWiseDesc.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

template <typename T>
std::vector<T> requireSpatialCount(std::vector<T> values,
                                   std::int64_t spatialDims,
                                   cudnnBackendAttributeName_t name)
{
    if(static_cast<std::int64_t>(values.size()) != spatialDims)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        attributeName(name) + " has " + std::to_string(values.size())
                            + " elements, expected the spatial dimension count "
                            + std::to_string(spatialDims));
    }
    return values;
}

inline std::int64_t requireSpatialDims(std::optional<std::int64_t> value,
                                       cudnnBackendAttributeName_t name)
{
    const std::int64_t spatialDims = requireAttribute(value, name);
    if(spatialDims < 1)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, attributeName(name) + " must be >= 1");
    }
    return spatialDims;
}

class ConvolutionDescriptor final : public Descriptor
{
public:
    ConvolutionDescriptor()
        : Descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)
    {
    }

    // Lowering facts; valid once finalized.
    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

    ConvolutionMode_t mode() const noexcept
    {
        return _mode;
    }

    std::int64_t spatialDims() const noexcept
    {
        return _spatialDims;
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _prePaddings;
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _postPaddings;
    }

    const std::vector<std::int64_t>& dilations() const noexcept
    {
        return _dilations;
    }

    const std::vector<std::int64_t>& filterStrides() const noexcept
    {
        return _filterStrides;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawComputeType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_CONVOLUTION_COMP_TYPE),
                               CUDNN_ATTR_CONVOLUTION_COMP_TYPE);
        const auto rawMode = requireAttribute(
            scalarAttribute<cudnnConvolutionMode_t>(CUDNN_ATTR_CONVOLUTION_CONV_MODE),
            CUDNN_ATTR_CONVOLUTION_CONV_MODE);
        const std::int64_t spatialDims
            = requireSpatialDims(scalarAttribute<std::int64_t>(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS),
                                 CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS);
        std::vector<std::int64_t> prePaddings
            = spatialArray(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, spatialDims);
        std::vector<std::int64_t> postPaddings
            = spatialArray(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, spatialDims);
        std::vector<std::int64_t> dilations
            = spatialArray(CUDNN_ATTR_CONVOLUTION_DILATIONS, spatialDims);
        std::vector<std::int64_t> filterStrides
            = spatialArray(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, spatialDims);
        if(filterStrides.front() <= 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES[0] must be > 0");
        }

        const DataType_t computeType
            = requireLowerable(rawComputeType, CUDNN_ATTR_CONVOLUTION_COMP_TYPE);
        const ConvolutionMode_t mode = requireLowerable(rawMode, CUDNN_ATTR_CONVOLUTION_CONV_MODE);

        _computeType = computeType;
        _mode = mode;
        _spatialDims = spatialDims;
        _prePaddings = std::move(prePaddings);
        _postPaddings = std::move(postPaddings);
        _dilations = std::move(dilations);
        _filterStrides = std::move(filterStrides);
    }

private:
    std::vector<std::int64_t> spatialArray(cudnnBackendAttributeName_t name,
                                           std::int64_t spatialDims) const
    {
        return requireSpatialCount(arrayAttribute<std::int64_t>(name), spatialDims, name);
    }

    DataType_t _computeType = DataType_t::NOT_SET;
    ConvolutionMode_t _mode = ConvolutionMode_t::NOT_SET;
    std::int64_t _spatialDims = 0;
    std::vector<std::int64_t> _prePaddings;
    std::vector<std::int64_t> _postPaddings;
    std::vector<std::int64_t> _dilations;
    std::vector<std::int64_t> _filterStrides;
};

class PointwiseDescriptor final : public Descriptor
{
public:
    PointwiseDescriptor()
        : Descriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR)
    {
    }

    // Lowering facts; valid once finalized.
    PointwiseMode_t mode() const noexcept
    {
        return _mode;
    }

    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

    // Mode parameters that differ from upstream's defaults. Upstream sets every
    // parameter of the mode, defaults included, and MIOpen rejects a fusion that
    // carries one at all, so lowering forwards only these.
    std::optional<double> reluLowerClip() const noexcept
    {
        return _reluLowerClip;
    }

    std::optional<double> reluUpperClip() const noexcept
    {
        return _reluUpperClip;
    }

    std::optional<double> reluLowerClipSlope() const noexcept
    {
        return _reluLowerClipSlope;
    }

    std::optional<double> eluAlpha() const noexcept
    {
        return _eluAlpha;
    }

    std::optional<double> softplusBeta() const noexcept
    {
        return _softplusBeta;
    }

    std::optional<double> swishBeta() const noexcept
    {
        return _swishBeta;
    }

    std::optional<std::int64_t> axis() const noexcept
    {
        return _axis;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawMode
            = requireAttribute(scalarAttribute<cudnnPointwiseMode_t>(CUDNN_ATTR_POINTWISE_MODE),
                               CUDNN_ATTR_POINTWISE_MODE);
        const auto rawComputeType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_POINTWISE_MATH_PREC),
                               CUDNN_ATTR_POINTWISE_MATH_PREC);
        for(const cudnnBackendAttributeName_t name : {CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                                      CUDNN_ATTR_POINTWISE_ELU_ALPHA,
                                                      CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                                      CUDNN_ATTR_POINTWISE_SWISH_BETA})
        {
            const std::optional<double> value = realAttribute(name);
            if(value && !std::isfinite(*value))
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM, attributeName(name) + " must be finite");
            }
        }

        _mode = requireLowerable(rawMode, CUDNN_ATTR_POINTWISE_MODE);
        _computeType = requireLowerable(rawComputeType, CUDNN_ATTR_POINTWISE_MATH_PREC);
        // Defaults from PointWiseDesc_v8 (cudnn_frontend_PointWiseDesc.h).
        _reluLowerClip = nonDefault(CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, 0.0);
        _reluLowerClipSlope = nonDefault(CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, 0.0);
        _eluAlpha = nonDefault(CUDNN_ATTR_POINTWISE_ELU_ALPHA, 1.0);
        _softplusBeta = nonDefault(CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, 1.0);
        _swishBeta = nonDefault(CUDNN_ATTR_POINTWISE_SWISH_BETA, 1.0);
        // The upper-clip default is DBL_MAX, which upstream clamps to FLT_MAX for
        // FLOAT math; any bound at or above FLT_MAX clips nothing once narrowed.
        const std::optional<double> upperClip = realAttribute(CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP);
        _reluUpperClip
            = upperClip && *upperClip < static_cast<double>(std::numeric_limits<float>::max())
                  ? upperClip
                  : std::nullopt;
        const std::optional<std::int64_t> axis
            = scalarAttribute<std::int64_t>(CUDNN_ATTR_POINTWISE_AXIS);
        _axis = axis && *axis != -1 ? axis : std::nullopt;
    }

private:
    std::optional<double> nonDefault(cudnnBackendAttributeName_t name, double defaultValue) const
    {
        const std::optional<double> value = realAttribute(name);
        return value && *value != defaultValue ? value : std::nullopt;
    }

    PointwiseMode_t _mode = PointwiseMode_t::NOT_SET;
    DataType_t _computeType = DataType_t::NOT_SET;
    std::optional<double> _reluLowerClip;
    std::optional<double> _reluUpperClip;
    std::optional<double> _reluLowerClipSlope;
    std::optional<double> _eluAlpha;
    std::optional<double> _softplusBeta;
    std::optional<double> _swishBeta;
    std::optional<std::int64_t> _axis;
};

class MatmulDescriptor final : public Descriptor
{
public:
    MatmulDescriptor()
        : Descriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR)
    {
    }

    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawComputeType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_MATMUL_COMP_TYPE),
                               CUDNN_ATTR_MATMUL_COMP_TYPE);
        if(hasAttribute(CUDNN_ATTR_MATMUL_PADDING_VALUE))
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_MATMUL_PADDING_VALUE is not supported");
        }
        _computeType = requireLowerable(rawComputeType, CUDNN_ATTR_MATMUL_COMP_TYPE);
    }

private:
    DataType_t _computeType = DataType_t::NOT_SET;
};

class ReductionDescriptor final : public Descriptor
{
public:
    ReductionDescriptor()
        : Descriptor(CUDNN_BACKEND_REDUCTION_DESCRIPTOR)
    {
    }

    ReductionMode_t mode() const noexcept
    {
        return _mode;
    }

    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

    bool isDeterministic() const noexcept
    {
        return _isDeterministic;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawMode = requireAttribute(
            scalarAttribute<cudnnReduceTensorOp_t>(CUDNN_ATTR_REDUCTION_OPERATOR),
            CUDNN_ATTR_REDUCTION_OPERATOR);
        const auto rawComputeType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_REDUCTION_COMP_TYPE),
                               CUDNN_ATTR_REDUCTION_COMP_TYPE);
        const ReductionMode_t mode = requireLowerable(rawMode, CUDNN_ATTR_REDUCTION_OPERATOR);
        const DataType_t computeType
            = requireLowerable(rawComputeType, CUDNN_ATTR_REDUCTION_COMP_TYPE);
        _mode = mode;
        _computeType = computeType;
        _isDeterministic
            = scalarAttribute<bool>(CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC).value_or(false);
    }

private:
    ReductionMode_t _mode = ReductionMode_t::NOT_SET;
    DataType_t _computeType = DataType_t::NOT_SET;
    bool _isDeterministic = false;
};

class ResampleDescriptor final : public Descriptor
{
public:
    ResampleDescriptor()
        : Descriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR)
    {
    }

    // Lowering facts; valid once finalized. The FRACTION arrays are integral
    // here (finalize rejects anything else).
    ResampleMode_t mode() const noexcept
    {
        return _mode;
    }

    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

    PaddingMode_t paddingMode() const noexcept
    {
        return _paddingMode;
    }

    const std::vector<std::int64_t>& windowDims() const noexcept
    {
        return _windowDims;
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _prePaddings;
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _postPaddings;
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _strides;
    }

protected:
    void finalizeImpl() override
    {
        const auto rawMode
            = requireAttribute(scalarAttribute<cudnnResampleMode_t>(CUDNN_ATTR_RESAMPLE_MODE),
                               CUDNN_ATTR_RESAMPLE_MODE);
        const auto rawComputeType
            = requireAttribute(scalarAttribute<cudnnDataType_t>(CUDNN_ATTR_RESAMPLE_COMP_TYPE),
                               CUDNN_ATTR_RESAMPLE_COMP_TYPE);
        // Upstream's builder refuses to build without a padding mode, and hipDNN's
        // resample descriptors require one.
        const auto rawPadding = requireAttribute(
            scalarAttribute<cudnnPaddingMode_t>(CUDNN_ATTR_RESAMPLE_PADDING_MODE),
            CUDNN_ATTR_RESAMPLE_PADDING_MODE);
        // Upstream sends 0 when the builder never set it, so this is a finalize
        // check, not a SET check.
        const std::int64_t spatialDims
            = requireSpatialDims(scalarAttribute<std::int64_t>(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS),
                                 CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS);
        const std::vector<cudnnFraction_t> windowDims
            = spatialArray(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, spatialDims);
        const std::vector<cudnnFraction_t> prePaddings
            = spatialArray(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, spatialDims);
        const std::vector<cudnnFraction_t> postPaddings
            = spatialArray(CUDNN_ATTR_RESAMPLE_POST_PADDINGS, spatialDims);
        const std::vector<cudnnFraction_t> strides
            = spatialArray(CUDNN_ATTR_RESAMPLE_STRIDES, spatialDims);

        // NEAREST and BILINEAR have no hipDNN resample mode.
        const ResampleMode_t mode = requireLowerable(rawMode, CUDNN_ATTR_RESAMPLE_MODE);
        const DataType_t computeType
            = requireLowerable(rawComputeType, CUDNN_ATTR_RESAMPLE_COMP_TYPE);
        const PaddingMode_t paddingMode
            = requireLowerable(rawPadding, CUDNN_ATTR_RESAMPLE_PADDING_MODE);
        std::vector<std::int64_t> integralWindowDims
            = integral(windowDims, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS);
        std::vector<std::int64_t> integralPrePaddings
            = integral(prePaddings, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS);
        std::vector<std::int64_t> integralPostPaddings
            = integral(postPaddings, CUDNN_ATTR_RESAMPLE_POST_PADDINGS);
        std::vector<std::int64_t> integralStrides = integral(strides, CUDNN_ATTR_RESAMPLE_STRIDES);

        _mode = mode;
        _computeType = computeType;
        _paddingMode = paddingMode;
        _windowDims = std::move(integralWindowDims);
        _prePaddings = std::move(integralPrePaddings);
        _postPaddings = std::move(integralPostPaddings);
        _strides = std::move(integralStrides);
    }

private:
    std::vector<cudnnFraction_t> spatialArray(cudnnBackendAttributeName_t name,
                                              std::int64_t spatialDims) const
    {
        return requireSpatialCount(arrayAttribute<cudnnFraction_t>(name), spatialDims, name);
    }

    // SET already rejected zero denominators.
    static std::vector<std::int64_t> integral(const std::vector<cudnnFraction_t>& fractions,
                                              cudnnBackendAttributeName_t name)
    {
        std::vector<std::int64_t> values;
        values.reserve(fractions.size());
        for(std::size_t i = 0; i < fractions.size(); ++i)
        {
            const cudnnFraction_t& fraction = fractions[i];
            // INT64_MIN / -1 overflows; its quotient is unrepresentable anyway.
            const bool overflows
                = fraction.denominator == -1
                  && fraction.numerator == std::numeric_limits<std::int64_t>::min();
            if(overflows || fraction.numerator % fraction.denominator != 0)
            {
                throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                                attributeName(name) + "[" + std::to_string(i)
                                    + "] = " + std::to_string(fraction.numerator) + "/"
                                    + std::to_string(fraction.denominator)
                                    + " is not an integer; fractional values are not supported");
            }
            values.push_back(fraction.numerator / fraction.denominator);
        }
        return values;
    }

    ResampleMode_t _mode = ResampleMode_t::NOT_SET;
    DataType_t _computeType = DataType_t::NOT_SET;
    PaddingMode_t _paddingMode = PaddingMode_t::NOT_SET;
    std::vector<std::int64_t> _windowDims;
    std::vector<std::int64_t> _prePaddings;
    std::vector<std::int64_t> _postPaddings;
    std::vector<std::int64_t> _strides;
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

protected:
    void finalizeImpl() override
    {
        requireFe(
            requireAttribute(scalarAttribute<cudnnRngDistribution_t>(CUDNN_ATTR_RNG_DISTRIBUTION),
                             CUDNN_ATTR_RNG_DISTRIBUTION),
            CUDNN_ATTR_RNG_DISTRIBUTION);
    }
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
