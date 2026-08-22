// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_data_sdk/utilities/ShapeUtilities.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceLayernorm
{
public:
    // Layer normalization forward pass.
    // Normalizes over the last `normalizedDimCount` dimensions of the input tensor.
    //
    // For input X with shape [d0, d1, ..., d_{n-1}] and normalizedDimCount = k:
    //   - Batch dimensions:      [d0, ..., d_{n-k-1}]
    //   - Normalized dimensions: [d_{n-k}, ..., d_{n-1}]
    //   - For each batch position b, uses Welford's online algorithm:
    //       Pass 1 (Welford): Incrementally computes mean and variance in a single pass.
    //           For each element x_n (n = 1, 2, ...):
    //               delta   = x_n - mean_{n-1}
    //               mean_n  = mean_{n-1} + delta / n
    //               delta2  = x_n - mean_n
    //               M2_n    = M2_{n-1} + delta * delta2
    //           var_b  = M2 / m,  rstd_b = 1 / sqrt(var_b + epsilon)
    //       Pass 2: y[b, i] = scale[i] * (x[b, i] - mean_b) * rstd_b + bias[i]
    //
    // Welford's algorithm is chosen for this reference implementation because it:
    //   - Avoids accumulator overflow (mean updated incrementally, never summed)
    //   - Avoids catastrophic cancellation (no E[x²] - E[x]² subtraction)
    //   - Is numerically stable for arbitrary value ranges and element counts
    //
    // Scale and bias, if provided, have shape matching the normalized dimensions.
    // Mean and rstd outputs, if provided, have shape matching the batch dimensions.
    template <class XDataType,
              class ScaleBiasDataType,
              class YDataType = XDataType,
              class MeanRstdDataType = ScaleBiasDataType,
              class ComputeDataType = float>
    static void fprop(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>* scale,
                      const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>* bias,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const double epsilon,
                      const int64_t normalizedDimCount,
                      hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* mean = nullptr,
                      hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* rstd = nullptr)
    {
        const auto& dims = x.dims();
        auto ndim = static_cast<int64_t>(dims.size());

        if(ndim < 1)
        {
            throw std::runtime_error("Layernorm fprop requires at least 1D tensor.");
        }

        if(normalizedDimCount < 1 || normalizedDimCount > ndim)
        {
            throw std::runtime_error(
                "normalizedDimCount must be between 1 and the number of tensor dimensions.");
        }

        if(scale != nullptr && bias != nullptr && scale->dims().size() != bias->dims().size())
        {
            throw std::runtime_error("Scale and bias tensors must have the same rank.");
        }

        // Split dimensions into batch dims and normalized dims
        std::vector<int64_t> batchDims;
        std::vector<int64_t> normalizedDims;
        if(mean != nullptr)
        {
            batchDims = mean->dims();
        }
        else if(rstd != nullptr)
        {
            batchDims = rstd->dims();
        }
        else
        {
            batchDims
                = std::vector<int64_t>(dims.begin(), dims.begin() + ndim - normalizedDimCount);
        }
        if(scale != nullptr)
        {
            normalizedDims = scale->dims();
        }
        else if(bias != nullptr)
        {
            normalizedDims = bias->dims();
        }
        else
        {
            normalizedDims
                = std::vector<int64_t>(dims.begin() + ndim - normalizedDimCount, dims.end());
        }

        for(auto d : normalizedDims)
        {
            if(d <= 0)
            {
                throw std::runtime_error(
                    "Normalized dimensions must all be positive (no zero-size dimensions).");
            }
        }
        for(auto d : batchDims)
        {
            if(d <= 0)
            {
                throw std::runtime_error(
                    "Batch dimensions must all be positive (no zero-size dimensions).");
            }
        }

        auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        // If batchDims is empty (entire tensor is normalized), use a single scalar iteration
        if(batchDims.empty())
        {
            batchDims.push_back(1);
        }

        auto layernormFpropFunc = [&](const std::vector<int64_t>& batchIndices) {
            // Pass 1: Welford's online algorithm for mean and variance
            int64_t count = 0;
            auto batchMean = static_cast<ComputeDataType>(0.0);
            auto m2 = static_cast<ComputeDataType>(0.0);

            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                normalizedDims, [&](const std::vector<int64_t>& normIndices) {
                    auto fullIndices
                        = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));

                    count++;
                    auto delta = xVal - batchMean;
                    batchMean += delta / static_cast<ComputeDataType>(count);
                    auto delta2 = xVal - batchMean;
                    m2 += delta * delta2;
                });

            auto batchVariance = m2 / static_cast<ComputeDataType>(count);
            auto invStd = static_cast<ComputeDataType>(1.0)
                          / hipdnn_data_sdk::types::sqrt(batchVariance + epsilonCompute);

            // Pass 2: normalize and apply scale/bias
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                normalizedDims, [&](const std::vector<int64_t>& normIndices) {
                    auto fullIndices
                        = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto xHat = (xVal - batchMean) * invStd;

                    ComputeDataType yVal = xHat;
                    if(scale != nullptr)
                    {
                        yVal
                            = static_cast<ComputeDataType>(scale->getHostValue(normIndices)) * yVal;
                    }
                    if(bias != nullptr)
                    {
                        yVal = yVal + static_cast<ComputeDataType>(bias->getHostValue(normIndices));
                    }

                    y.setHostValue(static_cast<YDataType>(yVal), fullIndices);
                });

            // Save mean and rstd for this batch position if requested
            if(mean != nullptr)
            {
                mean->setHostValue(static_cast<MeanRstdDataType>(batchMean), batchIndices);
            }
            if(rstd != nullptr)
            {
                rstd->setHostValue(static_cast<MeanRstdDataType>(invStd), batchIndices);
            }
        };

        // Parallelize over batch dimensions
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(layernormFpropFunc, batchDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();

        if(mean != nullptr)
        {
            mean->memory().markHostModified();
        }
        if(rstd != nullptr)
        {
            rstd->memory().markHostModified();
        }
    }

    // Layer normalization backward pass.
    // Calculates the gradients for a normalization over the last `normalizedDimCount` dimensions of the input tensor X
    //
    // For input dY, X with shape [d0, d1, ..., d_{n-1}] and normalizedDimCount = k:
    //   - Batch dimensions:      [d0, ..., d_{n-k-1}]
    //   - Normalized dimensions: [d_{n-k}, ..., d_{n-1}]
    //   - Stage 1 (backward values):
    //       For each batch position b:
    //           For each element dy_b_n, x_b_n, scale_n (n = 1, 2, ..., N):
    //               sum_dy_scale_x_n = sum_dy_scale_x_{n-1} + dy_b_n * scale_n * x_b_n
    //               sum_dy_scale_n = sum_dy_scale_{n-1} + dy_b_n * scale_n
    //           a = rstd_b * rstd_b * rstd_b * (sum_dy_scale_x - sum_dy_scale * mean_b) / N
    //           b = rstd_b * sum_dy_scale / N - a * mean_b
    //           For each element dy_b_n, x_b_n, dx_b_n (n = 1, 2, ..., N):
    //               dx_b_n = rstd_b * dy_b_n * scale_n - a * x_b_n - b
    //   - Stage 2 (backward weights):
    //       For each normalized position n:
    //           For each element dy_n_b, x_n_b, mean_b, rstd_b (b = 1, 2, ...):
    //               dscale_sum_b = dscale_sum_{b-1} + dy_n_b * (x_n_b - mean_b) * rstd_b
    //               dbias_sum_b = dbias_sum_{b-1} + dy_n_b
    //           dscale_n = dscale_sum_b
    //           dbias_n = dbias_sum_b
    //
    // Scale and bias have shape matching the normalized dimensions.
    // Mean and rstd inputs, if provided, have shape matching the batch dimensions.
    template <class DyDataType,
              class ScaleBiasDataType,
              class DxDataType = DyDataType,
              class MeanRstdDataType = ScaleBiasDataType,
              class ComputeDataType = float>
    static void bprop(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                      const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& x,
                      const hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& scale,
                      hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dscale,
                      hipdnn_data_sdk::utilities::TensorBase<ScaleBiasDataType>& dbias,
                      [[maybe_unused]] const double epsilon,
                      const hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* mean,
                      const hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* rstd,
                      const int64_t normalizedDimCount)
    {
        const auto& dims = dy.dims();
        auto ndim = static_cast<int64_t>(dims.size());

        if(ndim < 1)
        {
            throw std::runtime_error("Layernorm bprop requires at least 1D tensor.");
        }

        if(normalizedDimCount < 1 || normalizedDimCount > ndim)
        {
            throw std::runtime_error(
                "normalizedDimCount must be between 1 and the number of tensor dimensions.");
        }

        if(scale.dims() != dscale.dims() || scale.dims() != dbias.dims())
        {
            throw std::runtime_error(
                "Scale, dscale and dbias tensors must have the same dimensions.");
        }

        if((mean == nullptr) != (rstd == nullptr))
        {
            throw std::runtime_error(
                "Layernorm backward requires both mean and rstd to be provided, or neither.");
        }

        if(mean != nullptr && mean->dims() != rstd->dims())
        {
            throw std::runtime_error("Mean and rstd tensors must have the same dimensions.");
        }

        // Split dimensions into batch dims and normalized dims
        auto normalizedDims = scale.dims();
        const int64_t normalizedDimsSize = std::accumulate(
            normalizedDims.begin(), normalizedDims.end(), int64_t{1}, std::multiplies<int64_t>{});
        std::vector<int64_t> batchDims;
        if(mean != nullptr)
        {
            batchDims = mean->dims();
        }
        else if(dims.size() == normalizedDims.size())
        {
            batchDims = std::vector<int64_t>(dims.size(), 1);
            for(size_t i = 0; i < dims.size(); ++i)
            {
                if(dims[i] != normalizedDims[i])
                {
                    batchDims[i] = dims[i];
                }
            }
        }
        else
        {
            batchDims = std::vector<int64_t>(static_cast<size_t>(ndim - normalizedDimCount), 1);
            for(size_t i = 0; i < static_cast<size_t>(ndim - normalizedDimCount); ++i)
            {
                batchDims[i] = dims[i];
            }
        }
        auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(x.strides());
        auto batchStrides = hipdnn_data_sdk::utilities::generateStrides(batchDims, strideOrder);
        const int64_t batchDimsSize = std::accumulate(
            batchDims.begin(), batchDims.end(), int64_t{1}, std::multiplies<int64_t>{});

        std::vector<ComputeDataType> tmpMean;
        std::vector<ComputeDataType> tmpRstd;
        if(mean == nullptr || rstd == nullptr)
        {
            tmpMean = std::vector<ComputeDataType>(static_cast<size_t>(batchDimsSize));
            tmpRstd = std::vector<ComputeDataType>(static_cast<size_t>(batchDimsSize));
        }

        // If batchDims is empty (entire tensor is normalized), use a single scalar iteration
        if(batchDims.empty())
        {
            batchDims.push_back(1);
        }

        // Pass 1: backward values
        auto layernormBpropValuesFunc = [&](const std::vector<int64_t>& batchIndices) {
            auto sumDyScaleX = static_cast<ComputeDataType>(0.0);
            auto sumDyScale = static_cast<ComputeDataType>(0.0);
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                normalizedDims, [&](const std::vector<int64_t>& normIndices) {
                    auto fullIndices
                        = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                    auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(fullIndices));
                    auto scaleVal = static_cast<ComputeDataType>(scale.getHostValue(normIndices));
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));

                    sumDyScaleX += dyVal * scaleVal * xVal;
                    sumDyScale += dyVal * scaleVal;
                });

            ComputeDataType meanVal;
            ComputeDataType rstdVal;
            if(mean == nullptr || rstd == nullptr)
            {
                int64_t count = 0;
                meanVal = static_cast<ComputeDataType>(0.0);
                auto m2 = static_cast<ComputeDataType>(0.0);

                hipdnn_data_sdk::utilities::iterateAlongDimensions(
                    normalizedDims, [&](const std::vector<int64_t>& normIndices) {
                        auto fullIndices
                            = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                        auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));

                        count++;
                        auto delta = xVal - meanVal;
                        meanVal += delta / static_cast<ComputeDataType>(count);
                        auto delta2 = xVal - meanVal;
                        m2 += delta * delta2;
                    });

                auto batchVariance = m2 / static_cast<ComputeDataType>(count);
                rstdVal = static_cast<ComputeDataType>(1.0)
                          / hipdnn_data_sdk::types::sqrt(batchVariance
                                                         + static_cast<ComputeDataType>(epsilon));

                auto idx = static_cast<size_t>(std::inner_product(
                    batchIndices.begin(), batchIndices.end(), batchStrides.begin(), int64_t{0}));
                tmpMean[idx] = meanVal;
                tmpRstd[idx] = rstdVal;
            }
            else
            {
                meanVal = static_cast<ComputeDataType>(mean->getHostValue(batchIndices));
                rstdVal = static_cast<ComputeDataType>(rstd->getHostValue(batchIndices));
            }

            auto a = rstdVal * rstdVal * rstdVal * (sumDyScaleX - sumDyScale * meanVal)
                     / static_cast<ComputeDataType>(normalizedDimsSize);
            auto b = rstdVal * sumDyScale / static_cast<ComputeDataType>(normalizedDimsSize)
                     - a * meanVal;
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                normalizedDims, [&](const std::vector<int64_t>& normIndices) {
                    auto fullIndices
                        = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                    auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(fullIndices));
                    auto scaleVal = static_cast<ComputeDataType>(scale.getHostValue(normIndices));
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    auto dxVal = rstdVal * dyVal * scaleVal - a * xVal - b;
                    dx.setHostValue(static_cast<DxDataType>(dxVal), fullIndices);
                });
        };

        // Pass 2: backward weights
        auto layernormBpropWeightsFunc = [&](const std::vector<int64_t>& normIndices) {
            auto dscaleVal = static_cast<ComputeDataType>(0.0);
            auto dbiasVal = static_cast<ComputeDataType>(0.0);
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                batchDims, [&](const std::vector<int64_t>& batchIndices) {
                    auto fullIndices
                        = buildFullIndices(batchIndices, normIndices, ndim, normalizedDimCount);
                    auto dyVal = static_cast<ComputeDataType>(dy.getHostValue(fullIndices));
                    auto xVal = static_cast<ComputeDataType>(x.getHostValue(fullIndices));
                    ComputeDataType meanVal;
                    ComputeDataType rstdVal;
                    if(mean == nullptr || rstd == nullptr)
                    {
                        auto idx = static_cast<size_t>(std::inner_product(batchIndices.begin(),
                                                                          batchIndices.end(),
                                                                          batchStrides.begin(),
                                                                          int64_t{0}));
                        meanVal = tmpMean[idx];
                        rstdVal = tmpRstd[idx];
                    }
                    else
                    {
                        meanVal = static_cast<ComputeDataType>(mean->getHostValue(batchIndices));
                        rstdVal = static_cast<ComputeDataType>(rstd->getHostValue(batchIndices));
                    }
                    dscaleVal += dyVal * (xVal - meanVal) * rstdVal;
                    dbiasVal += dyVal;
                });
            dscale.setHostValue(static_cast<ScaleBiasDataType>(dscaleVal), normIndices);
            dbias.setHostValue(static_cast<ScaleBiasDataType>(dbiasVal), normIndices);
        };

        // Parallelize over batch dimensions
        auto parallelValuesFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            layernormBpropValuesFunc, batchDims);
        parallelValuesFunc(std::thread::hardware_concurrency());
        auto parallelWeightsFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            layernormBpropWeightsFunc, normalizedDims);
        parallelWeightsFunc(std::thread::hardware_concurrency());

        dx.memory().markHostModified();
        dscale.memory().markHostModified();
        dbias.memory().markHostModified();
    }

private:
    // Build full N-dimensional indices from batch indices and normalized indices.
    // For a tensor of ndim dimensions with the last normalizedDimCount being normalized:
    //   fullIndices = [batchIndices..., normIndices...]
    // Handles the case where batchDims was padded with a leading 1 (scalar batch).
    static std::vector<int64_t> buildFullIndices(const std::vector<int64_t>& batchIndices,
                                                 const std::vector<int64_t>& normIndices,
                                                 int64_t ndim,
                                                 int64_t normalizedDimCount)
    {
        auto batchDimCount = ndim - normalizedDimCount;
        std::vector<int64_t> fullIndices;
        fullIndices.reserve(static_cast<size_t>(ndim));
        fullIndices.insert(
            fullIndices.end(), batchIndices.begin(), batchIndices.begin() + batchDimCount);
        fullIndices.insert(
            fullIndices.end(), normIndices.end() - normalizedDimCount, normIndices.end());
        return fullIndices;
    }
};

} // namespace hipdnn_test_sdk::utilities
