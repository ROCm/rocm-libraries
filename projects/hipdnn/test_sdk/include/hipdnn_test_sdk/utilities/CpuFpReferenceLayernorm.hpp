// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
                      double epsilon,
                      int64_t normalizedDimCount,
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
        if((scale != nullptr && scale->dims().size() == dims.size())
           || (bias != nullptr && bias->dims().size() == dims.size())) // Pad with ones
        {
            batchDims = std::vector<int64_t>(dims.size(), 1);
            normalizedDims = std::vector<int64_t>(dims.size(), 1);
            for(size_t i = 0; i < dims.size(); ++i)
            {
                if(static_cast<int64_t>(i) < ndim - normalizedDimCount)
                {
                    batchDims[i] = dims[i];
                }
                else
                {
                    normalizedDims[i] = dims[i];
                }
            }
        }
        else // Don't pad with ones
        {
            batchDims
                = std::vector<int64_t>(dims.begin(), dims.begin() + (ndim - normalizedDimCount));
            normalizedDims
                = std::vector<int64_t>(dims.begin() + (ndim - normalizedDimCount), dims.end());
        }

        for(auto d : normalizedDims)
        {
            if(d <= 0)
            {
                throw std::runtime_error(
                    "Normalized dimensions must all be positive (no zero-size dimensions).");
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

        if(batchIndices.size() == static_cast<size_t>(ndim)
           && normIndices.size() == static_cast<size_t>(ndim)) // Padded with 1
        {
            fullIndices.insert(fullIndices.end(),
                               batchIndices.begin(),
                               batchIndices.begin() + ndim - normalizedDimCount);
            fullIndices.insert(fullIndices.end(),
                               normIndices.begin() + ndim - normalizedDimCount,
                               normIndices.end());
        }
        else // Not padded with 1
        {
            // If batchDimCount is 0, the batch iteration was over a padded [1] dim, skip it
            if(batchDimCount > 0)
            {
                fullIndices.insert(fullIndices.end(), batchIndices.begin(), batchIndices.end());
            }

            fullIndices.insert(fullIndices.end(), normIndices.begin(), normIndices.end());
        }

        return fullIndices;
    }
};

} // namespace hipdnn_test_sdk::utilities
