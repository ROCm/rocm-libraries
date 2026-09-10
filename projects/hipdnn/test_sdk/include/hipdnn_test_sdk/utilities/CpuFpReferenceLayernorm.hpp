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

        // The passes address memory through hoisted base pointers and strides, which is the
        // dense layout only; a ragged tensor rebases every batch at its own offset.
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, PREFIX, "y");
        if(scale != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*scale, PREFIX, "scale");
        }
        if(bias != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*bias, PREFIX, "bias");
        }
        if(mean != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*mean, PREFIX, "mean");
        }
        if(rstd != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*rstd, PREFIX, "rstd");
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

        // Raw pointers and strides are hoisted once. getHostValue/setHostValue each cost a
        // virtual memory() call plus a stride reduction over an index vector built per
        // element, which is pure heap traffic in a loop that visits every element twice.
        const XDataType* xBase = x.memory().hostData();
        YDataType* yBase = y.memory().hostData();
        const ScaleBiasDataType* scaleBase
            = (scale != nullptr) ? scale->memory().hostData() : nullptr;
        const ScaleBiasDataType* biasBase = (bias != nullptr) ? bias->memory().hostData() : nullptr;
        MeanRstdDataType* meanBase = (mean != nullptr) ? mean->memory().hostData() : nullptr;
        MeanRstdDataType* rstdBase = (rstd != nullptr) ? rstd->memory().hostData() : nullptr;

        const auto& xStrides = x.strides();
        const auto& yStrides = y.strides();

        // Every batch position walks the normalized dims the same way, so hoist that walk
        // into flat offset tables built once. x and y are addressed by
        // [batchIndices..., trailing normIndices...], so only the trailing normalizedDimCount
        // axes of the walk move their address - the leading axes get a zero stride. scale and
        // bias are addressed by the whole normIndices, so they use their own strides directly.
        const auto batchDimCount = static_cast<size_t>(ndim - normalizedDimCount);
        const auto normIndexCount = normalizedDims.size();
        const auto normSuffixStart = normIndexCount - static_cast<size_t>(normalizedDimCount);

        std::vector<int64_t> xWalkStrides(normIndexCount, 0);
        std::vector<int64_t> yWalkStrides(normIndexCount, 0);
        for(size_t axis = 0; axis < static_cast<size_t>(normalizedDimCount); ++axis)
        {
            xWalkStrides[normSuffixStart + axis] = xStrides[batchDimCount + axis];
            yWalkStrides[normSuffixStart + axis] = yStrides[batchDimCount + axis];
        }

        const auto xNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, xWalkStrides.data());
        const auto yNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, yWalkStrides.data());
        const auto scaleOffsets = (scale != nullptr) ? hipdnn_test_sdk::detail::buildDenseOffsets(
                                                           normalizedDims, scale->strides().data())
                                                     : std::vector<int64_t>{};
        const auto biasOffsets = (bias != nullptr) ? hipdnn_test_sdk::detail::buildDenseOffsets(
                                                         normalizedDims, bias->strides().data())
                                                   : std::vector<int64_t>{};

        const auto normElementCount = xNormOffsets.size();

        auto layernormFpropFunc = [&](const std::vector<int64_t>& batchIndices) {
            const int64_t xBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), xStrides.data(), batchDimCount);
            const int64_t yBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), yStrides.data(), batchDimCount);

            // Pass 1: Welford's online algorithm for mean and variance
            int64_t count = 0;
            auto batchMean = static_cast<ComputeDataType>(0.0);
            auto m2 = static_cast<ComputeDataType>(0.0);

            for(size_t element = 0; element < normElementCount; ++element)
            {
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xBatchOffset + xNormOffsets[element]]);

                count++;
                auto delta = xVal - batchMean;
                batchMean += delta / static_cast<ComputeDataType>(count);
                auto delta2 = xVal - batchMean;
                m2 += delta * delta2;
            }

            auto batchVariance = m2 / static_cast<ComputeDataType>(count);
            auto invStd = static_cast<ComputeDataType>(1.0)
                          / hipdnn_data_sdk::types::sqrt(batchVariance + epsilonCompute);

            // Pass 2: normalize and apply scale/bias
            for(size_t element = 0; element < normElementCount; ++element)
            {
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xBatchOffset + xNormOffsets[element]]);
                auto xHat = (xVal - batchMean) * invStd;

                ComputeDataType yVal = xHat;
                if(scaleBase != nullptr)
                {
                    yVal = static_cast<ComputeDataType>(scaleBase[scaleOffsets[element]]) * yVal;
                }
                if(biasBase != nullptr)
                {
                    yVal = yVal + static_cast<ComputeDataType>(biasBase[biasOffsets[element]]);
                }

                yBase[yBatchOffset + yNormOffsets[element]] = static_cast<YDataType>(yVal);
            }

            // Save mean and rstd for this batch position if requested
            if(meanBase != nullptr)
            {
                meanBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), mean->strides().data(), batchIndices.size())]
                    = static_cast<MeanRstdDataType>(batchMean);
            }
            if(rstdBase != nullptr)
            {
                rstdBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), rstd->strides().data(), batchIndices.size())]
                    = static_cast<MeanRstdDataType>(invStd);
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
                      const hipdnn_data_sdk::utilities::TensorBase<DxDataType>& x,
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

        // The passes address memory through hoisted base pointers and strides, which is the
        // dense layout only; a ragged tensor rebases every batch at its own offset.
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dy, PREFIX, "dy");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(scale, PREFIX, "scale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dx, PREFIX, "dx");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dscale, PREFIX, "dscale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dbias, PREFIX, "dbias");
        if(mean != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*mean, PREFIX, "mean");
        }
        if(rstd != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*rstd, PREFIX, "rstd");
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

        // If batchDims is empty (entire tensor is normalized), use a single scalar iteration.
        // batchStrides was generated before this, so keep it the same length as the walk.
        if(batchDims.empty())
        {
            batchDims.push_back(1);
        }
        batchStrides.resize(batchDims.size(), 0);

        // Raw pointers and strides are hoisted once; see fprop.
        const DyDataType* dyBase = dy.memory().hostData();
        const DxDataType* xBase = x.memory().hostData();
        const ScaleBiasDataType* scaleBase = scale.memory().hostData();
        DxDataType* dxBase = dx.memory().hostData();
        ScaleBiasDataType* dscaleBase = dscale.memory().hostData();
        ScaleBiasDataType* dbiasBase = dbias.memory().hostData();
        const MeanRstdDataType* meanBase = (mean != nullptr) ? mean->memory().hostData() : nullptr;
        const MeanRstdDataType* rstdBase = (rstd != nullptr) ? rstd->memory().hostData() : nullptr;

        const auto& dyStrides = dy.strides();
        const auto& xStrides = x.strides();
        const auto& dxStrides = dx.strides();
        const auto& scaleStrides = scale.strides();
        const auto& dscaleStrides = dscale.strides();
        const auto& dbiasStrides = dbias.strides();

        // Each pass holds one index space fixed and walks the other, and the walk is identical
        // every time, so hoist both into flat offset tables. dy/x/dx are addressed by
        // [batchIndices..., trailing normIndices...]: the normalized walk moves only their
        // trailing normalizedDimCount axes and the batch walk only their leading batchDimCount
        // axes, so the axes the walk does not reach get a zero stride. scale/dscale/dbias are
        // addressed by the whole normIndices and mean/rstd by the whole batchIndices, so those
        // use their own strides directly.
        const auto batchDimCount = static_cast<size_t>(ndim - normalizedDimCount);
        const auto normIndexCount = normalizedDims.size();
        const auto normSuffixStart = normIndexCount - static_cast<size_t>(normalizedDimCount);

        std::vector<int64_t> dyNormWalk(normIndexCount, 0);
        std::vector<int64_t> xNormWalk(normIndexCount, 0);
        std::vector<int64_t> dxNormWalk(normIndexCount, 0);
        for(size_t axis = 0; axis < static_cast<size_t>(normalizedDimCount); ++axis)
        {
            dyNormWalk[normSuffixStart + axis] = dyStrides[batchDimCount + axis];
            xNormWalk[normSuffixStart + axis] = xStrides[batchDimCount + axis];
            dxNormWalk[normSuffixStart + axis] = dxStrides[batchDimCount + axis];
        }

        std::vector<int64_t> dyBatchWalk(batchDims.size(), 0);
        std::vector<int64_t> xBatchWalk(batchDims.size(), 0);
        for(size_t axis = 0; axis < batchDimCount; ++axis)
        {
            dyBatchWalk[axis] = dyStrides[axis];
            xBatchWalk[axis] = xStrides[axis];
        }

        const auto dyNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, dyNormWalk.data());
        const auto xNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, xNormWalk.data());
        const auto dxNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, dxNormWalk.data());
        const auto scaleNormOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(normalizedDims, scaleStrides.data());

        const auto dyBatchOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(batchDims, dyBatchWalk.data());
        const auto xBatchOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(batchDims, xBatchWalk.data());
        const auto meanBatchOffsets
            = (mean != nullptr)
                  ? hipdnn_test_sdk::detail::buildDenseOffsets(batchDims, mean->strides().data())
                  : std::vector<int64_t>{};
        const auto rstdBatchOffsets
            = (rstd != nullptr)
                  ? hipdnn_test_sdk::detail::buildDenseOffsets(batchDims, rstd->strides().data())
                  : std::vector<int64_t>{};
        // tmpMean/tmpRstd are indexed by the batch position's flat index under batchStrides.
        const auto tmpBatchOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(batchDims, batchStrides.data());

        const auto normElementCount = dyNormOffsets.size();
        const auto batchElementCount = dyBatchOffsets.size();

        // Pass 1: backward values
        auto layernormBpropValuesFunc = [&](const std::vector<int64_t>& batchIndices) {
            const int64_t dyBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), dyStrides.data(), batchDimCount);
            const int64_t xBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), xStrides.data(), batchDimCount);
            const int64_t dxBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), dxStrides.data(), batchDimCount);

            auto sumDyScaleX = static_cast<ComputeDataType>(0.0);
            auto sumDyScale = static_cast<ComputeDataType>(0.0);
            for(size_t element = 0; element < normElementCount; ++element)
            {
                auto dyVal
                    = static_cast<ComputeDataType>(dyBase[dyBatchOffset + dyNormOffsets[element]]);
                auto scaleVal = static_cast<ComputeDataType>(scaleBase[scaleNormOffsets[element]]);
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xBatchOffset + xNormOffsets[element]]);

                sumDyScaleX += dyVal * scaleVal * xVal;
                sumDyScale += dyVal * scaleVal;
            }

            ComputeDataType meanVal;
            ComputeDataType rstdVal;
            if(meanBase == nullptr || rstdBase == nullptr)
            {
                int64_t count = 0;
                meanVal = static_cast<ComputeDataType>(0.0);
                auto m2 = static_cast<ComputeDataType>(0.0);

                for(size_t element = 0; element < normElementCount; ++element)
                {
                    auto xVal
                        = static_cast<ComputeDataType>(xBase[xBatchOffset + xNormOffsets[element]]);

                    count++;
                    auto delta = xVal - meanVal;
                    meanVal += delta / static_cast<ComputeDataType>(count);
                    auto delta2 = xVal - meanVal;
                    m2 += delta * delta2;
                }

                auto batchVariance = m2 / static_cast<ComputeDataType>(count);
                rstdVal = static_cast<ComputeDataType>(1.0)
                          / hipdnn_data_sdk::types::sqrt(batchVariance
                                                         + static_cast<ComputeDataType>(epsilon));

                auto idx = static_cast<size_t>(hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), batchStrides.data(), batchIndices.size()));
                tmpMean[idx] = meanVal;
                tmpRstd[idx] = rstdVal;
            }
            else
            {
                meanVal = static_cast<ComputeDataType>(meanBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), mean->strides().data(), batchIndices.size())]);
                rstdVal = static_cast<ComputeDataType>(rstdBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), rstd->strides().data(), batchIndices.size())]);
            }

            auto a = rstdVal * rstdVal * rstdVal * (sumDyScaleX - sumDyScale * meanVal)
                     / static_cast<ComputeDataType>(normalizedDimsSize);
            auto b = rstdVal * sumDyScale / static_cast<ComputeDataType>(normalizedDimsSize)
                     - a * meanVal;
            for(size_t element = 0; element < normElementCount; ++element)
            {
                auto dyVal
                    = static_cast<ComputeDataType>(dyBase[dyBatchOffset + dyNormOffsets[element]]);
                auto scaleVal = static_cast<ComputeDataType>(scaleBase[scaleNormOffsets[element]]);
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xBatchOffset + xNormOffsets[element]]);
                auto dxVal = rstdVal * dyVal * scaleVal - a * xVal - b;
                dxBase[dxBatchOffset + dxNormOffsets[element]] = static_cast<DxDataType>(dxVal);
            }
        };

        // Pass 2: backward weights
        auto layernormBpropWeightsFunc = [&](const std::vector<int64_t>& normIndices) {
            const int64_t dyNormOffset
                = hipdnn_test_sdk::detail::flatOffset(normIndices.data() + normSuffixStart,
                                                      dyStrides.data() + batchDimCount,
                                                      static_cast<size_t>(normalizedDimCount));
            const int64_t xNormOffset
                = hipdnn_test_sdk::detail::flatOffset(normIndices.data() + normSuffixStart,
                                                      xStrides.data() + batchDimCount,
                                                      static_cast<size_t>(normalizedDimCount));

            auto dscaleVal = static_cast<ComputeDataType>(0.0);
            auto dbiasVal = static_cast<ComputeDataType>(0.0);
            for(size_t element = 0; element < batchElementCount; ++element)
            {
                auto dyVal
                    = static_cast<ComputeDataType>(dyBase[dyNormOffset + dyBatchOffsets[element]]);
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xNormOffset + xBatchOffsets[element]]);
                ComputeDataType meanVal;
                ComputeDataType rstdVal;
                if(meanBase == nullptr || rstdBase == nullptr)
                {
                    auto idx = static_cast<size_t>(tmpBatchOffsets[element]);
                    meanVal = tmpMean[idx];
                    rstdVal = tmpRstd[idx];
                }
                else
                {
                    meanVal = static_cast<ComputeDataType>(meanBase[meanBatchOffsets[element]]);
                    rstdVal = static_cast<ComputeDataType>(rstdBase[rstdBatchOffsets[element]]);
                }
                dscaleVal += dyVal * (xVal - meanVal) * rstdVal;
                dbiasVal += dyVal;
            }

            dscaleBase[hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), dscaleStrides.data(), normIndices.size())]
                = static_cast<ScaleBiasDataType>(dscaleVal);
            dbiasBase[hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), dbiasStrides.data(), normIndices.size())]
                = static_cast<ScaleBiasDataType>(dbiasVal);
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
    static constexpr auto PREFIX = "CpuFpReferenceLayernorm: ";
};

} // namespace hipdnn_test_sdk::utilities
