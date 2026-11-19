// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_data_sdk/utilities/ShapeUtilities.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceLayernorm
{
    static constexpr auto PREFIX = "CpuFpReferenceLayernorm: ";

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
    // y has the shape of x.
    // Scale and bias, if provided, share one shape: the normalized dimensions, optionally
    // preceded by 1s.
    // Mean and rstd outputs, if provided, share one shape: the batch dimensions, optionally
    // followed by 1s, with at least one dimension.
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

        validateNormalizedDimCount(dims, normalizedDimCount, "fprop");

        if(std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d <= 0; }))
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "fprop requires every dimension to be positive.");
        }

        if(y.dims() != dims)
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "fprop requires y to have the same shape as x.");
        }
        if(scale != nullptr && bias != nullptr && scale->dims() != bias->dims())
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "fprop requires scale and bias to have the same shape.");
        }
        if(mean != nullptr && rstd != nullptr && mean->dims() != rstd->dims())
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "fprop requires mean and rstd to have the same shape.");
        }

        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, PREFIX, "y");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(scale, PREFIX, "scale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(bias, PREFIX, "bias");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(mean, PREFIX, "mean");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(rstd, PREFIX, "rstd");

        const auto* affine = (scale != nullptr) ? scale : bias;
        if(affine != nullptr)
        {
            validateAffineShape(dims, affine->dims(), normalizedDimCount, "fprop");
        }
        const auto* stat = (mean != nullptr) ? mean : rstd;
        if(stat != nullptr)
        {
            validateStatShape(dims, stat->dims(), normalizedDimCount, "fprop");
        }

        const auto normCount = static_cast<size_t>(normalizedDimCount);
        const size_t batchDimCount = dims.size() - normCount;
        const auto split = dims.begin() + static_cast<std::ptrdiff_t>(batchDimCount);
        std::vector<int64_t> batchExtents(dims.begin(), split);
        const std::vector<int64_t> normExtents(split, dims.end());

        // A whole-tensor normalization has a single batch position, indexed by no dimension.
        if(batchExtents.empty())
        {
            batchExtents.push_back(1);
        }

        const auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        const XDataType* xBase = x.memory().hostData();
        YDataType* yBase = y.memory().hostData();
        const ScaleBiasDataType* scaleBase
            = (scale != nullptr) ? scale->memory().hostData() : nullptr;
        const ScaleBiasDataType* biasBase = (bias != nullptr) ? bias->memory().hostData() : nullptr;
        MeanRstdDataType* meanBase = (mean != nullptr) ? mean->memory().hostData() : nullptr;
        MeanRstdDataType* rstdBase = (rstd != nullptr) ? rstd->memory().hostData() : nullptr;

        const auto& xStrides = x.strides();
        const auto& yStrides = y.strides();

        // Every batch position walks the normalized dims the same way, so that walk is a
        // flat offset table built once. x and y cover the normalized dims with their
        // trailing strides; scale and bias with their trailing strides too, since any
        // leading dims they have are 1s.
        const auto xNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, xStrides.data() + batchDimCount);
        const auto yNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, yStrides.data() + batchDimCount);
        const auto scaleOffsets = (scale != nullptr)
                                      ? hipdnn_test_sdk::detail::buildDenseOffsets(
                                            normExtents, trailingStrides(*scale, normCount))
                                      : std::vector<int64_t>{};
        const auto biasOffsets = (bias != nullptr)
                                     ? hipdnn_test_sdk::detail::buildDenseOffsets(
                                           normExtents, trailingStrides(*bias, normCount))
                                     : std::vector<int64_t>{};
        const int64_t* meanStrides = (mean != nullptr) ? mean->strides().data() : nullptr;
        const int64_t* rstdStrides = (rstd != nullptr) ? rstd->strides().data() : nullptr;

        const auto normElementCount = xNormOffsets.size();

        auto layernormFpropFunc = [&](const std::vector<int64_t>& batchIndices) {
            const int64_t xBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), xStrides.data(), batchDimCount);
            const int64_t yBatchOffset = hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), yStrides.data(), batchDimCount);

            // Pass 1: Welford's online algorithm for mean and variance
            const auto [batchMean, invStd] = welfordMeanAndRstd<ComputeDataType>(
                xBase + xBatchOffset, xNormOffsets, epsilonCompute);

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
                    batchIndices.data(), meanStrides, batchDimCount)]
                    = static_cast<MeanRstdDataType>(batchMean);
            }
            if(rstdBase != nullptr)
            {
                rstdBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), rstdStrides, batchDimCount)]
                    = static_cast<MeanRstdDataType>(invStd);
            }
        };

        // Parallelize over batch dimensions
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(layernormFpropFunc, batchExtents);
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
    // x and dx have the shape of dy.
    // Scale, dscale and dbias share one shape: the normalized dimensions, optionally
    // preceded by 1s.
    // Mean and rstd inputs, if provided, share one shape: the batch dimensions, optionally
    // followed by 1s, with at least one dimension.
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
                      const double epsilon,
                      const hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* mean,
                      const hipdnn_data_sdk::utilities::TensorBase<MeanRstdDataType>* rstd,
                      const int64_t normalizedDimCount)
    {
        const auto& dims = dy.dims();

        validateNormalizedDimCount(dims, normalizedDimCount, "bprop");

        if(x.dims() != dims || dx.dims() != dims)
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "bprop requires x and dx to have the same shape as dy.");
        }
        if(scale.dims() != dscale.dims() || scale.dims() != dbias.dims())
        {
            throw std::runtime_error(
                std::string(PREFIX)
                + "bprop requires scale, dscale and dbias to have the same shape.");
        }
        if((mean == nullptr) != (rstd == nullptr))
        {
            throw std::runtime_error(
                std::string(PREFIX)
                + "bprop requires both mean and rstd to be provided, or neither.");
        }
        if(mean != nullptr && mean->dims() != rstd->dims())
        {
            throw std::runtime_error(std::string(PREFIX)
                                     + "bprop requires mean and rstd to have the same shape.");
        }

        hipdnn_test_sdk::detail::validateNoRaggedTensor(dy, PREFIX, "dy");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(scale, PREFIX, "scale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dx, PREFIX, "dx");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dscale, PREFIX, "dscale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dbias, PREFIX, "dbias");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(mean, PREFIX, "mean");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(rstd, PREFIX, "rstd");

        validateAffineShape(dims, scale.dims(), normalizedDimCount, "bprop");
        if(mean != nullptr)
        {
            validateStatShape(dims, mean->dims(), normalizedDimCount, "bprop");
        }

        const auto normCount = static_cast<size_t>(normalizedDimCount);
        const size_t batchDimCount = dims.size() - normCount;
        const auto split = dims.begin() + static_cast<std::ptrdiff_t>(batchDimCount);
        std::vector<int64_t> batchExtents(dims.begin(), split);
        const std::vector<int64_t> normExtents(split, dims.end());

        const int64_t normalizedDimsSize = std::accumulate(
            normExtents.begin(), normExtents.end(), int64_t{1}, std::multiplies<int64_t>{});

        // A whole-tensor normalization has a single batch position, indexed by no dimension.
        if(batchExtents.empty())
        {
            batchExtents.push_back(1);
        }

        const auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

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
        const int64_t* meanStrides = (mean != nullptr) ? mean->strides().data() : nullptr;
        const int64_t* rstdStrides = (rstd != nullptr) ? rstd->strides().data() : nullptr;

        // Each pass holds one index space fixed and walks the other, and the walk is identical
        // every time, so both walks are flat offset tables built once. dy/x/dx cover the
        // normalized dims with their trailing strides and the batch dims with their leading
        // ones; scale covers the normalized dims with its trailing strides, since any leading
        // dims it has are 1s.
        const auto dyNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, dyStrides.data() + batchDimCount);
        const auto xNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, xStrides.data() + batchDimCount);
        const auto dxNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, dxStrides.data() + batchDimCount);
        const auto scaleNormOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            normExtents, trailingStrides(scale, normCount));

        const auto dyBatchOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(batchExtents, dyStrides.data());
        const auto xBatchOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(batchExtents, xStrides.data());

        const auto normElementCount = dyNormOffsets.size();
        const auto batchElementCount = dyBatchOffsets.size();

        // The statistics of every batch position, row-major over batchExtents: read from
        // mean/rstd when provided, recomputed from x otherwise. Pass 1 fills them and pass 2
        // reads them back in the order it walks the batch.
        std::vector<ComputeDataType> batchMean(batchElementCount);
        std::vector<ComputeDataType> batchRstd(batchElementCount);
        const auto batchRowMajorStrides = hipdnn_data_sdk::utilities::generateStrides(batchExtents);

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
            if(meanBase == nullptr)
            {
                std::tie(meanVal, rstdVal) = welfordMeanAndRstd<ComputeDataType>(
                    xBase + xBatchOffset, xNormOffsets, epsilonCompute);
            }
            else
            {
                meanVal = static_cast<ComputeDataType>(meanBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), meanStrides, batchDimCount)]);
                rstdVal = static_cast<ComputeDataType>(rstdBase[hipdnn_test_sdk::detail::flatOffset(
                    batchIndices.data(), rstdStrides, batchDimCount)]);
            }

            const auto statIdx = static_cast<size_t>(hipdnn_test_sdk::detail::flatOffset(
                batchIndices.data(), batchRowMajorStrides.data(), batchIndices.size()));
            batchMean[statIdx] = meanVal;
            batchRstd[statIdx] = rstdVal;

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
        const int64_t* dscaleStrides = trailingStrides(dscale, normCount);
        const int64_t* dbiasStrides = trailingStrides(dbias, normCount);
        auto layernormBpropWeightsFunc = [&](const std::vector<int64_t>& normIndices) {
            const int64_t dyNormOffset = hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), dyStrides.data() + batchDimCount, normCount);
            const int64_t xNormOffset = hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), xStrides.data() + batchDimCount, normCount);

            auto dscaleVal = static_cast<ComputeDataType>(0.0);
            auto dbiasVal = static_cast<ComputeDataType>(0.0);
            for(size_t element = 0; element < batchElementCount; ++element)
            {
                auto dyVal
                    = static_cast<ComputeDataType>(dyBase[dyNormOffset + dyBatchOffsets[element]]);
                auto xVal
                    = static_cast<ComputeDataType>(xBase[xNormOffset + xBatchOffsets[element]]);
                dscaleVal += dyVal * (xVal - batchMean[element]) * batchRstd[element];
                dbiasVal += dyVal;
            }

            dscaleBase[hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), dscaleStrides, normCount)]
                = static_cast<ScaleBiasDataType>(dscaleVal);
            dbiasBase[hipdnn_test_sdk::detail::flatOffset(
                normIndices.data(), dbiasStrides, normCount)]
                = static_cast<ScaleBiasDataType>(dbiasVal);
        };

        // Parallelize pass 1 over the batch and pass 2 over the normalized dimensions
        auto parallelValuesFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            layernormBpropValuesFunc, batchExtents);
        parallelValuesFunc(std::thread::hardware_concurrency());
        auto parallelWeightsFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            layernormBpropWeightsFunc, normExtents);
        parallelWeightsFunc(std::thread::hardware_concurrency());

        dx.memory().markHostModified();
        dscale.memory().markHostModified();
        dbias.memory().markHostModified();
    }

private:
    static void validateNormalizedDimCount(const std::vector<int64_t>& dims,
                                           int64_t normalizedDimCount,
                                           const char* pass)
    {
        const auto ndim = static_cast<int64_t>(dims.size());
        if(ndim < 1)
        {
            throw std::runtime_error(std::string(PREFIX) + pass + " requires at least 1D tensor.");
        }
        if(normalizedDimCount < 1 || normalizedDimCount > ndim)
        {
            throw std::runtime_error(
                std::string(PREFIX)
                + "normalizedDimCount must be between 1 and the number of tensor dimensions.");
        }
    }

    // scale/bias shape: the trailing normalizedDimCount dims of x, optionally preceded by 1s.
    static void validateAffineShape(const std::vector<int64_t>& xDims,
                                    const std::vector<int64_t>& affineDims,
                                    int64_t normalizedDimCount,
                                    const char* pass)
    {
        const auto count = static_cast<std::ptrdiff_t>(normalizedDimCount);
        if(static_cast<std::ptrdiff_t>(affineDims.size()) < count
           || !std::equal(affineDims.end() - count, affineDims.end(), xDims.end() - count)
           || std::any_of(
               affineDims.begin(), affineDims.end() - count, [](int64_t d) { return d != 1; }))
        {
            throw std::runtime_error(std::string(PREFIX) + pass
                                     + " requires scale/bias to have the trailing "
                                       "normalizedDimCount dims of the input, optionally "
                                       "preceded by 1s.");
        }
    }

    // mean/rstd shape: the leading batch dims of x, optionally followed by 1s. A rank-0
    // tensor holds no element, so the shape keeps at least one dimension even when every
    // dimension of x is normalized.
    static void validateStatShape(const std::vector<int64_t>& xDims,
                                  const std::vector<int64_t>& statDims,
                                  int64_t normalizedDimCount,
                                  const char* pass)
    {
        const auto count = static_cast<std::ptrdiff_t>(xDims.size())
                           - static_cast<std::ptrdiff_t>(normalizedDimCount);
        if(statDims.empty() || static_cast<std::ptrdiff_t>(statDims.size()) < count
           || !std::equal(xDims.begin(), xDims.begin() + count, statDims.begin())
           || std::any_of(
               statDims.begin() + count, statDims.end(), [](int64_t d) { return d != 1; }))
        {
            throw std::runtime_error(std::string(PREFIX) + pass
                                     + " requires mean/rstd to have the leading batch dims of "
                                       "the input, optionally followed by 1s, and at least "
                                       "one dimension.");
        }
    }

    // Strides of the last `count` dims of a validated scale-shaped tensor.
    static const int64_t* trailingStrides(const hipdnn_data_sdk::utilities::ITensor& tensor,
                                          size_t count)
    {
        return tensor.strides().data() + (tensor.strides().size() - count);
    }

    // Welford's online mean and variance over one batch position's normalized elements.
    // Returns {mean, 1 / sqrt(variance + epsilon)}.
    template <class ComputeDataType, class XDataType>
    static std::pair<ComputeDataType, ComputeDataType> welfordMeanAndRstd(
        const XDataType* xBatch, const std::vector<int64_t>& xNormOffsets, ComputeDataType epsilon)
    {
        int64_t count = 0;
        auto mean = static_cast<ComputeDataType>(0.0);
        auto m2 = static_cast<ComputeDataType>(0.0);

        for(const auto offset : xNormOffsets)
        {
            auto xVal = static_cast<ComputeDataType>(xBatch[offset]);

            count++;
            auto delta = xVal - mean;
            mean += delta / static_cast<ComputeDataType>(count);
            auto delta2 = xVal - mean;
            m2 += delta * delta2;
        }

        auto variance = m2 / static_cast<ComputeDataType>(count);
        auto rstd
            = static_cast<ComputeDataType>(1.0) / hipdnn_data_sdk::types::sqrt(variance + epsilon);
        return {mean, rstd};
    }
};

} // namespace hipdnn_test_sdk::utilities
