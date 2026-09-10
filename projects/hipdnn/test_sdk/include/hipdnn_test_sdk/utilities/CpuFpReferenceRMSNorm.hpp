// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceRMSNorm
{
    static constexpr auto PREFIX = "CpuFpReferenceRMSNorm: ";

    static void computeLeadingAndReductionDims(const std::vector<int64_t>& xDims,
                                               const std::vector<int64_t>& scaleDims,
                                               std::vector<int64_t>& leadingDims,
                                               std::vector<int64_t>& reductionDims)
    {
        const size_t rank = xDims.size();

        // Normalized shape = maximal trailing suffix of dims where scale[i] == input[i].
        // Clamp so batch is always leading (never normalized).
        // matchCount = number of trailing dims where scaleDims[i] == xDims[i]
        const auto [scaleMismatch, _]
            = std::mismatch(scaleDims.rbegin(), scaleDims.rend(), xDims.rbegin(), xDims.rend());
        const auto matchCount
            = static_cast<size_t>(std::distance(scaleDims.rbegin(), scaleMismatch));
        const size_t reductionStart = (matchCount >= rank) ? 1 : rank - matchCount;
        if(reductionStart == rank)
        {
            // Validator should have rejected this; defensive guard for direct callers.
            throw std::runtime_error("RMSNorm: scale has no trailing dims matching input — no "
                                     "normalized axes can be derived.");
        }

        // Leading dims: [0, reductionStart) — batch + leading dims preserved through
        // the op (these are "kept" at full size in invRms).
        // Reduction dims: [reductionStart, rank) — normalized, collapsed per leading position.
        const auto splitOffset = static_cast<std::ptrdiff_t>(reductionStart);
        leadingDims.assign(xDims.begin(), xDims.begin() + splitOffset);
        reductionDims.assign(xDims.begin() + splitOffset, xDims.end());
    }

public:
    /// RMSNorm forward: y = x / RMS(x) * scale [+ bias]
    ///
    /// Normalized axes are derived from the scale tensor's shape: the non-1 dims
    /// of scale form a contiguous trailing suffix matching input, and those dims
    /// are normalized over (analogous to PyTorch's `normalized_shape`, encoded
    /// implicitly via the scale tensor). Dims where scale is 1 are the leading
    /// dims — they are "kept" in invRms (preserved at full size) while the
    /// normalized dims collapse to 1. Examples:
    ///   input [N, C, H, W], scale [1, C, H, W] → normalize over (C, H, W),
    ///                                           invRms shape [N, 1, 1, 1]
    ///   input [N, C, H, W], scale [1, 1, H, W] → normalize over (H, W),
    ///                                           invRms shape [N, C, 1, 1]
    ///   input [N, C, H, W], scale [1, 1, 1, W] → normalize over (W),
    ///                                           invRms shape [N, C, H, 1]
    ///
    /// @param x           Input tensor
    /// @param scale       Scale tensor, same rank as x; dim 0 = 1 (batch broadcast);
    ///                    non-1 dims must form a trailing suffix matching input.
    /// @param y           Output tensor (same shape as x)
    /// @param epsilon     Small scalar for numerical stability
    /// @param invRms      Optional output: 1 / RMS(x); shape = input with normalized dims → 1
    /// @param bias        Optional bias tensor, same shape as scale
    template <class XDataType, class ScaleDataType, class YDataType, class ComputeDataType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                        hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                        double epsilon,
                        hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>* invRms = nullptr,
                        const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* bias = nullptr)
    {
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.size() < 2)
        {
            throw std::runtime_error("RMSNorm forward requires at least 2D input tensor (batch and "
                                     "at least one feature dim).");
        }
        if(scaleDims.size() != xDims.size())
        {
            throw std::runtime_error("RMSNorm forward requires scale rank to equal input rank.");
        }

        // The kernel below addresses memory through hoisted base pointers and flat offset
        // tables, which assumes a dense layout; a ragged tensor rebases each batch at its
        // own offset and would silently read/write the wrong element.
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(scale, PREFIX, "scale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, PREFIX, "y");
        if(invRms != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*invRms, PREFIX, "invRms");
        }
        if(bias != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*bias, PREFIX, "bias");
        }

        // Compute leading and reduction dims based on input and scale shapes
        std::vector<int64_t> leadingDims;
        std::vector<int64_t> reductionDims;
        computeLeadingAndReductionDims(xDims, scaleDims, leadingDims, reductionDims);
        const auto reductionStart = leadingDims.size();

        const auto reductionCount = std::accumulate(
            reductionDims.begin(), reductionDims.end(), int64_t{1}, std::multiplies<>{});
        const auto reductionCountCompute = static_cast<ComputeDataType>(reductionCount);
        const auto epsilonCompute = static_cast<ComputeDataType>(epsilon);

        // Raw pointers are hoisted once, outside the parallel functor. The inner loops
        // never call getHostValue/setHostValue: each of those costs a virtual memory()
        // call plus a stride reduction over a freshly allocated index vector, which
        // dominates runtime on allocators without a per-thread cache (the Windows heap).
        const XDataType* xBase = x.memory().hostData();
        const ScaleDataType* scaleBase = scale.memory().hostData();
        YDataType* yBase = y.memory().hostData();
        ComputeDataType* invRmsBase = (invRms != nullptr) ? invRms->memory().hostData() : nullptr;
        const ScaleDataType* biasBase = (bias != nullptr) ? bias->memory().hostData() : nullptr;

        const auto& xStrides = x.strides();
        const auto& yStrides = y.strides();
        const auto& scaleStrides = scale.strides();

        // Reduction-region offset tables: one flat offset per reduction position, built
        // once and shared by every leading position. scale/bias need only this table -
        // their leading-region strides never contribute, since scaleIdx is zero there.
        // Running example: input [N, C, H, W], scale [1, 1, H, W]
        //   leadingDims = [N, C]     reductionDims = [H, W]     redIdx = [h, w]
        //   xRedOffsets[i] / yRedOffsets[i]   = flat offset of (h, w) in x's / y's H,W strides
        //   scaleRedOffsets[i] / biasRedOffsets[i] = flat offset of (h, w) in scale's / bias's
        //                                            H,W strides
        const std::vector<int64_t> xRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, xStrides.data() + reductionStart);
        const std::vector<int64_t> yRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, yStrides.data() + reductionStart);
        const std::vector<int64_t> scaleRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, scaleStrides.data() + reductionStart);
        std::vector<int64_t> biasRedOffsets;
        if(bias != nullptr)
        {
            biasRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
                reductionDims, bias->strides().data() + reductionStart);
        }
        const size_t redCount = xRedOffsets.size();

        // Compute RMS-normalized output for one leading position.
        //
        // Running example (continued): leadingIdx = [n, c]
        //   xLeadOffset / yLeadOffset = flat offset of (n, c) in x's / y's N,C strides
        //   x's element at redIdx i   = xBase[xLeadOffset + xRedOffsets[i]]
        //   scale's/bias's element    = scaleBase[scaleRedOffsets[i]] (no leading component)
        //   invRms's element          = invRmsBase[flat offset of (n, c) in invRms's N,C strides]
        auto rmsnormFwdFunc = [&](const std::vector<int64_t>& leadingIdx) {
            const int64_t xLeadOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), xStrides.data(), reductionStart);

            // Pass 1: accumulate sum(x^2) over the reduction dims at this leading position.
            //   invRms[n, c, 0, 0] = 1 / sqrt(mean(x^2 over [h, w]) + epsilon)
            auto sumSquares = static_cast<ComputeDataType>(0.0);
            for(size_t i = 0; i < redCount; ++i)
            {
                const auto inVal
                    = static_cast<ComputeDataType>(xBase[xLeadOffset + xRedOffsets[i]]);
                sumSquares += inVal * inVal;
            }

            const auto meanSquares = sumSquares / reductionCountCompute;
            const auto invRmsValue = static_cast<ComputeDataType>(1.0)
                                     / hipdnn_data_sdk::types::sqrt(meanSquares + epsilonCompute);

            // Pass 2: write y = scale * x * invRms (+ bias), walking the same reduction
            // offsets but now also reading scale/bias at their leading-zeroed positions.
            //   y[n, c, h, w] = scale[0, 0, h, w] * x[n, c, h, w] * invRms[n, c, 0, 0]
            //                   (+ bias[0, 0, h, w])
            const int64_t yLeadOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), yStrides.data(), reductionStart);
            for(size_t i = 0; i < redCount; ++i)
            {
                const auto xVal = static_cast<ComputeDataType>(xBase[xLeadOffset + xRedOffsets[i]]);
                const auto xNorm = xVal * invRmsValue;
                ComputeDataType yVal
                    = static_cast<ComputeDataType>(scaleBase[scaleRedOffsets[i]]) * xNorm;
                if(bias != nullptr)
                {
                    yVal += static_cast<ComputeDataType>(biasBase[biasRedOffsets[i]]);
                }
                yBase[yLeadOffset + yRedOffsets[i]] = static_cast<YDataType>(yVal);
            }

            // invRms keeps the leading dims and collapses the reduction dims to 1, so its
            // offset has only a leading component - no reduction-walk table needed.
            // Running example: invRms[n, c, 0, 0].
            if(invRms != nullptr)
            {
                const int64_t invRmsOffset = hipdnn_test_sdk::detail::flatOffset(
                    leadingIdx.data(), invRms->strides().data(), reductionStart);
                invRmsBase[invRmsOffset] = static_cast<ComputeDataType>(invRmsValue);
            }
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormFwdFunc, leadingDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
        if(invRms != nullptr)
        {
            invRms->memory().markHostModified();
        }
    }

    template <class DyDataType,
              class XDataType,
              class ScaleDataType,
              class DxDataType = XDataType,
              class ComputeDataType = float>
    static void backward(const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& dy,
                         const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                         const hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& scale,
                         const hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>& invRms,
                         hipdnn_data_sdk::utilities::TensorBase<DxDataType>& dx,
                         hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>& dscale,
                         hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>* dbias = nullptr)
    {
        const auto& xDims = x.dims();
        const auto& scaleDims = scale.dims();

        if(xDims.size() < 2)
        {
            throw std::runtime_error(
                "RMSNorm backward requires at least 2D input tensor (batch and "
                "at least one feature dim).");
        }

        if(dy.dims().size() != xDims.size() || scaleDims.size() != xDims.size()
           || invRms.dims().size() != xDims.size())
        {
            throw std::runtime_error("RMSNorm backward requires dy, scale, and invRms to all have "
                                     "the same rank as input.");
        }

        // The kernel below addresses memory through hoisted base pointers and flat offset
        // tables, which assumes a dense layout; a ragged tensor rebases each batch at its
        // own offset and would silently read/write the wrong element.
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dy, PREFIX, "dy");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(scale, PREFIX, "scale");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(invRms, PREFIX, "invRms");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dx, PREFIX, "dx");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dscale, PREFIX, "dscale");
        if(dbias != nullptr)
        {
            hipdnn_test_sdk::detail::validateNoRaggedTensor(*dbias, PREFIX, "dbias");
        }

        // Compute leading and reduction dims based on input and scale shapes
        std::vector<int64_t> leadingDims;
        std::vector<int64_t> reductionDims;
        computeLeadingAndReductionDims(xDims, scaleDims, leadingDims, reductionDims);
        const auto reductionStart = leadingDims.size();

        const auto reductionCount = std::accumulate(
            reductionDims.begin(), reductionDims.end(), int64_t{1}, std::multiplies<>{});
        const auto reductionCountCompute = static_cast<ComputeDataType>(reductionCount);

        // Raw pointers and strides are hoisted once; see forward() for why.
        const DyDataType* dyBase = dy.memory().hostData();
        const XDataType* xBase = x.memory().hostData();
        const ScaleDataType* scaleBase = scale.memory().hostData();
        const ComputeDataType* invRmsBase = invRms.memory().hostData();
        DxDataType* dxBase = dx.memory().hostData();
        ScaleDataType* dscaleBase = dscale.memory().hostData();
        ScaleDataType* dbiasBase = (dbias != nullptr) ? dbias->memory().hostData() : nullptr;

        const auto& dyStrides = dy.strides();
        const auto& xStrides = x.strides();
        const auto& scaleStrides = scale.strides();
        const auto& invRmsStrides = invRms.strides();
        const auto& dxStrides = dx.strides();
        const auto& dscaleStrides = dscale.strides();

        // Leading-region offset tables: one flat offset per leading position, built once
        // and shared by every reduction position walked in this functor. Roles are
        // inverted from forward()/rmsnormBwdDataFunc below - this functor is parallelized
        // over reductionDims and walks leadingDims internally, so it is the leading walk
        // (not the reduction walk) that benefits from a shared table.
        // Running example: input [N, C, H, W], scale [1, 1, H, W]
        //   leadingDims = [N, C]     reductionDims = [H, W]
        //   dyLeadOffsets[i] / xLeadOffsets[i] / invRmsLeadOffsets[i] = flat offset of the
        //     i-th (n, c) position in dy's / x's / invRms's N,C strides
        const std::vector<int64_t> dyLeadOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(leadingDims, dyStrides.data());
        const std::vector<int64_t> xLeadOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(leadingDims, xStrides.data());
        const std::vector<int64_t> invRmsLeadOffsets
            = hipdnn_test_sdk::detail::buildDenseOffsets(leadingDims, invRmsStrides.data());
        const size_t leadCount = dyLeadOffsets.size();

        // redIdx = [h, w] is fixed for the whole functor call, so its contribution to the
        // dy/x/dscale/dbias offsets (dyRedOffset, xRedOffset, dscaleOffset, dbiasOffset
        // below) is a scalar computed once, not a table.
        auto rmsnormBwdWeightBiasFunc = [&](const std::vector<int64_t>& redIdx) {
            auto sumDScale = static_cast<ComputeDataType>(0.0);
            auto sumDBias = static_cast<ComputeDataType>(0.0);

            const int64_t dyRedOffset = hipdnn_test_sdk::detail::flatOffset(
                redIdx.data(), dyStrides.data() + reductionStart, reductionDims.size());
            const int64_t xRedOffset = hipdnn_test_sdk::detail::flatOffset(
                redIdx.data(), xStrides.data() + reductionStart, reductionDims.size());

            // Compute the reduction-sum components of dscale and dbias for this reduction position.
            for(size_t i = 0; i < leadCount; ++i)
            {
                const auto pdy
                    = static_cast<ComputeDataType>(dyBase[dyLeadOffsets[i] + dyRedOffset]);
                const auto px = static_cast<ComputeDataType>(xBase[xLeadOffsets[i] + xRedOffset]);
                const auto prstd = static_cast<ComputeDataType>(invRmsBase[invRmsLeadOffsets[i]]);

                sumDScale += pdy * (px * prstd);
                sumDBias += pdy;
            }

            // Write dscale and dbias for this reduction position.
            const int64_t dscaleOffset = hipdnn_test_sdk::detail::flatOffset(
                redIdx.data(), dscaleStrides.data() + reductionStart, reductionDims.size());
            dscaleBase[dscaleOffset] = static_cast<ScaleDataType>(sumDScale);
            if(dbias != nullptr)
            {
                const int64_t dbiasOffset = hipdnn_test_sdk::detail::flatOffset(
                    redIdx.data(), dbias->strides().data() + reductionStart, reductionDims.size());
                dbiasBase[dbiasOffset] = static_cast<ScaleDataType>(sumDBias);
            }
        };

        auto parallelWeightBiasFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            rmsnormBwdWeightBiasFunc, reductionDims);
        parallelWeightBiasFunc(std::thread::hardware_concurrency());
        dscale.memory().markHostModified();
        if(dbias != nullptr)
        {
            dbias->memory().markHostModified();
        }

        // Reduction-region offset tables for the data-gradient pass below, built once and
        // shared by every leading position (mirrors forward()'s tables).
        const std::vector<int64_t> dyRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, dyStrides.data() + reductionStart);
        const std::vector<int64_t> xRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, xStrides.data() + reductionStart);
        const std::vector<int64_t> scaleRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, scaleStrides.data() + reductionStart);
        const std::vector<int64_t> dxRedOffsets = hipdnn_test_sdk::detail::buildDenseOffsets(
            reductionDims, dxStrides.data() + reductionStart);
        const size_t redCount = dyRedOffsets.size();

        auto rmsnormBwdDataFunc = [&](const std::vector<int64_t>& leadingIdx) {
            auto meanDyXW = static_cast<ComputeDataType>(0.0);

            const int64_t dyLeadOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), dyStrides.data(), reductionStart);
            const int64_t xLeadOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), xStrides.data(), reductionStart);

            // Compute the reduction-sum component of dx = (dy * scale - meanDyXW * x) * invRms,
            // where meanDyXW = mean(dy * scale * x) over the reduction dims at this leading position.
            for(size_t i = 0; i < redCount; ++i)
            {
                const auto pdy
                    = static_cast<ComputeDataType>(dyBase[dyLeadOffset + dyRedOffsets[i]]);
                const auto px = static_cast<ComputeDataType>(xBase[xLeadOffset + xRedOffsets[i]]);
                const auto pw = static_cast<ComputeDataType>(scaleBase[scaleRedOffsets[i]]);

                meanDyXW += pdy * pw * px;
            }

            meanDyXW /= reductionCountCompute;

            // Get invRms for this leading position
            const int64_t invRmsOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), invRmsStrides.data(), reductionStart);
            const auto prstd = static_cast<ComputeDataType>(invRmsBase[invRmsOffset]);
            const auto invRmsCube = prstd * prstd * prstd;

            // Compute dx = (dy * scale - meanDyXW * x) * invRms
            const int64_t dxLeadOffset = hipdnn_test_sdk::detail::flatOffset(
                leadingIdx.data(), dxStrides.data(), reductionStart);
            for(size_t i = 0; i < redCount; ++i)
            {
                const auto pdy
                    = static_cast<ComputeDataType>(dyBase[dyLeadOffset + dyRedOffsets[i]]);
                const auto px = static_cast<ComputeDataType>(xBase[xLeadOffset + xRedOffsets[i]]);
                const auto pw = static_cast<ComputeDataType>(scaleBase[scaleRedOffsets[i]]);

                const auto dxVal = (pdy * pw * prstd) - (meanDyXW * px * invRmsCube);

                dxBase[dxLeadOffset + dxRedOffsets[i]] = static_cast<DxDataType>(dxVal);
            }
        };

        auto parallelDataFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rmsnormBwdDataFunc, leadingDims);
        parallelDataFunc(std::thread::hardware_concurrency());
        dx.memory().markHostModified();
    }
};

} // namespace hipdnn_test_sdk::utilities
