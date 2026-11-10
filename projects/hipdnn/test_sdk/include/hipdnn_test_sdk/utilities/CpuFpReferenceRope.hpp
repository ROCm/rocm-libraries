// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for rotary position embedding (RoPE) forward, rotate-half (non-interleaved).
/// With D the last dim of x, r = rope_dim (0 means D), h = r / 2, o = rope_offset (absent:
/// D - r), s = output_scale and f the angles of freqs broadcast to x (every freqs dim but the
/// last is x's or 1; the last is r), for 0 <= i < h:
///   y[o + i]     = s * (x[o + i] cos f[i]         - x[o + i + h] sin f[i])
///   y[o + i + h] = s * (x[o + i + h] cos f[i + h] + x[o + i] sin f[i + h])
/// and y[j] = s * x[j] outside the segment. Arithmetic runs in ComputeType. Each element is
/// read before its output is written, so y may alias x.
class CpuFpReferenceRope
{
public:
    template <class DataType, class ComputeType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                        const hipdnn_data_sdk::utilities::TensorBase<float>& freqs,
                        hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
                        float outputScale,
                        int64_t ropeDim,
                        std::optional<int64_t> ropeOffset)
    {
        const auto segment = resolveSegment(x, freqs, y, ropeDim, ropeOffset);
        const int64_t rotDim = segment.rotDim;
        const int64_t offset = segment.offset;

        const auto& xDims = x.dims();
        const auto rank = xDims.size();
        const int64_t headDim = xDims[rank - 1];
        const std::vector<int64_t> rowDims(xDims.begin(),
                                           xDims.begin() + static_cast<std::ptrdiff_t>(rank - 1));
        int64_t rowCount = 1;
        for(const auto dim : rowDims)
        {
            rowCount *= dim;
        }
        if(rowCount == 0 || headDim == 0)
        {
            return;
        }

        const DataType* xBase = x.memory().hostData();
        const float* freqsBase = freqs.memory().hostData();
        DataType* yBase = y.memory().hostData();
        const auto& xStrides = x.strides();
        const auto& yStrides = y.strides();
        const auto& freqsStrides = freqs.strides();
        const auto& freqsDims = freqs.dims();
        const auto scale = static_cast<ComputeType>(outputScale);
        const int64_t halfDim = rotDim / 2;
        const int64_t xLast = xStrides[rank - 1];
        const int64_t yLast = yStrides[rank - 1];
        const int64_t freqsLast = freqsStrides[rank - 1];

        auto rotateRow = [&](const std::vector<int64_t>& row) {
            int64_t xRow = 0;
            int64_t yRow = 0;
            int64_t freqsRow = 0;
            for(size_t d = 0; d < row.size(); ++d)
            {
                xRow += row[d] * xStrides[d];
                yRow += row[d] * yStrides[d];
                if(freqsDims[d] != 1)
                {
                    freqsRow += row[d] * freqsStrides[d];
                }
            }

            for(int64_t j = 0; j < headDim; ++j)
            {
                if(j < offset || j >= offset + rotDim)
                {
                    const auto value = static_cast<ComputeType>(xBase[xRow + j * xLast]);
                    yBase[yRow + j * yLast] = static_cast<DataType>(scale * value);
                }
            }

            for(int64_t i = 0; i < halfDim; ++i)
            {
                const int64_t lo = offset + i;
                const int64_t hi = lo + halfDim;
                const auto xLo = static_cast<ComputeType>(xBase[xRow + lo * xLast]);
                const auto xHi = static_cast<ComputeType>(xBase[xRow + hi * xLast]);
                const auto fLo = static_cast<ComputeType>(freqsBase[freqsRow + i * freqsLast]);
                const auto fHi
                    = static_cast<ComputeType>(freqsBase[freqsRow + (i + halfDim) * freqsLast]);

                const ComputeType yLo = scale * (xLo * std::cos(fLo) - xHi * std::sin(fLo));
                const ComputeType yHi = scale * (xHi * std::cos(fHi) + xLo * std::sin(fHi));
                yBase[yRow + lo * yLast] = static_cast<DataType>(yLo);
                yBase[yRow + hi * yLast] = static_cast<DataType>(yHi);
            }
        };

        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(rotateRow, rowDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }

private:
    struct Segment
    {
        int64_t rotDim;
        int64_t offset;
    };

    template <class DataType>
    static Segment resolveSegment(const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
                                  const hipdnn_data_sdk::utilities::TensorBase<float>& freqs,
                                  const hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
                                  int64_t ropeDim,
                                  std::optional<int64_t> ropeOffset)
    {
        const std::string prefix = "CpuFpReferenceRope: ";

        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, prefix, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(freqs, prefix, "freqs");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, prefix, "y");

        const auto& xDims = x.dims();
        const auto& freqsDims = freqs.dims();
        if(xDims.size() < 2)
        {
            throw std::runtime_error(prefix + "x must have rank >= 2");
        }
        if(freqsDims.size() != xDims.size())
        {
            throw std::runtime_error(prefix + "freqs must have the same rank as x");
        }
        if(y.dims() != xDims)
        {
            throw std::runtime_error(prefix + "y dims must equal x dims");
        }

        const int64_t headDim = xDims.back();
        if(ropeDim < 0)
        {
            throw std::runtime_error(prefix + "rope_dim must not be negative");
        }
        const int64_t rotDim = ropeDim == 0 ? headDim : ropeDim;
        if(rotDim <= 0 || rotDim > headDim || rotDim % 2 != 0)
        {
            throw std::runtime_error(prefix + "rope_dim " + std::to_string(rotDim)
                                     + " must be even and in (0, " + std::to_string(headDim) + "]");
        }
        const int64_t offset = ropeOffset.value_or(headDim - rotDim);
        if(offset < 0 || offset + rotDim > headDim)
        {
            throw std::runtime_error(prefix + "rope_offset " + std::to_string(offset)
                                     + " + rope_dim must be within the last dim of x");
        }

        if(freqsDims.back() != rotDim)
        {
            throw std::runtime_error(prefix + "freqs last dim must equal rope_dim");
        }
        for(size_t d = 0; d + 1 < xDims.size(); ++d)
        {
            if(freqsDims[d] != xDims[d] && freqsDims[d] != 1)
            {
                throw std::runtime_error(prefix + "freqs dim " + std::to_string(d)
                                         + " must equal x's or be 1");
            }
        }

        return {rotDim, offset};
    }
};

} // namespace hipdnn_test_sdk::utilities
