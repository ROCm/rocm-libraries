// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for the depthwise causal 1-D convolution (causal_conv1d_attributes.fbs).
///
/// `x` `[T, C]` holds N token-packed sequences with offsets `cuSeqlens` `[N + 1]`
/// (0 = cu[0] <= cu[1] <= ... <= cu[N] = T). For sequence n and channel c, let xs be the
/// W - 1 state columns followed by the sequence's tokens (W = weight `[C, W]` last dim):
///   `y[t, c] = act(bias[c] + sum_{j < W} weight[c, j] * xs[t + j, c])`
///   `finalState[row(n), c, :]` = the last W - 1 entries of xs
/// `weight[c, W - 1]` multiplies the current token. The state is `[N, C, W - 1]`, oldest
/// column first; an absent `initialState` means zeros. With `stateIndices` `[N]`,
/// `row(n) = stateIndices[n]` addresses a pool `[P, C, W - 1]`; otherwise `row(n) = n`.
/// Every initial state is read before any final state is written, so `finalState` may
/// alias `initialState`'s buffer. Sums run in `ComputeDataType`; SILU is `v * sigmoid(v)`.
class CpuFpReferenceCausalConv1d
{
public:
    using Activation = hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation;

    /// Keeps the optional tensor pointers out of template deduction, so `nullptr` binds.
    template <class T>
    using NonDeduced = std::enable_if_t<true, T>;

    template <class DataType,
              class SeqlensDataType,
              class StateIndexDataType = int32_t,
              class ComputeDataType = float>
    static void forward(
        const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>& weight,
        const hipdnn_data_sdk::utilities::TensorBase<SeqlensDataType>& cuSeqlens,
        hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
        Activation activation = Activation::IDENTITY,
        NonDeduced<const hipdnn_data_sdk::utilities::TensorBase<DataType>*> bias = nullptr,
        NonDeduced<const hipdnn_data_sdk::utilities::TensorBase<DataType>*> initialState = nullptr,
        NonDeduced<const hipdnn_data_sdk::utilities::TensorBase<StateIndexDataType>*> stateIndices
        = nullptr,
        NonDeduced<hipdnn_data_sdk::utilities::TensorBase<DataType>*> finalState = nullptr)
    {
        validateShapes(x, weight, cuSeqlens, y, bias, initialState, stateIndices, finalState);

        const int64_t tokens = x.dims()[0];
        const int64_t channels = x.dims()[1];
        const int64_t width = weight.dims()[1];
        const int64_t stateWidth = width - 1;
        const int64_t sequences = cuSeqlens.dims()[0] - 1;

        const auto offsets = readOffsets(cuSeqlens, sequences, tokens);
        const auto rows = stateRows(stateIndices,
                                    initialState,
                                    finalState,
                                    sequences,
                                    initialState != nullptr ? initialState->dims()[0] : sequences);

        // Read every incoming state before anything is written: finalState may alias it.
        const auto stateIndex = [channels, stateWidth](int64_t n, int64_t c, int64_t k) {
            return static_cast<size_t>((n * channels + c) * stateWidth + k);
        };
        std::vector<ComputeDataType> states(static_cast<size_t>(sequences * channels * stateWidth),
                                            ComputeDataType(0));
        if(initialState != nullptr)
        {
            for(int64_t n = 0; n < sequences; ++n)
            {
                const auto row = rows[static_cast<size_t>(n)];
                for(int64_t c = 0; c < channels; ++c)
                {
                    for(int64_t k = 0; k < stateWidth; ++k)
                    {
                        states[stateIndex(n, c, k)]
                            = static_cast<ComputeDataType>(initialState->getHostValue(row, c, k));
                    }
                }
            }
        }

        std::vector<ComputeDataType> finals(finalState != nullptr ? states.size() : 0);
        for(int64_t n = 0; n < sequences; ++n)
        {
            const int64_t start = offsets[static_cast<size_t>(n)];
            const int64_t length = offsets[static_cast<size_t>(n) + 1] - start;
            for(int64_t c = 0; c < channels; ++c)
            {
                // xs: stateWidth state columns, then the sequence's tokens.
                const auto xs = [&](int64_t i) {
                    return i < stateWidth ? states[stateIndex(n, c, i)]
                                          : static_cast<ComputeDataType>(
                                                x.getHostValue(start + i - stateWidth, c));
                };
                const auto biasValue = bias != nullptr
                                           ? static_cast<ComputeDataType>(bias->getHostValue(c))
                                           : ComputeDataType(0);
                for(int64_t t = 0; t < length; ++t)
                {
                    auto sum = biasValue;
                    for(int64_t j = 0; j < width; ++j)
                    {
                        sum += static_cast<ComputeDataType>(weight.getHostValue(c, j)) * xs(t + j);
                    }
                    y.setHostValue(static_cast<DataType>(activate(sum, activation)), start + t, c);
                }
                if(finalState != nullptr)
                {
                    for(int64_t k = 0; k < stateWidth; ++k)
                    {
                        finals[stateIndex(n, c, k)] = xs(length + k);
                    }
                }
            }
        }
        y.memory().markHostModified();

        if(finalState != nullptr)
        {
            for(int64_t n = 0; n < sequences; ++n)
            {
                const auto row = rows[static_cast<size_t>(n)];
                for(int64_t c = 0; c < channels; ++c)
                {
                    for(int64_t k = 0; k < stateWidth; ++k)
                    {
                        finalState->setHostValue(
                            static_cast<DataType>(finals[stateIndex(n, c, k)]), row, c, k);
                    }
                }
            }
            finalState->memory().markHostModified();
        }
    }

private:
    template <class ComputeDataType>
    static ComputeDataType activate(ComputeDataType value, Activation activation)
    {
        if(activation == Activation::SILU)
        {
            const auto v = static_cast<float>(value);
            return static_cast<ComputeDataType>(v / (1.0F + std::exp(-v)));
        }
        return value;
    }

    template <class SeqlensDataType>
    static std::vector<int64_t>
        readOffsets(const hipdnn_data_sdk::utilities::TensorBase<SeqlensDataType>& cuSeqlens,
                    int64_t sequences,
                    int64_t tokens)
    {
        std::vector<int64_t> offsets(static_cast<size_t>(sequences) + 1);
        for(int64_t n = 0; n <= sequences; ++n)
        {
            offsets[static_cast<size_t>(n)] = static_cast<int64_t>(cuSeqlens.getHostValue(n));
        }
        if(offsets.front() != 0 || offsets.back() != tokens)
        {
            throw std::invalid_argument(
                "CpuFpReferenceCausalConv1d: cu_seqlens must start at 0 and end at T");
        }
        for(size_t n = 1; n < offsets.size(); ++n)
        {
            if(offsets[n] < offsets[n - 1])
            {
                throw std::invalid_argument(
                    "CpuFpReferenceCausalConv1d: cu_seqlens must be non-decreasing");
            }
        }
        return offsets;
    }

    /// State row of each sequence: stateIndices[n] when given, else n. Rows must lie in
    /// [0, stateRowCount), and the rows written to finalState must be distinct.
    template <class DataType, class StateIndexDataType>
    static std::vector<int64_t>
        stateRows(const hipdnn_data_sdk::utilities::TensorBase<StateIndexDataType>* stateIndices,
                  const hipdnn_data_sdk::utilities::TensorBase<DataType>* initialState,
                  const hipdnn_data_sdk::utilities::TensorBase<DataType>* finalState,
                  int64_t sequences,
                  int64_t stateRowCount)
    {
        std::vector<int64_t> rows(static_cast<size_t>(sequences));
        for(int64_t n = 0; n < sequences; ++n)
        {
            rows[static_cast<size_t>(n)]
                = stateIndices != nullptr ? static_cast<int64_t>(stateIndices->getHostValue(n)) : n;
        }
        if(initialState == nullptr && finalState == nullptr)
        {
            return rows;
        }
        std::vector<bool> written(static_cast<size_t>(stateRowCount), false);
        for(const auto row : rows)
        {
            if(row < 0 || row >= stateRowCount)
            {
                throw std::out_of_range("CpuFpReferenceCausalConv1d: state row "
                                        + std::to_string(row) + " is outside the "
                                        + std::to_string(stateRowCount) + " state rows");
            }
            if(finalState != nullptr)
            {
                if(written[static_cast<size_t>(row)])
                {
                    throw std::invalid_argument("CpuFpReferenceCausalConv1d: state row "
                                                + std::to_string(row)
                                                + " is written by two sequences");
                }
                written[static_cast<size_t>(row)] = true;
            }
        }
        return rows;
    }

    template <class DataType, class SeqlensDataType, class StateIndexDataType>
    static void validateShapes(
        const hipdnn_data_sdk::utilities::TensorBase<DataType>& x,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>& weight,
        const hipdnn_data_sdk::utilities::TensorBase<SeqlensDataType>& cuSeqlens,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>& y,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>* bias,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>* initialState,
        const hipdnn_data_sdk::utilities::TensorBase<StateIndexDataType>* stateIndices,
        const hipdnn_data_sdk::utilities::TensorBase<DataType>* finalState)
    {
        const auto& xDims = x.dims();
        const auto& wDims = weight.dims();
        const auto& cuDims = cuSeqlens.dims();
        if(xDims.size() != 2 || y.dims() != xDims)
        {
            throw std::invalid_argument("CpuFpReferenceCausalConv1d: x and y must be [T, C]");
        }
        if(wDims.size() != 2 || wDims[0] != xDims[1] || wDims[1] < 2)
        {
            throw std::invalid_argument(
                "CpuFpReferenceCausalConv1d: weight must be [C, W] with W >= 2");
        }
        if(cuDims.size() != 1 || cuDims[0] < 2)
        {
            throw std::invalid_argument(
                "CpuFpReferenceCausalConv1d: cu_seqlens must be [N + 1] with N >= 1");
        }
        if(bias != nullptr && bias->dims() != std::vector<int64_t>{xDims[1]})
        {
            throw std::invalid_argument("CpuFpReferenceCausalConv1d: bias must be [C]");
        }
        const int64_t sequences = cuDims[0] - 1;
        if(stateIndices != nullptr)
        {
            if(initialState == nullptr)
            {
                throw std::invalid_argument(
                    "CpuFpReferenceCausalConv1d: state_indices requires initial_state");
            }
            if(stateIndices->dims() != std::vector<int64_t>{sequences})
            {
                throw std::invalid_argument(
                    "CpuFpReferenceCausalConv1d: state_indices must be [N]");
            }
        }
        const auto checkState
            = [&](const hipdnn_data_sdk::utilities::TensorBase<DataType>* state, const char* name) {
                  if(state == nullptr)
                  {
                      return;
                  }
                  const auto& sDims = state->dims();
                  const bool rowsOk = stateIndices != nullptr
                                          ? sDims.size() == 3 && sDims[0] >= 1
                                          : sDims.size() == 3 && sDims[0] == sequences;
                  if(!rowsOk || sDims[1] != xDims[1] || sDims[2] != wDims[1] - 1)
                  {
                      throw std::invalid_argument(std::string("CpuFpReferenceCausalConv1d: ") + name
                                                  + " must be [N or P, C, W - 1]");
                  }
              };
        checkState(initialState, "initial_state");
        checkState(finalState, "final_state");
        if(stateIndices != nullptr && finalState != nullptr
           && finalState->dims()[0] != initialState->dims()[0])
        {
            throw std::invalid_argument(
                "CpuFpReferenceCausalConv1d: final_state pool must match initial_state's");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
