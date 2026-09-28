// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Deterministic, always-valid FirstTokenOffset value for expert `e`: splits
/// `tokenRows` evenly across `experts` groups. Shared by the CPU-reference test
/// bundle and GPU integration test bundles so both synthesize identical
/// routing instead of maintaining separate copies of the same formula.
constexpr int32_t
    defaultMoeGroupedMatmulBwdRoutingOffset(int64_t e, int64_t tokenRows, int64_t experts)
{
    return static_cast<int32_t>((e * tokenRows) / experts);
}

/// Packed column-major strides `[K*N, 1, K]` for a `[experts, K, N]` DWeight — the
/// layout MoeGroupedMatmulBwdNode::infer_properties_node() assigns when the caller
/// leaves DWeight strides unset. DWeight itself accepts any strides: a caller that
/// sets them on the graph tensor keeps its own layout, and this reference honors
/// whatever layout the tensor carries. Use this helper only to match the inferred
/// default, since a host buffer must always agree with the strides the graph hands
/// the executor.
inline std::vector<int64_t>
    moeGroupedMatmulBwdDweightStrides(const std::vector<int64_t>& dweightDims)
{
    const std::vector<int64_t> columnMajorStrideOrder = {2, 0, 1};
    return hipdnn_data_sdk::utilities::generateStrides(dweightDims, columnMajorStrideOrder);
}

/// CPU reference for the backward MoE grouped matmul (NONE-mode routing only —
/// this schema has no gather/scatter fields, so group `g` always reduces exactly
/// expert `g`, and `FirstTokenOffset` row count must equal the expert count).
///
/// `FirstTokenOffset` gives the non-decreasing token-row boundaries per expert:
/// expert `e` owns rows `[FirstTokenOffset[e], FirstTokenOffset[e+1])` (or up to
/// `tokenRows` for the last expert). For each expert:
///   `DWeight[e, k, n] = sum over r in the expert's row range of Token[r, k] * DOutput[r, n]`
class CpuFpReferenceMoeGroupedMatmulBwd
{
public:
    template <class DoutputDataType,
              class TokenDataType,
              class DweightDataType,
              class ComputeDataType = float>
    static void backward(const hipdnn_data_sdk::utilities::TensorBase<DoutputDataType>& doutput,
                         const hipdnn_data_sdk::utilities::TensorBase<TokenDataType>& token,
                         const hipdnn_data_sdk::utilities::TensorBase<int32_t>& firstTokenOffset,
                         hipdnn_data_sdk::utilities::TensorBase<DweightDataType>& dweight)
    {
        validateInput(doutput, token, firstTokenOffset, dweight);

        const int64_t expertCount = dweight.dims()[0];
        const int64_t hiddenK = dweight.dims()[1];
        const int64_t outputN = dweight.dims()[2];
        const int64_t tokenRows = token.dims()[1];

        // Validate the whole offset table before consuming any of it, and append
        // tokenRows as a sentinel so the compute loop can read each expert's row
        // range as [offsets[expert], offsets[expert + 1]) uniformly, including the
        // last expert, without re-branching on expertCount per (expert, k) call.
        std::vector<int64_t> offsets(static_cast<size_t>(expertCount) + 1);
        int64_t previousOffset = 0;
        for(int64_t expert = 0; expert < expertCount; ++expert)
        {
            const int64_t offset = firstTokenOffset.getHostValue({expert, 0, 0});
            if(offset < 0 || offset > tokenRows)
            {
                throw std::runtime_error("CpuFpReferenceMoeGroupedMatmulBwd: FirstTokenOffset["
                                         + std::to_string(expert) + "] = " + std::to_string(offset)
                                         + " is out of range [0, " + std::to_string(tokenRows)
                                         + "]");
            }
            if(expert > 0 && offset < previousOffset)
            {
                throw std::runtime_error(
                    "CpuFpReferenceMoeGroupedMatmulBwd: FirstTokenOffset must be non-decreasing "
                    "(violated at expert "
                    + std::to_string(expert) + ")");
            }
            if(expert == 0 && offset != 0)
            {
                throw std::runtime_error(
                    "CpuFpReferenceMoeGroupedMatmulBwd: FirstTokenOffset[0] must be 0, got "
                    + std::to_string(offset)
                    + " (rows before the first expert's range would be silently excluded)");
            }
            offsets[static_cast<size_t>(expert)] = offset;
            previousOffset = offset;
        }
        offsets[static_cast<size_t>(expertCount)] = tokenRows;

        // --- Compute, parallel over (expert, k) pairs. Each work item reduces over
        // its expert's token-row range and produces one full DWeight row of length
        // outputN. Raw pointers/strides are hoisted once; the inner loops never
        // call getHostValue/setHostValue.
        const TokenDataType* tokenBase = token.memory().hostData();
        const DoutputDataType* doutputBase = doutput.memory().hostData();
        DweightDataType* dweightBase = dweight.memory().hostData();

        const auto& tokenStrides = token.strides();
        const auto& doutputStrides = doutput.strides();
        const auto& dweightStrides = dweight.strides();

        const int64_t tokenStride1 = tokenStrides[1];
        const int64_t tokenStride2 = tokenStrides[2];
        const int64_t doutputStride1 = doutputStrides[1];
        const int64_t doutputStride2 = doutputStrides[2];
        const int64_t dweightStride0 = dweightStrides[0];
        const int64_t dweightStride1 = dweightStrides[1];
        const int64_t dweightStride2 = dweightStrides[2];

        auto dweightRowFunc = [&](const std::vector<int64_t>& indices) {
            const int64_t expert = indices[0];
            const int64_t k = indices[1];

            const int64_t start = offsets[static_cast<size_t>(expert)];
            const int64_t end = offsets[static_cast<size_t>(expert + 1)];

            DweightDataType* dweightRow
                = dweightBase + expert * dweightStride0 + k * dweightStride1;

            // Accumulate the expert's token-row range into a contiguous scratch
            // row, then scatter it out through dweight's actual stride (unit for
            // row-major {K*N, N, 1}, K for column-major {K*N, 1, K}). thread_local
            // keeps the scratch buffer off the per-work-item allocation path - this
            // functor body runs once per (expert, k) pair, on every worker thread.
            thread_local std::vector<ComputeDataType> s_acc;
            s_acc.assign(static_cast<size_t>(outputN), ComputeDataType{0});

            for(int64_t r = start; r < end; ++r)
            {
                const TokenDataType* tokRow = tokenBase + r * tokenStride1;
                const auto a = static_cast<ComputeDataType>(tokRow[k * tokenStride2]);
                const DoutputDataType* doutRow = doutputBase + r * doutputStride1;
                for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
                {
                    s_acc[static_cast<size_t>(nIdx)]
                        += a * static_cast<ComputeDataType>(doutRow[nIdx * doutputStride2]);
                }
            }

            for(int64_t nIdx = 0; nIdx < outputN; ++nIdx)
            {
                dweightRow[nIdx * dweightStride2]
                    = hipdnn_test_sdk::detail::safeConvert<DweightDataType>(
                        s_acc[static_cast<size_t>(nIdx)]);
            }
        };

        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(
            dweightRowFunc, {expertCount, hiddenK});
        parallelFunc(std::thread::hardware_concurrency());

        dweight.memory().markHostModified();
    }

private:
    template <class DoutputDataType, class TokenDataType, class DweightDataType>
    static void
        validateInput(const hipdnn_data_sdk::utilities::TensorBase<DoutputDataType>& doutput,
                      const hipdnn_data_sdk::utilities::TensorBase<TokenDataType>& token,
                      const hipdnn_data_sdk::utilities::TensorBase<int32_t>& firstTokenOffset,
                      const hipdnn_data_sdk::utilities::TensorBase<DweightDataType>& dweight)
    {
        const std::string prefix = "CpuFpReferenceMoeGroupedMatmulBwd: ";

        hipdnn_test_sdk::detail::validateNoRaggedTensor(doutput, prefix, "doutput");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(token, prefix, "token");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(
            firstTokenOffset, prefix, "first_token_offset");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(dweight, prefix, "dweight");

        if(doutput.dims().size() != 3 || token.dims().size() != 3
           || firstTokenOffset.dims().size() != 3 || dweight.dims().size() != 3)
        {
            throw std::runtime_error(
                prefix + "doutput, token, first_token_offset and dweight must all be rank 3");
        }
        if(doutput.dims()[0] != 1)
        {
            throw std::runtime_error(prefix + "doutput must have a singleton leading dimension");
        }
        if(token.dims()[0] != 1)
        {
            throw std::runtime_error(prefix + "token must have a singleton leading dimension");
        }
        if(firstTokenOffset.dims()[1] != 1 || firstTokenOffset.dims()[2] != 1)
        {
            throw std::runtime_error(prefix + "first_token_offset must have shape [E, 1, 1]");
        }

        const int64_t expertCount = dweight.dims()[0];
        if(expertCount <= 0)
        {
            throw std::runtime_error(prefix + "expert count (dweight.dims[0]) must be positive");
        }
        if(firstTokenOffset.dims()[0] != expertCount)
        {
            throw std::runtime_error(prefix
                                     + "first_token_offset row count must equal the expert count");
        }

        if(token.dims()[2] != dweight.dims()[1])
        {
            throw std::runtime_error(prefix
                                     + "token hidden size (dims[2]) must equal dweight.dims[1]");
        }
        if(doutput.dims()[2] != dweight.dims()[2])
        {
            throw std::runtime_error(prefix + "doutput width (dims[2]) must equal dweight.dims[2]");
        }
        if(doutput.dims()[1] != token.dims()[1])
        {
            throw std::runtime_error(
                prefix + "doutput row count (dims[1]) must equal token row count (dims[1])");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
