// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>

#include <algorithm>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

namespace matmul
{
// Broadcasting rule (matches MatmulNode.hpp):
// For each batch dim i, A[i] and B[i] are compatible if one divides the other.
// Output batch dim is max(A[i], B[i]).
template <typename ADims, typename BDims, typename CDims>
inline bool validateBatchBroadcastDims(size_t batchDims,
                                       const ADims& aDims,
                                       const BDims& bDims,
                                       const CDims& cDims)
{
    for(size_t i = 0; i < batchDims; ++i)
    {
        const auto aDimVal = static_cast<int64_t>(aDims[i]);
        const auto bDimVal = static_cast<int64_t>(bDims[i]);
        const auto cDimVal = static_cast<int64_t>(cDims[i]);

        if(aDimVal <= 0 || bDimVal <= 0 || cDimVal <= 0)
        {
            return false;
        }

        if(aDimVal % bDimVal != 0 && bDimVal % aDimVal != 0)
        {
            return false;
        }

        const int64_t expectedOut = std::max(aDimVal, bDimVal);
        if(cDimVal != expectedOut)
        {
            return false;
        }
    }

    return true;
}
} // namespace matmul

class CpuFpReferenceMatmul
{
public:
    // Check if this CPU implementation supports the given node configuration
    static bool isApplicable(const hipdnn_data_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_data_sdk::data_objects;
        return node.attributes_type() == NodeAttributes::MatmulAttributes;
    }

    template <class ADataType, class BDataType, class CDataType, class ComputeDataType = float>
    static void matmul(const hipdnn_data_sdk::utilities::TensorBase<ADataType>& a,
                       const hipdnn_data_sdk::utilities::TensorBase<BDataType>& b,
                       hipdnn_data_sdk::utilities::TensorBase<CDataType>& c)
    {
        validateInput(a, b, c);

        const auto& aDims = a.dims(); // [..., M, K]
        const auto& bDims = b.dims(); // [..., K, N]
        const auto& cDims = c.dims(); // [..., M, N]

        const auto rank = static_cast<int64_t>(cDims.size());
        const auto batchDims = rank - 2;

        auto computeElement = [&](const std::vector<int64_t>& indices) {
            // C dims: [...batch..., M, N]
            const int64_t m = indices[static_cast<size_t>(rank - 2)];
            const int64_t n = indices[static_cast<size_t>(rank - 1)];

            std::vector<int64_t> aIndices(static_cast<size_t>(rank));
            std::vector<int64_t> bIndices(static_cast<size_t>(rank));

            // Broadcasting for batch dims (divisibility rule):
            // outDim = C[dim] = max(A[dim], B[dim])
            // inIdx  = outIdx / (outDim / inDim)
            for(int64_t i = 0; i < batchDims; ++i)
            {
                const auto idx = static_cast<size_t>(i);
                const int64_t outDim = cDims[idx];

                const int64_t aScale = outDim / aDims[idx];
                const int64_t bScale = outDim / bDims[idx];

                aIndices[idx] = indices[idx] / aScale;
                bIndices[idx] = indices[idx] / bScale;
            }

            aIndices[static_cast<size_t>(rank - 2)] = m;
            bIndices[static_cast<size_t>(rank - 1)] = n;

            auto acc = static_cast<ComputeDataType>(0);
            const int64_t kDim = aDims[static_cast<size_t>(rank - 1)];
            for(int64_t k = 0; k < kDim; ++k)
            {
                aIndices[static_cast<size_t>(rank - 1)] = k;
                bIndices[static_cast<size_t>(rank - 2)] = k;

                const ADataType aVal = a.getHostValue(aIndices);
                const BDataType bVal = b.getHostValue(bIndices);
                acc = acc
                      + (static_cast<ComputeDataType>(aVal) * static_cast<ComputeDataType>(bVal));
            }

            c.setHostValue(safeConvert<CDataType>(acc), indices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::utilities::makeParallelTensorFunctor(computeElement, c.dims());
        parallelFunc(std::thread::hardware_concurrency());

        c.memory().markHostModified();
    }

private:
    template <typename TA, typename TB, typename TC>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<TA>& a,
                              const hipdnn_data_sdk::utilities::TensorBase<TB>& b,
                              const hipdnn_data_sdk::utilities::TensorBase<TC>& c)
    {
        // Matmul node requires A and B have the same rank
        const auto rankA = a.dims().size();
        const auto rankB = b.dims().size();
        const auto rankC = c.dims().size();
        if(rankA != rankB || rankA != rankC)
        {
            throw std::invalid_argument(
                "Matmul expects A, B, and C to have the same rank (A rank="
                + std::to_string(rankA) + ", B rank=" + std::to_string(rankB)
                + ", C rank=" + std::to_string(rankC) + ")");
        }
        if(rankA < 2)
        {
            throw std::invalid_argument("Matmul expects rank >= 2 tensors");
        }

        const auto& aDims = a.dims();
        const auto& bDims = b.dims();
        const auto& cDims = c.dims();

        // For each batch dim: A[i] and B[i] are compatible if one divides the other
        // Output batch dim is max(A[i], B[i])
        const auto batchDims = rankA - 2;
        if(!hipdnn_test_sdk::utilities::matmul::validateBatchBroadcastDims(
               batchDims, aDims, bDims, cDims))
        {
            throw std::invalid_argument("Matmul batch dimensions are not broadcast-compatible");
        }

        // Matrix dimensions:
        // A[..., M, K] x B[..., K, N] -> C[..., M, N]
        const int64_t mDim = aDims[rankA - 2];
        const int64_t kDim = aDims[rankA - 1];
        const int64_t bKDim = bDims[rankB - 2];
        const int64_t nDim = bDims[rankB - 1];

        if(kDim != bKDim)
        {
            throw std::invalid_argument("Matmul shape mismatch: A.K must equal B.K");
        }
        if(cDims[rankC - 2] != mDim || cDims[rankC - 1] != nDim)
        {
            throw std::invalid_argument("Matmul shape mismatch: C must be [..., A.M, B.N]");
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
