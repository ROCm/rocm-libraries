// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <algorithm>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceMatmul
{
public:
    // Check if this CPU implementation supports the given node configuration
    static bool isApplicable(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
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
        const auto batchDims = rank - static_cast<int64_t>(K_BATCH_IDX);

        auto matmulFunc = [&](const std::vector<int64_t>& indices) {
            // C dims: [...batch..., M, N]
            const int64_t m = *(indices.rbegin() + K_M_IDX);
            const int64_t n = *(indices.rbegin() + K_N_IDX);

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

            // Set matrix indices from the last dimension to match the [..., M, K] and [..., K, N] shapes
            *(aIndices.rbegin() + K_M_IDX) = m;
            *(bIndices.rbegin() + K_N_IDX) = n;

            auto acc = static_cast<ComputeDataType>(0.f);
            const int64_t kDim = *(aDims.rbegin() + K_K_IDX_A);
            for(int64_t k = 0; k < kDim; ++k)
            {
                *(aIndices.rbegin() + K_K_IDX_A) = k;
                *(bIndices.rbegin() + K_K_IDX_B) = k;

                const ADataType aVal = a.getHostValue(aIndices);
                const BDataType bVal = b.getHostValue(bIndices);
                acc = acc
                      + (static_cast<ComputeDataType>(aVal) * static_cast<ComputeDataType>(bVal));
            }

            c.setHostValue(hipdnn_test_sdk::detail::safeConvert<CDataType>(acc), indices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(matmulFunc, c.dims());
        parallelFunc(std::thread::hardware_concurrency());

        c.memory().markHostModified();
    }

private:
    template <typename TA, typename TB, typename TC>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<TA>& a,
                              const hipdnn_data_sdk::utilities::TensorBase<TB>& b,
                              const hipdnn_data_sdk::utilities::TensorBase<TC>& c)
    {
        const auto& aDims = a.dims();
        const auto& bDims = b.dims();
        const auto& cDims = c.dims();

        // Matmul node requires A and B have the same rank
        if(aDims.size() != bDims.size() || aDims.size() != cDims.size())
        {
            throw std::invalid_argument("Matmul expects A, B, and C to have the same rank (A rank="
                                        + std::to_string(aDims.size())
                                        + ", B rank=" + std::to_string(bDims.size())
                                        + ", C rank=" + std::to_string(cDims.size()) + ")");
        }

        const auto rank = aDims.size();
        if(rank < K_BATCH_IDX)
        {
            throw std::invalid_argument("Matmul expects matrices with rank >= 2");
        }

        const auto batchDims = rank - K_BATCH_IDX;
        if(!validateBroadcastableBatchDims(batchDims, aDims, bDims, cDims))
        {
            throw std::invalid_argument("Matmul batch dimensions are not broadcast-compatible");
        }

        // Matrix dimensions:
        // A[..., M, K] x B[..., K, N] -> C[..., M, N]
        const int64_t mDim = *(aDims.rbegin() + K_M_IDX);
        const int64_t kDimA = *(aDims.rbegin() + K_K_IDX_A);
        const int64_t kDimB = *(bDims.rbegin() + K_K_IDX_B);
        const int64_t nDim = *(bDims.rbegin() + K_N_IDX);

        if(kDimA != kDimB)
        {
            throw std::invalid_argument("Matmul shape mismatch: A.K must equal B.K");
        }
        if((*(cDims.rbegin() + K_M_IDX)) != mDim || (*(cDims.rbegin() + K_N_IDX)) != nDim)
        {
            throw std::invalid_argument("Matmul shape mismatch: C must be [..., A.M, B.N]");
        }
    }

    static bool validateBroadcastableBatchDims(size_t batchDims,
                                               const std::vector<int64_t>& aDims,
                                               const std::vector<int64_t>& bDims,
                                               const std::vector<int64_t>& cDims)
    {
        for(size_t i = 0; i < batchDims; ++i)
        {
            const auto aDimVal = aDims[i];
            const auto bDimVal = bDims[i];
            const auto cDimVal = cDims[i];

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

    // Indexes for the matrix dimensions starting from the last dimension
    constexpr static size_t K_BATCH_IDX = 2;
    constexpr static size_t K_M_IDX = 1;
    constexpr static size_t K_K_IDX_A = 0;
    constexpr static size_t K_K_IDX_B = 1;
    constexpr static size_t K_N_IDX = 0;
};

} // namespace hipdnn_test_sdk::utilities
