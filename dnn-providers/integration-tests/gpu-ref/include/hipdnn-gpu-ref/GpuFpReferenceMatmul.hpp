// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn-gpu-ref/detail/HipRtcTypeName.hpp"

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <stdexcept>

namespace hipdnn_gpu_ref
{

using namespace hipdnn_data_sdk::utilities;

namespace detail
{

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          unsigned int TileSize>
inline std::vector<std::string> buildMatmulDefines(const std::vector<int64_t>& aDims,
                                                   const std::vector<int64_t>& bDims)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DA_TYPE=") + HipRtcTypeName<ADataType>::VALUE);
    defines.emplace_back(std::string("-DB_TYPE=") + HipRtcTypeName<BDataType>::VALUE);
    defines.emplace_back(std::string("-DC_TYPE=") + HipRtcTypeName<CDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DMATMUL_K=") + std::to_string(aDims[aDims.size() - 1]));
    defines.emplace_back(std::string("-DMATMUL_M=") + std::to_string(aDims[aDims.size() - 2]));
    defines.emplace_back(std::string("-DMATMUL_N=") + std::to_string(bDims[bDims.size() - 1]));
    defines.emplace_back(std::string("-DMATMUL_BATCH_DIM_COUNT=")
                         + std::to_string(aDims.size() - 2));
    defines.emplace_back(std::string("-DTILE_SIZE=") + std::to_string(TileSize));
    return defines;
}

} // namespace detail

class GpuFpReferenceMatmul
{
public:
    static constexpr unsigned int TILE_SIZE = 32;

    // Matrix multiplication
    template <class ADataType, class BDataType, class CDataType, class ComputeDataType = float>
    static void matmul(TensorBase<ADataType>& a, TensorBase<BDataType>& b, TensorBase<CDataType>& c)
    {
        validateMatmul(a, b, c);

        auto defines = detail::
            buildMatmulDefines<ADataType, BDataType, CDataType, ComputeDataType, TILE_SIZE>(
                a.dims(), b.dims());

        launchMatmul(a.rawDeviceData(),
                     a.dims(),
                     a.strides(),
                     b.rawDeviceData(),
                     b.dims(),
                     b.strides(),
                     c.rawDeviceData(),
                     c.dims(),
                     c.strides(),
                     TILE_SIZE,
                     defines);

        c.markDeviceModified();
    }

private:
    // --- Validators ---

    template <class T>
    static constexpr bool IS_SUPPORTED_DATA_TYPE
        = std::is_same_v<T, float> || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    static void validateConsistentDimensions(const std::vector<int64_t>& aDims,
                                             const std::vector<int64_t>& bDims,
                                             const std::vector<int64_t>& cDims)
    {
        if(aDims.size() != bDims.size())
        {
            throw std::invalid_argument("Matmul requires A and B tensors to have the same rank.");
        }

        if(aDims.size() < 2 || aDims.size() > 5)
        {
            throw std::invalid_argument("Matmul requires A and B tensor ranks to be 2, 3, 4 or 5.");
        }

        std::vector<int64_t> aBatchDims(aDims.begin(), aDims.end() - 2);
        std::vector<int64_t> bBatchDims(bDims.begin(), bDims.end() - 2);
        for(size_t i = 0; i < aBatchDims.size(); ++i)
        {
            if(aBatchDims[i] % bBatchDims[i] != 0 && bBatchDims[i] % aBatchDims[i] != 0)
            {
                throw std::invalid_argument(
                    "Matmul requires A and B tensors to have broadcast-compatible batch dimensions "
                    "(all but the last two dimensions).");
            }
        }

        if(aDims[aDims.size() - 1] != bDims[bDims.size() - 2])
        {
            throw std::invalid_argument("Matmul requires K to match between A tensor (last "
                                        "dimension) and B tensor (second to last dimension).");
        }

        std::vector<int64_t> expectedCDims(aDims.size());
        for(size_t i = 0; i < aBatchDims.size(); ++i)
        {
            expectedCDims[i] = std::max(aBatchDims[i], bBatchDims[i]);
        }
        expectedCDims[expectedCDims.size() - 2] = aDims[aDims.size() - 2];
        expectedCDims[expectedCDims.size() - 1] = bDims[bDims.size() - 1];
        if(cDims != expectedCDims)
        {
            throw std::invalid_argument(
                "Matmul requires C tensor dimensions to match expected C tensor dimensions "
                "(broadcasted batch..., M, N) from A tensor dimensions (A batch..., M, K) and B "
                "tensor dimensions(B batch..., K, N).");
        }
    }

    static void validateConsistentLayouts(const std::vector<int64_t>& aStrides,
                                          const std::vector<int64_t>& bStrides,
                                          const std::vector<int64_t>& cStrides)
    {
        const auto aStrideOrder = extractStrideOrder(aStrides);
        const auto bStrideOrder = extractStrideOrder(bStrides);
        const auto cStrideOrder = extractStrideOrder(cStrides);

        if(aStrideOrder != bStrideOrder || aStrideOrder != cStrideOrder)
        {
            throw std::invalid_argument(
                "Matmul requires A, B and C tensors to have the same stride order.");
        }

        for(size_t i = 1; i < aStrides.size(); ++i)
        {
            if(aStrides[i] > aStrides[i - 1])
            {
                throw std::invalid_argument(
                    "Matmul requires A, B and C tensors to have a contiguous layout.");
            }
        }
    }

    template <class ADataType, class BDataType, class CDataType>
    static void validateMatmul(const TensorBase<ADataType>& a,
                               const TensorBase<BDataType>& b,
                               TensorBase<CDataType>& c)
    {
        // Validate tensor dimensions
        validateConsistentDimensions(a.dims(), b.dims(), c.dims());
        validateConsistentLayouts(a.strides(), b.strides(), c.strides());
        if(!a.isPacked() || !b.isPacked() || !c.isPacked())
        {
            throw std::invalid_argument(
                "Matmul requires A, B and C tensors to have a contiguous layout.");
        }

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<ADataType>,
                      "Matmul supports only float, half and bfloat16 A data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<BDataType>,
                      "Matmul supports only float, half and bfloat16 B data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<CDataType>,
                      "Matmul supports only float, half and bfloat16 C data types.");
    }

    // --- Kernel launcher (defined in GpuFpReferenceMatmul.cpp) ---

    static void launchMatmul(const void* aPtr,
                             const std::vector<int64_t>& aDims,
                             const std::vector<int64_t>& aStrides,
                             const void* bPtr,
                             const std::vector<int64_t>& bDims,
                             const std::vector<int64_t>& bStrides,
                             void* cPtr,
                             const std::vector<int64_t>& cDims,
                             const std::vector<int64_t>& cStrides,
                             int64_t tileSize,
                             const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
