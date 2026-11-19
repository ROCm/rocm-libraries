// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn-gpu-ref/detail/HipRtcTypeName.hpp"

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
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
inline std::vector<std::string> buildMatmulDefines(size_t rank)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DA_TYPE=") + HipRtcTypeName<ADataType>::VALUE);
    defines.emplace_back(std::string("-DB_TYPE=") + HipRtcTypeName<BDataType>::VALUE);
    defines.emplace_back(std::string("-DC_TYPE=") + HipRtcTypeName<CDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    defines.emplace_back(std::string("-DMATMUL_BATCH_DIM_COUNT=") + std::to_string(rank - 2));
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
        validateMatmul<ADataType, BDataType, CDataType, ComputeDataType>(a, b, c);

        auto defines = detail::
            buildMatmulDefines<ADataType, BDataType, CDataType, ComputeDataType, TILE_SIZE>(
                a.dims().size());

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

    template <class T>
    static constexpr bool IS_SUPPORTED_COMPUTE_DATA_TYPE
        = std::is_same_v<T, double> || std::is_same_v<T, float>
          || std::is_same_v<T, hipdnn_data_sdk::types::half>
          || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>;

    static void validateConsistentDimensions(const std::vector<int64_t>& aDims,
                                             const std::vector<int64_t>& bDims,
                                             const std::vector<int64_t>& cDims)
    {
        if(aDims.size() != bDims.size() || aDims.size() != cDims.size())
        {
            throw std::invalid_argument(
                "Matmul requires A, B and C tensors to have the same rank.");
        }

        if(aDims.size() < 2 || aDims.size() > 5)
        {
            throw std::invalid_argument(
                "Matmul requires A, B and C tensor ranks to be 2, 3, 4 or 5.");
        }

        for(size_t i = 0; i < aDims.size(); ++i)
        {
            if(aDims[i] <= 0 || bDims[i] <= 0 || cDims[i] <= 0)
            {
                throw std::invalid_argument(
                    "Matmul requires A, B and C tensors to have positive dimensions.");
            }
        }

        if(!hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::isBroadcastCompatible(
               aDims, bDims, cDims))
        {
            throw std::invalid_argument(
                "Matmul requires A and B tensors to have broadcast-compatible batch dimensions "
                "(all but the last two dimensions).");
        }

        if(!hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::isMatrixDimensionsValid(
               aDims, bDims, cDims))
        {
            throw std::invalid_argument(
                "Matmul requires C tensor dimensions to match expected C tensor dimensions "
                "(broadcasted batch..., M, N) from A tensor dimensions (A batch..., M, K) and B "
                "tensor dimensions(B batch..., K, N).");
        }
    }

    template <class ADataType, class BDataType, class CDataType, class ComputeDataType>
    static void validateMatmul(const TensorBase<ADataType>& a,
                               const TensorBase<BDataType>& b,
                               TensorBase<CDataType>& c)
    {
        // Validate tensor dimensions
        validateConsistentDimensions(a.dims(), b.dims(), c.dims());

        // Validate data types
        static_assert(IS_SUPPORTED_DATA_TYPE<ADataType>,
                      "Matmul supports only float, half and bfloat16 A data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<BDataType>,
                      "Matmul supports only float, half and bfloat16 B data types.");
        static_assert(IS_SUPPORTED_DATA_TYPE<CDataType>,
                      "Matmul supports only float, half and bfloat16 C data types.");
        static_assert(IS_SUPPORTED_COMPUTE_DATA_TYPE<ComputeDataType>,
                      "Matmul supports only double, float, half and bfloat16 compute data types.");
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
