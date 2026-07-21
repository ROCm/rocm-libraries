// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace test_matmul_common
{

struct MatmulTestCase
{
    std::vector<int64_t> aDims;
    std::vector<int64_t> bDims;
    std::vector<int64_t> cDims;
    bool transA;
    bool transB;
    unsigned seed;

    MatmulTestCase(std::vector<int64_t>&& aDimsLocal,
                   std::vector<int64_t>&& bDimsLocal,
                   bool transALocal,
                   bool transBLocal,
                   unsigned seedLocal)
        : aDims(std::move(aDimsLocal))
        , bDims(std::move(bDimsLocal))
        , transA(transALocal)
        , transB(transBLocal)
        , seed(seedLocal)
    {
        if(aDims.size() != bDims.size())
        {
            throw std::invalid_argument("aDims and bDims must have the same number of dimensions.");
        }

        const auto rank = aDims.size();
        if(rank < 2)
        {
            throw std::invalid_argument("Tensor rank must have at least 2 dimensions.");
        }

        const auto batchDims = rank - 2;
        cDims = std::vector<int64_t>(rank, 1);

        for(size_t i = 0; i < batchDims; ++i)
        {
            const auto aDimVal = aDims[i];
            const auto bDimVal = bDims[i];

            if(aDimVal <= 0 || bDimVal <= 0)
            {
                throw std::invalid_argument("Dimensions must be positive.");
            }

            if(aDimVal % bDimVal != 0 && bDimVal % aDimVal != 0)
            {
                throw std::invalid_argument("Batch dimensions must be broadcastable.");
            }

            cDims[i] = std::max(aDimVal, bDimVal);
        }

        // Matrix dimensions:
        // A[..., M, K] x B[..., K, N] -> C[..., M, N]
        if(aDims[batchDims + 1] != bDims[batchDims])
        {
            throw std::invalid_argument("Matmul shape mismatch: A.K must equal B.K");
        }

        cDims[batchDims] = aDims[batchDims];
        cDims[batchDims + 1] = bDims[batchDims + 1];
    }

    friend std::ostream& operator<<(std::ostream& ss, const MatmulTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(a:";
        vecToStream(ss, tc.aDims);
        ss << " b:";
        vecToStream(ss, tc.bDims);
        ss << " c:";
        vecToStream(ss, tc.cDims);
        ss << " transA:";
        ss << tc.transA;
        ss << " transB:";
        ss << tc.transB;
        ss << " seed:" << tc.seed;
        ss << ")";

        return ss;
    }
};

// Row-major strides for the given dims, with the last two swapped when the
// operand is transposed (column-major).
inline std::vector<int64_t> generateInputStrideOrder(const std::vector<int64_t>& dims,
                                                     bool transpose)
{
    std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    if(transpose)
    {
        const size_t rank = dims.size();
        strides[rank - 1] = dims[rank - 2];
        strides[rank - 2] = 1;
    }
    return strides;
}

} // namespace test_matmul_common
