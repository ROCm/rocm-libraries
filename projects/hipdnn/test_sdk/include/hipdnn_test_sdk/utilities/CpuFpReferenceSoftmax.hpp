// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// CPU reference for softmax over the last dimension (cudnn-frontend Softmax_attributes /
/// CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR). For every row r of p (all dims but the last):
///   m        = max_j p[r, j]
///   sum      = sum_j exp(p[r, j] - m)
///   s[r, j]  = exp(p[r, j] - m) / sum
///   stats[r] = m + ln(sum)               (log-sum-exp; divided by ln 2 with statsUseLog2)
/// Arithmetic runs in ComputeType. A row whose entries are all -inf (fully masked) has no
/// probability mass: s is 0 across the row and stats is -inf, instead of the NaN a naive
/// exp(-inf - -inf) would give. NaN inputs propagate. Each element of p is read before the
/// matching element of s is written, so s may alias p.
class CpuFpReferenceSoftmax
{
public:
    template <class PType, class SType = PType, class StatsType = float, class ComputeType = float>
    static void forward(const hipdnn_data_sdk::utilities::TensorBase<PType>& p,
                        hipdnn_data_sdk::utilities::TensorBase<SType>& s,
                        hipdnn_data_sdk::utilities::TensorBase<StatsType>* stats,
                        bool statsUseLog2)
    {
        validate(p, s, stats);

        const auto& pDims = p.dims();
        const auto rank = pDims.size();
        const int64_t rowLength = pDims[rank - 1];

        // Rows are indexed by every dim but the last; a rank-1 tensor is a single row.
        std::vector<int64_t> rowDims(pDims.begin(),
                                     pDims.begin() + static_cast<std::ptrdiff_t>(rank - 1));
        if(rowDims.empty())
        {
            rowDims.push_back(1);
        }
        for(const auto dim : rowDims)
        {
            if(dim == 0)
            {
                return;
            }
        }

        const PType* pBase = p.memory().hostData();
        SType* sBase = s.memory().hostData();
        StatsType* statsBase = stats == nullptr ? nullptr : stats->memory().hostData();
        const auto& pStrides = p.strides();
        const auto& sStrides = s.strides();
        const std::vector<int64_t> noStrides;
        const auto& statsStrides = stats == nullptr ? noStrides : stats->strides();
        const int64_t pLast = pStrides[rank - 1];
        const int64_t sLast = sStrides[rank - 1];
        const auto leadingRank = rank - 1;

        const auto negInf = -std::numeric_limits<ComputeType>::infinity();
        const auto invLn2 = static_cast<ComputeType>(1.0 / std::log(2.0));

        auto softmaxRow = [&](const std::vector<int64_t>& row) {
            int64_t pRow = 0;
            int64_t sRow = 0;
            int64_t statsRow = 0;
            for(size_t d = 0; d < leadingRank; ++d)
            {
                pRow += row[d] * pStrides[d];
                sRow += row[d] * sStrides[d];
                if(statsBase != nullptr)
                {
                    statsRow += row[d] * statsStrides[d];
                }
            }

            auto rowMax = negInf;
            bool hasNan = false;
            for(int64_t j = 0; j < rowLength; ++j)
            {
                const auto value = static_cast<ComputeType>(pBase[pRow + j * pLast]);
                if(std::isnan(value))
                {
                    hasNan = true;
                }
                else if(value > rowMax)
                {
                    rowMax = value;
                }
            }

            if(rowMax == negInf && !hasNan)
            {
                // Fully masked (or empty) row: no probability mass.
                for(int64_t j = 0; j < rowLength; ++j)
                {
                    sBase[sRow + j * sLast] = static_cast<SType>(ComputeType{0});
                }
                if(statsBase != nullptr)
                {
                    statsBase[statsRow] = static_cast<StatsType>(negInf);
                }
                return;
            }

            if(hasNan)
            {
                rowMax = std::numeric_limits<ComputeType>::quiet_NaN();
            }

            auto sum = ComputeType{0};
            for(int64_t j = 0; j < rowLength; ++j)
            {
                sum += std::exp(static_cast<ComputeType>(pBase[pRow + j * pLast]) - rowMax);
            }

            for(int64_t j = 0; j < rowLength; ++j)
            {
                const auto e = std::exp(static_cast<ComputeType>(pBase[pRow + j * pLast]) - rowMax);
                sBase[sRow + j * sLast] = static_cast<SType>(e / sum);
            }

            if(statsBase != nullptr)
            {
                auto lse = rowMax + std::log(sum);
                if(statsUseLog2)
                {
                    lse *= invLn2;
                }
                statsBase[statsRow] = static_cast<StatsType>(lse);
            }
        };

        auto parallelFunc = hipdnn_test_sdk::detail::makeParallelTensorFunctor(softmaxRow, rowDims);
        parallelFunc(std::thread::hardware_concurrency());

        s.memory().markHostModified();
        if(stats != nullptr)
        {
            stats->memory().markHostModified();
        }
    }

private:
    template <class PType, class SType, class StatsType>
    static void validate(const hipdnn_data_sdk::utilities::TensorBase<PType>& p,
                         const hipdnn_data_sdk::utilities::TensorBase<SType>& s,
                         const hipdnn_data_sdk::utilities::TensorBase<StatsType>* stats)
    {
        const std::string prefix = "CpuFpReferenceSoftmax: ";

        hipdnn_test_sdk::detail::validateNoRaggedTensor(p, prefix, "p");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(s, prefix, "s");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(stats, prefix, "stats");

        const auto& pDims = p.dims();
        if(pDims.empty())
        {
            throw std::runtime_error(prefix + "p must have rank >= 1");
        }
        if(s.dims() != pDims)
        {
            throw std::runtime_error(prefix + "s dims must equal p dims");
        }
        if(stats != nullptr)
        {
            auto expected = pDims;
            expected.back() = 1;
            if(stats->dims() != expected)
            {
                throw std::runtime_error(prefix + "stats dims must equal p dims with last dim 1");
            }
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
