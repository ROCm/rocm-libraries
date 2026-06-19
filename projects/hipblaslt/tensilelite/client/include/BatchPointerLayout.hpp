// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace TensileLite::Client
{
    struct BatchPointerLayout
    {
        std::vector<size_t> offsets;

        size_t count() const noexcept
        {
            return offsets.size();
        }
    };

    inline std::vector<size_t> batchPointerTensorBatchIndices(
        ContractionProblemGemm::BatchIndices const& batchIndices,
        ContractionProblemGemm::TENSOR              tensor)
    {
        switch(tensor)
        {
        case ContractionProblemGemm::TENSOR::A:
        case ContractionProblemGemm::TENSOR::B:
        case ContractionProblemGemm::TENSOR::C:
        case ContractionProblemGemm::TENSOR::D:
            break;
        default:
            throw std::invalid_argument(
                "Batch pointer tensor index mapping only supports A/B/C/D.");
        }

        std::vector<size_t> batchIdx;
        batchIdx.reserve(batchIndices.size());

        for(auto const& batchIndex : batchIndices)
        {
            switch(tensor)
            {
            case ContractionProblemGemm::TENSOR::A:
                batchIdx.push_back(batchIndex.a);
                break;
            case ContractionProblemGemm::TENSOR::B:
                batchIdx.push_back(batchIndex.b);
                break;
            case ContractionProblemGemm::TENSOR::C:
                batchIdx.push_back(batchIndex.c);
                break;
            case ContractionProblemGemm::TENSOR::D:
                batchIdx.push_back(batchIndex.d);
                break;
            default:
                throw std::logic_error("Unexpected batch pointer tensor after validation.");
            }
        }

        return batchIdx;
    }

    inline BatchPointerLayout makeBatchPointerLayout(TensorDescriptor const& tensor,
                                                     std::vector<size_t> const& batchIdx)
    {
        BatchPointerLayout layout;
        std::vector<size_t> batchSizes;
        std::vector<size_t> batchStrides;

        batchSizes.reserve(batchIdx.size());
        batchStrides.reserve(batchIdx.size());

        for(size_t index : batchIdx)
        {
            if(index >= tensor.dimensions())
                throw std::out_of_range("Batch pointer layout batch dimension out of range.");

            batchSizes.push_back(tensor.sizes()[index]);
            batchStrides.push_back(tensor.strides()[index]);
        }

        size_t const count = TensileLite::CoordCount(batchSizes.begin(), batchSizes.end());
        layout.offsets.reserve(count);

        std::vector<size_t> coord(batchSizes.size(), 0);
        for(size_t idx = 0; idx < count; ++idx)
        {
            TensileLite::CoordNumbered(
                idx, coord.begin(), coord.end(), batchSizes.begin(), batchSizes.end());

            size_t offset = 0;
            for(size_t i = 0; i < batchSizes.size(); ++i)
            {
                offset += coord[i] * batchStrides[i];
            }
            layout.offsets.push_back(offset);
        }

        return layout;
    }
} // namespace TensileLite::Client
