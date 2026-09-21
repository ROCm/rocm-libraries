// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceReduction
{
public:
    static bool isApplicable(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        return node.attributes_type() == NodeAttributes::ReductionAttributes;
    }

    template <class XDataType, class YDataType, class ComputeDataType = float>
    static void reduce(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                       hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                       hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        validateInput(x, y);
        // Validate mode is supported/set.
        initAccumulator<ComputeDataType>(mode);

        const auto& xDims = x.dims();
        const auto& yDims = y.dims();
        const auto rank = yDims.size();

        // Compute the reduction domain - the set of input elements that map to
        // each output element. For a 1d reduction this is a vector; for 2d, a
        // matrix; etc.
        //
        // Example with ADD (sum), X shape=(2, 3, 4):
        //   X = [[[ 0,  1,  2,  3],
        //         [ 4,  5,  6,  7],
        //         [ 8,  9, 10, 11]],
        //        [[12, 13, 14, 15],
        //         [16, 17, 18, 19],
        //         [20, 21, 22, 23]]]
        //
        //   Y shape=(1, 3, 4) -> reduction axis: 0.
        //   Each output element's reduction domain is a vector along 0 axis:
        //     Y[0,0,0]: [0, 12] -> 12
        //     Y[0,0,1]: [1, 13] -> 14
        //   Y = [[[12, 14, 16, 18],
        //         [20, 22, 24, 26],
        //         [28, 30, 32, 34]]]
        //
        //   Y shape=(1, 3, 1) -> reduction axes: 0 and 2.
        //   Each output element's reduction domain is a 2x4 matrix along 0 and 2 axes:
        //     Y[0,0,0]: [[ 0,  1,  2,  3],
        //                [12, 13, 14, 15]]  -> 60
        //     Y[0,1,0]: [[ 4,  5,  6,  7],
        //                [16, 17, 18, 19]]  -> 92
        //   Y = [[[ 60],
        //         [ 92],
        //         [124]]]
        std::vector<size_t> reductionDomainAxes;
        std::vector<int64_t> reductionDomainShape;
        int64_t reductionDomainSize = 1;
        for(size_t i = 0; i < rank; ++i)
        {
            if(yDims[i] == 1 && xDims[i] > 1)
            {
                reductionDomainAxes.push_back(i);
                reductionDomainShape.push_back(xDims[i]);
                reductionDomainSize *= xDims[i];
            }
        }

        // Compute strides for the reduction domain, so we can de-linearize
        // iterator into reduction domain.
        //
        // Continuing the 2x4 example (Y shape=(1,3,1), reduction axes 0 and 2):
        //   Y[0,0,0]: [[ 0,  1,  2,  3],
        //              [12, 13, 14, 15]]
        //   reductionDomainShape  = [2, 4]
        //   reductionDomainStride = [4, 1]
        std::vector<int64_t> reductionDomainStride(reductionDomainAxes.size());
        if(!reductionDomainStride.empty())
        {
            reductionDomainStride.back() = 1;
            for(size_t i = 1; i < reductionDomainStride.size(); ++i)
            {
                auto j = reductionDomainStride.size() - 1 - i;
                reductionDomainStride[j]
                    = reductionDomainStride[j + 1] * reductionDomainShape[j + 1];
            }
        }

        auto reduceFunc = [&](const std::vector<int64_t>& yIndices) {
            auto acc = initAccumulator<ComputeDataType>(mode);
            std::vector<int64_t> xIndices = yIndices;

            // Iterate over linearized index for reduction domain.
            for(int64_t flatIdx = 0; flatIdx < reductionDomainSize; ++flatIdx)
            {
                // De-linearize index
                //
                // Continuing the 2x4 example (Y shape=(1,3,1), reduction axes 0 and 2):
                //   Y[0,0,0]: [[ 0,  1,  2,  3],
                //              [12, 13, 14, 15]]
                //   flatIdx=0 -> (0/4, 0%4) = (0,0) -> X[0, row, 0]
                //   flatIdx=7 -> (7/4, 7%4) = (1,3) -> X[1, row, 3]
                int64_t remaining = flatIdx;
                for(size_t r = 0; r < reductionDomainAxes.size(); ++r)
                {
                    xIndices[reductionDomainAxes[r]] = remaining / reductionDomainStride[r];
                    remaining %= reductionDomainStride[r];
                }

                // Accumulate
                const XDataType val = x.getHostValue(xIndices);
                accumulate(acc, static_cast<ComputeDataType>(val), mode);
            }

            auto result = finalize(acc, reductionDomainSize, mode);
            y.setHostValue(hipdnn_test_sdk::detail::safeConvert<YDataType>(result), yIndices);
        };

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(reduceFunc, y.dims());
        // Run on all available CPU threads
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }

private:
    template <typename TX, typename TY>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<TX>& x,
                              const hipdnn_data_sdk::utilities::TensorBase<TY>& y)
    {
        const auto& xDims = x.dims();
        const auto& yDims = y.dims();

        if(xDims.size() != yDims.size())
        {
            throw std::invalid_argument("Reduction expects X and Y to have the same rank (X rank="
                                        + std::to_string(xDims.size())
                                        + ", Y rank=" + std::to_string(yDims.size()) + ")");
        }

        bool hasReducedDim = false;
        for(size_t i = 0; i < xDims.size(); ++i)
        {
            if(yDims[i] != xDims[i] && yDims[i] != 1)
            {
                throw std::invalid_argument("Reduction: Y dim[" + std::to_string(i)
                                            + "]=" + std::to_string(yDims[i]) + " must equal X dim["
                                            + std::to_string(i) + "]=" + std::to_string(xDims[i])
                                            + " or be 1");
            }
            if(yDims[i] == 1 && xDims[i] > 1)
            {
                hasReducedDim = true;
            }
        }

        if(!hasReducedDim)
        {
            throw std::invalid_argument("Reduction: at least one dimension must be reduced");
        }
    }

    template <class ComputeDataType>
    static ComputeDataType initAccumulator(hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
        switch(mode)
        {
        case ReductionMode::ADD:
        case ReductionMode::AVG:
        case ReductionMode::AMAX:
        case ReductionMode::NORM1:
        case ReductionMode::NORM2:
            return static_cast<ComputeDataType>(0);
        case ReductionMode::MUL:
        case ReductionMode::MUL_NO_ZEROS:
            return static_cast<ComputeDataType>(1);
        case ReductionMode::MIN_OP:
            return std::numeric_limits<ComputeDataType>::max();
        case ReductionMode::MAX_OP:
            return std::numeric_limits<ComputeDataType>::lowest();
        default:
            throw std::invalid_argument("Unsupported reduction mode for initAccumulator");
        }
    }

    template <class ComputeDataType>
    static void accumulate(ComputeDataType& acc,
                           ComputeDataType val,
                           hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
        switch(mode)
        {
        case ReductionMode::ADD:
        case ReductionMode::AVG:
            acc += val;
            break;
        case ReductionMode::MUL:
            acc *= val;
            break;
        case ReductionMode::MIN_OP:
            acc = std::min(acc, val);
            break;
        case ReductionMode::MAX_OP:
            acc = std::max(acc, val);
            break;
        case ReductionMode::AMAX:
            acc = std::max(acc, std::abs(val));
            break;
        case ReductionMode::NORM1:
            acc += std::abs(val);
            break;
        case ReductionMode::NORM2:
            acc += val * val;
            break;
        case ReductionMode::MUL_NO_ZEROS:
            if(val != static_cast<ComputeDataType>(0))
            {
                acc *= val;
            }
            break;
        default:
            throw std::invalid_argument("Unsupported reduction mode for accumulate");
        }
    }

    template <class ComputeDataType>
    static ComputeDataType finalize(ComputeDataType acc,
                                    int64_t count,
                                    hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
    {
        using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
        switch(mode)
        {
        case ReductionMode::AVG:
            return acc / static_cast<ComputeDataType>(count);
        case ReductionMode::NORM2:
            return std::sqrt(acc);
        default:
            return acc;
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
