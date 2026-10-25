/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <cstdint>
#include <numeric>

#include <hip/hip_runtime.h>

#include <ck/utility/data_type.hpp>
#include <ck/utility/type_convert.hpp>

#include "data_types.hpp"
#include "hip_error.hpp"
#include "hiptensor_element_wise_operation.hpp"
#include "reduction_epilogue.hpp"
#include "util.hpp"

namespace hiptensor
{
    namespace
    {
        struct EpilogueLayout
        {
            int32_t rank;
            int64_t lengths[kReductionEpilogueMaxRank];
            int64_t stridesD[kReductionEpilogueMaxRank];
            int64_t stridesA[kReductionEpilogueMaxRank];
            int64_t stridesC[kReductionEpilogueMaxRank];
        };

        template <typename T, typename ComputeT>
        __global__ void reductionEpilogueKernel(T*                  D,
                                                T const*            A,
                                                T const*            C,
                                                EpilogueLayout      layout,
                                                hiptensorOperator_t opA,
                                                ComputeT            alpha,
                                                hiptensorOperator_t opC,
                                                ComputeT            beta,
                                                int64_t             count)
        {
            using ck::tensor_operation::element_wise::HiptensorUnaryOp;

            for(int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
                i += int64_t(gridDim.x) * blockDim.x)
            {
                // The first mode varies fastest; any order visits every element once.
                int64_t rest = i, offsetD = 0, offsetA = 0, offsetC = 0;
                for(int32_t d = 0; d < layout.rank; d++)
                {
                    int64_t index = rest % layout.lengths[d];
                    rest /= layout.lengths[d];
                    offsetD += index * layout.stridesD[d];
                    offsetA += index * layout.stridesA[d];
                    offsetC += index * layout.stridesC[d];
                }

                ComputeT result = ComputeT(0);
                if(A != nullptr)
                {
                    ComputeT value;
                    HiptensorUnaryOp{opA}(value, ck::type_convert<ComputeT>(A[offsetA]));
                    result = alpha * value;
                }
                if(C != nullptr)
                {
                    ComputeT value;
                    HiptensorUnaryOp{opC}(value, ck::type_convert<ComputeT>(C[offsetC]));
                    result += beta * value;
                }
                D[offsetD] = ck::type_convert<T>(result);
            }
        }

        template <typename T, typename ComputeT>
        hiptensorStatus_t launchEpilogue(EpilogueLayout const& layout,
                                         int64_t               count,
                                         void*                 D,
                                         void const*           A,
                                         hiptensorOperator_t   opA,
                                         double                alpha,
                                         void const*           C,
                                         hiptensorOperator_t   opC,
                                         double                beta,
                                         hipStream_t           stream)
        {
            constexpr int64_t blockSize = 256;
            constexpr int64_t maxBlocks = int64_t(1) << 20;
            auto const        blocks    = std::max<int64_t>(
                1, std::min<int64_t>((count + blockSize - 1) / blockSize, maxBlocks));

            reductionEpilogueKernel<T, ComputeT>
                <<<dim3(static_cast<uint32_t>(blocks)), dim3(blockSize), 0, stream>>>(
                    static_cast<T*>(D),
                    static_cast<T const*>(A),
                    static_cast<T const*>(C),
                    layout,
                    opA,
                    static_cast<ComputeT>(alpha),
                    opC,
                    static_cast<ComputeT>(beta),
                    count);
            HIPTENSOR_RETURN_IF_HIP_ERROR(hipGetLastError());
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace

    hiptensorStatus_t reductionEpilogue(hiptensorDataType_t             type,
                                        std::vector<std::size_t> const& lengths,
                                        std::vector<std::size_t> const& stridesD,
                                        void*                           D,
                                        std::vector<std::size_t> const& stridesA,
                                        void const*                     A,
                                        hiptensorOperator_t             opA,
                                        double                          alpha,
                                        std::vector<std::size_t> const& stridesC,
                                        void const*                     C,
                                        hiptensorOperator_t             opC,
                                        double                          beta,
                                        hipStream_t                     stream)
    {
        if(lengths.size() > kReductionEpilogueMaxRank || stridesD.size() != lengths.size()
           || (A != nullptr && stridesA.size() != lengths.size())
           || (C != nullptr && stridesC.size() != lengths.size()))
        {
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }

        EpilogueLayout layout{};
        layout.rank   = static_cast<int32_t>(lengths.size());
        int64_t count = 1;
        for(std::size_t d = 0; d < lengths.size(); d++)
        {
            layout.lengths[d]  = static_cast<int64_t>(lengths[d]);
            layout.stridesD[d] = static_cast<int64_t>(stridesD[d]);
            layout.stridesA[d] = A != nullptr ? static_cast<int64_t>(stridesA[d]) : 0;
            layout.stridesC[d] = C != nullptr ? static_cast<int64_t>(stridesC[d]) : 0;
            count *= layout.lengths[d];
        }
        if(count == 0)
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }

        switch(type)
        {
        case HIPTENSOR_R_16F:
            return launchEpilogue<ck::half_t, float>(
                layout, count, D, A, opA, alpha, C, opC, beta, stream);
        case HIPTENSOR_R_16BF:
            return launchEpilogue<ck::bhalf_t, float>(
                layout, count, D, A, opA, alpha, C, opC, beta, stream);
        case HIPTENSOR_R_32F:
            return launchEpilogue<float, float>(
                layout, count, D, A, opA, alpha, C, opC, beta, stream);
        case HIPTENSOR_R_64F:
            return launchEpilogue<double, double>(
                layout, count, D, A, opA, alpha, C, opC, beta, stream);
        default:
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }
    }

    hiptensorStatus_t reductionCopyTensor(hiptensorDataType_t             type,
                                          std::vector<std::size_t> const& lengths,
                                          std::vector<std::size_t> const& strides,
                                          void*                           D,
                                          void const*                     C,
                                          hipStream_t                     stream)
    {
        // The elements fill a contiguous range when, ordered by stride, every mode of extent
        // greater than one starts where the previous one ends.
        std::vector<std::size_t> order(lengths.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return strides[a] < strides[b];
        });
        std::size_t expected   = 1;
        bool        contiguous = strides.size() == lengths.size();
        for(auto d : order)
        {
            if(!contiguous || lengths[d] == 1)
            {
                continue;
            }
            contiguous = strides[d] == expected;
            expected *= lengths[d];
        }

        if(contiguous)
        {
            auto const bytes = elementsFromLengths(lengths) * hiptensorDataTypeSize(type);
            HIPTENSOR_RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(D, C, bytes, hipMemcpyDeviceToDevice, stream));
            return HIPTENSOR_STATUS_SUCCESS;
        }

        return reductionEpilogue(type,
                                 lengths,
                                 strides,
                                 D,
                                 {},
                                 nullptr,
                                 HIPTENSOR_OP_IDENTITY,
                                 0.0,
                                 strides,
                                 C,
                                 HIPTENSOR_OP_IDENTITY,
                                 1.0,
                                 stream);
    }
} // namespace hiptensor
