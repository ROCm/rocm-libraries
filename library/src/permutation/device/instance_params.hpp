/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef INSTANCE_PARAMS_HPP
#define INSTANCE_PARAMS_HPP

#include "../permutation_types.hpp"
#include "data_types.hpp"
#include "hash.hpp"

namespace ck::tensor_operation::device::instance
{
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Scale,
              index_t NumDim,
              index_t BlockSize                  = 0,
              index_t M0PerBlock                 = 0,
              index_t M1PerBlock                 = 0,
              index_t M0PerThread                = 0,
              index_t M1PerThread                = 0,
              typename ThreadClusterArrangeOrder = ck::Sequence<0, 0>,
              typename InScalarPerVectorSeq      = ck::Sequence<0>,
              typename OutScalarPerVectorSeq     = ck::Sequence<0>>
    struct DeviceElementwiseParams
    {
        static hiptensor::Uid hashCode()
        {
            return hiptensor::Hash{}(
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, InDataTypeTuple>>,
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, OutDataTypeTuple>>,
                hiptensor::PermutationOperatorType_v<Scale>,
                NumDim,
                BlockSize,
                M0PerBlock,
                M1PerBlock,
                M0PerThread,
                M1PerThread,
                ThreadClusterArrangeOrder::At(0),
                ThreadClusterArrangeOrder::At(1),
                InScalarPerVectorSeq::At(0),
                OutScalarPerVectorSeq::At(0));
        }
    };

    // `getHashCodeOfBestPerfInstances` generates a hash code based on the arguments. This hash code represents
    // the best perf instance. And it appends hash codes of 2 more instances which can handle the input tensors
    // that cannot be handled by the best perf instance.
    //
    // Ck requires that the length of fastest changing dimonsion must be multiple times of `InScalarPerVectorSeq`
    // and `OutScalarPerVectorSeq`. For example, `tensor.lengths[0] == 1777`, it cannot be handled by instance with
    // `InScalarPerVectorSeq == 8`.

    // The caller should test the returned hash code in order since earlier instances have better perf.
    std::vector<hiptensor::Uid>
        getHashCodeOfBestPerfInstances(hipDataType                           typeIn,
                                       hipDataType                           typeOut,
                                       hiptensor::PermutationOpId_t          scale,
                                       index_t                               numDim,
                                       hiptensor::InstanceHyperParams const& hyperParams);

}
#endif //  INSTANCE_PARAMS_HPP
