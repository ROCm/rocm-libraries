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
#include "util.hpp"

namespace std
{
    template <typename T>
    struct hash<std::vector<T>>
    {
        constexpr std::size_t operator()(const std::vector<T>& vec) const
        {
            std::size_t seed = 0;
            for(const auto& elem : vec)
            {
                // Combine the hash of each element into the overall hash
                seed ^= std::hash<T>{}(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

namespace ck::tensor_operation::device::instance
{
    template <typename DataTypeTuple>
    inline auto convertTypeTupleToHipDataTypeVector()
    {
        using HiptensorDataTypeTuple
            = hiptensor::tuple_ck_type_tuple_to_hiptensor_type_tuple_t<DataTypeTuple>;
        std::vector<hipDataType> hipDataTypeVector;
        ck::static_for<0, HiptensorDataTypeTuple::Size(), 1>{}([&hipDataTypeVector](auto i) {
            hipDataTypeVector.push_back(
                hiptensor::HipDataType_v<typename ck::tuple_element_t<i, HiptensorDataTypeTuple>>);
        });
        return hipDataTypeVector;
    }

    class DeviceElementwiseParams
    {
    public:
        constexpr static hiptensor::Uid hashCode(DeviceElementwiseParams const& params)
        {
            return hiptensor::Hash{}(params.mInDataTypes,
                                     params.mOutDataTypes,
                                     params.mScale,
                                     params.mNumDim,
                                     params.mInstanceHyperParams.mBlockSize,
                                     params.mInstanceHyperParams.mM0PerBlock,
                                     params.mInstanceHyperParams.mM1PerBlock,
                                     params.mInstanceHyperParams.mM0PerThread,
                                     params.mInstanceHyperParams.mM1PerThread,
                                     params.mInstanceHyperParams.mThreadClusterArrangeOrder,
                                     params.mInstanceHyperParams.mInScalarPerVectorSeq,
                                     params.mInstanceHyperParams.mOutScalarPerVectorSeq);
        }

        template <typename InDataTypeTuple,
                  typename OutDataTypeTuple,
                  hiptensor::PermutationOpId_t Scale,
                  index_t                      NumDim,
                  index_t                      BlockSize,
                  index_t                      M0PerBlock,
                  index_t                      M1PerBlock,
                  index_t                      M0PerThread,
                  index_t                      M1PerThread,
                  typename ThreadClusterArrangeOrder,
                  typename InScalarPerVectorSeq,
                  typename OutScalarPerVectorSeq>
        static auto Gen()
        {
            DeviceElementwiseParams params;
            params.mInDataTypes  = convertTypeTupleToHipDataTypeVector<InDataTypeTuple>();
            params.mOutDataTypes = convertTypeTupleToHipDataTypeVector<OutDataTypeTuple>();
            params.mScale        = Scale;
            params.mNumDim       = NumDim;
            params.mInstanceHyperParams.mBlockSize   = BlockSize;
            params.mInstanceHyperParams.mM0PerBlock  = M0PerBlock;
            params.mInstanceHyperParams.mM1PerBlock  = M1PerBlock;
            params.mInstanceHyperParams.mM0PerThread = M0PerThread;
            params.mInstanceHyperParams.mM1PerThread = M1PerThread;
            params.mInstanceHyperParams.mThreadClusterArrangeOrder
                = {ThreadClusterArrangeOrder::At(0), ThreadClusterArrangeOrder::At(1)};
            ck::static_for<0, InScalarPerVectorSeq::Size(), 1>{}([&params](auto i) {
                params.mInstanceHyperParams.mInScalarPerVectorSeq.push_back(
                    InScalarPerVectorSeq::At(i));
            });
            params.mInstanceHyperParams.mOutScalarPerVectorSeq = {OutScalarPerVectorSeq::At(0)};
            return params;
        }
        template <typename InDataTypeTuple,
                  typename OutDataTypeTuple,
                  hiptensor::PermutationOpId_t Scale,
                  index_t                      NumDim>
        static auto Gen()
        {
            DeviceElementwiseParams params;
            params.mInDataTypes  = convertTypeTupleToHipDataTypeVector<InDataTypeTuple>();
            params.mOutDataTypes = convertTypeTupleToHipDataTypeVector<OutDataTypeTuple>();
            params.mScale        = Scale;
            params.mNumDim       = NumDim;

            // This function is only used for reference instance.
            // Referenece instances are not affected by member variables below. So set them to
            // default values.
            //
            // Important: Need to use `InstanceHyperParams{}`, the default value of InstanceHyperParams,
            // to query a reference instance since `InstanceHyperParams{}` matches InstanceHyperParams value
            // return from this function.
            params.mInstanceHyperParams = {};
            return params;
        }
        static auto Gen(std::vector<hipDataType> const&       inDataTypes,
                        std::vector<hipDataType> const&       outDataTypes,
                        hiptensor::PermutationOpId_t          scale,
                        index_t                               numDim,
                        hiptensor::InstanceHyperParams const& instanceHyperParams)
        {
            DeviceElementwiseParams params;
            params.mInDataTypes         = inDataTypes;
            params.mOutDataTypes        = outDataTypes;
            params.mScale               = scale;
            params.mNumDim              = numDim;
            params.mInstanceHyperParams = instanceHyperParams;
            return params;
        }

    private:
        DeviceElementwiseParams() = default;

        std::vector<hipDataType>     mInDataTypes;
        std::vector<hipDataType>     mOutDataTypes;
        hiptensor::PermutationOpId_t mScale;
        index_t                      mNumDim;

        hiptensor::InstanceHyperParams mInstanceHyperParams;
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
        getHashCodeOfBestPerfInstances(std::vector<hipDataType> const&       typeIn,
                                       std::vector<hipDataType> const&       typeOut,
                                       hiptensor::PermutationOpId_t          scale,
                                       index_t                               numDim,
                                       hiptensor::InstanceHyperParams const& hyperParams);

}

namespace std
{
    template <>
    struct hash<ck::tensor_operation::device::instance::DeviceElementwiseParams>
    {
        constexpr std::size_t operator()(
            ck::tensor_operation::device::instance::DeviceElementwiseParams const& params) const
        {
            return ck::tensor_operation::device::instance::DeviceElementwiseParams::hashCode(
                params);
        }
    };
}
#endif //  INSTANCE_PARAMS_HPP
