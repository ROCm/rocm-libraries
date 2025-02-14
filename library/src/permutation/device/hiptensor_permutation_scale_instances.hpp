/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP

// Stdlib includes
#include <cstdlib>
#include <memory>
#include <vector>

// CK includes
#include <add_device_operation_instance.hpp>
#include <ck.hpp>
#include <ck/tensor_operation/gpu/device/device_elementwise.hpp>
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>

#include "instance_params.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                template <typename InDataTypeTuple,
                          typename OutDataTypeTuple,
                          typename ElementwiseOperation,
                          index_t NumDim,
                          index_t BlockSize,
                          index_t M0PerBlock,
                          index_t M1PerBlock,
                          index_t M0PerThread,
                          index_t M1PerThread,
                          typename ThreadClusterArrangeOrder,
                          typename InScalarPerVectorSeq,
                          typename OutScalarPerVectorSeq>
                struct HiptensorDeviceElementwiseImpl
                    : public ck::tensor_operation::device::DeviceElementwiseImpl<
                          InDataTypeTuple,
                          OutDataTypeTuple,
                          ElementwiseOperation,
                          NumDim,
                          BlockSize,
                          M0PerBlock,
                          M1PerBlock,
                          M0PerThread,
                          M1PerThread,
                          ThreadClusterArrangeOrder,
                          InScalarPerVectorSeq,
                          OutScalarPerVectorSeq>
                {

                    std::string GetTypeString() const override
                    {
                        auto str = std::stringstream();
                        str << NumDim << "_";
                        str << BlockSize << "_";
                        str << M0PerBlock << "_";
                        str << M1PerBlock << "_";
                        str << M0PerThread << "_";
                        str << M1PerThread << "_";
                        str << ThreadClusterArrangeOrder::At(0) << "_";
                        str << ThreadClusterArrangeOrder::At(1) << "_";
                        str << InScalarPerVectorSeq::At(0) << "_";
                        str << OutScalarPerVectorSeq::At(0);
                        return str.str();
                    }
                };

                template <typename InDataTypeTuple,
                          typename OutDataTypeTuple,
                          typename Aop,
                          typename Bop,
                          typename Scale,
                          index_t NumDim>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceElementwise<
                        InDataTypeTuple,
                        OutDataTypeTuple,
                        ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
                        NumDim>>
                {
                    using ElementwiseOperation
                        = ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>;
                    using DeviceOp = DeviceElementwise<InDataTypeTuple,
                                                       OutDataTypeTuple,
                                                       ElementwiseOperation,
                                                       NumDim>;

                    template <index_t BlockSize,
                              index_t M0PerBlock,
                              index_t M1PerBlock,
                              index_t M0PerThread,
                              index_t M1PerThread,
                              typename ThreadClusterArrangeOrder,
                              typename InScalarPerVectorSeq,
                              typename OutScalarPerVectorSeq,
                              typename Container>
                    static void addInstance(Container& container)
                    {
                        container.insert(
                            {DeviceElementwiseParams<InDataTypeTuple,
                                                     OutDataTypeTuple,
                                                     Scale,
                                                     NumDim,
                                                     BlockSize,
                                                     M0PerBlock,
                                                     M1PerBlock,
                                                     M0PerThread,
                                                     M1PerThread,
                                                     ThreadClusterArrangeOrder,
                                                     InScalarPerVectorSeq,
                                                     OutScalarPerVectorSeq>::hashCode(),
                             std::make_unique<
                                 HiptensorDeviceElementwiseImpl<InDataTypeTuple,
                                                                OutDataTypeTuple,
                                                                ElementwiseOperation,
                                                                NumDim,
                                                                BlockSize,
                                                                M0PerBlock,
                                                                M1PerBlock,
                                                                M0PerThread,
                                                                M1PerThread,
                                                                ThreadClusterArrangeOrder,
                                                                InScalarPerVectorSeq,
                                                                OutScalarPerVectorSeq>>()});
                    }
                    static auto GetInstances()
                    {
                        std::unordered_map<hiptensor::Uid, std::unique_ptr<DeviceOp>> opPtrs;
                        // clang-format off
                        if constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<float>> && NumDim == 2) {

                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 16  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 32  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 16  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 32  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 32  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 64  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 16  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 32  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 128 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 64  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 32  , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<128 , 64  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 256 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);

                            // the following instances are the safety net to float and rank2
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                        } else if  constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<float>> && NumDim == 3) {

                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 64  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 64  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 32  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 32  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 16  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 32  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 128 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 128 , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 64  , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 16  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);

                            // the following instances are the safety net to float and rank3
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                        } else if  constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<float>> && NumDim == 4) {

                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 64  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 64  , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 256 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 64  , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 16  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 128 , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 32  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 32  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 32  , 16  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);

                            // the following instances are the safety net to float and rank4
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                        } else if  constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<ck::half_t>> && NumDim == 2) {

                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 16  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 32  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 32  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 64  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 64  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<64  , 64  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<64  , 16  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<32  , 32  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 16  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 128 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);

                            // the following instances are the safety net to half and rank2
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                        } else if  constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<ck::half_t>> && NumDim == 3) {

                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 64  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<128 , 256 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 32  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 64  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 16  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 64  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 64  , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 128 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 256 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 32  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);

                            // the following instances are the safety net to half and rank3
                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                        } else if  constexpr(std::is_same_v<InDataTypeTuple, ck::Tuple<ck::half_t>> && NumDim == 4) {

                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 256 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<32  , 64  , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 64  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 64  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<1 , 0> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<64  , 32  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 256 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 32  , 32  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<128 , 128 , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<64  , 64  , 64  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 64  , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<16> , ck::Sequence<16>>(opPtrs);
                            addInstance<256 , 128 , 128 , 16 , 16 , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<128 , 32  , 256 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<8>  , ck::Sequence<8>>(opPtrs);
                            addInstance<256 , 32  , 128 , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 128 , 128 , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);

                            // the following instances are the safety net to half and rank4
                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<64  , 128 , 32  , 8  , 8  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);
                        } else if  constexpr(NumDim == 5 || NumDim == 6) {
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<4>  , ck::Sequence<4>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<2>  , ck::Sequence<2>>(opPtrs);
                            addInstance<256 , 64  , 64  , 4  , 4  , ck::Sequence<0 , 1> , ck::Sequence<1>  , ck::Sequence<1>>(opPtrs);
                        }
                        // clang-format on
                        return opPtrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
