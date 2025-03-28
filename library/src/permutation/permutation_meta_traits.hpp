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

#ifndef HIPTENSOR_PERMUTATION_META_TRAITS_HPP
#define HIPTENSOR_PERMUTATION_META_TRAITS_HPP

// CK includes
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>
#include <device_operation_instance_factory.hpp>

// hiptensor includes
#include "data_types.hpp"
#include "device/hiptensor_permutation_scale_instances.hpp"
#include "meta_traits.hpp"
#include <hiptensor/internal/native_types.hpp>

namespace hiptensor
{
    // Meta traits for Scalar permutation
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Scale,
              ck::index_t NumDim>
    struct MetaTraits<ck::tensor_operation::device::DeviceElementwise<
        InDataTypeTuple,
        OutDataTypeTuple,
        ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale>,
        NumDim>>
    {
        constexpr static ck::index_t               NDim = NumDim;
        constexpr static ElementwiseInstanceType_t InstanceType
            = ElementwiseInstanceType_t::PERMUTATION;

        using InDataT  = InDataTypeTuple;
        using OutDataT = OutDataTypeTuple;

        using AOp        = Aop;
        using ScaleOp    = Scale;
        using CombinedOp = ck::tensor_operation::element_wise::UnaryCombinedOp<AOp, ScaleOp>;
    };

    // Meta traits for Scalar elementwise_binary
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Cop,
              typename ACop,
              ck::index_t NumDim>
    struct MetaTraits<ck::tensor_operation::device::DeviceElementwise<
        InDataTypeTuple,
        OutDataTypeTuple,
        ck::tensor_operation::element_wise::BinaryWithUnaryCombinedOp<ACop, Aop, Cop>,
        NumDim>>
    {
        constexpr static ck::index_t               NDim = NumDim;
        constexpr static ElementwiseInstanceType_t InstanceType
            = ElementwiseInstanceType_t::ELEMENTWISE_BINARY_OP;

        /*
         * CK does not use hip_bfloat16, instead it use ushort(ck::bhalf_t) for cuda bhalf_t type.
         * What we want here is that we can use ck::bhalf_t with ck instances and use hip_bfloat16
         * with hiptensor classes.
         *
         * When creating a solution, ck::bhalf_t was passed in to create ck instance.
         * When registering the solution, MetaTraits will returen hip_bfloat16 to create key.
         */
        // using InDataT  = InDataTypeTuple;
        using InDataT  = tuple_ck_type_tuple_to_hiptensor_type_tuple_t<InDataTypeTuple>;
        using OutDataT = tuple_ck_type_tuple_to_hiptensor_type_tuple_t<OutDataTypeTuple>;

        using AOp  = Aop;
        using COp  = Cop;
        using ACOp = ACop;
        using CombinedOp
            = ck::tensor_operation::element_wise::BinaryWithUnaryCombinedOp<ACop, Aop, Cop>;
    };

    // Meta traits for Scalar elementwise_trinary
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Cop,
              typename ABop,
              typename ABCop,
              ck::index_t NumDim>
    struct MetaTraits<ck::tensor_operation::device::DeviceElementwise<
        InDataTypeTuple,
        OutDataTypeTuple,
        ck::tensor_operation::element_wise::TrinaryWithUnaryCombinedOp<ABop, ABCop, Aop, Bop, Cop>,
        NumDim>>
    {
        constexpr static ck::index_t               NDim = NumDim;
        constexpr static ElementwiseInstanceType_t InstanceType
            = ElementwiseInstanceType_t::ELEMENTWISE_TRINARY_OP;

        using InDataT  = InDataTypeTuple;
        using OutDataT = OutDataTypeTuple;

        using AOp        = Aop;
        using BOp        = Bop;
        using COp        = Cop;
        using ABOp       = ABop;
        using ABCOp      = ABCop;
        using CombinedOp = ck::tensor_operation::element_wise::
            TrinaryWithUnaryCombinedOp<ABop, ABCop, Aop, Bop, Cop>;
    };
} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_META_TRAITS_HPP
