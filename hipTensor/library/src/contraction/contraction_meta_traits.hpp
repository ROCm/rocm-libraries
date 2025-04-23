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

#ifndef HIPTENSOR_CONTRACTION_META_TRAITS_HPP
#define HIPTENSOR_CONTRACTION_META_TRAITS_HPP

// CK includes
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>

// hiptensor includes
#include "data_types.hpp"
#include "device/device_element_wise_operation_complex.hpp"
#include "meta_traits.hpp"

#define MaxNumDimsM 6
#define MaxNumDimsN 6
#define MaxNumDimsK 6

namespace hiptensor
{
    // Partial specialize for Bilinear contraction
    template <ck::index_t NumDimsM,
              ck::index_t NumDimsN,
              ck::index_t NumDimsK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType>
    struct MetaTraits<
        ck::tensor_operation::device::DeviceContractionMultipleD<NumDimsM,
                                                                 NumDimsN,
                                                                 NumDimsK,
                                                                 ADataType,
                                                                 BDataType,
                                                                 ck::Tuple<DsDataType>,
                                                                 EDataType,
                                                                 AElementwiseOperation,
                                                                 BElementwiseOperation,
                                                                 CDEElementwiseOperation,
                                                                 ComputeDataType>,
        std::enable_if_t<
            (std::is_same_v<CDEElementwiseOperation, ck::tensor_operation::element_wise::Bilinear>)
            || (std::is_same_v<CDEElementwiseOperation,
                               ck::tensor_operation::element_wise::BilinearComplex>)>>
    {
        constexpr static ck::index_t DimsM = NumDimsM;
        constexpr static ck::index_t DimsN = NumDimsN;
        constexpr static ck::index_t DimsK = NumDimsK;
        /*
         * CK does not use hip_bfloat16, instead it use ushort(ck::bhalf_t) for cuda bhalf_t type.
         * What we want here is that we can use ck::bhalf_t with ck instances and use hip_bfloat16
         * with hiptensor classes.
         *
         * When creating a solution, ck::bhalf_t was passed in to create ck instance.
         * When registering the solution, MetaTraits will returen hip_bfloat16 to create key.
         */
        using ADataT
            = std::conditional_t<std::is_same_v<ADataType, ck::bhalf_t>, hip_bfloat16, ADataType>;
        using BDataT
            = std::conditional_t<std::is_same_v<BDataType, ck::bhalf_t>, hip_bfloat16, BDataType>;
        using DDataT
            = std::conditional_t<std::is_same_v<DsDataType, ck::bhalf_t>, hip_bfloat16, DsDataType>;
        using EDataT
            = std::conditional_t<std::is_same_v<EDataType, ck::bhalf_t>, hip_bfloat16, EDataType>;
        using ComputeDataT = std::conditional_t<std::is_same_v<ComputeDataType, ck::bhalf_t>,
                                                hip_bfloat16,
                                                ComputeDataType>;
        using AOp          = AElementwiseOperation;
        using BOp          = BElementwiseOperation;
        using CDEOp        = CDEElementwiseOperation;
    };

    // Partial specialize for Scale contraction
    template <ck::index_t NumDimsM,
              ck::index_t NumDimsN,
              ck::index_t NumDimsK,
              typename ADataType,
              typename BDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType>
    struct MetaTraits<
        ck::tensor_operation::device::DeviceContractionMultipleD<NumDimsM,
                                                                 NumDimsN,
                                                                 NumDimsK,
                                                                 ADataType,
                                                                 BDataType,
                                                                 ck::Tuple<>,
                                                                 EDataType,
                                                                 AElementwiseOperation,
                                                                 BElementwiseOperation,
                                                                 CDEElementwiseOperation,
                                                                 ComputeDataType>,
        std::enable_if_t<
            (std::is_same_v<CDEElementwiseOperation, ck::tensor_operation::element_wise::Scale>)
            || (std::is_same_v<CDEElementwiseOperation,
                               ck::tensor_operation::element_wise::ScaleComplex>)>>
    {
        constexpr static ck::index_t DimsM = NumDimsM;
        constexpr static ck::index_t DimsN = NumDimsN;
        constexpr static ck::index_t DimsK = NumDimsK;
        using ADataT
            = std::conditional_t<std::is_same_v<ADataType, ck::bhalf_t>, hip_bfloat16, ADataType>;
        using BDataT
            = std::conditional_t<std::is_same_v<BDataType, ck::bhalf_t>, hip_bfloat16, BDataType>;
        using DDataT = NoneType;
        using EDataT
            = std::conditional_t<std::is_same_v<EDataType, ck::bhalf_t>, hip_bfloat16, EDataType>;
        using ComputeDataT = std::conditional_t<std::is_same_v<ComputeDataType, ck::bhalf_t>,
                                                hip_bfloat16,
                                                ComputeDataType>;
        using AOp          = AElementwiseOperation;
        using BOp          = BElementwiseOperation;
        using CDEOp        = CDEElementwiseOperation;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_META_TRAITS_HPP
