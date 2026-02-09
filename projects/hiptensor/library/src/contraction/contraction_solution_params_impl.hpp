/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "contraction_meta_traits.hpp"
#include "contraction_solution_params.hpp"
#include "data_types.hpp"
#include "hash.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::ContractionSolutionParams>
    {
        size_t operator()(hiptensor::ContractionSolutionParams const& s) const noexcept
        {
            return hiptensor::Hash{}(s.dimsM(),
                                     s.dimsN(),
                                     s.dimsK(),
                                     s.typeCompute(),
                                     s.typeA(),
                                     s.typeB(),
                                     s.typeC(),
                                     s.typeD(),
                                     s.opA(),
                                     s.opB(),
                                     s.opCDE());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp>
    struct ContractionSolutionParamsImpl : public ContractionSolutionParams
    {
        ContractionSolutionParamsImpl()                                                = default;
        ~ContractionSolutionParamsImpl()                                               = default;
        ContractionSolutionParamsImpl(ContractionSolutionParamsImpl const&)            = default;
        ContractionSolutionParamsImpl(ContractionSolutionParamsImpl&&)                 = default;
        ContractionSolutionParamsImpl& operator=(ContractionSolutionParamsImpl const&) = default;
        ContractionSolutionParamsImpl& operator=(ContractionSolutionParamsImpl&&)      = default;

        using MetaTraitsT = MetaTraits<DeviceOp>;

        int32_t dimsM() const override
        {
            return MetaTraitsT::DimsM;
        }

        int32_t dimsN() const override
        {
            return MetaTraitsT::DimsN;
        }

        int32_t dimsK() const override
        {
            return MetaTraitsT::DimsK;
        }

        hiptensorDataType_t typeA() const override
        {
            return HipTensorDataType_v<typename MetaTraitsT::ADataT>;
        }

        hiptensorDataType_t typeB() const override
        {
            return HipTensorDataType_v<typename MetaTraitsT::BDataT>;
        }

        hiptensorDataType_t typeC() const override
        {
            return HipTensorDataType_v<typename MetaTraitsT::DDataT>;
        }

        hiptensorDataType_t typeD() const override
        {
            return HipTensorDataType_v<typename MetaTraitsT::EDataT>;
        }

        hiptensorComputeDescriptor_t typeCompute() const override
        {
            return convertToComputeType(HipTensorDataType_v<typename MetaTraitsT::ComputeDataT>);
        }

        hiptensorOperator_t opA() const override
        {
            return ElementWiseOperatorType_v<typename MetaTraitsT::AOp>;
        }

        hiptensorOperator_t opB() const override
        {
            return ElementWiseOperatorType_v<typename MetaTraitsT::BOp>;
        }

        ContractionOpId_t opCDE() const override
        {
            return ContractionOperatorType_v<typename MetaTraitsT::CDEOp>;
        }
    };

} // namespace hiptensor
