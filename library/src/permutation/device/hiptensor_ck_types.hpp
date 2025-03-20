
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

#ifndef HIPTENSOR_CK_TYPES_HPP
#define HIPTENSOR_CK_TYPES_HPP

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/device_elementwise.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp>
#include <ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp>

#include <hiptensor_unary_element_wise_operation.hpp>

namespace hiptensor
{
    using CkScale             = ck::tensor_operation::element_wise::Scale;
    using CkPassThrough       = ck::tensor_operation::element_wise::PassThrough;
    using CkHiptensorUnaryOp  = ck::tensor_operation::element_wise::HiptensorUnaryOp;
    using CkHiptensorBinaryOp = ck::tensor_operation::element_wise::HiptensorBinaryOp;
    using CkUnaryCombinedOp
        = ck::tensor_operation::element_wise::UnaryCombinedOp<CkHiptensorUnaryOp, CkScale>;
    using CkPermutationUnaryCombinedOp
        = ck::tensor_operation::element_wise::UnaryCombinedOp<CkHiptensorUnaryOp, CkScale>;
    using CkPermutationPassThroughCombinedOp
        = ck::tensor_operation::element_wise::UnaryCombinedOp<CkPassThrough, CkPassThrough>;
    using CkBinaryWithUnaryCombinedOp = ck::tensor_operation::element_wise::
        BinaryWithUnaryCombinedOp<CkHiptensorBinaryOp, CkUnaryCombinedOp, CkUnaryCombinedOp>;
    using CkTrinaryWithUnaryCombinedOp
        = ck::tensor_operation::element_wise::TrinaryWithUnaryCombinedOp<CkHiptensorBinaryOp,
                                                                         CkHiptensorBinaryOp,
                                                                         CkUnaryCombinedOp,
                                                                         CkUnaryCombinedOp,
                                                                         CkUnaryCombinedOp>;
}

#endif // HIPTENSOR_CK_TYPES_HPP
