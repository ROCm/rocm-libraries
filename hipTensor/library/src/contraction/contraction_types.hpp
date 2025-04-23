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

#ifndef HIPTENSOR_CONTRACTION_TYPES_HPP
#define HIPTENSOR_CONTRACTION_TYPES_HPP

#include <ostream>

namespace hiptensor
{
    /**
     * \brief This enum decides the over the operation based on the inputs.
     * \details This enum decides the operation based on the in puts passed in the
     * hipTensorContractionGetWorkspaceSize
     */
    enum struct ContractionOpId_t : int32_t
    {
        SCALE            = 0, ///< \f${C=\alpha\mathcal{A}\mathcal{B}}\f$
        BILINEAR         = 1, ///< \f${D=\alpha\mathcal{A}\mathcal{B}+\beta\mathcal{C}}\f$
        SCALE_COMPLEX    = 2,
        BILINEAR_COMPLEX = 3,
        UNKNOWN,
    };

    // Map type to runtime hiptensorOperator_t
    template <typename OpId>
    struct ElementWiseOperatorType;

    template <typename OpId>
    static constexpr auto ElementWiseOperatorType_v = ElementWiseOperatorType<OpId>::value;

    // Map type to runtime ContractionOpId_t
    template <typename OpId>
    struct ContractionOperatorType;

    template <typename OpId>
    static constexpr auto ContractionOperatorType_v = ContractionOperatorType<OpId>::value;

} // namespace hiptensor

namespace std
{
    ostream& operator<<(ostream& os, hiptensor::ContractionOpId_t const&);

} // namespace std

#include "contraction_types_impl.hpp"

#endif // HIPTENSOR_CONTRACTION_TYPES_HPP
