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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP

#include "register_layout_transforms_impl.hpp"

namespace rocwmma
{
    // Array of Structures (AOS) layouts
    // - ColInlineVW
    // - RowInlineVW
    // - ColInlineInt
    // - RowInlineInt
    // Structure of Arrays (SOA) layouts
    // - ColOrthoVW
    // - RowOrthoVW
    // - ColOrthoInt
    // - RowInlineInt

    template <class Transform>
    struct Driver
    {
        using Func = Transform;

        template <typename DataT, uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& v)
        {
            auto result = VecT<DataT, VecSize>{};

            auto rIt = makeVectorIterator<Func::VecSize>(v).begin();
            auto wIt = makeVectorIterator<Func::VecSize>(result).begin();

#pragma unroll
            for(uint32_t i = 0; i < decltype(rIt)::range(); i++)
            {
                *wIt = Func::exec(*rIt);
                rIt++;
                wIt++;
            }
            return result;
        }
    };

    /*! \class aos_to_soa
    *  \brief  A class that transforms register layout from Array of Structures (AOS)
    *  to Structure of Arrays (SOA) for non-interleaved layouts.
    *  @tparam BlockDim The leading block dimension
    *  @tparam MaxVW The maximum vector width
    */
    template <uint32_t BlockDim, uint32_t MaxVW>
    using aos_to_soa = Driver<TransformsImpl::Ops::AosToSoa<BlockDim, MaxVW>>;

    /*! \class soa_to_aos
    *  \brief  A class that transforms register layout from Structure of Arrays (SOA)
    *  to Array of Structures (AOS) for non-interleaved layouts.
    *  @tparam BlockDim The leading block dimension
    *  @tparam MaxVW The maximum vector width
    */
    template <uint32_t BlockDim, uint32_t MaxVW>
    using soa_to_aos = Driver<TransformsImpl::Ops::SoaToAos<BlockDim, MaxVW>>;

    /*! \class aos_to_soa
    *  \brief  A class that transforms register layout from Array of Structures (AOS)
    *  to Structure of Arrays (SOA) for interleaved layouts.
    *  @tparam BlockDim The leading block dimension
    *  @tparam MaxVW The maximum vector width
    */
    template <uint32_t BlockDim, uint32_t MaxVW>
    using aos_to_soa_int = Driver<TransformsImpl::Ops::AosToSoa<BlockDim, MaxVW>>;

    /*! \class soa_to_aos
    *  \brief  A class that transforms register layout from Structure of Arrays (SOA)
    *  to Array of Structures (AOS) for interleaved layouts.
    *  @tparam BlockDim The leading block dimension
    *  @tparam MaxVW The maximum vector width
    */
    template <uint32_t BlockDim, uint32_t MaxVW>
    using soa_to_aos_int = Driver<TransformsImpl::Ops::SoaToAos<BlockDim, MaxVW>>;

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
