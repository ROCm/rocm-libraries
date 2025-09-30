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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    constexpr uint64_t GemmTypeId(rocisa::DataType aType,
                                  rocisa::DataType bType,
                                  rocisa::DataType cType,
                                  rocisa::DataType dType,
                                  rocisa::DataType alphaType,
                                  rocisa::DataType betaType,
                                  rocisa::DataType computeInputType)
    {
        static_assert(BitFieldGenerator::ElementWidth((uint32_t)rocisa::DataType::Count) * 7
                          <= BitFieldGenerator::maxBitFieldWidth,
                      "Max bitfield width exceeded");

        return BitFieldGenerator::GenerateBitField(
            BitFieldGenerator::ElementWidth((uint32_t)rocisa::DataType::Count),
            (uint32_t)aType,
            (uint32_t)bType,
            (uint32_t)cType,
            (uint32_t)dType,
            (uint32_t)alphaType,
            (uint32_t)betaType,
            (uint32_t)computeInputType);
    }

    template <typename A            = float,
              typename B            = A,
              typename C            = A,
              typename D            = C,
              typename Alpha        = D,
              typename Beta         = D,
              typename ComputeInput = A>
    struct TypedGemm
    {
        using AType            = A;
        using BType            = B;
        using CType            = C;
        using DType            = D;
        using AlphaType        = Alpha;
        using BetaType         = Beta;
        using ComputeInputType = ComputeInput;

        constexpr static uint64_t TypeId()
        {
            return GemmTypeId(TypeInfo<A>::Enum,
                              TypeInfo<B>::Enum,
                              TypeInfo<C>::Enum,
                              TypeInfo<D>::Enum,
                              TypeInfo<Alpha>::Enum,
                              TypeInfo<Beta>::Enum,
                              TypeInfo<ComputeInput>::Enum);
        }
    };

    // Commonly used type groupings
    // Naming: _[Ti_To_Tc]_:
    // S=float, D=double, C=complex<float>, Z=complex<double>,
    // H=Half, B=BF16, I8x4=Int8x4, I32=int32_t
    using TypedGemm_S_S_S = TypedGemm<float>;
    using TypedGemm_D_D_D = TypedGemm<double>;
    using TypedGemm_C_C_C = TypedGemm<std::complex<float>>;
    using TypedGemm_Z_Z_Z = TypedGemm<std::complex<double>>;
#ifdef TENSILE_USE_HALF
    using TypedGemm_H_H_H = TypedGemm<Half>;
    using TypedGemm_H_H_S = TypedGemm<Half, Half, Half, Half, float, float>;
    using TypedGemm_H_S_S = TypedGemm<Half, Half, float, float>;
    // Mix precision
    using TypedGemm_HS_H_H_S = TypedGemm<Half, float, Half, Half, float, float, Half>;
    using TypedGemm_SH_H_H_S = TypedGemm<float, Half, Half, Half, float, float, Half>;
#endif // TENSILE_USE_HALF
    using TypedGemm_I8x4_I32_I32 = TypedGemm<Int8x4, Int8x4, int32_t, int32_t>;
    using TypedGemm_I8_I8_I32    = TypedGemm<int8_t, int8_t, int8_t, int8_t, int32_t, int32_t>;
    using TypedGemm_I8_I32_I32   = TypedGemm<int8_t, int8_t, int32_t, int32_t>;
    using TypedGemm_I8_I32_S     = TypedGemm<int8_t, int8_t, int32_t, int32_t, float, float>;
    using TypedGemm_I8_I8_S      = TypedGemm<int8_t, int8_t, int8_t, int8_t, float, float>;
    using TypedGemm_I8_H_S       = TypedGemm<int8_t, int8_t, Half, Half, float, float>;
    using TypedGemm_I32_I32_I32  = TypedGemm<int32_t>;
#ifdef TENSILE_USE_BF16
    using TypedGemm_B_B_S   = TypedGemm<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
    using TypedGemm_B_S_S   = TypedGemm<BFloat16, BFloat16, float, float>;
    using TypedGemm_H_B_H_S = TypedGemm<Half, Half, Half, Half, float, float, BFloat16>;
    using TypedGemm_I8_B_S  = TypedGemm<int8_t, int8_t, BFloat16, BFloat16, float, float>;
    using TypedGemm_S_B_S   = TypedGemm<float, float, float, float, float, float, BFloat16>;
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
    using TypedGemm_F8_F8_S = TypedGemm<Float8, Float8, Float8, Float8, float, float>;
    using TypedGemm_F8_B8_S = TypedGemm<Float8, Float8, BFloat8, BFloat8, float, float>;
    using TypedGemm_F8_H_S  = TypedGemm<Float8, Float8, Half, Half, float, float>;
    using TypedGemm_F8_B_S  = TypedGemm<Float8, Float8, BFloat16, BFloat16, float, float>;
    using TypedGemm_F8_S_S  = TypedGemm<Float8, Float8, float, float>;
    using TypedGemm_B8_F8_S = TypedGemm<BFloat8, BFloat8, Float8, Float8, float, float>;
    using TypedGemm_B8_B8_S = TypedGemm<BFloat8, BFloat8, BFloat8, BFloat8, float, float>;
    using TypedGemm_B8_S_S  = TypedGemm<BFloat8, BFloat8, float, float>;
    using TypedGemm_B8_H_S  = TypedGemm<BFloat8, BFloat8, Half, Half, float, float>;
    using TypedGemm_B8_B_S  = TypedGemm<BFloat8, BFloat8, BFloat16, BFloat16, float, float>;
    // hybrid
    using TypedGemm_F8B8_F8_S
        = TypedGemm<Float8, BFloat8, Float8, Float8, float, float, Float8BFloat8>;
    using TypedGemm_F8B8_H_S = TypedGemm<Float8, BFloat8, Half, Half, float, float, Float8BFloat8>;
    using TypedGemm_F8B8_B_S
        = TypedGemm<Float8, BFloat8, BFloat16, BFloat16, float, float, Float8BFloat8>;
    using TypedGemm_F8B8_S_S
        = TypedGemm<Float8, BFloat8, float, float, float, float, Float8BFloat8>;
    using TypedGemm_B8F8_F8_S
        = TypedGemm<BFloat8, Float8, Float8, Float8, float, float, BFloat8Float8>;
    using TypedGemm_B8F8_H_S = TypedGemm<BFloat8, Float8, Half, Half, float, float, BFloat8Float8>;
    using TypedGemm_B8F8_B_S
        = TypedGemm<BFloat8, Float8, BFloat16, BFloat16, float, float, BFloat8Float8>;
    using TypedGemm_B8F8_S_S
        = TypedGemm<BFloat8, Float8, float, float, float, float, BFloat8Float8>;
    using TypedGemm_F8B8_B8_S
        = TypedGemm<Float8, BFloat8, BFloat8, BFloat8, float, float, Float8BFloat8>;
    using TypedGemm_B8F8_B8_S
        = TypedGemm<BFloat8, Float8, BFloat8, BFloat8, float, float, BFloat8Float8>;
    using TypedGemm_H_F8B8_H_S = TypedGemm<Half, Half, Half, Half, float, float, Float8BFloat8>;
    using TypedGemm_H_B8F8_H_S = TypedGemm<Half, Half, Half, Half, float, float, BFloat8Float8>;

    // NANOO F8
    using TypedGemm_F8N_F8N_S
        = TypedGemm<Float8_fnuz, Float8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    using TypedGemm_F8N_B8N_S
        = TypedGemm<Float8_fnuz, Float8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
    using TypedGemm_F8N_H_S = TypedGemm<Float8_fnuz, Float8_fnuz, Half, Half, float, float>;
    using TypedGemm_F8N_B_S = TypedGemm<Float8_fnuz, Float8_fnuz, BFloat16, BFloat16, float, float>;
    using TypedGemm_F8N_S_S = TypedGemm<Float8_fnuz, Float8_fnuz, float, float>;
    using TypedGemm_B8N_F8N_S
        = TypedGemm<BFloat8_fnuz, BFloat8_fnuz, Float8_fnuz, Float8_fnuz, float, float>;
    using TypedGemm_B8N_B8N_S
        = TypedGemm<BFloat8_fnuz, BFloat8_fnuz, BFloat8_fnuz, BFloat8_fnuz, float, float>;
    using TypedGemm_B8N_S_S = TypedGemm<BFloat8_fnuz, BFloat8_fnuz, float, float>;
    using TypedGemm_B8N_H_S = TypedGemm<BFloat8_fnuz, BFloat8_fnuz, Half, Half, float, float>;
    using TypedGemm_B8N_B_S
        = TypedGemm<BFloat8_fnuz, BFloat8_fnuz, BFloat16, BFloat16, float, float>;
    // hybrid
    using TypedGemm_F8B8N_F8N_S = TypedGemm<Float8_fnuz,
                                            BFloat8_fnuz,
                                            Float8_fnuz,
                                            Float8_fnuz,
                                            float,
                                            float,
                                            Float8BFloat8_fnuz>;
    using TypedGemm_F8B8N_H_S
        = TypedGemm<Float8_fnuz, BFloat8_fnuz, Half, Half, float, float, Float8BFloat8_fnuz>;
    using TypedGemm_F8B8N_B_S = TypedGemm<Float8_fnuz,
                                          BFloat8_fnuz,
                                          BFloat16,
                                          BFloat16,
                                          float,
                                          float,
                                          Float8BFloat8_fnuz>;
    using TypedGemm_F8B8N_S_S
        = TypedGemm<Float8_fnuz, BFloat8_fnuz, float, float, float, float, Float8BFloat8_fnuz>;
    using TypedGemm_B8F8N_F8N_S = TypedGemm<BFloat8_fnuz,
                                            Float8_fnuz,
                                            Float8_fnuz,
                                            Float8_fnuz,
                                            float,
                                            float,
                                            BFloat8Float8_fnuz>;
    using TypedGemm_B8F8N_H_S
        = TypedGemm<BFloat8_fnuz, Float8_fnuz, Half, Half, float, float, BFloat8Float8_fnuz>;
    using TypedGemm_B8F8N_B_S = TypedGemm<BFloat8_fnuz,
                                          Float8_fnuz,
                                          BFloat16,
                                          BFloat16,
                                          float,
                                          float,
                                          BFloat8Float8_fnuz>;
    using TypedGemm_B8F8N_S_S
        = TypedGemm<BFloat8_fnuz, Float8_fnuz, float, float, float, float, BFloat8Float8_fnuz>;
    using TypedGemm_F8B8N_B8N_S = TypedGemm<Float8_fnuz,
                                            BFloat8_fnuz,
                                            BFloat8_fnuz,
                                            BFloat8_fnuz,
                                            float,
                                            float,
                                            Float8BFloat8_fnuz>;
    using TypedGemm_B8F8N_B8N_S = TypedGemm<BFloat8_fnuz,
                                            Float8_fnuz,
                                            BFloat8_fnuz,
                                            BFloat8_fnuz,
                                            float,
                                            float,
                                            BFloat8Float8_fnuz>;
    using TypedGemm_H_F8B8N_H_S
        = TypedGemm<Half, Half, Half, Half, float, float, Float8BFloat8_fnuz>;
    using TypedGemm_H_B8F8N_H_S
        = TypedGemm<Half, Half, Half, Half, float, float, BFloat8Float8_fnuz>;

#ifdef TENSILE_USE_HALF
    // Mix precision: OCPFP8
    using TypedGemm_H_F8_H_S      = TypedGemm<Half, Half, Half, Half, float, float, Float8>;
    using TypedGemm_H_B8_H_S      = TypedGemm<Half, Half, Half, Half, float, float, BFloat8>;
    using TypedGemm_HF8_H_S_S     = TypedGemm<Half, Float8, float, float, float, float, Half>;
    using TypedGemm_F8H_H_S_S     = TypedGemm<Float8, Half, float, float, float, float, Half>;
    using TypedGemm_HF8_H_H_S     = TypedGemm<Half, Float8, Half, Half, float, float, Half>;
    using TypedGemm_F8H_H_H_S     = TypedGemm<Float8, Half, Half, Half, float, float, Half>;
    using TypedGemm_HF8_H_FP8_S   = TypedGemm<Half, Float8, Float8, Float8, float, float, Half>;
    using TypedGemm_F8H_H_FP8_S   = TypedGemm<Float8, Half, Float8, Float8, float, float, Half>;
    using TypedGemm_HF8_FP8_S_S   = TypedGemm<Half, Float8, float, float, float, float, Float8>;
    using TypedGemm_F8H_FP8_S_S   = TypedGemm<Float8, Half, float, float, float, float, Float8>;
    using TypedGemm_HF8_FP8_H_S   = TypedGemm<Half, Float8, Half, Half, float, float, Float8>;
    using TypedGemm_F8H_FP8_H_S   = TypedGemm<Float8, Half, Half, Half, float, float, Float8>;
    using TypedGemm_HF8_FP8_FP8_S = TypedGemm<Half, Float8, Float8, Float8, float, float, Float8>;
    using TypedGemm_F8H_FP8_FP8_S = TypedGemm<Float8, Half, Float8, Float8, float, float, Float8>;

    // Mix precision: NANOO
    using TypedGemm_H_F8N_H_S  = TypedGemm<Half, Half, Half, Half, float, float, Float8_fnuz>;
    using TypedGemm_H_B8N_H_S  = TypedGemm<Half, Half, Half, Half, float, float, BFloat8_fnuz>;
    using TypedGemm_HF8N_H_S_S = TypedGemm<Half, Float8_fnuz, float, float, float, float, Half>;
    using TypedGemm_F8NH_H_S_S = TypedGemm<Float8_fnuz, Half, float, float, float, float, Half>;
    using TypedGemm_HF8N_H_H_S = TypedGemm<Half, Float8_fnuz, Half, Half, float, float, Half>;
    using TypedGemm_F8NH_H_H_S = TypedGemm<Float8_fnuz, Half, Half, Half, float, float, Half>;
    using TypedGemm_HF8N_H_FP8_S
        = TypedGemm<Half, Float8_fnuz, Float8_fnuz, Float8_fnuz, float, float, Half>;
    using TypedGemm_F8NH_H_FP8_S
        = TypedGemm<Float8_fnuz, Half, Float8_fnuz, Float8_fnuz, float, float, Half>;
    using TypedGemm_HF8N_FP8_S_S
        = TypedGemm<Half, Float8_fnuz, float, float, float, float, Float8_fnuz>;
    using TypedGemm_F8NH_FP8_S_S
        = TypedGemm<Float8_fnuz, Half, float, float, float, float, Float8_fnuz>;
    using TypedGemm_HF8N_FP8_H_S
        = TypedGemm<Half, Float8_fnuz, Half, Half, float, float, Float8_fnuz>;
    using TypedGemm_F8NH_FP8_H_S
        = TypedGemm<Float8_fnuz, Half, Half, Half, float, float, Float8_fnuz>;
    using TypedGemm_HF8N_FP8_FP8_S
        = TypedGemm<Half, Float8_fnuz, Float8_fnuz, Float8_fnuz, float, float, Float8_fnuz>;
    using TypedGemm_F8NH_FP8_FP8_S
        = TypedGemm<Float8_fnuz, Half, Float8_fnuz, Float8_fnuz, float, float, Float8_fnuz>;
#endif // TENSILE_USE_HALF
#endif // TENSILE_USE_FP8_BF8
} // namespace TensileLite
