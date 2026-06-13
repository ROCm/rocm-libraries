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
#ifndef ROCWMMA_WMMA_IMPL_HPP
#define ROCWMMA_WMMA_IMPL_HPP

#include "constants.hpp"
#include "layout/transforms/transforms.hpp"
#include "permute.hpp"
#include "types.hpp"
#include "utility/type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        struct Unsupported;

        // Enabler for targets.
        // Given a TargetId, enable if it exists in the TargetIds list
        template <uint32_t TargetId, uint32_t... TargetIds>
        using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;

        // Enabler for all of gfx103 (RDNA2)
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx103_t = enable_if_t<contains_number_v<uint32_t,
                                                              TargetId,
                                                              Constants::AMDGCN_ARCH_ID_GFX1030,
                                                              Constants::AMDGCN_ARCH_ID_GFX1031,
                                                              Constants::AMDGCN_ARCH_ID_GFX1032>
                                            && Cond>;

        // Enabler for all of gfx11
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx11_t = enable_if_t<contains_number_v<uint32_t,
                                                             TargetId,
                                                             Constants::AMDGCN_ARCH_ID_GFX1100,
                                                             Constants::AMDGCN_ARCH_ID_GFX1101,
                                                             Constants::AMDGCN_ARCH_ID_GFX1102,
                                                             Constants::AMDGCN_ARCH_ID_GFX1103,
                                                             Constants::AMDGCN_ARCH_ID_GFX1150,
                                                             Constants::AMDGCN_ARCH_ID_GFX1151,
                                                             Constants::AMDGCN_ARCH_ID_GFX1152,
                                                             Constants::AMDGCN_ARCH_ID_GFX1153>
                                           && Cond>;

        // Enabler for all of gfx103 and gfx11 (same WMMA instruction set)
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx103_gfx11_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1030,
                                            Constants::AMDGCN_ARCH_ID_GFX1031,
                                            Constants::AMDGCN_ARCH_ID_GFX1032,
                                            Constants::AMDGCN_ARCH_ID_GFX1100,
                                            Constants::AMDGCN_ARCH_ID_GFX1101,
                                            Constants::AMDGCN_ARCH_ID_GFX1102,
                                            Constants::AMDGCN_ARCH_ID_GFX1103,
                                            Constants::AMDGCN_ARCH_ID_GFX1150,
                                            Constants::AMDGCN_ARCH_ID_GFX1151,
                                            Constants::AMDGCN_ARCH_ID_GFX1152,
                                            Constants::AMDGCN_ARCH_ID_GFX1153>
                          && Cond>;

        // Enabler for all of gfx12
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx12_t = enable_if_t<contains_number_v<uint32_t,
                                                             TargetId,
                                                             Constants::AMDGCN_ARCH_ID_GFX1200,
                                                             Constants::AMDGCN_ARCH_ID_GFX1201,
                                                             Constants::AMDGCN_ARCH_ID_GFX1250>
                                           && Cond>;

        // Enabler for all of gfx11 and gfx12
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx11_gfx12_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1100,
                                            Constants::AMDGCN_ARCH_ID_GFX1101,
                                            Constants::AMDGCN_ARCH_ID_GFX1102,
                                            Constants::AMDGCN_ARCH_ID_GFX1103,
                                            Constants::AMDGCN_ARCH_ID_GFX1150,
                                            Constants::AMDGCN_ARCH_ID_GFX1151,
                                            Constants::AMDGCN_ARCH_ID_GFX1152,
                                            Constants::AMDGCN_ARCH_ID_GFX1153,
                                            Constants::AMDGCN_ARCH_ID_GFX1200,
                                            Constants::AMDGCN_ARCH_ID_GFX1201,
                                            Constants::AMDGCN_ARCH_ID_GFX1250>
                          && Cond>;

        /*! \class amdgcn_wmma
        *  \brief  Builtin wrapper for wmma instructions
        *  @tparam InputTA Datatype of input A
        *  @tparam InputTB Datatype of input B
        *  @tparam ComputeT Datatype of accumulator
        *  @tparam BlockM M-dimension of wmma block
        *  @tparam BlockN N-dimension of wmma block
        *  @tparam BlockK K-dimension of wmma block
        *  @tparam GfxTarget The current gfx family target of interest being compiled
        *  @tparam TargetEnable Enabler for the current target if supported
        */
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t GfxTargetId = Constants::AMDGCN_CURRENT_ARCH_ID,
                  typename Enabler     = void>
        struct amdgcn_wmma
        {
            // This is a pass-through implementation that isn't supported, and doesn't
            // do anything practical. The following trait will allow us to identify
            // unsupported instances, as we won't include it in the overloads to follow.
            using Unsupported = Unsupported;

        private:
            using PackTraitsA   = PackTraits<InputTA>;
            using PackTraitsB   = PackTraits<InputTB>;
            using PackTraitsAcc = PackTraits<ComputeT>;

            constexpr static uint32_t InputASize
                = BlockM * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsA::PackRatio);
            constexpr static uint32_t InputBSize
                = BlockN * BlockK / (Constants::AMDGCN_WAVE_SIZE * PackTraitsB::PackRatio);
            constexpr static uint32_t AccumSize
                = BlockM * BlockM / (Constants::AMDGCN_WAVE_SIZE * PackTraitsAcc::PackRatio);

        public:
            using ARegsT = VecT<typename PackTraitsA::PackedT, InputASize>;
            using BRegsT = VecT<typename PackTraitsB::PackedT, InputBSize>;
            using CRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
            using DRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
        };

        enum struct WmmaCtrlFlags : bool
        {
            // Output register selection of WMMA.
            // Low = bits [15:0]
            // High = bits[31:16]
            LOW  = false,
            HIGH = true,

            // Signage indicator of inputs / accum
            UNSIGNED = false,
            SIGNED   = true,

            // Input matrix reuse indicators
            NO_REUSE = false,
            REUSE    = true,
        };

        enum struct WmmaInputModifiers : int16_t
        {
            // A, B, C input modifiers
            NONE   = 0,
            NEGATE = 1,

            // C input only modifiers
            ABS     = 2,
            NEG_ABS = 3
        };

        // gfx103 software WMMA helpers
        // Note: reinterpret_cast< uint32_t& > is strict-aliasing UB by the C++ standard
        // (same pattern as gfx103_swap_rows / gfx103_half2 below, and as used elsewhere
        // in rocwmma: permute_impl.hpp, mapping_util_impl.hpp). Safe on all current
        // device compilers (clang/hipcc). Should switch to __builtin_bit_cast when
        // rocwmma moves to C++20.
        template <typename DataT>
        ROCWMMA_DEVICE static inline DataT gfx103_bpermute(DataT input, uint32_t laneId)
        {
            static_assert(sizeof(DataT) == sizeof(uint32_t), "Inputs must be 32 bit");
            reinterpret_cast<uint32_t&>(input) = __builtin_amdgcn_ds_bpermute(
                laneId << 2, reinterpret_cast<uint32_t const&>(input));
            return input;
        }

        template <typename VecT>
        ROCWMMA_DEVICE static inline auto gfx103_bpermute_vec(VecT const& input, uint32_t laneId)
        {
            VecT result{};

            for(uint32_t i = 0; i < VecTraits<VecT>::size(); ++i)
            {
                result[i] = gfx103_bpermute(input[i], laneId);
            }

            return result;
        }

        template <typename DataT>
        ROCWMMA_DEVICE static inline DataT gfx103_swap_rows(DataT input)
        {
            static_assert(sizeof(DataT) == sizeof(uint32_t), "Inputs must be 32 bit");
            reinterpret_cast<uint32_t&>(input)
                = __builtin_amdgcn_permlanex16(0u,
                                               reinterpret_cast<uint32_t const&>(input),
                                               0x76543210u,
                                               0xfedcba98u,
                                               false,
                                               true);
            return input;
        }

        template <typename VecT>
        ROCWMMA_DEVICE static inline auto gfx103_swap_rows_vec(VecT const& input)
        {
            VecT result{};

            for(uint32_t i = 0; i < VecTraits<VecT>::size(); ++i)
            {
                result[i] = gfx103_swap_rows(input[i]);
            }

            return result;
        }

        template <uint32_t SrcLaneIdx>
        ROCWMMA_DEVICE static inline auto
            gfx103_fdot2_dpp8(float16_t __attribute__((ext_vector_type(2))) a,
                              float16_t __attribute__((ext_vector_type(2))) b,
                              float32_t                                     accum)
        {
            static_assert(SrcLaneIdx < 8u, "DPP8 src lane out of range");
            if constexpr(SrcLaneIdx == 0u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[0, 0, 0, 0, 0, 0, 0, 0]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 1u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[1, 1, 1, 1, 1, 1, 1, 1]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 2u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[2, 2, 2, 2, 2, 2, 2, 2]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 3u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[3, 3, 3, 3, 3, 3, 3, 3]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 4u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[4, 4, 4, 4, 4, 4, 4, 4]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 5u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[5, 5, 5, 5, 5, 5, 5, 5]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 6u)
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[6, 6, 6, 6, 6, 6, 6, 6]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else
            {
                asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[7, 7, 7, 7, 7, 7, 7, 7]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            return accum;
        }

        template <typename PackedPairT>
        ROCWMMA_DEVICE static inline uint32_t gfx103_select_half2_bits(
            PackedPairT const& lowPacked, PackedPairT const& highPacked, uint32_t index)
        {
            auto const packedIndex = index >> 1;
            auto const shift       = (index & 0x1u) << 4;
            auto const lowBits     = (lowPacked[packedIndex] >> shift) & 0xFFFFu;
            auto const highBits    = (highPacked[packedIndex] >> shift) & 0xFFFFu;
            return lowBits | (highBits << 16);
        }

        // Union type-punning between uint32_t and an ext_vector_type(2) of half
        // is non-portable per the C++ standard but is the common HIP/CUDA device
        // idiom (matches __half2 semantics, and is used elsewhere in rocwmma).
        // See the note above gfx103_bpermute for the strict-aliasing story.
        ROCWMMA_DEVICE static inline auto gfx103_half2(uint32_t packedBits)
        {
            using Half2T = float16_t __attribute__((ext_vector_type(2)));

            union
            {
                uint32_t packed;
                Half2T   half2;
            } result{packedBits};

            return result.half2;
        }

        template <uint32_t VecSize>
        ROCWMMA_DEVICE static inline auto
            gfx103_pack_half_pairs(VecT<float16_t, VecSize> const& input)
        {
            using PairBitsT = VecT<uint32_t, 4>;
            static_assert(VecSize >= 8u, "Inputs must contain at least 8 values");
            PairBitsT result{};

#pragma unroll
            for(uint32_t i = 0; i < 4u; ++i)
            {
                auto pair = gfx103_half2(0u);
                pair[0]   = input[2u * i];
                pair[1]   = input[2u * i + 1u];
                result[i] = reinterpret_cast<uint32_t const&>(pair);
            }

            return result;
        }

        template <typename InputRegsT, typename AccRegsT>
        ROCWMMA_DEVICE static inline auto gfx103_wmma_f32_16x16x16_f16(InputRegsT const& regsA,
                                                                       InputRegsT const& regsB,
                                                                       AccRegsT const&   regsC)
            -> AccRegsT
        {
            using PackedInputT = VecT<float16_t, 16>;
            using PairBitsT    = VecT<uint32_t, 8>;
            using SoaAccumT    = VecT<float32_t, 8>;

            static_assert(sizeof(PackedInputT) == sizeof(InputRegsT), "Inconsistent data formats");

            auto const laneId     = threadIdx.x & (Constants::AMDGCN_WAVE_SIZE - 1u);
            auto const useLowHalf = laneId < 16u;

            auto const aSoa
                = Transforms::from_wmma_input_gfx11(reinterpret_cast<PackedInputT const&>(regsA));
            auto const bSoa
                = Transforms::from_wmma_input_gfx11(reinterpret_cast<PackedInputT const&>(regsB));

            auto const aPacked      = gfx103_pack_half_pairs(aSoa);
            auto const bPacked      = gfx103_pack_half_pairs(bSoa);
            auto const aPackedOther = gfx103_swap_rows_vec(aPacked);
            auto const bPackedOther = gfx103_swap_rows_vec(bPacked);
            PairBitsT  aPairs{};
            PairBitsT  bPairs{};

#pragma unroll
            for(uint32_t i = 0; i < 8u; ++i)
            {
                aPairs[i] = useLowHalf ? gfx103_select_half2_bits(aPacked, aPackedOther, i)
                                       : gfx103_select_half2_bits(aPackedOther, aPacked, i);
                bPairs[i] = useLowHalf ? gfx103_select_half2_bits(bPacked, bPackedOther, i)
                                       : gfx103_select_half2_bits(bPackedOther, bPacked, i);
            }

            // Each lane needs the A row for its column: source from the low or high 8-lane group
            // depending on which half-wave this lane belongs to.
            auto const aRows
                = gfx103_bpermute_vec(aPairs, (laneId & 0x7u) + (useLowHalf ? 0u : 8u));

            auto accumRows = SoaAccumT{};

#pragma unroll
            for(uint32_t pair = 0; pair < 8u; ++pair)
            {
                auto const aPair = gfx103_half2(aRows[pair]);
                auto const bPair = gfx103_half2(bPairs[pair]);
                accumRows[0]     = gfx103_fdot2_dpp8<0>(aPair, bPair, accumRows[0]);
                accumRows[1]     = gfx103_fdot2_dpp8<1>(aPair, bPair, accumRows[1]);
                accumRows[2]     = gfx103_fdot2_dpp8<2>(aPair, bPair, accumRows[2]);
                accumRows[3]     = gfx103_fdot2_dpp8<3>(aPair, bPair, accumRows[3]);
                accumRows[4]     = gfx103_fdot2_dpp8<4>(aPair, bPair, accumRows[4]);
                accumRows[5]     = gfx103_fdot2_dpp8<5>(aPair, bPair, accumRows[5]);
                accumRows[6]     = gfx103_fdot2_dpp8<6>(aPair, bPair, accumRows[6]);
                accumRows[7]     = gfx103_fdot2_dpp8<7>(aPair, bPair, accumRows[7]);
            }

            auto const partnerRows = gfx103_swap_rows_vec(accumRows);
            auto       result      = Transforms::wmma_acc_gfx11_to_soa(regsC);

            if(useLowHalf)
            {
                result[0] += accumRows[0];
                result[1] += accumRows[2];
                result[2] += accumRows[4];
                result[3] += accumRows[6];
                result[4] += partnerRows[0];
                result[5] += partnerRows[2];
                result[6] += partnerRows[4];
                result[7] += partnerRows[6];
            }
            else
            {
                result[0] += partnerRows[1];
                result[1] += partnerRows[3];
                result[2] += partnerRows[5];
                result[3] += partnerRows[7];
                result[4] += accumRows[1];
                result[5] += accumRows[3];
                result[6] += accumRows[5];
                result[7] += accumRows[7];
            }

            return Transforms::soa_to_wmma_acc_gfx11(result);
        }

        template <uint32_t SrcLaneIdx>
        ROCWMMA_DEVICE static inline float gfx103_fmac_dpp8(float a, float b, float accum)
        {
            static_assert(SrcLaneIdx < 8u, "DPP8 src lane out of range");
            if constexpr(SrcLaneIdx == 0u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[0,0,0,0,0,0,0,0]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 1u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[1,1,1,1,1,1,1,1]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 2u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[2,2,2,2,2,2,2,2]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 3u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[3,3,3,3,3,3,3,3]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 4u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[4,4,4,4,4,4,4,4]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 5u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[5,5,5,5,5,5,5,5]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 6u)
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[6,6,6,6,6,6,6,6]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else
            {
                asm volatile("v_fmac_f32_dpp %0, %1, %2 dpp8:[7,7,7,7,7,7,7,7]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            return accum;
        }

        // bf16->f32 fallback for gfx103. Hardware WMMA returns zeros for bf16 on this silicon
        // (v_dot2c_f32_bf16 is absent from gfx10 ISA). v_fma_mix_f32 exists but is VOP3P so
        // DPP8 is unavailable on gfx10 (only gfx11+). So we reuse the f16-style pack/bpermute
        // preamble and manually upcast each bf16 pair to f32 before the v_fmac_f32_dpp8 loop.
        template <typename InputRegsT, typename AccRegsT>
        ROCWMMA_DEVICE static inline auto gfx103_wmma_f32_16x16x16_bf16(InputRegsT const& regsA,
                                                                         InputRegsT const& regsB,
                                                                         AccRegsT const&   regsC)
            -> AccRegsT
        {
            using PackedInputT = VecT<bfloat16_t, 16>;
            using PairBitsT = VecT<uint32_t, 8>;
            using SoaAccumT = VecT<float32_t, 8>;

            static_assert(sizeof(PackedInputT) == sizeof(InputRegsT), "Inconsistent data formats");

            auto const laneId     = threadIdx.x & (Constants::AMDGCN_WAVE_SIZE - 1u);
            auto const useLowHalf = laneId < 16u;

            auto const aSoa
                = Transforms::from_wmma_input_gfx11(reinterpret_cast<PackedInputT const&>(regsA));
            auto const bSoa
                = Transforms::from_wmma_input_gfx11(reinterpret_cast<PackedInputT const&>(regsB));

            auto const aPacked
                = gfx103_pack_half_pairs(reinterpret_cast<VecT<float16_t, 8> const&>(aSoa));
            auto const bPacked
                = gfx103_pack_half_pairs(reinterpret_cast<VecT<float16_t, 8> const&>(bSoa));
            auto const aPackedOther = gfx103_swap_rows_vec(aPacked);
            auto const bPackedOther = gfx103_swap_rows_vec(bPacked);

            PairBitsT aPairs{};
            PairBitsT bPairs{};

            for(uint32_t i = 0; i < 8u; ++i)
            {
                aPairs[i] = useLowHalf ? gfx103_select_half2_bits(aPacked, aPackedOther, i)
                                       : gfx103_select_half2_bits(aPackedOther, aPacked, i);
                bPairs[i] = useLowHalf ? gfx103_select_half2_bits(bPacked, bPackedOther, i)
                                       : gfx103_select_half2_bits(bPackedOther, bPacked, i);
            }

            auto const aRows
                = gfx103_bpermute_vec(aPairs, (laneId & 0x7u) + (useLowHalf ? 0u : 8u));

            SoaAccumT accumRows{};

#pragma unroll
            for(uint32_t pair = 0; pair < 8u; ++pair)
            {
                uint32_t aWord = aRows[pair];
                uint32_t bWord = bPairs[pair];

                // bf16 upcast: shift low half into f32 exponent position, high half already aligned
                // (tried __builtin_amdgcn_cvt_f32_bf8 but that's a different format entirely)
                uint32_t aLoBits = (aWord & 0xFFFFu) << 16;
                uint32_t aHiBits = aWord & 0xFFFF0000u;
                uint32_t bLoBits = (bWord & 0xFFFFu) << 16;
                uint32_t bHiBits = bWord & 0xFFFF0000u;
                float aLoF = reinterpret_cast<float const&>(aLoBits);
                float aHiF = reinterpret_cast<float const&>(aHiBits);
                float bLoF = reinterpret_cast<float const&>(bLoBits);
                float bHiF = reinterpret_cast<float const&>(bHiBits);

                accumRows[0] = gfx103_fmac_dpp8<0>(aLoF, bLoF, accumRows[0]);
                accumRows[1] = gfx103_fmac_dpp8<1>(aLoF, bLoF, accumRows[1]);
                accumRows[2] = gfx103_fmac_dpp8<2>(aLoF, bLoF, accumRows[2]);
                accumRows[3] = gfx103_fmac_dpp8<3>(aLoF, bLoF, accumRows[3]);
                accumRows[4] = gfx103_fmac_dpp8<4>(aLoF, bLoF, accumRows[4]);
                accumRows[5] = gfx103_fmac_dpp8<5>(aLoF, bLoF, accumRows[5]);
                accumRows[6] = gfx103_fmac_dpp8<6>(aLoF, bLoF, accumRows[6]);
                accumRows[7] = gfx103_fmac_dpp8<7>(aLoF, bLoF, accumRows[7]);

                accumRows[0] = gfx103_fmac_dpp8<0>(aHiF, bHiF, accumRows[0]);
                accumRows[1] = gfx103_fmac_dpp8<1>(aHiF, bHiF, accumRows[1]);
                accumRows[2] = gfx103_fmac_dpp8<2>(aHiF, bHiF, accumRows[2]);
                accumRows[3] = gfx103_fmac_dpp8<3>(aHiF, bHiF, accumRows[3]);
                accumRows[4] = gfx103_fmac_dpp8<4>(aHiF, bHiF, accumRows[4]);
                accumRows[5] = gfx103_fmac_dpp8<5>(aHiF, bHiF, accumRows[5]);
                accumRows[6] = gfx103_fmac_dpp8<6>(aHiF, bHiF, accumRows[6]);
                accumRows[7] = gfx103_fmac_dpp8<7>(aHiF, bHiF, accumRows[7]);
            }

            auto const partnerRows = gfx103_swap_rows_vec(accumRows);
            auto       result      = Transforms::wmma_acc_gfx11_to_soa(regsC);

            if(useLowHalf)
            {
                result[0] += accumRows[0];
                result[1] += accumRows[2];
                result[2] += accumRows[4];
                result[3] += accumRows[6];
                result[4] += partnerRows[0];
                result[5] += partnerRows[2];
                result[6] += partnerRows[4];
                result[7] += partnerRows[6];
            }
            else
            {
                result[0] += partnerRows[1];
                result[1] += partnerRows[3];
                result[2] += partnerRows[5];
                result[3] += partnerRows[7];
                result[4] += accumRows[1];
                result[5] += accumRows[3];
                result[6] += accumRows[5];
                result[7] += accumRows[7];
            }

            return Transforms::soa_to_wmma_acc_gfx11(result);
        }

        template <uint32_t SrcLaneIdx>
        ROCWMMA_DEVICE static inline int32_t
            gfx103_idot4_dpp8(int32_t a, int32_t b, int32_t accum)
        {
            static_assert(SrcLaneIdx < 8u, "DPP8 src lane out of range");
            if constexpr(SrcLaneIdx == 0u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[0,0,0,0,0,0,0,0]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 1u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[1,1,1,1,1,1,1,1]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 2u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[2,2,2,2,2,2,2,2]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 3u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[3,3,3,3,3,3,3,3]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 4u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[4,4,4,4,4,4,4,4]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 5u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[5,5,5,5,5,5,5,5]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else if constexpr(SrcLaneIdx == 6u)
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[6,6,6,6,6,6,6,6]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            else
            {
                asm volatile("v_dot4c_i32_i8_dpp %0, %1, %2 dpp8:[7,7,7,7,7,7,7,7]"
                             : "=v"(accum)
                             : "v"(a), "v"(b), "0"(accum));
            }
            return accum;
        }

        // Each lane holds 2 int32 words (8 int8 elements, K=0..7) after the wmma input layout
        // unpacks the duplicate. The other 8 K-elements (K=8..15) live in the partner half-wave
        // (lanes 0-15 vs 16-31), accessible via permlanex16. We broadcast each A row via bpermute
        // (once per half-wave group), then accumulate 4 v_dot4c_i32_i8_dpp8 products covering
        // all 16 K-elements.
        template <typename InputRegsT, typename AccRegsT>
        ROCWMMA_DEVICE static inline auto gfx103_wmma_i32_16x16x16_i8(InputRegsT const& regsA,
                                                                       InputRegsT const& regsB,
                                                                       AccRegsT const&   regsC)
            -> AccRegsT
        {
            using PackedInputT = VecT<int32_t, 4>;
            using WordPairT    = VecT<int32_t, 2>;
            using SoaAccumT    = VecT<int32_t, 8>;

            static_assert(sizeof(PackedInputT) == sizeof(InputRegsT), "Inconsistent data formats");

            auto const laneId     = threadIdx.x & (Constants::AMDGCN_WAVE_SIZE - 1u);
            auto const useLowHalf = laneId < 16u;

            auto const aSoa = Transforms::from_wmma_input_gfx11(
                reinterpret_cast<PackedInputT const&>(regsA));
            auto const bSoa = Transforms::from_wmma_input_gfx11(
                reinterpret_cast<PackedInputT const&>(regsB));

            // Get the K=8..15 words from the partner half-wave.
            auto const aSoaOther = gfx103_swap_rows_vec(aSoa);
            auto const bSoaOther = gfx103_swap_rows_vec(bSoa);

            // This half-wave holds K=0..7 (or K=8..15 for high lanes); partner holds the rest.
            auto const& aLo = useLowHalf ? aSoa : aSoaOther;
            auto const& bLo = useLowHalf ? bSoa : bSoaOther;
            auto const& aHi = useLowHalf ? aSoaOther : aSoa;
            auto const& bHi = useLowHalf ? bSoaOther : bSoa;

            // Each lane needs the A row for its output column. After permlanex16, aHi already
            // has the partner-half K-data at the same logical lane positions as aLo, so both
            // broadcasts use the same source lane.
            auto const srcLane = (laneId & 0x7u) + (useLowHalf ? 0u : 8u);
            auto const aRowsLo = gfx103_bpermute_vec(aLo, srcLane);
            auto const aRowsHi = gfx103_bpermute_vec(aHi, srcLane);

            SoaAccumT accumRows{};

#pragma unroll
            for(uint32_t word = 0; word < 2u; ++word)
            {
                accumRows[0] = gfx103_idot4_dpp8<0>(aRowsLo[word], bLo[word], accumRows[0]);
                accumRows[1] = gfx103_idot4_dpp8<1>(aRowsLo[word], bLo[word], accumRows[1]);
                accumRows[2] = gfx103_idot4_dpp8<2>(aRowsLo[word], bLo[word], accumRows[2]);
                accumRows[3] = gfx103_idot4_dpp8<3>(aRowsLo[word], bLo[word], accumRows[3]);
                accumRows[4] = gfx103_idot4_dpp8<4>(aRowsLo[word], bLo[word], accumRows[4]);
                accumRows[5] = gfx103_idot4_dpp8<5>(aRowsLo[word], bLo[word], accumRows[5]);
                accumRows[6] = gfx103_idot4_dpp8<6>(aRowsLo[word], bLo[word], accumRows[6]);
                accumRows[7] = gfx103_idot4_dpp8<7>(aRowsLo[word], bLo[word], accumRows[7]);
            }
#pragma unroll
            for(uint32_t word = 0; word < 2u; ++word)
            {
                accumRows[0] = gfx103_idot4_dpp8<0>(aRowsHi[word], bHi[word], accumRows[0]);
                accumRows[1] = gfx103_idot4_dpp8<1>(aRowsHi[word], bHi[word], accumRows[1]);
                accumRows[2] = gfx103_idot4_dpp8<2>(aRowsHi[word], bHi[word], accumRows[2]);
                accumRows[3] = gfx103_idot4_dpp8<3>(aRowsHi[word], bHi[word], accumRows[3]);
                accumRows[4] = gfx103_idot4_dpp8<4>(aRowsHi[word], bHi[word], accumRows[4]);
                accumRows[5] = gfx103_idot4_dpp8<5>(aRowsHi[word], bHi[word], accumRows[5]);
                accumRows[6] = gfx103_idot4_dpp8<6>(aRowsHi[word], bHi[word], accumRows[6]);
                accumRows[7] = gfx103_idot4_dpp8<7>(aRowsHi[word], bHi[word], accumRows[7]);
            }

            auto const partnerRows = gfx103_swap_rows_vec(accumRows);
            auto       result      = Transforms::wmma_acc_gfx11_to_soa(regsC);

            if(useLowHalf)
            {
                result[0] += accumRows[0];
                result[1] += accumRows[2];
                result[2] += accumRows[4];
                result[3] += accumRows[6];
                result[4] += partnerRows[0];
                result[5] += partnerRows[2];
                result[6] += partnerRows[4];
                result[7] += partnerRows[6];
            }
            else
            {
                result[0] += partnerRows[1];
                result[1] += partnerRows[3];
                result[2] += partnerRows[5];
                result[3] += partnerRows[7];
                result[4] += accumRows[1];
                result[5] += accumRows[3];
                result[6] += accumRows[5];
                result[7] += accumRows[7];
            }

            return Transforms::soa_to_wmma_acc_gfx11(result);
        }

        // gfx103 implementations
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return gfx103_wmma_f32_16x16x16_f16(regsA, regsB, regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return gfx103_wmma_i32_16x16x16_i8(regsA, regsB, regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return gfx103_wmma_f32_16x16x16_bf16(regsA, regsB, regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx103_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        // gfx11 implementations
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx103_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx103_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of float16.
                using TypeIn = VecT<float16_t, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of float16.
                using TypeIn  = VecT<float16_t, 16>;
                using TypeOut = VecT<float16_t, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsC)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeOut) == sizeof(decay_t<DRegsT>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(reinterpret_cast<TypeOut&>(result))
                    = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsC)),
                        (bool)AccumBits)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of short.
                using TypeIn = VecT<short, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of short.
                using TypeIn  = VecT<short, 16>;
                using TypeOut = VecT<short, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsC)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeOut) == sizeof(decay_t<DRegsT>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(reinterpret_cast<TypeOut&>(result))
                    = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsC)),
                        (bool)AccumBits)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_gfx11_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32((bool)InputSign,
                                                                  to_native_vector(regsA),
                                                                  (bool)InputSign,
                                                                  to_native_vector(regsB),
                                                                  to_native_vector(regsC),
                                                                  (bool)AccumSign)};
                return result;
            }
        };

        // gfx12 implementations
        // f16
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of float16_t.
                using TypeIn = VecT<float16_t, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_f32_16x16x32_f16((bool)InputAMod,
                                                              to_native_vector(regsA),
                                                              (bool)InputBMod,
                                                              to_native_vector(regsB),
                                                              (int16_t)InputCMod,
                                                              to_native_vector(regsC),
                                                              (bool)ReuseA,
                                                              (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of float16_t.
                using TypeIn  = VecT<float16_t, 8>;
                using TypeOut = VecT<float16_t, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsC)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeOut) == sizeof(decay_t<DRegsT>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(reinterpret_cast<TypeIn&>(result))
                    = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsC)))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float16_t,
                           float16_t,
                           float16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_f16_16x16x32_f16((bool)InputAMod,
                                                              to_native_vector(regsA),
                                                              (bool)InputBMod,
                                                              to_native_vector(regsB),
                                                              (int16_t)InputCMod,
                                                              to_native_vector(regsC),
                                                              (bool)ReuseA,
                                                              (bool)ReuseB)};
                return result;
            }
        };

        // bf16
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of short.
                using TypeIn = VecT<short, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_f32_16x16x32_bf16((bool)InputAMod,
                                                               to_native_vector(regsA),
                                                               (bool)InputBMod,
                                                               to_native_vector(regsB),
                                                               (int16_t)InputCMod,
                                                               to_native_vector(regsC),
                                                               (bool)ReuseA,
                                                               (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 8u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of short.
                using TypeIn  = VecT<short, 8>;
                using TypeOut = VecT<short, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsC)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeOut) == sizeof(decay_t<DRegsT>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(reinterpret_cast<TypeOut&>(result))
                    = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                        to_native_vector(reinterpret_cast<TypeIn const&>(regsC)))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat16_t,
                           bfloat16_t,
                           bfloat16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_bf16_16x16x32_bf16((bool)InputAMod,
                                                                to_native_vector(regsA),
                                                                (bool)InputBMod,
                                                                to_native_vector(regsB),
                                                                (int16_t)InputCMod,
                                                                to_native_vector(regsC),
                                                                (bool)ReuseA,
                                                                (bool)ReuseB)};
                return result;
            }
        };

        // f32
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float32_t,
                           float32_t,
                           float32_t,
                           16u,
                           16u,
                           2u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float32_t, float32_t, float32_t, 16u, 16u, 4u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float32_t,
                           float32_t,
                           float32_t,
                           16u,
                           16u,
                           4u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputAMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputBMod = WmmaInputModifiers::NONE;
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_f32_16x16x4_f32((bool)InputAMod,
                                                             to_native_vector(regsA),
                                                             (bool)InputBMod,
                                                             to_native_vector(regsB),
                                                             (int16_t)InputCMod,
                                                             to_native_vector(regsC),
                                                             (bool)ReuseA,
                                                             (bool)ReuseB)};
                return result;
            }
        };

        // int8
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)InputSign,
                                                                        to_native_vector(regsA),
                                                                        (bool)InputSign,
                                                                        to_native_vector(regsB),
                                                                        to_native_vector(regsC),
                                                                        (bool)AccumSign)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            using PaddedFwd = amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            using PaddedFwd = amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<int8_t,
                           int8_t,
                           int32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegI32x8;
            using BRegsT = VRegI32x8;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x64_iu8((bool)InputSign,
                                                              to_native_vector(regsA),
                                                              (bool)InputSign,
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (bool)ReuseA,
                                                              (bool)ReuseB)};
                return result;
            }
        };

        // uint8
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<uint8_t,
                           uint8_t,
                           uint32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::UNSIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::UNSIGNED;

            // Packed register types
            using ARegsT = VRegUI32x1;
            using BRegsT = VRegUI32x1;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<uint8_t, uint8_t, int32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<uint8_t,
                           uint8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::UNSIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::UNSIGNED;

            // Packed register types
            using ARegsT = VRegUI32x2;
            using BRegsT = VRegUI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of signed 32-bit int
                using TypeIn = VRegI32x2;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    (bool)InputSign,
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    (bool)InputSign,
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC),
                    (bool)AccumSign)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<uint8_t,
                           uint8_t,
                           int32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::UNSIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::UNSIGNED;

            // Packed register types
            using ARegsT = VRegUI32x2;
            using BRegsT = VRegUI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            using PaddedFwd = amdgcn_wmma<uint8_t, uint8_t, int32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<uint8_t,
                           uint8_t,
                           int32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::UNSIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::UNSIGNED;

            // Packed register types
            using ARegsT = VRegUI32x4;
            using BRegsT = VRegUI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            using PaddedFwd = amdgcn_wmma<uint8_t, uint8_t, int32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<uint8_t,
                           uint8_t,
                           int32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::UNSIGNED;
            constexpr static WmmaCtrlFlags ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegUI32x8;
            using BRegsT = VRegUI32x8;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                to_native_vector(result)
                    = {__builtin_amdgcn_wmma_i32_16x16x64_iu8((bool)InputSign,
                                                              to_native_vector(regsA),
                                                              (bool)InputSign,
                                                              to_native_vector(regsB),
                                                              to_native_vector(regsC),
                                                              (bool)ReuseA,
                                                              (bool)ReuseB)};
                return result;
            }
        };

        // fp8 homogeneous inputs
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<float8_t, float8_t, float32_t, 16u, 16u, 16u, GfxTargetId>::exec(
                    concat(regsA, ARegsT{0}),
                    concat(regsB, BRegsT{0}),
                    forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float8_t, float8_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float8_t, float8_t, float32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<float8_t, float8_t, float16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<float8_t, float8_t, float16_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           8u,
                           GfxTargetId,
                           enable_gfx12_t<GfxTargetId>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return amdgcn_wmma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 16u, GfxTargetId>::
                    exec(concat(regsA, ARegsT{0}),
                         concat(regsB, BRegsT{0}),
                         forward<CRegsT const&>(regsC));
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, bfloat8_t, float16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, bfloat8_t, float16_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        // fp8 mixed inputs
        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float8_t, bfloat8_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<float8_t, bfloat8_t, float32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float32_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<float8_t, bfloat8_t, float16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<float8_t, bfloat8_t, float16_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<float8_t,
                           bfloat8_t,
                           float16_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId,
                                              Constants::AMDGCN_ARCH_ID_GFX1200,
                                              Constants::AMDGCN_ARCH_ID_GFX1201>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    to_native_vector(regsC))};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, float8_t, float32_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, float8_t, float32_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float32_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           16u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, float8_t, float16_t, 16u, 16u, 32u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           32u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            using PaddedFwd
                = amdgcn_wmma<bfloat8_t, float8_t, float16_t, 16u, 16u, 64u, GfxTargetId>;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return PaddedFwd::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           64u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 8>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        template <uint32_t GfxTargetId>
        struct amdgcn_wmma<bfloat8_t,
                           float8_t,
                           float16_t,
                           16u,
                           16u,
                           128u,
                           GfxTargetId,
                           enable_target_id_t<GfxTargetId, Constants::AMDGCN_ARCH_ID_GFX1250>>
        {
            constexpr static WmmaInputModifiers InputCMod = WmmaInputModifiers::NONE;
            constexpr static WmmaCtrlFlags      ReuseA    = WmmaCtrlFlags::NO_REUSE;
            constexpr static WmmaCtrlFlags      ReuseB    = WmmaCtrlFlags::NO_REUSE;

            // Packed register types
            using ARegsT = VRegF32x16;
            using BRegsT = VRegF32x16;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 16>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>),
                              "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>),
                              "Inconsistent data formats");

                DRegsT result;
                to_native_vector(result) = {__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8(
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsA)),
                    to_native_vector(reinterpret_cast<TypeIn const&>(regsB)),
                    (int16_t)InputCMod,
                    to_native_vector(regsC),
                    (bool)ReuseA,
                    (bool)ReuseB)};
                return result;
            }
        };

        // Derivative implementations
        template <uint32_t GfxTargetId, uint32_t BlockK>
        struct amdgcn_wmma<hfloat16_t,
                           hfloat16_t,
                           float32_t,
                           16u,
                           16u,
                           BlockK,
                           GfxTargetId,
                           enable_gfx11_gfx12_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, BlockK, GfxTargetId>
        {
        };

        template <uint32_t GfxTargetId, uint32_t BlockK>
        struct amdgcn_wmma<hfloat16_t,
                           hfloat16_t,
                           hfloat16_t,
                           16u,
                           16u,
                           BlockK,
                           GfxTargetId,
                           enable_gfx11_gfx12_t<GfxTargetId, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, BlockK, GfxTargetId>
        {
        };

    } // namespace detail

    namespace MmaTraits_impl
    {
        template <typename WmmaOp>
        struct is_wmma : public false_type
        {
        };

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_in,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct is_wmma<
            detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_in, BlockM_In, BlockN_In, BlockK_In>>
            : public true_type
        {
        };

        template <typename WmmaOp>
        constexpr static bool is_wmma_v = is_wmma<WmmaOp>::value;

        // All of the overrides won't have the Unsupported tag
        template <typename WmmaOp, typename Enabler = void>
        struct is_wmma_supported : public true_type
        {
        };

        // Default implementation will have the Unsupported tag
        template <typename WmmaOp>
        struct is_wmma_supported<
            WmmaOp,
            enable_if_t<is_same_v<typename WmmaOp::Unsupported, detail::Unsupported>>>
            : public false_type
        {
        };

        template <typename WmmaOp>
        constexpr static bool is_wmma_supported_v = is_wmma_supported<WmmaOp>::value;

        template <typename MfmaOp>
        struct wmma_traits;

        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_In,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct wmma_traits<
            detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>>
        {
            // Base implementation
            using Impl = detail::
                amdgcn_wmma<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK_In>;

            // Operand types
            using InputTA  = InputTA_In;
            using InputTB  = InputTB_In;
            using ComputeT = ComputeT_In;

            // Raw input / output types
            using ARegsT = typename Impl::ARegsT;
            using BRegsT = typename Impl::BRegsT;
            using CRegsT = typename Impl::CRegsT;
            using DRegsT = typename Impl::DRegsT;

            // Geometric block sizes
            constexpr static uint32_t BlockM = BlockM_In;
            constexpr static uint32_t BlockN = BlockN_In;
            constexpr static uint32_t BlockK = BlockK_In;

            // Vector sizes per block (packed)
            constexpr static uint32_t BlockSizeA = VecTraits<ARegsT>::size();
            constexpr static uint32_t BlockSizeB = VecTraits<BRegsT>::size();
            constexpr static uint32_t BlockSizeC = VecTraits<CRegsT>::size();

            // Backend flags
            constexpr static bool is_wmma      = is_wmma_v<Impl>;
            constexpr static bool is_mfma      = false;
            constexpr static bool is_supported = is_wmma_supported_v<Impl>;
        };

        // MmaTraits implemented for mfma backend
        template <typename MmaOp>
        struct MmaTraits<MmaOp, enable_if_t<is_wmma_v<MmaOp>>> : public wmma_traits<MmaOp>
        {
        };

    } // namespace MmaTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_WMMA_IMPL_HPP
