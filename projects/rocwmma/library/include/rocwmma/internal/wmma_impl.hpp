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
#include "types.hpp"
#include "utility/type_traits.hpp"
#include "wmma/resolve.hpp"

namespace rocwmma
{
    namespace detail
    {
        struct Unsupported;

        // Enabler for targets.
        // Given a TargetId, enable if it exists in the TargetIds list
        template <uint32_t TargetId, uint32_t... TargetIds>
        using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;

        // Enabler for the legacy gfx11 WMMA register layout
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

        // Enabler for gfx117x and gfx12 WMMA register layouts
        template <uint32_t TargetId, bool Cond = true>
        using enable_gfx12_t
            = enable_if_t<contains_number_v<uint32_t,
                                            TargetId,
                                            Constants::AMDGCN_ARCH_ID_GFX1170,
                                            Constants::AMDGCN_ARCH_ID_GFX1171,
                                            Constants::AMDGCN_ARCH_ID_GFX1200,
                                            Constants::AMDGCN_ARCH_ID_GFX1201,
                                            Constants::AMDGCN_ARCH_ID_GFX1250> && Cond>;

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
                                            Constants::AMDGCN_ARCH_ID_GFX1170,
                                            Constants::AMDGCN_ARCH_ID_GFX1171,
                                            Constants::AMDGCN_ARCH_ID_GFX1200,
                                            Constants::AMDGCN_ARCH_ID_GFX1201,
                                            Constants::AMDGCN_ARCH_ID_GFX1250> && Cond>;

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

        // Operations described in wmma/ (native descriptors, capability lists and
        // K padding). Only Enabler = void resolves; other enablers keep selecting
        // the unsupported primary template.
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK,
                  uint32_t GfxTargetId>
        struct amdgcn_wmma<
            InputTA,
            InputTB,
            ComputeT,
            BlockM,
            BlockN,
            BlockK,
            GfxTargetId,
            enable_if_t<!is_void_v<
                wmma_resolve_t<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK, GfxTargetId>>>>
            : public wmma_resolve_t<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK, GfxTargetId>
        {
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
            REUSE = true,
        };

        enum struct WmmaInputModifiers : int16_t
        {
            // A, B, C input modifiers
            NONE = 0,
            NEGATE = 1,

            // C input only modifiers
            ABS = 2,
            NEG_ABS = 3
        };

        // gfx11 implementations
        // gfx12 implementations
        // f16
        // bf16
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
                  uint32_t BlockK_In,
                  uint32_t GfxTargetId_In,
                  typename Enabler_In>
        struct is_wmma<detail::amdgcn_wmma<InputTA_In,
                                           InputTB_In,
                                           ComputeT_in,
                                           BlockM_In,
                                           BlockN_In,
                                           BlockK_In,
                                           GfxTargetId_In,
                                           Enabler_In>> : public true_type
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
                  uint32_t BlockK_In,
                  uint32_t GfxTargetId_In,
                  typename Enabler_In>
        struct wmma_traits<detail::amdgcn_wmma<InputTA_In,
                                               InputTB_In,
                                               ComputeT_In,
                                               BlockM_In,
                                               BlockN_In,
                                               BlockK_In,
                                               GfxTargetId_In,
                                               Enabler_In>>
        {
            // Base implementation
            using Impl = detail::amdgcn_wmma<InputTA_In,
                                             InputTB_In,
                                             ComputeT_In,
                                             BlockM_In,
                                             BlockN_In,
                                             BlockK_In,
                                             GfxTargetId_In,
                                             Enabler_In>;

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
