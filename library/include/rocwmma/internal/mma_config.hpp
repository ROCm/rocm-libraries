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
#ifndef ROCWMMA_MMA_CONFIG_HPP
#define ROCWMMA_MMA_CONFIG_HPP

#include "layout/register_layout_transforms.hpp"
#include "io_config.hpp"
#include "mfma.hpp"
#include "pack_util.hpp"
#include "types.hpp"
#include "wmma.hpp"

namespace rocwmma
{
    template <uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              typename DataLayoutA,
              typename DataLayoutB,
              typename DataLayoutC,
              typename DataLayoutD>
    struct MmaConfig
    {
        using IOConfigA = IOConfig<matrix_a, FragM, FragN, FragK, InputTA, DataLayoutA>;
        using IOConfigB = IOConfig<matrix_b, FragM, FragN, FragK, InputTB, DataLayoutB>;
        using IOConfigC = IOConfig<accumulator, FragM, FragN, FragK, ComputeT, DataLayoutC>;
        using IOConfigD = IOConfig<accumulator, FragM, FragN, FragK, ComputeT, DataLayoutD>;

        using IOLayoutA = typename IOConfigA::IOLayout;
        using IOLayoutB = typename IOConfigB::IOLayout;
        using IOLayoutC = typename IOConfigC::IOLayout;
        using IOLayoutD = typename IOConfigD::IOLayout;

        constexpr static uint32_t MmaDimM = IOLayoutA::MmaDim;
        constexpr static uint32_t MmaDimN = IOLayoutB::MmaDim;

        // Sanity checks:
        // - For now, MmaDimM/N/Acc must match
        // - MmaLayout for input A/B must match
        // - MmaLayout for accumulators must match
        static_assert(MmaDimM == MmaDimN, "MmaDims must match");
        static_assert((MmaDimN == IOLayoutC::MmaDim) && (MmaDimN == IOLayoutD::MmaDim), "Mismatched accumulator MmaDim");
        static_assert(is_layout_same_v<typename IOLayoutA::MmaLayout, typename IOLayoutB::MmaLayout>, "Input fragment register layouts do not match");
        static_assert(is_layout_same_v<typename IOLayoutC::MmaLayout, typename IOLayoutD::MmaLayout>, "Accumulator fragment register layouts do not match");

        // Check valid mma layouts
        // TODO: eventually should enforce
        // static_assert(layout_traits<typename IOLayoutA::MmaLayout>::is_valid, "Invalid MmaLayout for matrix_a");
        // static_assert(layout_traits<typename IOLayoutB::MmaLayout>::is_valid, "Invalid MmaLayout for matrix_b");
        // static_assert(layout_traits<typename IOLayoutC::MmaLayout>::is_valid, "Invalid MmaLayout for accumulator C");
        // static_assert(layout_traits<typename IOLayoutD::MmaLayout>::is_valid, "Invalid MmaLayout for accumulator D");

        // Input transforms
        using PreMmaXFormA = register_layout_transform<typename IOLayoutA::FragmentLayout,
                                                       typename IOLayoutA::MmaLayout,
                                                       1u>;

        using PreMmaXFormB = register_layout_transform<typename IOLayoutB::FragmentLayout,
                                                       typename IOLayoutB::MmaLayout,
                                                       1u>;

        using PreMmaXFormC = register_layout_transform<typename IOLayoutC::FragmentLayout,
                                                       typename IOLayoutC::MmaLayout,
                                                       1u>;

        // Output accum transform
        using PostMmaXFormD = register_layout_transform<typename IOLayoutD::MmaLayout,
                                                        typename IOLayoutD::FragmentLayout,
                                                        1u>;

        // Pack util
        using PackB = typename IOConfigB::PackUtil;
        using PackA = typename IOConfigA::PackUtil;
        using PackC = typename IOConfigC::PackUtil;
        using PackD = typename IOConfigD::PackUtil;

        // Gfx9 uses MFMA, gfx11/12 uses WMMA
        using Mma = conditional_t<(bool)ROCWMMA_ARCH_GFX9,
                                  Mfma<FragM, FragN, FragK, InputTA, InputTB, ComputeT, MmaDimM, MmaDimN>,
                                  Wmma<FragM, FragN, FragK, InputTA, InputTB, ComputeT, MmaDimM, MmaDimN>>;
    };

} // namespace rocwmma

#endif // ROCWMMA_MMA_CONFIG_HPP
