// swmmac.hpp — SWMMAC public interface (follows mfma.hpp pattern)
// AMD vendor-level integration into rocWMMA core.
//
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "mma.hpp"
#include "mma_selector.hpp"
#include "mma_traits.hpp"
#include "swmmac_impl.hpp"
#include "swmmac_traits.hpp"

namespace rocwmma {

// Expose SWMMAC implementation backend
template <typename InputTA,   typename InputTB,   typename ComputeT,
          uint32_t BlockM,    uint32_t BlockN,    uint32_t BlockK>
using Swmmac_impl = detail::amdgcn_swmmac<InputTA, InputTB, ComputeT,
                                          BlockM, BlockN, BlockK>;

// ============================================================================
// Architecture-dispatching backend selector
//
// gfx12 (RDNA4): SWMMAC — peak throughput, asymmetric A/B registers
// gfx11 (RDNA3): WMMA fallback — functional, symmetric registers, K=16
// ============================================================================
template <typename InputTA,   typename InputTB,   typename ComputeT,
          uint32_t BlockM,    uint32_t BlockN,
          uint32_t BlockKTest = 64u>
struct SwmmacSelector
#if ROCWMMA_HAS_SWMMAC
    : public MmaSelector<Swmmac_impl, InputTA, InputTB, ComputeT,
                         BlockM, BlockN, BlockKTest> {};
#else
    // gfx11 fallback: use WMMA (symmetric registers, K=16 max)
    : public MmaSelector<Swmmac_impl, InputTA, InputTB, ComputeT,
                         BlockM, BlockN, 16u> {};
#endif

// Unified SWMMAC traits — backend + IO + driver
template <typename InputTA,   typename InputTB,   typename ComputeT,
          uint32_t BlockM,    uint32_t BlockN,    uint32_t BlockK>
using SwmmacTraitsT = SwmmacTraits<InputTA, InputTB, ComputeT,
                                   BlockM, BlockN, BlockK>;

// SWMMAC interface through Mma<>
template <uint32_t FragM,      uint32_t FragN,      uint32_t FragK,
          typename InputTA,    typename InputTB,    typename ComputeT,
          uint32_t BlockM,     uint32_t BlockN,
          uint32_t BlockK      = FragK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
struct Swmmac
    : public Mma<FragM, FragN, FragK,
                 typename SwmmacSelector<InputTA, InputTB, ComputeT,
                                         BlockM, BlockN, BlockK>::SelectedOp,
                 AccumPolicy>
{
    using SelectedOp =
        typename SwmmacSelector<InputTA, InputTB, ComputeT,
                                BlockM, BlockN, BlockK>::SelectedOp;
};

} // namespace rocwmma
