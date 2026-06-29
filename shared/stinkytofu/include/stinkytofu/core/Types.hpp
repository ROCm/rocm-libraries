/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * ************************************************************************ */
#pragma once

#include <array>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

namespace stinkytofu {
// Error codes for StinkyIRConverter operations
enum class StinkyErrorCode : int {
    SUCCESS = 0,
    PASSCTX_EMPTY = 1,
    PARSE_ERROR = 2,
};

/// GEMM-specific tile configuration
/// This configuration is specific to GEMM kernels and their tiling strategy
/// Note: WavefrontSize is NOT included here as it's derived from architecture,
///       not a user-configurable parameter. Use getWaveFrontSize(arch) to query it.
struct GemmTileConfig {
    std::array<int, 3> arch{0, 0, 0};  ///< GPU architecture [gfx, major, minor]
    uint32_t TileA0;                   ///< Tile size for A dimension 0
    uint32_t TileB0;                   ///< Tile size for B dimension 0
    uint32_t TileM0;                   ///< Tile size for M dimension 0
    uint32_t NumGRA;                   ///< Number of global read A
    uint32_t NumGRB;                   ///< Number of global read B
    uint32_t NumGRM;                   ///< Number of global read M
    uint32_t NumWaves;                 ///< Number of waves
};

/// Pass-specific feature configuration
/// Categorizes optimization behaviors into semantics, properties, and features
struct PassFeatureConfig {
    /// Loop structure and unrolling properties
    /// These are code structure PROPERTIES (not optional features)
    struct LoopConfig {
        bool unrollGemm = false;  ///< Whether GEMM loops are unrolled
    };

    /// DAG scheduler switches.
    /// DS read reorder strategy for WMMA operand scheduling.
    enum class DsReadOrder {
        ProgramOrder,    ///< No reorder (AABB)
        Ascending,       ///< Pair by WMMA affinity: A0 B0 A1 B1
        AscendingCache,  ///< Zigzag for cache reuse: A0 B0 B1 A1
    };

    struct DagFeatures {
        bool distributeGlobalRead = false;                 ///< Enable global read distribution
        DsReadOrder dsReadOrder = DsReadOrder::Ascending;  ///< DS read reorder strategy
        /// Max in-flight tensor_load_to_lds credits (HW queue depth, to connect to sw math cycles).
        /// 0 disables the throttle (current behavior).
        int globalReadQueueDepth = 0;
        /// Modeled cycles until one tensor_load_to_lds credit frees. Fed from the
        /// cost/cycle model; varies with layout and problem size.
        int globalReadDrainLatency = 0;
        int dsReadQueueDepth = 0;
        int dsReadDrainLatency = 0;
        int dsReadPerWmma = INT_MAX;
    };

    LoopConfig loopConfig;
    DagFeatures dagFeatures;
};

/// VGPR MSB encoding mode supported by the toolchain.
enum class VgprMsbMode : uint8_t {
    None,   ///< Toolchain does not support `s_set_vgpr_msb`
    Msb8,   ///< 8-bit form only (`s_set_vgpr_msb 0`)
    Msb16,  ///< 16-bit form (`s_set_vgpr_msb 0x0101`) — packs prev + curr MSB
};

/// Toolchain capabilities discovered by probing the assembler (via comgr or
/// rocisa's initAsmCaps).  Populated either by the rocisa conversion layer or
/// by ToolchainCaps::probe() for the standalone path.
struct AsmCapsConfig {
    VgprMsbMode vgprMsbMode = VgprMsbMode::None;
};
}  // namespace stinkytofu
