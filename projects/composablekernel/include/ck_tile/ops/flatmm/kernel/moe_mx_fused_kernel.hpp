// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp"

namespace ck_tile {

// A whole MX MoE layer in one launch, gfx1250:
//
//   Y[t] += sum_s w[t, s] * down_e( quant_mxfp4( SiLU(X[t] W_gate_e) * (X[t] W_up_e) ) )
//
// Each workgroup takes one sorted row tile (M_Tile rows of one expert) and one slice of the
// intermediate dimension (slice columns, a multiple of the down projection's K tile):
//   1. GateUpKernel (MoeFlatmmKind::kFFN_gemm1_gate_up_mx_quant) computes the slice's gate_up
//      tiles for the row tile and writes the intermediate as MXFP4 (data per token slot, scales
//      per sorted row). Only this workgroup reads that part of the intermediate, right after it
//      is written, so it stays in the caches.
//   2. A workgroup fence and barrier make those writes visible to the whole workgroup.
//   3. DownKernel (MoeFlatmmKind::kFFN_gemm2) computes every output column tile of the down
//      projection with K restricted to the slice, and adds the routing-weighted partial sum
//      into Y with fp32 atomics. The slices of a row tile add up to the full product.
// Splitting the intermediate gives decode-size batches, which have few row tiles, enough
// workgroups. Y must be zero before the launch.
template <typename GateUpKernel_, typename DownKernel_>
struct MoeMxFusedKernel
{
    using GateUpKernel    = remove_cvref_t<GateUpKernel_>;
    using DownKernel      = remove_cvref_t<DownKernel_>;
    using TilePartitioner = typename GateUpKernel::TilePartitioner;

    static_assert(GateUpKernel::Kind == MoeFlatmmKind::kFFN_gemm1_gate_up_mx_quant &&
                      DownKernel::Kind == MoeFlatmmKind::kFFN_gemm2,
                  "phase 1 quantizes gate_up to MXFP4, phase 2 is the atomic gemm2");
    static_assert(GateUpKernel::kBlockSize == DownKernel::kBlockSize &&
                      TilePartitioner::MPerBlock == DownKernel::TilePartitioner::MPerBlock,
                  "both phases must use the same block and row tile");

    static constexpr index_t kBlockSize = GateUpKernel::kBlockSize;
    static constexpr index_t MPerBlock  = TilePartitioner::MPerBlock;
    // Intermediate columns one gate_up tile produces (gate and up interleaved in its N tile).
    static constexpr index_t InterPerGateUpTile = TilePartitioner::NPerBlock / 2;
    static constexpr index_t DownNPerBlock      = DownKernel::TilePartitioner::NPerBlock;
    static constexpr index_t DownKPerBlock      = DownKernel::TilePartitioner::KPerBlock;

    template <typename GateUpKargs, typename DownKargs>
    struct Kargs
    {
        GateUpKargs gate_up;
        DownKargs down;
        index_t inter_slices; // slices of the intermediate dimension per row tile
        index_t slice;        // intermediate columns per slice
    };

    template <typename GateUpHostArgs, typename DownHostArgs>
    CK_TILE_HOST static auto
    MakeKargs(const GateUpHostArgs& gate_up, const DownHostArgs& down, index_t inter_slices)
    {
        auto g = GateUpKernel::MakeKernelArgs(gate_up);
        auto d = DownKernel::MakeKernelArgs(down);
        return Kargs<decltype(g), decltype(d)>{
            g, d, inter_slices, inter_slices > 0 ? down.K / inter_slices : 0};
    }

    template <typename K>
    CK_TILE_HOST static bool IsSupportedArgument(const K& kargs)
    {
        const index_t inter = kargs.down.K;
        if(kargs.inter_slices <= 0 || inter % kargs.inter_slices != 0 ||
           kargs.slice % DownKPerBlock != 0 || kargs.slice % InterPerGateUpTile != 0 ||
           kargs.gate_up.N != 2 * inter || kargs.down.N % DownNPerBlock != 0 ||
           kargs.gate_up.M != kargs.down.M)
        {
            std::cerr << "MoeMxFusedKernel: needs N of gate_up = 2 x intermediate, intermediate "
                         "split into slices that are multiples of "
                      << DownKPerBlock << " and " << InterPerGateUpTile
                      << ", and the same sorted rows for both phases" << std::endl;
            return false;
        }
        return GateUpKernel::IsSupportedArgument(kargs.gate_up) &&
               DownKernel::IsSupportedArgument(kargs.down);
    }

    CK_TILE_HOST static auto BlockSize() { return GateUpKernel::BlockSize(); }

    template <typename K>
    CK_TILE_HOST static auto GridSize(const K& kargs)
    {
        return dim3(integer_divide_ceil(kargs.gate_up.M, MPerBlock) * kargs.inter_slices);
    }

    [[nodiscard]] CK_TILE_HOST static std::string GetName()
    {
        return "moe_mx_fused_" + GateUpKernel::GetName();
    }

    template <typename K>
    CK_TILE_DEVICE void operator()(K kargs) const
    {
        const index_t iM = blockIdx.x / kargs.inter_slices;
        const index_t s  = blockIdx.x % kargs.inter_slices;
        if(iM * MPerBlock >= kargs.gate_up.p_max_token_id[0])
            return; // trailing padding of the sorted token list

        // Phase 1: this slice of SiLU(gate) * up for the row tile, written as MXFP4.
        const index_t n_begin = s * kargs.slice / InterPerGateUpTile;
        const index_t n_end   = n_begin + kargs.slice / InterPerGateUpTile;
        for(index_t iN = n_begin; iN < n_end; ++iN)
        {
            block_sync_lds(); // the previous tile's LDS reads before this tile's LDS writes
            GateUpKernel{}(kargs.gate_up, iM, iN);
        }

        // The down projection reads what this workgroup just wrote.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

        // Phase 2: every output column tile, K restricted to the slice; weighted atomic add.
        const index_t n_tiles = kargs.down.N / DownNPerBlock;
        for(index_t iN = 0; iN < n_tiles; ++iN)
        {
            block_sync_lds();
            DownKernel{}(kargs.down, iM, iN, s * kargs.slice, kargs.slice);
        }
    }
};

} // namespace ck_tile
