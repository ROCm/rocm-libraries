// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp"

namespace ck_tile {

// A whole MX MoE layer in one persistent launch, gfx1250, scheduled as tasks:
//
//   gate_up task (row tile m, gate_up column tile n): GateUpKernel
//       (MoeFlatmmKind::kFFN_gemm1_gate_up_mx_quant) writes its 32 x (N_Tile / 2) part of the
//       MXFP4 intermediate, then publishes it by incrementing ready[m].
//   down task (row tile m, output column tile n): waits until ready[m] counts every gate_up
//       tile of m, then runs DownKernel (kFFN_gemm2: weighted fp32 atomics into Y, or
//       kFFN_gemm2_per_slot: plain store per (token, slot), combined afterwards) over all of K.
//
// The task list holds every gate_up task, row tile by row tile, then every down task in the same
// order. Workgroup w runs tasks w, w + grid, w + 2 grid, ...: it finishes all its gate_up tasks
// before any down task, so a down task only waits for gate_up tasks of workgroups that are
// running, provided every workgroup of the grid is resident at once. GridSize() keeps the grid
// within that (occupancy times compute units). Taking row tiles in order lets the first row
// tiles' down tasks run while the last gate_up tasks are still running.
//
// Synchronization cost decides whether one launch beats two, so it is kept to a store wait, a
// barrier and a relaxed atomic increment per gate_up task (the gate_up epilogue writes the
// intermediate with device-scope stores, visible device-wide once complete, so no cache writeback
// is needed), and a compare-exchange poll per down task. On gfx1250 a shared task counter, a
// device-scope release (cache writeback) or acquire (cache invalidate) per task, or counters
// sharing a cache line each cost more than the tasks' work at decode sizes.
//
// Workspace (int32, zero when first allocated; the kernel leaves it zero): ready[max row tiles],
// then waiters[max row tiles], kCounterStride words apart. The last down task of a row tile to
// pass its wait resets that tile's counters, so consecutive launches on a stream need no
// clearing.
template <typename GateUpKernel_, typename DownKernel_>
struct MoeMxPersistentKernel
{
    using GateUpKernel    = remove_cvref_t<GateUpKernel_>;
    using DownKernel      = remove_cvref_t<DownKernel_>;
    using TilePartitioner = typename GateUpKernel::TilePartitioner;

    static_assert(GateUpKernel::Kind == MoeFlatmmKind::kFFN_gemm1_gate_up_mx_quant &&
                      (DownKernel::Kind == MoeFlatmmKind::kFFN_gemm2 ||
                       DownKernel::Kind == MoeFlatmmKind::kFFN_gemm2_per_slot),
                  "gate_up quantizes to MXFP4; down is gemm2 or gemm2_per_slot");
    static_assert(GateUpKernel::kBlockSize == DownKernel::kBlockSize &&
                      TilePartitioner::MPerBlock == DownKernel::TilePartitioner::MPerBlock,
                  "both task kinds must use the same block and row tile");

    static constexpr index_t kBlockSize      = GateUpKernel::kBlockSize;
    static constexpr index_t MPerBlock       = TilePartitioner::MPerBlock;
    static constexpr index_t GateUpNPerBlock = TilePartitioner::NPerBlock;
    static constexpr index_t DownNPerBlock   = DownKernel::TilePartitioner::NPerBlock;
    // int32 words between counters: each in its own 256 bytes, so polls of different row tiles
    // go to different L2 channels instead of queueing on one line.
    static constexpr index_t kCounterStride = 64;

    template <typename GateUpKargs, typename DownKargs>
    struct Kargs
    {
        GateUpKargs gate_up;
        DownKargs down;
        int32_t* workspace;
        index_t max_row_tiles; // row tiles the workspace has counters for
    };

    // int32 words of workspace for up to max_row_tiles row tiles.
    CK_TILE_HOST static constexpr std::size_t WorkspaceWords(index_t max_row_tiles)
    {
        return 2 * std::size_t(max_row_tiles) * kCounterStride;
    }

    template <typename GateUpHostArgs, typename DownHostArgs>
    CK_TILE_HOST static auto
    MakeKargs(const GateUpHostArgs& gate_up, const DownHostArgs& down, void* workspace)
    {
        auto g = GateUpKernel::MakeKernelArgs(gate_up);
        auto d = DownKernel::MakeKernelArgs(down);
        return Kargs<decltype(g), decltype(d)>{
            g, d, static_cast<int32_t*>(workspace), integer_divide_ceil(gate_up.M, MPerBlock)};
    }

    template <typename K>
    CK_TILE_HOST static bool IsSupportedArgument(const K& kargs)
    {
        // The down tasks have no device-scope acquire, so no cache may hold part of a row
        // tile's intermediate before that row tile is complete. Caches fill 256 bytes at a time
        // on gfx1250, so every slot row (stride / 2 bytes) and every row tile's scale block
        // (K bytes) must be whole 256-byte blocks: neighbouring slot rows belong to other row
        // tiles, and with a row stride that is not a multiple of 256 bytes the down projection
        // reads wrong values.
        if(kargs.workspace == nullptr || kargs.gate_up.N != 2 * kargs.down.K ||
           kargs.down.K % 256 != 0 || kargs.gate_up.stride_C != kargs.down.stride_A ||
           kargs.down.stride_A % 512 != 0 || kargs.gate_up.M != kargs.down.M ||
           kargs.gate_up.N % GateUpNPerBlock != 0 || kargs.down.N % DownNPerBlock != 0)
        {
            std::cerr << "MoeMxPersistentKernel: needs a workspace, N of gate_up = 2 x the down "
                         "projection's K (a multiple of 256), the intermediate's row stride "
                         "(stride_C of gate_up = stride_A of down) a multiple of 512, and the "
                         "same sorted rows for both"
                      << std::endl;
            return false;
        }
        return GateUpKernel::IsSupportedArgument(kargs.gate_up) &&
               DownKernel::IsSupportedArgument(kargs.down);
    }

    CK_TILE_HOST static auto BlockSize() { return GateUpKernel::BlockSize(); }

    // Workgroups that fit on the device at once (more would only queue behind them), and no
    // more than there are tasks (row tiles bounded by the sorted rows).
    template <typename K>
    CK_TILE_HOST static auto GridSize(const K& kargs)
    {
        hipDeviceProp_t prop;
        int device = 0;
        HIP_CHECK_ERROR(hipGetDevice(&device));
        HIP_CHECK_ERROR(hipGetDeviceProperties(&prop, device));
        int per_cu = 0;
        HIP_CHECK_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_cu,
            reinterpret_cast<const void*>(kentry<1, MoeMxPersistentKernel, K>),
            BlockSize().x,
            0));
        // No more workgroups than tasks: idle ones would only take launch slots.
        const index_t row_tiles = integer_divide_ceil(kargs.gate_up.M, MPerBlock);
        const index_t tasks =
            row_tiles * (kargs.gate_up.N / GateUpNPerBlock + kargs.down.N / DownNPerBlock);
        return dim3(min(prop.multiProcessorCount * max(per_cu, 1), max(tasks, 1)));
    }

    [[nodiscard]] CK_TILE_HOST static std::string GetName()
    {
        return "moe_mx_persistent_" + GateUpKernel::GetName();
    }

    template <typename K>
    CK_TILE_DEVICE void operator()(K kargs) const
    {
        int32_t* ready   = kargs.workspace;
        int32_t* waiters = ready + kargs.max_row_tiles * kCounterStride;

        const index_t row_tiles = integer_divide_ceil(kargs.gate_up.p_max_token_id[0], MPerBlock);
        const index_t gate_up_n = kargs.gate_up.N / GateUpNPerBlock;
        const index_t down_n    = kargs.down.N / DownNPerBlock;
        const index_t gate_up_tasks = row_tiles * gate_up_n;
        const index_t tasks         = gate_up_tasks + row_tiles * down_n;

        for(index_t task = blockIdx.x; task < tasks; task += gridDim.x)
        {
            block_sync_lds(); // the previous task's LDS reads before this task's LDS writes
            if(task < gate_up_tasks)
            {
                const index_t m = task / gate_up_n;
                GateUpKernel{}(kargs.gate_up, m, task % gate_up_n);
                // Publish this tile of the intermediate: all its device-scope stores complete,
                // then one count for the row tile.
                asm volatile("s_wait_storecnt 0x0" ::: "memory");
                __builtin_amdgcn_s_barrier();
                if(threadIdx.x == 0)
                    __scoped_atomic_fetch_add(
                        &ready[m * kCounterStride], 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            }
            else
            {
                const index_t t = task - gate_up_tasks;
                const index_t m = t / down_n;
                if(threadIdx.x == 0)
                {
                    // Poll with a compare-exchange that never changes the count: a real atomic,
                    // done at the L2. (An idempotent fetch_add(0) is folded into a plain load,
                    // which on gfx1250 can keep returning a stale count.)
                    for(;;)
                    {
                        int32_t seen = gate_up_n;
                        if(__scoped_atomic_compare_exchange_n(&ready[m * kCounterStride],
                                                              &seen,
                                                              gate_up_n,
                                                              false,
                                                              __ATOMIC_RELAXED,
                                                              __ATOMIC_RELAXED,
                                                              __MEMORY_SCOPE_DEVICE))
                            break;
                        __builtin_amdgcn_s_sleep(1);
                    }
                    // The last waiter of this row tile resets its counters for the next launch.
                    if(__scoped_atomic_fetch_add(&waiters[m * kCounterStride],
                                                 1,
                                                 __ATOMIC_RELAXED,
                                                 __MEMORY_SCOPE_DEVICE) == down_n - 1)
                    {
                        __scoped_atomic_store_n(&waiters[m * kCounterStride],
                                                0,
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                        __scoped_atomic_store_n(
                            &ready[m * kCounterStride], 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                    }
                }
                __builtin_amdgcn_s_barrier();
                // Workgroup acquire: keeps the compiler from moving the intermediate's loads above
                // the wait. No device-scope acquire is needed (an agent-scope cache invalidate
                // costs more than the down task at decode sizes): no workgroup reads any 256-byte
                // block of row tile m's intermediate before ready[m] is complete (see the layout
                // checks in IsSupportedArgument; padded sorted rows are masked without a memory
                // access), so no cache holds a stale copy, and the launch drops lines left by
                // earlier launches.
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
                DownKernel{}(kargs.down, m, t % down_n);
            }
        }
    }
};

} // namespace ck_tile
