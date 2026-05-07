// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/utility/amd_semaphore.hpp"
#include "ck/utility/amd_named_barrier.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch,
          TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
struct GridwiseGemmPipeline_v5;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v5<1>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds_async_load();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // block_sync_lds_async_load();
                block_sync_lds();
                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds_async_load();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <>
struct GridwiseGemmPipeline_v5<1,
                               TensorLoadOption::CLUSTER_ASYNC_MULTICAST_LDS_LOAD,
                               TensorLoadOption::DEFAULT_LOAD>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
#if defined(__gfx13__)
                __builtin_amdgcn_s_wait_asynccnt(0);
                __builtin_amdgcn_s_barrier_signal(-3);
                __builtin_amdgcn_s_barrier_wait(-3);
#endif
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // block_sync_lds_async_load();
                block_sync_lds();
                a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
#if defined(__gfx13__)
            __builtin_amdgcn_s_wait_asynccnt(0);
            __builtin_amdgcn_s_barrier_signal(-3);
            __builtin_amdgcn_s_barrier_wait(-3);
#endif
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <index_t NumPrefetch>
struct GridwiseGemmPipeline_Wavegroup_v5;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_Wavegroup_v5<1>
{
    static constexpr auto I0 = Number<0>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        __shared__ NamedBarrier<4> barrierLds;
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            barrierLds.init();
            barrierLds.join();
        }

        semaDataReady.init();
        semaDataFree.init();
        __syncthreads();

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
            b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    barrierLds.template sync_lds<true>();
                    semaDataReady.template signal<0>();

                    semaDataFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();

                    a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf);
                    b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            {
                barrierLds.template sync_lds<true>();
                semaDataReady.template signal<0>();
            }
        }

        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            c_thread_buf.Clear();

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    semaDataReady.template wait<0>();
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
                    semaDataFree.template signal<0>();

                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            {
                semaDataReady.template wait<0>();
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
            }
        }
    }
};

} // namespace ck
