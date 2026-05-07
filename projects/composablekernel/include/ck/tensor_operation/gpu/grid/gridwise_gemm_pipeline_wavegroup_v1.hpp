// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/utility/amd_semaphore.hpp"
#include "ck/utility/amd_named_barrier.hpp"

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch,
          bool AEnableLds,
          bool BEnableLds,
          TensorLoadOption ALoadOption = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BLoadOption = TensorLoadOption::DEFAULT_LOAD>
struct GridwiseGemmPipeline_Wavegroup_v1;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_Wavegroup_v1<1, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
            // preload data into LDS
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    barrierLds.template sync_lds<false>();
                    semaDataReady.template signal<0>();

                    semaDataFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            barrierLds.template sync_lds<false>();
            semaDataReady.template signal<0>();
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
                semaDataFree.template signal<0>();
            }
        }
    }

#ifdef CK_EXTENSION_MX_TYPE
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
              typename CThreadBuffer,
              typename AScaleGridDesc,
              typename AScaleGridBuffer,
              typename AScaleBlockDesc,
              typename AScaleBlockBuffer,
              typename AScaleBlockTransfer,
              typename AScaleBlockTransferStep,
              typename BScaleGridDesc,
              typename BScaleGridBuffer,
              typename BScaleBlockDesc,
              typename BScaleBlockBuffer,
              typename BScaleBlockTransfer,
              typename BScaleBlockTransferStep>
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
                               index_t num_loop,
                               const AScaleGridDesc& a_scale_grid_desc,
                               const AScaleGridBuffer& a_scale_grid_buf,
                               const AScaleBlockDesc& a_scale_block_desc,
                               AScaleBlockBuffer& a_scale_block_buf,
                               AScaleBlockTransfer& a_scale_blockwise_copy,
                               const AScaleBlockTransferStep& a_scale_block_copy_step,
                               const BScaleGridDesc& b_scale_grid_desc,
                               const BScaleGridBuffer& b_scale_grid_buf,
                               const BScaleBlockDesc& b_scale_block_desc,
                               BScaleBlockBuffer& b_scale_block_buf,
                               BScaleBlockTransfer& b_scale_blockwise_copy,
                               const BScaleBlockTransferStep& b_scale_block_copy_step)
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
            // preload data into LDS
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

            a_scale_blockwise_copy.RunRead(a_scale_grid_desc, a_scale_grid_buf);
            b_scale_blockwise_copy.RunRead(b_scale_grid_desc, b_scale_grid_buf);

            a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc, a_scale_block_copy_step);
            b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_block_copy_step);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            a_scale_blockwise_copy.RunWrite(a_scale_block_desc, a_scale_block_buf);
            b_scale_blockwise_copy.RunWrite(b_scale_block_desc, b_scale_block_buf);

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    a_scale_blockwise_copy.RunRead(a_scale_grid_desc, a_scale_grid_buf);

                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                    b_scale_blockwise_copy.RunRead(b_scale_grid_desc, b_scale_grid_buf);

                    barrierLds.template sync_lds<false>();
                    semaDataReady.template signal<0>();

                    semaDataFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                              a_scale_block_copy_step);
                    b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                              b_scale_block_copy_step);

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                    a_scale_blockwise_copy.RunWrite(a_scale_block_desc, a_scale_block_buf);
                    b_scale_blockwise_copy.RunWrite(b_scale_block_desc, b_scale_block_buf);
                    ++i;
                } while(i < (num_loop - 1));
            }

            {
                // tail
                barrierLds.template sync_lds<false>();
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
                    blockwise_gemm.Run(a_block_buf,
                                       b_block_buf,
                                       a_scale_block_buf,
                                       b_scale_block_buf,
                                       c_thread_buf);
                    semaDataFree.template signal<0>();

                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            {
                semaDataReady.template wait<0>();
                blockwise_gemm.Run(
                    a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);
                semaDataFree.template signal<0>();
            }
        }
    }
#endif
};

// 2-stage prefetch
template <>
struct GridwiseGemmPipeline_Wavegroup_v1<2, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return (num_loop / 2) > 1;
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
    static __device__ void Run(const AGridDesc& a_grid_desc,
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
            // preload data into LDS
            {
                // Read 0
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Read 1
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);
            }

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    // Move
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    // Write i
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

                    // Read i+2
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                    barrierLds.template sync_lds<false>();
                    semaDataReady.template signal<0>();

                    // Move
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    // Sync
                    semaDataFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    // Write i+1
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

                    // Read i+3
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                    // Sync
                    barrierLds.template sync_lds<false>();
                    semaDataReady.template signal<0>();

                    i += 2;
                } while(i < (num_loop - 2));
            }

            // tail
            {
                // Write num_loop - 2
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

                // Sync
                barrierLds.template sync_lds<false>();
                semaDataReady.template signal<0>();

                // Sync
                semaDataFree.template wait<0>();
                barrierLds.signal();
                barrierLds.wait();

                // Write num_loop - 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

                // Sync
                barrierLds.template sync_lds<false>();
                semaDataReady.template signal<0>();

                // Gemm num_loop - 1
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

                    // Sync
                    semaDataReady.template wait<0>();

                    // Gemm i
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    // Sync
                    semaDataFree.template signal<0>();
                    semaDataReady.template wait<0>();

                    // Gemm i+1
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    // Sync
                    semaDataFree.template signal<0>();

                    i += 2;
                } while(i < (num_loop - 2));
            }

            // tail
            {
                // Sync
                semaDataReady.template wait<0>();

                // Gemm num_loop - 2
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                semaDataFree.template signal<0>();
                semaDataReady.template wait<0>();

                // Gemm num_loop - 1
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                semaDataFree.template signal<0>();
            }
        }
    }
};

template <bool AEnableLds, bool BEnableLds>
struct GridwiseGemmPipeline_Wavegroup_v1<1, AEnableLds, BEnableLds>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;
        __shared__ NamedBarrier<4> barrierLds;

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            barrierLds.init();
            barrierLds.join();
        }

        semaDataReady.init();
        semaDataFree.init();
        __syncthreads();

        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);

        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            // preload data into LDS
            if constexpr(AEnableLds)
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            }
            else
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
                a_block_buf.SwitchBuffer();
            }

            if constexpr(BEnableLds)
            {
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
            }
            else
            {
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
                b_block_buf.SwitchBuffer();
            }
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;
                do
                {
#if defined(__gfx13__)
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
#endif
                    if constexpr(AEnableLds)
                    {
                        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    }
                    else
                    {
                        a_blockwise_copy.Run(
                            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
                        a_block_buf.SwitchBuffer();
                    }
                    if constexpr(BEnableLds)
                    {
                        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                    }
                    else
                    {
                        b_blockwise_copy.Run(
                            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
                        b_block_buf.SwitchBuffer();
                    }
                    barrierLds.template sync_lds<false>();
                    semaDataReady.template signal<0>();

                    semaDataFree.template wait<0>();
                    barrierLds.signal();
                    barrierLds.wait();
                    if constexpr(AEnableLds)
                    {
                        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                    }

                    if constexpr(BEnableLds)
                    {
                        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                    }

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            {
                barrierLds.template sync_lds<false>();
                semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
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
                semaDataFree.template signal<0>();
            }
        }
    }
};

template <>
struct GridwiseGemmPipeline_Wavegroup_v1<1, false, false>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;

        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);

        semaDataReady.init();
        semaDataFree.init();
        // wait semaphore init
        __syncthreads();

        // preload data
        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            a_blockwise_copy.Run(
                a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
            b_blockwise_copy.Run(
                b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
            semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                a_block_buf.SwitchBuffer();
                b_block_buf.SwitchBuffer();

                do
                {
                    a_blockwise_copy.Run(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
                    b_blockwise_copy.Run(
                        b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
                    semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();

                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    semaDataFree.template wait<0>();
                    ++i;
                } while(i < (num_loop - 1));
            }
        }
        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            c_thread_buf.Clear();

            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    semaDataReady.template wait<0>();
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
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

#ifdef CK_EXTENSION_MX_TYPE
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
              typename CThreadBuffer,
              typename AScaleGridDesc,
              typename AScaleGridBuffer,
              typename AScaleBlockDesc,
              typename AScaleBlockBuffer,
              typename AScaleBlockTransfer,
              typename AScaleBlockTransferStep,
              typename BScaleGridDesc,
              typename BScaleGridBuffer,
              typename BScaleBlockDesc,
              typename BScaleBlockBuffer,
              typename BScaleBlockTransfer,
              typename BScaleBlockTransferStep>
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
                               index_t num_loop,
                               const AScaleGridDesc& a_scale_grid_desc,
                               const AScaleGridBuffer& a_scale_grid_buf,
                               const AScaleBlockDesc& a_scale_block_desc,
                               AScaleBlockBuffer& a_scale_block_buf,
                               AScaleBlockTransfer& a_scale_blockwise_copy,
                               const AScaleBlockTransferStep& a_scale_block_copy_step,
                               const BScaleGridDesc& b_scale_grid_desc,
                               const BScaleGridBuffer& b_scale_grid_buf,
                               const BScaleBlockDesc& b_scale_block_desc,
                               BScaleBlockBuffer& b_scale_block_buf,
                               BScaleBlockTransfer& b_scale_blockwise_copy,
                               const BScaleBlockTransferStep& b_scale_block_copy_step)
    {
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;

        constexpr auto b_block_origin_idx       = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx       = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto b_scale_block_origin_idx = make_tuple(I0, I0, I0, I0, I0);
        constexpr auto a_scale_block_origin_idx = make_tuple(I0, I0, I0, I0, I0);

        semaDataReady.init();
        semaDataFree.init();
        // wait semaphore init
        __syncthreads();

        // preload data
        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            // preload data into LDS
            a_blockwise_copy.Run(
                a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
            b_blockwise_copy.Run(
                b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
            a_scale_blockwise_copy.Run(a_scale_grid_desc,
                                       a_scale_grid_buf,
                                       a_scale_block_desc,
                                       a_scale_block_origin_idx,
                                       a_scale_block_buf);
            b_scale_blockwise_copy.Run(b_scale_grid_desc,
                                       b_scale_grid_buf,
                                       b_scale_block_desc,
                                       b_scale_block_origin_idx,
                                       b_scale_block_buf);
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
            a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc, a_scale_block_copy_step);
            b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_block_copy_step);
            semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();

            // main body
            if constexpr(HasMainLoop)
            {

                a_block_buf.SwitchBuffer();
                b_block_buf.SwitchBuffer();
                a_scale_block_buf.SwitchBuffer();
                b_scale_block_buf.SwitchBuffer();
                index_t i = 0;

                do
                {
                    a_blockwise_copy.Run(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
                    b_blockwise_copy.Run(
                        b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
                    a_scale_blockwise_copy.Run(a_scale_grid_desc,
                                               a_scale_grid_buf,
                                               a_scale_block_desc,
                                               a_scale_block_origin_idx,
                                               a_scale_block_buf);
                    b_scale_blockwise_copy.Run(b_scale_grid_desc,
                                               b_scale_grid_buf,
                                               b_scale_block_desc,
                                               b_scale_block_origin_idx,
                                               b_scale_block_buf);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                              a_scale_block_copy_step);
                    b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                              b_scale_block_copy_step);
                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
                    a_scale_block_buf.SwitchBuffer();
                    b_scale_block_buf.SwitchBuffer();
                    semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
                    semaDataFree.template wait<0>();
                    ++i;
                } while(i < (num_loop - 1));
            }
        }

        // preload data
        if(get_wave_id_in_wavegroup() == WaveIdRun)
        { // Initialize C
            c_thread_buf.Clear();

            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    semaDataReady.template wait<0>();

                    blockwise_gemm.Run(a_block_buf,
                                       b_block_buf,
                                       a_scale_block_buf,
                                       b_scale_block_buf,
                                       c_thread_buf);

                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
                    a_scale_block_buf.SwitchBuffer();
                    b_scale_block_buf.SwitchBuffer();
                    semaDataFree.template signal<0>();
                    ++i;
                } while(i < (num_loop - 1));
            }

            // tail
            {
                semaDataReady.template wait<0>();
                blockwise_gemm.Run(
                    a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);
            }
        }
    }
#endif
};

template <TensorLoadOption ALoadOption, TensorLoadOption BLoadOption>
struct GridwiseGemmPipeline_Wavegroup_v1<1, false, false, ALoadOption, BLoadOption>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
        __shared__ WavegroupSemaphore<WaveIdRun> semaDataReady;
        __shared__ WavegroupSemaphore<WaveIdLoad> semaDataFree;

        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);

        semaDataReady.init();
        semaDataFree.init();
        // wait semaphore init
        __syncthreads();

        // preload data
        if(get_wave_id_in_wavegroup() == WaveIdLoad)
        {
            // Need add this if condition when the LLVM fix the scratch_store only in wave_roup_id
            // == 0 issue.
            if((ALoadOption == TensorLoadOption::DEFAULT_LOAD) ||
               (ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD && (get_wavegroup_id() == 0)))
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
            }
            if((BLoadOption == TensorLoadOption::DEFAULT_LOAD) ||
               (BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD && (get_wavegroup_id() == 0)))
            {
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
            }
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
            semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();
            // main body
            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                a_block_buf.SwitchBuffer();
                b_block_buf.SwitchBuffer();

                do
                {
                    if((ALoadOption == TensorLoadOption::DEFAULT_LOAD) ||
                       (ALoadOption == TensorLoadOption::WGP_MULTICAST_LOAD &&
                        (get_wavegroup_id() == 0)))
                    {
                        a_blockwise_copy.Run(
                            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
                    }
                    if((BLoadOption == TensorLoadOption::DEFAULT_LOAD) ||
                       (BLoadOption == TensorLoadOption::WGP_MULTICAST_LOAD &&
                        (get_wavegroup_id() == 0)))
                    {
                        b_blockwise_copy.Run(
                            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);
                    }
                    semaDataReady.template signal<SemaphoreAddressSpaceGlobal>();

                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    semaDataFree.template wait<0>();
                    ++i;
                } while(i < (num_loop - 1));
            }
        }
        if(get_wave_id_in_wavegroup() == WaveIdRun)
        {
            // Initialize C
            c_thread_buf.Clear();

            if constexpr(HasMainLoop)
            {
                index_t i = 0;

                do
                {
                    semaDataReady.template wait<0>();
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    a_block_buf.SwitchBuffer();
                    b_block_buf.SwitchBuffer();
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
