// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch,
          bool AEnableLds,
          bool BEnableLds,
          TensorLoadOption AMulctiCastLoad = TensorLoadOption::DEFAULT_LOAD,
          TensorLoadOption BMulticastLoad  = TensorLoadOption::DEFAULT_LOAD>
struct GridwiseGemmPipeline_v1;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<1, true, true>
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
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
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
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_scale_blockwise_copy.RunRead(a_scale_grid_desc, a_scale_grid_buf);
        b_scale_blockwise_copy.RunRead(b_scale_grid_desc, b_scale_grid_buf);

        a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc, a_scale_block_copy_step);
        b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_block_copy_step);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

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
                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                b_scale_blockwise_copy.RunRead(b_scale_grid_desc, b_scale_grid_buf);

                blockwise_gemm.Run(
                    a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);
                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                          a_scale_block_copy_step);
                b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                          b_scale_block_copy_step);

                a_scale_blockwise_copy.RunWrite(a_scale_block_desc, a_scale_block_buf);
                b_scale_blockwise_copy.RunWrite(b_scale_block_desc, b_scale_block_buf);
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            blockwise_gemm.Run(
                a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);
        }
    }
#endif
};

// 2-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<2, true, true>
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

        // Initialize C
        c_thread_buf.Clear();

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

                // Sync
                block_sync_lds();

                // Gemm i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

                // Read i+3
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                // Sync
                block_sync_lds();

                // Gemm i+1
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            // Write num_loop - 2
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            // Sync
            block_sync_lds();

            // Write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, false, true>
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
#if defined(__gfx13__)
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto a_block_buf_switch = a_block_buf;

        // preload data into LDS
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                a_block_buf = a_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, true, false>
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
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto b_block_buf_switch = b_block_buf;

        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, false, false>
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
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto b_block_buf_switch = b_block_buf;
        auto a_block_buf_switch = a_block_buf;

        // preload data into LDS
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

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
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_block_buf = a_block_buf_switch;
                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
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
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto b_block_buf_switch = b_block_buf;
        auto a_block_buf_switch = a_block_buf;

        constexpr auto b_scale_block_origin_idx = make_tuple(I0, I0, I0, I0, I0);
        constexpr auto a_scale_block_origin_idx = make_tuple(I0, I0, I0, I0, I0);
        auto b_scale_block_buf_switch           = b_scale_block_buf;
        auto a_scale_block_buf_switch           = a_scale_block_buf;

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
        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);
                a_scale_blockwise_copy.Run(a_scale_grid_desc,
                                           a_scale_grid_buf,
                                           a_scale_block_desc,
                                           a_scale_block_origin_idx,
                                           a_scale_block_buf_switch);
                b_scale_blockwise_copy.Run(b_scale_grid_desc,
                                           b_scale_grid_buf,
                                           b_scale_block_desc,
                                           b_scale_block_origin_idx,
                                           b_scale_block_buf_switch);
                block_sync_lds();

                blockwise_gemm.Run(
                    a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                a_scale_blockwise_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                          a_scale_block_copy_step);
                b_scale_blockwise_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                          b_scale_block_copy_step);
                a_block_buf = a_block_buf_switch;
                b_block_buf = b_block_buf_switch;

                a_scale_block_buf = a_scale_block_buf_switch;
                b_scale_block_buf = b_scale_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(
                a_block_buf, b_block_buf, a_scale_block_buf, b_scale_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
#endif
};

template <TensorLoadOption ALoadOption, TensorLoadOption BLoadOption>
struct GridwiseGemmPipeline_v1<1, false, false, ALoadOption, BLoadOption>
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
                               index_t num_loop,
                               index_t a_cluster_size,
                               index_t b_cluster_size)
    {
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto b_block_buf_switch = b_block_buf;
        auto a_block_buf_switch = a_block_buf;
        ignore                  = a_cluster_size;
        ignore                  = b_cluster_size;

#if defined(__gfx13__)
        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                     BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD) // ClustMultiCastLoad
        {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
            __builtin_amdgcn_s_barrier_signal(-3);
            __builtin_amdgcn_s_barrier_wait(-3);
        }
#endif
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

#if defined(__gfx13__)
        if constexpr(ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                     BLoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD) // ClustMultiCastLoad
        {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
            __builtin_amdgcn_s_barrier_signal(-3);
            __builtin_amdgcn_s_barrier_wait(-3);
        }
#endif

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
#if defined(__gfx13__)
                if constexpr(ALoadOption == TensorLoadOption::CLUSTER_MULTICAST_LOAD ||
                             BLoadOption ==
                                 TensorLoadOption::CLUSTER_MULTICAST_LOAD) // ClustMultiCastLoad
                {
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
                    __builtin_amdgcn_s_barrier_signal(-3);
                    __builtin_amdgcn_s_barrier_wait(-3);
                }
#endif
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_block_buf = a_block_buf_switch;
                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <TensorLoadOption ALoadOption, TensorLoadOption BLoadOption>
struct GridwiseGemmPipeline_v1<1, true, false, ALoadOption, BLoadOption>
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
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        auto b_block_buf_switch = b_block_buf;

        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

// template <bool AEnableLds, bool BEnableLds, TensorLoadOption ATmaLoad, TensorLoadOption
// BTmaLoad>
template <>
struct GridwiseGemmPipeline_v1<1,
                               true,
                               false,
                               TensorLoadOption::CLUSTER_DDS_LOAD,
                               TensorLoadOption::DEFAULT_LOAD>
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
                               index_t num_loop,
                               index_t a_cluster_size,
                               index_t b_cluster_size)
    {
#if defined(__gfx13__)
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        // Initialize C
        c_thread_buf.Clear();

        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

#if defined(__gfx13__)
        // Cluster sync
        bool isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
        if(isFirst)
            __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);

        const int wgRank = __builtin_amdgcn_cluster_workgroup_flat_id();
#else
        const int wgRank = 0;
#endif

        ignore    = b_cluster_size;
        index_t k = 0;
        do
        {
            b_blockwise_copy.Run(
                b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

            block_sync_lds();

            const index_t a_map_rank_id = (wgRank & ~(a_cluster_size - 1)) | k;

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, a_map_rank_id, 0);

            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            ++k;
        } while(k < a_cluster_size);

        block_sync_lds();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                index_t j = 0;

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

#if defined(__gfx13__)
                // Cluster sync
                isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
                __builtin_amdgcn_s_barrier_wait(-1);
                if(isFirst)
                    __builtin_amdgcn_s_barrier_signal(-3);
                __builtin_amdgcn_s_barrier_wait(-3);
#endif

                do
                {
                    b_blockwise_copy.Run(
                        b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

                    block_sync_lds();

                    const index_t a_map_rank_id = (wgRank & ~(a_cluster_size - 1)) | j;

                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, a_map_rank_id, 0);

                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    ++j;
                } while(j < a_cluster_size);

                block_sync_lds();

                ++i;
            } while(i < (num_loop / a_cluster_size - 1));
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1,
                               false,
                               true,
                               TensorLoadOption::DEFAULT_LOAD,
                               TensorLoadOption::CLUSTER_DDS_LOAD>
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
                               index_t num_loop,
                               index_t a_cluster_size,
                               index_t b_cluster_size)
    {
#if defined(__gfx13__)
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        // Initialize C
        c_thread_buf.Clear();

        // preload data into LDS
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

#if defined(__gfx13__)
        // Cluster sync
        bool isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
        if(isFirst)
            __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);

        const int wgRank = __builtin_amdgcn_cluster_workgroup_flat_id();
#else
        const int wgRank = 0;
#endif
        ignore    = a_cluster_size;
        index_t k = 0;
        do
        {
            a_blockwise_copy.Run(
                a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

            block_sync_lds();

            const index_t b_map_rank_id = (wgRank & ~(b_cluster_size - 1)) | k;

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, 0, b_map_rank_id);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

            ++k;
        } while(k < b_cluster_size);

        block_sync_lds();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                index_t j = 0;

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
#if defined(__gfx13__)
                // Cluster sync
                isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
                __builtin_amdgcn_s_barrier_wait(-1);
                if(isFirst)
                    __builtin_amdgcn_s_barrier_signal(-3);
                __builtin_amdgcn_s_barrier_wait(-3);
#endif
                do
                {
                    a_blockwise_copy.Run(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

                    block_sync_lds();

                    const index_t b_map_rank_id = (wgRank & ~(b_cluster_size - 1)) | j;

                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, 0, b_map_rank_id);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    ++j;
                } while(j < b_cluster_size);

                block_sync_lds();

                ++i;
            } while(i < (num_loop / b_cluster_size - 1));
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1,
                               false,
                               true,
                               TensorLoadOption::CLUSTER_MULTICAST_LOAD,
                               TensorLoadOption::CLUSTER_DDS_LOAD>
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
                               index_t num_loop,
                               index_t a_cluster_size,
                               index_t b_cluster_size)
    {
#if defined(__gfx13__)
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0, I0);
#else
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
#endif
        // Initialize C
        c_thread_buf.Clear();

        // preload data into LDS
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

#if defined(__gfx13__)
        // Cluster sync
        bool isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
        if(isFirst)
            __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);

        const int wgRank = __builtin_amdgcn_cluster_workgroup_flat_id();
#else
        const int wgRank = 0;
#endif
        ignore = a_cluster_size;

        index_t k = 0;
        do
        {
            a_blockwise_copy.Run(
                a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

            block_sync_lds();

#if defined(__gfx13__)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
            __builtin_amdgcn_s_barrier_signal(-3);
            __builtin_amdgcn_s_barrier_wait(-3);
#endif

            const index_t b_map_rank_id = (wgRank & ~(b_cluster_size - 1)) | k;

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, 0, b_map_rank_id);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

            ++k;
        } while(k < b_cluster_size);

        block_sync_lds();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                index_t j = 0;

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

#if defined(__gfx13__)
                // Cluster sync
                isFirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);
                __builtin_amdgcn_s_barrier_wait(-1);
                if(isFirst)
                    __builtin_amdgcn_s_barrier_signal(-3);
                __builtin_amdgcn_s_barrier_wait(-3);

                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
                __builtin_amdgcn_s_barrier_signal(-3);
                __builtin_amdgcn_s_barrier_wait(-3);
#endif

                do
                {
                    a_blockwise_copy.Run(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

                    block_sync_lds();

                    const index_t b_map_rank_id = (wgRank & ~(b_cluster_size - 1)) | j;

                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf, 0, b_map_rank_id);

#if defined(__gfx13__)
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
                    __builtin_amdgcn_s_barrier_signal(-3);
                    __builtin_amdgcn_s_barrier_wait(-3);
#endif

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    ++j;
                } while(j < b_cluster_size);

                block_sync_lds();

                ++i;
            } while(i < (num_loop / b_cluster_size - 1));
        }
    }
};

template <index_t NumPrefetch, bool AEnableLds, bool BEnableLds>
struct GridwiseGemmPipeline_v1_WeightOnly;

template <>
struct GridwiseGemmPipeline_v1_WeightOnly<1, true, true>
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
              typename ScaleGridDesc,
              typename ScaleGridBuffer,
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
                               const ScaleGridDesc& scale_grid_desc,
                               const ScaleGridBuffer& scale_grid_buf,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // Global Prefetch Stage 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        // Scale read once
        b_blockwise_copy.RunScaleRead(scale_grid_desc, scale_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        // Dequantization fused in blockwise_copy
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <index_t NumPrefetch>
struct GridwiseGemmPipelineInterwave_v1;

template <>
struct GridwiseGemmPipelineInterwave_v1<1>
{
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
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // block_sync_lds(); // moved into blockwise_gemm

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// Note: 2 stage prefetch not optimized for inter-wave loop scheduler
template <>
struct GridwiseGemmPipelineInterwave_v1<2> : public GridwiseGemmPipeline_v1<2, true, true>
{
};

// TODO: deprecate as GridwiseGemmPipeline_Selector covers the functionality
template <index_t NumPrefetch, LoopScheduler LoopSched>
constexpr auto GridwiseGemmPipeline_v1_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return GridwiseGemmPipeline_v1<NumPrefetch, true, true>{};
    }
    else if constexpr(LoopSched == LoopScheduler::Interwave)
    {
        return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
    }
}

} // namespace ck
