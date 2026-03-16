// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "depthwise_conv_utils.hpp"

struct DepthwiseConvFwdInvoker
{
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              ck_tile::index_t TileH,
              ck_tile::index_t TileW,
              ck_tile::index_t FilterSize,
              ck_tile::index_t DilationH,
              ck_tile::index_t DilationW,
              ck_tile::index_t StrideH,
              ck_tile::index_t StrideW,
              ck_tile::index_t PadH,
              ck_tile::index_t PadW,
              ck_tile::index_t NBatch,
              ck_tile::index_t SubTileH,
              ck_tile::index_t SubTileW,
              ck_tile::index_t InVecSize,
              ck_tile::index_t OutVecSize>
    static KernelRunResult try_instance(const ck_tile::DepthwiseConvFwdHostArgs& args,
                                        const ck_tile::stream_config& s,
                                        const VerificationInfo<OutDataType>& verify_info,
                                        const ck_tile::index_t instance_idx,
                                        const std::size_t flop,
                                        const std::size_t num_byte,
                                        bool& first_error_printed)
    {
        KernelRunResult result;

        // TODO: BlockSize is hardcoded to 64; make it a template parameter when supporting other
        // sizes
        using Traits = ck_tile::DepthwiseConvFwdTraits<InDataType,
                                                       WeiDataType,
                                                       AccDataType,
                                                       OutDataType,
                                                       64,
                                                       TileH,
                                                       TileW,
                                                       FilterSize,
                                                       FilterSize,
                                                       StrideH,
                                                       StrideW,
                                                       DilationH,
                                                       DilationW,
                                                       PadH,
                                                       PadW,
                                                       NBatch,
                                                       SubTileH,
                                                       SubTileW,
                                                       InVecSize,
                                                       OutVecSize>;

        using Pipeline = ck_tile::DepthwiseConvFwdPipeline<Traits>;
        using Kernel   = ck_tile::DepthwiseConvFwdKernel<Traits, Pipeline>;

        if(!Kernel::IsSupportedArgument(args))
        {
            return result;
        }

        result.config_name = Kernel::GetName();

        const auto kargs = Kernel::MakeKernelArgs(args);

        const auto grids  = Kernel::GridSize(static_cast<ck_tile::index_t>(args.G_),
                                            static_cast<ck_tile::index_t>(args.N_));
        const auto blocks = Kernel::BlockSize();

        // TODO: make kBlockPerCu configurable instead of hardcoding 1
        const float time_ms =
            ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

        result.is_supported = true;
        result.time_ms      = time_ms;

        if(flop > 0 && time_ms > 0)
        {
            result.tflops     = static_cast<float>(flop) / 1.E9 / time_ms;
            result.gb_per_sec = static_cast<float>(num_byte) / 1.E6 / time_ms;
        }

        // TODO: remove after depthwise conv generalization/optimization is complete
#if 0
        dump_output_tensor(verify_info, args, s, time_ms);
#endif

        if(verify_info.do_verification)
        {
            verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());
            const bool print_errors = (s.log_level_ > 0) && !first_error_printed;

            const bool pass      = verify_gpu_result<OutDataType>(verify_info.p_out_host->data(),
                                                             verify_info.p_out_ref,
                                                             verify_info.output_size,
                                                             print_errors);
            result.verify_status = pass ? VerifyStatus::kPass : VerifyStatus::kFail;

            if(!pass && print_errors)
            {
                first_error_printed = true;
            }
        }

        if(s.log_level_ > 0)
        {
            std::cout << "[Instance " << instance_idx << "] ";
            if(result.verify_status != VerifyStatus::kSkipped)
            {
                std::cout << (result.verify_status == VerifyStatus::kPass ? "[PASS]" : "[FAIL]")
                          << " ";
            }
            std::cout << std::fixed << std::setprecision(6) << time_ms << " ms, "
                      << std::setprecision(4) << result.tflops << " TFlops, "
                      << std::setprecision(3) << result.gb_per_sec << " GB/s, "
                      << result.config_name << std::endl;
        }

        return result;
    }

    template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
    static InvokerResult run_all_instances(const ck_tile::DepthwiseConvFwdHostArgs& args,
                                           const ck_tile::stream_config& s,
                                           const VerificationInfo<OutDataType>& verify_info,
                                           const std::size_t flop,
                                           const std::size_t num_byte)
    {
        float best_time = std::numeric_limits<float>::max();
        std::string best_config;
        VerifyStatus best_verify_status    = VerifyStatus::kSkipped;
        ck_tile::index_t best_instance_idx = -1;
        ck_tile::index_t instance_count    = 0;
        ck_tile::index_t valid_count       = 0;
        bool first_error_printed           = false;

        if(s.log_level_ > 0)
        {
            std::cout << "\n=== Testing all instances ===" << std::endl;
        }

        auto process_result = [&](const KernelRunResult& result) {
            if(result.is_supported)
            {
                valid_count++;
                if(result.time_ms < best_time)
                {
                    best_time          = result.time_ms;
                    best_config        = result.config_name;
                    best_verify_status = result.verify_status;
                    best_instance_idx  = instance_count;
                }
            }
            instance_count++;
        };

        // Parameters: TileH, TileW, Filter (square, FilterH=FilterW), StrH, StrW, PadH, PadW,
        //             NBatch, SubTileH, SubTileW, InVecSize, OutVecSize
        // TODO: Dilation is hardcoded to 1x1; expand when non-unit dilation is supported
#define TRY_INSTANCE(                                                                \
    TileH, TileW, Filter, StrH, StrW, PadH, PadW, NBatch, SubH, SubW, InVec, OutVec) \
    process_result(try_instance<InDataType,                                          \
                                WeiDataType,                                         \
                                AccDataType,                                         \
                                OutDataType,                                         \
                                TileH,                                               \
                                TileW,                                               \
                                Filter,                                              \
                                1,                                                   \
                                1,                                                   \
                                StrH,                                                \
                                StrW,                                                \
                                PadH,                                                \
                                PadW,                                                \
                                NBatch,                                              \
                                SubH,                                                \
                                SubW,                                                \
                                InVec,                                               \
                                OutVec>(                                             \
        args, s, verify_info, instance_count, flop, num_byte, first_error_printed))

        // ============================================================================
        // FilterSize = 3, Pad = 1
        // ============================================================================
        // --- 3x3 stride=1 ---
        TRY_INSTANCE(8, 8, 3, 1, 1, 1, 1, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 3, 1, 1, 1, 1, 8, 1, 4, 8, 8); // mid tile, large batch
        TRY_INSTANCE(16, 16, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2); // tiny image fallback (H/W<=4)
        TRY_INSTANCE(28, 28, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8); // large tile, NBatch=1

        // --- 3x3 stride=2 ---
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 2, 1, 4, 8, 8); // mid tile, NBatch=2
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 1, 4, 8, 8); // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 8, 8); // small output (Ho/Wo~7-14)
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 2, 2); // tiny image fallback (H/W<=4)
        TRY_INSTANCE(14, 28, 3, 2, 2, 1, 1, 1, 2, 4, 8, 8); // asymmetric tile
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 2, 4, 4, 8, 8); // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 4, 4); // large tile, reduced vec
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 2, 8, 8, 8); // large output (Ho/Wo~256-512)

        // ============================================================================
        // FilterSize = 5, Pad = 2
        // ============================================================================
        // --- 5x5 stride=1 ---
        TRY_INSTANCE(8, 8, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE(8, 8, 5, 1, 1, 2, 2, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 2, 1, 1, 4, 8, 8); // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 2, 8, 1, 4, 8, 8); // mid tile, large batch
        TRY_INSTANCE(28, 28, 5, 1, 1, 2, 2, 8, 4, 4, 8, 8); // large tile, large batch
        TRY_INSTANCE(32, 32, 5, 1, 1, 2, 2, 4, 4, 4, 8, 8); // large tile, mid batch

        // --- 5x5 stride=2 ---
        TRY_INSTANCE(8, 8, 5, 2, 2, 2, 2, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE(8, 8, 5, 2, 2, 2, 2, 1, 2, 2, 2, 2);   // small tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 2, 2, 2, 2, 1, 1, 4, 8, 8); // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 2, 2, 2, 2, 1, 2, 2, 8, 8); // small output (Ho/Wo~7-14)
        TRY_INSTANCE(14, 28, 5, 2, 2, 2, 2, 2, 2, 4, 8, 8); // asymmetric tile
        TRY_INSTANCE(16, 32, 5, 2, 2, 2, 2, 4, 1, 8, 8, 8); // wide tile
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 4, 4, 4, 4); // large tile, reduced vec
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 2, 8, 8, 8); // large output (Ho/Wo~256-512)

        // ============================================================================
        // FilterSize = 7, Pad = 3
        // ============================================================================
        // --- 7x7 stride=1 ---
        TRY_INSTANCE(8, 8, 7, 1, 1, 3, 3, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE(8, 8, 7, 1, 1, 3, 3, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 3, 1, 1, 4, 8, 8); // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 3, 8, 1, 4, 8, 8); // mid tile, large batch
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 3, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 3, 8, 4, 4, 8, 8); // large tile, large batch
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 3, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 3, 4, 4, 4, 8, 8); // large tile, mid batch

        // --- 7x7 stride=2 ---
        TRY_INSTANCE(8, 8, 7, 2, 2, 3, 3, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE(16, 16, 7, 2, 2, 3, 3, 2, 1, 4, 8, 8); // mid tile, NBatch=2
        TRY_INSTANCE(14, 28, 7, 2, 2, 3, 3, 2, 2, 4, 8, 8); // asymmetric tile
        TRY_INSTANCE(16, 32, 7, 2, 2, 3, 3, 4, 1, 8, 8, 8); // wide tile
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 3, 2, 4, 4, 8, 8); // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 3, 1, 4, 4, 8, 8); // large tile, NBatch=1

        // ============================================================================
        // FilterSize = 9, Pad = 4
        // ============================================================================
        // --- 9x9 stride=1 ---
        TRY_INSTANCE(8, 8, 9, 1, 1, 4, 4, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE(8, 8, 9, 1, 1, 4, 4, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 4, 1, 1, 4, 8, 8); // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 4, 8, 1, 4, 8, 8); // mid tile, large batch
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 4, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 4, 8, 4, 4, 8, 8); // large tile, large batch
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 1, 4, 4, 8, 8); // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 4, 4, 4, 8, 8); // large tile, mid batch

        // --- 9x9 stride=2 ---
        TRY_INSTANCE(8, 8, 9, 2, 2, 4, 4, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE(16, 16, 9, 2, 2, 4, 4, 2, 1, 4, 8, 8); // mid tile, NBatch=2
        TRY_INSTANCE(14, 28, 9, 2, 2, 4, 4, 2, 2, 4, 8, 8); // asymmetric tile
        TRY_INSTANCE(16, 32, 9, 2, 2, 4, 4, 4, 1, 8, 8, 8); // wide tile
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 4, 2, 4, 4, 8, 8); // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 4, 1, 4, 4, 8, 8); // large tile, NBatch=1

#undef TRY_INSTANCE

        if(valid_count == 0)
        {
            // TODO: standardize error message format
            throw std::runtime_error(
                "Kernel launch failed: No suitable kernel configuration found!");
        }

        return {best_time, best_config, best_verify_status, best_instance_idx};
    }
};
