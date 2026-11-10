// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>

#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_cr_kernel.hpp"
#include "fmha_fwd_cr_support.hpp"
#include "ck_tile/host/device_prop.hpp"

template <typename Kernel, typename Arch>
struct FmhaFwdLaunch;

template <typename Problem, typename Epilogue, typename Arch>
struct FmhaFwdLaunch<
    ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSTdmCr<Problem>, Epilogue>,
    Arch>
{
    using Kernel = ck_tile::FmhaFwdKernel<ck_tile::BlockFmhaPipelineQRKSVSTdmCr<Problem>, Epilogue>;

    template <typename Args>
    static float Run(const ck_tile::stream_config& s, const Args& a)
    {
        using Pipeline = typename Kernel::FmhaPipeline;
        if(ck_tile::get_device_name().compare(0, 7, "gfx1250") != 0 || !Kernel::IsSupportedInput(a))
            return -1;

        auto launch_strided = [&] {
            using K      = typename Kernel::StridedKernel;
            const auto x = K::MakeKargs(a);
            assert(K::IsSupportedArgument(x));
            return ck_tile::launch_kernel(s,
                                          ck_tile::make_kernel<K::kBlockPerCu, Arch>(
                                              K{}, K::GridSize(x), K::BlockSize(), 0, x));
        };
        if constexpr(!Pipeline::kCompact)
            return launch_strided();
        else
        {
            if(!fmha_fwd_cr_compact_args(a))
                return launch_strided();
            const auto kargs = Kernel::MakeKargs(a);
            auto launch      = [&](auto tuning) {
                using K = typename Kernel::template WithTuning<decltype(tuning)>;
                assert(K::IsSupportedArgument(kargs));
                return ck_tile::launch_kernel(
                    s,
                    ck_tile::make_kernel<K::kBlockPerCu, Arch>(
                        K{}, K::GridSize(kargs), K::BlockSize(), 0, kargs));
            };
            if constexpr(Pipeline::kM0 == 64)
                return launch(ck_tile::FmhaTdmCrTuning<0, true, true>{});
            else
            {
                if constexpr(Pipeline::kM0 == 128 && !Problem::kPadSeqLenQ && !Problem::kPadSeqLenK)
                {
                    if constexpr(Pipeline::kCausal)
                    {
                        // A single query tile has no traversal to group across heads.
                        if(a.seqlen_q == 128 && a.seqlen_k == 128)
                            return launch(ck_tile::FmhaTdmCrTuning<0, true, false, 0>{});
                        // Four causal query tiles benefit from larger head groups once
                        // every XCD owns at least two complete groups of 32 rows.
                        if(a.nhead_q == 8 && a.seqlen_q == 512 && a.seqlen_k == 512 &&
                           a.batch >= 64 && a.batch % 32 == 0)
                            return launch(ck_tile::FmhaTdmCrTileTuning<4>{});
                    }
                    if(fmha_fwd_cr_has_static_tuning(a))
                    {
                        switch(a.seqlen_q)
                        {
                        case 512: return launch(ck_tile::FmhaTdmCrTuning<512>{});
                        case 1024: return launch(ck_tile::FmhaTdmCrTuning<1024>{});
                        case 2048: return launch(ck_tile::FmhaTdmCrTuning<2048>{});
                        case 4096: return launch(ck_tile::FmhaTdmCrTuning<4096>{});
                        case 8192: return launch(ck_tile::FmhaTdmCrTuning<8192>{});
                        case 16384: return launch(ck_tile::FmhaTdmCrTuning<16384>{});
                        case 32768: return launch(ck_tile::FmhaTdmCrTuning<32768>{});
                        }
                    }
                    if constexpr(!Pipeline::kCausal)
                    {
                        if(a.nhead_q == 8)
                        {
                            if(a.seqlen_q == a.seqlen_k)
                            {
                                switch(a.seqlen_q)
                                {
                                case 128: return launch(ck_tile::FmhaTdmCrTileTuning<1>{});
                                case 256: return launch(ck_tile::FmhaTdmCrTileTuning<2>{});
                                case 512: return launch(ck_tile::FmhaTdmCrTileTuning<4>{});
                                case 640: return launch(ck_tile::FmhaTdmCrTileTuning<5>{});
                                case 1024: return launch(ck_tile::FmhaTdmCrTileTuning<8>{});
                                case 4096:
                                    if(a.batch <= 4)
                                        return launch(ck_tile::FmhaTdmCrTileTuning<32>{});
                                    break;
                                }
                            }
                            if(a.seqlen_q == 1024 && a.seqlen_k == 2048)
                                return launch(ck_tile::FmhaTdmCrTileTuning<8, 16>{});
                            if(a.seqlen_q == 2048 && a.seqlen_k == 1024)
                                return launch(ck_tile::FmhaTdmCrTileTuning<16, 8>{});
                        }
                    }
                }
                if constexpr(Pipeline::kM0 == 128 && !Pipeline::kCausal && Problem::kPadSeqLenQ &&
                             Problem::kPadSeqLenK)
                {
                    if(a.nhead_q == 8 && a.seqlen_q > 512 && a.seqlen_q < 640 && a.seqlen_k > 512 &&
                       a.seqlen_k < 640)
                    {
                        // Nearly full tails have a fixed K tile count and benefit from
                        // fragment masks and the 15-row asynchronous store.
                        if(a.seqlen_q % 16 == 15 && a.seqlen_k % 128 > 112)
                            return launch(ck_tile::FmhaTdmCrTileTuning<5, 0, true>{});
                        return launch(ck_tile::FmhaTdmCrTileTuning<5, 0>{});
                    }
                }
                if(a.seqlen_q <= 512)
                    return launch(ck_tile::FmhaTdmCrTuning<0, true, false>{});
                if(a.seqlen_q <= 2048)
                    return launch(ck_tile::FmhaTdmCrTuning<0, true, true>{});
                return launch(ck_tile::FmhaTdmCrTuning<>{});
            }
        }
    }
};
