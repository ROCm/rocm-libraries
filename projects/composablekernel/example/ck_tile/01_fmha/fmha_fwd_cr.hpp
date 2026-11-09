// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_cr_kernel.hpp"
#include "fmha_fwd_cr_support.hpp"
#include "ck_tile/host/device_prop.hpp"

template <typename Kernel, typename Arch, typename Args>
float fmha_fwd_cr_launch(const ck_tile::stream_config& s, const Args& a)
{
    using Pipeline = typename Kernel::FmhaPipeline;
    using Problem  = typename Pipeline::Problem;
    if(ck_tile::get_device_name().compare(0, 7, "gfx1250") != 0 || !fmha_fwd_cr_supported_args(a) ||
       (Pipeline::kCausal != (a.mask_type != 0)) || (Pipeline::kStoreLSE && !a.lse_ptr) ||
       (!Problem::kPadSeqLenQ && a.seqlen_q % 128) || (!Problem::kPadSeqLenK && a.seqlen_k % 128))
        return -1;

    const typename Kernel::Kargs kargs{a.q_ptr,
                                       a.k_ptr,
                                       a.v_ptr,
                                       a.o_ptr,
                                       static_cast<float*>(a.lse_ptr),
                                       a.batch,
                                       a.nhead_q,
                                       a.seqlen_q,
                                       a.seqlen_k,
                                       a.scale_s};
    auto launch = [&](auto tuning) {
        using K = typename Kernel::template WithTuning<decltype(tuning)>;
        return ck_tile::launch_kernel(s,
                                      ck_tile::make_kernel<K::kBlockPerCu, Arch>(
                                          K{}, K::GridSize(kargs), K::BlockSize(), 0, kargs));
    };
    if constexpr(!Problem::kPadSeqLenQ && !Problem::kPadSeqLenK)
    {
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
    }
    if(a.seqlen_q <= 512)
        return launch(ck_tile::FmhaTdmCrTuning<0, true, false>{});
    if(a.seqlen_q <= 2048)
        return launch(ck_tile::FmhaTdmCrTuning<0, true, true>{});
    return launch(ck_tile::FmhaTdmCrTuning<>{});
}
