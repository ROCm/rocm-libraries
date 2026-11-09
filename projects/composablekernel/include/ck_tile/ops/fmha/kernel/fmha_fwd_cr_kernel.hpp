// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_cr.hpp"

namespace ck_tile {

// Native BSHD wrapper: the grid mapping and output shuffle belong to this
// pipeline, which has a compact, contiguous-layout argument contract.
template <typename Pipeline_, typename Epilogue_, typename Tuning_ = FmhaTdmCrTuning<>>
struct FmhaFwdTdmCrKernel
{
    using FmhaPipeline                   = Pipeline_;
    using Tuning                         = Tuning_;
    using Kargs                          = typename FmhaPipeline::Arguments;
    static constexpr index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    template <typename OtherTuning>
    using WithTuning = FmhaFwdTdmCrKernel<Pipeline_, Epilogue_, OtherTuning>;

    CK_TILE_HOST static dim3 BlockSize() { return dim3(128); }
    CK_TILE_HOST static dim3 GridSize(const Kargs& a)
    {
        return dim3((a.sq + 127) / 128, a.batch * a.heads);
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return FmhaPipeline::GetSmemSize();
    }
    CK_TILE_DEVICE void operator()(Kargs a) const { FmhaPipeline::template Run<Tuning>(a); }
};

} // namespace ck_tile
