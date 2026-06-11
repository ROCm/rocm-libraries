// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Hand-written template instantiation for SpargeBlockMapKernel (fp16, D=128).

#include "sparge_blockmap_trek.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "ck_tile/host/device_memory.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

// ============================================================================
// Type configuration for block map kernel (reuses FmhaSparseFwdTypeConfig)
// ============================================================================

// fp16: D=128, kM0=64, kN0=128
using bmap_fp16_block_tile = ck_tile::sequence<64, 128, 128, 128, 128, 128>;
//                                              kM0 kN0  kK0  kN1  kK1  kQKHeaddim(D)

using bmap_fp16_shape =
    ck_tile::TileFmhaShape<bmap_fp16_block_tile,
                           ck_tile::sequence<4, 1, 1>,    // Gemm0BlockWarps
                           ck_tile::sequence<16, 16, 16>, // Gemm0WarpTile (unused by blockmap, but
                                                          // needed by shape)
                           ck_tile::sequence<4, 1, 1>,    // Gemm1BlockWarps
                           ck_tile::sequence<16, 16, 16>, // Gemm1WarpTile
                           true>;                         // VLayout row-major

using bmap_fp16_trait = ck_tile::TileFmhaTraits<true,  // kPadSeqLenQ
                                                true,  // kPadSeqLenK
                                                true,  // kPadHeadDimQ
                                                true,  // kPadHeadDimV
                                                false, // kHasLogitsSoftCap
                                                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                false, // kStoreLSE
                                                false, // kHasDropout
                                                false, // kHasRandVal
                                                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                -1,     // kBlockPerCu
                                                false>; // kIsVRowMajorSkip

using bmap_fp16_variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using bmap_fp16_mask    = ck_tile::GenericAttentionMask<false>;

using bmap_fp16_problem = ck_tile::BlockFmhaPipelineProblem<ck_tile::half_t, // QDataType
                                                            ck_tile::half_t, // KDataType
                                                            ck_tile::half_t, // VDataType
                                                            float,           // SaccDataType
                                                            float,           // SMPLComputeDataType
                                                            ck_tile::half_t, // BiasDataType
                                                            uint8_t, // RandValOutputDataType
                                                            float,   // LSEDataType
                                                            ck_tile::half_t, // PDataType
                                                            float,           // OaccDataType
                                                            ck_tile::half_t, // ODataType
                                                            bmap_fp16_shape,
                                                            false, // kIsGroupMode
                                                            bmap_fp16_variant,
                                                            bmap_fp16_mask,
                                                            false, // kUseTrLoad
                                                            bmap_fp16_trait>;

using bmap_fp16_pipeline = ck_tile::SpargeBlockMapPipeline<bmap_fp16_problem>;
using bmap_fp16_kernel   = ck_tile::SpargeBlockMapKernel<bmap_fp16_pipeline>;

using kstats_fp16_pipeline = ck_tile::SpargeKStatsPipeline<bmap_fp16_problem>;
using kstats_fp16_kernel   = ck_tile::SpargeKStatsKernel<kstats_fp16_pipeline>;

// bf16: dtype-independent aliases share fp16 chain; only problem differs.
using bmap_bf16_block_tile = bmap_fp16_block_tile;
using bmap_bf16_shape      = bmap_fp16_shape;
using bmap_bf16_trait      = bmap_fp16_trait;
using bmap_bf16_variant    = bmap_fp16_variant;
using bmap_bf16_mask       = bmap_fp16_mask;

using bmap_bf16_problem = ck_tile::BlockFmhaPipelineProblem<ck_tile::bf16_t, // QDataType
                                                            ck_tile::bf16_t, // KDataType
                                                            ck_tile::bf16_t, // VDataType
                                                            float,           // SaccDataType
                                                            float,           // SMPLComputeDataType
                                                            ck_tile::bf16_t, // BiasDataType
                                                            uint8_t, // RandValOutputDataType
                                                            float,   // LSEDataType
                                                            ck_tile::bf16_t, // PDataType
                                                            float,           // OaccDataType
                                                            ck_tile::bf16_t, // ODataType
                                                            bmap_bf16_shape,
                                                            false, // kIsGroupMode
                                                            bmap_bf16_variant,
                                                            bmap_bf16_mask,
                                                            false, // kUseTrLoad
                                                            bmap_bf16_trait>;

using bmap_bf16_pipeline = ck_tile::SpargeBlockMapPipeline<bmap_bf16_problem>;
using bmap_bf16_kernel   = ck_tile::SpargeBlockMapKernel<bmap_bf16_pipeline>;

using kstats_bf16_pipeline = ck_tile::SpargeKStatsPipeline<bmap_bf16_problem>;
using kstats_bf16_kernel   = ck_tile::SpargeKStatsKernel<kstats_bf16_pipeline>;

// ============================================================================
// Workspace layout: caller owns the buffer; we just compute size + offsets.
// Layout = [pooled_k (KDataType) | sim_k (uint8)]. sim_k follows pooled_k with
// no padding (uint8 has alignment 1).
// ============================================================================

namespace {

constexpr int sparge_kN0_for(int hdim_q)
{
    // d=128 instances use kN0=128 (see bmap_fp16_block_tile).
    return (hdim_q == 128) ? 128 : 0;
}

size_t dtype_bytes(const std::string& dt)
{
    if(dt == "fp16" || dt == "bf16")
        return 2;
    return 0;
}

} // namespace

sparge_blockmap_workspace_layout
sparge_blockmap_compute_workspace_layout(sparge_blockmap_traits traits, sparge_blockmap_args args)
{
    const int kN0              = sparge_kN0_for(traits.hdim_q);
    const int N_k              = (kN0 > 0) ? ck_tile::integer_divide_ceil(args.seqlen_k, kN0) : 0;
    const int D                = traits.hdim_q;
    const size_t element_bytes = dtype_bytes(traits.data_type);
    const size_t total_blocks  = static_cast<size_t>(args.batch) * args.nhead_k * N_k;

    sparge_blockmap_workspace_layout layout{};
    layout.pooled_k_offset = 0;
    layout.pooled_k_bytes  = total_blocks * D * element_bytes;
    layout.sim_k_offset    = layout.pooled_k_bytes;
    layout.sim_k_bytes     = total_blocks * sizeof(uint8_t);
    // k_scale LUT: align to fp32 boundary after sim_k (uint8 stream).
    const size_t k_scale_unaligned = layout.sim_k_offset + layout.sim_k_bytes;
    layout.k_scale_offset = (k_scale_unaligned + alignof(float) - 1) & ~(alignof(float) - 1);
    layout.k_scale_bytes  = total_blocks * sizeof(float);
    layout.total_bytes    = layout.k_scale_offset + layout.k_scale_bytes;
    return layout;
}

// ============================================================================
// Stage launchers: read args.workspace_ptr split per layout, run one kernel.
// ============================================================================

namespace {

template <typename KStatsKernel>
void launch_kstats_only(sparge_blockmap_traits traits,
                        sparge_blockmap_args args,
                        const ck_tile::stream_config& s)
{
    const auto layout  = sparge_blockmap_compute_workspace_layout(traits, args);
    auto* ws_base      = static_cast<char*>(args.workspace_ptr);
    void* pooled_k_ptr = ws_base + layout.pooled_k_offset;
    void* sim_k_ptr    = ws_base + layout.sim_k_offset;
    void* k_scale_ptr  = ws_base + layout.k_scale_offset;

    auto [kargs, grids] = sparge_kstats_create_kargs_and_grids<KStatsKernel>(
        args, pooled_k_ptr, sim_k_ptr, k_scale_ptr);
    const dim3 blocks                      = KStatsKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = KStatsKernel::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu>(KStatsKernel{}, grids, blocks, 0, kargs)(s);
}

template <typename BlockMapKernel>
void launch_blockmap_only(sparge_blockmap_traits traits,
                          sparge_blockmap_args args,
                          const ck_tile::stream_config& s)
{
    const auto layout  = sparge_blockmap_compute_workspace_layout(traits, args);
    auto* ws_base      = static_cast<char*>(args.workspace_ptr);
    void* pooled_k_ptr = ws_base + layout.pooled_k_offset;
    void* sim_k_ptr    = ws_base + layout.sim_k_offset;

    auto [kargs, grids] =
        sparge_blockmap_create_kargs_and_grids<BlockMapKernel>(args, pooled_k_ptr, sim_k_ptr);
    const dim3 blocks                      = BlockMapKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = BlockMapKernel::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu>(BlockMapKernel{}, grids, blocks, 0, kargs)(s);
}

} // namespace

// ============================================================================
// Oneshot stages (no timing): caller chains them via launch_kernel.
// ============================================================================

void sparge_kstats_fwd_oneshot(sparge_blockmap_traits traits,
                               sparge_blockmap_args args,
                               const ck_tile::stream_config& s)
{
    if(traits.data_type == "fp16" && traits.hdim_q == 128)
    {
        launch_kstats_only<kstats_fp16_kernel>(traits, args, s);
        return;
    }
    if(traits.data_type == "bf16" && traits.hdim_q == 128)
    {
        launch_kstats_only<kstats_bf16_kernel>(traits, args, s);
        return;
    }
    std::cerr << "sparge_kstats_fwd_oneshot: unsupported config (data_type=" << traits.data_type
              << ", hdim_q=" << traits.hdim_q << ")" << std::endl;
}

void sparge_blockmap_only_fwd_oneshot(sparge_blockmap_traits traits,
                                      sparge_blockmap_args args,
                                      const ck_tile::stream_config& s)
{
    if(traits.data_type == "fp16" && traits.hdim_q == 128)
    {
        launch_blockmap_only<bmap_fp16_kernel>(traits, args, s);
        return;
    }
    if(traits.data_type == "bf16" && traits.hdim_q == 128)
    {
        launch_blockmap_only<bmap_bf16_kernel>(traits, args, s);
        return;
    }
    std::cerr << "sparge_blockmap_only_fwd_oneshot: unsupported config (data_type="
              << traits.data_type << ", hdim_q=" << traits.hdim_q << ")" << std::endl;
}

// ============================================================================
// Combined functions: kstats + blockmap + attention timed together.
// ============================================================================

float sparge_jenga_fwd(sparge_blockmap_traits bmap_t,
                       sparge_blockmap_args bmap_a,
                       fmha_jenga_fwd_traits attn_t,
                       fmha_jenga_fwd_args attn_a,
                       const ck_tile::stream_config& s)
{
    if(s.log_level_ > 0)
        std::cout << ", sparge_kstats_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", sparge_blockmap_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", fmha_jenga_fwd_" << attn_t.data_type << "_d" << attn_t.hdim_q << std::flush;

    return ck_tile::launch_kernel(
        s,
        [=](const ck_tile::stream_config& s_) { sparge_kstats_fwd_oneshot(bmap_t, bmap_a, s_); },
        [=](const ck_tile::stream_config& s_) {
            sparge_blockmap_only_fwd_oneshot(bmap_t, bmap_a, s_);
        },
        [=](const ck_tile::stream_config& s_) { fmha_jenga_fwd_oneshot(attn_t, attn_a, s_); });
}

// Split-launch by per-head pv_threshold: sentinel (>=1e29f) -> kNone binary, finite ->
// kPerWave/kPerBlock
float sparge_sparge_fwd_combined(sparge_blockmap_traits bmap_t,
                                 sparge_blockmap_args bmap_a,
                                 fmha_sparge_fwd_traits attn_t,
                                 fmha_sparge_fwd_args attn_a,
                                 const ck_tile::stream_config& s)
{
    if(s.log_level_ > 0)
        std::cout << ", sparge_kstats_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", sparge_blockmap_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", fmha_sparge_fwd_" << attn_t.data_type << "_d" << attn_t.hdim_q
                  << std::flush;

    // Decide bucket plan. Pull per-head thresholds from device buffer when set,
    // else broadcast the scalar across all heads to a single bucket.
    const int nhead_q = attn_a.nhead_q;
    std::vector<int> false_heads; // pv_threshold >= 1e29f -> kNone binary (mode 0)
    std::vector<int> true_heads;  // finite               -> kPerWave or kPerBlock binary
    false_heads.reserve(nhead_q);
    true_heads.reserve(nhead_q);

    if(attn_a.hp.pv_threshold_per_head_ptr != nullptr)
    {
        std::vector<float> pv_host(nhead_q);
        auto err = hipMemcpy(pv_host.data(),
                             attn_a.hp.pv_threshold_per_head_ptr,
                             static_cast<size_t>(nhead_q) * sizeof(float),
                             hipMemcpyDeviceToHost);
        if(err != hipSuccess)
        {
            std::cerr << "sparge_sparge_fwd_combined: hipMemcpy pv_threshold_per_head failed: "
                      << hipGetErrorString(err) << std::endl;
            return -1.f;
        }
        for(int h = 0; h < nhead_q; ++h)
        {
            if(pv_host[h] >= 1e29f)
                false_heads.push_back(h);
            else
                true_heads.push_back(h);
        }
    }
    else
    {
        // scalar mode: sentinel pv_threshold -> kNone bucket, else mode_compile bucket
        if(attn_a.hp.pv_threshold >= 1e29f)
            for(int h = 0; h < nhead_q; ++h)
                false_heads.push_back(h);
        else
            for(int h = 0; h < nhead_q; ++h)
                true_heads.push_back(h);
    }

    // Skip empty buckets so we never schedule a zero-grid launch.
    const bool need_false = !false_heads.empty();
    const bool need_true  = !true_heads.empty();

    // per-bucket head-remap device buffers, kept alive across the launch
    ck_tile::DeviceMem false_remap_dev(std::max<size_t>(1, false_heads.size() * sizeof(int32_t)));
    ck_tile::DeviceMem true_remap_dev(std::max<size_t>(1, true_heads.size() * sizeof(int32_t)));
    if(need_false)
        false_remap_dev.ToDevice(false_heads.data());
    if(need_true)
        true_remap_dev.ToDevice(true_heads.data());

    // per-bucket attn args; scalar pv_threshold left as-is for the null-remap device fallback
    fmha_sparge_fwd_args attn_false = attn_a;
    fmha_sparge_fwd_args attn_true  = attn_a;
    if(need_false)
    {
        attn_false.hp.head_remap_ptr  = static_cast<const int*>(false_remap_dev.GetDeviceBuffer());
        attn_false.hp.nhead_in_launch = static_cast<int>(false_heads.size());
        attn_false.hp.pv_skip_compile = false; // legacy bool — kept consistent
        attn_false.hp.pv_mode_compile = 0;     // route to kNone binary
    }
    if(need_true)
    {
        attn_true.hp.head_remap_ptr  = static_cast<const int*>(true_remap_dev.GetDeviceBuffer());
        attn_true.hp.nhead_in_launch = static_cast<int>(true_heads.size());
        attn_true.hp.pv_skip_compile = true; // legacy bool — kept consistent
        // pv_mode_compile inherits from attn_a so --pv_mode=none with finite threshold stays a
        // control measurement
    }

    // chain: kstats -> blockmap -> [fmha_false?] -> [fmha_true?]; empty buckets emit a no-op lambda
    auto cb_kstats = [=](const ck_tile::stream_config& s_) {
        sparge_kstats_fwd_oneshot(bmap_t, bmap_a, s_);
    };
    auto cb_bmap = [=](const ck_tile::stream_config& s_) {
        sparge_blockmap_only_fwd_oneshot(bmap_t, bmap_a, s_);
    };
    auto cb_fmha_false = [=](const ck_tile::stream_config& s_) {
        if(need_false)
            fmha_sparge_fwd_oneshot(attn_t, attn_false, s_);
    };
    auto cb_fmha_true = [=](const ck_tile::stream_config& s_) {
        if(need_true)
            fmha_sparge_fwd_oneshot(attn_t, attn_true, s_);
    };

    // launch_kernel returns elapsed ms for the whole chain when timing is on.
    // We always pass 4 callables and gate execution inside the lambda; this
    // keeps the timing contract stable, while a no-op lambda has negligible
    // (~ns) cost compared to the saved 5-15us host launch.
    return ck_tile::launch_kernel(s, cb_kstats, cb_bmap, cb_fmha_false, cb_fmha_true);
}
