// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>
#include <tuple>

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {

const ck_tile::stream_config stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

// D128 batch-mode split-KV over a paged cache, checked against the CPU reference.
template <typename DataTypeConfig>
fwd_result run_paged_splitkv(int seqlen_q, int num_splits, int page_block_size)
{
    return fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                        1,               // batch
                                        8,               // nhead
                                        8,               // nhead_k
                                        {seqlen_q},      // seqlen_qs
                                        {128},           // seqlen_ks
                                        128,             // hdim_q
                                        128,             // hdim_v
                                        0,               // seqlen_knew
                                        {-1},            // seqlen_qpads
                                        {-1},            // seqlen_kpads
                                        {},              // q_eff_lens_per_batch
                                        {},              // kv_eff_lens_per_batch
                                        0,               // rotary_dim
                                        true,            // i_perm
                                        true,            // o_perm
                                        0,               // scale_s
                                        0,               // logits_soft_cap
                                        true,            // is_v_rowmajor
                                        false,           // lse
                                        page_block_size, // page_block_size
                                        false,           // use_cache_batch_idx
                                        "n",             // bias_str
                                        0.0f,            // p_drop
                                        0,               // drop_seed
                                        0,               // drop_offset
                                        false,           // drop_prefs
                                        "0",             // mask_str
                                        "n",             // qscale_str
                                        true,            // is_rotary_interleaved
                                        num_splits,      // num_splits
                                        "uf",            // init_method
                                        11939,           // seed
                                        1,               // do_validation
                                        0,               // init_sink_value
                                        0,               // pack_gqa
                                        stream_config);
}

class FmhaFwdSplitKvGfx1100
    : public TestWithParam<std::tuple<std::tuple<int, int, int>, std::string>>
{
};

TEST_P(FmhaFwdSplitKvGfx1100, CpuValidated)
{
    if(ck_tile::get_device_name().compare(0, 7, "gfx1100") != 0)
        GTEST_SKIP() << "gfx1100-only split-KV regression";

    const auto [shape, dtype]                          = GetParam();
    const auto [seqlen_q, num_splits, page_block_size] = shape;
    // num_splits == 0 asks the runner's heuristic, which must resolve to one split.
    if(num_splits == 0)
        ASSERT_EQ(override_num_splits_if_necessary(1, 8, seqlen_q, 128, 0.0f, num_splits), 1);

    const auto result = dtype == "bf16"
                            ? run_paged_splitkv<FmhaFwdBf16>(seqlen_q, num_splits, page_block_size)
                            : run_paged_splitkv<FmhaFwdFp16>(seqlen_q, num_splits, page_block_size);

    ASSERT_EQ(result, fwd_result::success);
}

// {seqlen_q, num_splits, page_block_size}
INSTANTIATE_TEST_SUITE_P(PageAlignment,
                         FmhaFwdSplitKvGfx1100,
                         Combine(Values(std::tuple{128, 2, 80},
                                        std::tuple{128, 2, 96},
                                        std::tuple{128, 1, 16},
                                        std::tuple{2048, 0, 16}),
                                 Values("fp16", "bf16")));

} // namespace
