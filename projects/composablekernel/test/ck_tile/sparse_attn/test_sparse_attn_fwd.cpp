// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>
#include <tuple>

#include "example/ck_tile/50_sparse_attn/sparse_attn_fwd.hpp"
#include "example/ck_tile/50_sparse_attn/sparse_attn_fwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaSparseFwdBf16 // or FmhaSparseFwdFp16
#endif

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run the long test matrix (mirrors smoke_test_sparse_attn.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_SPARSE_ATTN_LONG_TESTS)

#define CHECK_RESULT(result)                                            \
    do                                                                  \
    {                                                                   \
        if(result == sparse_attn_result::skipped)                       \
            GTEST_SKIP() << "Unsupported configuration for this build"; \
        ASSERT_EQ(result, sparse_attn_result::success);                 \
    } while(0)

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

auto EnableTestIf(bool condition)
{
    return ValuesIn(condition ? std::vector<bool>{true} : std::vector<bool>{});
}

// Thin wrapper over sparse_attn_fwd_run that pins the many rarely-varied knobs to sane defaults,
// exposing only the parameters the test groups below sweep. d/block are fixed at 128 (the only
// supported tile). seqlen is kept small so the CPU block reference stays tractable.
template <typename T>
sparse_attn_result run(const std::string& api,
                       int batch,
                       int nhead,
                       int nhead_k,
                       int seqlen,
                       float sparsity,
                       const std::string& mask_str,
                       sparse_attn_mode mode,
                       bool perm                      = true,
                       const std::string& bias_str    = "n",
                       bool attention_sink            = false,
                       float simthreshold             = 0.0f,
                       float pvthreshd                = 0.0f,
                       const std::string& sparge_mode = "topk",
                       bool smooth_k                  = true,
                       float logits_soft_cap          = 0.0f,
                       const std::string& qscale      = "perwarp",
                       const std::string& qkdtype     = "int8")
{
    if(nhead_k < 0)
        nhead_k = nhead;
    return sparse_attn_fwd_run<T>(
        api,
        batch,
        nhead,
        nhead_k,
        seqlen, // seqlen_q
        seqlen, // seqlen_k
        128,    // hdim_q
        128,    // hdim_v
        sparsity,
        simthreshold,
        mask_str,
        attention_sink,
        128,  // block_size
        perm, // i_perm
        perm, // o_perm
        true, // is_v_rowmajor
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        pvthreshd,
        sparge_mode,
        false, // perhead_test
        "",    // sparsity_per_head_csv
        "",    // sim_per_head_csv
        "",    // pvthreshd_per_head_csv
        smooth_k,
        false, // print_sparsity
        stream_config,
        mode,
        false,         // json_out
        std::string(), // json_file
        bias_str,
        0.0f, // scale_s_user
        logits_soft_cap,
        qscale,
        qkdtype);
}

// ---------------------------------------------------------------------------
// Jenga
// ---------------------------------------------------------------------------
class Jenga : public TestWithParam<
                  std::tuple<sparse_attn_mode, std::tuple<int, int, int, int, float, std::string>>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Jenga);

INSTANTIATE_TEST_SUITE_P(TestCkTileSparseAttn,
                         Jenga,
                         Combine(Values(sparse_attn_mode::batch, sparse_attn_mode::group),
                                 Values(std::tuple{1, 4, -1, 1024, 0.5f, "n"},
                                        std::tuple{2, 4, -1, 2048, 0.3f, "n"},
                                        std::tuple{1, 8, 2, 1024, 0.5f, "n"}, // GQA
                                        std::tuple{2, 4, -1, 1024, 0.5f, "e:0"},
                                        std::tuple{2, 4, -1, 1024, 0.5f, "e:1"})));

TEST_P(Jenga, DataTypeConfig)
{
    auto [mode, dims]                         = GetParam();
    auto [batch, nhead, nhead_k, s, sp, bias] = dims;
    auto result = run<DataTypeConfig>("jenga", batch, nhead, nhead_k, s, sp, "0", mode, true, bias);
    CHECK_RESULT(result);
}

// ---------------------------------------------------------------------------
// VSA
// ---------------------------------------------------------------------------
class VSA : public TestWithParam<
                std::tuple<sparse_attn_mode,
                           std::tuple<int, int, int, int, float, std::string, std::string>>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VSA);

INSTANTIATE_TEST_SUITE_P(
    TestCkTileSparseAttn,
    VSA,
    Combine(Values(sparse_attn_mode::batch, sparse_attn_mode::group),
            Values(std::tuple{1, 4, -1, 1024, 0.5f, "0", "n"},
                   std::tuple{1, 4, -1, 2048, 0.5f, "t", "n"},        // top-left causal
                   std::tuple{1, 4, -1, 2048, 0.5f, "b", "n"},        // bottom-right causal
                   std::tuple{1, 4, -1, 2048, 0.5f, "t:128,32", "n"}, // SWA
                   std::tuple{1, 8, 2, 2048, 0.5f, "t", "n"},         // GQA
                   std::tuple{2, 4, -1, 2048, 0.5f, "0", "e:1"},
                   std::tuple{2, 4, -1, 2048, 0.5f, "t", "a"})));

TEST_P(VSA, DataTypeConfig)
{
    auto [mode, dims]                               = GetParam();
    auto [batch, nhead, nhead_k, s, sp, mask, bias] = dims;
    auto result = run<DataTypeConfig>("vsa", batch, nhead, nhead_k, s, sp, mask, mode, true, bias);
    CHECK_RESULT(result);
}

// ---------------------------------------------------------------------------
// Sparge (block selection: topk / cdf)
// ---------------------------------------------------------------------------
class Sparge
    : public TestWithParam<std::tuple<
          sparse_attn_mode,
          std::tuple<int, int, int, int, float, std::string, std::string, std::string, bool>>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Sparge);

INSTANTIATE_TEST_SUITE_P(
    TestCkTileSparseAttn,
    Sparge,
    Combine(Values(sparse_attn_mode::batch, sparse_attn_mode::group),
            // batch, nhead, nhead_k, seqlen, sparsity, mask, bias, sparge_mode, sink
            Values(std::tuple{1, 4, -1, 2048, 0.5f, "0", "n", "topk", false},
                   std::tuple{1, 4, -1, 2048, 0.5f, "t", "n", "topk", false},
                   std::tuple{1, 4, -1, 2048, 0.5f, "b", "n", "topk", false},
                   std::tuple{1, 4, -1, 2048, 0.6f, "0", "n", "cdf", false},
                   std::tuple{1, 4, -1, 2048, 0.5f, "t", "n", "topk", true}, // attention sink
                   std::tuple{1, 8, 2, 2048, 0.5f, "0", "n", "topk", false}, // GQA
                   std::tuple{2, 4, -1, 2048, 0.5f, "0", "e:1", "topk", false},
                   std::tuple{2, 4, -1, 2048, 0.5f, "t", "a", "topk", false})));

TEST_P(Sparge, DataTypeConfig)
{
    auto [mode, dims]                                            = GetParam();
    auto [batch, nhead, nhead_k, s, sp, mask, bias, smode, sink] = dims;
    auto result                                                  = run<DataTypeConfig>(
        "sparge", batch, nhead, nhead_k, s, sp, mask, mode, true, bias, sink, 0.0f, 0.0f, smode);
    CHECK_RESULT(result);
}

// Sparge extra knobs (batch, no-mask): pv-skip threshold, K smoothing off, sim threshold.
class SpargeKnobs : public TestWithParam<std::tuple<float, bool, float>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SpargeKnobs);

INSTANTIATE_TEST_SUITE_P(TestCkTileSparseAttn,
                         SpargeKnobs,
                         // pvthreshd, smooth_k, simthreshold
                         Values(std::tuple{3.0f, true, 0.0f},
                                std::tuple{0.0f, false, 0.0f},
                                std::tuple{0.0f, true, 0.5f}));

TEST_P(SpargeKnobs, DataTypeConfig)
{
    auto [pvthreshd, smooth_k, simthreshold] = GetParam();
    auto result                              = run<DataTypeConfig>("sparge",
                                      1,
                                      4,
                                      -1,
                                      2048,
                                      0.6f,
                                      "0",
                                      sparse_attn_mode::batch,
                                      true,
                                      "n",
                                      false,
                                      simthreshold,
                                      pvthreshd,
                                      "topk",
                                      smooth_k);
    CHECK_RESULT(result);
}

// ---------------------------------------------------------------------------
// Sparge-Sage (quantized). bf16 only; fp16 build skips via runner.
// ---------------------------------------------------------------------------
class SpargeSage : public TestWithParam<
                       std::tuple<sparse_attn_mode,
                                  std::string, // qkdtype
                                  std::string, // qscale
                                  std::tuple<int, int, int, int, float, std::string, std::string>>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SpargeSage);

INSTANTIATE_TEST_SUITE_P(TestCkTileSparseAttn,
                         SpargeSage,
                         Combine(Values(sparse_attn_mode::batch, sparse_attn_mode::group),
                                 Values("int8", "fp8"),
                                 Values("perwarp", "perblock", "perthread", "pertensor"),
                                 // batch, nhead, nhead_k, seqlen, sparsity, mask, bias
                                 Values(std::tuple{2, 4, -1, 1024, 0.5f, "0", "n"},
                                        std::tuple{2, 4, -1, 1024, 0.5f, "t", "n"},
                                        std::tuple{2, 8, 2, 1024, 0.5f, "0", "n"}, // GQA
                                        std::tuple{2, 4, -1, 1024, 0.5f, "t", "a"})));

TEST_P(SpargeSage, DataTypeConfig)
{
    // sparge_sage uses ds_read_tr transpose-load + FP8/INT8 MFMA, supported only on gfx950/MI350.
    if(!ck_tile::is_gfx95_supported())
        GTEST_SKIP() << "sparge_sage requires gfx950";

    auto [mode, qkdtype, qscale, dims]              = GetParam();
    auto [batch, nhead, nhead_k, s, sp, mask, bias] = dims;
    auto result                                     = run<DataTypeConfig>("sparge_sage",
                                      batch,
                                      nhead,
                                      nhead_k,
                                      s,
                                      sp,
                                      mask,
                                      mode,
                                      true,
                                      bias,
                                      false,
                                      0.0f,
                                      0.0f,
                                      "topk",
                                      true,
                                      0.0f,
                                      qscale,
                                      qkdtype);
    CHECK_RESULT(result);
}

// ---------------------------------------------------------------------------
// Logits soft cap (Gemma-style; NO_BIAS only) across jenga/vsa/sparge.
// ---------------------------------------------------------------------------
class LogitsSoftCap
    : public TestWithParam<std::tuple<sparse_attn_mode, std::tuple<std::string, std::string>>>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LogitsSoftCap);

INSTANTIATE_TEST_SUITE_P(TestCkTileSparseAttn,
                         LogitsSoftCap,
                         Combine(Values(sparse_attn_mode::batch, sparse_attn_mode::group),
                                 // api, mask
                                 Values(std::tuple{std::string("jenga"), std::string("0")},
                                        std::tuple{std::string("vsa"), std::string("t")},
                                        std::tuple{std::string("sparge"), std::string("t")})));

TEST_P(LogitsSoftCap, DataTypeConfig)
{
    auto [mode, cfg] = GetParam();
    auto [api, mask] = cfg;
    auto result      = run<DataTypeConfig>(
        api, 2, 4, -1, 1024, 0.5f, mask, mode, true, "n", false, 0.0f, 0.0f, "topk", true, 8.0f);
    CHECK_RESULT(result);
}
