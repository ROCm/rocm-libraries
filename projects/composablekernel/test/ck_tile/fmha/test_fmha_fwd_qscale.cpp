// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_fmha_fwd_common.hpp"

// Every other suite pins the scale to TestConfigs<T>::qscale_str, leaving the rest untested.
// Only gfx125x has more than one, so the sweep is compiled in there alone.
#ifdef CK_TILE_TEST_FMHA_QSCALE_SWEEP
enum class sink_kind
{
    none,
    gptoss,
    streamllm
};

// Full causal (left=-1) makes y == y_total, so x_start is 0 and the sink phase collapses.
// This row checks that the has_sink instantiation matches the sinkless answer; the live
// sink phase is carried by the local-window tuples below instead.
constexpr auto kStreamLlmMask = "b:-1,0,2";

constexpr bool kIsMxFp8Config =
    ck_tile::is_any_of<DataTypeConfig, FmhaFwdMxFp8, FmhaFwdMxFp8Bf16>::value;
constexpr bool kIsFp8Config = ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp8, FmhaFwdFp8Bf16>::value;

// On gfx125x the FP8 suites below must reach qr_tdm_sched, not fall back to another pipeline.
bool ExpectsFp8Sched(bool is_v_rowmajor)
{
    // MX has no other gfx125x pipeline; FP8 prefers it at every seqlen_q.
    return ck_tile::is_gfx125_supported() && (kIsFp8Config || (kIsMxFp8Config && is_v_rowmajor));
}

void ExpectFp8Sched(const std::string& kernel_name, bool is_v_rowmajor)
{
    if(ExpectsFp8Sched(is_v_rowmajor))
        EXPECT_NE(kernel_name.find("qr_tdm_sched"), std::string::npos) << kernel_name;
}

// Unlike CHECK_RESULT, a missing instance fails wherever qr_tdm_sched is expected.
#define CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor)                           \
    do                                                                          \
    {                                                                           \
        if(result == fwd_result::no_instance && ExpectsFp8Sched(is_v_rowmajor)) \
            FAIL() << "No FP8 qr_tdm_sched instance for current parameters";    \
        CHECK_RESULT(result);                                                   \
    } while(0)

// Skips what qr_tdm_sched leaves out by design: MX bias and gfx125x MX on col-major V.
#define SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, bias_str)                          \
    do                                                                           \
    {                                                                            \
        if(kIsMxFp8Config && std::string_view(bias_str) != "n")                  \
            GTEST_SKIP() << "MX has no elementwise bias";                        \
        if(kIsMxFp8Config && !(is_v_rowmajor) && ck_tile::is_gfx125_supported()) \
            GTEST_SKIP() << "gfx125x MX runs on row-major V only";               \
    } while(0)

class QuantScale
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      sink_kind,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

// hdim 128 is where perhead and blockscale exist. The non-multiple seqlens select the
// seqlen-padded instances; the last tuple is a tile multiple so the unpadded pack-GQA path
// is covered too. This sweep pins bias to "n"; fp8 bias + sink is carried by SinkWindowMask.
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScale,
    Combine(ModeValues,
            QScaleValues,
            Values(sink_kind::none, sink_kind::gptoss, sink_kind::streamllm),
            Values(std::tuple{2, 2, 1, 55, 256, "0"},   // GQA, seqlen_q << seqlen_k
                   std::tuple{1, 3, -1, 100, 51, "0"},  // plain MHA, seqlen_q > seqlen_k
                   std::tuple{2, 1, -1, 99, 256, "1"},  // causal
                   std::tuple{1, 2, 1, 1024, 256, "2"}, // GQA, causal bottom-right
                   std::tuple{1, 4, 2, 256, 256, "0"},  // Pack-GQA: ratio 2, no mask, s%128==0
                   // The two local-window tuples below are the only non-empty sink phases
                   // here; causal masks collapse it, and only then does the descale index
                   // leave k_origin. sink=128 == kN0 is the prologue jump.
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128"},
                   // sink=512 gives num_sink_loop 4, reaching the mainloop jump that
                   // sink <= kN0 never does; GQA ratio 2 crosses it with the descale stride.
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

// init=3 fills Q/K/V up to the fp8 maximum, which only stands for a real tensor when a
// descale maps that maximum back to qkv_max. Without a descale the values stay at the fp8
// scale, the logits reach ~3e5 in the exp2 domain and the softmax collapses to an argmax.
// There half an fp32 ulp of the row max is worth 1% of the exponential, and the fp8 P
// operand snaps that factor out of the numerator while the row sum keeps it, so the whole
// row lands 1% off. Anchor the no-scale inputs on qkv_max, as the three descales do.
const char* qscale_init_method(std::string_view qscale)
{
    return qscale == "n" ? "0" : init_method;
}

TEST_P(QuantScale, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    const std::string mask = sink == sink_kind::streamllm ? kStreamLlmMask : mask_str;
    const int init_sink    = sink == sink_kind::gptoss ? 1 : 0;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// Elementwise bias across the quantization scales, with the sink logit and sink tokens.
class QuantScaleBias : public QuantScale
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleBias,
    Combine(ModeValues,
            QScaleValues,
            // streamllm swaps in kStreamLlmMask; the t:128,30 rows carry the live sink tokens.
            Values(sink_kind::none, sink_kind::gptoss, sink_kind::streamllm),
            Values(std::tuple{2, 2, 1, 55, 256, "0"},   // GQA, seqlen_q << seqlen_k
                   std::tuple{1, 3, -1, 300, 129, "0"}, // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1"}, // causal
                   std::tuple{1, 4, 2, 256, 256, "0"},  // GQA ratio 2 (bias turns Pack-GQA off)
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128"},
                   // num_sink_loop 4: the bias window jumps from the sink tokens mid-loop
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

TEST_P(QuantScaleBias, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    const std::string mask = sink == sink_kind::streamllm ? kStreamLlmMask : mask_str;
    const int init_sink    = sink == sink_kind::gptoss ? 1 : 0;
    SKIP_IF_MX_UNSUPPORTED(def_is_v_rowmajor, "e");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "e",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, def_is_v_rowmajor);
    ExpectFp8Sched(kernel_name, def_is_v_rowmajor);
}

// LSE across every quantization scale; MXFP8 adds row-major V, the layout gfx125x MX runs on.
class QuantScaleLse
    : public TestWithParam<
          std::tuple<mode_enum, const char*, bool, std::tuple<int, int, int, int, int>>>
{
};

static auto LseVRowmajorValues =
    ValuesIn(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8> ? std::vector<bool>{false, true}
                                                          : std::vector<bool>{def_is_v_rowmajor});

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleLse,
                         Combine(ModeValues,
                                 QScaleValues,
                                 LseVRowmajorValues,
                                 Values(std::tuple{2, 2, 1, 55, 256},       // GQA, short query
                                        std::tuple{1, 3, -1, 100, 51},      // seqlen_q > seqlen_k
                                        std::tuple{1, 4, 2, 256, 256},      // tile multiple
                                        std::tuple{2, 2, -1, 300, 1000}))); // query/key tails

TEST_P(QuantScaleLse, DataTypeConfig)
{
    auto [mode, qscale, is_v_rowmajor, dims]         = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        true,  // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask_str
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Causal and sliding-window masks across every quantization scale, with and without LSE.
class QuantScaleMask
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      bool,
                                      bool,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMask,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true), // lse
            LseVRowmajorValues,
            Values(std::tuple{1, 2, 1, 300, 1000, "t"},       // causal top-left, key tail
                   std::tuple{2, 2, -1, 1000, 300, "b"},      // causal bottom-right, empty rows
                   std::tuple{1, 2, 1, 256, 256, "t:128,30"}, // sliding window, tile multiple
                   std::tuple{1, 3, -1, 500, 700, "b:128,30"},
                   std::tuple{2, 2, 1, 129, 161, "b:-1,64"})));

TEST_P(QuantScaleMask, DataTypeConfig)
{
    auto [mode, qscale, lse, is_v_rowmajor, dims_mask]         = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Variable-length groups with physical Q/K padding; the runner drops K padding for MX types.
class QuantScaleGroupPad
    : public TestWithParam<
          std::tuple<std::tuple<int, int>, const char*, bool, std::string, std::tuple<int, int>>>
{
};

// Every head dim of the scheduled FP8 pipeline; other data types keep head dim 128.
static const auto kGroupPadHdims =
    kIsMxFp8Config || kIsFp8Config
        ? std::vector<std::tuple<int, int>>{{64, 64}, {128, 128}, {192, 128}, {256, 256}}
        : std::vector<std::tuple<int, int>>{{128, 128}};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleGroupPad,
                         Combine(ValuesIn(kGroupPadHdims),
                                 QScaleValues,
                                 Values(false, true), // lse
                                 Values("0", "t", "b:128,30"),
                                 Values(std::tuple{2, 1}, std::tuple{4, 2})));

TEST_P(QuantScaleGroupPad, DataTypeConfig)
{
    auto [hdims, qscale, lse, mask_str, heads] = GetParam();
    auto [hdim, hdim_v]                        = hdims;
    auto [nhead, nhead_k]                      = heads;

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        3, // batch
        nhead,
        nhead_k,
        {100, 300, 129},
        {161, 600, 129},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v),
        0,               // seqlen_knew
        {128, 384, 256}, // seqlen_qpads
        {256, 640, 256}, // seqlen_kpads
        {},              // q_eff_lens_per_batch
        {},              // kv_eff_lens_per_batch
        0,               // rotary_dim
        true,            // i_perm
        true,            // o_perm
        0,               // scale_s
        0,               // logits_soft_cap
        true,            // is_v_rowmajor
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, true);
    ExpectFp8Sched(kernel_name, true);
}

// Per-head sink logit (init_sink) and mask sink tokens across every quantization scale and LSE.
class QuantScaleSink
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      bool,
                                      bool,
                                      std::tuple<int, int, int, int, int, std::string, int>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleSink,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true), // lse
            LseVRowmajorValues,
            Values(std::tuple{2, 2, 1, 300, 1000, "0", 1},  // sink logit, GQA, key tail
                   std::tuple{1, 3, -1, 1000, 256, "0", 1}, // sink logit, seqlen_q > seqlen_k
                   std::tuple{1, 4, 2, 256, 256, "0", 1},   // sink logit, tile multiple
                   std::tuple{2, 2, -1, 1000, 300, "b", 1}, // sink logit, empty rows
                   // gpt-oss: sliding window plus sink logit, GQA ratio 4
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", 1},
                   std::tuple{1, 2, -1, 700, 1300, "b:128,0", 1},
                   // StreamingLLM sink tokens: prologue jump, num_sink_loop 4, key tail
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128", 0},
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", 1},
                   std::tuple{1, 2, -1, 1000, 1300, "b:128,30,200", 1})));

TEST_P(QuantScaleSink, DataTypeConfig)
{
    auto [mode, qscale, lse, is_v_rowmajor, dims_mask]                    = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, init_sink] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Head dim 64 across every quantization scale: masks, sink logit and tokens, bias, LSE.
using QuantScaleHdimParam =
    std::tuple<mode_enum,
               const char*,
               bool,
               std::tuple<int, int, int, int, int, std::string, std::string, int>>;

class QuantScaleHdim64 : public TestWithParam<QuantScaleHdimParam>
{
};

void RunQuantScaleHdim(int hdim, const QuantScaleHdimParam& param, int hdim_v = -1)
{
    auto [mode, qscale, lse, dims]                                                  = param;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, bias_str, init_sink] = dims;
    // gfx125x MXFP8 runs on row-major V.
    const bool is_v_rowmajor = kIsMxFp8Config || def_is_v_rowmajor;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, bias_str);

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v < 0 ? hdim : hdim_v),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        bias_str,
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim64,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim64, DataTypeConfig) { RunQuantScaleHdim(64, GetParam()); }

// Head dim 256 (one workgroup per CU on gfx125x) with the same rows as head dim 64.
class QuantScaleHdim256 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim256,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim256, DataTypeConfig) { RunQuantScaleHdim(256, GetParam()); }

// Decode-length queries in batch mode on every head dim, GQA and MHA.
class QuantScaleShortQuery
    : public TestWithParam<
          std::tuple<int,
                     const char*,
                     bool,
                     std::tuple<int, int, int, int, int, std::string, std::string, int>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleShortQuery,
    Combine(Values(64, 128, 256),
            QScaleValues,
            Values(false, true),                                           // lse
            Values(std::tuple{2, 8, 2, 1, 1024, "0", "n", 0},              // GQA 4, Pack-GQA n/pt
                   std::tuple{3, 4, -1, 1, 500, "0", "n", 0},              // MHA, key tail
                   std::tuple{2, 8, 2, 4, 700, "b", "n", 0},               // causal bottom-right
                   std::tuple{1, 4, 1, 16, 2048, "b", "n", 0},             // causal, GQA ratio 4
                   std::tuple{2, 8, 2, 16, 1000, "b:127,0", "n", 1},       // gpt-oss
                   std::tuple{1, 8, 2, 4, 2048, "b:1023,0,4", "n", 0},     // sink tokens
                   std::tuple{2, 4, 1, 1, 300, "0", "n", 1},               // sink logit
                   std::tuple{1, 2, -1, 55, 256, "b", "e", 0},             // elementwise bias
                   std::tuple{2, 4, 2, 127, 300, "b:1023,0,4", "n", 1}))); // 127 = tile - 1

TEST_P(QuantScaleShortQuery, DataTypeConfig)
{
    auto [hdim, qscale, lse, dims] = GetParam();
    RunQuantScaleHdim(hdim, {mode_enum::batch, qscale, lse, dims});
}

// Decode without Pack-GQA in both layouts; asserts which decode kernel each shape reaches.
using MergedDecodeDims = std::tuple<int, int, int, int, int, std::string, int, bool>;

class QuantScaleMergedDecode
    : public TestWithParam<std::tuple<int, const char*, bool, MergedDecodeDims>>
{
};

// Packed GQA on the 32-row tile up to 32 rows, on the 128-row tile up to 128 with GQA, else plain.
std::string ExpectedDecodeKernel(int nhead, int nhead_k, int seqlen_q, bool perm)
{
    const int ratio = nhead_k < 0 ? 1 : nhead / nhead_k;
    const int rows  = seqlen_q * ratio;
    // b*s*h*d query heads only fold into rows for a single query.
    if(seqlen_q > 1 && !perm)
        return "sched";
    if(rows <= 32)
        return "packgqa_32row";
    if(rows <= 128 && ratio > 1)
        return "packgqa";
    return "sched";
}

std::string DecodeKernelOf(const std::string& kernel_name)
{
    if(kernel_name.find("_qr_tdm_sched_") == std::string::npos)
        return "other";
    if(kernel_name.find("_packgqa_") == std::string::npos)
        return "sched";
    return kernel_name.find("_b32x") != std::string::npos ? "packgqa_32row" : "packgqa";
}

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMergedDecode,
    Combine(Values(64, 128, 256),
            QScaleValues,
            Values(false, true),                                            // lse
            Values(MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, true},        // GQA 4, no mask
                   MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, false},       // same, b*s*h*d
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, true},        // 64 rows, causal
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, false},       // b*s*h*d, s > 1
                   MergedDecodeDims{2, 32, 2, 1, 513, "b:127,0", 1, false}, // gpt-oss, b*s*h*d
                   MergedDecodeDims{1, 32, 4, 8, 300, "0", 1, true},        // sink logit, 64 rows
                   MergedDecodeDims{2, 8, -1, 2, 300, "b", 0, true},        // MHA, 2 rows
                   MergedDecodeDims{1, 8, 1, 20, 300, "b", 0, true},        // 160 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 1, true},  // empty KV, sink logit, 4 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 0, true},  // empty KV, 4 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 1, true}, // empty KV, sink logit, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 0, true}, // empty KV, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 5, "b", 1, true}, // fully masked rows, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 2, "b", 0, true},  // fully masked rows, 16 rows
                   MergedDecodeDims{1, 16, 2, 8, 8, "1", 0, true}, // square causal, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 4, "1", 0, true},  // square causal, 16 rows
                   MergedDecodeDims{1, 32, 2, 8, 700, "0", 0, true}, // full 128-row tile, key tail
                   MergedDecodeDims{1, 32, 4, 8, 2048, "b:128,30,512", 0, true}))); // sink tokens

void RunMergedDecode(
    int hdim, int hdim_v, const char* qscale, bool lse, const MergedDecodeDims& dims)
{
    if(!(kIsMxFp8Config || kIsFp8Config) || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "decode kernels are gfx125x FP8 only";
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, init_sink, perm] = dims;
    const bool is_v_rowmajor = kIsMxFp8Config || def_is_v_rowmajor;
    // Blockscale's stochastically rounded P over only 8 rows per head needs a 3-sigma gain check.
    const bool wide_alpha = std::string_view(qscale) == "bs" && hdim_v == 128 &&
                            dims == MergedDecodeDims{1, 32, 2, 8, 700, "0", 0, true};

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        perm, // i_perm
        perm, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        0,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name,
        wide_alpha ? 3.0 : 2.0);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    EXPECT_EQ(DecodeKernelOf(kernel_name), ExpectedDecodeKernel(nhead, nhead_k, seqlen_q, perm))
        << kernel_name;
}

TEST_P(QuantScaleMergedDecode, DataTypeConfig)
{
    auto [hdim, qscale, lse, dims] = GetParam();
    RunMergedDecode(hdim, hdim, qscale, lse, dims);
}

// Head dim 192 with a 128-wide V (D128 PV, K64 QK) on the rows of head dims 64 and 256.
class QuantScaleHdim192 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim192,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim192, DataTypeConfig) { RunQuantScaleHdim(192, GetParam(), 128); }

// Decode-length queries on head dim 192 / V 128, as QuantScaleShortQuery.
class QuantScaleShortQueryHdim192 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleShortQueryHdim192,
    Combine(Values(mode_enum::batch),
            QScaleValues,
            Values(false, true),                                           // lse
            Values(std::tuple{2, 8, 2, 1, 1024, "0", "n", 0},              // GQA 4, Pack-GQA n/pt
                   std::tuple{3, 4, -1, 1, 500, "0", "n", 0},              // MHA, key tail
                   std::tuple{2, 8, 2, 4, 700, "b", "n", 0},               // causal bottom-right
                   std::tuple{1, 4, 1, 16, 2048, "b", "n", 0},             // causal, GQA ratio 4
                   std::tuple{2, 8, 2, 16, 1000, "b:127,0", "n", 1},       // gpt-oss
                   std::tuple{1, 8, 2, 4, 2048, "b:1023,0,4", "n", 0},     // sink tokens
                   std::tuple{2, 4, 1, 1, 300, "0", "n", 1},               // sink logit
                   std::tuple{1, 2, -1, 55, 256, "b", "e", 0},             // elementwise bias
                   std::tuple{2, 4, 2, 127, 300, "b:1023,0,4", "n", 1}))); // 127 = tile - 1

TEST_P(QuantScaleShortQueryHdim192, DataTypeConfig) { RunQuantScaleHdim(192, GetParam(), 128); }

// Decode kernel selection on head dim 192 / V 128, as QuantScaleMergedDecode.
class QuantScaleMergedDecodeHdim192
    : public TestWithParam<std::tuple<const char*, bool, MergedDecodeDims>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMergedDecodeHdim192,
    Combine(QScaleValues,
            Values(false, true),                                            // lse
            Values(MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, true},        // GQA 4, no mask
                   MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, false},       // same, b*s*h*d
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, true},        // 64 rows, causal
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, false},       // b*s*h*d, s > 1
                   MergedDecodeDims{2, 32, 2, 1, 513, "b:127,0", 1, false}, // gpt-oss, b*s*h*d
                   MergedDecodeDims{1, 32, 4, 8, 300, "0", 1, true},        // sink logit, 64 rows
                   MergedDecodeDims{2, 8, -1, 2, 300, "b", 0, true},        // MHA, 2 rows
                   MergedDecodeDims{1, 8, 1, 20, 300, "b", 0, true},        // 160 rows
                   MergedDecodeDims{2, 64, 8, 4, 1000, "0", 0, true},       // GQA 8, 32 rows
                   MergedDecodeDims{1, 32, 4, 8, 700, "b:127,0", 1, true},  // masked sink, 64 rows
                   MergedDecodeDims{1, 32, 2, 1, 2048, "b:1023,0,4", 0, true}, // 16 groups
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 1, true},  // empty KV, sink logit, 4 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 0, true},  // empty KV, 4 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 1, true}, // empty KV, sink logit, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 0, true}, // empty KV, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 5, "b", 1, true}, // fully masked rows, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 2, "b", 0, true},  // fully masked rows, 16 rows
                   MergedDecodeDims{1, 16, 2, 8, 8, "1", 0, true}, // square causal, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 4, "1", 0, true},  // square causal, 16 rows
                   MergedDecodeDims{1, 32, 2, 8, 700, "0", 0, true}, // full 128-row tile, key tail
                   MergedDecodeDims{1, 32, 4, 8, 2048, "b:128,30,512", 0, true}))); // sink tokens

TEST_P(QuantScaleMergedDecodeHdim192, DataTypeConfig)
{
    auto [qscale, lse, dims] = GetParam();
    RunMergedDecode(192, 128, qscale, lse, dims);
}

// Elementwise bias with LSE, including the early exit of empty rows next to a sink logit.
class QuantScaleBiasLse : public QuantScale
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBiasLse,
                         Combine(ModeValues,
                                 QScaleValues,
                                 Values(sink_kind::none, sink_kind::gptoss),
                                 Values(std::tuple{1, 3, -1, 300, 129, "0"},  // tails
                                        std::tuple{2, 2, -1, 1000, 300, "b"}, // early exit
                                        std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

TEST_P(QuantScaleBiasLse, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(def_is_v_rowmajor, "e");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        true,  // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "e",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,                                 // do_validation
        sink == sink_kind::gptoss ? 1 : 0, // init_sink_value
        1,                                 // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, def_is_v_rowmajor);
    ExpectFp8Sched(kernel_name, def_is_v_rowmajor);
}

// MX batch instances without LSE, mask or sink; every other MX sweep adds one of them.
class QuantScaleMxBase
    : public TestWithParam<std::tuple<bool, bool, std::tuple<int, int, int, int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleMxBase);
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMxBase,
    Combine(EnableTestIf(ck_tile::is_any_of<DataTypeConfig, FmhaFwdMxFp8, FmhaFwdMxFp8Bf16>::value),
            LseVRowmajorValues,
            Values(std::tuple{1, 2, -1, 128, 128},   // one tile
                   std::tuple{2, 4, 2, 256, 256},    // GQA, tile multiple
                   std::tuple{2, 3, 1, 300, 1000},   // query/key tails
                   std::tuple{1, 2, -1, 1000, 161},  // seqlen_q > seqlen_k
                   std::tuple{2, 2, -1, 17, 333}))); // short query

TEST_P(QuantScaleMxBase, DataTypeConfig)
{
    auto [_, is_v_rowmajor, dims]                    = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        false, // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask_str
        "mx",
        true, // is_rotary_interleaved
        1,    // num_splits
        "3",  // init_method
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Batch mode with per-batch query/key lengths: workgroups past a batch's seqlen_q exit early.
using EarlyExitDims =
    std::tuple<int, int, int, int, int, std::vector<int>, std::vector<int>, std::string, bool>;

class QuantScaleBatchEarlyExit
    : public TestWithParam<std::tuple<std::tuple<int, int>, bool, EarlyExitDims>>
{
};

// Every head dim of the scheduled FP8 pipeline, as (hdim_q, hdim_v).
static const auto kEarlyExitHdims =
    kIsMxFp8Config || kIsFp8Config
        ? std::vector<std::tuple<int, int>>{{64, 64}, {128, 128}, {192, 128}, {256, 256}}
        : std::vector<std::tuple<int, int>>{};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBatchEarlyExit);
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleBatchEarlyExit,
    Combine(ValuesIn(kEarlyExitHdims),
            Values(false, true), // lse
            // batch, nhead, nhead_k, seqlen_q, seqlen_k, q_eff_lens, kv_eff_lens, mask, bhsd
            Values(EarlyExitDims{3, 4, -1, 300, 400, {300, 64, 1}, {400, 100, 7}, "0", true},
                   EarlyExitDims{3, 8, 2, 300, 300, {129, 0, 257}, {300, 50, 300}, "b", false},
                   EarlyExitDims{2, 4, 1, 600, 512, {513, 17}, {}, "t:128,30", true},
                   // Merge-eligible decode shapes stay unmerged once cu_seqlen_q or _k is set.
                   EarlyExitDims{2, 8, 2, 1, 1024, {1, 1}, {}, "0", true},
                   EarlyExitDims{2, 16, 2, 8, 700, {}, {700, 9}, "b", true})));

TEST_P(QuantScaleBatchEarlyExit, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "FP8 qr_tdm_sched is gfx125x only";
    auto [hdims, lse, dims]                                                     = GetParam();
    auto [hdim, hdim_v]                                                         = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, q_eff, kv_eff, mask, perm] = dims;
    const char* qscale = kIsMxFp8Config ? "mx" : "bs";

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {seqlen_q},
        {seqlen_k},
        hdim,
        hdim_v,
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        q_eff,
        kv_eff,
        0,    // rotary_dim
        perm, // i_perm
        perm, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        true, // is_v_rowmajor
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        "3",  // init_method
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, true);
    EXPECT_EQ(DecodeKernelOf(kernel_name), "sched") << kernel_name;
}

// Direct fmha_fwd() calls with arguments the runner never produces.
constexpr const char* DirectDataType()
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8>)
        return "fp8";
    else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8>)
        return "mxfp8";
    else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8Bf16>)
        return "mxfp8bf16";
    else
        return "fp8bf16";
}

struct DirectCase
{
    int batch               = 2;
    int nhead               = 2;
    int nhead_k             = 1;
    int seqlen_q            = 200;
    int seqlen_k            = 700;
    int hdim_q              = 128;
    int hdim_v              = 128;
    std::string mask        = "0";
    quant_scale_enum qscale = quant_scale_enum::blockscale;
    int bss_q               = 128;
    int bss_kv              = 32; // keys per K/V descale array element
    int bss_kv_value        = 32; // keys per distinct K/V descale value
    int max_seqlen_q        = -1;
    int lse_head_pad        = 0;
    std::vector<float> sink; // one logit per query head; empty for none
};

struct DirectResult
{
    float rc = -1;
    std::string kernel;
    std::vector<float> o;
    std::vector<float> lse;
};

// Batch mode, b*h*s*d layout, row-major V, LSE stored; inputs depend only on the shape.
DirectResult RunDirect(const DirectCase& c)
{
    using QDataType = typename FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
    using ODataType = typename FmhaFwdTypeConfig<DataTypeConfig>::ODataType;
    const int b = c.batch, h = c.nhead, hk = c.nhead_k, sq = c.seqlen_q, sk = c.seqlen_k;
    const int dq = c.hdim_q, dv = c.hdim_v, lse_stride = sq + c.lse_head_pad;
    const bool bs   = c.qscale == quant_scale_enum::blockscale;
    const int nbq   = bs ? ck_tile::integer_divide_ceil(sq, c.bss_q > 0 ? c.bss_q : 128) : 1;
    const int nbk   = bs ? ck_tile::integer_divide_ceil(sk, c.bss_kv > 0 ? c.bss_kv : 32) : 1;
    const int nbval = bs ? ck_tile::integer_divide_ceil(sk, c.bss_kv_value) : 1;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> value(-2.f, 2.f), scale(0.5f, 1.5f);
    const auto fill = [&](int n) {
        std::vector<QDataType> t(n);
        for(auto& x : t)
            x = ck_tile::type_convert<QDataType>(value(gen));
        return t;
    };
    const auto q = fill(b * h * sq * dq), k = fill(b * hk * sk * dq), v = fill(b * hk * sk * dv);
    std::vector<float> q_ds(b * h * nbq), k_val(b * hk * nbval), v_val(b * hk * nbval);
    for(auto& x : q_ds)
        x = scale(gen);
    for(auto& x : k_val)
        x = scale(gen);
    // The V descale must be a power of two.
    for(auto& x : v_val)
        x = std::ldexp(1.f, static_cast<int>(gen() % 3) - 1);
    std::vector<float> k_ds(b * hk * nbk), v_ds(b * hk * nbk);
    for(int bh = 0; bh < b * hk; ++bh)
        for(int i = 0; i < nbk; ++i)
        {
            const int j        = bs ? i * c.bss_kv / c.bss_kv_value : 0;
            k_ds[bh * nbk + i] = k_val[bh * nbval + j];
            v_ds[bh * nbk + i] = v_val[bh * nbval + j];
        }

    const auto upload = [](const auto& host) {
        auto dev = std::make_unique<ck_tile::DeviceMem>(
            std::max<std::size_t>(1, host.size() * sizeof(host[0])));
        dev->ToDevice(host.data());
        return dev;
    };
    const auto q_buf = upload(q), k_buf = upload(k), v_buf = upload(v);
    const auto qd_buf = upload(q_ds), kd_buf = upload(k_ds), vd_buf = upload(v_ds);
    const auto sink_buf = upload(c.sink.empty() ? std::vector<float>{0.f} : c.sink);
    ck_tile::DeviceMem o_buf(b * h * sq * dv * sizeof(ODataType));
    ck_tile::DeviceMem lse_buf(b * h * lse_stride * sizeof(float));
    o_buf.SetZero();
    lse_buf.SetZero();

    const mask_info mask = mask_info::decode(c.mask, sq, sk);
    DirectResult r;
    fmha_fwd_args a{};
    a.selected_kernel_name   = &r.kernel;
    a.q_ptr                  = q_buf->GetDeviceBuffer();
    a.k_ptr                  = k_buf->GetDeviceBuffer();
    a.v_ptr                  = v_buf->GetDeviceBuffer();
    a.q_descale_ptr          = qd_buf->GetDeviceBuffer();
    a.k_descale_ptr          = kd_buf->GetDeviceBuffer();
    a.v_descale_ptr          = vd_buf->GetDeviceBuffer();
    a.o_ptr                  = o_buf.GetDeviceBuffer();
    a.lse_ptr                = lse_buf.GetDeviceBuffer();
    a.sink_ptr               = c.sink.empty() ? nullptr : sink_buf->GetDeviceBuffer();
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = b;
    a.max_seqlen_q           = c.max_seqlen_q < 0 ? sq : c.max_seqlen_q;
    a.hdim_q                 = dq;
    a.hdim_v                 = dv;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(dq));
    a.stride_q               = dq;
    a.stride_k               = dq;
    a.stride_v               = dv;
    a.stride_o               = dv;
    a.nhead_stride_q         = sq * dq;
    a.nhead_stride_k         = sk * dq;
    a.nhead_stride_v         = sk * dv;
    a.nhead_stride_o         = sq * dv;
    a.nhead_stride_lse       = lse_stride;
    a.nhead_stride_q_descale = nbq;
    a.nhead_stride_k_descale = nbk;
    a.nhead_stride_v_descale = nbk;
    a.batch_stride_q         = h * sq * dq;
    a.batch_stride_k         = hk * sk * dq;
    a.batch_stride_v         = hk * sk * dv;
    a.batch_stride_o         = h * sq * dv;
    a.batch_stride_lse       = h * lse_stride;
    a.batch_stride_q_descale = h * nbq;
    a.batch_stride_k_descale = hk * nbk;
    a.batch_stride_v_descale = hk * nbk;
    a.window_size_left       = mask.left;
    a.window_size_right      = mask.right;
    a.sink_size              = mask.sink;
    a.mask_type              = static_cast<ck_tile::index_t>(mask.type);
    a.block_scale_size_q     = c.bss_q;
    a.block_scale_size_kv    = c.bss_kv;

    const fmha_fwd_traits traits{dq,
                                 dv,
                                 DirectDataType(),
                                 false, // batch mode
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask.type,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 c.qscale,
                                 false, // no skip_min_seqlen_q
                                 mask.sink > 0 || !c.sink.empty()};
    r.rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(r.rc < 0)
        return r;
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<ODataType> o(b * h * sq * dv);
    o_buf.FromDevice(o.data());
    r.o.resize(o.size());
    std::transform(o.begin(), o.end(), r.o.begin(), [](ODataType x) {
        return ck_tile::type_convert<float>(x);
    });
    r.lse.resize(b * h * lse_stride);
    lse_buf.FromDevice(r.lse.data());
    return r;
}

bool SameBits(const std::vector<float>& x, const std::vector<float>& y)
{
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

static const auto kDirectHdims =
    kIsFp8Config ? kEarlyExitHdims : std::vector<std::tuple<int, int>>{};

// K/V descale blocks of 64-256 keys must match the same values spelled out per 32 keys.
class QuantScaleBlockScaleKv
    : public TestWithParam<std::tuple<std::tuple<int, int>, int, std::string>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBlockScaleKv);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBlockScaleKv,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(64, 128, 256), // block_scale_size_kv
                                 // window starts mid-block for 256; sink tokens jump to it
                                 Values("0", "b:128,30", "b:128,30,200")));

TEST_P(QuantScaleBlockScaleKv, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "FP8 qr_tdm_sched is gfx125x only";
    auto [hdims, bss_kv, mask] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v) = hdims;
    c.mask                       = mask;
    c.bss_kv = c.bss_kv_value = bss_kv;
    const auto wide           = RunDirect(c);
    c.bss_kv                  = 32;
    const auto narrow         = RunDirect(c);
    ASSERT_GE(wide.rc, 0) << wide.kernel;
    ASSERT_GE(narrow.rc, 0) << narrow.kernel;
    ExpectFp8Sched(wide.kernel, true);
    EXPECT_EQ(wide.kernel, narrow.kernel);
    EXPECT_TRUE(std::any_of(wide.o.begin(), wide.o.end(), [](float x) { return x != 0.f; }));
    EXPECT_TRUE(SameBits(wide.o, narrow.o));
    EXPECT_TRUE(SameBits(wide.lse, narrow.lse));
}

// qr_tdm_sched rejects scale blocks that split its M tile or a 32-key group; zero ones fail.
class QuantScaleBlockScaleReject : public TestWithParam<std::tuple<bool, std::tuple<int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBlockScaleReject);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBlockScaleReject,
                         Combine(EnableTestIf(kIsFp8Config),
                                 Values(std::tuple{128, 0},
                                        std::tuple{128, 16},
                                        std::tuple{128, 48},
                                        std::tuple{64, 128},
                                        std::tuple{0, 128})));

TEST_P(QuantScaleBlockScaleReject, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "FP8 qr_tdm_sched is gfx125x only";
    DirectCase c;
    std::tie(c.bss_q, c.bss_kv) = std::get<1>(GetParam());
    const auto r                = RunDirect(c);
    // qr_tdm, when built, takes the M64 Q scale blocks that qr_tdm_sched leaves.
    if(c.bss_q > 0 && c.bss_kv > 0 && r.rc >= 0)
        EXPECT_EQ(DecodeKernelOf(r.kernel), "other") << r.kernel;
    else
        EXPECT_LT(r.rc, 0) << r.kernel;
}

// A larger max_seqlen_q or a padded LSE head stride keeps decode heads unmerged, same results.
class QuantScaleMergeStride
    : public TestWithParam<std::tuple<std::tuple<int, int>, std::tuple<int, int, int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleMergeStride);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleMergeStride,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(std::tuple{2, 8, 2, 2},     // 8 rows, one-wavefront
                                        std::tuple{1, 16, 2, 8}))); // 64 rows, merged-GQA

TEST_P(QuantScaleMergeStride, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "FP8 qr_tdm_sched is gfx125x only";
    auto [hdims, dims] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v)                      = hdims;
    std::tie(c.batch, c.nhead, c.nhead_k, c.seqlen_q) = dims;
    c.seqlen_k                                        = 300;
    c.qscale                                          = quant_scale_enum::pertensor;
    const auto merged                                 = RunDirect(c);
    ASSERT_GE(merged.rc, 0) << merged.kernel;
    EXPECT_EQ(DecodeKernelOf(merged.kernel),
              ExpectedDecodeKernel(c.nhead, c.nhead_k, c.seqlen_q, true));
    for(int pad : {0, 1})
    {
        DirectCase u     = c;
        u.max_seqlen_q   = pad ? -1 : c.seqlen_q + 1;
        u.lse_head_pad   = pad;
        const auto plain = RunDirect(u);
        ASSERT_GE(plain.rc, 0) << plain.kernel;
        EXPECT_EQ(DecodeKernelOf(plain.kernel), "sched") << plain.kernel;
        EXPECT_TRUE(SameBits(plain.o, merged.o)) << plain.kernel;
        for(int bh = 0; bh < c.batch * c.nhead; ++bh)
            for(int i = 0; i < c.seqlen_q; ++i)
                EXPECT_EQ(plain.lse[bh * (c.seqlen_q + pad) + i], merged.lse[bh * c.seqlen_q + i]);
    }
}

// An infinite sink logit takes the whole row (O = 0, LSE = +inf); a NaN one poisons it.
class QuantScaleSinkNonFinite
    : public TestWithParam<std::tuple<std::tuple<int, int>, std::tuple<int, int, int, int>, bool>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleSinkNonFinite);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleSinkNonFinite,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(std::tuple{2, 2, 1, 200}, // tiled
                                        std::tuple{2, 8, 2, 1},   // one-wavefront
                                        std::tuple{1, 16, 2, 8}), // merged-GQA
                                 Bool()));                        // NaN

TEST_P(QuantScaleSinkNonFinite, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "FP8 qr_tdm_sched is gfx125x only";
    auto [hdims, dims, nan] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v)                      = hdims;
    std::tie(c.batch, c.nhead, c.nhead_k, c.seqlen_q) = dims;
    c.seqlen_k                                        = 300;
    c.qscale                                          = quant_scale_enum::pertensor;
    const float bad =
        nan ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    // Odd query heads keep a finite logit.
    for(int i = 0; i < c.nhead; ++i)
        c.sink.push_back(i % 2 ? 0.5f : bad);
    const auto r = RunDirect(c);
    ASSERT_GE(r.rc, 0) << r.kernel;
    ExpectFp8Sched(r.kernel, true);
    const int dv = c.hdim_v, sq = c.seqlen_q;
    for(int bh = 0; bh < c.batch * c.nhead; ++bh)
    {
        const bool finite = (bh % c.nhead) % 2;
        for(int i = 0; i < sq; ++i)
        {
            const float lse = r.lse[bh * sq + i];
            EXPECT_TRUE(finite ? std::isfinite(lse) : nan ? std::isnan(lse) : lse == bad) << lse;
            for(int j = 0; j < dv; ++j)
            {
                const float o = r.o[(bh * sq + i) * dv + j];
                EXPECT_TRUE(finite ? std::isfinite(o) : nan ? std::isnan(o) : o == 0.f) << o;
            }
        }
    }
}

// MX dispatch on zero data: unfoldable Q scale strides and group mode without V scale starts.
float RunMxProbe(bool group, int q_scale_head_pad, bool v_scale_seqstart, std::string& kernel)
{
    const int h = 8, hk = 2, sq = 2, sk = 256, d = 128, ds = d / 32, sv = sk / 32;
    constexpr std::size_t kSlot = 1 << 20;
    ck_tile::DeviceMem arena(8 * kSlot);
    arena.SetZero();
    const auto slot = [&](int i) {
        return static_cast<char*>(arena.GetDeviceBuffer()) + i * kSlot;
    };
    const std::vector<int32_t> seqstarts{0, sq, 0, sk, 0, sv};
    HIP_CHECK_ERROR(hipMemcpy(
        slot(7), seqstarts.data(), seqstarts.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    const auto* starts = reinterpret_cast<const int32_t*>(slot(7));

    fmha_fwd_args a{};
    a.selected_kernel_name   = &kernel;
    a.q_ptr                  = slot(0);
    a.k_ptr                  = slot(1);
    a.v_ptr                  = slot(2);
    a.q_descale_ptr          = slot(3);
    a.k_descale_ptr          = slot(4);
    a.v_descale_ptr          = slot(5);
    a.o_ptr                  = slot(6);
    a.lse_ptr                = slot(6) + kSlot / 2;
    a.seqstart_q_ptr         = group ? starts : nullptr;
    a.seqstart_k_ptr         = group ? starts + 2 : nullptr;
    a.seqstart_v_scale_ptr   = group && v_scale_seqstart ? starts + 4 : nullptr;
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = 1;
    a.max_seqlen_q           = sq;
    a.hdim_q                 = d;
    a.hdim_v                 = d;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(d));
    a.stride_q               = d;
    a.stride_k               = d;
    a.stride_v               = d;
    a.stride_o               = d;
    a.nhead_stride_q         = sq * d;
    a.nhead_stride_k         = sk * d;
    a.nhead_stride_v         = sk * d;
    a.nhead_stride_o         = sq * d;
    a.nhead_stride_lse       = sq;
    a.batch_stride_q         = h * sq * d;
    a.batch_stride_k         = hk * sk * d;
    a.batch_stride_v         = hk * sk * d;
    a.batch_stride_o         = h * sq * d;
    a.batch_stride_lse       = h * sq;
    a.stride_q_descale       = ds;
    a.stride_k_descale       = ds;
    a.stride_v_descale       = sv;
    a.nhead_stride_q_descale = sq * ds + q_scale_head_pad;
    a.nhead_stride_k_descale = sk * ds;
    a.nhead_stride_v_descale = d * sv;
    a.batch_stride_q_descale = h * a.nhead_stride_q_descale;
    a.batch_stride_k_descale = hk * sk * ds;
    a.batch_stride_v_descale = hk * d * sv;
    a.window_size_left       = -1;
    a.window_size_right      = -1;

    const fmha_fwd_traits traits{d,
                                 d,
                                 DirectDataType(),
                                 group,
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask_enum::no_mask,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 quant_scale_enum::mx};
    const float rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(rc >= 0)
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    return rc;
}

TEST(TestCkTileFmhaFwd, QuantScaleMxDispatch)
{
    if(!kIsMxFp8Config || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125x MX only";
    std::string kernel;
    EXPECT_GE(RunMxProbe(false, 0, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "packgqa_32row") << kernel;
    EXPECT_GE(RunMxProbe(false, 4, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
    EXPECT_GE(RunMxProbe(true, 0, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
    kernel.clear();
    EXPECT_LT(RunMxProbe(true, 0, false, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
}

// MX scales on random FP8 data, dispatched under the given data type string.
DirectResult RunMxAlias(const std::string& data_type, bool group, int sq)
{
    using ODataType = typename FmhaFwdTypeConfig<DataTypeConfig>::ODataType;
    const int h = 8, hk = 2, sk = 256, d = 128, ds = d / 32, sv = sk / 32;
    std::mt19937 gen(321);
    std::uniform_real_distribution<float> value(-2.f, 2.f);
    std::vector<ck_tile::fp8_t> q(h * sq * d), k(hk * sk * d), v(hk * sk * d);
    for(auto* t : {&q, &k, &v})
        for(auto& x : *t)
            x = ck_tile::type_convert<ck_tile::fp8_t>(value(gen));
    // E8M0 exponents of 0.5, 1 and 2.
    std::vector<uint8_t> qs(h * sq * ds), ks(hk * sk * ds), vs(hk * d * sv);
    for(auto* t : {&qs, &ks, &vs})
        for(auto& x : *t)
            x = static_cast<uint8_t>(126 + gen() % 3);
    const std::vector<int32_t> seqstarts{0, sq, 0, sk, 0, sv};
    const auto upload = [](const auto& host) {
        auto dev = std::make_unique<ck_tile::DeviceMem>(host.size() * sizeof(host[0]));
        dev->ToDevice(host.data());
        return dev;
    };
    const auto q_buf = upload(q), k_buf = upload(k), v_buf = upload(v);
    const auto qs_buf = upload(qs), ks_buf = upload(ks), vs_buf = upload(vs);
    const auto starts_buf = upload(seqstarts);
    const auto* starts    = static_cast<const int32_t*>(starts_buf->GetDeviceBuffer());
    ck_tile::DeviceMem o_buf(h * sq * d * sizeof(ODataType)), lse_buf(h * sq * sizeof(float));
    o_buf.SetZero();

    DirectResult r;
    fmha_fwd_args a{};
    a.selected_kernel_name   = &r.kernel;
    a.q_ptr                  = q_buf->GetDeviceBuffer();
    a.k_ptr                  = k_buf->GetDeviceBuffer();
    a.v_ptr                  = v_buf->GetDeviceBuffer();
    a.q_descale_ptr          = qs_buf->GetDeviceBuffer();
    a.k_descale_ptr          = ks_buf->GetDeviceBuffer();
    a.v_descale_ptr          = vs_buf->GetDeviceBuffer();
    a.o_ptr                  = o_buf.GetDeviceBuffer();
    a.lse_ptr                = lse_buf.GetDeviceBuffer();
    a.seqstart_q_ptr         = group ? starts : nullptr;
    a.seqstart_k_ptr         = group ? starts + 2 : nullptr;
    a.seqstart_v_scale_ptr   = group ? starts + 4 : nullptr;
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = 1;
    a.max_seqlen_q           = sq;
    a.hdim_q                 = d;
    a.hdim_v                 = d;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(d));
    a.stride_q               = d;
    a.stride_k               = d;
    a.stride_v               = d;
    a.stride_o               = d;
    a.nhead_stride_q         = sq * d;
    a.nhead_stride_k         = sk * d;
    a.nhead_stride_v         = sk * d;
    a.nhead_stride_o         = sq * d;
    a.nhead_stride_lse       = sq;
    a.batch_stride_q         = h * sq * d;
    a.batch_stride_k         = hk * sk * d;
    a.batch_stride_v         = hk * sk * d;
    a.batch_stride_o         = h * sq * d;
    a.batch_stride_lse       = h * sq;
    a.stride_q_descale       = ds;
    a.stride_k_descale       = ds;
    a.stride_v_descale       = sv;
    a.nhead_stride_q_descale = sq * ds;
    a.nhead_stride_k_descale = sk * ds;
    a.nhead_stride_v_descale = d * sv;
    a.batch_stride_q_descale = h * sq * ds;
    a.batch_stride_k_descale = hk * sk * ds;
    a.batch_stride_v_descale = hk * d * sv;
    a.window_size_left       = -1;
    a.window_size_right      = -1;

    const fmha_fwd_traits traits{d,
                                 d,
                                 data_type,
                                 group,
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask_enum::no_mask,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 quant_scale_enum::mx};
    r.rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(r.rc < 0)
        return r;
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<ODataType> o(h * sq * d);
    o_buf.FromDevice(o.data());
    r.o.resize(o.size());
    std::transform(o.begin(), o.end(), r.o.begin(), [](ODataType x) {
        return ck_tile::type_convert<float>(x);
    });
    r.lse.resize(h * sq);
    lse_buf.FromDevice(r.lse.data());
    return r;
}

// fp8/fp8bf16 with MX scales run the mxfp8/mxfp8bf16 kernel and give its exact output.
TEST(TestCkTileFmhaFwd, QuantScaleMxFromFp8)
{
    if(!kIsFp8Config || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125x FP8 only";
    const std::string dtype = DirectDataType();
    for(auto [group, sq] : {std::pair{false, 1}, std::pair{false, 200}, std::pair{true, 200}})
    {
        const auto alias = RunMxAlias(dtype, group, sq);
        const auto mx    = RunMxAlias("mx" + dtype, group, sq);
        ASSERT_GE(alias.rc, 0) << alias.kernel;
        ASSERT_GE(mx.rc, 0) << mx.kernel;
        EXPECT_EQ(alias.kernel, mx.kernel);
        EXPECT_NE(alias.kernel.find("_mx" + dtype + "_"), std::string::npos) << alias.kernel;
        EXPECT_NE(alias.kernel.find("qr_tdm_sched"), std::string::npos) << alias.kernel;
        EXPECT_TRUE(std::any_of(alias.o.begin(), alias.o.end(), [](float x) { return x != 0.f; }));
        EXPECT_TRUE(SameBits(alias.o, mx.o));
        EXPECT_TRUE(SameBits(alias.lse, mx.lse));
    }
}

// Round-to-nearest P drops small probabilities on long rows; the runner's gain check catches it.
TEST(TestCkTileFmhaFwd, QuantScaleLongRowNoGain)
{
    if(!kIsFp8Config || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125x FP8 qr_tdm_sched only";
    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        1, // batch
        1, // nhead
        1, // nhead_k
        {8192},
        {32768},
        128,
        128,
        0,     // seqlen_knew
        {-1},  // seqlen_qpads
        {-1},  // seqlen_kpads
        {},    // q_eff_lens_per_batch
        {},    // kv_eff_lens_per_batch
        0,     // rotary_dim
        true,  // i_perm
        true,  // o_perm
        0,     // scale_s
        0,     // logits_soft_cap
        true,  // is_v_rowmajor
        false, // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask
        "pt",
        true, // is_rotary_interleaved
        1,    // num_splits
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        0, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, true);
    EXPECT_EQ(DecodeKernelOf(kernel_name), "sched") << kernel_name;
}
#endif
