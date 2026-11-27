// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaFwdFp16 // or FmhaFwdBf16 / FmhaFwdFp8Bf16 / FmhaFwdFp32
#endif

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

template <typename T>
struct TestConfigs
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, 128},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, -1},
        std::tuple{128, -1},
        std::tuple{256, -1},
    };
    static constexpr auto AppendKVHDimValues = std::array{
        std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto ModeValues        = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues = std::array{true};
    static constexpr auto qscale_str        = "n";
    static constexpr auto QScaleValues      = std::array{"n"};
    static constexpr bool def_lse           = true;
    static constexpr bool def_is_v_rowmajor = true;
    static constexpr auto init_method       = "uf";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 8);
    }
};

template <>
struct TestConfigs<FmhaFwdFp8Bf16>
{
    static constexpr auto HDimValues =
        std::array{std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto AppendKVHDimValues = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "pt";
    static constexpr auto QScaleValues       = std::array{"n", "pt", "ph", "bs"};
    static constexpr bool def_lse            = false;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "3";
    // When there are no fp8 instances with padding, pad seqlen to avoid skipping most of the tests:
    // return ck_tile::integer_least_multiple(seqlen, 128);
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

// d=128 is the only head dim the whole fp8 family shares; fp8 has no d=64 tile. No
// splitkv/appendkv instances are generated, and all four quantization scales exist here.
struct Fp8FamilyTestConfigs
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "pt";
    static constexpr auto QScaleValues       = std::array{"n", "pt", "ph", "bs"};
    static constexpr bool def_lse            = false;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

template <>
struct TestConfigs<FmhaFwdFp8> : Fp8FamilyTestConfigs
{
};

template <>
struct TestConfigs<FmhaFwdMxFp8>
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{false};
    static constexpr auto qscale_str         = "mx";
    static constexpr auto QScaleValues       = std::array{"mx"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = false;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

// MXFP8 with a BF16 O exists only on gfx125x qr_tdm_sched; d64/d256 run in the Hdim suites.
template <>
struct TestConfigs<FmhaFwdMxFp8Bf16> : TestConfigs<FmhaFwdMxFp8>
{
    static constexpr auto HDimValues        = std::array{std::tuple{128, -1}};
    static constexpr auto IsVRowmajorValues = std::array{true};
    static constexpr bool def_is_v_rowmajor = true;
};

template <>
struct TestConfigs<FmhaFwdMxFp4>
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{false};
    static constexpr auto qscale_str         = "mx";
    static constexpr auto QScaleValues       = std::array{"mx"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = false;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen)
    {
        return seqlen < 0 ? seqlen : ck_tile::integer_least_multiple(seqlen, 2);
    }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 32);
    }
};

template <>
struct TestConfigs<FmhaFwdFp32>
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{48, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "n";
    static constexpr auto QScaleValues       = std::array{"n"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "uf";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim) { return hdim; }
};

static auto HDimValues         = ValuesIn(TestConfigs<DataTypeConfig>::HDimValues);
static auto SplitKVHDimValues  = ValuesIn(TestConfigs<DataTypeConfig>::SplitKVHDimValues);
static auto AppendKVHDimValues = ValuesIn(TestConfigs<DataTypeConfig>::AppendKVHDimValues);
static auto ModeValues         = ValuesIn(TestConfigs<DataTypeConfig>::ModeValues);
static auto IsVRowmajorValues  = ValuesIn(TestConfigs<DataTypeConfig>::IsVRowmajorValues);
#ifdef CK_TILE_TEST_FMHA_QSCALE_SWEEP
static auto QScaleValues = ValuesIn(TestConfigs<DataTypeConfig>::QScaleValues);
#endif
constexpr static auto qscale_str = TestConfigs<DataTypeConfig>::qscale_str;
constexpr bool def_lse           = TestConfigs<DataTypeConfig>::def_lse;
constexpr bool def_is_v_rowmajor = TestConfigs<DataTypeConfig>::def_is_v_rowmajor;
constexpr auto init_method       = TestConfigs<DataTypeConfig>::init_method;
inline int adjust_seqlen(int seqlen) { return TestConfigs<DataTypeConfig>::adjust_seqlen(seqlen); }
inline int adjust_hdim(int hdim) { return TestConfigs<DataTypeConfig>::adjust_hdim(hdim); }

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

// For the qr_tdm split-kv suites: gfx1250 fp16/bf16 has an instance for every
// case they run, so no_instance there means codegen dropped one and must fail.
#define CHECK_SPLITKV_RESULT(result)                                              \
    do                                                                            \
    {                                                                             \
        if(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value && \
           ck_tile::is_gfx125_supported())                                        \
            ASSERT_EQ(result, fwd_result::success);                               \
        else                                                                      \
            CHECK_RESULT(result);                                                 \
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

#define COMMON_ARGS_INIT(init)                                                               \
    init, static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, 0, 1, \
        stream_config

#define COMMON_ARGS COMMON_ARGS_INIT(init_method)

inline auto EnableTestIf(bool condition)
{
    return ValuesIn(condition ? std::vector<bool>{true} : std::vector<bool>{});
}
