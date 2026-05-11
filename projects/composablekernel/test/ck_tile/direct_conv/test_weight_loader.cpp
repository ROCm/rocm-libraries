// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test for weight_load_to_lds with padded descriptors.
//
// Verifies that when c_per_group < GROUP_SIZE or k_per_group < GROUP_SIZE,
// the padded weight loader correctly:
//   1. Loads real weight data into the correct LDS positions.
//   2. Zero-pads the extra C and K channels.
//
// Also verifies that is_valid_config correctly gates configs by vector_size
// versus c_per_group divisibility.

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_tile_conv_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_fp16_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_fp16_tile_conv_impl_v2.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

using namespace ck_tile::direct_conv;
using namespace grouped_4c_tile::v3;

// ============================================================================
// Config indices used by the tests
//
// configs[9]  — Fprop, vector_size=8 (default, full GROUP_SIZE)  → unpadded
// configs[46] — Fprop, CyclicShift, vector_size=2               → c%2==0
// configs[47] — Fprop, CyclicShift, vector_size=1               → any c
// ============================================================================
static constexpr int CFG_UNPADDED = 9;
static constexpr int CFG_VEC2    = 46;
static constexpr int CFG_VEC1    = 47;

// ============================================================================
// Test kernel: templated on config index.
// Calls weight_load_to_lds and copies LDS content to global memory.
// ============================================================================
template <int CfgIdx, bool Padding>
__global__ void test_weight_load_kernel(const _Float16* __restrict__ wei,
                                        _Float16* __restrict__ lds_out,
                                        int groups,
                                        int c_per_group,
                                        int k_per_group)
{
    using TC = TileConstants<configs[CfgIdx]>;
    constexpr auto cfg = configs[CfgIdx];

    ck_tile::direct_conv::BlockCoords<cfg> bc(groups);

    constexpr int LDS_UINT4 = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
    constexpr int BLOCK_SIZE = cfg.block_size();

    __shared__ uint4 lds_buf[LDS_UINT4];

    for(int i = threadIdx.x; i < LDS_UINT4; i += BLOCK_SIZE)
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    __syncthreads();

    weight_load_to_lds<TC, cfg, Padding>(bc, lds_buf, wei, c_per_group, k_per_group);
    ck_tile::direct_conv::wait_vmcnt<0>();
    __syncthreads();

    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_UINT4 * 8; i += BLOCK_SIZE)
        lds_out[i] = lds_fp16[i];
}

// ============================================================================
// Test fixture
// ============================================================================
class WeightLoaderTest : public ::testing::Test
{
protected:
    static constexpr int KH = configs[CFG_VEC1].kh;
    static constexpr int KW = configs[CFG_VEC1].kw;
    static constexpr int GROUP_SIZE = 4; // fixed for 4c kernel

    // All padded configs share the same block_groups (waves_c64=2 → 32 groups).
    static constexpr int BLOCK_GROUPS = configs[CFG_VEC1].block_groups();

    static std::vector<_Float16> make_weight_tensor(int groups,
                                                    int k_per_group,
                                                    int c_per_group)
    {
        const int total = groups * k_per_group * KH * KW * c_per_group;
        std::vector<_Float16> wei(total);
        for(int i = 0; i < total; i++)
            wei[i] = static_cast<_Float16>(static_cast<float>((i % 7) + 1));
        return wei;
    }

    static float read_weight(const std::vector<_Float16>& wei,
                             int k_per_group, int c_per_group,
                             int g, int k, int y, int x, int c)
    {
        int idx = g * (k_per_group * KH * KW * c_per_group)
                + k * (KH * KW * c_per_group)
                + (y * KW + x) * c_per_group
                + c;
        return static_cast<float>(wei[idx]);
    }

    // Launch the kernel for the given config index and verify LDS contents.
    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int k_per_group)
    {
        using TC = TileConstants<configs[CfgIdx]>;
        constexpr int LDS_FP16 = TC::Weight::WEIGHT_LDS_SIZE_UINT4 * 8;
        constexpr int BLOCK_SIZE = configs[CfgIdx].block_size();

        // Use the config-specific block_groups, not the fixture constant.
        // Config 9 (unpadded) has waves_c64=1 → block_groups()=16,
        // while configs 47-49 have waves_c64=2 → block_groups()=32.
        constexpr int groups = configs[CfgIdx].block_groups();
        auto wei_host = make_weight_tensor(groups, k_per_group, c_per_group);

        _Float16* d_wei     = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_wei,     wei_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16        * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_wei, wei_host.data(), wei_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_weight_load_kernel<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_wei, d_lds_out, groups, c_per_group, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        // Verify LDS layout: [g, k, yx, c] with GROUP_SIZE-padded K and C.
        const int kh_kw = KH * KW;
        for(int g = 0; g < groups; g++)
        for(int k = 0; k < GROUP_SIZE; k++)
        for(int yx = 0; yx < kh_kw; yx++)
        for(int c = 0; c < GROUP_SIZE; c++)
        {
            int lds_idx = g * GROUP_SIZE * kh_kw * GROUP_SIZE
                        + k * kh_kw * GROUP_SIZE
                        + yx * GROUP_SIZE
                        + c;

            float actual   = static_cast<float>(lds_host[lds_idx]);
            float expected = (k < k_per_group && c < c_per_group)
                ? read_weight(wei_host, k_per_group, c_per_group,
                              g, k, yx / KW, yx % KW, c)
                : 0.0f;

            EXPECT_EQ(actual, expected)
                << "cfg=" << CfgIdx
                << " g=" << g << " k=" << k << " yx=" << yx << " c=" << c;
        }

        ck_tile::hip_check_error(hipFree(d_wei));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// ============================================================================
// Correctness tests — weight loading
// ============================================================================

// Unpadded path (config 9, vector_size=8): c==GROUP_SIZE, k==GROUP_SIZE.
TEST_F(WeightLoaderTest, Unpadded_C4_K4)
{
    run_and_verify<CFG_UNPADDED, false>(GROUP_SIZE, GROUP_SIZE);
}

// C padding only (k == GROUP_SIZE).
TEST_F(WeightLoaderTest, Vec2_C2_K4) { run_and_verify<CFG_VEC2>(2, GROUP_SIZE); }
TEST_F(WeightLoaderTest, Vec1_C1_K4) { run_and_verify<CFG_VEC1>(1, GROUP_SIZE); }
TEST_F(WeightLoaderTest, Vec1_C3_K4) { run_and_verify<CFG_VEC1>(3, GROUP_SIZE); }

// Both C and K padded.
TEST_F(WeightLoaderTest, Vec2_C2_K3)      { run_and_verify<CFG_VEC2>(2, 3); }
TEST_F(WeightLoaderTest, Vec2_C2_K2)      { run_and_verify<CFG_VEC2>(2, 2); }
TEST_F(WeightLoaderTest, Vec1_C3_K3)      { run_and_verify<CFG_VEC1>(3, 3); }
TEST_F(WeightLoaderTest, Vec1_C1_K1)      { run_and_verify<CFG_VEC1>(1, 1); }

// C < K cases (input channels fewer than output channels).
TEST_F(WeightLoaderTest, Vec1_C2_K4)      { run_and_verify<CFG_VEC1>(2, 4); }
TEST_F(WeightLoaderTest, Vec1_C1_K3)      { run_and_verify<CFG_VEC1>(1, 3); }
TEST_F(WeightLoaderTest, Vec2_C2_K4_asym) { run_and_verify<CFG_VEC2>(2, 4); }
TEST_F(WeightLoaderTest, Vec1_C1_K2)      { run_and_verify<CFG_VEC1>(1, 2); }
TEST_F(WeightLoaderTest, Vec1_C3_K2)      { run_and_verify<CFG_VEC1>(3, 2); }

// ============================================================================
// Validity tests — is_valid_config gates configs by vector_size vs c_per_group
// ============================================================================

class ValidConfigTest : public ::testing::Test
{
protected:
    // Minimal Fprop params with the given c_per_group.
    static Conv2dParams make_params(int c_per_group, int k_per_group = 4)
    {
        // groups must be a multiple of block_groups() for any of the padded configs.
        const int groups = configs[CFG_VEC1].block_groups();
        // Spatial dims must be >= block_q() for the largest waves_q4 config.
        // configs 47-49 have waves_q4=8 → block_q()=32, so use q=w=32.
        Conv2dParams p;
        p.direction = Direction::Fprop;
        p.n         = 1;
        p.c_tot     = groups * c_per_group;
        p.k_tot     = groups * k_per_group;
        p.h         = 32;
        p.w         = 32;
        p.q         = 32;
        p.groups    = groups;
        p.kh        = configs[CFG_VEC1].kh;
        p.kw        = configs[CFG_VEC1].kw;
        return p;
    }
};

// c%4==0: vec4 valid, vec2 valid, vec1 valid; unpadded valid only if c==4.
TEST_F(ValidConfigTest, C4_valid_all)
{
    auto p = make_params(4);
    EXPECT_TRUE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

// c==3: not divisible by 2 or 4 → only vec1 valid.
TEST_F(ValidConfigTest, C3_only_vec1)
{
    auto p = make_params(3);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_FALSE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

// c==2: divisible by 2 → vec2 and vec1 valid, vec4 invalid.
TEST_F(ValidConfigTest, C2_vec2_and_vec1)
{
    auto p = make_params(2);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

// c==1: only vec1 valid.
TEST_F(ValidConfigTest, C1_only_vec1)
{
    auto p = make_params(1);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_FALSE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

// Wrong direction: all Fprop configs reject Dgrad params.
TEST_F(ValidConfigTest, WrongDirection_rejected)
{
    auto p = make_params(4);
    p.direction = Direction::Dgrad;
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
}

// groups not a multiple of block_groups(): all configs reject.
TEST_F(ValidConfigTest, NonAlignedGroups_rejected)
{
    auto p = make_params(4);
    p.groups = configs[CFG_VEC1].block_groups() + 1; // not a multiple
    p.c_tot  = p.groups * 4;
    p.k_tot  = p.groups * 4;
    EXPECT_FALSE(is_valid_config(p, configs[CFG_VEC1]));
}

// C != K: vector_size must divide both c_per_group and k_per_group.
TEST_F(ValidConfigTest, C3_K2_only_vec1)
{
    auto p = make_params(3, 2);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_FALSE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

TEST_F(ValidConfigTest, C2_K4_vec2_and_vec1)
{
    auto p = make_params(2, 4);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

TEST_F(ValidConfigTest, C1_K3_only_vec1)
{
    auto p = make_params(1, 3);
    EXPECT_FALSE(is_valid_config(p, configs[CFG_UNPADDED]));
    EXPECT_FALSE(is_valid_config(p, configs[CFG_VEC2]));
    EXPECT_TRUE(is_valid_config(p, configs[CFG_VEC1]));
}

// =============================================================================
// 16c WeightLoader tests
// =============================================================================

namespace ns_16c = ck_tile::direct_conv::grouped_16c_tile::v2;

// Config indices for 16c kernel:
//   9  — Fprop, no swizzle, vector_size=16 (unpadded)
//   81 — Fprop, CyclicShift, vector_size=4 (padded, c%4==0)
//   83 — Fprop, CyclicShift, vector_size=1 (padded, any c)
static constexpr int CFG_16C_UNPADDED = 9;
static constexpr int CFG_16C_VEC4     = 81;
static constexpr int CFG_16C_VEC1     = 83;

template <int CfgIdx, bool Padding>
__global__ void test_weight_load_kernel_16c(const _Float16* __restrict__ wei,
                                            _Float16* __restrict__ lds_out,
                                            int groups,
                                            int c_per_group,
                                            int k_per_group)
{
    using TC = ns_16c::TileConstants<ns_16c::configs[CfgIdx]>;
    constexpr auto cfg = ns_16c::configs[CfgIdx];

    ck_tile::direct_conv::BlockCoords<cfg> bc(groups);

    constexpr int LDS_UINT4 = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
    constexpr int BLOCK_SIZE = cfg.block_size();

    __shared__ uint4 lds_buf[LDS_UINT4];

    for(int i = threadIdx.x; i < LDS_UINT4; i += BLOCK_SIZE)
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    __syncthreads();

    weight_load_to_lds<TC, cfg, Padding>(bc, lds_buf, wei, c_per_group, k_per_group);
    ck_tile::direct_conv::wait_vmcnt<0>();
    __syncthreads();

    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_UINT4 * 8; i += BLOCK_SIZE)
        lds_out[i] = lds_fp16[i];
}

class WeightLoader16cTest : public ::testing::Test
{
protected:
    static constexpr int KH = ns_16c::configs[CFG_16C_VEC1].kh;
    static constexpr int KW = ns_16c::configs[CFG_16C_VEC1].kw;
    static constexpr int GROUP_SIZE = 16;

    static std::vector<_Float16> make_weight_tensor(int groups, int k_per_group, int c_per_group)
    {
        const int total = groups * k_per_group * KH * KW * c_per_group;
        std::vector<_Float16> wei(total);
        for(int i = 0; i < total; i++)
            wei[i] = static_cast<_Float16>(static_cast<float>((i % 7) + 1));
        return wei;
    }

    static float read_weight(const std::vector<_Float16>& wei,
                             int k_per_group, int c_per_group,
                             int g, int k, int y, int x, int c)
    {
        int idx = g * (k_per_group * KH * KW * c_per_group)
                + k * (KH * KW * c_per_group)
                + (y * KW + x) * c_per_group
                + c;
        return static_cast<float>(wei[idx]);
    }

    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int k_per_group)
    {
        using TC = ns_16c::TileConstants<ns_16c::configs[CfgIdx]>;
        constexpr int LDS_FP16 = TC::Weight::WEIGHT_LDS_SIZE_UINT4 * 8;
        constexpr int BLOCK_SIZE = ns_16c::configs[CfgIdx].block_size();
        constexpr int groups = ns_16c::configs[CfgIdx].block_groups();

        auto wei_host = make_weight_tensor(groups, k_per_group, c_per_group);

        _Float16* d_wei     = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_wei,     wei_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16        * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_wei, wei_host.data(), wei_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_weight_load_kernel_16c<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_wei, d_lds_out, groups, c_per_group, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        const int kh_kw = KH * KW;
        for(int g = 0; g < groups; g++)
        for(int k = 0; k < GROUP_SIZE; k++)
        for(int yx = 0; yx < kh_kw; yx++)
        for(int c = 0; c < GROUP_SIZE; c++)
        {
            int lds_idx = g * GROUP_SIZE * kh_kw * GROUP_SIZE
                        + k * kh_kw * GROUP_SIZE
                        + yx * GROUP_SIZE
                        + c;

            float actual   = static_cast<float>(lds_host[lds_idx]);
            float expected = (k < k_per_group && c < c_per_group)
                ? read_weight(wei_host, k_per_group, c_per_group,
                              g, k, yx / KW, yx % KW, c)
                : 0.0f;

            EXPECT_EQ(actual, expected)
                << "cfg=" << CfgIdx
                << " g=" << g << " k=" << k << " yx=" << yx << " c=" << c;
        }

        ck_tile::hip_check_error(hipFree(d_wei));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// Unpadded 16c
TEST_F(WeightLoader16cTest, Unpadded_C16_K16) { run_and_verify<CFG_16C_UNPADDED, false>(16, 16); }

// C padding only (k == GROUP_SIZE)
TEST_F(WeightLoader16cTest, Vec1_C12_K16) { run_and_verify<CFG_16C_VEC1>(12, 16); }
TEST_F(WeightLoader16cTest, Vec1_C10_K16) { run_and_verify<CFG_16C_VEC1>(10, 16); }
TEST_F(WeightLoader16cTest, Vec1_C9_K16)  { run_and_verify<CFG_16C_VEC1>(9, 16); }
TEST_F(WeightLoader16cTest, Vec4_C12_K16) { run_and_verify<CFG_16C_VEC4>(12, 16); }

// Both C and K padded
TEST_F(WeightLoader16cTest, Vec1_C12_K12) { run_and_verify<CFG_16C_VEC1>(12, 12); }
TEST_F(WeightLoader16cTest, Vec1_C10_K10) { run_and_verify<CFG_16C_VEC1>(10, 10); }
TEST_F(WeightLoader16cTest, Vec1_C9_K9)   { run_and_verify<CFG_16C_VEC1>(9, 9); }

// C != K
TEST_F(WeightLoader16cTest, Vec1_C9_K12)  { run_and_verify<CFG_16C_VEC1>(9, 12); }
TEST_F(WeightLoader16cTest, Vec1_C12_K9)  { run_and_verify<CFG_16C_VEC1>(12, 9); }
TEST_F(WeightLoader16cTest, Vec1_C16_K9)  { run_and_verify<CFG_16C_VEC1>(16, 9); }
TEST_F(WeightLoader16cTest, Vec1_C9_K16_asym) { run_and_verify<CFG_16C_VEC1>(9, 16); }

// =============================================================================
// 8c WeightLoader tests
// =============================================================================

namespace ns_8c = ck_tile::direct_conv::grouped_8c_tile::v2;

// Config indices for 8c kernel:
//   9  — Fprop, no swizzle, vector_size=8 (unpadded)
//   79 — Fprop, CyclicShift, vector_size=4 (padded, c%4==0)
//   81 — Fprop, CyclicShift, vector_size=1 (padded, any c)
static constexpr int CFG_8C_UNPADDED = 9;
static constexpr int CFG_8C_VEC4     = 79;
static constexpr int CFG_8C_VEC1     = 81;

template <int CfgIdx, bool Padded>
__global__ void test_weight_load_kernel_8c(const _Float16* __restrict__ wei,
                                           _Float16* __restrict__ lds_out,
                                           int groups,
                                           int c_per_group,
                                           int k_per_group)
{
    using TC = ns_8c::TileConstants<ns_8c::configs[CfgIdx]>;
    constexpr auto cfg = ns_8c::configs[CfgIdx];

    ck_tile::direct_conv::BlockCoords<cfg> bc(groups);

    constexpr int LDS_UINT4 = TC::Weight::WEIGHT_LDS_SIZE_UINT4;
    constexpr int BLOCK_SIZE = cfg.block_size();

    __shared__ uint4 lds_buf[LDS_UINT4];

    for(int i = threadIdx.x; i < LDS_UINT4; i += BLOCK_SIZE)
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    __syncthreads();

    weight_load_to_lds<TC, cfg, Padded>(bc, lds_buf, wei, c_per_group, k_per_group);
    ck_tile::direct_conv::wait_vmcnt<0>();
    __syncthreads();

    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_UINT4 * 8; i += BLOCK_SIZE)
        lds_out[i] = lds_fp16[i];
}

class WeightLoader8cTest : public ::testing::Test
{
protected:
    static constexpr int KH = ns_8c::configs[CFG_8C_VEC1].kh;
    static constexpr int KW = ns_8c::configs[CFG_8C_VEC1].kw;
    static constexpr int GROUP_SIZE = 8;

    static std::vector<_Float16> make_weight_tensor(int groups, int k_per_group, int c_per_group)
    {
        const int total = groups * k_per_group * KH * KW * c_per_group;
        std::vector<_Float16> wei(total);
        for(int i = 0; i < total; i++)
            wei[i] = static_cast<_Float16>(static_cast<float>((i % 7) + 1));
        return wei;
    }

    static float read_weight(const std::vector<_Float16>& wei,
                             int k_per_group, int c_per_group,
                             int g, int k, int y, int x, int c)
    {
        int idx = g * (k_per_group * KH * KW * c_per_group)
                + k * (KH * KW * c_per_group)
                + (y * KW + x) * c_per_group
                + c;
        return static_cast<float>(wei[idx]);
    }

    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int k_per_group)
    {
        using TC = ns_8c::TileConstants<ns_8c::configs[CfgIdx]>;
        constexpr int LDS_FP16 = TC::Weight::WEIGHT_LDS_SIZE_UINT4 * 8;
        constexpr int BLOCK_SIZE = ns_8c::configs[CfgIdx].block_size();
        constexpr int groups = ns_8c::configs[CfgIdx].block_groups();

        auto wei_host = make_weight_tensor(groups, k_per_group, c_per_group);

        _Float16* d_wei     = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_wei,     wei_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16        * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_wei, wei_host.data(), wei_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_weight_load_kernel_8c<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_wei, d_lds_out, groups, c_per_group, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        const int kh_kw = KH * KW;
        for(int g = 0; g < groups; g++)
        for(int k = 0; k < GROUP_SIZE; k++)
        for(int yx = 0; yx < kh_kw; yx++)
        for(int c = 0; c < GROUP_SIZE; c++)
        {
            int lds_idx = g * GROUP_SIZE * kh_kw * GROUP_SIZE
                        + k * kh_kw * GROUP_SIZE
                        + yx * GROUP_SIZE
                        + c;

            float actual   = static_cast<float>(lds_host[lds_idx]);
            float expected = (k < k_per_group && c < c_per_group)
                ? read_weight(wei_host, k_per_group, c_per_group,
                              g, k, yx / KW, yx % KW, c)
                : 0.0f;

            EXPECT_EQ(actual, expected)
                << "cfg=" << CfgIdx
                << " g=" << g << " k=" << k << " yx=" << yx << " c=" << c;
        }

        ck_tile::hip_check_error(hipFree(d_wei));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// Unpadded 8c
TEST_F(WeightLoader8cTest, Unpadded_C8_K8) { run_and_verify<CFG_8C_UNPADDED, false>(8, 8); }

// C padding only (k == GROUP_SIZE)
TEST_F(WeightLoader8cTest, Vec1_C6_K8) { run_and_verify<CFG_8C_VEC1>(6, 8); }
TEST_F(WeightLoader8cTest, Vec1_C5_K8) { run_and_verify<CFG_8C_VEC1>(5, 8); }
TEST_F(WeightLoader8cTest, Vec4_C4_K8) { run_and_verify<CFG_8C_VEC4>(4, 8); }

// Both C and K padded
TEST_F(WeightLoader8cTest, Vec1_C6_K6) { run_and_verify<CFG_8C_VEC1>(6, 6); }
TEST_F(WeightLoader8cTest, Vec1_C5_K5) { run_and_verify<CFG_8C_VEC1>(5, 5); }

// C != K
TEST_F(WeightLoader8cTest, Vec1_C5_K7) { run_and_verify<CFG_8C_VEC1>(5, 7); }
TEST_F(WeightLoader8cTest, Vec1_C6_K5) { run_and_verify<CFG_8C_VEC1>(6, 5); }
TEST_F(WeightLoader8cTest, Vec1_C8_K5) { run_and_verify<CFG_8C_VEC1>(8, 5); }
TEST_F(WeightLoader8cTest, Vec1_C5_K8_asym) { run_and_verify<CFG_8C_VEC1>(5, 8); }
