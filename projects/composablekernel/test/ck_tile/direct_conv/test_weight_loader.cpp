// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test for weight_load_to_lds with padded descriptors.
//
// Verifies that when c_per_group < GROUP_SIZE or k_per_group < GROUP_SIZE,
// the padded weight loader correctly:
//   1. Loads real weight data into the correct LDS positions.
//   2. Zero-pads the extra C and K channels.
//
// The test launches a minimal GPU kernel that calls weight_load_to_lds,
// then reads back LDS via a device-to-host copy of the LDS buffer.

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_tile_conv_impl_v3.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>

using namespace ck_tile::direct_conv;

// ============================================================================
// Test kernel: calls weight_load_to_lds and copies LDS content to global memory
// ============================================================================

// Use configs[9] from the 4c kernel as the test configuration.
// configs[9] is Fprop with waves_c64=1, waves_q4=1 (smallest block size = 64 threads).
static constexpr auto test_cfg = grouped_4c_tile::v3::configs[9];
using TestTC = grouped_4c_tile::v3::TileConstants<test_cfg>;

static constexpr int WEIGHT_LDS_SIZE_UINT4 = TestTC::Weight::WEIGHT_LDS_SIZE_UINT4;
static constexpr int GROUP_SIZE = TestTC::GROUP_SIZE; // = 4
static constexpr int BLOCK_GROUPS = test_cfg.block_groups();
static constexpr int KH = test_cfg.kh;
static constexpr int KW = test_cfg.kw;
static constexpr int BLOCK_SIZE = test_cfg.block_size();
static constexpr int LDS_FP16_ELEMS = WEIGHT_LDS_SIZE_UINT4 * 8;

// Kernel: load weights to LDS, then copy LDS to global for host verification.
__global__ void test_weight_load_kernel(const _Float16* __restrict__ wei,
                                        _Float16* __restrict__ lds_out,
                                        int groups,
                                        int c_per_group,
                                        int k_per_group)
{
    grouped_4c_tile::v3::BlockCoords<test_cfg> bc(groups);

    __shared__ uint4 lds_buf[WEIGHT_LDS_SIZE_UINT4];

    // Initialize LDS with sentinel values to detect unwritten positions.
    for(int i = threadIdx.x; i < WEIGHT_LDS_SIZE_UINT4; i += BLOCK_SIZE)
    {
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    }
    __syncthreads();

    weight_load_to_lds<TestTC, test_cfg>(bc, lds_buf, wei, c_per_group, k_per_group);

    __syncthreads();

    // Copy LDS to global for host verification.
    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_FP16_ELEMS; i += BLOCK_SIZE)
    {
        lds_out[i] = lds_fp16[i];
    }
}

// ============================================================================
// Test fixture
// ============================================================================

class WeightLoaderTest : public ::testing::Test
{
protected:
    // Build a GKYXC weight tensor with distinct non-zero values.
    static std::vector<_Float16> make_weight_tensor(int groups,
                                                    int k_per_group,
                                                    int c_per_group)
    {
        const int total = groups * k_per_group * KH * KW * c_per_group;
        std::vector<_Float16> wei(total);
        for(int i = 0; i < total; i++)
        {
            // Values 1-7, never 0, so we can distinguish data from zero-padding.
            wei[i] = static_cast<_Float16>(static_cast<float>((i % 7) + 1));
        }
        return wei;
    }

    // Read element at [g, k, y, x, c] from the host weight tensor.
    static float read_weight(const std::vector<_Float16>& wei,
                             int k_per_group, int c_per_group,
                             int g, int k, int y, int x, int c)
    {
        const int filter_size = KH * KW;
        int idx = g * (k_per_group * filter_size * c_per_group)
                + k * (filter_size * c_per_group)
                + (y * KW + x) * c_per_group
                + c;
        return static_cast<float>(wei[idx]);
    }

    // Launch kernel, read back LDS, and verify the padded layout.
    void run_and_verify(int c_per_group, int k_per_group)
    {
        const int groups = BLOCK_GROUPS;
        auto wei_host = make_weight_tensor(groups, k_per_group, c_per_group);

        _Float16* d_wei = nullptr;
        _Float16* d_lds_out = nullptr;
        const size_t wei_bytes = wei_host.size() * sizeof(_Float16);
        const size_t lds_out_bytes = LDS_FP16_ELEMS * sizeof(_Float16);

        ck_tile::hip_check_error(hipMalloc(&d_wei, wei_bytes));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, lds_out_bytes));
        ck_tile::hip_check_error(
            hipMemcpy(d_wei, wei_host.data(), wei_bytes, hipMemcpyHostToDevice));

        test_weight_load_kernel<<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_wei, d_lds_out, groups, c_per_group, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16_ELEMS);
        ck_tile::hip_check_error(
            hipMemcpy(lds_host.data(), d_lds_out, lds_out_bytes, hipMemcpyDeviceToHost));

        // Verify LDS layout: [g, k, yx, c] with GROUP_SIZE-padded K and C.
        // Real positions: LDS[g,k,yx,c] = DRAM weight[g,k,y,x,c]
        // Padded positions: LDS[g,k,yx,c] = 0.0
        const int kh_kw = KH * KW;
        for(int g = 0; g < BLOCK_GROUPS; g++)
        {
            for(int k = 0; k < GROUP_SIZE; k++)
            {
                for(int yx = 0; yx < kh_kw; yx++)
                {
                    for(int c = 0; c < GROUP_SIZE; c++)
                    {
                        int lds_idx = g * GROUP_SIZE * kh_kw * GROUP_SIZE
                                    + k * kh_kw * GROUP_SIZE
                                    + yx * GROUP_SIZE
                                    + c;

                        float actual = static_cast<float>(lds_host[lds_idx]);
                        float expected;

                        if(k < k_per_group && c < c_per_group)
                        {
                            expected = read_weight(wei_host, k_per_group, c_per_group,
                                                   g, k, yx / KW, yx % KW, c);
                        }
                        else
                        {
                            expected = 0.0f;
                        }

                        EXPECT_EQ(actual, expected)
                            << "Mismatch at g=" << g << " k=" << k
                            << " yx=" << yx << " c=" << c
                            << " lds_idx=" << lds_idx;
                    }
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_wei));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// ============================================================================
// Tests
// ============================================================================

// Baseline: c_per_group == GROUP_SIZE, k_per_group == GROUP_SIZE (no padding).
TEST_F(WeightLoaderTest, NoPadding_C4_K4)
{
    const int c_per_group = GROUP_SIZE; // 4
    const int k_per_group = GROUP_SIZE; // 4
    const int groups = BLOCK_GROUPS;

    auto wei_host = make_weight_tensor(groups, k_per_group, c_per_group);

    _Float16* d_wei = nullptr;
    _Float16* d_lds_out = nullptr;
    const size_t wei_bytes = wei_host.size() * sizeof(_Float16);
    const size_t lds_out_bytes = LDS_FP16_ELEMS * sizeof(_Float16);

    ck_tile::hip_check_error(hipMalloc(&d_wei, wei_bytes));
    ck_tile::hip_check_error(hipMalloc(&d_lds_out, lds_out_bytes));
    ck_tile::hip_check_error(
        hipMemcpy(d_wei, wei_host.data(), wei_bytes, hipMemcpyHostToDevice));

    test_weight_load_kernel<<<dim3(1, 1, 1), BLOCK_SIZE>>>(
        d_wei, d_lds_out, groups, c_per_group, k_per_group);
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<_Float16> lds_host(LDS_FP16_ELEMS);
    ck_tile::hip_check_error(
        hipMemcpy(lds_host.data(), d_lds_out, lds_out_bytes, hipMemcpyDeviceToHost));

    // Unpadded path: flat contiguous copy from DRAM to LDS.
    const int total_weight_elems = BLOCK_GROUPS * GROUP_SIZE * KH * KW * GROUP_SIZE;
    ASSERT_EQ(total_weight_elems, LDS_FP16_ELEMS)
        << "Weight element count must match LDS capacity for the no-padding case";

    for(int i = 0; i < total_weight_elems; i++)
    {
        float expected = static_cast<float>(wei_host[i]);
        float actual = static_cast<float>(lds_host[i]);
        EXPECT_EQ(actual, expected) << "Mismatch at LDS element " << i;
    }

    ck_tile::hip_check_error(hipFree(d_wei));
    ck_tile::hip_check_error(hipFree(d_lds_out));
}

// Padded: c_per_group=3, k_per_group=4 (C padding only).
TEST_F(WeightLoaderTest, PaddedC3_K4) { run_and_verify(3, GROUP_SIZE); }

// Padded: c_per_group=2, k_per_group=3 (both C and K padding).
TEST_F(WeightLoaderTest, PaddedC2_K3) { run_and_verify(2, 3); }

// Padded: c_per_group=2, k_per_group=2 (both C and K padding).
TEST_F(WeightLoaderTest, PaddedC2_K2) { run_and_verify(2, 2); }

// Padded: c_per_group=1, k_per_group=1 (depthwise conv - maximum padding).
TEST_F(WeightLoaderTest, PaddedC1_K1) { run_and_verify(1, 1); }
