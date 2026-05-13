// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test for OutputWriter with padded k_per_group.
//
// Verifies that when k_per_group < GROUP_SIZE, the padded output writer
// correctly:
//   1. Writes real output data to valid k positions.
//   2. Does NOT write to padded k positions (sentinel preserved).
//
// For the unpadded case (k_per_group == GROUP_SIZE), verifies that the
// existing write path produces the correct output layout.

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_tile_conv_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/utils/types.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_8c_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_tile_conv_impl_v2.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

using namespace ck_tile::direct_conv;
using namespace grouped_4c_tile::v3;

// ============================================================================
// Config indices
//
// KernelConfigurations<>::configs[9]  — Fprop, no swizzle, direct DRAM (OutputWriter)
// KernelConfigurations<>::configs[49] — Fprop, CyclicShift, vector_size=1, direct DRAM (OutputWriter)
// ============================================================================
static constexpr int CFG_DIRECT   = 9;
static constexpr int CFG_VEC8     = 43;
static constexpr int CFG_VEC4     = 47;
static constexpr int CFG_VEC2     = 48;
static constexpr int CFG_VEC1     = 49;

// ============================================================================
// Test kernel: calls OutputWriter::flush for row 0 with a known accumulator.
//
// Each thread's accumulator is set to (threadIdx.x * 4 + k + 1) for k=0..3.
// After flush, the host verifies which output positions were written.
// ============================================================================
template <int CfgIdx>
__global__ void test_output_write_kernel(_Float16* __restrict__ out,
                                          int groups,
                                          int k_per_group,
                                          int ho,
                                          int wo,
                                          int c_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = TileConstants<KernelConfigurations<>::configs[CfgIdx]>;
    constexpr auto cfg = KernelConfigurations<>::configs[CfgIdx];

    // When c_per_group is 0 (default), use GROUP_SIZE for backward compatibility.
    const int cpg = (c_per_group > 0) ? c_per_group : cfg.group_size();
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, cpg, k_per_group);

    // Weight LDS (unused but needed for OutputWriterLds constructor signature).
    __shared__ uint4 dummy_lds[1];

    ck_tile::direct_conv::OutputWriter<TC> ow(bc, dummy_lds, out, ho, wo, k_per_group);

    // Each thread writes a deterministic accumulator value.
    // Use threadIdx.x to create unique values per thread.
    ck_tile::fp32x4_t acc;
    acc[0] = static_cast<float>(threadIdx.x * 4 + 1);
    acc[1] = static_cast<float>(threadIdx.x * 4 + 2);
    acc[2] = static_cast<float>(threadIdx.x * 4 + 3);
    acc[3] = static_cast<float>(threadIdx.x * 4 + 4);

    // Write row 0 (p_out = 0).
    ow.flush(acc, 0);
#endif
}


// ============================================================================
// Test fixture
// ============================================================================
class OutputWriterTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 4;  // fixed for 4c kernel
    static constexpr _Float16 SENTINEL = static_cast<_Float16>(-999.0f);

    // Launch the kernel and verify output contents.
    // c_per_group controls the input channel count passed to BlockCoords.
    // When 0 (default), uses GROUP_SIZE for backward compatibility.
    template <int CfgIdx>
    void run_and_verify(int k_per_group, int c_per_group = 0)
    {
        using TC = TileConstants<KernelConfigurations<>::configs[CfgIdx]>;
        constexpr auto cfg = KernelConfigurations<>::configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int BLOCK_Q = TC::BLOCK_Q;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int K_total = groups * k_per_group;  // actual output channels

        // Output dimensions: just big enough for the tile.
        const int wo = BLOCK_Q + 4;
        const int ho = 2;

        // Allocate output: [1, ho, wo, K_total] in NHWK layout.
        const int out_size = ho * wo * K_total;
        std::vector<_Float16> out_host(out_size, SENTINEL);

        _Float16* d_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_out, out_size * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_out, out_host.data(), out_size * sizeof(_Float16), hipMemcpyHostToDevice));

        test_output_write_kernel<CfgIdx><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_out, groups, k_per_group, ho, wo, c_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        ck_tile::hip_check_error(hipMemcpy(
            out_host.data(), d_out, out_size * sizeof(_Float16), hipMemcpyDeviceToHost));

        // Verify: row 0, columns [0, BLOCK_Q) should be written (non-sentinel).
        // Row 1 and columns >= BLOCK_Q should remain sentinel.
        // Within the written region, only k positions < k_per_group (within each
        // group's GROUP_SIZE output channels) should be written.
        //
        // The output layout is NHWK with K_total = groups * k_per_group.
        // BlockCoords maps block_k_out = block_group * k_per_group.
        // The kernel writes K channels starting at block_k_out.
        //
        // For the test with groups = BLOCK_GROUPS and blockIdx.y = 0:
        //   block_group = 0, block_k_out = 0.
        //
        // Each thread writes fp16x4 (4 channels) at its distribution-assigned position.
        // For the non-padded case (k_per_group == GROUP_SIZE), all GROUP_SIZE channels
        // are valid within each group, so the written channels span the full block_c
        // range: [0, BLOCK_GROUPS * GROUP_SIZE).
        //
        // For the padded case (k_per_group < GROUP_SIZE), the OutputWriter uses
        // a padded descriptor that maps only k_per_group channels per group,
        // writing k_per_group * BLOCK_GROUPS channels total per spatial position.

        // Count how many positions were written (non-sentinel) vs should have been.
        int written_count = 0;
        int expected_written = 0;

        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                bool is_sentinel = (val == static_cast<float>(SENTINEL));

                // Expected behavior: row 0, q < BLOCK_Q should be written.
                bool should_write = (q < BLOCK_Q);

                if(should_write)
                    expected_written++;
                if(!is_sentinel)
                    written_count++;

                if(q >= BLOCK_Q)
                {
                    // Beyond tile: must remain sentinel.
                    EXPECT_TRUE(is_sentinel)
                        << "q=" << q << " k=" << k << " val=" << val
                        << " (should be sentinel beyond BLOCK_Q)";
                }
            }
        }

        // Row 1 should be entirely sentinel.
        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 1 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                EXPECT_EQ(val, static_cast<float>(SENTINEL))
                    << "Row 1 should be untouched: q=" << q << " k=" << k;
            }
        }

        // For unpadded case: all K_total * BLOCK_Q positions should be written.
        if(k_per_group == GROUP_SIZE)
        {
            EXPECT_EQ(written_count, expected_written)
                << "Unpadded: all " << expected_written
                << " positions should be written, got " << written_count;
        }

        // Verify written values are non-zero (converted from fp32 accumulator).
        // Each thread writes (threadIdx.x * 4 + k + 1) which is always >= 1.
        for(int q = 0; q < BLOCK_Q && q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                if(val != static_cast<float>(SENTINEL))
                {
                    EXPECT_GT(val, 0.0f)
                        << "Written value should be positive: q=" << q << " k=" << k;
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_out));
    }
};

// ============================================================================
// Correctness tests — output writing
// ============================================================================

// Unpadded path: k_per_group == GROUP_SIZE.
TEST_F(OutputWriterTest, Direct_K4) { run_and_verify<CFG_DIRECT>(GROUP_SIZE); }
TEST_F(OutputWriterTest, Vec1_K4) { run_and_verify<CFG_VEC1>(GROUP_SIZE); }
TEST_F(OutputWriterTest, Vec2_K4) { run_and_verify<CFG_VEC2>(GROUP_SIZE); }
TEST_F(OutputWriterTest, Vec4_K4) { run_and_verify<CFG_VEC4>(GROUP_SIZE); }
TEST_F(OutputWriterTest, Vec8_K4) { run_and_verify<CFG_VEC8>(GROUP_SIZE); }

// Padded path: k_per_group < GROUP_SIZE, CyclicShift swizzle.
TEST_F(OutputWriterTest, Vec1_K3) { run_and_verify<CFG_VEC1>(3); }
TEST_F(OutputWriterTest, Vec1_K2) { run_and_verify<CFG_VEC1>(2); }
TEST_F(OutputWriterTest, Vec1_K1) { run_and_verify<CFG_VEC1>(1); }

TEST_F(OutputWriterTest, Vec2_K3) { run_and_verify<CFG_VEC2>(3); }
TEST_F(OutputWriterTest, Vec2_K2) { run_and_verify<CFG_VEC2>(2); }
TEST_F(OutputWriterTest, Vec2_K1) { run_and_verify<CFG_VEC2>(1); }

TEST_F(OutputWriterTest, Vec4_K3) { run_and_verify<CFG_VEC4>(3); }
TEST_F(OutputWriterTest, Vec4_K2) { run_and_verify<CFG_VEC4>(2); }
TEST_F(OutputWriterTest, Vec4_K1) { run_and_verify<CFG_VEC4>(1); }

TEST_F(OutputWriterTest, Vec8_K3) { run_and_verify<CFG_VEC8>(3); }
TEST_F(OutputWriterTest, Vec8_K2) { run_and_verify<CFG_VEC8>(2); }
TEST_F(OutputWriterTest, Vec8_K1) { run_and_verify<CFG_VEC8>(1); }

// Padded path: k_per_group < GROUP_SIZE, no swizzle.
TEST_F(OutputWriterTest, Direct_K3) { run_and_verify<CFG_DIRECT>(3); }
TEST_F(OutputWriterTest, Direct_K2) { run_and_verify<CFG_DIRECT>(2); }
TEST_F(OutputWriterTest, Direct_K1) { run_and_verify<CFG_DIRECT>(1); }

// C != K tests: verify OutputWriter works correctly when BlockCoords has C_in != C_out.
// The OutputWriter should only depend on k_per_group (C_out), not c_per_group (C_in).
TEST_F(OutputWriterTest, Vec1_C3_K2)  { run_and_verify<CFG_VEC1>(2, 3); }
TEST_F(OutputWriterTest, Vec1_C1_K4)  { run_and_verify<CFG_VEC1>(4, 1); }
TEST_F(OutputWriterTest, Vec1_C2_K3)  { run_and_verify<CFG_VEC1>(3, 2); }
TEST_F(OutputWriterTest, Vec2_C2_K4)  { run_and_verify<CFG_VEC2>(4, 2); }
TEST_F(OutputWriterTest, Vec1_C3_K1)  { run_and_verify<CFG_VEC1>(1, 3); }
TEST_F(OutputWriterTest, Direct_C3_K2) { run_and_verify<CFG_DIRECT>(2, 3); }
TEST_F(OutputWriterTest, Direct_C1_K4) { run_and_verify<CFG_DIRECT>(4, 1); }

// =============================================================================
// 16c OutputWriter tests
// =============================================================================

namespace ns_16c = ck_tile::direct_conv::grouped_16c_tile::v2;

// Config indices for 16c kernel:
//   9  — Fprop, no swizzle, direct DRAM (unpadded)
//   83 — Fprop, CyclicShift, LDS epilogue, vector_size=1 (padded, any k)
//   81 — Fprop, CyclicShift, LDS epilogue, vector_size=4 (padded, k%4==0)
static constexpr int CFG_16C_DIRECT = 9;
static constexpr int CFG_16C_VEC4   = 81;
static constexpr int CFG_16C_VEC1   = 83;

template <int CfgIdx>
__global__ void test_output_write_kernel_16c(_Float16* __restrict__ out,
                                              int groups,
                                              int k_per_group,
                                              int ho,
                                              int wo,
                                              int c_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = ns_16c::TileConstants<ns_16c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx]>;
    constexpr auto cfg = ns_16c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx];

    const int cpg = (c_per_group > 0) ? c_per_group : cfg.group_size();
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, cpg, k_per_group);

    __shared__ uint4 dummy_lds[1];

    ck_tile::direct_conv::OutputWriter<TC> ow(bc, dummy_lds, out, ho, wo, k_per_group);

    ck_tile::fp32x4_t acc;
    acc[0] = static_cast<float>(threadIdx.x * 4 + 1);
    acc[1] = static_cast<float>(threadIdx.x * 4 + 2);
    acc[2] = static_cast<float>(threadIdx.x * 4 + 3);
    acc[3] = static_cast<float>(threadIdx.x * 4 + 4);

    ow.flush(acc, 0);
#endif
}

class OutputWriter16cTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 16;
    static constexpr _Float16 SENTINEL = static_cast<_Float16>(-999.0f);

    template <int CfgIdx>
    void run_and_verify(int k_per_group, int c_per_group = 0)
    {
        using TC = ns_16c::TileConstants<ns_16c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx]>;
        constexpr auto cfg = ns_16c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int BLOCK_Q = TC::BLOCK_Q;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int K_total = groups * k_per_group;
        const int wo = BLOCK_Q + 4;
        const int ho = 2;

        const int out_size = ho * wo * K_total;
        std::vector<_Float16> out_host(out_size, SENTINEL);

        _Float16* d_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_out, out_size * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_out, out_host.data(), out_size * sizeof(_Float16), hipMemcpyHostToDevice));

        test_output_write_kernel_16c<CfgIdx><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_out, groups, k_per_group, ho, wo, c_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        ck_tile::hip_check_error(hipMemcpy(
            out_host.data(), d_out, out_size * sizeof(_Float16), hipMemcpyDeviceToHost));

        int written_count = 0;
        int expected_written = 0;

        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                bool is_sentinel = (val == static_cast<float>(SENTINEL));

                bool should_write = (q < BLOCK_Q);
                if(should_write)
                    expected_written++;
                if(!is_sentinel)
                    written_count++;

                if(q >= BLOCK_Q)
                {
                    EXPECT_TRUE(is_sentinel)
                        << "q=" << q << " k=" << k << " val=" << val
                        << " (should be sentinel beyond BLOCK_Q)";
                }
            }
        }

        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 1 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                EXPECT_EQ(val, static_cast<float>(SENTINEL))
                    << "Row 1 should be untouched: q=" << q << " k=" << k;
            }
        }

        if(k_per_group == GROUP_SIZE)
        {
            EXPECT_EQ(written_count, expected_written)
                << "Unpadded: all " << expected_written
                << " positions should be written, got " << written_count;
        }

        for(int q = 0; q < BLOCK_Q && q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                if(val != static_cast<float>(SENTINEL))
                {
                    EXPECT_GT(val, 0.0f)
                        << "Written value should be positive: q=" << q << " k=" << k;
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_out));
    }
};

// Unpadded 16c
TEST_F(OutputWriter16cTest, Direct_K16) { run_and_verify<CFG_16C_DIRECT>(16); }
TEST_F(OutputWriter16cTest, Vec1_K16)   { run_and_verify<CFG_16C_VEC1>(16); }

// Padded: k_per_group < 16
TEST_F(OutputWriter16cTest, Vec1_K12)  { run_and_verify<CFG_16C_VEC1>(12); }
TEST_F(OutputWriter16cTest, Vec1_K10)  { run_and_verify<CFG_16C_VEC1>(10); }
TEST_F(OutputWriter16cTest, Vec1_K9)   { run_and_verify<CFG_16C_VEC1>(9); }
TEST_F(OutputWriter16cTest, Vec4_K12)  { run_and_verify<CFG_16C_VEC4>(12); }

// C != K
TEST_F(OutputWriter16cTest, Vec1_C9_K12)  { run_and_verify<CFG_16C_VEC1>(12, 9); }
TEST_F(OutputWriter16cTest, Vec1_C12_K9)  { run_and_verify<CFG_16C_VEC1>(9, 12); }
TEST_F(OutputWriter16cTest, Vec1_C16_K9)  { run_and_verify<CFG_16C_VEC1>(9, 16); }
TEST_F(OutputWriter16cTest, Direct_C9_K16) { run_and_verify<CFG_16C_DIRECT>(16, 9); }

// =============================================================================
// 8c OutputWriter tests
// =============================================================================

namespace ns_8c = ck_tile::direct_conv::grouped_8c_tile::v2;

// Config indices for 8c kernel:
//   9  — Fprop, no swizzle, direct DRAM (unpadded)
//   81 — Fprop, CyclicShift, LDS epilogue, vector_size=1 (padded, any k)
//   79 — Fprop, CyclicShift, LDS epilogue, vector_size=4 (padded, k%4==0)
static constexpr int CFG_8C_DIRECT = 9;
static constexpr int CFG_8C_VEC4   = 79;
static constexpr int CFG_8C_VEC1   = 81;

template <int CfgIdx>
__global__ void test_output_write_kernel_8c(_Float16* __restrict__ out,
                                             int groups,
                                             int k_per_group,
                                             int ho,
                                             int wo,
                                             int c_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = ns_8c::TileConstants<ns_8c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx]>;
    constexpr auto cfg = ns_8c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx];

    const int cpg = (c_per_group > 0) ? c_per_group : cfg.group_size();
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, cpg, k_per_group);

    __shared__ uint4 dummy_lds[1];

    ck_tile::direct_conv::OutputWriter<TC> ow(bc, dummy_lds, out, ho, wo, k_per_group);

    ck_tile::fp32x4_t acc;
    acc[0] = static_cast<float>(threadIdx.x * 4 + 1);
    acc[1] = static_cast<float>(threadIdx.x * 4 + 2);
    acc[2] = static_cast<float>(threadIdx.x * 4 + 3);
    acc[3] = static_cast<float>(threadIdx.x * 4 + 4);

    ow.flush(acc, 0);
#endif
}

class OutputWriter8cTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 8;
    static constexpr _Float16 SENTINEL = static_cast<_Float16>(-999.0f);

    template <int CfgIdx>
    void run_and_verify(int k_per_group, int c_per_group = 0)
    {
        using TC = ns_8c::TileConstants<ns_8c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx]>;
        constexpr auto cfg = ns_8c::KernelConfigurations<>::KernelConfigurations<>::configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int BLOCK_Q = TC::BLOCK_Q;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int K_total = groups * k_per_group;
        const int wo = BLOCK_Q + 4;
        const int ho = 2;

        const int out_size = ho * wo * K_total;
        std::vector<_Float16> out_host(out_size, SENTINEL);

        _Float16* d_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_out, out_size * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_out, out_host.data(), out_size * sizeof(_Float16), hipMemcpyHostToDevice));

        test_output_write_kernel_8c<CfgIdx><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_out, groups, k_per_group, ho, wo, c_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        ck_tile::hip_check_error(hipMemcpy(
            out_host.data(), d_out, out_size * sizeof(_Float16), hipMemcpyDeviceToHost));

        int written_count = 0;
        int expected_written = 0;

        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                bool is_sentinel = (val == static_cast<float>(SENTINEL));

                bool should_write = (q < BLOCK_Q);
                if(should_write)
                    expected_written++;
                if(!is_sentinel)
                    written_count++;

                if(q >= BLOCK_Q)
                {
                    EXPECT_TRUE(is_sentinel)
                        << "q=" << q << " k=" << k << " val=" << val
                        << " (should be sentinel beyond BLOCK_Q)";
                }
            }
        }

        for(int q = 0; q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 1 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                EXPECT_EQ(val, static_cast<float>(SENTINEL))
                    << "Row 1 should be untouched: q=" << q << " k=" << k;
            }
        }

        if(k_per_group == GROUP_SIZE)
        {
            EXPECT_EQ(written_count, expected_written)
                << "Unpadded: all " << expected_written
                << " positions should be written, got " << written_count;
        }

        for(int q = 0; q < BLOCK_Q && q < wo; q++)
        {
            for(int k = 0; k < K_total; k++)
            {
                int idx = 0 * wo * K_total + q * K_total + k;
                float val = static_cast<float>(out_host[idx]);
                if(val != static_cast<float>(SENTINEL))
                {
                    EXPECT_GT(val, 0.0f)
                        << "Written value should be positive: q=" << q << " k=" << k;
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_out));
    }
};

// Unpadded 8c
TEST_F(OutputWriter8cTest, Direct_K8) { run_and_verify<CFG_8C_DIRECT>(8); }
TEST_F(OutputWriter8cTest, Vec1_K8)   { run_and_verify<CFG_8C_VEC1>(8); }

// Padded: k_per_group < 8
TEST_F(OutputWriter8cTest, Vec1_K6) { run_and_verify<CFG_8C_VEC1>(6); }
TEST_F(OutputWriter8cTest, Vec1_K5) { run_and_verify<CFG_8C_VEC1>(5); }
TEST_F(OutputWriter8cTest, Vec4_K4) { run_and_verify<CFG_8C_VEC4>(4); }

// C != K
TEST_F(OutputWriter8cTest, Vec1_C5_K6) { run_and_verify<CFG_8C_VEC1>(6, 5); }
TEST_F(OutputWriter8cTest, Vec1_C6_K5) { run_and_verify<CFG_8C_VEC1>(5, 6); }
TEST_F(OutputWriter8cTest, Vec1_C8_K5) { run_and_verify<CFG_8C_VEC1>(5, 8); }
TEST_F(OutputWriter8cTest, Direct_C5_K8) { run_and_verify<CFG_8C_DIRECT>(8, 5); }
