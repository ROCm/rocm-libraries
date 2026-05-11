// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit test for InputLoader with padded c_per_group.
//
// Verifies that when c_per_group < GROUP_SIZE, the padded input loader
// correctly:
//   1. Loads real input data into the correct LDS positions.
//   2. Zero-pads the extra C channels within each group.
//
// For the unpadded case (c_per_group == GROUP_SIZE), verifies that the
// existing async load path produces the correct LDS layout.

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
// configs[9]  — Fprop, vector_size=8, waves_c64=1 (unpadded baseline)
// configs[49] — Fprop, CyclicShift, vector_size=1, waves_c64=2 (any c)
// ============================================================================
static constexpr int CFG_UNPADDED = 9;
static constexpr int CFG_VEC2     = 48;
static constexpr int CFG_VEC1     = 49;

// ============================================================================
// Test kernel: templated on config index.
// Constructs InputLoader, loads one row into LDS, copies LDS to global.
// ============================================================================
template <int CfgIdx, bool Padding>
__global__ void test_input_load_kernel(const _Float16* __restrict__ in,
                                       _Float16* __restrict__ lds_out,
                                       int groups,
                                       int c_per_group,
                                       int hi,
                                       int wi,
                                       int px,
                                       int k_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = TileConstants<configs[CfgIdx]>;
    constexpr auto cfg = configs[CfgIdx];
    constexpr int BLOCK_SIZE = cfg.block_size();

    // When k_per_group is 0 (default), use c_per_group for both (backward compat).
    const int kpg = (k_per_group > 0) ? k_per_group : c_per_group;
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, c_per_group, kpg);

    constexpr int LDS_C8 = TC::INPUT_LDS_BUFFER_SIZE_C8;
    constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
    __shared__ uint4 lds_buf[LDS_C8];

    // Fill LDS with sentinel to detect uninitialized reads.
    for(int i = threadIdx.x; i < LDS_C8; i += BLOCK_SIZE)
    {
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    }
    __syncthreads();

    // Construct InputLoader and prefetch first row (row 0) into LDS buffer 0.
    ck_tile::direct_conv::InputLoader<TC, cfg, ck_tile::fp16x4_t, Padding> il(bc, lds_buf, in, hi, wi, px, 0, 1, 1, 1, 1, c_per_group);
    il.prefetch_tile_to_lds(0);

    __syncthreads();

    // Copy LDS contents to global memory for host verification.
    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_FP16; i += BLOCK_SIZE)
    {
        lds_out[i] = lds_fp16[i];
    }
#endif
}

// ============================================================================
// Test fixture
// ============================================================================
class InputLoaderTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 4;  // fixed for 4c kernel

    // Create input tensor in NHWC layout: [1, hi, wi, groups * c_per_group].
    static std::vector<_Float16> make_input_tensor(int hi, int wi,
                                                    int groups,
                                                    int c_per_group)
    {
        const int C_total = groups * c_per_group;
        const int total = hi * wi * C_total;
        std::vector<_Float16> inp(total);
        for(int i = 0; i < total; i++)
        {
            inp[i] = static_cast<_Float16>(static_cast<float>((i % 13) + 1));
        }
        return inp;
    }

    // Read a single element from the NHWC input tensor.
    static float read_input(const std::vector<_Float16>& inp,
                            int wi, int C_total,
                            int h, int w, int c)
    {
        int idx = h * wi * C_total + w * C_total + c;
        return static_cast<float>(inp[idx]);
    }

    // Launch the kernel for the given config index and verify LDS contents.
    // k_per_group controls the output channel count passed to BlockCoords.
    // When 0 (default), uses c_per_group for backward compatibility.
    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int px = 0, int k_per_group = 0)
    {
        using TC = TileConstants<configs[CfgIdx]>;
        constexpr auto cfg = configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
        constexpr int BLOCK_W = TC::BLOCK_W;
        constexpr int BLOCK_C8 = TC::BLOCK_C8;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int C_total = groups * c_per_group;

        // Choose input dimensions large enough for the tile.
        // BLOCK_W = BLOCK_Q + (kw - 1). wi must be >= BLOCK_W - px.
        const int wi = BLOCK_W + 4;  // a few extra columns
        const int hi = 4;            // only need a few rows (we load row 0)

        auto inp_host = make_input_tensor(hi, wi, groups, c_per_group);

        _Float16* d_in      = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_in,      inp_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16        * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_in, inp_host.data(), inp_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_input_load_kernel<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_in, d_lds_out, groups, c_per_group, hi, wi, px, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        // Verify LDS layout: [BLOCK_W, BLOCK_C8, 8].
        //
        // For non-swizzled configs (SwizzleType::None), the layout is direct:
        //   LDS[w, c8, c] holds input channel c8 * 8 + c at spatial position w.
        //
        // For CyclicShift configs, the LDS is stored in swizzled order to avoid
        // bank conflicts.  The DRAM descriptor applies the CyclicShift, so the
        // data at LDS position (w, c8, c) comes from DRAM channel c8_phys, where:
        //   c8_phys = (c8 + w) % BLOCK_C8     [for wi_padded spatial index w]
        // The MFMA read path undoes this with an inverse_cyclic_shift on the LDS
        // read descriptor, so the end-to-end computation is correct.  This test
        // must account for the same shift when computing the expected LDS value.
        //
        // For XOR configs the analogous relationship is:
        //   c8_phys = c8 ^ w  (but XOR configs are not tested here)
        //
        // w_actual = w - px is the spatial position in the real input tensor.
        constexpr auto swizzle = cfg.swizzle_type;
        for(int w = 0; w < BLOCK_W; w++)
        {
            for(int c8 = 0; c8 < BLOCK_C8; c8++)
            {
                for(int c = 0; c < 8; c++)
                {
                    int lds_idx = w * BLOCK_C8 * 8 + c8 * 8 + c;
                    float actual = static_cast<float>(lds_host[lds_idx]);

                    // Compute the physical DRAM c8 that was loaded into this LDS slot.
                    int c8_phys;
                    if constexpr(swizzle == SwizzleType::CyclicShift)
                        c8_phys = (c8 + w) % BLOCK_C8;
                    else if constexpr(swizzle == SwizzleType::XOR)
                        c8_phys = c8 ^ w;
                    else
                        c8_phys = c8;

                    int global_ch    = c8_phys * 8 + c;
                    int group_idx    = global_ch / GROUP_SIZE;
                    int group_local_c = global_ch % GROUP_SIZE;
                    int w_actual     = w - px;

                    float expected;
                    if(group_local_c < c_per_group && w_actual >= 0 && w_actual < wi)
                    {
                        int flat_c = group_idx * c_per_group + group_local_c;
                        expected = read_input(inp_host, wi, C_total, 0, w_actual, flat_c);
                    }
                    else
                    {
                        expected = 0.0f;
                    }

                    EXPECT_EQ(actual, expected)
                        << "cfg=" << CfgIdx
                        << " w=" << w << " c8=" << c8 << " c=" << c
                        << " c8_phys=" << c8_phys
                        << " group=" << group_idx << " gc=" << group_local_c
                        << " w_actual=" << w_actual;
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_in));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// ============================================================================
// Correctness tests — input loading
// ============================================================================

// Unpadded path: c_per_group == GROUP_SIZE.
TEST_F(InputLoaderTest, Unpadded_C4) { run_and_verify<CFG_UNPADDED, false>(GROUP_SIZE); }

// Padded path: c_per_group < GROUP_SIZE.
TEST_F(InputLoaderTest, Vec1_C3) { run_and_verify<CFG_VEC1>(3); }
TEST_F(InputLoaderTest, Vec1_C2) { run_and_verify<CFG_VEC1>(2); }
TEST_F(InputLoaderTest, Vec1_C1) { run_and_verify<CFG_VEC1>(1); }
TEST_F(InputLoaderTest, Vec2_C2) { run_and_verify<CFG_VEC2>(2); }

// Padded path with spatial padding (px = 1).
TEST_F(InputLoaderTest, Vec1_C3_px1) { run_and_verify<CFG_VEC1>(3, 1); }
TEST_F(InputLoaderTest, Vec1_C1_px1) { run_and_verify<CFG_VEC1>(1, 1); }
TEST_F(InputLoaderTest, Vec2_C2_px1) { run_and_verify<CFG_VEC2>(2, 1); }

// Padded path with spatial padding (px = 2, 3).
TEST_F(InputLoaderTest, Vec1_C3_px2) { run_and_verify<CFG_VEC1>(3, 2); }
TEST_F(InputLoaderTest, Vec1_C3_px3) { run_and_verify<CFG_VEC1>(3, 3); }
TEST_F(InputLoaderTest, Vec1_C1_px2) { run_and_verify<CFG_VEC1>(1, 2); }
TEST_F(InputLoaderTest, Vec2_C2_px2) { run_and_verify<CFG_VEC2>(2, 2); }

// Unpadded with spatial padding.
TEST_F(InputLoaderTest, Unpadded_C4_px1) { run_and_verify<CFG_UNPADDED, false>(GROUP_SIZE, 1); }
TEST_F(InputLoaderTest, Unpadded_C4_px2) { run_and_verify<CFG_UNPADDED, false>(GROUP_SIZE, 2); }
TEST_F(InputLoaderTest, Unpadded_C4_px3) { run_and_verify<CFG_UNPADDED, false>(GROUP_SIZE, 3); }

// C != K tests: verify InputLoader works correctly when BlockCoords has C_in != C_out.
// The InputLoader should only depend on c_per_group (C_in), not k_per_group (C_out).
TEST_F(InputLoaderTest, Vec1_C3_K2)     { run_and_verify<CFG_VEC1>(3, 0, 2); }
TEST_F(InputLoaderTest, Vec1_C1_K4)     { run_and_verify<CFG_VEC1>(1, 0, 4); }
TEST_F(InputLoaderTest, Vec1_C2_K3)     { run_and_verify<CFG_VEC1>(2, 0, 3); }
TEST_F(InputLoaderTest, Vec2_C2_K4)     { run_and_verify<CFG_VEC2>(2, 0, 4); }
TEST_F(InputLoaderTest, Vec1_C3_K1)     { run_and_verify<CFG_VEC1>(3, 0, 1); }

// C != K with spatial padding.
TEST_F(InputLoaderTest, Vec1_C3_K2_px1) { run_and_verify<CFG_VEC1>(3, 1, 2); }
TEST_F(InputLoaderTest, Vec1_C1_K4_px1) { run_and_verify<CFG_VEC1>(1, 1, 4); }

// =============================================================================
// 16c InputLoader tests
// =============================================================================

namespace ns_16c = ck_tile::direct_conv::grouped_16c_tile::v2;

// Config indices for 16c kernel:
//   17 — Fprop, no swizzle, vector_size=8 (unpadded)
//   83 — Fprop, CyclicShift, vector_size=1 (padded, any c)
//   81 — Fprop, CyclicShift, vector_size=4 (padded, c%4==0)
static constexpr int CFG_16C_UNPADDED = 17;
static constexpr int CFG_16C_VEC4     = 81;
static constexpr int CFG_16C_VEC1     = 83;

template <int CfgIdx, bool Padding>
__global__ void test_input_load_kernel_16c(const _Float16* __restrict__ in,
                                           _Float16* __restrict__ lds_out,
                                           int groups,
                                           int c_per_group,
                                           int hi,
                                           int wi,
                                           int px,
                                           int k_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = ns_16c::TileConstants<ns_16c::configs[CfgIdx]>;
    constexpr auto cfg = ns_16c::configs[CfgIdx];
    constexpr int BLOCK_SIZE = cfg.block_size();

    const int kpg = (k_per_group > 0) ? k_per_group : c_per_group;
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, c_per_group, kpg);

    constexpr int LDS_C8 = TC::INPUT_LDS_BUFFER_SIZE_C8;
    constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
    __shared__ uint4 lds_buf[LDS_C8];

    for(int i = threadIdx.x; i < LDS_C8; i += BLOCK_SIZE)
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    __syncthreads();

    ck_tile::direct_conv::InputLoader<TC, cfg, ck_tile::fp16x4_t, Padding> il(bc, lds_buf, in, hi, wi, px, 0, 1, 1, 1, 1, c_per_group);
    il.prefetch_tile_to_lds(0);
    __syncthreads();

    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_FP16; i += BLOCK_SIZE)
        lds_out[i] = lds_fp16[i];
#endif
}

class InputLoader16cTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 16;

    static std::vector<_Float16> make_input_tensor(int hi, int wi, int groups, int c_per_group)
    {
        const int C_total = groups * c_per_group;
        const int total = hi * wi * C_total;
        std::vector<_Float16> inp(total);
        for(int i = 0; i < total; i++)
            inp[i] = static_cast<_Float16>(static_cast<float>((i % 13) + 1));
        return inp;
    }

    static float read_input(const std::vector<_Float16>& inp, int wi, int C_total, int h, int w, int c)
    {
        return static_cast<float>(inp[h * wi * C_total + w * C_total + c]);
    }

    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int px = 0, int k_per_group = 0)
    {
        using TC = ns_16c::TileConstants<ns_16c::configs[CfgIdx]>;
        constexpr auto cfg = ns_16c::configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
        constexpr int BLOCK_W = TC::BLOCK_W;
        constexpr int BLOCK_C8 = TC::BLOCK_C8;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int C_total = groups * c_per_group;
        const int wi = BLOCK_W + 4;
        const int hi = 4;

        auto inp_host = make_input_tensor(hi, wi, groups, c_per_group);

        _Float16* d_in = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_in, inp_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16 * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_in, inp_host.data(), inp_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_input_load_kernel_16c<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_in, d_lds_out, groups, c_per_group, hi, wi, px, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        constexpr auto swizzle = cfg.swizzle_type;
        for(int w = 0; w < BLOCK_W; w++)
        for(int c8 = 0; c8 < BLOCK_C8; c8++)
        for(int c = 0; c < 8; c++)
        {
            int lds_idx = w * BLOCK_C8 * 8 + c8 * 8 + c;
            float actual = static_cast<float>(lds_host[lds_idx]);

            int c8_phys;
            if constexpr(swizzle == SwizzleType::CyclicShift)
                c8_phys = (c8 + w) % BLOCK_C8;
            else if constexpr(swizzle == SwizzleType::XOR)
                c8_phys = c8 ^ w;
            else
                c8_phys = c8;

            int global_ch = c8_phys * 8 + c;
            int group_idx = global_ch / GROUP_SIZE;
            int group_local_c = global_ch % GROUP_SIZE;
            int w_actual = w - px;

            float expected;
            if(group_local_c < c_per_group && w_actual >= 0 && w_actual < wi)
            {
                int flat_c = group_idx * c_per_group + group_local_c;
                expected = read_input(inp_host, wi, C_total, 0, w_actual, flat_c);
            }
            else
            {
                expected = 0.0f;
            }

            EXPECT_EQ(actual, expected)
                << "cfg=" << CfgIdx << " w=" << w << " c8=" << c8 << " c=" << c
                << " c8_phys=" << c8_phys
                << " group=" << group_idx << " gc=" << group_local_c
                << " w_actual=" << w_actual;
        }

        ck_tile::hip_check_error(hipFree(d_in));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// Unpadded 16c
TEST_F(InputLoader16cTest, Unpadded_C16)     { run_and_verify<CFG_16C_UNPADDED, false>(16); }
TEST_F(InputLoader16cTest, Unpadded_C16_px1) { run_and_verify<CFG_16C_UNPADDED, false>(16, 1); }

// Padded 16c: c_per_group < 16
TEST_F(InputLoader16cTest, Vec1_C12)     { run_and_verify<CFG_16C_VEC1>(12); }
TEST_F(InputLoader16cTest, Vec1_C10)     { run_and_verify<CFG_16C_VEC1>(10); }
TEST_F(InputLoader16cTest, Vec1_C9)      { run_and_verify<CFG_16C_VEC1>(9); }
TEST_F(InputLoader16cTest, Vec4_C12)     { run_and_verify<CFG_16C_VEC4>(12); }
TEST_F(InputLoader16cTest, Vec1_C12_px1) { run_and_verify<CFG_16C_VEC1>(12, 1); }
TEST_F(InputLoader16cTest, Vec1_C9_px1)  { run_and_verify<CFG_16C_VEC1>(9, 1); }

// Spatial padding px = 2, 3.
TEST_F(InputLoader16cTest, Unpadded_C16_px2) { run_and_verify<CFG_16C_UNPADDED, false>(16, 2); }
TEST_F(InputLoader16cTest, Unpadded_C16_px3) { run_and_verify<CFG_16C_UNPADDED, false>(16, 3); }
TEST_F(InputLoader16cTest, Vec1_C12_px2) { run_and_verify<CFG_16C_VEC1>(12, 2); }
TEST_F(InputLoader16cTest, Vec1_C12_px3) { run_and_verify<CFG_16C_VEC1>(12, 3); }
TEST_F(InputLoader16cTest, Vec1_C9_px2)  { run_and_verify<CFG_16C_VEC1>(9, 2); }

// C != K for 16c
TEST_F(InputLoader16cTest, Vec1_C9_K16)  { run_and_verify<CFG_16C_VEC1>(9, 0, 16); }
TEST_F(InputLoader16cTest, Vec1_C12_K9)  { run_and_verify<CFG_16C_VEC1>(12, 0, 9); }

// =============================================================================
// 8c InputLoader tests
// =============================================================================

namespace ns_8c = ck_tile::direct_conv::grouped_8c_tile::v2;

// Config indices for 8c kernel:
//   17 — Fprop, no swizzle, vector_size=8 (unpadded)
//   81 — Fprop, CyclicShift, vector_size=1 (padded, any c)
//   79 — Fprop, CyclicShift, vector_size=4 (padded, c%4==0)
static constexpr int CFG_8C_UNPADDED = 17;
static constexpr int CFG_8C_VEC4     = 79;
static constexpr int CFG_8C_VEC1     = 81;

template <int CfgIdx, bool Padding>
__global__ void test_input_load_kernel_8c(const _Float16* __restrict__ in,
                                          _Float16* __restrict__ lds_out,
                                          int groups,
                                          int c_per_group,
                                          int hi,
                                          int wi,
                                          int px,
                                          int k_per_group = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
    using TC = ns_8c::TileConstants<ns_8c::configs[CfgIdx]>;
    constexpr auto cfg = ns_8c::configs[CfgIdx];
    constexpr int BLOCK_SIZE = cfg.block_size();

    const int kpg = (k_per_group > 0) ? k_per_group : c_per_group;
    ck_tile::direct_conv::BlockCoords<cfg> bc(groups, c_per_group, kpg);

    constexpr int LDS_C8 = TC::INPUT_LDS_BUFFER_SIZE_C8;
    constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
    __shared__ uint4 lds_buf[LDS_C8];

    for(int i = threadIdx.x; i < LDS_C8; i += BLOCK_SIZE)
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    __syncthreads();

    ck_tile::direct_conv::InputLoader<TC, cfg, ck_tile::fp16x4_t, Padding> il(bc, lds_buf, in, hi, wi, px, 0, 1, 1, 1, 1, c_per_group);
    il.prefetch_tile_to_lds(0);
    __syncthreads();

    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_FP16; i += BLOCK_SIZE)
        lds_out[i] = lds_fp16[i];
#endif
}

class InputLoader8cTest : public ::testing::Test
{
protected:
    static constexpr int GROUP_SIZE = 8;

    static std::vector<_Float16> make_input_tensor(int hi, int wi, int groups, int c_per_group)
    {
        const int C_total = groups * c_per_group;
        const int total = hi * wi * C_total;
        std::vector<_Float16> inp(total);
        for(int i = 0; i < total; i++)
            inp[i] = static_cast<_Float16>(static_cast<float>((i % 13) + 1));
        return inp;
    }

    static float read_input(const std::vector<_Float16>& inp, int wi, int C_total, int h, int w, int c)
    {
        return static_cast<float>(inp[h * wi * C_total + w * C_total + c]);
    }

    template <int CfgIdx, bool Padding = true>
    void run_and_verify(int c_per_group, int px = 0, int k_per_group = 0)
    {
        using TC = ns_8c::TileConstants<ns_8c::configs[CfgIdx]>;
        constexpr auto cfg = ns_8c::configs[CfgIdx];
        constexpr int BLOCK_SIZE = cfg.block_size();
        constexpr int LDS_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
        constexpr int BLOCK_W = TC::BLOCK_W;
        constexpr int BLOCK_C8 = TC::BLOCK_C8;
        constexpr int BLOCK_GROUPS = cfg.block_groups();

        const int groups = BLOCK_GROUPS;
        const int C_total = groups * c_per_group;
        const int wi = BLOCK_W + 4;
        const int hi = 4;

        auto inp_host = make_input_tensor(hi, wi, groups, c_per_group);

        _Float16* d_in = nullptr;
        _Float16* d_lds_out = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_in, inp_host.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds_out, LDS_FP16 * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_in, inp_host.data(), inp_host.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_input_load_kernel_8c<CfgIdx, Padding><<<dim3(1, 1, 1), BLOCK_SIZE>>>(
            d_in, d_lds_out, groups, c_per_group, hi, wi, px, k_per_group);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds_out, LDS_FP16 * sizeof(_Float16), hipMemcpyDeviceToHost));

        constexpr auto swizzle = cfg.swizzle_type;
        for(int w = 0; w < BLOCK_W; w++)
        for(int c8 = 0; c8 < BLOCK_C8; c8++)
        for(int c = 0; c < 8; c++)
        {
            int lds_idx = w * BLOCK_C8 * 8 + c8 * 8 + c;
            float actual = static_cast<float>(lds_host[lds_idx]);

            int c8_phys;
            if constexpr(swizzle == SwizzleType::CyclicShift)
                c8_phys = (c8 + w) % BLOCK_C8;
            else if constexpr(swizzle == SwizzleType::XOR)
                c8_phys = c8 ^ w;
            else
                c8_phys = c8;

            int global_ch = c8_phys * 8 + c;
            int group_idx = global_ch / GROUP_SIZE;
            int group_local_c = global_ch % GROUP_SIZE;
            int w_actual = w - px;

            float expected;
            if(group_local_c < c_per_group && w_actual >= 0 && w_actual < wi)
            {
                int flat_c = group_idx * c_per_group + group_local_c;
                expected = read_input(inp_host, wi, C_total, 0, w_actual, flat_c);
            }
            else
            {
                expected = 0.0f;
            }

            EXPECT_EQ(actual, expected)
                << "cfg=" << CfgIdx << " w=" << w << " c8=" << c8 << " c=" << c
                << " c8_phys=" << c8_phys
                << " group=" << group_idx << " gc=" << group_local_c
                << " w_actual=" << w_actual;
        }

        ck_tile::hip_check_error(hipFree(d_in));
        ck_tile::hip_check_error(hipFree(d_lds_out));
    }
};

// Unpadded 8c
TEST_F(InputLoader8cTest, Unpadded_C8)     { run_and_verify<CFG_8C_UNPADDED, false>(8); }
TEST_F(InputLoader8cTest, Unpadded_C8_px1) { run_and_verify<CFG_8C_UNPADDED, false>(8, 1); }

// Padded 8c: c_per_group < 8
TEST_F(InputLoader8cTest, Vec1_C6)     { run_and_verify<CFG_8C_VEC1>(6); }
TEST_F(InputLoader8cTest, Vec1_C5)     { run_and_verify<CFG_8C_VEC1>(5); }
TEST_F(InputLoader8cTest, Vec4_C4)     { run_and_verify<CFG_8C_VEC4>(4); }
TEST_F(InputLoader8cTest, Vec1_C6_px1) { run_and_verify<CFG_8C_VEC1>(6, 1); }
TEST_F(InputLoader8cTest, Vec1_C5_px1) { run_and_verify<CFG_8C_VEC1>(5, 1); }

// Spatial padding px = 2, 3.
TEST_F(InputLoader8cTest, Unpadded_C8_px2) { run_and_verify<CFG_8C_UNPADDED, false>(8, 2); }
TEST_F(InputLoader8cTest, Unpadded_C8_px3) { run_and_verify<CFG_8C_UNPADDED, false>(8, 3); }
TEST_F(InputLoader8cTest, Vec1_C6_px2) { run_and_verify<CFG_8C_VEC1>(6, 2); }
TEST_F(InputLoader8cTest, Vec1_C6_px3) { run_and_verify<CFG_8C_VEC1>(6, 3); }
TEST_F(InputLoader8cTest, Vec1_C5_px2) { run_and_verify<CFG_8C_VEC1>(5, 2); }

// C != K for 8c
TEST_F(InputLoader8cTest, Vec1_C5_K8) { run_and_verify<CFG_8C_VEC1>(5, 0, 8); }
TEST_F(InputLoader8cTest, Vec1_C6_K5) { run_and_verify<CFG_8C_VEC1>(6, 0, 5); }
