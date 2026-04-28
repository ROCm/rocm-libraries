// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_tile_conv_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/kernel/grouped_16c_fp16_tile_conv_impl_v2.hpp"
#include "ck_tile/ops/direct_convolution/utils/swizzle.hpp"
#include "ck_tile/ops/direct_convolution/utils/matrix_layout.hpp"
#pragma clang diagnostic pop

#include <hip/hip_runtime.h>
#include <vector>
#include <set>
#include <cstdio>
#include <string>

// ============================================================================
// Namespace aliases
// ============================================================================

namespace v3 = ck_tile::direct_conv::grouped_4c_tile::v3;
namespace v2 = ck_tile::direct_conv::grouped_16c_tile::v2;

// ============================================================================
// GPU kernel: capture LDS byte offsets for each lane in the MFMA read pattern
// ============================================================================

// Each thread computes the LDS byte offset it would read during the MFMA
// accumulation loop at a given kw_slice position. This uses the production
// tile_distribution and LDS read descriptor.
template <typename TC>
__global__ void capture_lds_read_offsets_kernel(int* offsets, int kw_slice)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;

    // Create the MFMA distribution.
    // This doesn't contain swizzle.
    constexpr auto dist = TC::Mfma::MakeDistribution();

    // Calculate the X-space index for this (warp_id, lane_id).
    const auto xs_idx = dist.calculate_index(
        ck_tile::array<ck_tile::index_t, 2>{warp_id, lane_id});

    // xs_idx has 3 components: {q_local, c4_local, c_sub}
    const ck_tile::index_t q_local  = xs_idx[ck_tile::number<0>{}];
    const ck_tile::index_t c4_local = xs_idx[ck_tile::number<1>{}];
    const ck_tile::index_t c_sub    = xs_idx[ck_tile::number<2>{}];

    // LDS read descriptor has shape [BLOCK_W, BLOCK_C4, 4].
    // The MFMA reads at position (q_local + kw_slice, c4_local, c_sub).
    const ck_tile::index_t w_coord = q_local + kw_slice;

    // Create the LDS read descriptor.
    // This contains the swizzle (e.g., cyclic shift) that is applied to the MFMA read addresses.
    constexpr auto lds_desc = TC::Input::MakeLdsReadDescriptor();

    // Calculate element offset (in fp16 units), convert to byte offset.
    const auto elem_offset = lds_desc.calculate_offset(
        ck_tile::make_tuple(w_coord, c4_local, c_sub));

    offsets[tid] = static_cast<int>(elem_offset) * 2;
}

// ============================================================================
// Host-side bank conflict analysis
// ============================================================================

// Count LDS bank conflicts for ds_read_b64 (reads 8 bytes = 2 dwords per lane).
//
// LDS has 32 banks, each 4 bytes wide. bank = (byte_offset / 4) % 32.
// ds_read_b64 touches dwords at byte_offset/4 and byte_offset/4+1.
//
// This counts the worst-case conflicts across all 64 lanes per wave
// (all lanes dispatched simultaneously). A bank with N distinct dword
// addresses contributes (N-1) conflicts.
int count_bank_conflicts(const std::vector<int>& byte_offsets, int block_size)
{
    constexpr int WAVE_SIZE = 64;
    constexpr int NUM_BANKS = 64; // gfx950: 64 LDS banks
    int total_conflicts = 0;

    const int num_waves = block_size / WAVE_SIZE;

    for(int wave = 0; wave < num_waves; wave++)
    {
        std::set<int> bank_accesses[NUM_BANKS];

        for(int lane = 0; lane < WAVE_SIZE; lane++)
        {
            int byte_off = byte_offsets[wave * WAVE_SIZE + lane];
            int dword0 = byte_off / 4;
            int dword1 = dword0 + 1;

            bank_accesses[dword0 % NUM_BANKS].insert(dword0);
            bank_accesses[dword1 % NUM_BANKS].insert(dword1);
        }

        for(int b = 0; b < NUM_BANKS; b++)
        {
            if(bank_accesses[b].size() > 1)
            {
                total_conflicts += static_cast<int>(bank_accesses[b].size()) - 1;
            }
        }
    }

    return total_conflicts;
}

// Check if all lanes within each wave read from distinct byte addresses.
bool all_lanes_distinct(const std::vector<int>& byte_offsets, int block_size)
{
    constexpr int WAVE_SIZE = 64;
    const int num_waves = block_size / WAVE_SIZE;

    for(int wave = 0; wave < num_waves; wave++)
    {
        std::set<int> addresses;
        for(int lane = 0; lane < WAVE_SIZE; lane++)
        {
            addresses.insert(byte_offsets[wave * WAVE_SIZE + lane]);
        }
        if(static_cast<int>(addresses.size()) != WAVE_SIZE)
        {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Helper: launch kernel, capture offsets, return on host
// ============================================================================

template <typename TC>
std::vector<int> run_capture_kernel(int block_size, int kw_slice)
{
    int* d_offsets = nullptr;
    ck_tile::hip_check_error(
        hipMalloc(&d_offsets, block_size * sizeof(int)));

    capture_lds_read_offsets_kernel<TC>
        <<<1, block_size>>>(d_offsets, kw_slice);
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<int> h_offsets(block_size);
    ck_tile::hip_check_error(
        hipMemcpy(h_offsets.data(), d_offsets,
                  block_size * sizeof(int), hipMemcpyDeviceToHost));

    (void)hipFree(d_offsets);
    return h_offsets;
}

// ============================================================================
// Test fixture
// ============================================================================

class LdsBankConflictTest : public ::testing::Test {};

// ============================================================================
// Config index reference (verified from configs[] array in source):
//
// 4c v3 (40 configs, 4 groups × 10: 5 Dgrad + 5 Fprop each):
//   Each group has the same wave configs in order:
//     w2q8, w2q4, w2q2, w2q1, w1q1
//   Fprop indices within each group: +5 offset from Dgrad
//
//   Group 0 (None, DRAM epilogue):  Fprop = {5:w2q8, 6:w2q4, 7:w2q2, 8:w2q1, 9:w1q1}
//   Group 2 (XOR, DRAM epilogue):   Fprop = {25:w2q8, 26:w2q4, 27:w2q2, 28:w2q1, 29:w1q1}
//
//   Matched pairs (same wave config, None vs XOR):
//     w2q4: None=6,  XOR=26
//     w2q2: None=7,  XOR=27
//     w1q1: None=9,  XOR=29
//
// 16c v2 (72 configs, 4 groups × 18: 9 Dgrad + 9 Fprop each):
//   Each group has waves_per_wg in order: 16,8,7,6,5,4,3,2,1
//   Fprop indices within each group: +9 offset from Dgrad
//
//   Group 0 (None, DRAM epilogue):  Fprop = {9:w16, 10:w8, ..., 14:w4, ..., 17:w1}
//   Group 2 (XOR, DRAM epilogue):   Fprop = {45:w16, 46:w8, ..., 50:w4, ..., 53:w1}
//
//   Matched pairs:
//     w4:  None=14, XOR=50
//     w1:  None=17, XOR=53
// ============================================================================

// ============================================================================
// Tests: XOR swizzle does not increase bank conflicts (4c and 16c)
//
// The XOR swizzle permutes the c8 address based on (w % BLOCK_C8). This
// helps reduce conflicts for access patterns where lanes reading the same
// spatial position but different channels would alias to the same bank.
//
// For the whole-wave worst-case metric, the effect depends on BLOCK_C8
// relative to the number of spatial positions per wave:
//   - 16c w4 (BLOCK_C8=8, 16 lanes per spatial pos): XOR reduces conflicts
//   - 4c configs (4 lanes per spatial pos): XOR doesn't change whole-wave metric
//   - Small BLOCK_C8 (e.g., w1: BLOCK_C8=2): XOR has limited effect
//
// We verify that XOR never makes things worse (conflicts_xor <= conflicts_none).
// ============================================================================

TEST_F(LdsBankConflictTest, XorSwizzleDoesNotIncreaseBankConflicts_4c)
{
    // Paired configs (same wave params, None vs XOR):
    //   w2q4: None=6,  XOR=26
    //   w2q2: None=7,  XOR=27
    //   w1q1: None=9,  XOR=29

    // w2q4
    {
        constexpr auto cfg_n = v3::configs[6];
        constexpr auto cfg_x = v3::configs[26];
        using TC_n = v3::TileConstants<cfg_n>;
        using TC_x = v3::TileConstants<cfg_x>;
        for(int kw = 0; kw < cfg_n.kw; kw++)
        {
            int c_n = count_bank_conflicts(run_capture_kernel<TC_n>(cfg_n.block_size(), kw), cfg_n.block_size());
            int c_x = count_bank_conflicts(run_capture_kernel<TC_x>(cfg_x.block_size(), kw), cfg_x.block_size());
            EXPECT_LE(c_x, c_n) << "4c w2q4 kw=" << kw << ": XOR=" << c_x << " None=" << c_n;
        }
    }
    // w2q2
    {
        constexpr auto cfg_n = v3::configs[7];
        constexpr auto cfg_x = v3::configs[27];
        using TC_n = v3::TileConstants<cfg_n>;
        using TC_x = v3::TileConstants<cfg_x>;
        for(int kw = 0; kw < cfg_n.kw; kw++)
        {
            int c_n = count_bank_conflicts(run_capture_kernel<TC_n>(cfg_n.block_size(), kw), cfg_n.block_size());
            int c_x = count_bank_conflicts(run_capture_kernel<TC_x>(cfg_x.block_size(), kw), cfg_x.block_size());
            EXPECT_LE(c_x, c_n) << "4c w2q2 kw=" << kw << ": XOR=" << c_x << " None=" << c_n;
        }
    }
    // w1q1
    {
        constexpr auto cfg_n = v3::configs[9];
        constexpr auto cfg_x = v3::configs[29];
        using TC_n = v3::TileConstants<cfg_n>;
        using TC_x = v3::TileConstants<cfg_x>;
        for(int kw = 0; kw < cfg_n.kw; kw++)
        {
            int c_n = count_bank_conflicts(run_capture_kernel<TC_n>(cfg_n.block_size(), kw), cfg_n.block_size());
            int c_x = count_bank_conflicts(run_capture_kernel<TC_x>(cfg_x.block_size(), kw), cfg_x.block_size());
            EXPECT_LE(c_x, c_n) << "4c w1q1 kw=" << kw << ": XOR=" << c_x << " None=" << c_n;
        }
    }
}

TEST_F(LdsBankConflictTest, XorSwizzleDoesNotIncreaseBankConflicts_16c)
{
    // w4: None=14, XOR=50
    {
        constexpr auto cfg_n = v2::configs[14];
        constexpr auto cfg_x = v2::configs[50];
        using TC_n = v2::TileConstants<cfg_n>;
        using TC_x = v2::TileConstants<cfg_x>;
        for(int kw = 0; kw < cfg_n.kw; kw++)
        {
            int c_n = count_bank_conflicts(run_capture_kernel<TC_n>(cfg_n.block_size(), kw), cfg_n.block_size());
            int c_x = count_bank_conflicts(run_capture_kernel<TC_x>(cfg_x.block_size(), kw), cfg_x.block_size());
            EXPECT_LE(c_x, c_n) << "16c w4 kw=" << kw << ": XOR=" << c_x << " None=" << c_n;
        }
    }
    // w1: None=17, XOR=53
    {
        constexpr auto cfg_n = v2::configs[17];
        constexpr auto cfg_x = v2::configs[53];
        using TC_n = v2::TileConstants<cfg_n>;
        using TC_x = v2::TileConstants<cfg_x>;
        for(int kw = 0; kw < cfg_n.kw; kw++)
        {
            int c_n = count_bank_conflicts(run_capture_kernel<TC_n>(cfg_n.block_size(), kw), cfg_n.block_size());
            int c_x = count_bank_conflicts(run_capture_kernel<TC_x>(cfg_x.block_size(), kw), cfg_x.block_size());
            EXPECT_LE(c_x, c_n) << "16c w1 kw=" << kw << ": XOR=" << c_x << " None=" << c_n;
        }
    }
}

// ============================================================================
// Tests: No swizzle configs have bank conflicts (sanity check)
// ============================================================================

TEST_F(LdsBankConflictTest, NoSwizzleHasBankConflicts)
{
    {
        constexpr auto cfg = v3::configs[6]; // 4c w2q4 None
        using TC = v3::TileConstants<cfg>;
        auto off = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_GT(count_bank_conflicts(off, cfg.block_size()), 0)
            << "4c None w2q4 should have bank conflicts";
    }
    {
        constexpr auto cfg = v2::configs[14]; // 16c w4 None
        using TC = v2::TileConstants<cfg>;
        auto off = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_GT(count_bank_conflicts(off, cfg.block_size()), 0)
            << "16c None w4 should have bank conflicts";
    }
}

// ============================================================================
// Tests: All lanes read distinct addresses
// ============================================================================

TEST_F(LdsBankConflictTest, AllLanesDistinct_4c)
{
    // XOR swizzle configs
    {
        constexpr auto cfg = v3::configs[26]; // w2q4 XOR
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c XOR w2q4: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v3::configs[27]; // w2q2 XOR
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c XOR w2q2: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v3::configs[29]; // w1q1 XOR
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c XOR w1q1: lanes within a wave read aliased addresses";
    }
    // No-swizzle configs
    {
        constexpr auto cfg = v3::configs[6]; // w2q4 None
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c None w2q4: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v3::configs[7]; // w2q2 None
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c None w2q2: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v3::configs[9]; // w1q1 None
        using TC = v3::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "4c None w1q1: lanes within a wave read aliased addresses";
    }
}

TEST_F(LdsBankConflictTest, AllLanesDistinct_16c)
{
    {
        constexpr auto cfg = v2::configs[50]; // w4 XOR
        using TC = v2::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "16c XOR w4: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v2::configs[53]; // w1 XOR
        using TC = v2::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "16c XOR w1: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v2::configs[14]; // w4 None
        using TC = v2::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "16c None w4: lanes within a wave read aliased addresses";
    }
    {
        constexpr auto cfg = v2::configs[17]; // w1 None
        using TC = v2::TileConstants<cfg>;
        auto offsets = run_capture_kernel<TC>(cfg.block_size(), 0);
        EXPECT_TRUE(all_lanes_distinct(offsets, cfg.block_size()))
            << "16c None w1: lanes within a wave read aliased addresses";
    }
}

// ============================================================================
// GPU kernel: compute HIP conv SwizzleT LDS byte offsets for reference.
//
// This mirrors the HIP conv input read pattern:
//   lane_q  = lane % 16
//   lane_c4 = lane / 16
//   offset  = SwizzleT::offset_uint2(lane_q + kw_slice, wave * GROUP_SIZE_4 + lane_c4) * sizeof(uint2)
// ============================================================================

template <int BLOCK_C, int GROUP_SIZE_4>
__global__ void capture_hip_conv_lds_read_offsets_kernel(int* offsets, int kw_slice)
{
    using Sw = ck_tile::direct_conv::SwizzleT<BLOCK_C>;

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;

    const int lane_q  = lane % 16;
    const int lane_c4 = lane / 16;

    const int c4 = wave * GROUP_SIZE_4 + lane_c4;
    const int w  = lane_q + kw_slice;

    offsets[tid] = Sw::offset_uint2(w, c4) * static_cast<int>(sizeof(uint2));
}

template <int BLOCK_C, int GROUP_SIZE_4>
std::vector<int> run_hip_conv_capture_kernel(int block_size, int kw_slice)
{
    int* d_offsets = nullptr;
    ck_tile::hip_check_error(hipMalloc(&d_offsets, block_size * sizeof(int)));

    capture_hip_conv_lds_read_offsets_kernel<BLOCK_C, GROUP_SIZE_4>
        <<<1, block_size>>>(d_offsets, kw_slice);
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<int> h_offsets(block_size);
    ck_tile::hip_check_error(
        hipMemcpy(h_offsets.data(), d_offsets,
                  block_size * sizeof(int), hipMemcpyDeviceToHost));

    (void)hipFree(d_offsets);
    return h_offsets;
}

// ============================================================================
// Test: CK Tile cyclic-shift descriptor produces the same LDS read byte
// offsets as the HIP conv SwizzleT reference.
//
// The cyclic-shift swizzle in the CK Tile descriptor chain (via
// make_inverse_cyclic_shift_transform in MakeLdsReadDescriptor) should produce
// the same physical LDS read addresses as SwizzleT::offset_uint2.
// ============================================================================

// The CK Tile cyclic shift uses inverse direction (c8 - w) % C8 vs HIP conv's
// (c8 + w) % C8. Both are valid bijective swizzles with identical bank conflict
// properties, but they produce different physical LDS offsets.
//
// This test verifies that despite different offsets, the bank conflict counts match.
TEST_F(LdsBankConflictTest, CyclicShiftSameBankConflictsAsHipConv_16c)
{
    // Config 73: Fprop, waves_per_wg=8, CyclicShift, DRAM epilogue
    constexpr auto cfg_cs = v2::configs[73];
    static_assert(cfg_cs.swizzle_type == ck_tile::direct_conv::SwizzleType::CyclicShift,
                  "Expected CyclicShift config");

    using TC_cs = v2::TileConstants<cfg_cs>;
    constexpr int BLOCK_C = TC_cs::BLOCK_C;
    constexpr int GROUP_SIZE_4 = TC_cs::GROUP_SIZE_4;

    const int block_size = cfg_cs.block_size();

    for(int kw = 0; kw < cfg_cs.kw; kw++)
    {
        auto ck_offsets  = run_capture_kernel<TC_cs>(block_size, kw);
        auto hip_offsets = run_hip_conv_capture_kernel<BLOCK_C, GROUP_SIZE_4>(block_size, kw);

        int ck_conflicts  = count_bank_conflicts(ck_offsets, block_size);
        int hip_conflicts = count_bank_conflicts(hip_offsets, block_size);

        EXPECT_EQ(ck_conflicts, hip_conflicts)
            << "16c w8 CyclicShift kw=" << kw
            << ": CK=" << ck_conflicts << " HIP=" << hip_conflicts;

        // Also verify all lanes read distinct addresses
        EXPECT_TRUE(all_lanes_distinct(ck_offsets, block_size))
            << "16c w8 CyclicShift CK kw=" << kw << ": lane aliasing detected";
        EXPECT_TRUE(all_lanes_distinct(hip_offsets, block_size))
            << "16c w8 CyclicShift HIP kw=" << kw << ": lane aliasing detected";
    }
}

// ============================================================================
// Test: CK Tile cyclic-shift has same bank conflicts as HIP conv SwizzleT
// ============================================================================

TEST_F(LdsBankConflictTest, CyclicShiftBankConflictsMatchHipConv_16c)
{
    constexpr auto cfg_cs = v2::configs[73];
    using TC_cs = v2::TileConstants<cfg_cs>;
    constexpr int BLOCK_C = TC_cs::BLOCK_C;
    constexpr int GROUP_SIZE_4 = TC_cs::GROUP_SIZE_4;

    const int block_size = cfg_cs.block_size();

    printf("\nCyclicShift vs HIP conv bank conflict comparison (16c w8):\n");
    printf("  %-10s  %-12s  %-12s\n", "kw_slice", "CK_conflicts", "HIP_conflicts");

    for(int kw = 0; kw < cfg_cs.kw; kw++)
    {
        auto ck_offsets  = run_capture_kernel<TC_cs>(block_size, kw);
        auto hip_offsets = run_hip_conv_capture_kernel<BLOCK_C, GROUP_SIZE_4>(block_size, kw);

        int ck_conflicts  = count_bank_conflicts(ck_offsets, block_size);
        int hip_conflicts = count_bank_conflicts(hip_offsets, block_size);

        printf("  %-10d  %-12d  %-12d\n", kw, ck_conflicts, hip_conflicts);

        EXPECT_EQ(ck_conflicts, hip_conflicts)
            << "16c w8 kw=" << kw << ": CK=" << ck_conflicts << " HIP=" << hip_conflicts;
    }
}

// ============================================================================
// Diagnostic: Print bank conflicts for 16c w8 variants (None, XOR, CyclicShift)
// ============================================================================

TEST_F(LdsBankConflictTest, PrintConflictSummary_16c_w8)
{
    printf("\n16c w8 Bank conflict summary (all kw_slice):\n");
    printf("  %-45s", "Config");
    for(int kw = 0; kw < 3; kw++) printf("  kw=%d", kw);
    printf("\n");

    auto print_row = [](const char* name, auto& offsets_by_kw, int block_size) {
        printf("  %-45s", name);
        for(auto& off : offsets_by_kw)
            printf("  %4d", count_bank_conflicts(off, block_size));
        printf("\n");
    };

    // 16c w8 None (idx 10 = Fprop w8)
    {
        constexpr auto cfg = v2::configs[10];
        using TC = v2::TileConstants<cfg>;
        std::vector<std::vector<int>> offsets_by_kw;
        for(int kw = 0; kw < cfg.kw; kw++)
            offsets_by_kw.push_back(run_capture_kernel<TC>(cfg.block_size(), kw));
        print_row("16c w8 None (idx 10)", offsets_by_kw, cfg.block_size());
    }
    // 16c w8 XOR (idx 46 = Fprop w8 XOR)
    {
        constexpr auto cfg = v2::configs[46];
        using TC = v2::TileConstants<cfg>;
        std::vector<std::vector<int>> offsets_by_kw;
        for(int kw = 0; kw < cfg.kw; kw++)
            offsets_by_kw.push_back(run_capture_kernel<TC>(cfg.block_size(), kw));
        print_row("16c w8 XOR  (idx 46)", offsets_by_kw, cfg.block_size());
    }
    // 16c w8 CyclicShift CK Tile (idx 73)
    {
        constexpr auto cfg = v2::configs[73];
        using TC = v2::TileConstants<cfg>;
        std::vector<std::vector<int>> offsets_by_kw;
        for(int kw = 0; kw < cfg.kw; kw++)
            offsets_by_kw.push_back(run_capture_kernel<TC>(cfg.block_size(), kw));
        print_row("16c w8 CyclicShift CK (idx 73)", offsets_by_kw, cfg.block_size());
    }
    // 16c w8 HIP conv SwizzleT reference
    {
        constexpr auto cfg = v2::configs[73]; // same wave config
        using TC = v2::TileConstants<cfg>;
        constexpr int BLOCK_C = TC::BLOCK_C;
        constexpr int GROUP_SIZE_4 = TC::GROUP_SIZE_4;
        std::vector<std::vector<int>> offsets_by_kw;
        for(int kw = 0; kw < cfg.kw; kw++)
            offsets_by_kw.push_back(
                run_hip_conv_capture_kernel<BLOCK_C, GROUP_SIZE_4>(cfg.block_size(), kw));
        print_row("16c w8 HIP SwizzleT (reference)", offsets_by_kw, cfg.block_size());
    }
}

// ============================================================================
// Diagnostic test: print bank conflict summary
// ============================================================================

TEST_F(LdsBankConflictTest, PrintConflictSummary)
{
    printf("\nBank conflict summary (kw_slice=0, worst-case across full wave):\n");
    printf("  %-45s  conflicts\n", "Config");
    printf("  %-45s  ---------\n", "------");

    auto print_line = [](const char* name, const std::vector<int>& offsets, int block_size) {
        int conflicts = count_bank_conflicts(offsets, block_size);
        printf("  %-45s  %4d\n", name, conflicts);
    };

    // 4c configs
    {
        constexpr auto cfg = v3::configs[6];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w2q4 None (idx 6)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v3::configs[26];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w2q4 XOR  (idx 26)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v3::configs[7];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w2q2 None (idx 7)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v3::configs[27];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w2q2 XOR  (idx 27)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v3::configs[9];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w1q1 None (idx 9)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v3::configs[29];
        using TC = v3::TileConstants<cfg>;
        print_line("4c w1q1 XOR  (idx 29)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }

    // 16c configs
    {
        constexpr auto cfg = v2::configs[14];
        using TC = v2::TileConstants<cfg>;
        print_line("16c w4 None (idx 14)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v2::configs[50];
        using TC = v2::TileConstants<cfg>;
        print_line("16c w4 XOR  (idx 50)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v2::configs[17];
        using TC = v2::TileConstants<cfg>;
        print_line("16c w1 None (idx 17)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
    {
        constexpr auto cfg = v2::configs[53];
        using TC = v2::TileConstants<cfg>;
        print_line("16c w1 XOR  (idx 53)", run_capture_kernel<TC>(cfg.block_size(), 0), cfg.block_size());
    }
}
