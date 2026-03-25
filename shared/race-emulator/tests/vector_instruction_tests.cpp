// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/FloatTypes.h"
#include "race-emulator/Wave.h"
#include "race-emulator/Workgroup.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>

namespace {

using namespace raceemulator;

void tryExecute(Wave &regs, const std::string &line) {
  regs.tryExecute(line, false);
}

} // namespace

TEST(Instructions, InstructionTestZero) {
  Workgroup wg;
  Wave regs(10, 10, 16, wg);
  tryExecute(regs, "v_mov_b32_e32 v1, 17");
  EXPECT_EQ(regs.getVgpr(1, 0), 17);

  tryExecute(regs, "s_mov_b32 s3, 18");
  tryExecute(regs, "v_mov_b32_e32 v1, s3");
  EXPECT_EQ(regs.getVgpr(1, 0), 18);

  regs.setVgpr64(0, /* lane */ 3, 2);
  regs.setVgpr64(2, /* lane */ 3, 137);
  // 137 + (2 << 3) = 137 + 16 = 153
  tryExecute(regs, "v_lshl_add_u64 v[2:3], v[0:1], 3, v[2:3]");
  EXPECT_EQ(regs.getVgpr64(2, 3), 153);

  // Let's now test v_add_f64.
  regs.setVgpr64(4, /* lane */ 5, std::bit_cast<uint64_t>(2.5));
  regs.setVgpr64(6, /* lane */ 5, std::bit_cast<uint64_t>(3.5));
  tryExecute(regs, "v_add_f64 v[4:5], v[6:7], v[4:5]");
  double result = std::bit_cast<double>(regs.getVgpr64(4, 5));
  EXPECT_EQ(result, 6.0);
}

TEST(Instructions, MixedOperandMath) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 16, /*sgprCount*/ 16, /*waveSize*/ 32, wg);

  // Setup: s0 = 10.0f (stored as bits), v2 = 5.0f (all lanes)
  uint32_t float10bits = std::bit_cast<uint32_t>(10.0f);
  regs.setSgpr(0, float10bits);
  EXPECT_EQ(regs.getSgpr(0), float10bits);

  for (int i = 0; i < 32; ++i) {
    regs.setVgpr(2, /*lane*/ i, std::bit_cast<uint32_t>(5.0f));
  }

  // 2. Register + Literal
  // v_mul_f32 v1, s0, 2.0 -> v1 = 10.0 * 2.0 = 20.0
  tryExecute(regs, "v_mul_f32_e32 v1, s0, 2.0");
  float res2 = std::bit_cast<float>(regs.getVgpr(1, /*lane*/ 15));
  EXPECT_FLOAT_EQ(res2, 20.0f);

  // 1. Literal + Literal (Immediate math)
  // v_add_f32 v0, 1.5, 2.5 -> v0 should be 4.0
  tryExecute(regs, "v_add_f32_e32 v0, 1.5, 2.5");
  float res1 = std::bit_cast<float>(regs.getVgpr(0, /*lane*/ 0));
  EXPECT_FLOAT_EQ(res1, 4.0f);

  // 3. VGPR + SGPR
  // v_add_f32 v3, v1, s0 -> 20.0 + 10.0 = 30.0
  tryExecute(regs, "v_add_f32_e32 v3, v1, s0");
  float res3 = std::bit_cast<float>(regs.getVgpr(3, /*lane*/ 31));
  EXPECT_FLOAT_EQ(res3, 30.0f);
}

TEST(Instructions, LaneIndependence) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 4, wg);

  // Set distinct values in v0 for each lane
  regs.setVgpr(0, /*lane*/ 0, 10);
  regs.setVgpr(0, /*lane*/ 1, 20);
  regs.setVgpr(0, /*lane*/ 2, 30);
  regs.setVgpr(0, /*lane*/ 3, 40);

  // Execute a shift operation: v1 = (v0 << 1) | 0
  tryExecute(regs, "v_lshl_or_b32 v1, v0, 1, 0");

  EXPECT_EQ(regs.getVgpr(1, /*lane*/ 0), 20);
  EXPECT_EQ(regs.getVgpr(1, /*lane*/ 1), 40);
  EXPECT_EQ(regs.getVgpr(1, /*lane*/ 2), 60);
  EXPECT_EQ(regs.getVgpr(1, /*lane*/ 3), 80);
}

TEST(Instructions, Registers64Bit) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1, wg);

  uint64_t bigValue = 0x123456789ABCDEF0;

  // 1. Set via helper
  regs.setVgpr64(0, /*lane*/ 0, bigValue);

  // 2. Verify individual 32-bit registers (Little Endian)
  // v0 = Low 32 bits, v1 = High 32 bits
  EXPECT_EQ(regs.getVgpr(0, /*lane*/ 0), 0x9ABCDEF0);
  EXPECT_EQ(regs.getVgpr(1, /*lane*/ 0), 0x12345678);

  // 3. Move via instruction
  tryExecute(regs, "v_mov_b32_e32 v2, v0"); // v2 = low
  tryExecute(regs, "v_mov_b32_e32 v3, v1"); // v3 = high

  // 4. Reconstruct 64-bit from v[2:3]
  uint64_t reconstructed = regs.getVgpr64(2, /*lane*/ 0);
  EXPECT_EQ(reconstructed, bigValue);
}

TEST(Instructions, Conversions) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1, wg);

  // Float -> Double
  float input = 3.14159f;
  regs.setVgpr(0, /*lane*/ 0, std::bit_cast<uint32_t>(input));

  // v[2:3] = double(v0)
  tryExecute(regs, "v_cvt_f64_f32_e32 v[2:3], v0");

  double result = std::bit_cast<double>(regs.getVgpr64(2, /*lane*/ 0));
  // Allow small epsilon due to f32->f64 precision expansion
  EXPECT_NEAR(result, 3.14159, 1e-6);

  // Double -> Float
  regs.setVgpr64(2, /*lane*/ 0, std::bit_cast<uint64_t>(12345.6789));

  // v1 = float(v[2:3])
  tryExecute(regs, "v_cvt_f32_f64_e32 v1, v[2:3]");

  float resF = std::bit_cast<float>(regs.getVgpr(1, /*lane*/ 0));
  EXPECT_FLOAT_EQ(resF, 12345.6789f);
}

TEST(Instructions, VLshlOrB32) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1, wg);

  // 1. Basic Operation: (10 << 2) | 5
  // 10 (binary 1010) << 2 = 40 (binary 101000)
  // 40 | 5 (binary 101) = 45 (binary 101101)
  regs.setVgpr(0, /*lane*/ 0, 10);
  regs.setVgpr(1, /*lane*/ 0, 5);
  // v2 = (v0 << 2) | v1
  tryExecute(regs, "v_lshl_or_b32 v2, v0, 2, v1");
  EXPECT_EQ(regs.getVgpr(2, /*lane*/ 0), 45);

  // 2. Test Shift Masking (Shift > 31)
  // Shift 33 is equivalent to shift 1 (33 & 31 = 1).
  // (10 << 1) | 0 = 20
  tryExecute(regs, "v_lshl_or_b32 v3, v0, 33, 0");
  EXPECT_EQ(regs.getVgpr(3, /*lane*/ 0), 20);

  // 3. Test Full Register Inputs (No literals)
  // Setup shift amount in s0
  regs.setSgpr(0, 3);
  // v2 = (v0 << s0) | v1
  // (10 << 3) | 5 = 80 | 5 = 85
  tryExecute(regs, "v_lshl_or_b32 v2, v0, s0, v1");
  EXPECT_EQ(regs.getVgpr(2, /*lane*/ 0), 85);
}

TEST(Instructions, VCmpGtI32E32) {
  // Example: v_cmp_gt_i32_e32 vcc, s5, v0
  Workgroup wg;
  Wave regs(/*vgprCount*/ 2, /*sgprCount*/ 6, /*waveSize*/ 64, wg);
  regs.setSgpr(2, 25); // s2 = 25
  // set v1 iota from 0:
  for (int lane = 0; lane < regs.getWaveSize(); ++lane) {
    regs.setVgpr(1, lane, 100);
  }
  regs.setVgpr(1, 20, 24);
  regs.setVgpr(1, 40, 25); // 25 > 25 ? no.
  regs.setVgpr(1, 50, 7);
  uint64_t expected = (uint64_t(1) << 50) |
                      (uint64_t(1) << 20); // lanes 20 and 50 should be set.

  tryExecute(regs, "v_cmp_gt_i32_e32 vcc, s2, v1");
  EXPECT_EQ(regs.getVccU64(), expected);

  tryExecute(regs, "v_cmp_gt_i32_e32 vcc, 0, 1");
  EXPECT_EQ(regs.getVccU64(), 0);

  tryExecute(regs, "v_cmp_gt_i32_e32 vcc, 1, 0");
  EXPECT_EQ(regs.getVccU64(), uint64_t(-1));
}

TEST(Instructions, VReadFirstLane) {
  // Setup: 1 VGPR, 1 SGPR, WaveSize 64
  Workgroup wg;
  Wave regs(1, 1, 64, wg);

  // 1. Initialize v0 such that Lane N contains the value N.
  // This acts as our "Lane ID" map.
  for (int i = 0; i < 64; ++i) {
    regs.setVgpr(0, i, static_cast<uint32_t>(i));
  }

  // Case 1: Lowest bit set (Lane 0)
  // EXEC = ...0001
  regs.setExecU64(1);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  // Should read from Lane 0 (value 0)
  EXPECT_EQ(regs.getSgpr(0), 0);

  // Case 2: A middle bit set (Lane 4)
  // EXEC = ...00010000
  regs.setExecU64(1ULL << 4);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  // Should read from Lane 4 (value 4)
  EXPECT_EQ(regs.getSgpr(0), 4);

  // Case 3: Multiple bits set (Lanes 10 and 20 active)
  // Spec says: "Lowest active lane"
  regs.setExecU64((1ULL << 10) | (1ULL << 20));
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  // Should pick Lane 10, not 20
  EXPECT_EQ(regs.getSgpr(0), 10);

  // Case 4: High bit set (Lane 63)
  regs.setExecU64(1ULL << 63);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  EXPECT_EQ(regs.getSgpr(0), 63);

  // Case 5: EXEC is Zero (The Edge Case)
  // Spec says: "if EXEC == 0 then lane = 0"
  // Modify Lane 0 to a unique value to prove we are reading it freshly
  regs.setExecU64(~0ULL);
  regs.setVgpr(0, 0, 0xDEADBEEF);

  regs.setExecU64(0);

  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  EXPECT_EQ(regs.getSgpr(0), 0xDEADBEEF);
}

TEST(Instructions, VAshrRevI32_LLVMExamples) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  // LLVM Example 1: result = ashr i32 4, 1
  // Yields: 2
  // Syntax: v_ashrrev_i32_e32 dst, shift, value
  tryExecute(regs, "v_ashrrev_i32_e32 v0, 1, 4");
  EXPECT_EQ(static_cast<int32_t>(regs.getVgpr(0, 0)), 2);

  // LLVM Example 2: result = ashr i32 4, 2
  // Yields: 1
  tryExecute(regs, "v_ashrrev_i32_e32 v0, 2, 4");
  EXPECT_EQ(static_cast<int32_t>(regs.getVgpr(0, 0)), 1);

  // LLVM Example 3: result = ashr i8 -2, 1
  // -2 (0xFFFFFFFE) >> 1 = -1 (0xFFFFFFFF), sign bit preserved.
  regs.setVgpr(1, 0, -2); // Load -2 into v1
  tryExecute(regs, "v_ashrrev_i32_e32 v0, 1, v1");
  EXPECT_EQ(static_cast<int32_t>(regs.getVgpr(0, 0)), -1);

  // Case 4: Sign Extension Check (Negative Number)
  // -4 >> 1 should be -2
  // Binary: ...111100 >> 1 = ...111110
  regs.setVgpr(1, 0, -4);
  tryExecute(regs, "v_ashrrev_i32_e32 v0, 1, v1");
  EXPECT_EQ(static_cast<int32_t>(regs.getVgpr(0, 0)), -2);

  // Case 5: Large Shift (Hardware Masking)
  // LLVM says shifts >= bitwidth are undefined, but AMDGPU spec says
  // the shift amount is masked by 0x1F (31).
  // Shift 33 -> 33 & 0x1F = 1.
  // 4 >> 1 = 2.
  tryExecute(regs, "v_ashrrev_i32_e32 v0, 33, 4");
  EXPECT_EQ(static_cast<int32_t>(regs.getVgpr(0, 0)), 2);
}

TEST(Instructions, VBfeU32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  // Data: 0xABCD1234
  // Binary: ... 0001 0010 0011 0100
  regs.setVgpr(0, 0, 0xABCD1234);

  // Case 1: Extract "3" (Nibble at bit 4)
  // Offset = 4, Width = 4
  regs.setVgpr(1, 0, 4); // Offset
  regs.setVgpr(2, 0, 4); // Width

  tryExecute(regs, "v_bfe_u32 v3, v0, v1, v2");
  EXPECT_EQ(regs.getVgpr(3, 0), 3);

  // Case 2: Extract "12" (Byte at bit 8)
  // Offset = 8, Width = 8
  // Result should be 0x12
  tryExecute(regs, "v_bfe_u32 v3, v0, 8, 8"); // Using literals for offset/width
  EXPECT_EQ(regs.getVgpr(3, 0), 0x12);

  // Case 3: Zero Width -> Result 0
  tryExecute(regs, "v_bfe_u32 v3, v0, 5, 0");
  EXPECT_EQ(regs.getVgpr(3, 0), 0);
}
TEST(Instructions, VAdd3U32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg); // 4 VGPR, 4 SGPRs

  // Case: v0 = v1 + v2 + s0
  // v1 = 10, v2 = 20, s0 = 5
  regs.setVgpr(1, 0, 10);
  regs.setVgpr(2, 0, 20);
  regs.setSgpr(0, 5);

  tryExecute(regs, "v_add3_u32 v0, v1, v2, s0");

  EXPECT_EQ(regs.getVgpr(0, 0), 35);

  // Case: Overflow wrapping
  // Max + 1 + 1 = 1
  regs.setVgpr(1, 0, 0xFFFFFFFF);
  regs.setVgpr(2, 0, 1);
  regs.setSgpr(0, 1);

  tryExecute(regs, "v_add3_u32 v0, v1, v2, s0");
  EXPECT_EQ(regs.getVgpr(0, 0), 1);
}

TEST(Instructions, VCmpU32Sdwa_Unified) {
  Workgroup wg;
  Wave regs(4, 2, 1, wg);

  // Data Setup
  // v1: High=0xAAAA, Low=0x1234 -> 0xAAAA1234
  // v2: High=0xBBBB, Low=0x1234 -> 0xBBBB1234
  regs.setVgpr(1, 0, 0xAAAA1234);
  regs.setVgpr(2, 0, 0xBBBB1234);

  // Clear Destination (s[0:1])
  regs.setSgpr64(0, 0);

  // Case 1: Compare Lower Words (WORD_0)
  // v1.WORD_0 (0x1234) == v2.WORD_0 (0x1234) -> MATCH
  tryExecute(
      regs, "v_cmp_eq_u32_sdwa s[0:1], v1, v2 src0_sel:WORD_0 src1_sel:WORD_0");
  EXPECT_EQ(regs.getSgpr64(0), 1ULL);

  // Case 2: Compare Upper Words (WORD_1)
  // v1.WORD_1 (0xAAAA) == v2.WORD_1 (0xBBBB) -> NO MATCH
  regs.setSgpr64(0, 0); // Reset Dest
  tryExecute(
      regs, "v_cmp_eq_u32_sdwa s[0:1], v1, v2 src0_sel:WORD_1 src1_sel:WORD_1");
  EXPECT_EQ(regs.getSgpr64(0), 0ULL);

  // Case 3: Mixed Selectors (Byte vs Word)
  // v1 = 0xAAAA1234
  // Check if BYTE_2 (0x12) matches the value in v3
  regs.setVgpr(3, 0, 0x00000012);
  regs.setSgpr64(0, 0);

  // v1.BYTE_2 is 0x12 (extracted from 0xAAAA1234)
  // v3.DWORD is 0x12 (full value)
  tryExecute(regs,
             "v_cmp_eq_u32_sdwa s[0:1], v1, v3 src0_sel:BYTE_1 src1_sel:DWORD");
  EXPECT_EQ(regs.getSgpr64(0), 1ULL);

  // Case 4: Default Behavior (Implicit DWORD)
  // 0xAAAA1234 != 0xBBBB1234
  regs.setSgpr64(0, 0);
  // Missing selectors imply DWORD for both
  tryExecute(regs, "v_cmp_eq_u32_sdwa s[0:1], v1, v2");
  EXPECT_EQ(regs.getSgpr64(0), 0ULL);
}

TEST(Instructions, VCndMask_MultiLane) {
  // Setup:
  // 3 VGPRs (v0 dest, v1 src0, v2 src1)
  // 2 SGPRs (s[0:1] for VCC)
  // WaveSize = 2 (We need at least 2 lanes to test bit 0 vs bit 1)
  Workgroup wg;
  Wave regs(3, 2, 2, wg);

  // 1. Initialize Inputs per Lane
  // Lane 0: v1=100, v2=200
  regs.setVgpr(1, 0, 100);
  regs.setVgpr(2, 0, 200);

  // Lane 1: v1=100, v2=200
  regs.setVgpr(1, 1, 100);
  regs.setVgpr(2, 1, 200);

  // 2. Setup VCC (Mask)
  // We want:
  // Lane 0 (Bit 0) -> 0 (Select False/Src0 -> v1)
  // Lane 1 (Bit 1) -> 1 (Select True/Src1  -> v2)
  // VCC Pattern: ...10 (Binary) -> 0x2
  regs.setSgpr64(0, 0x2);

  // 3. Execute
  // v0 = VCC ? v2 : v1
  tryExecute(regs, "v_cndmask_b32_e32 v0, v1, v2, s[0:1]");

  // 4. Verify Results

  // Lane 0: VCC bit 0 was 0. Should have picked v1 (100).
  EXPECT_EQ(regs.getVgpr(0, 0), 100);

  // Lane 1: VCC bit 1 was 1. Should have picked v2 (200).
  EXPECT_EQ(regs.getVgpr(0, 1), 200);
}

TEST(Instructions, VCndMask_RespectsExec) {
  Workgroup wg;
  Wave regs(3, 2, 64, wg); // 64 lanes

  // Initial State:
  regs.setVgpr(0, 0, 99);
  regs.setVgpr(0, 1, 99);
  regs.setVgpr(1, 0, 10);
  regs.setVgpr(1, 1, 10);
  regs.setVgpr(2, 0, 20);
  regs.setVgpr(2, 1, 20);

  // Setup:
  // Lane 0: EXEC=1. Should update.
  // Lane 1: EXEC=0. Should PRESERVE old value.
  regs.setExecU64(0x1); // Only bit 0 is set

  // Mask: all zero.
  regs.setSgpr64(0, 0);

  //           Lane
  //           0  1
  // Reg  0   99 99
  //      1   10 10
  //      2   20 20

  // Exec: v0 = s[0:1] ? v2 : v1
  // Lane 0: Active, VCC=0 -> Takes v1 (10)
  // Lane 1: Inactive -> Should stay 99
  tryExecute(regs, "v_cndmask_b32_e32 v0, v1, v2, s[0:1]");

  EXPECT_EQ(regs.getVgpr(0, 0), 10); // Updated
  EXPECT_EQ(regs.getVgpr(0, 1), 99); // Preserved! (If this is 10, EXEC failed)
}

TEST(Instructions, VAddLshlU32) {
  Workgroup wg;
  Wave regs(4, 0, 1, wg); // 4 VGPRs, 0 SGPRs, WaveSize 1

  // Case 1: Simple Address Calculation
  // Result = (10 + 20) << 2
  //        = 30 << 2
  //        = 120
  regs.setVgpr(1, 0, 10); // Src0
  regs.setVgpr(2, 0, 20); // Src1
  regs.setVgpr(3, 0, 2);  // Shift

  tryExecute(regs, "v_add_lshl_u32 v0, v1, v2, v3");
  EXPECT_EQ(regs.getVgpr(0, 0), 120);

  // Case 2: Wrap-around Addition before Shift
  // (0xFFFFFFFF + 1) << 1
  // (0) << 1 = 0
  regs.setVgpr(1, 0, 0xFFFFFFFF);
  regs.setVgpr(2, 0, 1);
  regs.setVgpr(3, 0, 1);

  tryExecute(regs, "v_add_lshl_u32 v0, v1, v2, v3");
  EXPECT_EQ(regs.getVgpr(0, 0), 0);

  // Case 3: Masking Shift Amount
  // Shift by 33 should act like shift by 1 (33 & 0x1F = 1)
  // (1 + 1) << 33 -> 2 << 1 = 4
  regs.setVgpr(1, 0, 1);
  regs.setVgpr(2, 0, 1);
  regs.setVgpr(3, 0, 33); // 0x21, 0x21 & 0x1F = 1

  tryExecute(regs, "v_add_lshl_u32 v0, v1, v2, v3");
  EXPECT_EQ(regs.getVgpr(0, 0), 4);
}

TEST(Instructions, VOrB32Sdwa_Foundation) {
  // Setup: 4 VGPRs, 4 SGPRs, Wave32
  Workgroup wg;
  Wave regs(4, 4, 32, wg);

  // Initial State
  // v1 = 0x11112222
  // s0 = 0x00003333
  regs.setVgpr(1, 0, 0x11112222);
  regs.setSgpr(0, 0x00003333);

  // TEST 1: PAD Behavior (Basic)
  // v_or_b32_sdwa v1, s0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD ...
  //
  // Logic:
  //   Result = 0x00003333 | 0x11112222 = 0x11113333
  //   Slice  = Lower 16 bits (0x3333) -> Shift to WORD_1 (0x33330000)
  //   PAD    = Zero rest -> Final: 0x33330000
  tryExecute(regs, "v_or_b32_sdwa v1, s0, v1 dst_sel:WORD_1 "
                   "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD");
  EXPECT_EQ(regs.getVgpr(1, 0), 0x33330000);

  // TEST 2: PRESERVE Behavior
  // Reset v1 to original state
  regs.setVgpr(1, 0, 0x11112222);

  // Logic:
  //   Result = 0x11113333 -> Slice 0x3333 -> Shift 0x33330000
  //   PRESERVE = Keep lower 16 of old v1 (0x2222)
  //   Final = 0x33332222
  tryExecute(regs, "v_or_b32_sdwa v1, s0, v1 dst_sel:WORD_1 "
                   "dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD");
  EXPECT_EQ(regs.getVgpr(1, 0), 0x33332222);
}

TEST(Instructions, VOrB32Sdwa_LLVM_Patterns) {
  Workgroup wg;
  Wave regs(64, 4, 1, wg); // 1 active lane

  // Inputs matching HIP test
  // A (v22/v4/v44/v5) = 0x12345678
  // B (v3/v2/v62/v21) = 0xABCDEF90
  uint32_t valA = 0x12345678;
  uint32_t valB = 0xABCDEF90;

  // Pattern 1: Mixed DWORD / BYTE_0
  // v_or_b32_sdwa v3, v22, v3 dst_sel:DWORD ... src0_sel:DWORD src1_sel:BYTE_0
  regs.setVgpr(22, 0, valA);
  regs.setVgpr(3, 0, valB);

  // Exp: A | (B & 0xFF) = 0x12345678 | 0x90 = 0x123456F8
  tryExecute(regs, "v_or_b32_sdwa v3, v22, v3 dst_sel:DWORD "
                   "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0");
  EXPECT_EQ(regs.getVgpr(3, 0), 0x123456F8);

  // Pattern 2: Mixed WORD_0 / DWORD
  // v_or_b32_sdwa v2, v3, v2 ... src0_sel:WORD_0 src1_sel:DWORD
  // Note: v3 and v2 here refer to register INDICES, we load valA/valB fresh.
  regs.setVgpr(3, 0, valA); // src0
  regs.setVgpr(2, 0, valB); // src1

  // Exp: (A & 0xFFFF) | B = 0x5678 | 0xABCDEF90
  //                       =          0xABCDEF90
  //                                  0x00005678
  //                                  0xABCDFFF8
  tryExecute(regs, "v_or_b32_sdwa v2, v3, v2 dst_sel:DWORD "
                   "dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xABCDFFF8);

  // Pattern 3: Tied Operand + High Word Dest (The "Tied Trap")
  // v_or_b32_sdwa v4, v4, v3 dst_sel:WORD_1 ... src0_sel:DWORD src1_sel:BYTE_0
  regs.setVgpr(4, 0, valA); // src0 AND dst (old)
  regs.setVgpr(3, 0, valB); // src1

  // Calculation:
  //   Raw Res = A | (B & 0xFF) = 0x123456F8
  //   Slice   = Raw & 0xFFFF   = 0x56F8  (Lower 16 bits of result)
  //   Shift   = Slice << 16    = 0x56F80000
  //   Pad     = Zero rest      = 0x56F80000
  tryExecute(regs, "v_or_b32_sdwa v4, v4, v3 dst_sel:WORD_1 "
                   "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0");
  EXPECT_EQ(regs.getVgpr(4, 0), 0x56F80000);

  // Pattern 4: Src0 is BYTE_0 (Reverse of Pattern 1)
  // v_or_b32_sdwa v0, v44, v62 ... src0_sel:BYTE_0 src1_sel:DWORD
  regs.setVgpr(44, 0, valA);
  regs.setVgpr(62, 0, valB);

  // Exp: (A & 0xFF) | B = 0x78 | 0xABCDEF90 = 0xABCDEFF8
  tryExecute(regs, "v_or_b32_sdwa v0, v44, v62 dst_sel:DWORD "
                   "dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD");
  EXPECT_EQ(regs.getVgpr(0, 0), 0xABCDEFF8);
}

// Helper to verify packed results easily
void expectPackedFloats(const Wave &regs, int dstIdx, float expectedLo,
                        float expectedHi) {
  uint64_t raw = regs.getVgpr64(dstIdx, 0);

  // Unpack Low
  uint32_t rawLo = raw & 0xFFFFFFFF;
  float actualLo = std::bit_cast<float>(rawLo);

  // Unpack High
  uint32_t rawHi = (raw >> 32) & 0xFFFFFFFF;
  float actualHi = std::bit_cast<float>(rawHi);

  EXPECT_FLOAT_EQ(actualLo, expectedLo) << "Low word mismatch";
  EXPECT_FLOAT_EQ(actualHi, expectedHi) << "High word mismatch";
}

TEST(Instructions, V_PK_MUL_F32_OpSel_Logic) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 6, /*sgpr*/ 0, /*wave*/ 2, wg);

  // Setup Inputs
  // Src0 (v0:1): Lo = 2.0,  Hi = 10.0
  // Src1 (v2:3): Lo = 3.0,  Hi = 100.0
  float s0_lo = 2.0f, s0_hi = 10.0f;
  float s1_lo = 3.0f, s1_hi = 100.0f;

  // Pack into uint64_t and set registers
  uint64_t pack0 =
      (static_cast<uint64_t>(std::bit_cast<uint32_t>(s0_hi)) << 32) |
      std::bit_cast<uint32_t>(s0_lo);
  uint64_t pack1 =
      (static_cast<uint64_t>(std::bit_cast<uint32_t>(s1_hi)) << 32) |
      std::bit_cast<uint32_t>(s1_lo);

  regs.setVgpr64(0, 0, pack0);
  regs.setVgpr64(2, 0, pack1);

  // Case 1: Default Behavior
  // Implied: op_sel:[0,0] (Lo uses Lo)
  // Implied: op_sel_hi:[1,1] (Hi uses Hi)
  // Result: Lo = 2 * 3 = 6
  //         Hi = 10 * 100 = 1000
  tryExecute(regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3]");
  expectPackedFloats(regs, 4, 6.0f, 1000.0f);

  // Case 2: Broadcast Low Scalar (The MI300 Kernel Case)
  // op_sel_hi:[0,1] -> "For the High calc, use Src0.Lo (0) and Src1.Hi (1)"
  // Result: Lo = 2 * 3 = 6   (Default)
  //         Hi = 2 * 100 = 200 (Broadcast Src0.Lo)
  tryExecute(regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel_hi:[0,1]");
  expectPackedFloats(regs, 4, 6.0f, 200.0f);

  // Case 3: Mixed Selection (Cross Multiply)
  // op_sel:[1,0] -> "For Low calc, use Src0.Hi (1) and Src1.Lo (0)"
  // Result: Lo = 10 * 3 = 30
  //         Hi = 10 * 100 = 1000 (Default)
  tryExecute(regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel:[1,0]");
  expectPackedFloats(regs, 4, 30.0f, 1000.0f);

  // Case 4: Full Swap
  // Low uses High inputs: [1,1]
  // High uses Low inputs: [0,0]
  // Result: Lo = 10 * 100 = 1000
  //         Hi = 2 * 3 = 6
  tryExecute(
      regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] op_sel:[1,1] op_sel_hi:[0,0]");
  expectPackedFloats(regs, 4, 1000.0f, 6.0f);
}

TEST(Instructions, V_PK_MUL_F32_Negation) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 6, /*sgpr*/ 0, /*wave*/ 64, wg);

  // Setup Inputs: 2.0 and 3.0 (Packed logic same as above)
  // Src0 (v0:1): 2.0, 2.0
  // Src1 (v2:3): 3.0, 3.0
  uint64_t pack = (static_cast<uint64_t>(std::bit_cast<uint32_t>(2.0f)) << 32) |
                  std::bit_cast<uint32_t>(2.0f);
  regs.setVgpr64(0, 0, pack);

  pack = (static_cast<uint64_t>(std::bit_cast<uint32_t>(3.0f)) << 32) |
         std::bit_cast<uint32_t>(3.0f);
  regs.setVgpr64(2, 0, pack);

  // Case: Negate Low Source 0
  // neg_lo:[1,0] -> Negate Src0 for the Low calculation only
  // Lo = (-2.0) * 3.0 = -6.0
  // Hi = 2.0 * 3.0 = 6.0 (Default)
  tryExecute(regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] neg_lo:[1,0]");
  expectPackedFloats(regs, 4, -6.0f, 6.0f);

  // Case: Negate High Source 1
  // neg_hi:[0,1] -> Negate Src1 for the High calculation only
  // Lo = 2.0 * 3.0 = 6.0
  // Hi = 2.0 * (-3.0) = -6.0
  tryExecute(regs, "v_pk_mul_f32 v[4:5], v[0:1], v[2:3] neg_hi:[0,1]");
  expectPackedFloats(regs, 4, 6.0f, -6.0f);
}

TEST(Instructions, V_FMAC_F32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 64, wg);

  // Setup Accumulator (Dst/v0) = 10.0
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(10.0f));

  // Setup Src0 (v1) = 2.0
  regs.setVgpr(1, 0, std::bit_cast<uint32_t>(2.0f));

  // Setup Src1 (v2) = 3.0
  regs.setVgpr(2, 0, std::bit_cast<uint32_t>(3.0f));

  // Execute: v0 = v1 * v2 + v0
  // Expected: (2.0 * 3.0) + 10.0 = 16.0
  tryExecute(regs, "v_fmac_f32 v0, v1, v2");

  float result = std::bit_cast<float>(regs.getVgpr(0, 0));
  EXPECT_FLOAT_EQ(result, 16.0f);
}

TEST(Instructions, V_CMPX_EQ_U32_BasicLogic) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64, wg);

  // Setup: Initialize EXEC to all 1s (all lanes active)
  regs.setExecU64(0xFFFFFFFFFFFFFFFF);

  // Setup Inputs
  // v0: [0, 1, 2, 3, 0, 1, 2, 3, ...]
  // v1: 2 (Scalar comparison value)
  for (int i = 0; i < 64; ++i) {
    regs.setVgpr(0, i, i % 4);
    regs.setVgpr(1, i, 2);
  }

  // Execute: v_cmpx_eq_u32 vcc, v0, v1
  // Should set EXEC[i] = 1 IF (v0[i] == 2)
  // Lanes 0,1,3 -> 0
  // Lane 2      -> 1
  // Expected pattern: 0010 0010 ... (0x2 repeating)
  tryExecute(regs, "v_cmpx_eq_u32 vcc, v0, v1");

  // Hex digit '2' is binary 0010, so we expect 0x2222...
  // uint64_t expectedMask = 2*0x2222222222222222ULL;
  uint64_t expectedMask =
      0x4444444444444444ULL; // Corrected pattern for 64 lanes
  EXPECT_EQ(regs.getExecU64(), expectedMask);

  // VCC should also reflect the new EXEC mask
  EXPECT_EQ(regs.getVccU64(), expectedMask);
}

TEST(Instructions, V_CMPX_EQ_U32_ExecMaskingBehavior) {
  // Verifies v_cmpx ANDs with the current EXEC mask rather than
  // overwriting it (essential for nested control flow).
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64, wg);

  // 2. Setup Inputs: ALL lanes match
  // v0 = 5, v1 = 5
  for (int i = 0; i < 64; ++i) {
    regs.setVgpr(0, i, 5);
    regs.setVgpr(1, i, 5);
  }

  // 1. Setup Initial EXEC: Only LOWER 32 lanes are active
  regs.setExecU64(0x00000000FFFFFFFF);

  // 3. Execute comparison
  // If Assign: EXEC becomes 0xFFFFFFFFFFFFFFFF (All lanes revived) -> WRONG
  // If AND:    EXEC becomes 0x00000000FFFFFFFF (Upper lanes stay dead) ->
  // CORRECT
  tryExecute(regs, "v_cmpx_eq_u32 vcc, v0, v1");

  EXPECT_EQ(regs.getExecU64(), 0x00000000FFFFFFFFULL)
      << "V_CMPX must perform (Result & Old_EXEC). Threads already masked off "
         "must stay dead.";

  EXPECT_EQ(regs.getVccU64(), 0x00000000FFFFFFFFULL);
}

TEST(Instructions, V_CMPX_EQ_U32_WithScalars) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64, wg);
  regs.setExecU64(0xFFFFFFFFFFFFFFFF);

  // Test Scalar Operand: v_cmpx_eq_u32 vcc, s0, v0
  regs.setSgpr(0, 42);
  for (int i = 0; i < 64; ++i) {
    // Even lanes = 42 (Match), Odd lanes = 99 (No Match)
    regs.setVgpr(0, i, (i % 2 == 0) ? 42 : 99);
  }

  tryExecute(regs, "v_cmpx_eq_u32 vcc, s0, v0");

  // Expect: 0101 0101 ... (0x5)
  // Bit 0 is 1 because Lane 0 matches.
  EXPECT_EQ(regs.getExecU64(), 0x5555555555555555ULL);
}

TEST(Instructions, V_ADD_CO_U32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 64, wg);

  // Lane 0: Simple Addition (10 + 20 = 30)
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 20);

  // Lane 1: Overflow/Carry Generation (UINT_MAX + 1 = 0)
  regs.setVgpr(0, 1, 0xFFFFFFFF);
  regs.setVgpr(1, 1, 1);

  // Lane 2: Boundary Case (UINT_MAX + UINT_MAX = -2)
  regs.setVgpr(0, 2, 0xFFFFFFFF);
  regs.setVgpr(1, 2, 0xFFFFFFFF);

  // Ensure VCC is clean
  regs.setVccU64(0);

  // Execute: v2 = v0 + v1, writes carry to vcc
  tryExecute(regs, "v_add_co_u32 v2, vcc, v0, v1");

  // Check Lane 0
  EXPECT_EQ(regs.getVgpr(2, 0), 30);

  // Check Lane 1
  EXPECT_EQ(regs.getVgpr(2, 1), 0);

  // Check Lane 2
  EXPECT_EQ(regs.getVgpr(2, 2), 0xFFFFFFFE);

  // Check VCC (Carry bits)
  // Lane 0: 0
  // Lane 1: 1
  // Lane 2: 1
  // Expected VCC = ...000110 (Binary) = 6
  EXPECT_EQ(regs.getVccU64() & 0x7, 6);
}

TEST(Instructions, V_ADDC_CO_U32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 64, wg);

  // Lane 0: No Carry In, No Carry Out (10 + 20 + 0 = 30)
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 20);

  // Lane 1: Carry In, No Carry Out (10 + 20 + 1 = 31)
  regs.setVgpr(0, 1, 10);
  regs.setVgpr(1, 1, 20);

  // Lane 2: Carry In AND Carry Out (Max + 0 + 1 = 0 + Carry)
  regs.setVgpr(0, 2, 0xFFFFFFFF);
  regs.setVgpr(1, 2, 0);

  // Setup Input VCC: Set bits 1 and 2 to '1' (Carry In for Lanes 1 & 2)
  regs.setVccU64(0b110);

  // Execute: v2 = v0 + v1 + vcc(in), writes carry to vcc(out)
  tryExecute(regs, "v_addc_co_u32 v2, vcc, v0, v1, vcc");

  // Check Lane 0
  EXPECT_EQ(regs.getVgpr(2, 0), 30);

  // Check Lane 1
  EXPECT_EQ(regs.getVgpr(2, 1), 31);

  // Check Lane 2
  EXPECT_EQ(regs.getVgpr(2, 2), 0);

  // Check Output VCC (Carry Out)
  // Lane 0: 0
  // Lane 1: 0
  // Lane 2: 1
  // Expected VCC = ...000100 = 4
  EXPECT_EQ(regs.getVccU64() & 0x7, 4);
}

TEST(Instructions, V_LSHL_ADD_U32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 64, wg);

  // Lane 0: Standard case
  // (5 << 2) + 10 = 20 + 10 = 30
  regs.setVgpr(0, 0, 5);  // Base
  regs.setVgpr(1, 0, 2);  // Shift
  regs.setVgpr(2, 0, 10); // Offset

  // Lane 1: Overflow case (shift pushes bits out)
  // (0xFFFFFFFF << 1) + 5 = 0xFFFFFFFE + 5 = 3
  regs.setVgpr(0, 1, 0xFFFFFFFF);
  regs.setVgpr(1, 1, 1);
  regs.setVgpr(2, 1, 5);

  tryExecute(regs, "v_lshl_add_u32 v3, v0, v1, v2");

  // Check Lane 0
  EXPECT_EQ(regs.getVgpr(3, 0), 30);

  // Check Lane 1
  EXPECT_EQ(regs.getVgpr(3, 1), 3);
}

TEST(Instructions, V_READFIRSTLANE_B32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 1, /*wave*/ 64, wg);

  // Setup distinct values in v0 for each lane
  for (int i = 0; i < 64; ++i) {
    regs.setVgpr(0, i, 100 + i);
  }

  // Case 1: Full EXEC mask (All lanes active)
  // Should read Lane 0 -> 100
  regs.setExecU64(0xFFFFFFFFFFFFFFFF);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  EXPECT_EQ(regs.getSgpr(0), 100);

  // Case 2: Partial EXEC mask (Lanes 0-3 inactive)
  // First active lane is 4 -> Should read 104
  regs.setExecU64(0xFFFFFFFFFFFFFFF0);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  EXPECT_EQ(regs.getSgpr(0), 104);

  // Case 3: Single lane active (Lane 63)
  // Should read 163
  regs.setExecU64(0x8000000000000000);
  tryExecute(regs, "v_readfirstlane_b32 s0, v0");
  EXPECT_EQ(regs.getSgpr(0), 163);
}

// Helper to simplify testing conversions
// InT: C++ type of input (e.g., uint32_t, float, double)
// OutT: C++ type of output
template <typename InT, typename OutT>
void runVectorConvertTest(Wave &regs, std::string opName, InT inputVal,
                          OutT expectedVal) {
  // 1. Setup Input (Handle 32-bit vs 64-bit registers)
  if constexpr (sizeof(InT) == 8) {
    regs.setVgpr64(0, 0, std::bit_cast<uint64_t>(inputVal));
  } else {
    regs.setVgpr(0, 0, std::bit_cast<uint32_t>(inputVal));
  }

  // 2. Execute
  // Dst = v2 (or v[2:3]), Src = v0 (or v[0:1])
  std::string asmLine = opName + " ";
  if constexpr (sizeof(OutT) == 8) {
    asmLine += "v[2:3], ";
  } else {
    asmLine += "v2, ";
  }
  if constexpr (sizeof(InT) == 8) {
    asmLine += "v[0:1]";
  } else {
    asmLine += "v0";
  }

  tryExecute(regs, asmLine);

  // 3. Verify Output
  OutT actual;
  if constexpr (sizeof(OutT) == 8) {
    uint64_t raw = regs.getVgpr64(2, 0);
    actual = std::bit_cast<OutT>(raw);
  } else {
    uint32_t raw = regs.getVgpr(2, 0);
    actual = std::bit_cast<OutT>(raw);
  }

  // Use GTest expectations
  if constexpr (std::is_floating_point_v<OutT>) {
    if constexpr (sizeof(OutT) == 8) {
      EXPECT_DOUBLE_EQ(actual, expectedVal) << opName;
    } else {
      EXPECT_FLOAT_EQ(actual, expectedVal) << opName;
    }
  } else {
    EXPECT_EQ(actual, expectedVal) << opName;
  }
}

TEST(Instructions, CVT_F32_U32_FixVerification) {
  Workgroup wg;
  Wave regs(4, 4, 1, wg);

  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 16, 16.0f);

  // Edge case: 0 -> 0.0f
  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 0, 0.0f);

  // Edge case: Large Integer
  // 16777215 is the largest integer exactly representable in float without gaps
  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 16777215,
                                        16777215.0f);
}

TEST(Instructions, CVT_U32_F32) {
  Workgroup wg;
  Wave regs(4, 4, 64, wg);

  // Standard cases
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 16.0f, 16);
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 0.0f, 0);

  // Truncation check (standard C++ behavior for static_cast)
  // 16.9 -> 16
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 16.9f, 16);
}

TEST(Instructions, CVT_DoublePrecision) {
  Workgroup wg;
  Wave regs(4, 4, 64, wg);

  // v_cvt_f64_u32 (u32 -> double)
  runVectorConvertTest<uint32_t, double>(regs, "v_cvt_f64_u32", 1, 1.0);
  runVectorConvertTest<uint32_t, double>(regs, "v_cvt_f64_u32", 0xFFFFFFFF,
                                         4294967295.0);

  // v_cvt_u32_f64 (double -> u32)
  runVectorConvertTest<double, uint32_t>(regs, "v_cvt_u32_f64", 12345.6789,
                                         12345);

  // v_cvt_f64_f32 (float -> double)
  runVectorConvertTest<float, double>(regs, "v_cvt_f64_f32", 1.5f, 1.5);

  // v_cvt_f32_f64 (double -> float)
  runVectorConvertTest<double, float>(regs, "v_cvt_f32_f64", 1.5, 1.5f);
}

TEST(Instructions, S_MOVK_I32_SignExtension) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  // The source must be a literal in the assembly string.

  // Case 1: Negative 1 (0xFFFF represents -1 in 16-bit signed)
  // Result should be sign-extended to 32-bit -1 (0xFFFFFFFF)
  tryExecute(regs, "s_movk_i32 s1, 0xFFFF");
  EXPECT_EQ(regs.getSgpr(1), 0xFFFFFFFF);

  // Case 2: Positive 1
  tryExecute(regs, "s_movk_i32 s1, 1");
  EXPECT_EQ(regs.getSgpr(1), 1);

  // Case 3: Positive Limit (0x7FFF = 32767)
  tryExecute(regs, "s_movk_i32 s1, 0x7FFF");
  EXPECT_EQ(regs.getSgpr(1), 32767);

  // Case 4: Negative Limit (0x8000 = -32768 in 16-bit signed)
  // Result should be sign-extended to 0xFFFF8000
  tryExecute(regs, "s_movk_i32 s1, 0x8000");
  EXPECT_EQ(regs.getSgpr(1), 0xFFFF8000);
}

TEST(Instructions, MOV_Ops) {
  Workgroup wg;
  Wave regs(4, 4, 64, wg);

  // v_mov_b32
  runVectorConvertTest<uint32_t, uint32_t>(regs, "v_mov_b32", 0xDEADBEEF,
                                           0xDEADBEEF);

  // v_mov_b64 (Testing 64-bit vector move)
  runVectorConvertTest<uint64_t, uint64_t>(
      regs, "v_mov_b64", 0xCAFEBABE12345678, 0xCAFEBABE12345678);
}

TEST(Instructions, VMadU64U32) {
  Workgroup wg;
  Wave regs(10, 10, 10, wg);

  // Case 1: Simple multiply-add, no overflow
  // 3 * 5 + 10 = 25
  regs.setVgpr(0, 0, 3);    // S0 = 3
  regs.setVgpr(1, 0, 5);    // S1 = 5
  regs.setVgpr64(2, 0, 10); // S2 = 10
  regs.setSgpr64(0, 0);     // Clear carry destination
  tryExecute(regs, "v_mad_u64_u32 v[4:5], s[0:1], v0, v1, v[2:3]");
  EXPECT_EQ(regs.getVgpr64(4, 0), 25ULL);
  // No overflow, so carry bit for lane 0 should be 0
  EXPECT_EQ(regs.getSgpr64(0) & 1ULL, 0ULL);

  // Case 2: Large multiply that produces a 64-bit result, no 65-bit overflow
  // 0xFFFFFFFF * 0xFFFFFFFF + 0 = 0xFFFFFFFE00000001
  regs.setVgpr(0, 0, 0xFFFFFFFF);
  regs.setVgpr(1, 0, 0xFFFFFFFF);
  regs.setVgpr64(2, 0, 0);
  regs.setSgpr64(0, 0);
  tryExecute(regs, "v_mad_u64_u32 v[4:5], s[0:1], v0, v1, v[2:3]");
  EXPECT_EQ(regs.getVgpr64(4, 0), 0xFFFFFFFE00000001ULL);
  EXPECT_EQ(regs.getSgpr64(0) & 1ULL, 0ULL);

  // Case 3: 65-bit overflow (carry = 1)
  // 0xFFFFFFFF * 0xFFFFFFFF + 0xFFFFFFFFFFFFFFFF
  // = 0xFFFFFFFE00000001 + 0xFFFFFFFFFFFFFFFF = 0x1_FFFFFFFE00000000
  // Low 64 bits = 0xFFFFFFFE00000000, carry = 1
  regs.setVgpr(0, 0, 0xFFFFFFFF);
  regs.setVgpr(1, 0, 0xFFFFFFFF);
  regs.setVgpr64(2, 0, 0xFFFFFFFFFFFFFFFFULL);
  regs.setSgpr64(0, 0);
  tryExecute(regs, "v_mad_u64_u32 v[4:5], s[0:1], v0, v1, v[2:3]");
  EXPECT_EQ(regs.getVgpr64(4, 0), 0xFFFFFFFE00000000ULL);
  EXPECT_EQ(regs.getSgpr64(0) & 1ULL, 1ULL);

  // Case 4: Literal 0 as src2 (matches the GEMM assembly pattern)
  // v_mad_u64_u32 v[4:5], s[0:1], v0, v1, 0
  regs.setVgpr(0, 0, 100);
  regs.setVgpr(1, 0, 200);
  regs.setSgpr64(0, 0);
  tryExecute(regs, "v_mad_u64_u32 v[4:5], s[0:1], v0, v1, 0");
  EXPECT_EQ(regs.getVgpr64(4, 0), 20000ULL);
  EXPECT_EQ(regs.getSgpr64(0) & 1ULL, 0ULL);

  // Case 5: Multi-lane — verify carry mask is per-lane
  // Lane 0: no overflow, Lane 1: overflow
  regs.setExecU64(0x3); // lanes 0 and 1 active
  regs.setVgpr(0, 0, 2);
  regs.setVgpr(0, 1, 0xFFFFFFFF);
  regs.setVgpr(1, 0, 3);
  regs.setVgpr(1, 1, 0xFFFFFFFF);
  regs.setVgpr64(2, 0, 0);
  regs.setVgpr64(2, 1, 0xFFFFFFFFFFFFFFFFULL);
  regs.setSgpr64(0, 0);
  tryExecute(regs, "v_mad_u64_u32 v[4:5], s[0:1], v0, v1, v[2:3]");
  // Lane 0: 2*3+0 = 6, no carry
  EXPECT_EQ(regs.getVgpr64(4, 0), 6ULL);
  // Lane 1: 0xFFFFFFFF*0xFFFFFFFF + 0xFFFFFFFFFFFFFFFF, carry = 1
  EXPECT_EQ(regs.getVgpr64(4, 1), 0xFFFFFFFE00000000ULL);
  uint64_t carryMask = regs.getSgpr64(0);
  EXPECT_EQ(carryMask & 0x1, 0ULL);   // lane 0: no carry
  EXPECT_EQ(carryMask & 0x2, 0x2ULL); // lane 1: carry
}

// --- Wave-32 register consistency tests ---

// Verify that getExecU32 and setExecU64 operate on the same underlying
// storage. The exec mask is stored as a 64-bit SGPR pair. On wave-32,
// getExecU32 reads the low 32 bits, which must be the same bits that
// setExecU64 writes to the low half.
TEST(Instructions, ExecU32U64Consistency) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 0, /*wave*/ 32, wg);

  // Wave constructor initializes exec to all-ones via setExecU64(~0ULL).
  // On wave-32, getExecU32 should see all 32 bits set.
  EXPECT_EQ(regs.getExecU32(), 0xFFFFFFFFu);

  // setExecU64 with a known pattern, verify getExecU32 sees the low bits.
  regs.setExecU64(0x00000000ABCD1234ULL);
  EXPECT_EQ(regs.getExecU32(), 0xABCD1234u);

  // setExecU32, verify getExecU64 sees it in the low bits (high bits
  // should be zero since we never set them on wave-32).
  regs.setExecU32(0x12345678);
  EXPECT_EQ(regs.getExecU64(), 0x0000000012345678ULL);
}

// Same consistency test for VCC.
TEST(Instructions, VccU32U64Consistency) {
  // Use wave-64 to test the 64-bit path, then wave-32 for 32-bit.
  Workgroup wg;
  Wave regs64(/*vgpr*/ 1, /*sgpr*/ 0, /*wave*/ 64, wg);
  regs64.setVccU64(0xDEADBEEFCAFEBABEULL);
  EXPECT_EQ(regs64.getVccU64(), 0xDEADBEEFCAFEBABEULL);

  Wave regs32(/*vgpr*/ 1, /*sgpr*/ 0, /*wave*/ 32, wg);
  regs32.setVccU32(0x12345678);
  EXPECT_EQ(regs32.getVccU32(), 0x12345678u);
}

// --- RDNA (wave-32) instruction tests ---

// v_add_co_ci_u32 is the RDNA name for v_addc_co_u32 (add with carry-in).
// Semantics: D.u32 = S0.u32 + S1.u32 + carry_in; carry_out written to sDst.
// The carry-in is read from the per-lane bit of the carry source (vcc_lo on
// wave-32). The carry-out is 1 if the 33-bit result overflows 32 bits.
TEST(Instructions, V_ADD_CO_CI_U32_Wave32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);

  // Lane 0: No carry-in, no carry-out (10 + 20 + 0 = 30)
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 20);

  // Lane 1: Carry-in set, no carry-out (10 + 20 + 1 = 31)
  regs.setVgpr(0, 1, 10);
  regs.setVgpr(1, 1, 20);

  // Lane 2: Carry-in set, carry-out generated (0xFFFFFFFF + 0 + 1 = 0)
  regs.setVgpr(0, 2, 0xFFFFFFFF);
  regs.setVgpr(1, 2, 0);

  // Set carry-in: lanes 1 and 2 have carry-in set
  regs.setVccU32(0b110);

  // v_add_co_ci_u32_e32 v2, vcc_lo, v0, v1, vcc_lo
  tryExecute(regs, "v_add_co_ci_u32 v2, vcc_lo, v0, v1, vcc_lo");

  EXPECT_EQ(regs.getVgpr(2, 0), 30u); // 10 + 20 + 0
  EXPECT_EQ(regs.getVgpr(2, 1), 31u); // 10 + 20 + 1
  EXPECT_EQ(regs.getVgpr(2, 2), 0u);  // 0xFFFFFFFF + 0 + 1 = 0 (overflow)

  // Carry-out: only lane 2 overflowed
  EXPECT_EQ(regs.getVccU32() & 0x7, 0b100u);
}

// v_add_co_u32 on wave-32: carry-out written to vcc_lo (32-bit).
TEST(Instructions, V_ADD_CO_U32_Wave32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);

  regs.setVgpr(0, 0, 0xFFFFFFFF);
  regs.setVgpr(1, 0, 1);
  regs.setVgpr(0, 1, 5);
  regs.setVgpr(1, 1, 10);

  regs.setVccU32(0);
  tryExecute(regs, "v_add_co_u32 v2, vcc_lo, v0, v1");

  EXPECT_EQ(regs.getVgpr(2, 0), 0u);  // overflow
  EXPECT_EQ(regs.getVgpr(2, 1), 15u); // no overflow
  // Lane 0 carries, lane 1 does not
  EXPECT_EQ(regs.getVccU32() & 0x3, 0b01u);
}

// v_mad_u64_u32 with null carry-out destination (wave-32).
// Semantics: D.u64 = S0.u32 * S1.u32 + S2.u64; carry discarded.
TEST(Instructions, V_MAD_U64_U32_NullCarry_Wave32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 6, /*sgpr*/ 2, /*wave*/ 32, wg);

  // Lane 0: 3 * 4 + 100 = 112
  regs.setVgpr(0, 0, 3);
  regs.setVgpr(1, 0, 4);
  regs.setVgpr64(2, 0, 100);

  tryExecute(regs, "v_mad_u64_u32 v[4:5], null, v0, v1, v[2:3]");

  EXPECT_EQ(regs.getVgpr64(4, 0), 112ULL);
}

// s_clause and s_delay_alu are scheduling hints with no semantic effect.
// They should parse and execute as no-ops.
TEST(Instructions, SchedulingHintNoOps_Wave32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 2, /*sgpr*/ 4, /*wave*/ 32, wg);

  regs.setVgpr(0, 0, 42);

  // These should not modify any state
  tryExecute(regs, "s_clause 0x1");
  tryExecute(regs, "s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | "
                   "instid1(VALU_DEP_1)");

  // Verify no state was modified
  EXPECT_EQ(regs.getVgpr(0, 0), 42u);
}

// v_cvt_f32_i32: convert signed int32 to float32.
TEST(Instructions, V_CVT_F32_I32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 2, /*sgpr*/ 0, /*wave*/ 32, wg);

  regs.setVgpr(0, 0, 42);
  tryExecute(regs, "v_cvt_f32_i32 v1, v0");
  EXPECT_EQ(std::bit_cast<float>(regs.getVgpr(1, 0)), 42.0f);

  // Negative value
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(int32_t(-7)));
  tryExecute(regs, "v_cvt_f32_i32 v1, v0");
  EXPECT_EQ(std::bit_cast<float>(regs.getVgpr(1, 0)), -7.0f);
}

// v_mad_u32_u24: D.u32 = S0[23:0] * S1[23:0] + S2.u32
// Multiply-add using the low 24 bits of each multiplicand.
TEST(Instructions, V_MAD_U32_U24) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 32, wg);

  // 3 * 4 + 10 = 22
  regs.setVgpr(0, 0, 3);
  regs.setVgpr(1, 0, 4);
  regs.setVgpr(2, 0, 10);
  tryExecute(regs, "v_mad_u32_u24 v3, v0, v1, v2");
  EXPECT_EQ(regs.getVgpr(3, 0), 22u);

  // High bits beyond [23:0] should be masked off.
  // 0xFF000003 & 0xFFFFFF = 3, 0xFF000004 & 0xFFFFFF = 4
  regs.setVgpr(0, 0, 0xFF000003);
  regs.setVgpr(1, 0, 0xFF000004);
  regs.setVgpr(2, 0, 0);
  tryExecute(regs, "v_mad_u32_u24 v3, v0, v1, v2");
  EXPECT_EQ(regs.getVgpr(3, 0), 12u);
}

// exec_lo: alias for exec on wave-32 (like vcc_lo is alias for vcc).
// s_mov_b32 s0, exec_lo should read the exec mask into an SGPR.
TEST(Instructions, ExecLo_Register) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 2, /*wave*/ 32, wg);

  // exec is initialized to all-ones. On wave-32, exec_lo = 0xFFFFFFFF.
  tryExecute(regs, "s_mov_b32 s0, exec_lo");
  EXPECT_EQ(regs.getSgpr(0), 0xFFFFFFFFu);
}

// v_cmpx_gt_i32: signed comparison that writes the exec mask.
// D[lane] = (S0 > S1), exec = result mask.
// Tests signed semantics: -1 is NOT > 5.
TEST(Instructions, V_CMPX_GT_I32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 2, /*sgpr*/ 0, /*wave*/ 32, wg);

  // Lane 0: 10 > 5 → true
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 5);
  // Lane 1: 3 > 7 → false
  regs.setVgpr(0, 1, 3);
  regs.setVgpr(1, 1, 7);
  // Lane 2: -1 (0xFFFFFFFF) > 5 → false (signed)
  regs.setVgpr(0, 2, 0xFFFFFFFF);
  regs.setVgpr(1, 2, 5);

  // 4-token form: explicit dest (CDNA style)
  tryExecute(regs, "v_cmpx_gt_i32 vcc, v0, v1");
  EXPECT_EQ(regs.getExecU64() & 0x7, 0b001ULL);
}

// v_cmpx_gt_i32 with implicit exec dest (RDNA _e64 form, suffix stripped).
// On RDNA, v_cmpx_*_e64 takes only 2 source operands and writes the
// result directly to exec (no SGPR destination). On CDNA, v_cmpx_* takes
// 3 operands (dst, src0, src1) and writes BOTH the SGPR dest AND exec.
// VOPC_Compare handles both forms by checking operand count.
TEST(Instructions, V_CMPX_GT_I32_ImplicitExec) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 2, /*sgpr*/ 0, /*wave*/ 32, wg);

  // Lane 0: 10 > 5 → true
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 5);
  // Lane 1: -1 > 5 → false (signed)
  regs.setVgpr(0, 1, 0xFFFFFFFF);
  regs.setVgpr(1, 1, 5);

  // 3-token form as produced by hipcc for RDNA (after _e64 stripping)
  tryExecute(regs, "v_cmpx_gt_i32 v0, v1");
  EXPECT_EQ(regs.getExecU64() & 0x3, 0b01ULL);
}

// v_add_nc_u32: RDNA name for v_add_u32 (add without carry).
TEST(Instructions, V_ADD_NC_U32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 7);
  tryExecute(regs, "v_add_nc_u32 v2, v0, v1");
  EXPECT_EQ(regs.getVgpr(2, 0), 17u);
}

// v_add_nc_i32: signed variant, same bit operation as v_add_nc_u32.
TEST(Instructions, V_ADD_NC_I32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);
  regs.setVgpr(0, 0, 10);
  regs.setVgpr(1, 0, 7);
  tryExecute(regs, "v_add_nc_i32 v2, v0, v1");
  EXPECT_EQ(regs.getVgpr(2, 0), 17u);

  // Negative: 5 + (-3) = 2 (using two's complement)
  regs.setVgpr(0, 0, 5);
  regs.setVgpr(1, 0, static_cast<uint32_t>(-3));
  tryExecute(regs, "v_add_nc_i32 v2, v0, v1");
  EXPECT_EQ(regs.getVgpr(2, 0), 2u);
}

// v_cmp_eq_i32: signed compare equal, sets VCC per-lane.
TEST(Instructions, V_CMP_EQ_I32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);

  // Lane 0: equal
  regs.setVgpr(0, 0, 42);
  regs.setVgpr(1, 0, 42);
  tryExecute(regs, "v_cmp_eq_i32 vcc, v0, v1");
  EXPECT_EQ(regs.getVccU32() & 1, 1u); // lane 0 set

  // Lane 0: not equal
  regs.setVgpr(1, 0, 43);
  tryExecute(regs, "v_cmp_eq_i32 vcc, v0, v1");
  EXPECT_EQ(regs.getVccU32() & 1, 0u); // lane 0 clear

  // Negative values: -1 == -1
  regs.setVgpr(0, 0, static_cast<uint32_t>(-1));
  regs.setVgpr(1, 0, static_cast<uint32_t>(-1));
  tryExecute(regs, "v_cmp_eq_i32 vcc, v0, v1");
  EXPECT_EQ(regs.getVccU32() & 1, 1u);
}

// v_alignbit_b32: D.u32 = ({S0, S1} >> S2[4:0])[31:0]
// Concatenates S0 (high 32 bits) and S1 (low 32 bits) into a 64-bit
// value, shifts right by S2[4:0], returns low 32 bits.
TEST(Instructions, V_ALIGNBIT_B32) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 32, wg);
  // {0xDEADBEEF, 0xCAFEBABE} >> 4 = 0xFDEADBEECAFEBAB >> ... low 32 =
  // 0xECAFEBAB
  regs.setVgpr(0, 0, 0xDEADBEEF); // high
  regs.setVgpr(1, 0, 0xCAFEBABE); // low
  regs.setVgpr(2, 0, 4);          // shift
  tryExecute(regs, "v_alignbit_b32 v3, v0, v1, v2");
  EXPECT_EQ(regs.getVgpr(3, 0), 0xFCAFEBABu);

  // Shift by 1: ({S0, S1} >> 1)[31:0]
  regs.setVgpr(2, 0, 1);
  tryExecute(regs, "v_alignbit_b32 v3, v0, v1, v2");
  EXPECT_EQ(regs.getVgpr(3, 0), 0xE57F5D5Fu);
}

// VOPD dual-issue: two operations separated by "::" in assembly.
// The emulator splits on "::" and executes each half sequentially.
TEST(Instructions, VOPD_DualIssue) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);
  regs.setVgpr(0, 0, 7);

  // v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v2, 2, v0
  // Should: v1 = 0, v2 = 7 << 2 = 28
  tryExecute(regs, "v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v2, 2, v0");
  EXPECT_EQ(regs.getVgpr(1, 0), 0u);
  EXPECT_EQ(regs.getVgpr(2, 0), 28u);
}

// Verify that a dual-issue line advances the PC by exactly 1, not 2.
// The :: line sets v1=42, then the next line overwrites v1=99.
// If the :: line incorrectly advances PC by 2, v_mov_b32 is skipped
// and v1 remains 42.
TEST(Instructions, VOPD_DualIssue_PCAdvancement) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 32, wg);
  regs.setVgpr(0, 0, 7);

  // Execute dual-issue, then check PC advanced by 1.
  int pcBefore = regs.getPc();
  tryExecute(regs, "v_dual_mov_b32 v1, 42 :: v_dual_mov_b32 v2, 0");
  int pcAfter = regs.getPc();

  EXPECT_EQ(regs.getVgpr(1, 0), 42u);
  EXPECT_EQ(regs.getVgpr(2, 0), 0u);
  EXPECT_EQ(pcAfter - pcBefore, 1)
      << "Dual-issue line should advance PC by 1, not 2";
}

// vcc_hi: the upper 32 bits of VCC, used on wave-64 when VCC is 64-bit.
// On wave-32, VCC fits in vcc_lo (32 bits), so vcc_hi is the adjacent SGPR.
// Writing to vcc_hi should be accessible via getSgpr at vccIndex+1.
TEST(Instructions, VccHi_Register) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 2, /*wave*/ 64, wg);

  // Write a known value to vcc_hi via s_mov_b32
  tryExecute(regs, "s_mov_b32 vcc_hi, 0xABCD0000");
  // Read vcc_hi back into an SGPR
  tryExecute(regs, "s_mov_b32 s0, vcc_hi");
  EXPECT_EQ(regs.getSgpr(0), 0xABCD0000u);
}

// exec_hi: the upper 32 bits of the exec mask (wave-64).
// On wave-64, exec is 64 bits and exec_hi is the upper half.
TEST(Instructions, ExecHi_Register) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 2, /*wave*/ 64, wg);

  // exec is initialized to all-ones on wave-64, so exec_hi = 0xFFFFFFFF.
  tryExecute(regs, "s_mov_b32 s0, exec_hi");
  EXPECT_EQ(regs.getSgpr(0), 0xFFFFFFFFu);

  // Write a new value to exec_hi
  tryExecute(regs, "s_mov_b32 exec_hi, 0x12345678");
  tryExecute(regs, "s_mov_b32 s1, exec_hi");
  EXPECT_EQ(regs.getSgpr(1), 0x12345678u);
}

// v_wmma_f32_16x16x16_bf16: D = A * B + C
// Random bf16 inputs (±1.0), verified against CPU reference matmul.
// Lane mapping (verified with AMD matrix_instruction_calculator):
//   A[row][k]: lane=row (0-15, replicated 16-31), vgpr=k/2, lo/hi bf16
//   B[k][col]: lane=col (0-15, replicated 16-31), vgpr=k/2, lo/hi bf16
//   D[row][col]: vgpr=row/2, lane=col (even rows) or col+16 (odd rows)
TEST(Instructions, WMMA_F32_16x16x16_BF16) {
  Workgroup wg;
  Wave regs(/*vgpr*/ 24, /*sgpr*/ 0, /*wave*/ 32, wg);

  // bf16 conversion helpers
  auto f32_to_bf16 = [](float f) -> uint16_t {
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(f) >> 16);
  };

  // Generate random ±1.0 matrices (exact in bf16)
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(0, 1);
  float choices[2] = {1.0f, -1.0f};

  float A[16][16], B[16][16], C[16][16], D_ref[16][16];
  for (int i = 0; i < 16; ++i) {
    for (int k = 0; k < 16; ++k) {
      A[i][k] = choices[dist(rng)];
    }
  }
  for (int k = 0; k < 16; ++k) {
    for (int j = 0; j < 16; ++j) {
      B[k][j] = choices[dist(rng)];
      C[k][j] = 0.0f;
    }
  }

  // CPU reference: D = A * B + C
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float sum = C[i][j];
      for (int k = 0; k < 16; ++k) {
        sum += A[i][k] * B[k][j];
      }
      D_ref[i][j] = sum;
    }
  }

  // Pack A into VGPRs: lane L holds row L%16, vgpr v holds k=2v (lo), k=2v+1
  // (hi). Lanes 16-31 replicate lanes 0-15.
  for (int lane = 0; lane < 32; ++lane) {
    int row = lane % 16;
    for (int v = 0; v < 8; ++v) {
      uint16_t lo = f32_to_bf16(A[row][2 * v]);
      uint16_t hi = f32_to_bf16(A[row][2 * v + 1]);
      regs.setVgpr(8 + v, lane, lo | (static_cast<uint32_t>(hi) << 16));
    }
  }

  // Pack B: lane L holds col L%16, vgpr v holds k=2v (lo), k=2v+1 (hi).
  for (int lane = 0; lane < 32; ++lane) {
    int col = lane % 16;
    for (int v = 0; v < 8; ++v) {
      uint16_t lo = f32_to_bf16(B[2 * v][col]);
      uint16_t hi = f32_to_bf16(B[2 * v + 1][col]);
      regs.setVgpr(16 + v, lane, lo | (static_cast<uint32_t>(hi) << 16));
    }
  }

  // C = 0 in D registers
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 8; ++v) {
      regs.setVgpr(v, lane, 0);
    }
  }

  tryExecute(regs,
             "v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]");

  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 8; ++v) {
      int row = (lane < 16) ? (2 * v) : (2 * v + 1);
      int col = lane % 16;
      float result = std::bit_cast<float>(regs.getVgpr(v, lane));
      EXPECT_EQ(result, D_ref[row][col])
          << "D[" << row << "][" << col << "] (lane=" << lane << ", vgpr=" << v
          << "): expected " << D_ref[row][col] << ", got " << result;
    }
  }
}

// v_perm_b32: byte permute from the 64-bit value {src0, src1}.
//
// Constructs an 8-byte input from two 32-bit sources:
//   in[0..3] = bytes of src1 (least significant first)
//   in[4..7] = bytes of src0
//
// Each byte of the selector (src2) picks one result byte:
//   0-7:   select in[sel]
//   8-11:  sign-extend in[1], in[3], in[5], in[7] respectively
//   12:    0x00
//   >=13:  0xFF
//
// Reference: AMD Instinct MI300 CDNA3 ISA, V_PERM_B32 (opcode 493).

TEST(Instructions, VPermB32_ByteSelect) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  // src0 = 0xDDCCBBAA, src1 = 0x44332211
  // in[0]=0x11, in[1]=0x22, in[2]=0x33, in[3]=0x44
  // in[4]=0xAA, in[5]=0xBB, in[6]=0xCC, in[7]=0xDD
  regs.setVgpr(0, 0, 0xDDCCBBAA); // src0
  regs.setVgpr(1, 0, 0x44332211); // src1

  // Selector: pick in[0], in[1], in[2], in[3] → 0x44332211
  regs.setSgpr(0, 0x03020100);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0x44332211u);

  // Selector: pick in[4], in[5], in[6], in[7] → 0xDDCCBBAA
  regs.setSgpr(0, 0x07060504);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xDDCCBBAA);

  // Selector: reverse bytes of src1 → in[3], in[2], in[1], in[0]
  regs.setSgpr(0, 0x00010203);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0x11223344u);

  // Selector: interleave → in[0], in[4], in[1], in[5]
  regs.setSgpr(0, 0x05010400);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xBB22AA11u);
}

TEST(Instructions, VPermB32_Constants) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  regs.setVgpr(0, 0, 0xDDCCBBAA);
  regs.setVgpr(1, 0, 0x44332211);

  // Selector 12 = 0x00, selector 13 = 0xFF
  regs.setSgpr(0, 0x0D0C0D0C);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xFF00FF00u);

  // Selector >= 13 all return 0xFF (e.g. 14, 255)
  regs.setSgpr(0, 0xFF0E0D0C);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0xFFFFFF00u);
}

TEST(Instructions, VPermB32_SignExtend) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  // src0 = 0x00800000, src1 = 0x00FF0080
  // in[0]=0x80, in[1]=0x00, in[2]=0xFF, in[3]=0x00
  // in[4]=0x00, in[5]=0x00, in[6]=0x80, in[7]=0x00
  regs.setVgpr(0, 0, 0x00800000); // src0
  regs.setVgpr(1, 0, 0x00FF0080); // src1

  // Sel 8 = sign-extend in[1] (0x00, bit 7=0 → 0x00)
  // Sel 9 = sign-extend in[3] (0x00, bit 7=0 → 0x00)
  // Sel 10 = sign-extend in[5] (0x00, bit 7=0 → 0x00)
  // Sel 11 = sign-extend in[7] (0x00, bit 7=0 → 0x00)
  regs.setSgpr(0, 0x0B0A0908);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0x00000000u);

  // Now with src1 = 0x0080FF80 so in[1]=0xFF (bit 7=1), in[3]=0x00
  // src0 = 0x0080FF00 so in[5]=0xFF (bit 7=1), in[7]=0x00
  regs.setVgpr(0, 0, 0x0080FF00); // in[4]=0x00,in[5]=0xFF,in[6]=0x80,in[7]=0x00
  regs.setVgpr(1, 0, 0x0080FF80); // in[0]=0x80,in[1]=0xFF,in[2]=0x80,in[3]=0x00

  // Sel 8 = sign-extend in[1] (0xFF, bit 7=1 → 0xFF)
  // Sel 9 = sign-extend in[3] (0x00, bit 7=0 → 0x00)
  // Sel 10 = sign-extend in[5] (0xFF, bit 7=1 → 0xFF)
  // Sel 11 = sign-extend in[7] (0x00, bit 7=0 → 0x00)
  regs.setSgpr(0, 0x0B0A0908);
  tryExecute(regs, "v_perm_b32 v2, v0, v1, s0");
  EXPECT_EQ(regs.getVgpr(2, 0), 0x00FF00FFu);
}

TEST(Instructions, V_EXP_F32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(0.0f));
  regs.setVgpr(0, 1, std::bit_cast<uint32_t>(1.0f));
  regs.setVgpr(0, 2, std::bit_cast<uint32_t>(-1.0f));
  regs.setVgpr(0, 3, std::bit_cast<uint32_t>(2.0f));
  tryExecute(regs, "v_exp_f32 v1, v0");
  // exp2(0) = 1, exp2(1) = 2, exp2(-1) = 0.5, exp2(2) = 4
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 0)), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 1)), 2.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 2)), 0.5f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 3)), 4.0f);
}

TEST(Instructions, V_RCP_F32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(1.0f));
  regs.setVgpr(0, 1, std::bit_cast<uint32_t>(2.0f));
  regs.setVgpr(0, 2, std::bit_cast<uint32_t>(0.5f));
  regs.setVgpr(0, 3, std::bit_cast<uint32_t>(-4.0f));
  tryExecute(regs, "v_rcp_f32 v1, v0");
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 0)), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 1)), 0.5f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 2)), 2.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(1, 3)), -0.25f);

  // rcp(0) = inf, rcp(inf) = 0
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(0.0f));
  regs.setVgpr(0, 1,
               std::bit_cast<uint32_t>(std::numeric_limits<float>::infinity()));
  tryExecute(regs, "v_rcp_f32 v1, v0");
  EXPECT_TRUE(std::isinf(std::bit_cast<float>(regs.getVgpr(1, 0))));
  EXPECT_EQ(std::bit_cast<float>(regs.getVgpr(1, 1)), 0.0f);
}

TEST(Instructions, V_MAX_F32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(1.0f));
  regs.setVgpr(1, 0, std::bit_cast<uint32_t>(2.0f));
  regs.setVgpr(0, 1, std::bit_cast<uint32_t>(-3.0f));
  regs.setVgpr(1, 1, std::bit_cast<uint32_t>(-1.0f));
  tryExecute(regs, "v_max_f32 v2, v0, v1");
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(2, 0)), 2.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(2, 1)), -1.0f);
}

TEST(Instructions, V_MIN_F32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(1.0f));
  regs.setVgpr(1, 0, std::bit_cast<uint32_t>(2.0f));
  regs.setVgpr(0, 1, std::bit_cast<uint32_t>(-3.0f));
  regs.setVgpr(1, 1, std::bit_cast<uint32_t>(-1.0f));
  tryExecute(regs, "v_min_f32 v2, v0, v1");
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(2, 0)), 1.0f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(2, 1)), -3.0f);
}

TEST(Instructions, V_FMA_F32) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  // v3 = v0 * v1 + v2 = 2.0 * 3.0 + 10.0 = 16.0
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(2.0f));
  regs.setVgpr(1, 0, std::bit_cast<uint32_t>(3.0f));
  regs.setVgpr(2, 0, std::bit_cast<uint32_t>(10.0f));
  tryExecute(regs, "v_fma_f32 v3, v0, v1, v2");
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(3, 0)), 16.0f);

  // -1.5 * 4.0 + 7.0 = 1.0
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(-1.5f));
  regs.setVgpr(1, 0, std::bit_cast<uint32_t>(4.0f));
  regs.setVgpr(2, 0, std::bit_cast<uint32_t>(7.0f));
  tryExecute(regs, "v_fma_f32 v3, v0, v1, v2");
  EXPECT_FLOAT_EQ(std::bit_cast<float>(regs.getVgpr(3, 0)), 1.0f);
}

TEST(Instructions, V_CVT_U32_F32_Clamping) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);
  // Negative -> 0
  regs.setVgpr(0, 0, std::bit_cast<uint32_t>(-1.0f));
  // NaN -> 0
  regs.setVgpr(
      0, 1, std::bit_cast<uint32_t>(std::numeric_limits<float>::quiet_NaN()));
  // Normal
  regs.setVgpr(0, 2, std::bit_cast<uint32_t>(42.7f));
  // Zero
  regs.setVgpr(0, 3, std::bit_cast<uint32_t>(0.0f));
  tryExecute(regs, "v_cvt_u32_f32 v1, v0");
  EXPECT_EQ(regs.getVgpr(1, 0), 0u);
  EXPECT_EQ(regs.getVgpr(1, 1), 0u);
  EXPECT_EQ(regs.getVgpr(1, 2), 42u);
  EXPECT_EQ(regs.getVgpr(1, 3), 0u);
}

// v_pack_b32_f16: D[15:0] = S0[15:0], D[31:16] = S1[15:0].
// Packs two f16 values into one 32-bit register.
TEST(Instructions, VPackB32F16) {
  Workgroup wg;
  Wave regs(4, 4, 4, wg);

  uint16_t f16_one = raceemulator::floatToF16(1.0f);
  uint16_t f16_neg2 = raceemulator::floatToF16(-2.0f);

  // v0 = f16(1.0) in low 16 bits, v1 = f16(-2.0) in low 16 bits.
  regs.setVgpr(0, 0, f16_one);
  regs.setVgpr(1, 0, f16_neg2);

  tryExecute(regs, "v_pack_b32_f16 v2, v0, v1");

  uint32_t packed = regs.getVgpr(2, 0);
  EXPECT_EQ(packed & 0xFFFF, f16_one);
  EXPECT_EQ(packed >> 16, f16_neg2);

  // Verify round-trip: unpack and convert back to float.
  EXPECT_EQ(raceemulator::f16ToFloat(packed & 0xFFFF), 1.0f);
  EXPECT_EQ(raceemulator::f16ToFloat(packed >> 16), -2.0f);

  // Upper 16 bits of sources should be ignored.
  regs.setVgpr(0, 1, 0xDEAD0000u | f16_one);
  regs.setVgpr(1, 1, 0xBEEF0000u | f16_neg2);
  tryExecute(regs, "v_pack_b32_f16 v2, v0, v1");
  packed = regs.getVgpr(2, 1);
  EXPECT_EQ(packed & 0xFFFF, f16_one);
  EXPECT_EQ(packed >> 16, f16_neg2);
}

// ============================================================================
// v_mfma_f32_16x16x32_f16 (gfx950, wave-64, K=32)
//
// D[16x16] = A[16x32] * B[32x16] + C[16x16], all f32 output, f16 inputs.
// Each lane provides 8 f16 values (4 VGPRs) for A and B. The 64 lanes form
// 4 groups of 16, each group covering 8 K-elements → 4 × 8 = 32.
// ============================================================================

TEST(Instructions, Mfma_F32_16x16x32_F16) {
  Workgroup wg;
  // 4 input + 4 output VGPRs per operand, plus headroom.
  Wave wave(/*vgprCount=*/20, /*sgprCount=*/4, /*waveSize=*/64, wg);

  // Set up A and B as all-ones in f16. With K=32, each output element
  // should be 1.0 * 1.0 * 32 = 32.0.
  uint16_t f16_one = raceemulator::floatToF16(1.0f);
  uint32_t packed_ones = (static_cast<uint32_t>(f16_one) << 16) | f16_one;

  for (int lane = 0; lane < 64; ++lane) {
    // A in v[0:3], B in v[4:7], C=0 (literal).
    for (int r = 0; r < 4; ++r) {
      wave.setVgpr(r, lane, packed_ones);     // A
      wave.setVgpr(4 + r, lane, packed_ones); // B
    }
  }

  tryExecute(wave, "v_mfma_f32_16x16x32_f16 v[8:11], v[0:3], v[4:7], 0");

  // Every element of the 16×16 output should be 32.0.
  for (int lane = 0; lane < 64; ++lane) {
    for (int i = 0; i < 4; ++i) {
      float val = std::bit_cast<float>(wave.getVgpr(8 + i, lane));
      EXPECT_FLOAT_EQ(val, 32.0f) << "lane=" << lane << " vgpr=" << (8 + i);
    }
  }
}

// Same test with non-trivial values: A=identity-like, B=known, verify
// specific output elements.
TEST(Instructions, Mfma_F32_16x16x32_F16_NonTrivial) {
  Workgroup wg;
  Wave wave(/*vgprCount=*/20, /*sgprCount=*/4, /*waveSize=*/64, wg);

  // Zero all inputs first.
  for (int lane = 0; lane < 64; ++lane) {
    for (int r = 0; r < 12; ++r) {
      wave.setVgpr(r, lane, 0);
    }
  }

  // Set A[0][0] = 3.0, B[0][0] = 5.0, C=0.
  // With the lane mapping: A row 0 is lane % 16 == 0, K-elem 0 is
  // the low half of the first VGPR in group 0 (lane / 16 == 0).
  // So lane 0, VGPR 0, low 16 bits.
  uint16_t f16_3 = raceemulator::floatToF16(3.0f);
  uint16_t f16_5 = raceemulator::floatToF16(5.0f);
  wave.setVgpr(0, /*lane=*/0, static_cast<uint32_t>(f16_3)); // A[0][0] = 3.0
  wave.setVgpr(4, /*lane=*/0, static_cast<uint32_t>(f16_5)); // B[0][0] = 5.0

  tryExecute(wave, "v_mfma_f32_16x16x32_f16 v[8:11], v[0:3], v[4:7], 0");

  // D[0][0] = A[0][0] * B[0][0] = 3.0 * 5.0 = 15.0.
  // D[0][0] maps to: row=0, col=0 → lane with col=lane%16=0, row=4*(lane/16)+i.
  // row=0 → lane/16=0, i=0 → lane=0, VGPR offset=0.
  float d00 = std::bit_cast<float>(wave.getVgpr(8, /*lane=*/0));
  EXPECT_FLOAT_EQ(d00, 15.0f);

  // D[1][0] should be 0 (only A[0][0] was set, not A[1][0]).
  // row=1 → lane/16=0, i=1 → lane=0, VGPR offset=1.
  float d10 = std::bit_cast<float>(wave.getVgpr(9, /*lane=*/0));
  EXPECT_FLOAT_EQ(d10, 0.0f);

  // D[0][1] should be 0 (B[0][1] = 0).
  // col=1 → lane=1, row=0 → i=0.
  float d01 = std::bit_cast<float>(wave.getVgpr(8, /*lane=*/1));
  EXPECT_FLOAT_EQ(d01, 0.0f);
}
