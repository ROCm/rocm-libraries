#include "race-emulator/Wave.h"
#include <bit>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

namespace {

using namespace raceemulator;

void tryExecute(Wave &regs, const std::string &line) {
  regs.tryExecute(line, false);
}

} // namespace

TEST(GpuEmulatorUnitTestsZero, InstructionTestZero) {
  Wave regs(10, 10, 16);
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

TEST(GpuEmulatorUnitTests, MixedOperandMath) {
  Wave regs(/*vgprCount*/ 16, /*sgprCount*/ 16, /*waveSize*/ 32);

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

TEST(GpuEmulatorUnitTests, LaneIndependence) {
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 4);

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

TEST(GpuEmulatorUnitTests, Registers64Bit) {
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1);

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

TEST(GpuEmulatorUnitTests, Conversions) {
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1);

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

TEST(GpuEmulatorUnitTests, VLshlOrB32) {
  Wave regs(/*vgprCount*/ 4, /*sgprCount*/ 4, /*waveSize*/ 1);

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

TEST(GpuEmulatorUnitTests, VCmpGtI32E32) {
  // Example: v_cmp_gt_i32_e32 vcc, s5, v0
  Wave regs(/*vgprCount*/ 2, /*sgprCount*/ 6, /*waveSize*/ 64);
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

TEST(GpuEmulatorUnitTests, VReadFirstLane) {
  // Setup: 1 VGPR, 1 SGPR, WaveSize 64
  Wave regs(1, 1, 64);

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

TEST(GpuEmulatorUnitTests, VAshrRevI32_LLVMExamples) {
  Wave regs(4, 4, 4);

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

TEST(GpuEmulatorUnitTests, VBfeU32) {
  Wave regs(4, 4, 4);

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
TEST(GpuEmulatorUnitTests, VAdd3U32) {
  Wave regs(4, 4, 4); // 4 VGPR, 4 SGPRs

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

TEST(GpuEmulatorUnitTests, VCmpU32Sdwa_Unified) {
  Wave regs(4, 2, 1);

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

TEST(GpuEmulatorUnitTests, VCndMask_MultiLane) {
  // Setup:
  // 3 VGPRs (v0 dest, v1 src0, v2 src1)
  // 2 SGPRs (s[0:1] for VCC)
  // WaveSize = 2 (We need at least 2 lanes to test bit 0 vs bit 1)
  Wave regs(3, 2, 2);

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

TEST(GpuEmulatorUnitTests, VCndMask_RespectsExec) {
  Wave regs(3, 2, 64); // 64 lanes

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

TEST(GpuEmulatorUnitTests, VAddLshlU32) {
  Wave regs(4, 0, 1); // 4 VGPRs, 0 SGPRs, WaveSize 1

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

TEST(GpuEmulatorUnitTests, VOrB32Sdwa_Foundation) {
  // Setup: 4 VGPRs, 4 SGPRs, Wave32
  Wave regs(4, 4, 32);

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

TEST(GpuEmulatorUnitTests, VOrB32Sdwa_LLVM_Patterns) {
  Wave regs(64, 4, 1); // 1 active lane

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

TEST(GpuEmulatorUnitTests, V_PK_MUL_F32_OpSel_Logic) {
  Wave regs(/*vgpr*/ 6, /*sgpr*/ 0, /*wave*/ 2);

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

TEST(GpuEmulatorUnitTests, V_PK_MUL_F32_Negation) {
  Wave regs(/*vgpr*/ 6, /*sgpr*/ 0, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTests, V_FMAC_F32) {
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTests, V_CMPX_EQ_U32_BasicLogic) {
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTests, V_CMPX_EQ_U32_ExecMaskingBehavior) {
  // Verifies v_cmpx ANDs with the current EXEC mask rather than
  // overwriting it (essential for nested control flow).
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTests, V_CMPX_EQ_U32_WithScalars) {
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 4, /*wave*/ 64);
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

TEST(GpuEmulatorUnitTestsPartTwoPartTwo, V_ADD_CO_U32) {
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTestsPartTwoPartTwo, V_ADDC_CO_U32) {
  Wave regs(/*vgpr*/ 3, /*sgpr*/ 0, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTestsPartTwoPartTwo, V_LSHL_ADD_U32) {
  Wave regs(/*vgpr*/ 4, /*sgpr*/ 0, /*wave*/ 64);

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

TEST(GpuEmulatorUnitTestsPartTwoPartTwo, V_READFIRSTLANE_B32) {
  Wave regs(/*vgpr*/ 1, /*sgpr*/ 1, /*wave*/ 64);

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
  if constexpr (sizeof(OutT) == 8)
    asmLine += "v[2:3], ";
  else
    asmLine += "v2, ";
  if constexpr (sizeof(InT) == 8)
    asmLine += "v[0:1]";
  else
    asmLine += "v0";

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
    if constexpr (sizeof(OutT) == 8)
      EXPECT_DOUBLE_EQ(actual, expectedVal) << opName;
    else
      EXPECT_FLOAT_EQ(actual, expectedVal) << opName;
  } else {
    EXPECT_EQ(actual, expectedVal) << opName;
  }
}

TEST(GpuEmulatorUnitTestsPartTwo, CVT_F32_U32_FixVerification) {
  Wave regs(4, 4, 1);

  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 16, 16.0f);

  // Edge case: 0 -> 0.0f
  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 0, 0.0f);

  // Edge case: Large Integer
  // 16777215 is the largest integer exactly representable in float without gaps
  runVectorConvertTest<uint32_t, float>(regs, "v_cvt_f32_u32", 16777215,
                                        16777215.0f);
}

TEST(GpuEmulatorUnitTestsPartTwo, CVT_U32_F32) {
  Wave regs(4, 4, 64);

  // Standard cases
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 16.0f, 16);
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 0.0f, 0);

  // Truncation check (standard C++ behavior for static_cast)
  // 16.9 -> 16
  runVectorConvertTest<float, uint32_t>(regs, "v_cvt_u32_f32", 16.9f, 16);
}

TEST(GpuEmulatorUnitTestsPartTwo, CVT_DoublePrecision) {
  Wave regs(4, 4, 64);

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

TEST(GpuEmulatorUnitTestsPartTwo, S_MOVK_I32_SignExtension) {
  Wave regs(4, 4, 4);

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

TEST(GpuEmulatorUnitTestsPartTwo, MOV_Ops) {
  Wave regs(4, 4, 64);

  // v_mov_b32
  runVectorConvertTest<uint32_t, uint32_t>(regs, "v_mov_b32", 0xDEADBEEF,
                                           0xDEADBEEF);

  // v_mov_b64 (Testing 64-bit vector move)
  runVectorConvertTest<uint64_t, uint64_t>(
      regs, "v_mov_b64", 0xCAFEBABE12345678, 0xCAFEBABE12345678);
}

TEST(GpuEmulatorUnitTests, VMadU64U32) {
  Wave regs(10, 10, 10);

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
