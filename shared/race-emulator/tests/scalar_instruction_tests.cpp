// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Wave.h"
#include "race-emulator/Workgroup.h"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace raceemulator;

// MACRO: Creates a unique GTest for an instruction
// ARGS:
//   TEST_NAME: Unique name for the test
//   ASM_OP:    The assembly string (e.g., "s_add_u32")
//   CASES:     Initializer list of Scalar3OpCase
#define TEST_SCALAR_BINARY_OP(TEST_NAME, ASM_OP, ...)                          \
  TEST(Instructions, TEST_NAME) {                                              \
    Workgroup wg;                                                              \
    Wave regs(8, 8, WaveSize{8}, wg);                                          \
    std::vector<Scalar3OpCase> cases = {__VA_ARGS__};                          \
                                                                               \
    for (size_t i = 0; i < cases.size(); ++i) {                                \
      const auto &c = cases[i];                                                \
      regs.setSgpr(0, c.src0);                                                 \
      regs.setSgpr(1, c.src1);                                                 \
      /* Reset SCC to ensure we aren't seeing stale state */                   \
      regs.setScc(0);                                                          \
                                                                               \
      tryExecute(regs, std::string(ASM_OP) + " s2, s0, s1");                   \
                                                                               \
      EXPECT_EQ(regs.getSgpr(2), c.expectedDst)                                \
          << "Dst mismatch at index " << i << " (" #ASM_OP ")";                \
      EXPECT_EQ(regs.getScc(), c.expectedScc)                                  \
          << "SCC mismatch at index " << i << " (" #ASM_OP ")";                \
    }                                                                          \
  }

#define TEST_SCALAR_CMP(TEST_NAME, ASM_OP, ...)                                \
  TEST(Instructions, TEST_NAME) {                                              \
    Workgroup wg;                                                              \
    Wave regs(8, 8, WaveSize{8}, wg);                                          \
    struct Case {                                                              \
      uint32_t s0;                                                             \
      uint32_t s1;                                                             \
      uint32_t expectedScc;                                                    \
    };                                                                         \
    std::vector<Case> cases = {__VA_ARGS__};                                   \
    for (const auto &c : cases) {                                              \
      regs.setSgpr(0, c.s0);                                                   \
      regs.setSgpr(1, c.s1);                                                   \
      tryExecute(regs, std::string(ASM_OP) + " s0, s1");                       \
      EXPECT_EQ(regs.getScc(), c.expectedScc)                                  \
          << "Failed: " << c.s0 << " vs " << c.s1;                             \
    }                                                                          \
  }
// Define a struct to hold test case data
struct Scalar3OpCase {
  uint32_t src0;
  uint32_t src1;
  uint32_t expectedDst;
  uint32_t expectedScc;
};
void tryExecute(Wave &regs, const std::string &line) {
  // Hacky way to run each instruction as if it is in its own program.
  // static int pc = 0;
  // std::map<std::string, int> labels;
  // std::map<std::string, Macro> macros;
  // ++pc;
  // return regs.tryExecute(line, pc, labels, macros);
  regs.tryExecute(line, false);
}

} // namespace

TEST(Instructions, SAndB32_Detailed) {

  Workgroup wg;
  Wave regs(/*vgprCount*/ 1, /*sgprCount*/ 4, WaveSize{1}, wg);
  regs.setSgpr(0, 0b1100); // s0 = 12
  regs.setSgpr(1, 0b1010); // s1 = 10
  regs.setScc(false);
  tryExecute(regs, "s_and_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0b1000); // s2 = 8
  EXPECT_TRUE(regs.getScc());         // and was not zero, so scc true.

  tryExecute(regs, "s_and_b32 s2, s0, s0");
  EXPECT_EQ(regs.getSgpr(2), 0b1100); // s2 = 12
  EXPECT_TRUE(regs.getScc());         // and was not zero, so scc true
  regs.setSgpr(3, 0b0011);            // s3 = 3
  tryExecute(regs, "s_and_b32 s2, s0, s3");
  EXPECT_EQ(regs.getSgpr(2), 0b0000); // s2 = 0
  EXPECT_FALSE(regs.getScc());        // and was zero,

  tryExecute(regs, "s_and_b32 s2, 7, 8");
  EXPECT_EQ(regs.getSgpr(2), 0b0000); // s2 = 0
  EXPECT_FALSE(regs.getScc());        // and was zero

  tryExecute(regs, "s_and_b32 s2, 7, 9");
  EXPECT_EQ(regs.getSgpr(2), 1);
  EXPECT_TRUE(regs.getScc());
}

// Test s_cmp_eq_u32 with literal operands and same-register comparison.
TEST(Instructions, SCmpEqU32_Literals) {
  Workgroup wg;
  Wave regs(/*vgprCount*/ 1, /*sgprCount*/ 4, WaveSize{1}, wg);
  regs.setSgpr(0, 42);  // s0 = 42
  regs.setSgpr(1, 100); // s1 = 100
  tryExecute(regs, "s_cmp_eq_u32 s0, s0");
  EXPECT_TRUE(regs.getScc()); // 42 == 42
  tryExecute(regs, "s_cmp_eq_u32 s0, 42");
  EXPECT_TRUE(regs.getScc()); // 42 == 42
  tryExecute(regs, "s_cmp_eq_u32 s0, s1");
  EXPECT_FALSE(regs.getScc()); // 42 != 100
  tryExecute(regs, "s_cmp_eq_u32 50, s1");
  EXPECT_FALSE(regs.getScc()); // 50 != 100
  tryExecute(regs, "s_cmp_eq_u32 42, s0");
  EXPECT_TRUE(regs.getScc()); // 42 == 42
}

TEST(Instructions, SBfeU32) {
  // Setup: 1 VGPR (unused), 5 SGPRs, WaveSize 1
  Workgroup wg;
  Wave regs(1, 5, WaveSize{1}, wg);

  // Data to extract from: 0x12345678
  // Binary: ... 0001 0010 0011 0100 0101 0110 0111 1000
  regs.setSgpr(0, 0x12345678);

  // --- Test 1: Extract the lowest byte (0x78) ---
  // Offset: 0, Width: 8
  // Packed Control: (8 << 16) | 0 = 0x00080000
  regs.setSgpr(1, 0x00080000);

  // s2 = BFE(s0, s1)
  tryExecute(regs, "s_bfe_u32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0x78); // Result should be 0x78
  EXPECT_TRUE(regs.getScc());       // Result is non-zero, so SCC = 1

  // --- Test 2: Extract a middle nibble (0x5) ---
  // We want to extract the '5' from 0x...5678
  // '8' is bits 0-3, '7' is 4-7, '6' is 8-11, '5' is 12-15.
  // Offset: 12, Width: 4
  // Packed Control: (4 << 16) | 12 = 0x0004000C
  regs.setSgpr(3, 0x0004000C);

  tryExecute(regs, "s_bfe_u32 s2, s0, s3");

  EXPECT_EQ(regs.getSgpr(2), 0x5);
  EXPECT_TRUE(regs.getScc());

  // --- Test 3: Zero Width Edge Case ---
  // Offset: 0, Width: 0 -> Result should be 0
  regs.setSgpr(4, 0x0);

  tryExecute(regs, "s_bfe_u32 s2, s0, s4");

  EXPECT_EQ(regs.getSgpr(2), 0x0);
  EXPECT_FALSE(regs.getScc()); // Result is zero, so SCC = 0

  // --- Test 4: Literal Support ---
  // Extract bits 4-7 (the value 0x7) using a literal for the control
  // Offset 4, Width 4 -> (4 << 16) | 4 = 0x40004 = 262148

  tryExecute(regs, "s_bfe_u32 s2, s0, 262148");

  EXPECT_EQ(regs.getSgpr(2), 0x7);
  EXPECT_TRUE(regs.getScc());
}

TEST(Instructions, SAddI32) {
  Workgroup wg;
  Wave regs(1, 5, WaveSize{1}, wg);

  // Case 1: Simple Addition (No Overflow)
  // 10 + 20 = 30
  regs.setSgpr(0, 10);
  regs.setSgpr(1, 20);
  tryExecute(regs, "s_add_i32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 30);
  EXPECT_FALSE(regs.getScc()); // Result fits in i32

  // Case 2: Positive Overflow
  // MaxInt (0x7FFFFFFF) + 1 -> Should wrap to MinInt (0x80000000)
  // This is +2147483647 + 1 = -2147483648
  regs.setSgpr(0, 0x7FFFFFFF);
  regs.setSgpr(1, 1);
  tryExecute(regs, "s_add_i32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0x80000000);
  EXPECT_TRUE(regs.getScc()); // Overflow detected!

  // Case 3: Negative Addition (No Overflow)
  // -5 + (-10) = -15
  // (0xFFFFFFFB + 0xFFFFFFF6 = 0xFFFFFFF1)
  regs.setSgpr(0, -5);
  regs.setSgpr(1, -10);
  tryExecute(regs, "s_add_i32 s2, s0, s1");

  EXPECT_EQ(static_cast<int32_t>(regs.getSgpr(2)), -15);
  EXPECT_FALSE(regs.getScc()); // -15 fits in i32

  // Case 4: Negative Overflow
  // MinInt (-2147483648) + (-1) -> Should wrap to MaxInt
  // (0x80000000 + 0xFFFFFFFF = 0x7FFFFFFF) ... carry bit discarded
  regs.setSgpr(0, 0x80000000); // Min Signed Int
  regs.setSgpr(1, -1);         // -1
  tryExecute(regs, "s_add_i32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0x7FFFFFFF); // Wrapped to Max Positive
  EXPECT_TRUE(regs.getScc());             // Overflow detected!

  // Case 5: Mixed Signs (Never Overflows)
  // MaxInt + (-1) = MaxInt - 1
  regs.setSgpr(0, 0x7FFFFFFF);
  regs.setSgpr(1, -1);
  tryExecute(regs, "s_add_i32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0x7FFFFFFE);
  EXPECT_FALSE(regs.getScc());
}

TEST(Instructions, SCSelectB32) {
  // 5 SGPRs.
  Workgroup wg;
  Wave regs(1, 5, WaveSize{1}, wg);

  // Setup inputs
  // s0 = 0xAAAA (The "True" value)
  // s1 = 0xBBBB (The "False" value)
  regs.setSgpr(0, 0xAAAA);
  regs.setSgpr(1, 0xBBBB);

  // Case 1: SCC is True -> Select First Operand (s0)
  regs.setScc(true);
  tryExecute(regs, "s_cselect_b32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0xAAAA);

  // Case 2: SCC is False -> Select Second Operand (s1)
  regs.setScc(false);
  tryExecute(regs, "s_cselect_b32 s2, s0, s1");

  EXPECT_EQ(regs.getSgpr(2), 0xBBBB);

  // Case 3: Literals (Just to be sure)
  // s2 = SCC ? 100 : 200
  regs.setScc(true);
  tryExecute(regs, "s_cselect_b32 s2, 100, 200");
  EXPECT_EQ(regs.getSgpr(2), 100);

  regs.setScc(false);
  tryExecute(regs, "s_cselect_b32 s2, 100, 200");
  EXPECT_EQ(regs.getSgpr(2), 200);
}

TEST(Instructions, SXorB64) {
  // Setup: 1 VGPR, 6 SGPRs (s0-s5), WaveSize 1
  Workgroup wg;
  Wave regs(1, 6, WaveSize{1}, wg);

  // Case 1: Simple XOR (Result is Non-Zero)
  // s[0:1] = 0x...F0F0 ^ 0x...0F0F = 0x...FFFF
  regs.setSgpr64(2, 0xF0F0F0F0F0F0F0F0ULL);
  regs.setSgpr64(4, 0x0F0F0F0F0F0F0F0FULL);

  tryExecute(regs, "s_xor_b64 s[0:1], s[2:3], s[4:5]");

  EXPECT_EQ(regs.getSgpr64(0), 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_TRUE(regs.getScc()); // Result is non-zero

  // Case 2: Identical Inputs (Result is Zero)
  // X ^ X = 0
  regs.setSgpr64(2, 0x123456789ABCDEF0ULL);
  // Re-using s[2:3] as both inputs
  tryExecute(regs, "s_xor_b64 s[0:1], s[2:3], s[2:3]");

  EXPECT_EQ(regs.getSgpr64(0), 0ULL);
  EXPECT_FALSE(regs.getScc()); // Result is zero

  // Case 3: High-Bit Check
  // Verify we aren't accidentally doing 32-bit math.
  // A = 1 << 63, B = 0. Result should have bit 63 set.
  regs.setSgpr64(2, 1ULL << 63);
  regs.setSgpr64(4, 0ULL);

  tryExecute(regs, "s_xor_b64 s[0:1], s[2:3], s[4:5]");

  EXPECT_EQ(regs.getSgpr64(0), 1ULL << 63);
  EXPECT_TRUE(regs.getScc());
}

TEST(Instructions, SMovkI32_SignExtension) {
  Workgroup wg;
  Wave regs(1, 1, WaveSize{1}, wg);

  // Case 1: Positive Signed Integer
  // 0x7FFF = 32767 (Max positive 16-bit integer)
  // Sign bit (bit 15) is 0.
  tryExecute(regs, "s_movk_i32 s0, 0x7FFF");
  EXPECT_EQ(regs.getSgpr(0), 0x00007FFF);
  EXPECT_EQ(static_cast<int32_t>(regs.getSgpr(0)), 32767);

  // Case 2: Negative Signed Integer (-1)
  // 0xFFFF represents -1 in 16-bit.
  // It should be sign-extended to 32-bit: 0xFFFFFFFF (-1).
  tryExecute(regs, "s_movk_i32 s0, 0xFFFF");

  // This is the check that would FAIL if we used <int32_t, int32_t>
  EXPECT_EQ(regs.getSgpr(0), 0xFFFFFFFF);
  EXPECT_EQ(static_cast<int32_t>(regs.getSgpr(0)), -1);

  // Case 3: Smallest Negative Integer (-32768)
  // 0x8000 is -32768 in 16-bit.
  // Should become 0xFFFF8000 in 32-bit.
  tryExecute(regs, "s_movk_i32 s0, 0x8000");
  EXPECT_EQ(regs.getSgpr(0), 0xFFFF8000);
  EXPECT_EQ(static_cast<int32_t>(regs.getSgpr(0)), -32768);
}

TEST(Instructions, SAdd) {
  // Setup: 8 VGPRs, 8 SGPRs, 8 waves per thread.
  Workgroup wg;
  Wave regs(8, 8, WaveSize{8}, wg);

  // Tests of s_add_u32 //
  ////////////////////////
  regs.setSgpr(0, 15);
  regs.setSgpr(1, 27);
  tryExecute(regs, "s_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 42);
  EXPECT_EQ(regs.getScc(), 0);

  regs.setSgpr(0, 0xFFFFFFFF);
  regs.setSgpr(1, 1);
  tryExecute(regs, "s_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0);
  EXPECT_EQ(regs.getScc(), 1);

  regs.setSgpr(0, 0xFFFFFFFF);
  regs.setSgpr(1, 0xFFFFFFFF);
  tryExecute(regs, "s_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0xFFFFFFFE);
  EXPECT_EQ(regs.getScc(), 1);

  // Tests of s_add_i32 //
  ////////////////////////
  regs.setSgpr(0, int32_t(15));
  regs.setSgpr(1, int32_t(27));
  tryExecute(regs, "s_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 42);
  EXPECT_EQ(regs.getScc(), 0);

  regs.setSgpr(0, int32_t(-1));
  regs.setSgpr(1, int32_t(1));
  tryExecute(regs, "s_add_i32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0);
  EXPECT_EQ(regs.getScc(), 0);

  regs.setSgpr(0, int32_t(-1));
  regs.setSgpr(1, int32_t(-1));
  tryExecute(regs, "s_add_i32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), int32_t(-2));
  EXPECT_EQ(regs.getScc(), 0);

  regs.setSgpr(0, std::numeric_limits<int>::lowest());
  regs.setSgpr(1, int32_t(-2));
  tryExecute(regs, "s_add_i32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), std::numeric_limits<int>::max() - 1);
  EXPECT_EQ(regs.getScc(), 1);

  regs.setSgpr(0, std::numeric_limits<int>::max());
  regs.setSgpr(1, int32_t(2));
  tryExecute(regs, "s_add_i32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), std::numeric_limits<int>::lowest() + 1);
  EXPECT_EQ(regs.getScc(), 1);

  // etc.
}

// s_add_u32 s2, s0, s1
// SCC = Carry Out
TEST_SCALAR_BINARY_OP(SAddU32, "s_add_u32", {15, 27, 42, 0},
                      {0xFFFFFFFF, 1, 0, 1},
                      {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 1})

TEST_SCALAR_BINARY_OP(SSubU32, "s_sub_u32", {10, 3, 7, 0},
                      {3, 10, 0xFFFFFFF9, 1}, {10, 10, 0, 0})

TEST_SCALAR_BINARY_OP(SAndB32, "s_and_b32",
                      {0xF0F0F0F0, 0x0F0F0F0F, 0, 0},         // No overlap
                      {0xF0F0F0F0, 0xF0000000, 0xF0000000, 1} // Overlap
)

// s_lshl_b32 s2, s0, s1
// SCC = 1 if result != 0
TEST_SCALAR_BINARY_OP(SLshlB32, "s_lshl_b32", {1, 4, 16, 1},
                      {0xFFFFFFFF, 1, 0xFFFFFFFE, 1}, {1, 32, 1, 1},
                      {0x10000000, 4, 0, 0}, {0x10000000, 1, 0x20000000, 1})

TEST_SCALAR_BINARY_OP(SLshrB32, "s_lshr_b32", {16, 2, 4, 1}, {1, 1, 0, 0},
                      {1, 32, 1, 1}, )

// s_mul_hi_u32 s2, s0, s1
// Result = High 32-bits of (u64(s0) * u64(s1))
TEST_SCALAR_BINARY_OP(SMulHiU32, "s_mul_hi_u32",
                      {2, 3, 0, 0}, // 6 fits in 32 bits, Hi=0
                      {0xFFFFFFFF, 2, 1,
                       0}, // (2^32 - 1) * 2 = 0x1 FFFFFFFE -> Hi=1
                      {0x80000000, 4, 2, 0} // 2^31 * 4 = 2^33 -> Hi=2
)

// s_mul_hi_i32 s2, s0, s1
// Signed high part
TEST_SCALAR_BINARY_OP(SMulHiI32, "s_mul_hi_i32",
                      {(uint32_t)-2, 2, (uint32_t)-1,
                       0}, // -2 * 2 = -4 (0xFF...FC) -> Hi=0xFFFFFFFF (-1)
                      {0x7FFFFFFF, 2, 0, 0} // MaxInt * 2 = 0xFFFFFFFE. Hi=0.
)

TEST(Instructions, SAddCU32) {
  Workgroup wg;
  Wave regs(8, 8, WaveSize{8}, wg);

  // 1. No Carry In
  regs.setScc(0);
  regs.setSgpr(0, 10);
  regs.setSgpr(1, 20);
  tryExecute(regs, "s_addc_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 30);
  EXPECT_EQ(regs.getScc(), 0);

  // 2. Carry In
  regs.setScc(1);
  regs.setSgpr(0, 10);
  regs.setSgpr(1, 20);
  tryExecute(regs, "s_addc_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 31); // 10 + 20 + 1
  EXPECT_EQ(regs.getScc(), 0);

  // 3. Carry Out Generated
  regs.setScc(0);
  regs.setSgpr(0, 0xFFFFFFFF);
  regs.setSgpr(1, 1);
  tryExecute(regs, "s_addc_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0);
  EXPECT_EQ(regs.getScc(), 1);

  // 3. Carry Out Generated
  regs.setScc(1);
  regs.setSgpr(0, 0xFFFFFFFF);
  regs.setSgpr(1, 1);
  tryExecute(regs, "s_addc_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 1);
  EXPECT_EQ(regs.getScc(), 1);
}

TEST_SCALAR_CMP(SCmpEqU32, "s_cmp_eq_u32", {10, 10, 1}, {10, 11, 0})

TEST_SCALAR_CMP(SCmpLtU32, "s_cmp_lt_u32", {5, 10, 1}, // 5 < 10
                {10, 5, 0},                            // 10 is not < 5
                {10, 10, 0},         // equal: not strictly less
                {0, 1, 1},           // 0 < 1
                {(uint32_t)-1, 0, 0} // UINT_MAX < 0 is false (unsigned)
)

TEST_SCALAR_CMP(SCmpGtI32, "s_cmp_gt_i32", {10, 5, 1}, {5, 10, 0},
                {(uint32_t)-5, 5, 0} // -5 is NOT > 5
)
TEST(Instructions, S_LSHL_B64) {
  // Needs s0-s4 (5 registers)
  Workgroup wg;
  Wave regs(/*vgpr*/ 0, /*sgpr*/ 6, WaveSize{64}, wg);

  // Setup Input: s[0:1] = 1 (64-bit integer)
  // We can now set this directly as a 64-bit value
  regs.setSgpr64(0, 1);

  // Setup Shift Amount: s4 = 33
  regs.setSgpr(4, 33);

  // Execute: s[2:3] = s[0:1] << s4
  // Result should be 1 << 33 = 0x200000000
  tryExecute(regs, "s_lshl_b64 s[2:3], s[0:1], s4");

  // Verify Result using 64-bit getter
  EXPECT_EQ(regs.getSgpr64(2), 0x200000000ULL);

  // Verify SCC (Result is non-zero, so SCC=1)
  EXPECT_EQ(regs.getScc(), 1);
}

// s_ctz_i32_b32: RDNA3+ name for s_ff1_i32_b32 (count trailing zeros).
TEST(Instructions, S_CTZ_I32_B32) {
  Workgroup wg;
  Wave regs(1, 4, WaveSize{1}, wg);

  // Trailing zeros of 0b...1000 = 3
  regs.setSgpr(0, 0x18); // 0b11000
  tryExecute(regs, "s_ctz_i32_b32 s1, s0");
  EXPECT_EQ(regs.getSgpr(1), 3u);

  // Trailing zeros of 1 = 0
  regs.setSgpr(0, 1);
  tryExecute(regs, "s_ctz_i32_b32 s1, s0");
  EXPECT_EQ(regs.getSgpr(1), 0u);

  // Input 0 returns -1 (0xFFFFFFFF)
  regs.setSgpr(0, 0);
  tryExecute(regs, "s_ctz_i32_b32 s1, s0");
  EXPECT_EQ(regs.getSgpr(1), 0xFFFFFFFFu);

  // Only bit 31 set => 31 trailing zeros
  regs.setSgpr(0, 0x80000000);
  tryExecute(regs, "s_ctz_i32_b32 s1, s0");
  EXPECT_EQ(regs.getSgpr(1), 31u);
}

// s_bfm_b32: bitfield mask. D = ((1 << S0[4:0]) - 1) << S1[4:0].
TEST(Instructions, S_BFM_B32) {
  Workgroup wg;
  Wave regs(1, 4, WaveSize{1}, wg);

  // Width=8, Offset=0 => mask = 0xFF
  regs.setSgpr(0, 8);
  regs.setSgpr(1, 0);
  tryExecute(regs, "s_bfm_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0xFFu);

  // Width=4, Offset=4 => mask = 0xF0
  regs.setSgpr(0, 4);
  regs.setSgpr(1, 4);
  tryExecute(regs, "s_bfm_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0xF0u);

  // Width=0, any offset => mask = 0
  regs.setSgpr(0, 0);
  regs.setSgpr(1, 16);
  tryExecute(regs, "s_bfm_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0u);

  // Width=32, Offset=0 => uses only S0[4:0]=0, so width=0 => 0
  // (32 & 0x1F = 0)
  regs.setSgpr(0, 32);
  regs.setSgpr(1, 0);
  tryExecute(regs, "s_bfm_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0u);

  // Width=1, Offset=31 => mask = 0x80000000
  regs.setSgpr(0, 1);
  regs.setSgpr(1, 31);
  tryExecute(regs, "s_bfm_b32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 0x80000000u);
}

// s_lshl2_add_u32: D = (S0 << 2) + S1. SCC = carry out.
TEST(Instructions, S_LSHL2_ADD_U32) {
  Workgroup wg;
  Wave regs(1, 4, WaveSize{1}, wg);

  // Simple: (3 << 2) + 5 = 17
  regs.setSgpr(0, 3);
  regs.setSgpr(1, 5);
  tryExecute(regs, "s_lshl2_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 17u);
  EXPECT_FALSE(regs.getScc()); // no carry

  // Carry: (0xC0000000 << 2) overflows + 1 => carry
  regs.setSgpr(0, 0xC0000000);
  regs.setSgpr(1, 1);
  tryExecute(regs, "s_lshl2_add_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 1u); // 0x300000000 + 1 truncated
  EXPECT_TRUE(regs.getScc());     // carry
}

// s_cmpk_lg_u32: compare SGPR != immediate, set SCC.
TEST(Instructions, S_CMPK_LG_U32) {
  Workgroup wg;
  Wave regs(1, 4, WaveSize{1}, wg);

  regs.setSgpr(0, 42);
  tryExecute(regs, "s_cmpk_lg_u32 s0, 0x2a"); // 0x2a = 42
  EXPECT_FALSE(regs.getScc());                // equal => not "lg" => SCC=0

  tryExecute(regs, "s_cmpk_lg_u32 s0, 0x0");
  EXPECT_TRUE(regs.getScc()); // 42 != 0 => SCC=1
}

TEST(Instructions, MOV_Ops_V2) {
  Workgroup wg;
  Wave regs(4, 4, WaveSize{64}, wg);

  // s_mov_b32 (Scalar)
  regs.setSgpr(0, 0x12345678);
  tryExecute(regs, "s_mov_b32 s1, s0");
  EXPECT_EQ(regs.getSgpr(1), 0x12345678);

  // s_mov_b64 (Scalar 64-bit)
  regs.setSgpr64(0, 0xABCD1234FFFF0000);
  tryExecute(regs, "s_mov_b64 s[2:3], s[0:1]");
  EXPECT_EQ(regs.getSgpr64(2), 0xABCD1234FFFF0000);
}

TEST(Instructions, S_LSHR_B64) {
  Workgroup wg;
  Wave regs(1, 8, WaveSize{1}, wg);
  regs.setSgpr64(0, 0x800000000ULL); // bit 35 set
  regs.setSgpr(4, 35);
  tryExecute(regs, "s_lshr_b64 s[2:3], s[0:1], s4");
  EXPECT_EQ(regs.getSgpr64(2), 1ULL);
  EXPECT_EQ(regs.getScc(), 1);

  // Shift to zero.
  regs.setSgpr64(0, 0xF);
  regs.setSgpr(4, 4);
  tryExecute(regs, "s_lshr_b64 s[2:3], s[0:1], s4");
  EXPECT_EQ(regs.getSgpr64(2), 0ULL);
  EXPECT_EQ(regs.getScc(), 0);
}

TEST(Instructions, S_MIN_U32) {
  Workgroup wg;
  Wave regs(1, 4, WaveSize{1}, wg);
  regs.setSgpr(0, 10);
  regs.setSgpr(1, 20);
  tryExecute(regs, "s_min_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 10u);
  EXPECT_EQ(regs.getScc(), 1); // SCC=1 when s0 < s1

  regs.setSgpr(0, 20);
  regs.setSgpr(1, 10);
  tryExecute(regs, "s_min_u32 s2, s0, s1");
  EXPECT_EQ(regs.getSgpr(2), 10u);
  EXPECT_EQ(regs.getScc(), 0); // SCC=0 when s0 >= s1
}
