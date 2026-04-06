// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Wave.h"
#include "race-emulator/Workgroup.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace raceemulator;

void tryExecute(Wave &regs, const std::string &line) {
  regs.tryExecute(line, false);
  regs.getWorkgroup().dispatchPendingRaceEvents(regs.getWaveId());
}

// TODO(newling) add tests of control flow instructions

TEST(Instructions, SAndSaveExecB64) {

  Workgroup wg({.vgprCount = 1, .sgprCount = 10, .waveSize = WaveSize{64}});
  auto &regs = wg.getWave(0);

  // populate vcc first 33 bits to 1.
  // populate exec final 33 bits with 1.
  // We then expect the & to be 1 at bits 31 and 32.
  // We check that
  // 1) exec is now these 2 bits
  // 2) s[2:3] is the old exec value.
  // 3) SCC is 1 (because the new exec is non-zero).
  //
  // We then check that EXEC is exactly this value.
  // s_and_saveexec_b64 s[2:3], vcc

  uint64_t vcc = (uint64_t(1) << 33) - 1;
  uint64_t exec = (vcc << 31);
  auto observed = vcc & exec;
  auto expected = uint64_t(3) << 31;
  EXPECT_EQ(observed, expected);

  regs.setExecU64(exec);
  regs.setVccU64(vcc);
  tryExecute(regs, "s_and_saveexec_b64 s[2:3], vcc");
  // The checks:
  EXPECT_EQ(regs.getExecU64(), expected);
  EXPECT_EQ(regs.getSgpr64(2), exec); // old exec saved
  EXPECT_TRUE(regs.getScc());         // new exec non-zero
}

// s_swappc_b64 is a no-op in the emulator.  On real HW it saves the
// return address (PC+4) to the destination SGPR pair and jumps to the
// byte-address in the source pair.  Because the emulator uses line-index
// PCs instead of byte addresses, the target address cannot be resolved.
// Currently only used for activation function calls with activationType=0
// (identity), so no-op is correct.  Verify that:
//   1) PC advances by 1.
//   2) Neither operand is clobbered (the real instruction would write dst;
//      the no-op deliberately does not, because the caller's code flow
//      assumes the subroutine runs and returns, which our no-op skips).
TEST(Instructions, S_SwapPc_B64_NoOp) {
  Workgroup wg({.vgprCount = 4, .sgprCount = 16, .waveSize = WaveSize{32}});
  auto &regs = wg.getWave(0);

  // Set up operands as the kernel does:
  //   s_swappc_b64 s[4:5], s[12:13]
  // Pre-fill both dst and src with sentinel values.
  regs.setSgpr(4, 0xAAAAAAAA);
  regs.setSgpr(5, 0xBBBBBBBB);
  regs.setSgpr(12, 0x12345678);
  regs.setSgpr(13, 0x9ABCDEF0);

  // Also set a VGPR to check it isn't touched.
  regs.setVgpr(0, 0, 42);

  int pcBefore = regs.getPc();
  tryExecute(regs, "s_swappc_b64 s[4:5], s[12:13]");
  EXPECT_EQ(regs.getPc(), pcBefore + 1);

  // SGPRs should be untouched (no-op does not write return address).
  EXPECT_EQ(regs.getSgpr(4), 0xAAAAAAAAu);
  EXPECT_EQ(regs.getSgpr(5), 0xBBBBBBBBu);
  EXPECT_EQ(regs.getSgpr(12), 0x12345678u);
  EXPECT_EQ(regs.getSgpr(13), 0x9ABCDEF0u);

  // VGPR untouched.
  EXPECT_EQ(regs.getVgpr(0, 0), 42u);
}

// s_sendmsg is a no-op in the emulator (hardware sends a message to the
// SQUEUE, which has no emulator equivalent). Verify it executes without
// throwing and leaves registers unchanged.
TEST(Instructions, S_SENDMSG) {
  Workgroup wg({.vgprCount = 0, .sgprCount = 4, .waveSize = WaveSize{64}});
  auto &regs = wg.getWave(0);
  regs.setSgpr(2, 0xDEADBEEF);
  tryExecute(regs, "s_sendmsg sendmsg(MSG_INTERRUPT)");
  EXPECT_EQ(regs.getSgpr(2), 0xDEADBEEFu); // unchanged
}

// Verify that instruction caching works: when enableLineCaching is true,
// each line should only be compiled once regardless of how many times the
// PC revisits it.
TEST(Instructions, InstructionCaching) {
  std::map<std::string, int> labels = {{"loop", 0}};
  std::map<std::string, Macro> macros;
  Workgroup wg({.vgprCount = 4,
                .sgprCount = 4,
                .waveSize = WaveSize{1},
                .labels = &labels,
                .macros = &macros});
  auto &wave = wg.getWave(0);

  // A 3-instruction loop: set v0, set v1, branch back to start.
  // Run it 5 times (15 instruction executions total).
  std::vector<std::string> program = {
      "v_mov_b32 v0, 42", // PC 0
      "v_mov_b32 v1, 99", // PC 1
      "s_branch loop",    // PC 2 -> jumps to PC 0
  };

  int iterations = 5;
  int totalInstructions = iterations * static_cast<int>(program.size());

  EXPECT_EQ(wave.getCompileCount(), 0);
  for (int i = 0; i < totalInstructions; ++i) {
    int pc = wave.getPc();
    wave.tryExecute(program[pc], /*enableLineCaching=*/true);
  }

  // Each of the 3 unique lines should be compiled exactly once.
  EXPECT_EQ(wave.getCompileCount(), 3);
}

// Verify that without caching, instructions are recompiled each time.
TEST(Instructions, NoCachingRecompiles) {
  std::map<std::string, int> labels = {{"loop", 0}};
  std::map<std::string, Macro> macros;
  Workgroup wg({.vgprCount = 4,
                .sgprCount = 4,
                .waveSize = WaveSize{1},
                .labels = &labels,
                .macros = &macros});
  auto &wave = wg.getWave(0);

  std::vector<std::string> program = {
      "v_mov_b32 v0, 42",
      "v_mov_b32 v1, 99",
      "s_branch loop",
  };

  int iterations = 3;
  int totalInstructions = iterations * static_cast<int>(program.size());

  for (int i = 0; i < totalInstructions; ++i) {
    int pc = wave.getPc();
    wave.tryExecute(program[pc], /*enableLineCaching=*/false);
  }

  // Without caching, each execution compiles fresh.
  EXPECT_EQ(wave.getCompileCount(), totalInstructions);
}
