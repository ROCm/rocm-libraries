#include "race-emulator/Wave.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace raceemulator;

void tryExecute(Wave &regs, const std::string &line) {
  regs.tryExecute(line, false);
}

// TODO(newling) add tests of control flow instructions

TEST(GpuEmulatorUnitTests, SAndSaveExecB64) {

  Wave regs(/*vgprCount*/ 1, /*sgprCount*/ 10, /*waveSize*/ 64);

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
