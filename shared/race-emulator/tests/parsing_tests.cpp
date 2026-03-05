#include "race-emulator/Emulator.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string_view>

using namespace raceemulator;

TEST(GpuEmulatorUnitTestsZero, ParserTestZero) {
  static constexpr std::string_view my_kernel = R"ASM(
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: my_kernel
    .symbol: 'my_kernel.kd'
    .language:                    OpenCL C
    .kernarg_segment_size:        184
    .group_segment_fixed_size:    124672
    .vgpr_count:                  256
    .args:
      - .name:              Gemm info
        .size:              4
        .offset:            0
        .value_kind:        by_value
        .value_type:        u32
      - .name:              D
        .size:              8
        .offset:            32
        .value_kind:        global_buffer
        .address_space:     generic
...
)ASM";
  auto emu = Emulator::createGfx942(my_kernel);
  EXPECT_EQ(emu.name(), "my_kernel");
  EXPECT_EQ(emu.kernargSegmentSize(), 184);
  ASSERT_GE(emu.nKernargs(), 2);
  EXPECT_EQ(emu.kernargName(0), "Gemm info");
  EXPECT_EQ(emu.kernargSize(0), 4);
  EXPECT_EQ(emu.kernargOffset(0), 0);
  EXPECT_EQ(emu.kernargValueKind(0), "by_value");
  EXPECT_EQ(emu.kernargName(1), "D");
  EXPECT_EQ(emu.kernargValueKind(1), "global_buffer");
  EXPECT_EQ(emu.kernargOffset(1), 32);
}

TEST(GpuEmulatorUnitTestsZero, ParserTestOne) {
  static constexpr std::string_view simple_adder_kernel = R"ASM(
---
amdhsa.kernels:
  - .agpr_count:       0
    .args:
      - .offset:          0
        .size:            4
        .value_kind:      by_value
      - .address_space:   global
        .offset:          8
        .size:            8
        .value_kind:      global_buffer
    .kernarg_segment_size: 32
    .name:             _Z12adder
    .symbol:           _Z12adder.kd
amdhsa.target:    amdgcn-amd-amdhsa--gfx942
...
)ASM";
  auto emu = Emulator::createGfx942(simple_adder_kernel);
  EXPECT_EQ(emu.name(), "_Z12adder");
  EXPECT_EQ(emu.kernargSegmentSize(), 32);
  EXPECT_EQ(emu.nKernargs(), 2);
  EXPECT_EQ(emu.kernargOffset(0), 0);
  EXPECT_EQ(emu.kernargSize(0), 4);
  EXPECT_EQ(emu.kernargValueKind(0), "by_value");
  EXPECT_EQ(emu.kernargOffset(1), 8);
  EXPECT_EQ(emu.kernargSize(1), 8);
  EXPECT_EQ(emu.kernargAddressSpace(1), "global");
}
