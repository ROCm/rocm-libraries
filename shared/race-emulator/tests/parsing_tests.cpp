// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include <cstring>
#include <gtest/gtest.h>
#include <string_view>

using namespace raceemulator;

TEST(Parser, ParserTestZero) {
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
  EXPECT_EQ(emu.getName(), "my_kernel");
  EXPECT_EQ(emu.getKernargSegmentSize(), 184);
  ASSERT_GE(emu.getNumKernargs(), 2);
  EXPECT_EQ(emu.getKernargName(0), "Gemm info");
  EXPECT_EQ(emu.getKernargSize(0), 4);
  EXPECT_EQ(emu.getKernargOffset(0), 0);
  EXPECT_EQ(emu.getKernargValueKind(0), "by_value");
  EXPECT_EQ(emu.getKernargName(1), "D");
  EXPECT_EQ(emu.getKernargValueKind(1), "global_buffer");
  EXPECT_EQ(emu.getKernargOffset(1), 32);
}

TEST(Parser, ParserTestOne) {
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
  EXPECT_EQ(emu.getName(), "_Z12adder");
  EXPECT_EQ(emu.getKernargSegmentSize(), 32);
  EXPECT_EQ(emu.getNumKernargs(), 2);
  EXPECT_EQ(emu.getKernargOffset(0), 0);
  EXPECT_EQ(emu.getKernargSize(0), 4);
  EXPECT_EQ(emu.getKernargValueKind(0), "by_value");
  EXPECT_EQ(emu.getKernargOffset(1), 8);
  EXPECT_EQ(emu.getKernargSize(1), 8);
  EXPECT_EQ(emu.getKernargAddressSpace(1), "global");
}

TEST(Parser, Gfx1151ArchProperties) {
  Gfx1151 arch;
  EXPECT_EQ(arch.getName(), "gfx1151");
  EXPECT_EQ(arch.getMaxLdsSize(), 65536);
}

TEST(Parser, ArchitectureFromTargetGfx1151) {
  auto arch = architectureFromTarget("amdgcn-amd-amdhsa--gfx1151");
  EXPECT_EQ(arch->getName(), "gfx1151");
  EXPECT_EQ(arch->getMaxLdsSize(), 65536);
}

TEST(Parser, ArchMismatchGfx1151WithGfx942Assembly) {
  // Target is parsed from .amdgcn_target directive, not YAML amdhsa.target.
  static constexpr std::string_view gfx942_asm = R"ASM(
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
---
amdhsa.kernels:
  - .args:
      - .offset:          0
        .size:            4
        .value_kind:      by_value
    .kernarg_segment_size: 4
    .name:             test_kernel
    .symbol:           test_kernel.kd
...
)ASM";
  EXPECT_NO_THROW(Emulator::createGfx942(gfx942_asm));
  EXPECT_THROW(Emulator::createGfx1151(gfx942_asm), std::runtime_error);
}

TEST(Parser, ParserGfx1151Target) {
  // HIP source:
  //   __global__ void copy_kernel(int* out, const int* in) {
  //     int i = threadIdx.x;
  //     out[i] = in[i];
  //   }
  //
  // Compiled with:
  //   hipcc -S --offload-arch=gfx1151 -O2 copy_kernel.cpp
  //
  // Metadata copied verbatim from the YAML section of the hipcc output.
  static constexpr std::string_view gfx1151_asm = R"ASM(
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .name:           _Z11copy_kernelPiPKi
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         _Z11copy_kernelPiPKi.kd
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
)ASM";
  auto emu = Emulator::createGfx1151(gfx1151_asm);
  EXPECT_EQ(emu.getName(), "_Z11copy_kernelPiPKi");
  EXPECT_EQ(emu.getKernargSegmentSize(), 16);
  EXPECT_EQ(emu.getNumKernargs(), 2);
  EXPECT_EQ(emu.getArch().getName(), "gfx1151");
}
