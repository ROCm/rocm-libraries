// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/Util.h"
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

// --- stripComments unit tests ---

TEST(StripComments, LineCommentSemicolon) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code ; comment", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, LineCommentDoubleSlash) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code // comment", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, SingleLineBlockComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("a /* comment */ b", inBlock), "a  b");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, MultipleBlockCommentsOnOneLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("/* x */ a /* y */ b", inBlock), " a  b");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentOpensMultiLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code /* start of block", inBlock), "code ");
  EXPECT_TRUE(inBlock);
}

TEST(StripComments, InsideBlockComment) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("  still in comment", inBlock), "");
  EXPECT_TRUE(inBlock);
}

TEST(StripComments, BlockCommentCloses) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("  end of block */ code", inBlock), " code");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentClosesThenReopens) {
  bool inBlock = true;
  EXPECT_EQ(stripComments("in */ out /* in */ out", inBlock), " out  out");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentInsideLineComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code // not /* a block", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, BlockCommentInsideSemicolonComment) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("code ; not /* a block", inBlock), "code ");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, NoComments) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("v_mov_b32_e32 v0, 0", inBlock),
            "v_mov_b32_e32 v0, 0");
  EXPECT_FALSE(inBlock);
}

TEST(StripComments, EmptyLine) {
  bool inBlock = false;
  EXPECT_EQ(stripComments("", inBlock), "");
  EXPECT_FALSE(inBlock);
}

// Integration test: block comments work correctly through the full
// parser and emulator pipeline.
TEST(Parser, BlockCommentsIgnored) {
  static constexpr std::string_view asm_with_block_comment = R"ASM(
.amdhsa_kernel foo
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 4
  .amdhsa_group_segment_fixed_size 1024
  .amdhsa_system_sgpr_workgroup_id_x 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: foo
    .symbol: 'foo.kd'
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 1024
    .wavefront_size: 64
    .args:
      - .name: ptr
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: generic
...
.end_amdgpu_metadata
foo:
  v_mov_b32_e32 v0, 0
  /* This is a multi-line block comment.
     It should be completely ignored.
     even_if_it_looks_like_an_instruction v0, v1, v2
  */
  v_mov_b32_e32 v1, 1
  // This is not /* a block comment start!
  /* first */ v_mov_b32_e32 v2, /* second */ 2
  s_endpgm
)ASM";
  Emulator emu = Emulator::createGfx942(asm_with_block_comment);
  std::vector<int> data(2, 0);
  int *p = data.data();
  emu.addKernarg(0, &p);
  emu.run({0, 0, 0}, {64, 1, 1});
  EXPECT_EQ(emu.getWave(0).getVgpr(1, 0), 1u);
  EXPECT_EQ(emu.getWave(0).getVgpr(2, 0), 2u);
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
