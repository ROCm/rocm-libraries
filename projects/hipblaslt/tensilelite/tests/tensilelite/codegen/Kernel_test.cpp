// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/Kernel.hpp>

#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <iterator>
#include <stdexcept>
#include <string>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using rocm::tensilelite::codegen::test::expectLinesInOrder;

bool hasLine(const std::string &text, const std::string &line) {
  return text.find("\n" + line + "\n") != std::string::npos;
}

class KernelTest : public testing::Test {
protected:
  Target target = Target::get("gfx950");
};

TEST_F(KernelTest, DefaultEmitsCodeObjectV6) {
  Kernel kernel(target, "empty_kernel", 256);
  kernel.signature().addPointer("out");
  kernel.builder().createEndProgram();
  EXPECT_EQ(kernel.codeObjectVersion(), CodeObjectVersion::v6);

  expectLinesInOrder(
      kernel.toAssembly(),
      {
          ".amdgcn_target \"amdgcn-amd-amdhsa--gfx950\"",
          ".amdhsa_code_object_version 6",
          ".text",
          ".protected empty_kernel",
          ".globl empty_kernel",
          ".p2align 8",
          ".type empty_kernel,@function",
          "empty_kernel:",
          "s_endpgm",
          ".Lempty_kernel_end:",
          ".size empty_kernel, .Lempty_kernel_end - empty_kernel",
          ".rodata",
          ".p2align 6",
          ".amdhsa_kernel empty_kernel",
          ".amdhsa_user_sgpr_kernarg_segment_ptr 1",
          ".amdhsa_kernarg_size 8",
          ".amdhsa_accum_offset 8",
          ".amdhsa_next_free_vgpr 8",
          ".amdhsa_next_free_sgpr 3",
          ".amdhsa_group_segment_fixed_size 0",
          ".amdhsa_private_segment_fixed_size 0",
          ".amdhsa_system_sgpr_workgroup_id_x 1",
          ".amdhsa_system_vgpr_workitem_id 0",
          ".amdhsa_float_denorm_mode_32 3",
          ".amdhsa_float_denorm_mode_16_64 3",
          ".end_amdhsa_kernel",
          ".amdgpu_metadata",
          "amdhsa.version: [1, 2]",
          "amdhsa.target: amdgcn-amd-amdhsa--gfx950",
          "- .name: empty_kernel",
          ".symbol: empty_kernel.kd",
          ".max_flat_workgroup_size: 256",
          ".wavefront_size: 64",
          ".end_amdgpu_metadata",
      });
}

TEST_F(KernelTest, MapsCodeObjectVersions) {
  struct Case {
    CodeObjectVersion version;
    const char *directive;
    const char *metadata;
  };
  const Case cases[] = {
      {CodeObjectVersion::v5, ".amdhsa_code_object_version 5",
       "amdhsa.version: [1, 2]"},
      {CodeObjectVersion::v4, ".amdhsa_code_object_version 4",
       "amdhsa.version: [1, 1]"},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.directive);
    Kernel kernel(target, "k", 64, c.version);
    kernel.builder().createEndProgram();
    std::string assembly = kernel.toAssembly();
    EXPECT_TRUE(hasLine(assembly, c.directive)) << assembly;
    EXPECT_TRUE(hasLine(assembly, c.metadata)) << assembly;
  }
}

TEST_F(KernelTest, LaysOutArgumentsNaturally) {
  Kernel kernel(target, "args_kernel", 64);
  EXPECT_EQ(kernel.signature().addPointer("x"), 0);
  EXPECT_EQ(kernel.signature().addPointer("y"), 8);
  EXPECT_EQ(kernel.signature().addScalar("a", 4), 16);
  EXPECT_EQ(kernel.signature().addScalar("n", 4), 20);
  EXPECT_EQ(kernel.signature().segmentSize(), 24);
  kernel.builder().createEndProgram();

  std::string assembly = kernel.toAssembly();
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_kernarg_size 24")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .kernarg_segment_size: 24")) << assembly;
  expectLinesInOrder(assembly, {
                                   "    .args:",
                                   "      - .name: x",
                                   "        .size: 8",
                                   "        .offset: 0",
                                   "        .value_kind: global_buffer",
                                   "        .address_space: global",
                                   "      - .name: y",
                                   "        .size: 8",
                                   "        .offset: 8",
                                   "        .value_kind: global_buffer",
                                   "        .address_space: global",
                                   "      - .name: a",
                                   "        .size: 4",
                                   "        .offset: 16",
                                   "        .value_kind: by_value",
                                   "      - .name: n",
                                   "        .size: 4",
                                   "        .offset: 20",
                                   "        .value_kind: by_value",
                                   "...",
                               });
  // A by-value argument is followed directly by the next entry.
  EXPECT_NE(assembly.find("        .value_kind: by_value\n      - .name: n"),
            std::string::npos)
      << assembly;
  EXPECT_NE(assembly.find("        .value_kind: by_value\n...\n"),
            std::string::npos)
      << assembly;
}

TEST_F(KernelTest, PadsArgumentsToNaturalAlignment) {
  KernelSignature padded;
  EXPECT_EQ(padded.addScalar("u32", 4), 0);
  EXPECT_EQ(padded.addPointer("p"), 8);
  EXPECT_EQ(padded.segmentSize(), 16);

  KernelSignature lone;
  lone.addScalar("u32", 4);
  EXPECT_EQ(lone.segmentSize(), 8);

  KernelSignature wide;
  wide.addScalar("u32", 4);
  EXPECT_EQ(wide.addScalar("u64", 8), 8);

  Kernel kernel(target, "no_args", 64);
  kernel.builder().createEndProgram();
  EXPECT_EQ(kernel.signature().segmentSize(), 0);
  std::string assembly = kernel.toAssembly();
  EXPECT_EQ(assembly.find(".args:"), std::string::npos) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_kernarg_size 0")) << assembly;
}

TEST_F(KernelTest, RejectsInvalidArguments) {
  KernelSignature signature;
  EXPECT_THROW(signature.addScalar("a", 2), std::invalid_argument);
  EXPECT_THROW(signature.addScalar("a", 16), std::invalid_argument);
  signature.addPointer("x");
  EXPECT_THROW(signature.addPointer("x"), std::invalid_argument);
  EXPECT_THROW(signature.addScalar("x", 4), std::invalid_argument);
  EXPECT_THROW(signature.addPointer(""), std::invalid_argument);
  EXPECT_THROW(signature.addPointer("1x"), std::invalid_argument);
  EXPECT_THROW(signature.addPointer("a-b"), std::invalid_argument);
  EXPECT_EQ(std::ssize(signature.arguments()), 1);
}

TEST_F(KernelTest, RejectsInvalidKernelParameters) {
  EXPECT_THROW(Kernel(target, "bad-name", 64), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "", 64), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "k", 0), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "k", 1025), std::invalid_argument);
  EXPECT_NO_THROW(Kernel(target, "k", 1024));
  EXPECT_THROW(Kernel(target, "k", 64, static_cast<CodeObjectVersion>(3)),
               std::invalid_argument);
}

TEST_F(KernelTest, ReservesAbiRegisters) {
  Kernel kernel(target, "k", 64);
  EXPECT_EQ(kernel.kernargSegmentPtr().kind(), RegisterKind::sgpr);
  EXPECT_EQ(kernel.kernargSegmentPtr().start(), 0);
  EXPECT_EQ(kernel.kernargSegmentPtr().count(), 2);
  EXPECT_EQ(kernel.workgroupIdX().kind(), RegisterKind::sgpr);
  EXPECT_EQ(kernel.workgroupIdX().start(), 2);
  EXPECT_EQ(kernel.workgroupIdX().count(), 1);
  EXPECT_EQ(kernel.workitemId().kind(), RegisterKind::vgpr);
  EXPECT_EQ(kernel.workitemId().start(), 0);
  EXPECT_EQ(kernel.workitemId().count(), 1);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 3);
  EXPECT_EQ(kernel.builder().allocateVgpr(1).start(), 1);
  // No AGPR is reserved.
  EXPECT_EQ(kernel.builder().allocateAgpr(1).start(), 0);
}

TEST_F(KernelTest, DerivesRegisterCountsFromHighWaterMarks) {
  {
    Kernel kernel(target, "empty", 64);
    std::string assembly = kernel.toAssembly();
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_accum_offset 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_vgpr 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_sgpr 3")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .sgpr_count: 9")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .vgpr_count: 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .agpr_count: 0")) << assembly;
  }
  Kernel kernel(target, "grown", 64);
  // The registers are released before printing, so only the high-water mark
  // can account for them.
  kernel.builder().allocateVgpr(10);
  kernel.builder().allocateSgpr(4, 4);
  std::string assembly = kernel.toAssembly();
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_accum_offset 16")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_vgpr 16")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_sgpr 8")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .sgpr_count: 14")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .vgpr_count: 16")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .agpr_count: 0")) << assembly;
}

TEST_F(KernelTest, PlacesAgprsAfterAccumOffset) {
  Kernel kernel(target, "accumulators", 64);
  kernel.builder().allocateVgpr(10);
  kernel.builder().allocateAgpr(6);
  std::string assembly = kernel.toAssembly();
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_accum_offset 16")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_vgpr 22")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .vgpr_count: 22")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .agpr_count: 6")) << assembly;
}

TEST_F(KernelTest, ExceedingAgprLimitThrows) {
  Kernel kernel(target, "k", 64);
  kernel.builder().allocateAgpr(target.maxAgpr() + 1);
  EXPECT_THROW(kernel.toAssembly(), std::runtime_error);
}

TEST_F(KernelTest, ExceedingRegisterLimitThrows) {
  Kernel kernel(target, "k", 64);
  kernel.builder().allocateVgpr(target.maxVgpr());
  EXPECT_THROW(kernel.toAssembly(), std::runtime_error);
}

void addReferenceGemmExplicitArguments(KernelSignature &signature) {
  for (const char *name : {"a", "b", "c", "d", "e", "f"})
    signature.addPointer(name);
  for (const char *name : {"s0", "s1", "s2", "s3", "s4", "s5", "s6"})
    signature.addScalar(name, 4);
}

TEST_F(KernelTest, ReplicatesReferenceGemmArgumentLayout) {
  Kernel kernel(target, "gemm", 256);
  KernelSignature &signature = kernel.signature();
  addReferenceGemmExplicitArguments(signature);
  EXPECT_EQ(signature.arguments()[5].offset, 40);
  EXPECT_EQ(signature.arguments()[6].offset, 48);
  EXPECT_EQ(signature.arguments()[12].offset, 72);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4), 76);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4), 80);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4), 84);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4), 88);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4), 92);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4), 96);
  EXPECT_EQ(signature.segmentSize(), 104);
  EXPECT_TRUE(signature.arguments().back().name.empty());
  kernel.builder().createEndProgram();

  std::string assembly = kernel.toAssembly();
  EXPECT_NE(assembly.find("        .value_kind: by_value\n"
                          "      - .size: 4\n"
                          "        .offset: 76\n"
                          "        .value_kind: hidden_group_size_x\n"
                          "      - .size: 4\n"
                          "        .offset: 80\n"
                          "        .value_kind: hidden_group_size_y\n"
                          "      - .size: 4\n"
                          "        .offset: 84\n"
                          "        .value_kind: hidden_group_size_z\n"
                          "      - .size: 4\n"
                          "        .offset: 88\n"
                          "        .value_kind: hidden_block_count_x\n"
                          "      - .size: 4\n"
                          "        .offset: 92\n"
                          "        .value_kind: hidden_block_count_y\n"
                          "      - .size: 4\n"
                          "        .offset: 96\n"
                          "        .value_kind: hidden_block_count_z\n"
                          "...\n"),
            std::string::npos)
      << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_kernarg_size 104")) << assembly;
}

TEST_F(KernelTest, ReplicatesReferenceSplitkReduceArgumentLayout) {
  Kernel kernel(target, "splitk_reduce", 256);
  KernelSignature &signature = kernel.signature();
  signature.addPointer("a");
  signature.addPointer("b");
  for (const char *name : {"s0", "s1", "s2", "s3"})
    signature.addScalar(name, 4);
  EXPECT_EQ(signature.arguments()[2].offset, 16);
  EXPECT_EQ(signature.arguments()[5].offset, 28);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4), 32);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4), 36);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4), 40);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4), 44);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4), 48);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4), 52);
  EXPECT_EQ(signature.segmentSize(), 56);
  kernel.builder().createEndProgram();

  std::string assembly = kernel.toAssembly();
  EXPECT_NE(assembly.find("        .value_kind: by_value\n"
                          "      - .size: 4\n"
                          "        .offset: 32\n"
                          "        .value_kind: hidden_group_size_x\n"),
            std::string::npos)
      << assembly;
  EXPECT_NE(assembly.find("        .offset: 52\n"
                          "        .value_kind: hidden_block_count_z\n...\n"),
            std::string::npos)
      << assembly;
}

TEST_F(KernelTest, AcceptsLlvmGroupSizeWidth) {
  KernelSignature signature;
  signature.addPointer("p");
  signature.addScalar("n", 4);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 2), 12);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 2), 14);
  EXPECT_EQ(signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4), 16);
  EXPECT_EQ(signature.segmentSize(), 24);
}

TEST_F(KernelTest, RejectsInvalidHiddenArguments) {
  KernelSignature signature;
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::byValue, 4),
               std::invalid_argument);
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::globalPointer, 8),
               std::invalid_argument);
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 2),
               std::invalid_argument);
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 8),
               std::invalid_argument);
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 8),
               std::invalid_argument);
  EXPECT_EQ(std::ssize(signature.arguments()), 0);

  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  EXPECT_THROW(signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4),
               std::invalid_argument);
  EXPECT_THROW(signature.addScalar("n", 4), std::invalid_argument);
  EXPECT_THROW(signature.addPointer("p"), std::invalid_argument);
  EXPECT_EQ(std::ssize(signature.arguments()), 1);
}

TEST_F(KernelTest, EnablesWorkgroupIds) {
  {
    Kernel kernel(target, "none", 64);
    EXPECT_THROW(kernel.workgroupIdY(), std::logic_error);
    EXPECT_THROW(kernel.workgroupIdZ(), std::logic_error);
    std::string assembly = kernel.toAssembly();
    EXPECT_EQ(assembly.find("workgroup_id_y"), std::string::npos) << assembly;
    EXPECT_EQ(assembly.find("workgroup_id_z"), std::string::npos) << assembly;
  }
  {
    Kernel kernel(target, "both", 64);
    kernel.enableWorkgroupIds(true, true);
    EXPECT_EQ(kernel.workgroupIdY().start(), 3);
    EXPECT_EQ(kernel.workgroupIdZ().start(), 4);
    EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 5);
    std::string assembly = kernel.toAssembly();
    expectLinesInOrder(assembly, {".amdhsa_system_sgpr_workgroup_id_x 1",
                                  ".amdhsa_system_sgpr_workgroup_id_y 1",
                                  ".amdhsa_system_sgpr_workgroup_id_z 1",
                                  ".amdhsa_system_vgpr_workitem_id 0"});
    EXPECT_THROW(kernel.enableWorkgroupIds(true, false), std::logic_error);
  }
  {
    Kernel kernel(target, "both_empty", 64);
    kernel.enableWorkgroupIds(true, true);
    EXPECT_TRUE(hasLine(kernel.toAssembly(), "  .amdhsa_next_free_sgpr 5"));
  }
  {
    Kernel kernel(target, "z_only", 64);
    kernel.enableWorkgroupIds(false, true);
    EXPECT_EQ(kernel.workgroupIdZ().start(), 3);
    EXPECT_THROW(kernel.workgroupIdY(), std::logic_error);
    std::string assembly = kernel.toAssembly();
    EXPECT_EQ(assembly.find("workgroup_id_y"), std::string::npos) << assembly;
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_system_sgpr_workgroup_id_z 1"))
        << assembly;
  }
}

TEST_F(KernelTest, RejectsInvalidWorkgroupIdRequests) {
  Kernel none(target, "none", 64);
  EXPECT_THROW(none.enableWorkgroupIds(false, false), std::invalid_argument);

  Kernel busy(target, "busy", 64);
  RegisterRange held = busy.builder().allocateSgpr(1);
  ASSERT_EQ(held.start(), 3);
  EXPECT_THROW(busy.enableWorkgroupIds(true, false), std::invalid_argument);
}

TEST_F(KernelTest, SetsLdsSize) {
  Kernel kernel(target, "lds", 64);
  EXPECT_EQ(kernel.ldsSize(), 0);
  kernel.setLdsSize(4096);
  EXPECT_EQ(kernel.ldsSize(), 4096);
  std::string assembly = kernel.toAssembly();
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_group_segment_fixed_size 4096"))
      << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .group_segment_fixed_size: 4096"))
      << assembly;
  EXPECT_THROW(kernel.setLdsSize(-1), std::invalid_argument);
  EXPECT_THROW(kernel.setLdsSize(163841), std::invalid_argument);
  EXPECT_EQ(kernel.ldsSize(), 4096);
  EXPECT_NO_THROW(kernel.setLdsSize(163840));
}

TEST_F(KernelTest, LoadsArguments) {
  Kernel kernel(target, "loads", 64);
  KernelSignature &signature = kernel.signature();
  signature.addPointer("p");
  signature.addScalar("n", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  IRBuilder &builder = kernel.builder();
  RegisterRange regs = builder.allocateSgpr(8, 4);
  kernel.loadArgument(regs.ref(0, 1), 8);
  kernel.loadArgument(regs.ref(2, 2), 8);
  kernel.loadArgument(regs.ref(4, 4), 0);
  expectLinesInOrder(kernel.toAssembly(),
                     {"s_load_dword s4, s[0:1], 8",
                      "s_load_dwordx2 s[6:7], s[0:1], 8",
                      "s_load_dwordx4 s[8:11], s[0:1], 0"});

  RegisterRange vgpr = builder.allocateVgpr(1);
  EXPECT_THROW(kernel.loadArgument(vgpr.ref(), 0), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 3), 0), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(1, 2), 0), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 1), 2), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 1), -4), std::invalid_argument);
  int end = signature.segmentSize();
  EXPECT_NO_THROW(kernel.loadArgument(regs.ref(0, 1), end - 4));
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 1), end), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 2), end - 4),
               std::invalid_argument);
}

} // namespace
