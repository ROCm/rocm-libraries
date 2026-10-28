// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "TestSupport.hpp"

#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

static bool hasLine(const std::string &text, const std::string &line) {
  return text.find("\n" + line + "\n") != std::string::npos;
}

static KernelSignature
pointerSignature(std::initializer_list<const char *> names) {
  KernelSignature signature;
  for (const char *name : names)
    signature.addPointer(name);
  return signature;
}

static KernelOptions optionsFor(CodeObjectVersion version) {
  KernelOptions options;
  options.abi.codeObjectVersion = version;
  return options;
}

namespace {

class KernelTest : public testing::Test {
protected:
  Target target = Target::get("gfx950");
};

} // namespace

TEST_F(KernelTest, DefaultEmitsCodeObjectV6) {
  Kernel kernel(target, "empty_kernel", 256, pointerSignature({"out"}));
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
      {CodeObjectVersion::v4, ".amdhsa_code_object_version 4",
       "amdhsa.version: [1, 1]"},
      {CodeObjectVersion::v5, ".amdhsa_code_object_version 5",
       "amdhsa.version: [1, 2]"},
      {CodeObjectVersion::v6, ".amdhsa_code_object_version 6",
       "amdhsa.version: [1, 2]"},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.directive);
    Kernel kernel(target, "k", 64, KernelSignature(), optionsFor(c.version));
    EXPECT_EQ(kernel.codeObjectVersion(), c.version);
    kernel.builder().createEndProgram();
    std::string assembly = kernel.toAssembly();
    EXPECT_TRUE(hasLine(assembly, c.directive)) << assembly;
    EXPECT_TRUE(hasLine(assembly, c.metadata)) << assembly;
  }
}

TEST_F(KernelTest, LaysOutArgumentsNaturally) {
  KernelSignature signature;
  EXPECT_EQ(signature.addPointer("x"), 0);
  EXPECT_EQ(signature.addPointer("y"), 8);
  EXPECT_EQ(signature.addScalar("a", 4), 16);
  EXPECT_EQ(signature.addScalar("n", 4), 20);
  Kernel kernel(target, "args_kernel", 64, std::move(signature));
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

  Kernel kernel(target, "no_args", 64, KernelSignature());
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
  const KernelSignature none;
  EXPECT_THROW(Kernel(target, "bad-name", 64, none), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "", 64, none), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "k", 0, none), std::invalid_argument);
  EXPECT_THROW(Kernel(target, "k", 1025, none), std::invalid_argument);
  EXPECT_NO_THROW(Kernel(target, "k", 1024, none));
  for (int32_t invalid : {3, 7})
    EXPECT_THROW(Kernel(target, "k", 64, none,
                        optionsFor(static_cast<CodeObjectVersion>(invalid))),
                 std::invalid_argument)
        << "version " << invalid;
}

static KernelSignature signatureWithHidden(KernelArgumentKind kind) {
  KernelSignature signature;
  signature.addPointer("p");
  signature.addScalar("n", 4);
  signature.addHidden(kind, 4);
  return signature;
}

static constexpr KernelArgumentKind hiddenKinds[] = {
    KernelArgumentKind::hiddenBlockCountX,
    KernelArgumentKind::hiddenBlockCountY,
    KernelArgumentKind::hiddenBlockCountZ,
    KernelArgumentKind::hiddenGroupSizeX,
    KernelArgumentKind::hiddenGroupSizeY,
    KernelArgumentKind::hiddenGroupSizeZ,
};

TEST_F(KernelTest, RejectsHiddenArgumentsAtCodeObjectV4) {
  for (KernelArgumentKind kind : hiddenKinds) {
    SCOPED_TRACE(static_cast<int32_t>(kind));
    EXPECT_THROW(Kernel(target, "k", 64, signatureWithHidden(kind),
                        optionsFor(CodeObjectVersion::v4)),
                 std::invalid_argument);
  }
}

TEST_F(KernelTest, AcceptsHiddenArgumentsFromCodeObjectV5) {
  for (CodeObjectVersion version :
       {CodeObjectVersion::v5, CodeObjectVersion::v6})
    for (KernelArgumentKind kind : hiddenKinds) {
      SCOPED_TRACE(static_cast<int32_t>(kind));
      EXPECT_NO_THROW(Kernel(target, "k", 64, signatureWithHidden(kind),
                             optionsFor(version)));
    }
}

TEST_F(KernelTest, AcceptsExplicitArgumentsAtCodeObjectV4) {
  EXPECT_NO_THROW(Kernel(target, "k", 64, pointerSignature({"a", "b"}),
                         optionsFor(CodeObjectVersion::v4)));
}

TEST_F(KernelTest, RejectsNegativePreloadDwords) {
  KernelOptions options;
  options.abi.maxPreloadDwords = -1;
  EXPECT_THROW(Kernel(target, "k", 64, KernelSignature(), options),
               std::invalid_argument);
  EXPECT_NO_THROW(Kernel(target, "k", 64, KernelSignature(), KernelOptions()));
}

TEST_F(KernelTest, ReservesAbiRegisters) {
  Kernel kernel(target, "k", 64, KernelSignature());
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
    Kernel kernel(target, "empty", 64, KernelSignature());
    std::string assembly = kernel.toAssembly();
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_accum_offset 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_vgpr 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_sgpr 3")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .sgpr_count: 9")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .vgpr_count: 8")) << assembly;
    EXPECT_TRUE(hasLine(assembly, "    .agpr_count: 0")) << assembly;
  }
  Kernel kernel(target, "grown", 64, KernelSignature());
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
  Kernel kernel(target, "accumulators", 64, KernelSignature());
  kernel.builder().allocateVgpr(10);
  kernel.builder().allocateAgpr(6);
  std::string assembly = kernel.toAssembly();
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_accum_offset 16")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_next_free_vgpr 22")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .vgpr_count: 22")) << assembly;
  EXPECT_TRUE(hasLine(assembly, "    .agpr_count: 6")) << assembly;
}

TEST_F(KernelTest, ExceedingAgprLimitThrows) {
  Kernel kernel(target, "k", 64, KernelSignature());
  kernel.builder().allocateAgpr(target.maxAgpr() + 1);
  EXPECT_THROW(kernel.toAssembly(), std::runtime_error);
}

TEST_F(KernelTest, ExceedingRegisterLimitThrows) {
  Kernel kernel(target, "k", 64, KernelSignature());
  kernel.builder().allocateVgpr(target.maxVgpr());
  EXPECT_THROW(kernel.toAssembly(), std::runtime_error);
}

static void addReferenceGemmExplicitArguments(KernelSignature &signature) {
  for (const char *name : {"a", "b", "c", "d", "e", "f"})
    signature.addPointer(name);
  for (const char *name : {"s0", "s1", "s2", "s3", "s4", "s5", "s6"})
    signature.addScalar(name, 4);
}

TEST_F(KernelTest, ReplicatesReferenceGemmArgumentLayout) {
  KernelSignature signature;
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
  Kernel kernel(target, "gemm", 256, std::move(signature));
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
  KernelSignature signature;
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
  Kernel kernel(target, "splitk_reduce", 256, std::move(signature));
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

static KernelOptions idOptions(bool y, bool z) {
  KernelOptions options;
  options.workgroupIdY = y;
  options.workgroupIdZ = z;
  return options;
}

TEST_F(KernelTest, DisablesWorkgroupIdsYAndZByDefault) {
  Kernel kernel(target, "none", 64, KernelSignature());
  EXPECT_FALSE(kernel.hasWorkgroupIdY());
  EXPECT_FALSE(kernel.hasWorkgroupIdZ());
  EXPECT_THROW(kernel.workgroupIdY(), std::logic_error);
  EXPECT_THROW(kernel.workgroupIdZ(), std::logic_error);
  std::string assembly = kernel.toAssembly();
  EXPECT_EQ(assembly.find("workgroup_id_y"), std::string::npos) << assembly;
  EXPECT_EQ(assembly.find("workgroup_id_z"), std::string::npos) << assembly;
}

TEST_F(KernelTest, PlacesWorkgroupIdsYAndZAfterX) {
  Kernel kernel(target, "both", 64, KernelSignature(), idOptions(true, true));
  EXPECT_EQ(kernel.workgroupIdX().start(), 2);
  EXPECT_EQ(kernel.workgroupIdY().start(), 3);
  EXPECT_EQ(kernel.workgroupIdZ().start(), 4);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 5);
  std::string assembly = kernel.toAssembly();
  expectLinesInOrder(assembly, {".amdhsa_system_sgpr_workgroup_id_x 1",
                                ".amdhsa_system_sgpr_workgroup_id_y 1",
                                ".amdhsa_system_sgpr_workgroup_id_z 1",
                                ".amdhsa_system_vgpr_workitem_id 0"});
}

TEST_F(KernelTest, CountsReservedWorkgroupIdsWithoutBody) {
  Kernel kernel(target, "both_empty", 64, KernelSignature(),
                idOptions(true, true));
  EXPECT_TRUE(hasLine(kernel.toAssembly(), "  .amdhsa_next_free_sgpr 5"));
}

TEST_F(KernelTest, PlacesOnlyWorkgroupIdY) {
  Kernel kernel(target, "y_only", 64, KernelSignature(),
                idOptions(true, false));
  EXPECT_TRUE(kernel.hasWorkgroupIdY());
  EXPECT_FALSE(kernel.hasWorkgroupIdZ());
  EXPECT_EQ(kernel.workgroupIdY().start(), 3);
  EXPECT_THROW(kernel.workgroupIdZ(), std::logic_error);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 4);
  std::string assembly = kernel.toAssembly();
  EXPECT_EQ(assembly.find("workgroup_id_z"), std::string::npos) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_system_sgpr_workgroup_id_y 1"))
      << assembly;
}

TEST_F(KernelTest, PlacesOnlyWorkgroupIdZ) {
  Kernel kernel(target, "z_only", 64, KernelSignature(),
                idOptions(false, true));
  EXPECT_FALSE(kernel.hasWorkgroupIdY());
  EXPECT_TRUE(kernel.hasWorkgroupIdZ());
  EXPECT_EQ(kernel.workgroupIdZ().start(), 3);
  EXPECT_THROW(kernel.workgroupIdY(), std::logic_error);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 4);
  std::string assembly = kernel.toAssembly();
  EXPECT_EQ(assembly.find("workgroup_id_y"), std::string::npos) << assembly;
  EXPECT_TRUE(hasLine(assembly, "  .amdhsa_system_sgpr_workgroup_id_z 1"))
      << assembly;
}

TEST_F(KernelTest, SetsLdsSize) {
  Kernel kernel(target, "lds", 64, KernelSignature());
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
  KernelSignature signature;
  signature.addPointer("p");
  signature.addScalar("n", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  int32_t end = signature.segmentSize();
  Kernel kernel(target, "loads", 64, std::move(signature));
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
  EXPECT_NO_THROW(kernel.loadArgument(regs.ref(0, 1), end - 4));
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 1), end), std::invalid_argument);
  EXPECT_THROW(kernel.loadArgument(regs.ref(0, 2), end - 4),
               std::invalid_argument);
}

TEST(KernelLaunch, EmptyKernelAssemblesLoadsAndLaunches) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_empty_kernel", 256, pointerSignature({"out"}));
  kernel.builder().createEndProgram();

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(kernel.toAssembly(), kernel.target().isaName());
  EXPECT_FALSE(hsaco.empty());

  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, "codegen_empty_kernel", module, function);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  void *out = nullptr;
  void *args[] = {&out};
  launchWithParams(function, 4, 256, args);
}

static constexpr int32_t roundTripGroupSize = 256;
static constexpr int32_t roundTripGroups = 4;
static constexpr int32_t roundTripElements =
    roundTripGroupSize * roundTripGroups;
static constexpr const char *roundTripName = "codegen_agpr_round_trip";

// Loads the in and out pointers from the kernarg segment into s[4:7].
static RegisterRange loadRoundTripPointers(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange pointers = builder.allocateSgpr(4, 4);
  builder.create<rocisa::SLoadB128>(pointers.container(),
                                    kernel.kernargSegmentPtr().container(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  return pointers;
}

// Moves each element through an AGPR. Index 3 of 4 is used rather than 0 so
// that a wrong accum_offset or next_free_vgpr would corrupt the result. No wait
// states are needed between the AGPR write and read on gfx90a and later.
static void emitAgprRoundTrip(Kernel &kernel, const RegisterRange &pointers,
                              const RegisterRange &globalId) {
  IRBuilder &builder = kernel.builder();
  RegisterRange byteOffset = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(byteOffset.container(), 2,
                                         globalId.container());
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::GlobalLoadB32>(
      value.container(), byteOffset.container(), pointers.sub(0, 2));
  builder.createWaitCnt({.vectorLoads = 0});
  RegisterRange acc = builder.allocateAgpr(4);
  builder.create<rocisa::VAccvgprWriteB32>(acc.sub(3), value.container());
  RegisterRange result = builder.allocateVgpr(1);
  builder.create<rocisa::VAccvgprReadB32>(result.container(), acc.sub(3));
  builder.create<rocisa::GlobalStoreB32>(
      byteOffset.container(), result.container(), pointers.sub(2, 2));
}

static void buildAgprRoundTrip(Kernel &kernel) {
  RegisterRange pointers = loadRoundTripPointers(kernel);
  RegisterRange globalId = computeGlobalId(kernel);
  emitAgprRoundTrip(kernel, pointers, globalId);
  kernel.builder().createEndProgram();
}

// Uploads in, launches the kernel and downloads the result into out.
static void runRoundTrip(const std::vector<uint8_t> &hsaco, const char *name,
                         int32_t groups, int32_t groupSize,
                         const std::vector<int32_t> &in,
                         std::vector<int32_t> &out) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, name, module, function);
  DevicePtr deviceIn;
  DevicePtr deviceOut;
  upload(in, deviceIn);
  // The sentinel shows any element the kernel failed to write.
  uploadBytes(nullptr, out.size() * sizeof(int32_t), deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  void *rawIn = deviceIn.get();
  void *rawOut = deviceOut.get();
  void *args[] = {&rawIn, &rawOut};
  launchWithParams(function, groups, groupSize, args);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceOut, out);
}

TEST(KernelLaunch, AgprRoundTripOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, roundTripName, roundTripGroupSize,
                pointerSignature({"in", "out"}));
  buildAgprRoundTrip(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {"v_accvgpr_write_b32 acc3, v3", "v_accvgpr_read_b32 v4, acc3",
       "global_store_dword v2, v4, s[6:7]", ".amdhsa_accum_offset 8",
       ".amdhsa_next_free_vgpr 12", ".vgpr_count: 12", ".agpr_count: 4"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> in(roundTripElements);
  for (int32_t i = 0; i < roundTripElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(roundTripElements, -1);
  runRoundTrip(hsaco, roundTripName, roundTripGroups, roundTripGroupSize, in,
               out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int32_t i = 0; i < roundTripElements; ++i)
    ASSERT_EQ(out[i], in[i]) << "element " << i;
}

static constexpr const char *idsName = "codegen_ids_hidden";
static constexpr int32_t idsGrid[3] = {3, 4, 5};
static constexpr int32_t idsBlock[3] = {8, 2, 2};
static constexpr int32_t idsGroups = idsGrid[0] * idsGrid[1] * idsGrid[2];
static constexpr int32_t idsBlockSize = idsBlock[0] * idsBlock[1] * idsBlock[2];
// Workgroup ids x, y, z, group sizes x, y, z and block counts x, y, z.
static constexpr int32_t idsDwords = 9;

// Mimics the reference kernel: the hidden arguments start at offset 12, which
// is not 8-aligned, and follow the explicit ones.
static void addIdsSignature(KernelSignature &signature) {
  signature.addPointer("out");
  signature.addScalar("tag", 4);
  for (KernelArgumentKind kind : {KernelArgumentKind::hiddenGroupSizeX,
                                  KernelArgumentKind::hiddenGroupSizeY,
                                  KernelArgumentKind::hiddenGroupSizeZ,
                                  KernelArgumentKind::hiddenBlockCountX,
                                  KernelArgumentKind::hiddenBlockCountY,
                                  KernelArgumentKind::hiddenBlockCountZ})
    signature.addHidden(kind, 4);
}

// Copies the ids to words 0..2 of values and loads the hidden arguments into
// words 3..8, which are the group sizes followed by the block counts.
static void loadIdsAndHiddenArguments(Kernel &kernel,
                                      const RegisterRange &values) {
  IRBuilder &builder = kernel.builder();
  const RegisterRange *ids[] = {&kernel.workgroupIdX(), &kernel.workgroupIdY(),
                                &kernel.workgroupIdZ()};
  for (int32_t i = 0; i < 3; ++i)
    builder.create<rocisa::SMovB32>(values.sub(i), ids[i]->container());
  for (int32_t i = 0; i < 6; ++i)
    kernel.loadArgument(values.ref(3 + i), 12 + 4 * i);
  builder.createWaitCnt({.scalarMemory = 0});
  // The runtime writes only the low 2 bytes of a group size (LLVM's width), so
  // the upper half of a 4-byte slot is undefined.
  for (int32_t i = 3; i < 6; ++i)
    builder.create<rocisa::SAndB32>(values.sub(i), values.sub(i), 0xffff);
}

// Returns the byte offset of this group's record, computed from the loaded
// block counts so that a wrong count corrupts the layout.
static RegisterRange computeRecordOffset(Kernel &kernel,
                                         const RegisterRange &values) {
  IRBuilder &builder = kernel.builder();
  RegisterRange linear = builder.allocateSgpr(1);
  // linear = x + countX * (y + countY * z).
  builder.create<rocisa::SMulI32>(linear.container(), values.sub(7),
                                  values.sub(2));
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(1));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  values.sub(6));
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(0));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  4 * idsDwords);
  RegisterRange offset = builder.allocateVgpr(1);
  builder.create<rocisa::VMovB32>(offset.container(), linear.container());
  return offset;
}

static void buildIdsAndHiddenKernel(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange out = builder.allocateSgpr(2, 2);
  RegisterRange values = builder.allocateSgpr(idsDwords);
  kernel.loadArgument(out.ref(), 0);
  loadIdsAndHiddenArguments(kernel, values);
  RegisterRange base = computeRecordOffset(kernel, values);
  // One lane stores each record; the kernel ends right after, so shrinking
  // exec is safe.
  builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(), 1);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange data = builder.allocateVgpr(1);
  for (int32_t i = 0; i < idsDwords; ++i) {
    builder.create<rocisa::VAddU32>(address.container(), 4 * i,
                                    base.container());
    builder.create<rocisa::VMovB32>(data.container(), values.sub(i));
    builder.create<rocisa::GlobalStoreB32>(address.container(),
                                           data.container(), out.container());
  }
  builder.createEndProgram();
}

// Runs the kernel over idsGrid with idsBlock and downloads the records.
static void runIdsAndHidden(const std::vector<uint8_t> &hsaco,
                            std::vector<int32_t> &records) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, idsName, module, function);
  DevicePtr deviceOut;
  // The sentinel shows any record the kernel failed to write.
  uploadBytes(nullptr, records.size() * sizeof(int32_t), deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  void *out = deviceOut.get();
  int32_t tag = 0;
  void *args[] = {&out, &tag};
  launchWithParams(function, dim3(idsGrid[0], idsGrid[1], idsGrid[2]),
                   dim3(idsBlock[0], idsBlock[1], idsBlock[2]), args);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceOut, records);
}

TEST(KernelLaunch, WorkgroupIdsAndHiddenArgumentsOnDevice) {
  Target target = Target::get("gfx950");
  KernelSignature signature;
  addIdsSignature(signature);
  Kernel kernel(target, idsName, idsBlockSize, std::move(signature),
                idOptions(true, true));
  buildIdsAndHiddenKernel(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(assembly, {".amdhsa_system_sgpr_workgroup_id_x 1",
                                ".amdhsa_system_sgpr_workgroup_id_y 1",
                                ".amdhsa_system_sgpr_workgroup_id_z 1"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> records(idsGroups * idsDwords, -1);
  runIdsAndHidden(hsaco, records);
  if (testing::Test::HasFatalFailure())
    return;

  for (int32_t z = 0; z < idsGrid[2]; ++z)
    for (int32_t y = 0; y < idsGrid[1]; ++y)
      for (int32_t x = 0; x < idsGrid[0]; ++x) {
        int32_t linear = x + idsGrid[0] * (y + idsGrid[1] * z);
        const int32_t expected[idsDwords] = {
            x,           y,          z,          idsBlock[0], idsBlock[1],
            idsBlock[2], idsGrid[0], idsGrid[1], idsGrid[2]};
        for (int32_t i = 0; i < idsDwords; ++i)
          ASSERT_EQ(records[linear * idsDwords + i], expected[i])
              << "group (" << x << ", " << y << ", " << z << ") word " << i;
      }
}

static constexpr const char *abiName = "codegen_abi_versions";
static constexpr int32_t abiGrid[3] = {3, 2, 2};
static constexpr int32_t abiGroups = abiGrid[0] * abiGrid[1] * abiGrid[2];
static constexpr int32_t abiScalarA = 0x1234;
static constexpr int32_t abiScalarB = -77;
// Workgroup ids x, y, z followed by the two scalar arguments.
static constexpr int32_t abiDwords = 5;
// Byte offset of e_ident[EI_ABIVERSION] in an ELF file.
static constexpr int32_t elfAbiVersionIndex = 8;

static KernelSignature abiSignature() {
  KernelSignature signature;
  signature.addPointer("out");
  signature.addScalar("a", 4);
  signature.addScalar("b", 4);
  return signature;
}

// Returns the byte offset of this workgroup's record in a VGPR.
static RegisterRange computeAbiRecordOffset(Kernel &kernel,
                                            const RegisterRange &values) {
  IRBuilder &builder = kernel.builder();
  RegisterRange linear = builder.allocateSgpr(1);
  // linear = x + gridX * (y + gridY * z).
  builder.create<rocisa::SMulI32>(linear.container(), values.sub(2),
                                  abiGrid[1]);
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(1));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  abiGrid[0]);
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(0));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  4 * abiDwords);
  RegisterRange offset = builder.allocateVgpr(1);
  builder.create<rocisa::VMovB32>(offset.container(), linear.container());
  return offset;
}

// Stores the ids and the two arguments of every workgroup from lane 0.
static void buildAbiKernel(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange out = builder.allocateSgpr(2, 2);
  RegisterRange values = builder.allocateSgpr(abiDwords);
  kernel.loadArgument(out.ref(), 0);
  const RegisterRange *ids[] = {&kernel.workgroupIdX(), &kernel.workgroupIdY(),
                                &kernel.workgroupIdZ()};
  for (int32_t i = 0; i < 3; ++i)
    builder.create<rocisa::SMovB32>(values.sub(i), ids[i]->container());
  kernel.loadArgument(values.ref(3), 8);
  kernel.loadArgument(values.ref(4), 12);
  builder.createWaitCnt({.scalarMemory = 0});
  RegisterRange base = computeAbiRecordOffset(kernel, values);
  // The kernel ends right after the stores, so shrinking exec is safe.
  builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(), 1);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange data = builder.allocateVgpr(1);
  for (int32_t i = 0; i < abiDwords; ++i) {
    builder.create<rocisa::VAddU32>(address.container(), 4 * i,
                                    base.container());
    builder.create<rocisa::VMovB32>(data.container(), values.sub(i));
    builder.create<rocisa::GlobalStoreB32>(address.container(),
                                           data.container(), out.container());
  }
  builder.createEndProgram();
}

static void runAbiKernel(const std::vector<uint8_t> &hsaco,
                         std::vector<int32_t> &records) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, abiName, module, function);
  DevicePtr deviceOut;
  // The sentinel shows any record the kernel failed to write.
  uploadBytes(nullptr, records.size() * sizeof(int32_t), deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  void *out = deviceOut.get();
  int32_t a = abiScalarA;
  int32_t b = abiScalarB;
  void *args[] = {&out, &a, &b};
  launchWithParams(function, dim3(abiGrid[0], abiGrid[1], abiGrid[2]), 64,
                   args);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceOut, records);
}

static void checkAbiRecords(const std::vector<int32_t> &records) {
  for (int32_t z = 0; z < abiGrid[2]; ++z)
    for (int32_t y = 0; y < abiGrid[1]; ++y)
      for (int32_t x = 0; x < abiGrid[0]; ++x) {
        int32_t linear = x + abiGrid[0] * (y + abiGrid[1] * z);
        const int32_t expected[abiDwords] = {x, y, z, abiScalarA, abiScalarB};
        for (int32_t i = 0; i < abiDwords; ++i)
          ASSERT_EQ(records[linear * abiDwords + i], expected[i])
              << "group (" << x << ", " << y << ", " << z << ") word " << i;
      }
}

static void checkArgumentsAndIdsAtVersion(CodeObjectVersion version,
                                          uint8_t elfAbiVersion) {
  Target target = Target::get("gfx950");
  KernelOptions options = optionsFor(version);
  options.workgroupIdY = true;
  options.workgroupIdZ = true;
  Kernel kernel(target, abiName, 64, abiSignature(), options);
  for (const KernelArgument &arg : kernel.signature().arguments())
    ASSERT_FALSE(arg.name.empty()) << "unexpected hidden argument";
  buildAbiKernel(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_GT(std::ssize(hsaco), elfAbiVersionIndex);
  EXPECT_EQ(hsaco[elfAbiVersionIndex], elfAbiVersion);
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> records(abiGroups * abiDwords, -1);
  runAbiKernel(hsaco, records);
  if (testing::Test::HasFatalFailure())
    return;
  checkAbiRecords(records);
}

TEST(KernelLaunch, ArgumentsAndWorkgroupIdsOnDeviceAtCodeObjectV4) {
  checkArgumentsAndIdsAtVersion(CodeObjectVersion::v4, 2);
}

TEST(KernelLaunch, ArgumentsAndWorkgroupIdsOnDeviceAtCodeObjectV5) {
  checkArgumentsAndIdsAtVersion(CodeObjectVersion::v5, 3);
}

TEST(KernelLaunch, ArgumentsAndWorkgroupIdsOnDeviceAtCodeObjectV6) {
  checkArgumentsAndIdsAtVersion(CodeObjectVersion::v6, 4);
}

static constexpr const char *ldsName = "codegen_lds_round_trip";
static constexpr int32_t ldsGroupSize = 256;
static constexpr int32_t ldsGroups = 4;
static constexpr int32_t ldsElements = ldsGroupSize * ldsGroups;

// Returns the LDS byte address of lane (tid + delta) % ldsGroupSize.
static RegisterRange computeLdsAddress(Kernel &kernel, int32_t delta) {
  IRBuilder &builder = kernel.builder();
  RegisterRange address = builder.allocateVgpr(1);
  builder.create<rocisa::VAddU32>(address.container(), delta,
                                  kernel.workitemId().container());
  builder.create<rocisa::VAndB32>(address.container(), ldsGroupSize - 1,
                                  address.container());
  builder.create<rocisa::VLShiftLeftB32>(address.container(), 2,
                                         address.container());
  return address;
}

// Each lane publishes its element in LDS and reads its neighbor's, which sits
// in another wave at wave boundaries, so the barrier is what makes it correct.
static void emitLdsExchange(Kernel &kernel, const RegisterRange &pointers,
                            const RegisterRange &globalId) {
  IRBuilder &builder = kernel.builder();
  RegisterRange byteOffset = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(byteOffset.container(), 2,
                                         globalId.container());
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::GlobalLoadB32>(
      value.container(), byteOffset.container(), pointers.sub(0, 2));
  builder.createWaitCnt({.vectorLoads = 0});
  RegisterRange ownAddress = computeLdsAddress(kernel, 0);
  builder.create<rocisa::DSStoreB32>(ownAddress.container(), value.container());
  builder.createWaitCnt({.lds = 0});
  builder.create<rocisa::SBarrier>();
  RegisterRange neighborAddress = computeLdsAddress(kernel, 1);
  RegisterRange neighbor = builder.allocateVgpr(1);
  builder.create<rocisa::DSLoadB32>(neighbor.container(),
                                    neighborAddress.container());
  builder.createWaitCnt({.lds = 0});
  builder.create<rocisa::GlobalStoreB32>(
      byteOffset.container(), neighbor.container(), pointers.sub(2, 2));
}

static void buildLdsRoundTrip(Kernel &kernel) {
  kernel.setLdsSize(4 * ldsGroupSize);
  IRBuilder &builder = kernel.builder();
  RegisterRange pointers = builder.allocateSgpr(4, 4);
  kernel.loadArgument(pointers.ref(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  RegisterRange globalId = computeGlobalId(kernel);
  emitLdsExchange(kernel, pointers, globalId);
  builder.createEndProgram();
}

TEST(KernelLaunch, LdsRoundTripOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, ldsName, ldsGroupSize, pointerSignature({"in", "out"}));
  buildLdsRoundTrip(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(assembly,
                     {"ds_write_b32", "s_waitcnt lgkmcnt(0)", "s_barrier",
                      "ds_read_b32", ".amdhsa_group_segment_fixed_size 1024"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> in(ldsElements);
  for (int32_t i = 0; i < ldsElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(ldsElements, -1);
  runRoundTrip(hsaco, ldsName, ldsGroups, ldsGroupSize, in, out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int32_t g = 0; g < ldsGroups; ++g)
    for (int32_t t = 0; t < ldsGroupSize; ++t)
      ASSERT_EQ(out[g * ldsGroupSize + t],
                in[g * ldsGroupSize + (t + 1) % ldsGroupSize])
          << "group " << g << " lane " << t;
}

// Kernarg preloading.

static KernelOptions
preloadOptions(int32_t request,
               CodeObjectVersion version = CodeObjectVersion::v6,
               bool y = false, bool z = false) {
  KernelOptions options = optionsFor(version);
  options.abi.maxPreloadDwords = request;
  options.workgroupIdY = y;
  options.workgroupIdZ = z;
  return options;
}

// The assembly after the compat prologue, which the body label ends.
static std::string bodyText(const std::string &assembly) {
  std::size_t label = assembly.find("_body:\n");
  if (label == std::string::npos)
    return assembly;
  return assembly.substr(label);
}

// A pointer followed by count 4-byte scalars named a0, a1, and so on.
static KernelSignature pointerAndScalars(int32_t count) {
  KernelSignature signature;
  signature.addPointer("out");
  for (int32_t i = 0; i < count; ++i)
    signature.addScalar("a" + std::to_string(i), 4);
  return signature;
}

TEST_F(KernelTest, PreloadDirectivesFollowTheWindow) {
  struct Case {
    bool y, z;
    int32_t idY, idZ;
    std::vector<std::string_view> directives;
  };
  // The window is the pointer and 3 scalars: 5 dwords in s[2:6].
  const Case cases[] = {
      {false, false, -1, -1, {}},
      {true, false, 8, -1, {".amdhsa_system_sgpr_workgroup_id_y 1"}},
      {false, true, -1, 8, {".amdhsa_system_sgpr_workgroup_id_z 1"}},
      {true,
       true,
       8,
       9,
       {".amdhsa_system_sgpr_workgroup_id_y 1",
        ".amdhsa_system_sgpr_workgroup_id_z 1"}},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(std::string(c.y ? "y" : "") + (c.z ? "z" : ""));
    Kernel kernel(target, "k", 64, pointerAndScalars(3),
                  preloadOptions(14, CodeObjectVersion::v6, c.y, c.z));
    EXPECT_EQ(kernel.preloadDwords(), 5);
    EXPECT_EQ(kernel.workgroupIdX().start(), 7);
    if (c.y)
      EXPECT_EQ(kernel.workgroupIdY().start(), c.idY);
    if (c.z)
      EXPECT_EQ(kernel.workgroupIdZ().start(), c.idZ);
    kernel.builder().createEndProgram();
    std::string assembly = kernel.toAssembly();
    std::vector<std::string_view> needles = {
        "  .amdhsa_user_sgpr_kernarg_segment_ptr 1",
        "  .amdhsa_user_sgpr_count 7",
        "  .amdhsa_user_sgpr_kernarg_preload_length 5",
        "  .amdhsa_user_sgpr_kernarg_preload_offset 0",
        "  .amdhsa_system_sgpr_workgroup_id_x 1"};
    needles.insert(needles.end(), c.directives.begin(), c.directives.end());
    expectLinesInOrder(assembly, needles);
  }
}

TEST_F(KernelTest, PreloadOffEmitsNoPreloadDirectives) {
  for (int32_t request : {0, 1}) {
    // A request of 1 clamps to 0 dwords: the pointer needs 2.
    Kernel kernel(target, "k", 64, pointerAndScalars(3),
                  preloadOptions(request));
    EXPECT_EQ(kernel.preloadDwords(), 0);
    kernel.builder().createEndProgram();
    std::string assembly = kernel.toAssembly();
    EXPECT_EQ(assembly.find("user_sgpr_count"), std::string::npos) << assembly;
    EXPECT_EQ(assembly.find("preload"), std::string::npos) << assembly;
    EXPECT_EQ(countLinesContaining(assembly, ".p2align 8"), 1) << assembly;
  }
}

TEST_F(KernelTest, RoundsPreloadWindowToWholeArguments) {
  KernelSignature twoPointers = pointerSignature({"a", "b"});
  twoPointers.addScalar("n", 4);
  EXPECT_EQ(
      Kernel(target, "k", 64, twoPointers, preloadOptions(3)).preloadDwords(),
      2);

  EXPECT_EQ(Kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(3))
                .preloadDwords(),
            3);
  EXPECT_EQ(Kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(1))
                .preloadDwords(),
            0);
  // The request is only an upper bound.
  EXPECT_EQ(Kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(100))
                .preloadDwords(),
            3);
  EXPECT_EQ(Kernel(target, "k", 64, KernelSignature(), preloadOptions(5))
                .preloadDwords(),
            0);
}

TEST_F(KernelTest, ClampsPreloadWindowToUserSgprs) {
  // The pointer and 12 scalars fill the 14 dwords next to the kernarg pointer.
  Kernel exact(target, "k", 64, pointerAndScalars(12), preloadOptions(100));
  EXPECT_EQ(exact.preloadDwords(), 14);
  EXPECT_EQ(exact.workgroupIdX().start(), 16);
  Kernel longer(target, "k", 64, pointerAndScalars(20), preloadOptions(100));
  EXPECT_EQ(longer.preloadDwords(), 14);
  Kernel capped(target, "k", 64, pointerAndScalars(20), preloadOptions(6));
  EXPECT_EQ(capped.preloadDwords(), 6);
}

TEST_F(KernelTest, PreloadWindowContainsPadding) {
  KernelSignature signature;
  signature.addScalar("n", 4);
  signature.addPointer("p");
  // The 4 byte padding after n belongs to the window but to no argument.
  EXPECT_EQ(
      Kernel(target, "k", 64, signature, preloadOptions(4)).preloadDwords(), 4);
  EXPECT_EQ(
      Kernel(target, "k", 64, signature, preloadOptions(3)).preloadDwords(), 1);
}

TEST_F(KernelTest, PreloadWindowExcludesHiddenArguments) {
  KernelSignature signature = pointerAndScalars(1);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  Kernel kernel(target, "k", 64, std::move(signature),
                preloadOptions(100, CodeObjectVersion::v5));
  EXPECT_EQ(kernel.preloadDwords(), 3);
  EXPECT_FALSE(kernel.takePreloadedArgument(12));
}

// The target is the only supported one, so the unsupported-target error
// (std::invalid_argument for a positive request) cannot be constructed here.

TEST_F(KernelTest, ReservesPreloadedArgumentRegisters) {
  Kernel kernel(target, "k", 64, pointerAndScalars(2), preloadOptions(100));
  ASSERT_EQ(kernel.preloadDwords(), 4);
  std::optional<RegisterRange> out = kernel.takePreloadedArgument(0);
  ASSERT_TRUE(out);
  EXPECT_EQ(out->start(), 2);
  EXPECT_EQ(out->count(), 2);
  std::optional<RegisterRange> second = kernel.takePreloadedArgument(12);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->start(), 5);
  EXPECT_EQ(second->count(), 1);
  // The workgroup id follows the window.
  EXPECT_EQ(kernel.workgroupIdX().start(), 6);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 7);
}

TEST_F(KernelTest, LoadsArgumentsInsideThePreloadWindow) {
  // The window s[2:6] holds out in s[2:3] and a0, a1, a2 in s4, s5, s6.
  Kernel kernel(target, "k", 64, pointerAndScalars(3), preloadOptions(100));
  RegisterRange regs = kernel.builder().allocateSgpr(8, 4);
  ASSERT_EQ(regs.start(), 8);
  kernel.loadArgument(regs.ref(0, 2), 0);
  kernel.loadArgument(regs.ref(2, 1), 12);
  std::string assembly = bodyText(kernel.toAssembly());
  expectLinesInOrder(assembly,
                     {"s_mov_b64 s[8:9], s[2:3]", "s_mov_b32 s10, s5"});
  EXPECT_EQ(assembly.find("s_load"), std::string::npos) << assembly;
}

TEST_F(KernelTest, SplitsLoadsPerArgument) {
  Kernel kernel(target, "k", 64, pointerAndScalars(3), preloadOptions(100));
  RegisterRange regs = kernel.builder().allocateSgpr(4, 4);
  kernel.loadArgument(regs.ref(0, 4), 0);
  std::string assembly = bodyText(kernel.toAssembly());
  expectLinesInOrder(assembly, {"s_mov_b64 s[8:9], s[2:3]", "s_mov_b32 s10, s4",
                                "s_mov_b32 s11, s5"});
  EXPECT_EQ(assembly.find("s_load"), std::string::npos) << assembly;
}

TEST_F(KernelTest, LoadsStraddlingThePreloadWindowEdge) {
  // The window ends after a2; a3 is not preloaded.
  Kernel kernel(target, "k", 64, pointerAndScalars(4), preloadOptions(5));
  ASSERT_EQ(kernel.preloadDwords(), 5);
  RegisterRange regs = kernel.builder().allocateSgpr(4, 4);
  ASSERT_EQ(regs.start(), 8);
  kernel.loadArgument(regs.ref(0, 4), 8);
  std::string assembly = bodyText(kernel.toAssembly());
  expectLinesInOrder(assembly,
                     {"s_mov_b32 s8, s4", "s_mov_b32 s9, s5",
                      "s_mov_b32 s10, s6", "s_load_dword s11, s[0:1], 20"});
  EXPECT_EQ(countLinesContaining(assembly, "s_load"), 1) << assembly;
}

TEST_F(KernelTest, LoadsOutsideThePreloadWindow) {
  // The window is the pointer alone.
  Kernel kernel(target, "k", 64, pointerAndScalars(4), preloadOptions(2));
  RegisterRange regs = kernel.builder().allocateSgpr(4, 4);
  kernel.loadArgument(regs.ref(0, 4), 8);
  std::string assembly = kernel.toAssembly();
  expectLinesInOrder(assembly, {"s_load_dwordx4 s[8:11], s[0:1], 8"});
  EXPECT_EQ(assembly.find("s_mov"), std::string::npos) << assembly;
}

TEST_F(KernelTest, SplitsUnheldLoadsByDestinationAlignment) {
  // a0 is held; the next 3 dwords start at the odd register s9.
  Kernel kernel(target, "k", 64, pointerAndScalars(4), preloadOptions(3));
  RegisterRange regs = kernel.builder().allocateSgpr(4, 4);
  kernel.loadArgument(regs.ref(0, 4), 8);
  expectLinesInOrder(kernel.toAssembly(),
                     {"s_mov_b32 s8, s4", "s_load_dword s9, s[0:1], 12",
                      "s_load_dwordx2 s[10:11], s[0:1], 16"});
}

TEST_F(KernelTest, LoadsAfterTakeOrReleaseFallBackToSLoad) {
  for (bool take : {true, false}) {
    SCOPED_TRACE(take ? "take" : "release");
    Kernel kernel(target, "k", 64, pointerAndScalars(3), preloadOptions(100));
    RegisterRange regs = kernel.builder().allocateSgpr(4, 4);
    std::optional<RegisterRange> taken;
    if (take)
      taken = kernel.takePreloadedArgument(0);
    else
      kernel.releasePreloadedArgument(0);
    kernel.loadArgument(regs.ref(0, 2), 0);
    kernel.loadArgument(regs.ref(2, 1), 8);
    std::string assembly = bodyText(kernel.toAssembly());
    // Only the pointer left the window; a0 is still copied.
    expectLinesInOrder(
        assembly, {"s_load_dwordx2 s[8:9], s[0:1], 0", "s_mov_b32 s10, s4"});
    EXPECT_EQ(countLinesContaining(assembly, "s_load"), 1) << assembly;
  }
}

TEST_F(KernelTest, ReleasedPreloadRegistersAreReused) {
  Kernel kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(3));
  // out is s[2:3], a0 is s4, the workgroup id is s5.
  EXPECT_EQ(kernel.builder().allocateSgpr(2, 2).start(), 6);
  kernel.releasePreloadedArgument(0);
  EXPECT_EQ(kernel.builder().allocateSgpr(2, 2).start(), 2);
  kernel.releasePreloadedArgument(8);
  EXPECT_EQ(kernel.builder().allocateSgpr(1).start(), 2);
}

TEST_F(KernelTest, TakenPreloadRegistersAreReusedOnceDropped) {
  Kernel kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(3));
  std::optional<RegisterRange> taken = kernel.takePreloadedArgument(0);
  ASSERT_TRUE(taken);
  EXPECT_EQ(kernel.builder().allocateSgpr(2, 2).start(), 6);
  taken.reset();
  EXPECT_EQ(kernel.builder().allocateSgpr(2, 2).start(), 2);
}

TEST_F(KernelTest, TakesAndReleasesPreloadedArguments) {
  KernelSignature signature = pointerAndScalars(1);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  Kernel kernel(target, "k", 64, std::move(signature),
                preloadOptions(3, CodeObjectVersion::v5));
  // Only the start offsets of arguments are valid, hidden ones included.
  for (int32_t offset : {-4, 2, 4, 20, 100}) {
    EXPECT_THROW(kernel.takePreloadedArgument(offset), std::invalid_argument)
        << offset;
    EXPECT_THROW(kernel.releasePreloadedArgument(offset), std::invalid_argument)
        << offset;
  }
  EXPECT_FALSE(kernel.takePreloadedArgument(12));
  EXPECT_NO_THROW(kernel.releasePreloadedArgument(12));

  EXPECT_TRUE(kernel.takePreloadedArgument(8));
  EXPECT_FALSE(kernel.takePreloadedArgument(8));
  kernel.releasePreloadedArgument(0);
  kernel.releasePreloadedArgument(0);
  EXPECT_FALSE(kernel.takePreloadedArgument(0));
}

TEST_F(KernelTest, TakesNothingWithoutPreload) {
  Kernel kernel(target, "k", 64, pointerAndScalars(1), preloadOptions(0));
  EXPECT_FALSE(kernel.takePreloadedArgument(0));
  EXPECT_NO_THROW(kernel.releasePreloadedArgument(8));
  EXPECT_THROW(kernel.takePreloadedArgument(4), std::invalid_argument);
}

// Preload prologue.

static constexpr const char *objdumpPath = "/opt/rocm/llvm/bin/llvm-objdump";
// Hardware with kernarg preload support enters the kernel this many bytes after
// the symbol.
static constexpr int64_t preloadEntryBytes = 256;

namespace {

struct DisassembledInstruction {
  int64_t address;
  std::string text;
};

} // namespace

// Returns the instructions of the kernel in hsaco and sets entry to its
// address; entry must be -1 on the call.
static std::vector<DisassembledInstruction>
disassemble(const std::vector<uint8_t> &hsaco, int64_t &entry) {
  char path[] = "/tmp/codegen_preload_XXXXXX";
  int fd = mkstemp(path);
  EXPECT_GE(fd, 0);
  if (fd < 0)
    return {};
  ssize_t written = write(fd, hsaco.data(), hsaco.size());
  close(fd);
  EXPECT_EQ(written, std::ssize(hsaco));
  std::string command =
      std::string(objdumpPath) + " -d --mcpu=gfx950 " + path + " 2>&1";
  FILE *pipe = popen(command.c_str(), "r");
  std::string output;
  if (pipe) {
    char chunk[4096];
    for (std::size_t count; (count = fread(chunk, 1, sizeof(chunk), pipe)) > 0;)
      output.append(chunk, count);
    pclose(pipe);
  }
  unlink(path);

  std::vector<DisassembledInstruction> instructions;
  std::regex header("^([0-9a-f]+) <.*>:$");
  std::regex instruction("^\\s+(\\S.*?)\\s+// ([0-9A-F]+):");
  std::istringstream stream(output);
  for (std::string current; std::getline(stream, current);) {
    std::smatch match;
    // The body label is a symbol too, so only the first header is the kernel.
    if (std::regex_match(current, match, header)) {
      if (entry < 0)
        entry = std::stoll(match[1], nullptr, 16);
    } else if (std::regex_search(current, match, instruction))
      instructions.push_back({std::stoll(match[2], nullptr, 16), match[1]});
  }
  return instructions;
}

TEST(KernelPreloadPrologue, BodyStartsAtTheHardwareEntryOffset) {
  if (access(objdumpPath, X_OK) != 0)
    GTEST_SKIP() << "no llvm-objdump";
  Target target = Target::get("gfx950");
  struct Case {
    int32_t scalars;
    int32_t request;
    std::vector<std::string_view> loads;
  };
  // The loads are the widest whose destination is aligned: x8 and x4 need a
  // multiple of 4, x2 an even register, starting at s2.
  const Case cases[] = {
      {0, 2, {"s_load_dwordx2 s[2:3], s[0:1], 0x0"}},
      {1, 3, {"s_load_dwordx2 s[2:3], s[0:1], 0x0", "s_load_dword s4,"}},
      {3,
       5,
       {"s_load_dwordx2 s[2:3], s[0:1], 0x0",
        "s_load_dwordx2 s[4:5], s[0:1], 0x8", "s_load_dword s6,"}},
      {12,
       14,
       {"s_load_dwordx2 s[2:3], s[0:1], 0x0",
        "s_load_dwordx8 s[4:11], s[0:1], 0x8",
        "s_load_dwordx4 s[12:15], s[0:1], 0x28"}},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE("request " + std::to_string(c.request));
    Kernel kernel(target, "codegen_preload_prologue", 64,
                  pointerAndScalars(c.scalars), preloadOptions(c.request));
    ASSERT_EQ(kernel.preloadDwords(), c.request);
    kernel.builder().createEndProgram();
    std::string assembly = kernel.toAssembly();
    SCOPED_TRACE("assembly:\n" + assembly);
    std::vector<uint8_t> hsaco =
        assembleToHSACO(assembly, kernel.target().isaName());
    ASSERT_FALSE(hsaco.empty());

    int64_t entry = -1;
    std::vector<DisassembledInstruction> instructions =
        disassemble(hsaco, entry);
    ASSERT_GE(entry, 0);
    ASSERT_GT(std::ssize(instructions), std::ssize(c.loads) + 2);
    int64_t next = 0;
    for (std::string_view load : c.loads) {
      EXPECT_EQ(instructions[next].address, entry + 8 * next);
      EXPECT_NE(instructions[next].text.find(load), std::string::npos)
          << instructions[next].text;
      ++next;
    }
    EXPECT_EQ(instructions[next].text, "s_waitcnt lgkmcnt(0)");
    EXPECT_EQ(instructions[next + 1].text.find("s_branch"), 0u);
    // Skip the padding to the first body instruction.
    int64_t body = next + 2;
    while (body < std::ssize(instructions) &&
           instructions[body].text == "s_nop 0")
      ++body;
    ASSERT_LT(body, std::ssize(instructions));
    EXPECT_EQ(instructions[body].text, "s_endpgm");
    EXPECT_EQ(instructions[body].address, entry + preloadEntryBytes);
  }
}

// Preload on the device.

static constexpr const char *preloadName = "codegen_preload";
static constexpr int32_t preloadGrid[3] = {3, 2, 2};
static constexpr int32_t preloadGroups =
    preloadGrid[0] * preloadGrid[1] * preloadGrid[2];
static constexpr int32_t preloadScalars = 13;
// Workgroup ids x, y, z followed by the scalar arguments.
static constexpr int32_t recordDwords = 3 + preloadScalars;
static constexpr int32_t maxPreloadRequest = 100;
// The pointer and 12 scalars fill the 14 dwords next to the kernarg pointer.
static constexpr int32_t maxPreloadWindow = 14;

namespace {

enum class ArgumentUse {
  /// Copy with loadArgument, then release the preloaded register.
  copy,
  /// Take the preloaded register and use it directly.
  take,
  /// Release the preloaded register, then load the value with loadArgument.
  releaseThenLoad,
};

struct PreloadCase {
  CodeObjectVersion version;
  int32_t request;
  ArgumentUse use;
  /// Remove the preload directives, as old firmware ignores them, so the
  /// compat prologue loads the window.
  bool stripPreloadDirectives;
};

} // namespace

static int32_t scalarValue(int32_t index) {
  return 37 * index * index - 11 * index - 0x6000;
}

static KernelSignature preloadSignature() {
  return pointerAndScalars(preloadScalars);
}

// Returns SGPRs holding the count dwords at offset, owned by keep, in the way
// use prescribes.
static RegisterRef obtainArgument(Kernel &kernel, ArgumentUse use,
                                  int32_t offset, int32_t count,
                                  std::vector<RegisterRange> &keep) {
  if (use == ArgumentUse::take) {
    if (std::optional<RegisterRange> taken =
            kernel.takePreloadedArgument(offset)) {
      keep.push_back(std::move(*taken));
      return keep.back().ref();
    }
  }
  keep.push_back(kernel.builder().allocateSgpr(count, count));
  RegisterRef dst = keep.back().ref();
  if (use == ArgumentUse::releaseThenLoad)
    kernel.releasePreloadedArgument(offset);
  kernel.loadArgument(dst, offset);
  if (use == ArgumentUse::copy)
    kernel.releasePreloadedArgument(offset);
  return dst;
}

// Returns the byte offset of this workgroup's record in a VGPR, computed from
// the workgroup ids wherever the hardware placed them.
static RegisterRange computePreloadRecordOffset(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange linear = builder.allocateSgpr(1);
  // linear = x + gridX * (y + gridY * z).
  builder.create<rocisa::SMulI32>(
      linear.container(), kernel.workgroupIdZ().container(), preloadGrid[1]);
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  kernel.workgroupIdY().container());
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  preloadGrid[0]);
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  kernel.workgroupIdX().container());
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  4 * recordDwords);
  RegisterRange offset = builder.allocateVgpr(1);
  builder.create<rocisa::VMovB32>(offset.container(), linear.container());
  return offset;
}

// Stores the workgroup ids and the scalar arguments of every workgroup from
// lane 0, through the output pointer argument.
static void buildPreloadKernel(Kernel &kernel, ArgumentUse use) {
  IRBuilder &builder = kernel.builder();
  std::vector<RegisterRange> keep;
  keep.reserve(1 + preloadScalars);
  RegisterRef out = obtainArgument(kernel, use, 0, 2, keep);
  std::vector<RegisterRef> words = {kernel.workgroupIdX().ref(),
                                    kernel.workgroupIdY().ref(),
                                    kernel.workgroupIdZ().ref()};
  for (int32_t i = 0; i < preloadScalars; ++i)
    words.push_back(obtainArgument(kernel, use, 8 + 4 * i, 1, keep));
  builder.createWaitCnt({.scalarMemory = 0});
  RegisterRange base = computePreloadRecordOffset(kernel);
  // The kernel ends right after the stores, so shrinking exec is safe.
  builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(), 1);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange data = builder.allocateVgpr(1);
  for (int32_t i = 0; i < recordDwords; ++i) {
    builder.create<rocisa::VAddU32>(address.container(), 4 * i,
                                    base.container());
    builder.create<rocisa::VMovB32>(data.container(), words[i].container());
    builder.create<rocisa::GlobalStoreB32>(address.container(),
                                           data.container(), out.container());
  }
  builder.createEndProgram();
}

static std::string stripPreloadDirectives(const std::string &assembly) {
  std::istringstream stream(assembly);
  std::string stripped;
  for (std::string current; std::getline(stream, current);) {
    if (current.find("kernarg_preload_length") != std::string::npos ||
        current.find("kernarg_preload_offset") != std::string::npos)
      continue;
    stripped += current + "\n";
  }
  return stripped;
}

static void runPreloadKernel(const std::vector<uint8_t> &hsaco,
                             std::vector<int32_t> &records) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, preloadName, module, function);
  DevicePtr deviceOut;
  // The sentinel shows any record the kernel failed to write.
  uploadBytes(nullptr, records.size() * sizeof(int32_t), deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  void *out = deviceOut.get();
  std::vector<int32_t> scalars;
  for (int32_t i = 0; i < preloadScalars; ++i)
    scalars.push_back(scalarValue(i));
  std::vector<void *> args = {&out};
  for (int32_t &scalar : scalars)
    args.push_back(&scalar);
  launchWithParams(function,
                   dim3(preloadGrid[0], preloadGrid[1], preloadGrid[2]), 64,
                   args.data());
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceOut, records);
}

static void checkPreloadRecords(const std::vector<int32_t> &records) {
  for (int32_t z = 0; z < preloadGrid[2]; ++z)
    for (int32_t y = 0; y < preloadGrid[1]; ++y)
      for (int32_t x = 0; x < preloadGrid[0]; ++x) {
        int32_t linear = x + preloadGrid[0] * (y + preloadGrid[1] * z);
        for (int32_t i = 0; i < recordDwords; ++i) {
          int32_t expected = i < 3 ? (i == 0   ? x
                                      : i == 1 ? y
                                               : z)
                                   : scalarValue(i - 3);
          ASSERT_EQ(records[linear * recordDwords + i], expected)
              << "group (" << x << ", " << y << ", " << z << ") word " << i;
        }
      }
}

static std::string
preloadCaseName(const testing::TestParamInfo<PreloadCase> &info) {
  const PreloadCase &c = info.param;
  const char *uses[] = {"Copy", "Take", "ReleaseThenLoad"};
  return "V" + std::to_string(static_cast<int32_t>(c.version)) + "Request" +
         std::to_string(c.request) + uses[static_cast<int32_t>(c.use)] +
         (c.stripPreloadDirectives ? "Stripped" : "");
}

static std::vector<PreloadCase> preloadCases() {
  std::vector<PreloadCase> cases;
  for (CodeObjectVersion version :
       {CodeObjectVersion::v4, CodeObjectVersion::v5, CodeObjectVersion::v6})
    for (ArgumentUse use :
         {ArgumentUse::copy, ArgumentUse::take, ArgumentUse::releaseThenLoad}) {
      for (int32_t request : {0, 3, maxPreloadRequest})
        cases.push_back({version, request, use, false});
      cases.push_back({version, maxPreloadRequest, use, true});
    }
  return cases;
}

namespace {

class KernelPreloadLaunch : public testing::TestWithParam<PreloadCase> {};

} // namespace

TEST_P(KernelPreloadLaunch, ArgumentsAndWorkgroupIdsOnDevice) {
  const PreloadCase &c = GetParam();
  Target target = Target::get("gfx950");
  KernelOptions options = preloadOptions(c.request, c.version, true, true);
  Kernel kernel(target, preloadName, 64, preloadSignature(), options);
  int32_t window = c.request == 0 ? 0 : c.request == 3 ? 3 : maxPreloadWindow;
  ASSERT_EQ(kernel.preloadDwords(), window);
  EXPECT_EQ(kernel.workgroupIdX().start(), 2 + window);
  buildPreloadKernel(kernel, c.use);
  std::string assembly = kernel.toAssembly();
  if (c.stripPreloadDirectives)
    assembly = stripPreloadDirectives(assembly);
  SCOPED_TRACE("assembly:\n" + assembly);

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> records(preloadGroups * recordDwords, -1);
  runPreloadKernel(hsaco, records);
  if (testing::Test::HasFatalFailure())
    return;
  checkPreloadRecords(records);
}

INSTANTIATE_TEST_SUITE_P(Codegen, KernelPreloadLaunch,
                         testing::ValuesIn(preloadCases()), preloadCaseName);
