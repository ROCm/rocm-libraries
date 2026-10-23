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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

static bool hasLine(const std::string &text, const std::string &line) {
  return text.find("\n" + line + "\n") != std::string::npos;
}

namespace {

class KernelTest : public testing::Test {
protected:
  Target target = Target::get("gfx950");
};

} // namespace

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

static void addReferenceGemmExplicitArguments(KernelSignature &signature) {
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

TEST(KernelLaunch, EmptyKernelAssemblesLoadsAndLaunches) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_empty_kernel", 256);
  kernel.signature().addPointer("out");
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

constexpr int roundTripGroupSize = 256;
constexpr int roundTripGroups = 4;
constexpr int roundTripElements = roundTripGroupSize * roundTripGroups;
constexpr const char *roundTripName = "codegen_agpr_round_trip";

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
  kernel.signature().addPointer("in");
  kernel.signature().addPointer("out");
  RegisterRange pointers = loadRoundTripPointers(kernel);
  RegisterRange globalId = computeGlobalId(kernel);
  emitAgprRoundTrip(kernel, pointers, globalId);
  kernel.builder().createEndProgram();
}

// Uploads in, launches the kernel and downloads the result into out.
static void runRoundTrip(const std::vector<uint8_t> &hsaco, const char *name,
                         int groups, int groupSize,
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
  Kernel kernel(target, roundTripName, roundTripGroupSize);
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
  for (int i = 0; i < roundTripElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(roundTripElements, -1);
  runRoundTrip(hsaco, roundTripName, roundTripGroups, roundTripGroupSize, in,
               out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int i = 0; i < roundTripElements; ++i)
    ASSERT_EQ(out[i], in[i]) << "element " << i;
}

constexpr const char *idsName = "codegen_ids_hidden";
constexpr int idsGrid[3] = {3, 4, 5};
constexpr int idsBlock[3] = {8, 2, 2};
constexpr int idsGroups = idsGrid[0] * idsGrid[1] * idsGrid[2];
constexpr int idsBlockSize = idsBlock[0] * idsBlock[1] * idsBlock[2];
// Workgroup ids x, y, z, group sizes x, y, z and block counts x, y, z.
constexpr int idsDwords = 9;

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
  for (int i = 0; i < 3; ++i)
    builder.create<rocisa::SMovB32>(values.sub(i), ids[i]->container());
  for (int i = 0; i < 6; ++i)
    kernel.loadArgument(values.ref(3 + i), 12 + 4 * i);
  builder.createWaitCnt({.scalarMemory = 0});
  // The runtime writes only the low 2 bytes of a group size (LLVM's width), so
  // the upper half of a 4-byte slot is undefined.
  for (int i = 3; i < 6; ++i)
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
  kernel.enableWorkgroupIds(true, true);
  addIdsSignature(kernel.signature());
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
  for (int i = 0; i < idsDwords; ++i) {
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
  Kernel kernel(target, idsName, idsBlockSize);
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

  for (int z = 0; z < idsGrid[2]; ++z)
    for (int y = 0; y < idsGrid[1]; ++y)
      for (int x = 0; x < idsGrid[0]; ++x) {
        int linear = x + idsGrid[0] * (y + idsGrid[1] * z);
        const int32_t expected[idsDwords] = {
            x,           y,          z,          idsBlock[0], idsBlock[1],
            idsBlock[2], idsGrid[0], idsGrid[1], idsGrid[2]};
        for (int i = 0; i < idsDwords; ++i)
          ASSERT_EQ(records[linear * idsDwords + i], expected[i])
              << "group (" << x << ", " << y << ", " << z << ") word " << i;
      }
}

constexpr const char *ldsName = "codegen_lds_round_trip";
constexpr int ldsGroupSize = 256;
constexpr int ldsGroups = 4;
constexpr int ldsElements = ldsGroupSize * ldsGroups;

// Returns the LDS byte address of lane (tid + delta) % ldsGroupSize.
static RegisterRange computeLdsAddress(Kernel &kernel, int delta) {
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
  kernel.signature().addPointer("in");
  kernel.signature().addPointer("out");
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
  Kernel kernel(target, ldsName, ldsGroupSize);
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
  for (int i = 0; i < ldsElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(ldsElements, -1);
  runRoundTrip(hsaco, ldsName, ldsGroups, ldsGroupSize, in, out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int g = 0; g < ldsGroups; ++g)
    for (int t = 0; t < ldsGroupSize; ++t)
      ASSERT_EQ(out[g * ldsGroupSize + t],
                in[g * ldsGroupSize + (t + 1) % ldsGroupSize])
          << "group " << g << " lane " << t;
}
