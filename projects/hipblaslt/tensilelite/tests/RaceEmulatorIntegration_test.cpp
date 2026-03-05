// Minimal integration test: verify that RaceConditionException thrown inside
// the race-emulator shared library is correctly caught by a consumer compiled
// with a potentially different compiler/C++ standard.

#include "race-emulator/Emulator.h"
#include "race-emulator/EmulatorException.h"
#include <gtest/gtest.h>
#include <string>

// Kernel metadata shared by both test kernels. Describes a single kernel "foo"
// with one 8-byte global_buffer argument, targeting gfx942 with wave size 64.
static const std::string kKernelMetadata = R"ASM(
  .amdhsa_kernel foo
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_sgpr 10
    .amdhsa_next_free_vgpr 10
    .amdhsa_accum_offset 10
  .end_amdhsa_kernel
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .name:           foo
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
...
)ASM";

// A minimal kernel with a VGPR race: global_load writes to v1, then ds_write
// reads v1 without a s_waitcnt vmcnt(0) in between.
static const std::string kRaceyAsm = std::string(R"ASM(
foo:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  ; Missing s_waitcnt vmcnt(0) -- this is the race.
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
)ASM") + kKernelMetadata;

// A race-free version of the same kernel (with the wait).
static const std::string kCleanAsm = std::string(R"ASM(
foo:
  s_load_dwordx2 s[0:1], s[0:1], 0x0
  v_lshlrev_b32_e32 v0, 2, v0
  s_waitcnt lgkmcnt(0)
  global_load_dword v1, v0, s[0:1]
  s_waitcnt vmcnt(0)
  ds_write_b32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
)ASM") + kKernelMetadata;

// Helper to set up an emulator with dummy kernargs and race checking enabled.
static raceemulator::Emulator makeEmulator(const std::string &asmText) {
  auto emulator = raceemulator::Emulator::createGfx942(asmText);

  // Provide a dummy kernarg (8 bytes). The pointer value doesn't matter
  // since completeEmulation is off and global loads won't actually execute.
  static std::vector<int> data(4, 0);
  int *ptr = data.data();
  emulator.addKernarg(0, &ptr);
  emulator.enableRaceChecks(true);
  emulator.enableCompleteEmulation(false);
  return emulator;
}

TEST(RaceEmulatorIntegration, DetectsVgprRaceAcrossLibraryBoundary) {
  auto emulator = makeEmulator(kRaceyAsm);
  EXPECT_THROW(emulator.run({0, 0, 0}, {64, 1, 1}),
               raceemulator::RaceConditionException);
}

TEST(RaceEmulatorIntegration, CleanAsmPassesWithoutException) {
  auto emulator = makeEmulator(kCleanAsm);
  EXPECT_NO_THROW(emulator.run({0, 0, 0}, {64, 1, 1}));
}
