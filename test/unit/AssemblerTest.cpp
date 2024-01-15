

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Assemblers/Assembler.hpp>

using namespace rocRoller;

static constexpr auto simple_assembly{
    R"(
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack+"
.set .amdgcn.next_free_vgpr, 0 // Needed if we ever place 2 kernels in one file.
.set .amdgcn.next_free_sgpr, 0
.text
.globl hello_world
.p2align 8
.type hello_world,@function
hello_world:


// Allocated : 2 SGPRs (Float): s0, s1


// Allocated : 2 SGPRs (Float): s2, s3
s_load_dwordx2 s[2:3], s[0:1], 0 // Load pointer

// Allocated : 1 SGPR (Float): s4
s_load_dword s4, s[0:1], 8 // Load value
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0) //


// Allocated : 2 VGPRs (Float): v0, v1

v_mov_b32 v0, s2 // Move pointer
v_mov_b32 v1, s3 // Move pointer

// Allocated : 1 VGPR (Float): v2

v_mov_b32 v2, s4 // Move value
flat_store_dword v[0:1], v2 // Store value
s_endpgm // End of hello_world
.Lhello_world_end:

.size hello_world, .Lhello_world_end-hello_world
.rodata
.p2align 6
.amdhsa_kernel hello_world
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: hello_world
    .symbol: hello_world.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 5
    .vgpr_count: 3
    .max_flat_workgroup_size: 256
    .args:
      - .size: 8
        .offset: 0
        .address_space: global
        .actual_access: write_only
        .value_kind: global_buffer
      - .size: 4
        .offset: 8
        .value_kind: by_value
...
.end_amdgpu_metadata
)"};

TEST(AssemblerTest, Basic)
{
    auto myAssembler = rocRoller::Assembler::Get();

    std::vector<char> kernelObject = myAssembler->assembleMachineCode(
        simple_assembly, rocRoller::GPUArchitectureTarget("gfx90a:xnack+"));

    EXPECT_NE(kernelObject.size(), 0);
}

TEST(AssemblerTest, BadTarget)
{
    auto myAssembler = rocRoller::Assembler::Get();

    EXPECT_THROW(myAssembler->assembleMachineCode(
                     simple_assembly, rocRoller::GPUArchitectureTarget("gfx91a:xnack+")),
                 std::runtime_error);
}
