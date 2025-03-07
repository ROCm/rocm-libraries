/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Assemblers/Assembler.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "GenericContextFixture.hpp"
#include "SimpleFixture.hpp"

using namespace rocRoller;

class AssemblerTest : public SimpleFixture
{
};

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
  - 2
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

auto assemble()
{
    auto myAssembler = rocRoller::Assembler::Get();

    std::vector<char> kernelObject = myAssembler->assembleMachineCode(
        simple_assembly, {GPUArchitectureGFX::GFX90A, {.xnack = true}});

    return kernelObject;
}

TEST_F(AssemblerTest, Basic)
{
    auto kernelObject = assemble();
    EXPECT_NE(kernelObject.size(), 0);
}

TEST_F(AssemblerTest, BadTarget)
{
    auto myAssembler = rocRoller::Assembler::Get();

    EXPECT_THROW(myAssembler->assembleMachineCode(simple_assembly,
                                                  {GPUArchitectureGFX::UNKNOWN, {.xnack = true}}),
                 std::runtime_error);
}

namespace rocRollerTest
{
    class SubassemblerPathTest : public GenericContextFixture
    {
    };

    TEST_F(SubassemblerPathTest, VersionedROCM)
    {
        auto path = static_cast<std::string>(
            std::filesystem::canonical(Settings::getDefault(Settings::SubprocessAssemblerPath)));
        // Ensure symbolic link expanded to include version number
        EXPECT_TRUE(path.find("rocm-") != std::string::npos);
        Settings::getInstance()->set(Settings::SubprocessAssemblerPath, path);
        EXPECT_NE(assemble().size(), 0);
    }

    TEST_F(SubassemblerPathTest, MissingPath)
    {
        Settings::getInstance()->set(
            Settings::SubprocessAssemblerPath,
            static_cast<std::string>(std::filesystem::temp_directory_path() / "foo"));
        EXPECT_THROW(assemble(), FatalError);
    }
}
