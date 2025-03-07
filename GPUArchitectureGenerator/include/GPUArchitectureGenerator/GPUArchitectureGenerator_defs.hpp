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

#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "rocRoller/GPUArchitecture/GPUArchitecture.hpp"
#include "rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp"
#include "rocRoller/GPUArchitecture/GPUCapability.hpp"
#include "rocRoller/GPUArchitecture/GPUInstructionInfo.hpp"

/**
  *
  * If you make any changes to this file, you will likely need to
  * regeneratee the yaml files in the pregenerated folder.
  *
  */

namespace GPUArchitectureGenerator
{
    const std::string DEFAULT_ASSEMBLER = "/opt/rocm/bin/amdclang";

    // GPUCapability -> {Assembler Query, Assembler Options}
    const std::unordered_map<rocRoller::GPUCapability,
                             std::tuple<std::string, std::string>,
                             rocRoller::GPUCapability::Hash>
        AssemblerQueries = {
            {rocRoller::GPUCapability::SupportedISA, {"", ""}},
            {rocRoller::GPUCapability::HasExplicitCO, {"v_add_co_u32 v0,vcc,v0,1", ""}},
            {rocRoller::GPUCapability::HasExplicitNC, {"v_add_nc_u32 v0,v0,1", ""}},
            {rocRoller::GPUCapability::HasDirectToLds,
             {"buffer_load_dword v0, s[0:3], 0 offen offset:0 lds", ""}},
            {rocRoller::GPUCapability::HasWiderDirectToLds,
             {"buffer_load_dwordx4 v0, s[8:11], 0 offen offset: 0 lds", ""}},
            {rocRoller::GPUCapability::HasAddLshl, {"v_add_lshl_u32 v47, v36, v34, 0x2", ""}},
            {rocRoller::GPUCapability::HasLshlOr, {"v_lshl_or_b32 v47, v36, 0x2, v34", ""}},
            {rocRoller::GPUCapability::HasSMulHi, {"s_mul_hi_u32 s47, s36, s34", ""}},
            {rocRoller::GPUCapability::HasCodeObjectV3, {"", "-mcode-object-version=3"}},
            {rocRoller::GPUCapability::HasCodeObjectV4, {"", "-mcode-object-version=4"}},
            {rocRoller::GPUCapability::HasCodeObjectV5, {"", "-mcode-object-version=5"}},

            {rocRoller::GPUCapability::HasMFMA,
             {"v_mfma_f32_32x32x1f32 a[0:31], v32, v33, a[0:31]", ""}},
            {rocRoller::GPUCapability::HasMFMA_fp8,
             {"v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[32:33], v[36:37], v[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_f8f6f4,
             {"v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_f64,
             {"v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]", ""}},
            {rocRoller::GPUCapability::HasMFMA_bf16_32x32x4_1k,
             {"v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]", ""}},
            {rocRoller::GPUCapability::HasMFMA_bf16_32x32x4,
             {"v_mfma_f32_32x32x4bf16 a[0:15], v11, v10, a[0:15]", ""}},
            {rocRoller::GPUCapability::HasMFMA_bf16_32x32x8_1k,
             {"v_mfma_f32_32x32x8bf16_1k a[0:3], v[32:33], v[34:35], a[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_bf16_16x16x8,
             {"v_mfma_f32_16x16x8bf6 a[0:3], v[32], v[33], a[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_bf16_16x16x16_1k,
             {"v_mfma_f32_16x16x16bf16_1k a[0:3], v[32:33], v[36:37], a[0:3]", ""}},

            {rocRoller::GPUCapability::HasMFMA_16x16x32_f16,
             {"v_mfma_f32_16x16x32_f16 a[0:3], v[32:35], v[36:39], a[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_32x32x16_f16,
             {"v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[36:39], a[0:15]", ""}},
            {rocRoller::GPUCapability::HasMFMA_16x16x32_bf16,
             {"v_mfma_f32_16x16x32_bf16 a[0:3], v[32:35], v[36:39], a[0:3]", ""}},
            {rocRoller::GPUCapability::HasMFMA_32x32x16_bf16,
             {"v_mfma_f32_32x32x16_bf16 a[0:15], v[32:35], v[36:39], a[0:15]", ""}},

            {rocRoller::GPUCapability::HasWMMA,
             {"v_wmma_f16_16x16x16_f16 v[0:3], v[32:35], v[36:39], v[0:3]", ""}},

            {rocRoller::GPUCapability::HasAccumOffset,
             {".amdhsa_kernel hello_world\n  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr\n  "
              ".amdhsa_next_free_sgpr .amdgcn.next_free_sgpr\n  .amdhsa_accum_offset "
              "4\n.end_amdhsa_kernel",
              ""}},

            {rocRoller::GPUCapability::HasGlobalOffset,
             {"global_store_dword v[8:9], v5 off offset:-16", ""}},

            {rocRoller::GPUCapability::v_mac_f16, {"v_mac_f16 v47, v36, v34", ""}},

            {rocRoller::GPUCapability::v_fma_f16,
             {"v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]", ""}},
            {rocRoller::GPUCapability::v_fmac_f16, {"v_fma_f16 v47, v36, v34", ""}},

            {rocRoller::GPUCapability::v_pk_fma_f16,
             {"v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]", ""}},
            {rocRoller::GPUCapability::v_pk_fmac_f16, {"v_pk_fma_f16 v47, v36, v34", ""}},

            {rocRoller::GPUCapability::v_mad_mix_f32,
             {"v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", ""}},
            {rocRoller::GPUCapability::v_fma_mix_f32,
             {"v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]", ""}},

            {rocRoller::GPUCapability::v_dot2_f32_f16, {"v_dot2_f32_f16 v20, v36, v34, v20", ""}},
            {rocRoller::GPUCapability::v_dot2c_f32_f16, {"v_dot2c_f32_f16 v47, v36, v34", ""}},

            {rocRoller::GPUCapability::v_dot4c_i32_i8, {"v_dot4c_i32_i8 v47, v36, v34", ""}},
            {rocRoller::GPUCapability::v_dot4_i32_i8, {"v_dot4_i32_i8 v47, v36, v34", ""}},

            {rocRoller::GPUCapability::v_mac_f32, {"v_mac_f32 v20, v21, v22", ""}},
            {rocRoller::GPUCapability::v_fma_f32, {"v_fma_f32 v20, v21, v22, v23", ""}},
            {rocRoller::GPUCapability::v_fmac_f32, {"v_fmac_f32 v20, v21, v22", ""}},

            {rocRoller::GPUCapability::v_mov_b64, {"v_mov_b64 v[2:3], v[0:1]", ""}},

            {rocRoller::GPUCapability::HasAtomicAdd,
             {"buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0", ""}},

            {rocRoller::GPUCapability::UnalignedVGPRs, {"v_add_f64 v[0:1], v[0:1], v[3:4]", ""}},

            {rocRoller::GPUCapability::HasDSReadTransposeB16,
             {"ds_read_b64_tr_b16 v[4:5], v1, offset:0", ""}},
            {rocRoller::GPUCapability::HasDSReadTransposeB8,
             {"ds_read_b64_tr_b8  v[4:5], v1, offset:0", ""}},
            {rocRoller::GPUCapability::HasDSReadTransposeB6,
             {"ds_read_b96_tr_b6  v[4:6], v1, offset:0", ""}},
            {rocRoller::GPUCapability::HasDSReadTransposeB4,
             {"ds_read_b64_tr_b4  v[4:5], v1, offset:0", ""}},

            {rocRoller::GPUCapability::HasPermLanes16, {"v_permlane16_swap_b32 v4, v5", ""}},
            {rocRoller::GPUCapability::HasPermLanes32, {"v_permlane32_swap_b32 v4, v5", ""}},
            {rocRoller::GPUCapability::UnalignedSGPRs, {"s_cmp_eq_u64 s[0:1], s[0:1], s[3:4]", ""}},
            {rocRoller::GPUCapability::SeparateVscnt, {"s_waitcnt_vscnt 0", ""}},

    };

    // GPUCapability -> <Vector of ISAs That Support It>
    const std::unordered_map<rocRoller::GPUCapability,
                             std::vector<rocRoller::GPUArchitectureTarget>,
                             rocRoller::GPUCapability::Hash>
        ArchSpecificCaps
        = {{rocRoller::GPUCapability::Waitcnt0Disabled,
            {
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX908},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX90A},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX940},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.xnack = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.sramecc = true}},
            }},
           {rocRoller::GPUCapability::HasAccCD,
            {
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX90A},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX940},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.xnack = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.sramecc = true}},
            }},
           {rocRoller::GPUCapability::ArchAccUnifiedRegs,
            {
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX90A},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX90A,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX940},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX941,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942,
                                                 {.sramecc = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.xnack = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.sramecc = true}},
            }},
           {rocRoller::GPUCapability::HasNaNoo,
            {
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX942,
                                                 {.sramecc = true}},
            }},
           {rocRoller::GPUCapability::HasPRNG,
            {
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.xnack = true}},
                rocRoller::GPUArchitectureTarget{rocRoller::GPUArchitectureGFX::GFX950,
                                                 {.sramecc = true}},
            }},
           {rocRoller::GPUCapability::HasWave64,
            std::vector<rocRoller::GPUArchitectureTarget>(
                rocRoller::SupportedArchitectures.begin(),
                rocRoller::SupportedArchitectures.end())}};

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx908ISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isCDNA1GPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx90aISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isCDNA2GPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx94XISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isCDNA3GPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx9ISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isCDNAGPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx120XISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isRDNA4GPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx12ISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isRDNA4GPU(); });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> cdnaISAs_1_2()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(rocRoller::SupportedArchitectures.begin(),
                     rocRoller::SupportedArchitectures.end(),
                     std::back_inserter(retval),
                     [](rocRoller::GPUArchitectureTarget const& x) -> bool {
                         return x.isCDNA1GPU() || x.isCDNA2GPU();
                     });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> cdnaISAs_2_3()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(rocRoller::SupportedArchitectures.begin(),
                     rocRoller::SupportedArchitectures.end(),
                     std::back_inserter(retval),
                     [](rocRoller::GPUArchitectureTarget const& x) -> bool {
                         return x.isCDNA2GPU() || x.isCDNA3GPU();
                     });
        return retval;
    }

    inline std::vector<rocRoller::GPUArchitectureTarget> gfx95XISAs()
    {
        std::vector<rocRoller::GPUArchitectureTarget> retval;
        std::copy_if(
            rocRoller::SupportedArchitectures.begin(),
            rocRoller::SupportedArchitectures.end(),
            std::back_inserter(retval),
            [](rocRoller::GPUArchitectureTarget const& x) -> bool { return x.isCDNA35GPU(); });
        return retval;
    }

    // GPUCapability -> <Predicate that returns true given an isa that supports it.>
    const std::unordered_map<rocRoller::GPUCapability,
                             std::function<bool(const rocRoller::GPUArchitectureTarget&)>,
                             rocRoller::GPUCapability::Hash>
        PredicateCaps = {
            {rocRoller::GPUCapability::CMPXWritesSGPR,
             [](rocRoller::GPUArchitectureTarget x) -> bool { return !x.isRDNAGPU(); }},

            {rocRoller::GPUCapability::HasWave32,
             [](rocRoller::GPUArchitectureTarget x) -> bool { return x.isRDNAGPU(); }},

            {rocRoller::GPUCapability::HasXnack,
             [](rocRoller::GPUArchitectureTarget x) -> bool { return x.features.xnack; }},

            {rocRoller::GPUCapability::PackedWorkitemIDs,
             [](rocRoller::GPUArchitectureTarget x) -> bool {
                 return x.isCDNA2GPU() || x.isCDNA3GPU() || x.isCDNA35GPU();
             }},

    };
    // This is the way to add a set of instructions that have the same wait value and wait queues.
    const std::vector<std::tuple<std::vector<rocRoller::GPUArchitectureTarget>,
                                 std::tuple<std::vector<std::string>,
                                            int,
                                            std::vector<rocRoller::GPUWaitQueueType>,
                                            unsigned int>>>
        GroupedInstructionInfos = {
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "buffer_load_dword",
                 "buffer_load_dwordx2",
                 "buffer_load_dwordx3",
                 "buffer_load_dwordx4",
                 "buffer_load_sbyte",
                 "buffer_load_sbyte_d16",
                 "buffer_load_sbyte_d16_hi",
                 "buffer_load_short_d16",
                 "buffer_load_short_d16_hi",
                 "buffer_load_sshort",
                 "buffer_load_ubyte",
                 "buffer_load_ubyte_d16",
                 "buffer_load_ubyte_d16_hi",
                 "buffer_load_ushort",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::LoadQueue},
              (1 << 12) - 1}},
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "buffer_store_byte",
                 "buffer_store_byte_d16_hi",
                 "buffer_store_dword",
                 "buffer_store_dwordx2",
                 "buffer_store_dwordx3",
                 "buffer_store_dwordx4",
                 "buffer_store_short",
                 "buffer_store_short_d16_hi",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::StoreQueue},
              (1 << 12) - 1}},
            // single-address LDS instructions
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "ds_read_addtid_b32",
                 "ds_read_b128",
                 "ds_read_b32",
                 "ds_read_b64",
                 "ds_read_b96",
                 "ds_read_i16",
                 "ds_read_i8",
                 "ds_read_i8_d16",
                 "ds_read_i8_d16_hi",
                 "ds_read_u16",
                 "ds_read_u8",
                 "ds_read_u16_d16",
                 "ds_read_u16_d16_hi",
                 "ds_read_u8_d16",
                 "ds_read_u8_d16_hi",
                 "ds_write_addtid_b32",
                 "ds_write_b128",
                 "ds_write_b16",
                 "ds_write_b16_d16_hi",
                 "ds_write_b32",
                 "ds_write_b64",
                 "ds_write_b8",
                 "ds_write_b8_d16_hi",
                 "ds_write_b96",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::DSQueue},
              (1 << 16) - 1}},
            // two-address LDS instructions
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "ds_read2_b32",
                 "ds_read2_b64",
                 "ds_read2st64_b32",
                 "ds_read2st64_b64",
                 "ds_write2_b32",
                 "ds_write2_b64",
                 "ds_write2st64_b32",
                 "ds_write2st64_b64",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::DSQueue},
              (1 << 8) - 1}},
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "global_load_dword",
                 "global_load_dwordx2",
                 "global_load_dwordx3",
                 "global_load_dwordx4",
                 "global_load_sbyte",
                 "global_load_sbyte_d16",
                 "global_load_sbyte_d16_hi",
                 "global_load_short_d16",
                 "global_load_short_d16_hi",
                 "global_load_sshort",
                 "global_load_ubyte",
                 "global_load_ubyte_d16",
                 "global_load_ubyte_d16_hi",
                 "global_load_ushort",
                 "scratch_load_dword",
                 "scratch_load_dwordx2",
                 "scratch_load_dwordx3",
                 "scratch_load_dwordx4",
                 "scratch_load_sbyte",
                 "scratch_load_sbyte_d16",
                 "scratch_load_sbyte_d16_hi",
                 "scratch_load_short_d16",
                 "scratch_load_short_d16_hi",
                 "scratch_load_sshort",
                 "scratch_load_ubyte",
                 "scratch_load_ubyte_d16",
                 "scratch_load_ubyte_d16_hi",
                 "scratch_load_ushort",
                  // clang-format on
              },
              0,
              {rocRoller::GPUWaitQueueType::LoadQueue},
              (1 << 12) - 1}}, // the offset is a `signed` 13-bit  (i.e., -4096..4095)
            {gfx9ISAs(),
             {{
                  // clang-format off
                 "global_store_byte",
                 "global_store_byte_d16_hi",
                 "global_store_dword",
                 "global_store_dwordx2",
                 "global_store_dwordx3",
                 "global_store_dwordx4",
                 "global_store_short",
                 "global_store_short_d16_hi",
                 "scratch_store_byte",
                 "scratch_store_byte_d16_hi",
                 "scratch_store_dword",
                 "scratch_store_dwordx2",
                 "scratch_store_dwordx3",
                 "scratch_store_dwordx4",
                 "scratch_store_short",
                 "scratch_store_short_d16_hi",
                  // clang-format on
              },
              0,
              {rocRoller::GPUWaitQueueType::StoreQueue},
              (1 << 12) - 1}}, // the offset is a `signed` 13-bit  (i.e., -4096..4095)
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "buffer_load_dword",
                 "buffer_load_dwordx2",
                 "buffer_load_dwordx3",
                 "buffer_load_dwordx4",
                 "buffer_load_sbyte",
                 "buffer_load_sbyte_d16",
                 "buffer_load_sbyte_d16_hi",
                 "buffer_load_short_d16",
                 "buffer_load_short_d16_hi",
                 "buffer_load_sshort",
                 "buffer_load_ubyte",
                 "buffer_load_ubyte_d16",
                 "buffer_load_ubyte_d16_hi",
                 "buffer_load_ushort",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::LoadQueue},
              (1 << 12) - 1}},
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "buffer_store_byte",
                 "buffer_store_byte_d16_hi",
                 "buffer_store_dword",
                 "buffer_store_dwordx2",
                 "buffer_store_dwordx3",
                 "buffer_store_dwordx4",
                 "buffer_store_short",
                 "buffer_store_short_d16_hi",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::StoreQueue},
              (1 << 12) - 1}},
            // single-address LDS instructions
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "ds_read_addtid_b32",
                 "ds_read_b128",
                 "ds_read_b32",
                 "ds_read_b64",
                 "ds_read_b96",
                 "ds_read_i16",
                 "ds_read_i8",
                 "ds_read_i8_d16",
                 "ds_read_i8_d16_hi",
                 "ds_read_u16",
                 "ds_read_u8",
                 "ds_read_u16_d16",
                 "ds_read_u16_d16_hi",
                 "ds_read_u8_d16",
                 "ds_read_u8_d16_hi",
                 "ds_write_addtid_b32",
                 "ds_write_b128",
                 "ds_write_b16",
                 "ds_write_b16_d16_hi",
                 "ds_write_b32",
                 "ds_write_b64",
                 "ds_write_b8",
                 "ds_write_b8_d16_hi",
                 "ds_write_b96",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::DSQueue},
              (1 << 16) - 1}},
            // two-address LDS instructions
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "ds_read2_b32",
                 "ds_read2_b64",
                 "ds_read2st64_b32",
                 "ds_read2st64_b64",
                 "ds_write2_b32",
                 "ds_write2_b64",
                 "ds_write2st64_b32",
                 "ds_write2st64_b64",
                  // clang-format on
              },
              1,
              {rocRoller::GPUWaitQueueType::DSQueue},
              (1 << 8) - 1}},
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "global_load_dword",
                 "global_load_dwordx2",
                 "global_load_dwordx3",
                 "global_load_dwordx4",
                 "global_load_sbyte",
                 "global_load_sbyte_d16",
                 "global_load_sbyte_d16_hi",
                 "global_load_short_d16",
                 "global_load_short_d16_hi",
                 "global_load_sshort",
                 "global_load_ubyte",
                 "global_load_ubyte_d16",
                 "global_load_ubyte_d16_hi",
                 "global_load_ushort",
                 "scratch_load_dword",
                 "scratch_load_dwordx2",
                 "scratch_load_dwordx3",
                 "scratch_load_dwordx4",
                 "scratch_load_sbyte",
                 "scratch_load_sbyte_d16",
                 "scratch_load_sbyte_d16_hi",
                 "scratch_load_short_d16",
                 "scratch_load_short_d16_hi",
                 "scratch_load_sshort",
                 "scratch_load_ubyte",
                 "scratch_load_ubyte_d16",
                 "scratch_load_ubyte_d16_hi",
                 "scratch_load_ushort",
                  // clang-format on
              },
              0,
              {rocRoller::GPUWaitQueueType::LoadQueue},
              (1 << 12) - 1}}, // the offset is a `signed` 13-bit  (i.e., -4096..4095)
            {gfx12ISAs(),
             {{
                  // clang-format off
                 "global_store_byte",
                 "global_store_byte_d16_hi",
                 "global_store_dword",
                 "global_store_dwordx2",
                 "global_store_dwordx3",
                 "global_store_dwordx4",
                 "global_store_short",
                 "global_store_short_d16_hi",
                 "scratch_store_byte",
                 "scratch_store_byte_d16_hi",
                 "scratch_store_dword",
                 "scratch_store_dwordx2",
                 "scratch_store_dwordx3",
                 "scratch_store_dwordx4",
                 "scratch_store_short",
                 "scratch_store_short_d16_hi",
                  // clang-format on
              },
              0,
              {rocRoller::GPUWaitQueueType::StoreQueue},
              (1 << 12) - 1}}, // the offset is a `signed` 13-bit  (i.e., -4096..4095)
    };

    // Tuple mapping a <Vector of GPUInstructionInfo> to a <Vector of GPUArchitectureTarget>
    const std::vector<std::tuple<std::vector<rocRoller::GPUArchitectureTarget>,
                                 std::vector<rocRoller::GPUInstructionInfo>>>
        InstructionInfos = {
            {std::vector<rocRoller::GPUArchitectureTarget>(
                 rocRoller::SupportedArchitectures.begin(),
                 rocRoller::SupportedArchitectures.end()),
             {
                 rocRoller::GPUInstructionInfo(
                     "s_endpgm", -1, {rocRoller::GPUWaitQueueType::FinalInstruction}),
             }},
            {gfx9ISAs(),
             {
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dword", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx16", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx4", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx8", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_store_dword", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_store_dwordx2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_store_dwordx4", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_discard", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_discard_x2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_inv", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_inv_vol", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_wb", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_dcache_wb_vol", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dword", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx16", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx2", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx4", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx8", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_memrealtime", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_memtime", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_load_dword", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_load_dwordx2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_load_dwordx4", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_store_dword", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_store_dwordx2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_scratch_store_dwordx4", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_store_dword", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_store_dwordx2", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_store_dwordx4", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_sendmsg", 1, {rocRoller::GPUWaitQueueType::SendMsgQueue}),
                 rocRoller::GPUInstructionInfo("v_dot2_f32_f16", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_dot2_i32_i16", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_dot4_i32_i8", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_dot8_i32_i4", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_accvgpr_read", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_accvgpr_read_b32", 0, {}, 1),
                 rocRoller::GPUInstructionInfo("v_accvgpr_write", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_accvgpr_write_b32", 0, {}, 2),
             }},
            {gfx12ISAs(),
             {
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dword", 1, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx16", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx2", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx4", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_buffer_load_dwordx8", 2, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dword", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx16", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx2", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx4", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_load_dwordx8", 0, {rocRoller::GPUWaitQueueType::SMemQueue}),
                 rocRoller::GPUInstructionInfo(
                     "s_sendmsg", 1, {rocRoller::GPUWaitQueueType::SendMsgQueue}),
             }},
            {gfx908ISAs(),
             {
                 rocRoller::GPUInstructionInfo("exp", 1, {rocRoller::GPUWaitQueueType::EXPQueue}),
             }},
            {cdnaISAs_1_2(),
             {
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x1f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x2bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x8bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x1f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x1f32", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x2bf16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4f16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x16i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x4i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x4i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x8i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_4x4x4i8", 0, {}, 2),
             }},
            {cdnaISAs_2_3(),
             {
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4bf16_1k", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8bf16_1k", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4bf16_1k", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f64_16x16x4f64", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f64_4x4x4f64", 0, {}, 2),
             }},
            {gfx90aISAs(),
             {
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16bf16_1k", 0, {}, 8),
             }},
            {gfx94XISAs(),
             {
                 rocRoller::GPUInstructionInfo("v_mov_b64", -1, {}, 0),
                 // V_MFMA_F32_{*}_F32
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x1_2b_f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x1f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x1_4b_f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x1f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x1_16b_f32", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x1f32", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2_f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f32", 0, {}, 8),
                 // V_MFMA_F32_{*}_F16
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4_2b_f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_4b_f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4_16b_f16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4f16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8_f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16_f16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16f16", 0, {}, 4),
                 // V_MFMA_F32_{*}_BF16
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4_2b_bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_4b_bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4_16b_bf16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4bf16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8_bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16_bf16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16bf16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16bf16_1k", 0, {}, 4),
                 // V_MFMA_I32_{*}_I8
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x4_2b_i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x4i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x4_4b_i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x4i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_4x4x4_16b_i8", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_4x4x4i8", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x16_i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x16i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x32_i8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x32i8", 0, {}, 4),
                 // V_MFMA_F32_{*}_XF32
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x8_xf32", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x8xf32", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4_xf32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4xf32", 0, {}, 8),
                 // V_MFMA_F64_{*}_F64
                 rocRoller::GPUInstructionInfo("v_mfma_f64_16x16x4_f64", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f64_4x4x4_4b_f64", 0, {}, 4),
                 // V_MFMA_F32_{*}_BF8_BF8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_bf8_bf8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_bf8_bf8", 0, {}, 8),
                 // V_MFMA_F32_{*}_BF8_FP8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_bf8_fp8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_bf8_fp8", 0, {}, 8),
                 // V_MFMA_F32_{*}_FP8_BF8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_fp8_bf8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_fp8_bf8", 0, {}, 8),
                 // V_MFMA_F32_{*}_FP8_FP8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_fp8_fp8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_fp8_fp8", 0, {}, 8),
             }},
            {gfx95XISAs(),
             {
                 rocRoller::GPUInstructionInfo("v_mov_b64", -1, {}, 0),
                 // V_MFMA_F32_{*}_F32
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x1_2b_f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x1f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x1_4b_f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x1f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x1_16b_f32", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x1f32", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2_f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x2f32", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_f32", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f32", 0, {}, 8),
                 // V_MFMA_F32_{*}_F16
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4_2b_f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_4b_f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4_16b_f16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4f16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8_f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8f16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16_f16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16f16", 0, {}, 4),
                 // These two instructions are new
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_f16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_f16", 0, {}, 8),
                 // V_MFMA_F32_{*}_BF16
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4_2b_bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x4bf16_1k", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4_4b_bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x4bf16_1k", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4_16b_bf16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4bf16", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_4x4x4bf16_1k", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8_bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8bf16", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x8bf16_1k", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16_bf16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16bf16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x16bf16_1k", 0, {}, 4),
                 // These two instructions are new
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_bf16", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_bf16", 0, {}, 8),
                 // V_MFMA_I32_{*}_I8
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x4_2b_i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x4i8", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x4_4b_i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x4i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_4x4x4_16b_i8", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_4x4x4i8", 0, {}, 2),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x16_i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x16i8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x32_i8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x32i8", 0, {}, 4),
                 // These two instructions are new
                 rocRoller::GPUInstructionInfo("v_mfma_i32_16x16x64_i8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_i32_32x32x32_i8", 0, {}, 8),
                 // V_MFMA_F64_{*}_F64
                 rocRoller::GPUInstructionInfo("v_mfma_f64_16x16x4_f64", 0, {}, 8),
                 // gfx950 increases the cycles of v_mfma_f64_16x16x4f64 to 64 (32 in gfx940)
                 rocRoller::GPUInstructionInfo("v_mfma_f64_16x16x4f64", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_f64_4x4x4_4b_f64", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f64_4x4x4f64", 0, {}, 4),
                 // V_MFMA_F32_{*}_BF8_BF8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_bf8_bf8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_bf8_bf8", 0, {}, 8),
                 // V_MFMA_F32_{*}_BF8_FP8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_bf8_fp8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_bf8_fp8", 0, {}, 8),
                 // V_MFMA_F32_{*}_FP8_BF8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_fp8_bf8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_fp8_bf8", 0, {}, 8),
                 // V_MFMA_F32_{*}_BF8_BF8
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x32_fp8_fp8", 0, {}, 4),
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x16_fp8_fp8", 0, {}, 8),
                 // V_MFMA_F32_{*}_F8F6F4 (new in 950)
                 // TODO: instructions below have different cycles depending on the matrix type,
                 //       but currently GPUInstructionInfo assumes each instruction has only
                 //       a fixed latency.
                 // Using passes instead of cycles (1 pass = 4 cycles)
                 // if (either matrix is F8) -> 8 passes   else() -> 4 passes
                 rocRoller::GPUInstructionInfo("v_mfma_f32_16x16x128_f8f6f4", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_mfma_scale_f32_16x16x128_f8f6f4", 0, {}, 8),
                 // if (either matrix is F8) -> 16 passes  else() -> 8 passes
                 rocRoller::GPUInstructionInfo("v_mfma_f32_32x32x64_f8f6f4", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_mfma_scale_f32_32x32x64_f8f6f4", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_prng_b32", 0, {}, 1),
                 // DS_READ_B64_TR_B{16,8,4} and DS_READ_B96_TR_B6
                 rocRoller::GPUInstructionInfo("ds_read_b64_tr_b16",
                                               1,
                                               {rocRoller::GPUWaitQueueType::DSQueue},
                                               4,
                                               /*implicitAccess*/ false,
                                               /*branch*/ false,
                                               (1 << 16) - 1),
                 rocRoller::GPUInstructionInfo("ds_read_b64_tr_b8",
                                               1,
                                               {rocRoller::GPUWaitQueueType::DSQueue},
                                               4,
                                               /*implicitAccess*/ false,
                                               /*branch*/ false,
                                               (1 << 16) - 1),
                 rocRoller::GPUInstructionInfo("ds_read_b64_tr_b4",
                                               1,
                                               {rocRoller::GPUWaitQueueType::DSQueue},
                                               4,
                                               /*implicitAccess*/ false,
                                               /*branch*/ false,
                                               (1 << 16) - 1),
                 rocRoller::GPUInstructionInfo("ds_read_b96_tr_b6",
                                               1,
                                               {rocRoller::GPUWaitQueueType::DSQueue},
                                               8,
                                               /*implicitAccess*/ false,
                                               /*branch*/ false,
                                               (1 << 16) - 1),
             }},
            {gfx120XISAs(),
             {
                 rocRoller::GPUInstructionInfo("v_wmma_f32_16x16x16_f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_wmma_f32_16x16x16_bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_wmma_f16_16x16x16_f16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_wmma_bf16_16x16x16_bf16", 0, {}, 16),
                 rocRoller::GPUInstructionInfo("v_wmma_f32_16x16x16_fp8_bf8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_wmma_f32_16x16x16_bf8_fp8", 0, {}, 8),
                 rocRoller::GPUInstructionInfo("v_wmma_f32_16x16x16_bf8_bf8", 0, {}, 8),
             }},
            {gfx9ISAs(),
             {
                 rocRoller::GPUInstructionInfo(
                     "v_fma_mix_f32",
                     -1,
                     {},
                     -1), //TODO: Remove this when MRISA supports instruction, or instruction is not needed anymore
             }},
    };

    const std::unordered_map<std::string, std::vector<rocRoller::GPUArchitectureTarget>>
        ImplicitReadInstructions = {
            {"s_addc_u32", gfx9ISAs()},
            {"s_subb_u32", gfx9ISAs()},
            {"s_cbranch_scc0", gfx9ISAs()},
            {"s_cbranch_scc1", gfx9ISAs()},
            {"s_cbranch_vccz", gfx9ISAs()},
            {"s_cbranch_vccnz", gfx9ISAs()},
            {"s_cbranch_execz", gfx9ISAs()},
            {"s_cbranch_execnz", gfx9ISAs()},
            {"s_cselect_b32", gfx9ISAs()},
            {"s_cselect_b64", gfx9ISAs()},
            {"s_cmovk_i32", gfx9ISAs()},
            {"s_cmov_b32", gfx9ISAs()},
            {"s_cmov_b64", gfx9ISAs()},
    };
}
