// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

TEST(SkinnyGEMM, KernelNameMatchesTheReference) {
  EXPECT_EQ(skinnyGEMMKernelName({}), "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_"
                                      "PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_"
                                      "GRID_3D1");

  SkinnyGEMMConfig ringless;
  ringless.prefetch = 1;
  ringless.upIters = 1;
  EXPECT_EQ(skinnyGEMMKernelName(ringless),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS1_PARTIAL_STEPS0_PF1_GROUP1_"
            "NUM_XCDS1_GRID_3D1");

  SkinnyGEMMConfig persistent;
  persistent.tn = 4;
  persistent.lanesK = 16;
  persistent.waves = 1;
  persistent.partialSteps = 2;
  persistent.grid3d = false;
  EXPECT_EQ(skinnyGEMMKernelName(persistent),
            "gemm_BM1_TN4_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS2_PF2_GROUP1_"
            "NUM_XCDS1_GRID_3D0");

  SkinnyGEMMConfig grouped;
  grouped.group = 4;
  grouped.numXcds = 8;
  EXPECT_EQ(skinnyGEMMKernelName(grouped),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP4_"
            "NUM_XCDS8_GRID_3D1");
}

std::vector<SkinnyGEMMConfig> invalidConfigs() {
  std::vector<SkinnyGEMMConfig> configs(12);
  configs[0].lanesK = 32;
  configs[1].lanesK = 0;
  configs[2].bm = 0;
  configs[3].tn = 0;
  configs[4].waves = 0;
  configs[5].waves = 17;
  configs[6].prefetch = -1;
  configs[7].upIters = -1;
  configs[8].partialSteps = 3;
  configs[9].partialSteps = -1;
  configs[10].group = 0;
  configs[11].numXcds = 0;
  return configs;
}

TEST(SkinnyGEMM, RejectsInvalidKnobs) {
  Target target = Target::get("gfx950");
  for (const SkinnyGEMMConfig &config : invalidConfigs()) {
    EXPECT_THROW(skinnyGEMMKernelName(config), std::invalid_argument);
    EXPECT_THROW(emitSkinnyGEMM(target, config), std::invalid_argument);
  }
}

TEST(SkinnyGEMM, EmitIsNotImplementedYet) {
  Target target = Target::get("gfx950");
  try {
    emitSkinnyGEMM(target, {});
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("not implemented"),
              std::string::npos);
  }
}

bool isLabelOrComment(const std::string &line) {
  return line.empty() || line.back() == ':' || line[0] == ';' ||
         line.rfind("//", 0) == 0 || line.rfind("/*", 0) == 0;
}

// Returns the mnemonics from the kernel symbol to s_endpgm.
std::vector<std::string> bodyMnemonics(const std::string &assembly) {
  std::vector<std::string> mnemonics;
  std::istringstream stream(assembly);
  std::string line;
  while (std::getline(stream, line) && line != "splitk_reduce:")
    ;
  while (std::getline(stream, line)) {
    std::istringstream words(line);
    std::string first;
    words >> first;
    if (isLabelOrComment(first))
      continue;
    mnemonics.push_back(first);
    if (first == "s_endpgm")
      break;
  }
  return mnemonics;
}

TEST(SplitKReduce, EmitsTheReferenceSchedule) {
  std::string assembly = emitSplitKReduce(Target::get("gfx950"));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> expected = {
      "s_load_dword",       "s_load_dword",   "s_load_dword",
      "s_load_dword",       "s_load_dwordx2", "s_load_dwordx2",
      "s_lshl_b32",         "v_add_u32",      "s_waitcnt",
      "v_cmp_gt_u32",       "s_sub_u32",      "v_min_i32",
      "s_lshl_b32",         "s_mul_i32",      "s_mul_i32",
      "s_lshl_b32",         "s_add_u32",      "s_mul_i32",
      "s_lshl_b32",         "s_add_u32",      "s_mov_b32",
      "s_mov_b32",          "v_lshlrev_b32",  "s_mul_i32",
      "v_add_u32",          "s_mov_b32",      "v_mov_b32",
      "s_mul_i32",          "v_add_u32",      "buffer_load_dword",
      "s_waitcnt",          "v_add_f32",      "s_add_u32",
      "s_cmp_lt_i32",       "s_cbranch_scc1", "v_lshrrev_b32",
      "s_and_saveexec_b64", "s_cbranch_vccz", "v_lshlrev_b32",
      "s_lshl_b32",         "v_add_u32",      "s_mul_i32",
      "s_lshl_b32",         "v_add_u32",      "buffer_store_short",
      "s_mov_b64",          "s_endpgm"};
  EXPECT_EQ(bodyMnemonics(assembly), expected);
  EXPECT_EQ(countLinesContaining(assembly, " offen"), 2);

  expectLinesInOrder(
      assembly, {"s[0:1], 28", "s[0:1], 24", "s[0:1], 20", "s[0:1], 16",
                 "s[0:1], 8", "s[0:1], 0", "s_waitcnt lgkmcnt(0)",
                 "v_cmp_gt_u32 vcc", "buffer_load_dword", "s_waitcnt vmcnt(0)",
                 "v_lshrrev_b32", "s_and_saveexec_b64", "s_cbranch_vccz",
                 "buffer_store_short", "s_mov_b64 exec", "s_endpgm"});
}

TEST(SplitKReduce, DeclaresTheReferenceArguments) {
  std::string assembly = emitSplitKReduce(Target::get("gfx950"));
  SCOPED_TRACE("assembly:\n" + assembly);
  EXPECT_NE(assembly.find(".amdhsa_system_sgpr_workgroup_id_y 1"),
            std::string::npos);
  EXPECT_EQ(assembly.find("workgroup_id_z 1"), std::string::npos);
  expectLinesInOrder(assembly,
                     {".name: splitk_reduce", ".kernarg_segment_size: 56",
                      ".max_flat_workgroup_size: 256"});
  expectLinesInOrder(assembly, {".offset: 32", "hidden_group_size_x",
                                ".offset: 36", "hidden_group_size_y",
                                ".offset: 40", "hidden_group_size_z",
                                ".offset: 44", "hidden_block_count_x",
                                ".offset: 48", "hidden_block_count_y",
                                ".offset: 52", "hidden_block_count_z"});
}

} // namespace
