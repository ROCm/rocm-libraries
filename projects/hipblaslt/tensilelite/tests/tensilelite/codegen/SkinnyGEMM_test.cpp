// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

std::string rejectionMessage(const SkinnyGEMMConfig &config) {
  try {
    emitSkinnyGEMM(Target::get("gfx950"), config);
  } catch (const std::invalid_argument &error) {
    return error.what();
  }
  ADD_FAILURE() << "expected std::invalid_argument";
  return {};
}

TEST(SkinnyGEMM, RejectsUnimplementedFeatures) {
  SkinnyGEMMConfig partial;
  partial.partialSteps = 2;
  EXPECT_EQ(rejectionMessage(partial),
            "partialSteps != 0 (partial K steps) is not implemented yet");

  SkinnyGEMMConfig persistent;
  persistent.grid3d = false;
  EXPECT_EQ(rejectionMessage(persistent),
            "grid3d = false (persistent launch) is not implemented yet");

  SkinnyGEMMConfig grouped;
  grouped.group = 4;
  EXPECT_EQ(rejectionMessage(grouped), "grid3d requires group = 1");

  SkinnyGEMMConfig remapped;
  remapped.numXcds = 8;
  EXPECT_EQ(rejectionMessage(remapped), "grid3d requires numXcds = 1");
}

TEST(SkinnyGEMM, RejectsConfigsOverTheRegisterBudget) {
  SkinnyGEMMConfig wide;
  wide.bm = 8;
  wide.tn = 8;
  wide.lanesK = 16;
  wide.waves = 1;
  wide.prefetch = 4;
  EXPECT_NE(rejectionMessage(wide).find("config exceeds the register budget: "),
            std::string::npos);
}

SkinnyGEMMConfig config(int bm, int tn, int lanesK, int waves, int prefetch,
                        int upIters = 0) {
  SkinnyGEMMConfig result;
  result.upIters = upIters;
  result.bm = bm;
  result.tn = tn;
  result.lanesK = lanesK;
  result.waves = waves;
  result.prefetch = prefetch;
  return result;
}

std::string emitted(const SkinnyGEMMConfig &config) {
  return emitSkinnyGEMM(Target::get("gfx950"), config);
}

std::string trimmed(const std::string &line) {
  auto isBlank = [](char c) { return c == ' ' || c == '\t'; };
  std::string::const_iterator first =
      std::find_if_not(line.begin(), line.end(), isBlank);
  std::string::const_reverse_iterator last =
      std::find_if_not(line.rbegin(), line.rend(), isBlank);
  if (first >= last.base())
    return "";
  return std::string(first, last.base());
}

// Returns the lines from the first one that starts with from up to the first
// later one that contains to.
std::vector<std::string> linesBetween(const std::string &assembly,
                                      std::string_view from,
                                      std::string_view to) {
  std::vector<std::string> lines;
  std::istringstream stream(assembly);
  std::string line;
  while (std::getline(stream, line)) {
    line = trimmed(line);
    if (line.rfind(from, 0) == 0)
      break;
  }
  if (stream) {
    lines.push_back(line);
    while (std::getline(stream, line)) {
      line = trimmed(line);
      lines.push_back(line);
      if (line.find(to) != std::string::npos)
        break;
    }
  }
  return lines;
}

std::vector<std::string> waitsOf(const std::vector<std::string> &lines) {
  std::vector<std::string> waits;
  for (const std::string &line : lines)
    if (line.rfind("s_waitcnt", 0) == 0)
      waits.push_back(line);
  return waits;
}

std::string joined(const std::vector<std::string> &lines) {
  std::string text;
  for (const std::string &line : lines)
    text += line + "\n";
  return text;
}

std::string vmcnt(int count) {
  return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
}

std::string lgkmcnt(int count) {
  return "s_waitcnt lgkmcnt(" + std::to_string(count) + ")";
}

TEST(SkinnyGEMM, AssemblyHasTheReferenceAbi) {
  std::string assembly = emitted(config(1, 1, 16, 2, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {".name: gemm_BM1_TN1_LANES_K16_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF1_"
       "GROUP1_NUM_XCDS1_GRID_3D1",
       ".kernarg_segment_size: 104", ".group_segment_fixed_size: 0",
       ".max_flat_workgroup_size: 128"});
  // The pointers, the scalars and the hidden arguments of the reference.
  expectLinesInOrder(
      assembly,
      {".name: D",       ".offset: 0",  ".name: A",       ".offset: 8",
       ".name: B",       ".offset: 16", ".name: scalesA", ".offset: 24",
       ".name: scalesB", ".offset: 32", ".name: ws",      ".offset: 40",
       ".name: ldD",     ".offset: 48", ".name: ldA",     ".offset: 52",
       ".name: ldB",     ".offset: 56", ".name: m",       ".offset: 60",
       ".name: n",       ".offset: 64", ".name: k",       ".offset: 68",
       ".name: spk",     ".offset: 72"});
  expectLinesInOrder(assembly, {".offset: 76", "hidden_group_size_x",
                                ".offset: 80", "hidden_group_size_y",
                                ".offset: 84", "hidden_group_size_z",
                                ".offset: 88", "hidden_block_count_x",
                                ".offset: 92", "hidden_block_count_y",
                                ".offset: 96", "hidden_block_count_z"});
  EXPECT_NE(assembly.find(".amdhsa_system_sgpr_workgroup_id_y 1"),
            std::string::npos);
  EXPECT_NE(assembly.find(".amdhsa_system_sgpr_workgroup_id_z 1"),
            std::string::npos);
  EXPECT_NE(assembly.find(".amdhsa_group_segment_fixed_size 0"),
            std::string::npos);
}

// The waits of the steady loop follow from the issue order. A ring slot is
// L = 5 * NC loads in flight (NC scale dwords, then 4 chunks of NC), and a use
// waits for the loads issued after it: the rest of its slot, the PF - 1 later
// slots and the refill loads of the slot itself. For NC = 2 and PF = 1 that is
// the reference's 9 and 8 for every column.
TEST(SkinnyGEMM, PrimarySteadyLoopWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  constexpr int columns = 2;
  constexpr int loadsPerSlot = 5 * columns;
  std::vector<std::string> expected;
  for (int c = 0; c < columns; ++c)
    expected.push_back(vmcnt(loadsPerSlot - 1 - c));
  // The permutes of the two columns are the 2 * (NC - 1 - c) newest of the LDS
  // operations in flight.
  for (int c = 0; c < columns; ++c)
    expected.push_back(lgkmcnt(2 * (columns - 1 - c)));
  for (int chunk = 0; chunk < 4; ++chunk)
    for (int c = 0; c < columns; ++c)
      expected.push_back(vmcnt(loadsPerSlot - 1 - c));
  EXPECT_EQ(waitsOf(loop), expected);
  EXPECT_EQ(expected.size(), 12u);
  EXPECT_EQ(expected.front(), "s_waitcnt vmcnt(9)");
  EXPECT_EQ(expected[1], "s_waitcnt vmcnt(8)");
}

TEST(SkinnyGEMM, DecodeFirstSteadyLoopWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 2));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  // One more slot is in flight behind each use: 19 and 18, the reference's
  // values, in both unrolled slots.
  constexpr int columns = 2;
  constexpr int loadsPerSlot = 5 * columns;
  std::vector<std::string> expected;
  for (int slot = 0; slot < 2; ++slot) {
    for (int c = 0; c < columns; ++c)
      expected.push_back(vmcnt(loadsPerSlot + loadsPerSlot - 1 - c));
    for (int c = 0; c < columns; ++c)
      expected.push_back(lgkmcnt(2 * (columns - 1 - c)));
    for (int chunk = 0; chunk < 4; ++chunk)
      for (int c = 0; c < columns; ++c)
        expected.push_back(vmcnt(loadsPerSlot + loadsPerSlot - 1 - c));
  }
  EXPECT_EQ(waitsOf(loop), expected);
  EXPECT_EQ(expected.front(), "s_waitcnt vmcnt(19)");
}

// The prologue loads that the epilogue overwrites may still be in flight when
// a split has fewer iterations than the ring has slots.
TEST(SkinnyGEMM, EpilogueWaitsForThePrologueLoadsItOverwrites) {
  struct Case {
    int prefetch;
    int outstanding;
  };
  // The first register of the epilogue is the first chunk of the ring, so only
  // the loads issued after it are left: the reference's 7 for PF = 1, and 17
  // for PF = 2.
  for (Case current : {Case{1, 7}, Case{2, 17}}) {
    std::string assembly = emitted(config(1, 1, 16, 1, current.prefetch));
    SCOPED_TRACE("assembly:\n" + assembly);
    std::vector<std::string> epilogue =
        linesBetween(assembly, "label_kloop_end_", "quad_perm:[1,0,3,2]");
    // The last loop's exit label starts the arms, so start at the last wait
    // before the first reduction instruction.
    std::vector<std::string> waits = waitsOf(epilogue);
    ASSERT_FALSE(waits.empty());
    EXPECT_EQ(waits.back(), vmcnt(current.outstanding));
  }
}

// Without a ring each iteration issues its loads and consumes them at once, so
// the waits count down from the L loads of the iteration.
TEST(SkinnyGEMM, NoPrefetchLoopWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 0));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  std::vector<std::string> expected = {
      vmcnt(9), vmcnt(8), lgkmcnt(2), lgkmcnt(0), vmcnt(7), vmcnt(6),
      vmcnt(5), vmcnt(4), vmcnt(3),   vmcnt(2),   vmcnt(1), vmcnt(0)};
  EXPECT_EQ(waitsOf(loop), expected);
  // The loads of the iteration all issue before the first wait.
  std::string text = joined(loop);
  EXPECT_EQ(text.find("buffer_load", text.find("s_waitcnt")),
            std::string::npos);
}

// The remaining 1 to 2 * PF - 1 iterations run in an arm of a chain of ifs, and
// the steady loop end is max((NIfull / PF - 1) * PF, 0), as shifts for a power
// of two and as LLVM's multiply by a magic number otherwise.
TEST(SkinnyGEMM, RemainderArmsAndSteadyLoopEnd) {
  for (int prefetch : {1, 2, 3, 4}) {
    std::string assembly = emitted(config(1, 1, 16, 1, prefetch));
    SCOPED_TRACE("prefetch " + std::to_string(prefetch));
    EXPECT_EQ(countLinesContaining(assembly, "s_cmp_eq_i32 "),
              2 * prefetch - 1);
    EXPECT_EQ(countLinesContaining(assembly, "s_mul_hi_i32"),
              prefetch == 3 ? 1 : 0);
  }
  EXPECT_NE(emitted(config(1, 1, 16, 1, 3)).find("1431655766"),
            std::string::npos);
}

TEST(SkinnyGEMM, PerChunkOpOrder) {
  std::string assembly = emitted(config(1, 1, 16, 1, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  std::string text = joined(loop);
  // Chunk 0: the decode shifts by 23 and masks the exponent, then per dword
  // two converts and a dot for the low and the high pair, then the refill.
  expectLinesInOrder(text, {"v_lshlrev_b32", ", 23, ", "v_and_b32",
                            "2139095040", "v_lshlrev_b32", "v_and_b32",
                            "v_cvt_scalef32_pk_bf16_fp8",
                            "v_cvt_scalef32_pk_bf16_fp8", "v_dot2_f32_bf16",
                            "v_cvt_scalef32_pk_bf16_fp8 v", "op_sel:[1,0,0]",
                            "v_cvt_scalef32_pk_bf16_fp8", "op_sel:[1,0,0]",
                            "v_dot2_f32_bf16", "v_cvt_scalef32_pk_bf16_fp8",
                            "v_cvt_scalef32_pk_bf16_fp8", "v_dot2_f32_bf16"});
  EXPECT_EQ(countLinesContaining(text, "v_dot2_f32_bf16"), 4 * 8);
  EXPECT_EQ(countLinesContaining(text, "op_sel:[1,0,0]"), 4 * 4 * 2);
  // Each chunk refills its two columns after its compute, and the last decode
  // shifts right by 1.
  std::vector<std::string> chunk0 =
      linesBetween(text, "v_lshlrev_b32", "buffer_load_dwordx4");
  EXPECT_EQ(countLinesContaining(joined(chunk0), "v_dot2_f32_bf16"), 8);
  EXPECT_EQ(countLinesContaining(text, "buffer_load_dwordx4"), 8);
  EXPECT_EQ(countLinesContaining(text, "buffer_load_dword "), 2);
  expectLinesInOrder(text, {"v_lshrrev_b32", ", 1, ", "v_and_b32"});
}

// The mnemonic of a line with its ds offset, if any, and the other operands
// dropped.
std::string shape(const std::string &line) {
  std::string mnemonic = line.substr(0, line.find(' '));
  if (mnemonic.rfind("ds_", 0) != 0)
    return line.rfind("s_waitcnt", 0) == 0 ? line : mnemonic;
  std::string::size_type offset = line.find("offset:");
  return offset == std::string::npos ? mnemonic
                                     : mnemonic + line.substr(offset);
}

// The lines of the given mnemonics, reduced to their shape.
std::vector<std::string> shapesOf(const std::vector<std::string> &lines,
                                  const std::vector<std::string> &prefixes) {
  std::vector<std::string> shapes;
  for (const std::string &line : lines)
    for (const std::string &prefix : prefixes)
      if (line.rfind(prefix, 0) == 0) {
        shapes.push_back(shape(line));
        break;
      }
  return shapes;
}

// The up-front scales are one dword load per column and entry that issue
// before every other load.
TEST(SkinnyGEMM, UpFrontLoadsComeFirst) {
  std::string assembly = emitted(config(1, 1, 16, 1, 2, 5));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loads;
  std::istringstream stream(assembly);
  for (std::string line; std::getline(stream, line);) {
    line = trimmed(line);
    if (line.rfind("buffer_load", 0) == 0)
      loads.push_back(line.substr(0, line.find(' ')));
  }
  constexpr int scaleLoads = 5 * 2;
  ASSERT_GT(loads.size(), static_cast<size_t>(scaleLoads));
  for (int i = 0; i < scaleLoads; ++i)
    EXPECT_EQ(loads[i], "buffer_load_dword") << i;
  EXPECT_EQ(loads[scaleLoads], "buffer_load_dwordx4");
  // Nothing issues scales again: the prologue, loop and arms load chunks.
  EXPECT_EQ(countLinesContaining(assembly, "buffer_load_dword "), scaleLoads);
}

// Each entry packs in sequence: the vmcnt of the fetch of a column counts the
// later scale loads and the PF slots of the prologue (L = 8 for NC = 2), then
// the permutes wait for the LDS operations behind them.
TEST(SkinnyGEMM, UpFrontPerWavePackWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 2, 5));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> pack =
      linesBetween(assembly, vmcnt(25), "label_kloop_0:");
  ASSERT_FALSE(pack.empty());
  std::vector<std::string> expected;
  for (int entry = 0; entry < 5; ++entry) {
    expected.insert(expected.end(),
                    {vmcnt(25 - 2 * entry), "ds_bpermute_b32",
                     "ds_bpermute_b32", vmcnt(24 - 2 * entry),
                     "ds_bpermute_b32", "ds_bpermute_b32", lgkmcnt(2),
                     "v_perm_b32", lgkmcnt(0), "v_perm_b32"});
  }
  EXPECT_EQ(shapesOf(pack, {"s_waitcnt", "ds_bpermute_b32", "v_perm_b32"}),
            expected);
}

// The block stages the raw dwords in LDS, each write waiting only for its own
// load, publishes them with a barrier that leaves the prologue loads in
// flight, and every wave reads the two dwords of its entries back.
TEST(SkinnyGEMM, UpFrontSharedPackOrderAndWaits) {
  std::string assembly = emitted(config(1, 1, 64, 4, 2, 2));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> pack =
      linesBetween(assembly, vmcnt(19), "label_kloop_0:");
  ASSERT_FALSE(pack.empty());
  std::vector<std::string> expected = {vmcnt(19),  "ds_write_b32offset:0",
                                       vmcnt(18),  "ds_write_b32offset:1024",
                                       vmcnt(17),  "ds_write_b32offset:2048",
                                       vmcnt(16),  "ds_write_b32offset:3072",
                                       lgkmcnt(0), "s_barrier"};
  for (int entry = 0; entry < 2; ++entry) {
    std::string first = std::to_string(entry * 2048);
    std::string second = std::to_string(entry * 2048 + 1024);
    std::string hiFirst = std::to_string(entry * 2048 + 64);
    std::string hiSecond = std::to_string(entry * 2048 + 1088);
    expected.insert(
        expected.end(),
        {"ds_read_b32offset:" + first, "ds_read_b32offset:" + hiFirst,
         "ds_read_b32offset:" + second, "ds_read_b32offset:" + hiSecond,
         lgkmcnt(2), "v_perm_b32", lgkmcnt(0), "v_perm_b32"});
  }
  EXPECT_EQ(shapesOf(pack, {"s_waitcnt", "ds_write_b32", "ds_read_b32",
                            "s_barrier", "v_perm_b32"}),
            expected);
}

// The mask of each entry dies after its sweep: one compare and one 64-bit
// select per entry, then a v_cndmask_b32 per column, and the loop neither
// loads scales nor permutes lanes.
TEST(SkinnyGEMM, UpFrontSelectChainIsEntryOuter) {
  std::string assembly = emitted(config(1, 1, 16, 1, 2, 5));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  std::vector<std::string> chain;
  for (const std::string &line : loop) {
    if (line.rfind("s_cmp_eq_i32", 0) == 0)
      chain.push_back(line.substr(line.rfind(',') + 2));
    else if (line.rfind("s_cselect_b64 vcc, -1, 0", 0) == 0)
      chain.push_back("select");
    else if (line.rfind("v_cndmask_b32", 0) == 0)
      chain.push_back("cndmask");
  }
  std::vector<std::string> expected;
  for (int slot = 0; slot < 2; ++slot)
    for (int entry = 1; entry < 5; ++entry)
      expected.insert(expected.end(),
                      {std::to_string(entry), "select", "cndmask", "cndmask"});
  EXPECT_EQ(chain, expected);
  std::string text = joined(loop);
  EXPECT_EQ(countLinesContaining(text, "ds_bpermute_b32"), 0);
  EXPECT_EQ(countLinesContaining(text, "buffer_load_dword "), 0);
}

TEST(SkinnyGEMM, UpFrontLoopWaits) {
  // Every chunk of the loop waits for its two columns behind the next slot:
  // L = 8 loads per slot and one more slot in flight, so 15 and 14.
  std::string shared = emitted(config(1, 1, 64, 4, 2, 2));
  SCOPED_TRACE("assembly:\n" + shared);
  std::vector<std::string> loop =
      linesBetween(shared, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  std::vector<std::string> expected;
  for (int chunk = 0; chunk < 8; ++chunk)
    expected.insert(expected.end(), {vmcnt(15), vmcnt(14)});
  EXPECT_EQ(waitsOf(loop), expected);

  // Without a ring the loads of an iteration, 4 chunks of 4 columns, count
  // down to zero.
  std::string noPrefetch = emitted(config(2, 2, 16, 1, 0, 4));
  SCOPED_TRACE("assembly:\n" + noPrefetch);
  loop = linesBetween(noPrefetch,
                      "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  expected.clear();
  for (int load = 15; load >= 0; --load)
    expected.push_back(vmcnt(load));
  EXPECT_EQ(waitsOf(loop), expected);
}

std::string ldsSegmentSize(const SkinnyGEMMConfig &config) {
  return ".group_segment_fixed_size: " +
         std::to_string(config.upIters * (config.bm + config.tn) *
                        config.waves * 256);
}

// Only shared scales use LDS: UP * NC * WAVES * 256 bytes.
TEST(SkinnyGEMM, UpFrontLdsSize) {
  SkinnyGEMMConfig single = config(1, 1, 64, 4, 2, 2);
  EXPECT_NE(emitted(single).find(".group_segment_fixed_size: 4096"),
            std::string::npos);
  EXPECT_NE(emitted(config(2, 2, 64, 2, 1, 3))
                .find(ldsSegmentSize(config(2, 2, 64, 2, 1, 3))),
            std::string::npos);
  for (const SkinnyGEMMConfig &unshared :
       {config(1, 1, 16, 4, 3, 5), config(1, 1, 64, 1, 1, 3),
        config(1, 1, 64, 4, 2, 0)}) {
    std::string assembly = emitted(unshared);
    EXPECT_NE(assembly.find(".group_segment_fixed_size: 0"), std::string::npos);
    EXPECT_EQ(assembly.find("ds_write_b32"), std::string::npos);
  }

  // 11 * 4 * 16 * 256 bytes exceed the 160 KiB of the target.
  std::string message = rejectionMessage(config(2, 2, 64, 16, 1, 11));
  EXPECT_NE(message.find("config exceeds the LDS budget: "), std::string::npos);
}

// The rotation of up-front scales is modulo the literal UP_ITERS, a mask for a
// power of two, LLVM's magic division otherwise and nothing for one entry.
TEST(SkinnyGEMM, UpFrontRotationModulusIsUpIters) {
  auto masksBy = [](const std::string &assembly, std::string_view mask) {
    std::istringstream stream(assembly);
    for (std::string line; std::getline(stream, line);) {
      line = trimmed(line);
      if (line.rfind("s_and_b32", 0) == 0 && line.size() > mask.size() &&
          line.compare(line.size() - mask.size(), mask.size(), mask) == 0)
        return true;
    }
    return false;
  };
  std::string ring = emitted(config(1, 1, 64, 4, 2));
  EXPECT_NE(ring.find("v_rcp_iflag_f32"), std::string::npos);

  std::string two = emitted(config(1, 1, 64, 4, 2, 2));
  EXPECT_EQ(two.find("v_rcp_iflag_f32"), std::string::npos);
  EXPECT_TRUE(masksBy(two, ", 1"));

  std::string four = emitted(config(1, 1, 16, 1, 2, 4));
  EXPECT_TRUE(masksBy(four, ", 3"));

  std::string five = emitted(config(1, 1, 16, 1, 2, 5));
  EXPECT_EQ(five.find("v_rcp_iflag_f32"), std::string::npos);
  EXPECT_EQ(countLinesContaining(five, "s_mul_hi_i32"), 1);
  EXPECT_NE(five.find("1717986919"), std::string::npos);

  // One entry has no rotation, and its scales are decoded before the loop.
  std::string one = emitted(config(1, 1, 64, 4, 1, 1));
  SCOPED_TRACE("assembly:\n" + one);
  EXPECT_EQ(one.find("v_rcp_iflag_f32"), std::string::npos);
  EXPECT_EQ(countLinesContaining(one, "s_mul_hi_i32"), 0);
  EXPECT_FALSE(masksBy(one, ", 0"));
  std::vector<std::string> loop =
      linesBetween(one, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  EXPECT_EQ(countLinesContaining(joined(loop), ", 23, "), 0);
  // Chunk 0 of the two columns is the only decode that shifts by 23.
  EXPECT_EQ(countLinesContaining(one, ", 23, "), 2);
}

// Counts the v_nop that precede each DPP add of the reduction of a single
// accumulator, in order.
std::vector<int> nopsBeforeDpp(const std::string &assembly) {
  std::vector<int> counts;
  std::istringstream stream(assembly);
  int nops = 0;
  bool inReduction = false;
  for (std::string line; std::getline(stream, line);) {
    line = trimmed(line);
    if (line == "v_nop") {
      ++nops;
      continue;
    }
    bool dpp = line.rfind("v_add_f32", 0) == 0 &&
               (line.find("quad_perm") != std::string::npos ||
                line.find("mirror") != std::string::npos ||
                line.find("row_bcast") != std::string::npos);
    if (dpp) {
      inReduction = true;
      counts.push_back(nops);
    }
    if (line.rfind("v_readlane_b32", 0) == 0)
      counts.push_back(nops);
    if (inReduction && line.rfind("v_cmp_eq_i32", 0) == 0)
      break;
    nops = 0;
  }
  return counts;
}

TEST(SkinnyGEMM, ReduceForEveryLanesK) {
  // Three nops cover the dot's write, two every DPP read of a VALU result.
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 4, 1, 1))),
            (std::vector<int>{3, 2}));
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 8, 1, 1))),
            (std::vector<int>{3, 2, 2}));
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 16, 1, 1))),
            (std::vector<int>{3, 2, 2, 2}));
  // 64 lanes add the row broadcasts and read lane 63 after one nop; the move
  // of the lane's value takes two more, which the reference omits.
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 64, 1, 1))),
            (std::vector<int>{3, 2, 2, 2, 2, 2, 1}));

  std::string lanes16 = emitted(config(1, 1, 16, 1, 1));
  expectLinesInOrder(lanes16,
                     {"quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf "
                      "bound_ctrl:1",
                      "quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf "
                      "bound_ctrl:1",
                      "row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1",
                      "row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1",
                      "v_cmp_eq_i32"});
  EXPECT_EQ(lanes16.find("row_bcast"), std::string::npos);
  EXPECT_EQ(lanes16.find("v_readlane_b32"), std::string::npos);

  std::string lanes64 = emitted(config(1, 1, 64, 1, 1));
  expectLinesInOrder(lanes64,
                     {"row_mirror", "row_bcast:15 row_mask:0xa bank_mask:0xf",
                      "row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1",
                      "v_readlane_b32", "v_nop", "v_nop", "v_mov_b32",
                      "v_cmp_eq_i32"});
  EXPECT_NE(lanes64.find(", 63"), std::string::npos);
  // 4 lanes stop after the quad permutes.
  std::string lanes4 = emitted(config(1, 1, 4, 1, 1));
  EXPECT_EQ(lanes4.find("row_half_mirror"), std::string::npos);
}

TEST(SkinnyGEMM, VectorStoresOnlyWhenPreconditionHolds) {
  // One element per lane has no run to vectorize and no orientation split.
  std::string scalar = emitted(config(1, 1, 16, 1, 1));
  EXPECT_EQ(scalar.find("buffer_store_dwordx"), std::string::npos);
  EXPECT_EQ(scalar.find("s_cmp_eq_u32"), std::string::npos);
  EXPECT_EQ(countLinesContaining(scalar, "buffer_store_short"), 1);
  EXPECT_EQ(countLinesContaining(scalar, "buffer_store_dword "), 1);

  // BM = 2, TN = 4: bf16 runs of 2 (one dword) and 4 (dwordx2) elements,
  // behind the branch on the orientation and the one on a full tile.
  std::string vector = emitted(config(2, 4, 16, 1, 1));
  SCOPED_TRACE("assembly:\n" + vector);
  EXPECT_NE(vector.find("buffer_store_dwordx2"), std::string::npos);
  EXPECT_NE(vector.find("buffer_store_dword "), std::string::npos);
  EXPECT_NE(vector.find("v_pack_b32_f16"), std::string::npos);
  EXPECT_NE(vector.find("s_cmp_eq_u32"), std::string::npos);

  // 8 bf16 elements make a dwordx4, and 4 f32 elements do for the workspace.
  EXPECT_NE(emitted(config(8, 2, 16, 1, 1)).find("buffer_store_dwordx4"),
            std::string::npos);
  EXPECT_NE(emitted(config(4, 2, 16, 1, 1)).find("buffer_store_dwordx4"),
            std::string::npos);
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
