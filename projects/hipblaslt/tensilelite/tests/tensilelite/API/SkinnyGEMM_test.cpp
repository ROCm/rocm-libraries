// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/API/SkinnyGEMM.hpp>

#include "Codegen/TestSupport.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

// The symbol lists each knob with its value.
TEST(SkinnyGEMM, KernelNameEncodesTheKnobs) {
  EXPECT_EQ(skinnyGEMMKernelName({}), "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_"
                                      "PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_"
                                      "GRID_3D1_USE_BETA0");

  SkinnyGEMMConfig ringless;
  ringless.prefetch = 1;
  ringless.upIters = 1;
  EXPECT_EQ(skinnyGEMMKernelName(ringless),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS1_PARTIAL_STEPS0_PF1_GROUP1_"
            "NUM_XCDS1_GRID_3D1_USE_BETA0");

  SkinnyGEMMConfig persistent;
  persistent.tn = 4;
  persistent.lanesK = 16;
  persistent.waves = 1;
  persistent.partialSteps = 1;
  persistent.grid3d = false;
  EXPECT_EQ(skinnyGEMMKernelName(persistent),
            "gemm_BM1_TN4_LANES_K16_WAVES1_UP_ITERS0_PARTIAL_STEPS1_PF2_GROUP1_"
            "NUM_XCDS1_GRID_3D0_USE_BETA0");

  SkinnyGEMMConfig grouped;
  grouped.group = 4;
  grouped.numXcds = 8;
  EXPECT_EQ(skinnyGEMMKernelName(grouped),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP4_"
            "NUM_XCDS8_GRID_3D1_USE_BETA0");

  SkinnyGEMMConfig beta;
  beta.useBeta = true;
  EXPECT_EQ(skinnyGEMMKernelName(beta),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_"
            "NUM_XCDS1_GRID_3D1_USE_BETA1");
  beta.grid3d = false;
  EXPECT_EQ(skinnyGEMMKernelName(beta),
            "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_GROUP1_"
            "NUM_XCDS1_GRID_3D0_USE_BETA1");
}

static std::vector<SkinnyGEMMConfig> invalidConfigs() {
  std::vector<SkinnyGEMMConfig> configs(13);
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
  configs[12].partialSteps = 2;
  return configs;
}

// Out-of-range knobs are rejected by the name and by the emitter.
TEST(SkinnyGEMM, RejectsInvalidKnobs) {
  Target target = Target::get("gfx950");
  for (const SkinnyGEMMConfig &config : invalidConfigs()) {
    EXPECT_THROW(skinnyGEMMKernelName(config), std::invalid_argument);
    EXPECT_THROW(emitSkinnyGEMM(target, config), std::invalid_argument);
  }
}

static std::string rejectionMessage(const SkinnyGEMMConfig &config) {
  try {
    emitSkinnyGEMM(Target::get("gfx950"), config);
  } catch (const std::invalid_argument &error) {
    return error.what();
  }
  ADD_FAILURE() << "expected std::invalid_argument";
  return {};
}

// A configuration that needs more registers than the target has is rejected.
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

static SkinnyGEMMConfig config(int32_t bm, int32_t tn, int32_t lanesK,
                               int32_t waves, int32_t prefetch,
                               int32_t upIters = 0, int32_t partialSteps = 0) {
  SkinnyGEMMConfig result;
  result.upIters = upIters;
  result.partialSteps = partialSteps;
  result.bm = bm;
  result.tn = tn;
  result.lanesK = lanesK;
  result.waves = waves;
  result.prefetch = prefetch;
  return result;
}

static std::string emitted(const SkinnyGEMMConfig &config) {
  return emitSkinnyGEMM(Target::get("gfx950"), config);
}

static bool isBlank(char c) { return c == ' ' || c == '\t'; }

static std::string trimmed(const std::string &line) {
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
static std::vector<std::string> linesBetween(const std::string &assembly,
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

static std::vector<std::string> waitsOf(const std::vector<std::string> &lines) {
  std::vector<std::string> waits;
  for (const std::string &line : lines)
    if (line.rfind("s_waitcnt", 0) == 0)
      waits.push_back(line);
  return waits;
}

static std::string joined(const std::vector<std::string> &lines) {
  std::string text;
  for (const std::string &line : lines)
    text += line + "\n";
  return text;
}

static std::string vmcnt(int32_t count) {
  return "s_waitcnt vmcnt(" + std::to_string(count) + ")";
}

static std::string lgkmcnt(int32_t count) {
  return "s_waitcnt lgkmcnt(" + std::to_string(count) + ")";
}

// The kernel declares the kernargs, the workgroup counts and ids.
TEST(SkinnyGEMM, AssemblyHasTheKernelAbi) {
  std::string assembly = emitted(config(1, 1, 16, 2, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {".name: gemm_BM1_TN1_LANES_K16_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF1_"
       "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0",
       ".kernarg_segment_size: 112", ".group_segment_fixed_size: 0",
       ".max_flat_workgroup_size: 128"});
  // The pointers, the scalars and the workgroup counts.
  expectLinesInOrder(assembly, {".name: D",
                                ".offset: 0",
                                ".name: A",
                                ".offset: 8",
                                ".name: B",
                                ".offset: 16",
                                ".name: scalesA",
                                ".offset: 24",
                                ".name: scalesB",
                                ".offset: 32",
                                ".name: ws",
                                ".offset: 40",
                                ".name: C",
                                ".offset: 48",
                                ".name: ldD",
                                ".offset: 56",
                                ".name: ldA",
                                ".offset: 60",
                                ".name: ldB",
                                ".offset: 64",
                                ".name: m",
                                ".offset: 68",
                                ".name: n",
                                ".offset: 72",
                                ".name: k",
                                ".offset: 76",
                                ".name: spk",
                                ".offset: 80",
                                ".name: ldC",
                                ".offset: 84",
                                ".name: alpha",
                                ".offset: 88",
                                ".name: beta",
                                ".offset: 92",
                                ".name: numWorkGroupsX",
                                ".offset: 96",
                                ".name: numWorkGroupsY",
                                ".offset: 100",
                                ".name: numWorkGroupsZ",
                                ".offset: 104"});
  EXPECT_EQ(assembly.find("hidden_"), std::string::npos);
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
// 9 and 8 for every column.
TEST(SkinnyGEMM, PrimarySteadyLoopWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  constexpr int32_t columns = 2;
  constexpr int32_t loadsPerSlot = 5 * columns;
  std::vector<std::string> expected;
  for (int32_t c = 0; c < columns; ++c)
    expected.push_back(vmcnt(loadsPerSlot - 1 - c));
  // The permutes of the two columns are the 2 * (NC - 1 - c) newest of the LDS
  // operations in flight.
  for (int32_t c = 0; c < columns; ++c)
    expected.push_back(lgkmcnt(2 * (columns - 1 - c)));
  for (int32_t chunk = 0; chunk < 4; ++chunk)
    for (int32_t c = 0; c < columns; ++c)
      expected.push_back(vmcnt(loadsPerSlot - 1 - c));
  EXPECT_EQ(waitsOf(loop), expected);
  EXPECT_EQ(expected.size(), 12u);
  EXPECT_EQ(expected.front(), "s_waitcnt vmcnt(9)");
  EXPECT_EQ(expected[1], "s_waitcnt vmcnt(8)");
}

// With PF = 2 the scales are decoded first and one more slot is in flight.
TEST(SkinnyGEMM, DecodeFirstSteadyLoopWaits) {
  std::string assembly = emitted(config(1, 1, 16, 1, 2));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> loop =
      linesBetween(assembly, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  // One more slot is in flight behind each use: 19 and 18, in both
  // unrolled slots.
  constexpr int32_t columns = 2;
  constexpr int32_t loadsPerSlot = 5 * columns;
  std::vector<std::string> expected;
  for (int32_t slot = 0; slot < 2; ++slot) {
    for (int32_t c = 0; c < columns; ++c)
      expected.push_back(vmcnt(loadsPerSlot + loadsPerSlot - 1 - c));
    for (int32_t c = 0; c < columns; ++c)
      expected.push_back(lgkmcnt(2 * (columns - 1 - c)));
    for (int32_t chunk = 0; chunk < 4; ++chunk)
      for (int32_t c = 0; c < columns; ++c)
        expected.push_back(vmcnt(loadsPerSlot + loadsPerSlot - 1 - c));
  }
  EXPECT_EQ(waitsOf(loop), expected);
  EXPECT_EQ(expected.front(), "s_waitcnt vmcnt(19)");
}

// The prologue loads that the epilogue overwrites may still be in flight when
// a split has fewer iterations than the ring has slots.
TEST(SkinnyGEMM, EpilogueWaitsForThePrologueLoadsItOverwrites) {
  struct Case {
    int32_t prefetch;
    int32_t outstanding;
  };
  // The first register of the epilogue is the first chunk of the ring, so only
  // the loads issued after it are left: 7 for PF = 1, and 17 for PF = 2.
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
// of two and as a multiply by a magic number otherwise.
TEST(SkinnyGEMM, RemainderArmsAndSteadyLoopEnd) {
  for (int32_t prefetch : {1, 2, 3, 4}) {
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

// Each chunk decodes its scale, converts and accumulates, then refills.
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
static std::string shape(const std::string &line) {
  std::string mnemonic = line.substr(0, line.find(' '));
  if (mnemonic.rfind("ds_", 0) != 0)
    return line.rfind("s_waitcnt", 0) == 0 ? line : mnemonic;
  std::string::size_type offset = line.find("offset:");
  return offset == std::string::npos ? mnemonic
                                     : mnemonic + line.substr(offset);
}

// The lines of the given mnemonics, reduced to their shape.
static std::vector<std::string>
shapesOf(const std::vector<std::string> &lines,
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
  constexpr int32_t scaleLoads = 5 * 2;
  ASSERT_GT(std::ssize(loads), scaleLoads);
  for (int32_t i = 0; i < scaleLoads; ++i)
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
  for (int32_t entry = 0; entry < 5; ++entry) {
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
  for (int32_t entry = 0; entry < 2; ++entry) {
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
  for (int32_t slot = 0; slot < 2; ++slot)
    for (int32_t entry = 1; entry < 5; ++entry)
      expected.insert(expected.end(),
                      {std::to_string(entry), "select", "cndmask", "cndmask"});
  EXPECT_EQ(chain, expected);
  std::string text = joined(loop);
  EXPECT_EQ(countLinesContaining(text, "ds_bpermute_b32"), 0);
  EXPECT_EQ(countLinesContaining(text, "buffer_load_dword "), 0);
}

// The up-front loops wait per chunk behind the slots in flight.
TEST(SkinnyGEMM, UpFrontLoopWaits) {
  // Every chunk of the loop waits for its two columns behind the next slot:
  // L = 8 loads per slot and one more slot in flight, so 15 and 14.
  std::string shared = emitted(config(1, 1, 64, 4, 2, 2));
  SCOPED_TRACE("assembly:\n" + shared);
  std::vector<std::string> loop =
      linesBetween(shared, "label_kloop_0:", "s_cbranch_scc1 label_kloop_0");
  ASSERT_FALSE(loop.empty());
  std::vector<std::string> expected;
  for (int32_t chunk = 0; chunk < 8; ++chunk)
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
  for (int32_t load = 15; load >= 0; --load)
    expected.push_back(vmcnt(load));
  EXPECT_EQ(waitsOf(loop), expected);
}

static std::string ldsSegmentSize(const SkinnyGEMMConfig &config) {
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

// Whether an s_and_b32 of the assembly ends in the operand mask.
static bool masksBy(const std::string &assembly, std::string_view mask) {
  std::istringstream stream(assembly);
  for (std::string line; std::getline(stream, line);) {
    line = trimmed(line);
    if (line.rfind("s_and_b32", 0) == 0 && line.size() > mask.size() &&
        line.compare(line.size() - mask.size(), mask.size(), mask) == 0)
      return true;
  }
  return false;
}

// The rotation of up-front scales is modulo the literal UP_ITERS, a mask for a
// power of two, a magic division otherwise and nothing for one entry.
TEST(SkinnyGEMM, UpFrontRotationModulusIsUpIters) {
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

// The lines of the assembly, trimmed.
static std::vector<std::string> allLines(const std::string &assembly) {
  std::vector<std::string> lines;
  std::istringstream stream(assembly);
  for (std::string line; std::getline(stream, line);)
    lines.push_back(trimmed(line));
  return lines;
}

// The compute of the last partial step: the lines after the last chunk mask
// label up to the last dot before the epilogue's first DPP add.
static std::vector<std::string> stepCompute(const std::string &assembly) {
  std::vector<std::string> lines = allLines(assembly);
  int32_t first = 0;
  int32_t last = 0;
  for (int32_t i = 0; i < std::ssize(lines); ++i) {
    if (lines[i].rfind("label_tail_mask_skip_", 0) == 0)
      first = i;
    if (lines[i].find("quad_perm") != std::string::npos)
      break;
    if (lines[i].rfind("v_dot2_f32_bf16", 0) == 0)
      last = i;
  }
  return {lines.begin() + first, lines.begin() + last + 1};
}

// The tail step is emitted once, with its four chunk tests and no loop.
TEST(SkinnyGEMM, PartialStepBodyIsEmittedOnce) {
  for (int32_t upIters : {0, 2}) {
    SCOPED_TRACE("upIters " + std::to_string(upIters));
    std::string none = emitted(config(1, 4, 16, 1, 1, upIters, 0));
    EXPECT_EQ(countLinesContaining(none, "s_bitcmp1_b32"), 0);
    EXPECT_EQ(countLinesContaining(none, "label_step_loop_"), 0);

    std::string tail = emitted(config(1, 4, 16, 1, 1, upIters, 1));
    EXPECT_EQ(countLinesContaining(tail, "s_bitcmp1_b32"), 4);
    EXPECT_EQ(countLinesContaining(tail, "s_cbranch_scc1 label_step_loop_"), 0);
  }
}

// The tail of both paths classifies every chunk by scalar compares into the
// FULL (plain loads), EDGE (steered loads) and DEAD (zeroing) arms, waits once
// for all the loads and then masks the EDGE chunks: no dword is loaded and
// waited for one at a time.
static void expectThreeArmTail(const std::string &assembly) {
  constexpr int32_t chunks = 4;
  constexpr int32_t dwords = 4;
  std::vector<std::string> lines = allLines(assembly);
  for (std::string_view label :
       {"label_tail_edge_", "label_tail_zero_", "label_tail_chunk_end_",
        "label_tail_mask_skip_"})
    EXPECT_EQ(std::count_if(lines.begin(), lines.end(),
                            [&](const std::string &line) {
                              return line.rfind(label, 0) == 0;
                            }),
              chunks)
        << label;
  int32_t begin = 0;
  int32_t firstMask = 0;
  int32_t lastArm = 0;
  for (int32_t i = 0; i < std::ssize(lines); ++i) {
    if (begin == 0 && lines[i].rfind("label_tail_edge_", 0) == 0)
      begin = i;
    if (lines[i].rfind("label_tail_chunk_end_", 0) == 0)
      lastArm = i;
    if (lastArm != 0 && firstMask == 0 &&
        lines[i].rfind("s_bitcmp1_b32", 0) == 0)
      firstMask = i;
  }
  ASSERT_GT(begin, 0);
  ASSERT_GT(firstMask, lastArm);
  // One wait for the loads of all the chunks, before the first mask test.
  EXPECT_EQ(lines[lastArm + 1], vmcnt(0));
  int32_t rightMasks = 0;
  for (int32_t i = begin; i < firstMask; ++i)
    EXPECT_NE(lines[i].rfind("buffer_load_dword ", 0), 0u);
  for (int32_t i = firstMask; i < std::ssize(lines); ++i) {
    if (lines[i].rfind("label_tail_skip_", 0) == 0)
      break;
    bool minusOne = lines[i].size() > 4 &&
                    lines[i].compare(lines[i].size() - 4, 4, ", -1") == 0;
    rightMasks += lines[i].rfind("v_lshrrev_b32", 0) == 0 && minusOne;
  }
  EXPECT_EQ(rightMasks, chunks * dwords);
}

TEST(SkinnyGEMM, TailChunksUseThreeArmsAndOneWait) {
  expectThreeArmTail(emitted(config(1, 4, 16, 1, 1, 0, 1)));
  expectThreeArmTail(emitted(config(1, 4, 16, 1, 2, 4, 1)));
}

// The EDGE arm steers the lanes that are dead in the chunk out of range with a
// v_or_b32 of the voffset before its loads, which the FULL arm, before it, does
// not need.
static void expectSteeredEdgeArm(const std::string &assembly) {
  constexpr std::string_view armEnd = "s_branch label_tail_chunk_end_";
  std::vector<std::string> full =
      linesBetween(assembly, "label_tail_begin_", armEnd);
  std::vector<std::string> edge =
      linesBetween(assembly, "label_tail_edge_", armEnd);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(edge.empty());
  EXPECT_GT(countLinesContaining(joined(edge), "buffer_load_dwordx4"), 0);
  EXPECT_GT(countLinesContaining(joined(edge), "v_or_b32"), 0);
  EXPECT_EQ(countLinesContaining(joined(full), "buffer_load_dwordx4"),
            countLinesContaining(joined(edge), "buffer_load_dwordx4"));
}

TEST(SkinnyGEMM, TailEdgeArmSteersTheDeadLanes) {
  expectSteeredEdgeArm(emitted(config(1, 4, 16, 1, 1, 4, 1)));
}

// The decoded scales of the lane chunks that start at or after the end of the
// split are cleared by a byte mask, which the loop and a split without a tail
// do not have.
TEST(SkinnyGEMM, TailClearsTheScalesOfDeadLaneChunks) {
  for (int32_t upIters : {0, 2}) {
    std::string none = emitted(config(1, 4, 16, 1, 1, upIters, 0));
    std::string tail = emitted(config(1, 4, 16, 1, 1, upIters, 1));
    EXPECT_GT(countLinesContaining(tail, "v_ashrrev_i32"),
              countLinesContaining(none, "v_ashrrev_i32"));
  }
}

// A step with ring scales waits only for its own scale loads: the gather waits
// for column c behind the NC - 1 - c later scale loads, so the last wait is for
// everything, and no compute wait follows.
TEST(SkinnyGEMM, RingStepWaitsOnlyForItsScales) {
  constexpr int32_t columns = 2;
  std::string assembly = emitted(config(1, 1, 16, 1, 1, 0, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> expected;
  for (int32_t c = 0; c < columns; ++c)
    expected.push_back(vmcnt(columns - 1 - c));
  for (int32_t c = 0; c < columns; ++c)
    expected.push_back(lgkmcnt(2 * (columns - 1 - c)));
  EXPECT_EQ(waitsOf(stepCompute(assembly)), expected);
}

// With up-front scales a step issues no scale load, and the first convert of
// column c of every chunk waits behind the NC - 1 - c later columns of its own
// chunk, the only loads certain to be in flight.
TEST(SkinnyGEMM, UpFrontStepWaitsPerChunk) {
  constexpr int32_t columns = 5;
  constexpr int32_t chunks = 4;
  std::string assembly = emitted(config(1, 4, 16, 1, 2, 4, 1));
  SCOPED_TRACE("assembly:\n" + assembly);
  std::vector<std::string> compute = stepCompute(assembly);
  std::vector<std::string> expected;
  for (int32_t chunk = 0; chunk < chunks; ++chunk)
    for (int32_t c = 0; c < columns; ++c)
      expected.push_back(vmcnt(columns - 1 - c));
  EXPECT_EQ(waitsOf(compute), expected);
  for (int32_t i = 0; i + 1 < std::ssize(compute); ++i)
    if (compute[i].rfind("s_waitcnt", 0) == 0)
      EXPECT_EQ(compute[i + 1].rfind("v_cvt_scalef32_pk_bf16_fp8", 0), 0u);
}

// The wait before a zero arm of a step: the newest prologue load of the chunk's
// registers is followed by the later prologue slots, the rest of its own slot's
// loads are before it, and the step's NC scale loads were issued on every path.
TEST(SkinnyGEMM, ZeroedChunkWaitsForPrologueLoads) {
  constexpr int32_t columns = 2;
  constexpr int32_t scaleLoads = columns;
  constexpr int32_t loadsPerSlot = scaleLoads + 4 * columns;
  for (int32_t prefetch : {1, 2}) {
    SCOPED_TRACE("prefetch " + std::to_string(prefetch));
    std::vector<std::string> lines =
        allLines(emitted(config(1, 1, 16, 1, prefetch, 0, 1)));
    std::vector<std::string> waits;
    for (int32_t i = 0; i + 1 < std::ssize(lines); ++i)
      if (lines[i].rfind("label_tail_zero_", 0) == 0)
        waits.push_back(lines[i + 1]);
    std::vector<std::string> expected;
    for (int32_t chunk = 0; chunk < 4; ++chunk) {
      int32_t newest = scaleLoads + chunk * columns + columns - 1;
      expected.push_back(vmcnt(loadsPerSlot - 1 - newest +
                               (prefetch - 1) * loadsPerSlot + scaleLoads));
    }
    EXPECT_EQ(waits, expected);
  }
  // Without a ring there is no prologue load to wait for.
  std::vector<std::string> lines =
      allLines(emitted(config(1, 1, 16, 1, 0, 0, 1)));
  int32_t zeroArms = 0;
  for (int32_t i = 0; i + 1 < std::ssize(lines); ++i) {
    if (lines[i].rfind("label_tail_zero_", 0) != 0)
      continue;
    ++zeroArms;
    EXPECT_EQ(lines[i + 1].rfind("v_mov_b32", 0), 0u);
  }
  EXPECT_EQ(zeroArms, 4);
}

// The up-front entries of a split with partial steps are steered at k: every
// scale load ORs its voffset with the sign mask, which PARTIAL_STEPS = 0 does
// not need.
TEST(SkinnyGEMM, UpFrontEntriesSteerAtK) {
  constexpr int32_t entries = 3;
  constexpr int32_t columns = 2;
  for (int32_t partialSteps : {0, 1}) {
    SCOPED_TRACE("partialSteps " + std::to_string(partialSteps));
    std::vector<std::string> lines =
        allLines(emitted(config(1, 1, 16, 1, 2, entries, partialSteps)));
    int32_t loads = 0;
    for (int32_t i = 1; i < std::ssize(lines) && loads < entries * columns;
         ++i) {
      if (lines[i].rfind("buffer_load_dword ", 0) != 0)
        continue;
      ++loads;
      bool steered = lines[i - 1].rfind("v_or_b32", 0) == 0;
      EXPECT_EQ(steered, partialSteps != 0) << lines[i - 1];
    }
    EXPECT_EQ(loads, entries * columns);
  }
}

// Counts the v_nop that precede each DPP add of the reduction of a single
// accumulator, in order.
static std::vector<int32_t> nopsBeforeDpp(const std::string &assembly) {
  std::vector<int32_t> counts;
  std::istringstream stream(assembly);
  int32_t nops = 0;
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

// The reduction adds one DPP step per doubling of the cluster.
TEST(SkinnyGEMM, ReduceForEveryLanesK) {
  // Three nops cover the dot's write, two every DPP read of a VALU result.
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 4, 1, 1))),
            (std::vector<int32_t>{3, 2}));
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 8, 1, 1))),
            (std::vector<int32_t>{3, 2, 2}));
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 16, 1, 1))),
            (std::vector<int32_t>{3, 2, 2, 2}));
  // 64 lanes add the row broadcasts and read lane 63 after one nop; the move
  // of the lane's value takes two more.
  EXPECT_EQ(nopsBeforeDpp(emitted(config(1, 1, 64, 1, 1))),
            (std::vector<int32_t>{3, 2, 2, 2, 2, 2, 1}));

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

// Vector stores exist only when the runs are dword aligned.
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

// The 1-D launch form of base, with the tile order knobs.
static SkinnyGEMMConfig persistentConfig(SkinnyGEMMConfig base,
                                         int32_t group = 1,
                                         int32_t numXcds = 1) {
  base.grid3d = false;
  base.group = group;
  base.numXcds = numXcds;
  return base;
}

// The persistent kernel only enables the workgroup id x and reads only
// numWorkGroupsX, which is the number of blocks.
TEST(SkinnyGEMM, PersistentLaunchAbi) {
  std::string assembly = emitted(persistentConfig(config(1, 1, 16, 1, 1)));
  SCOPED_TRACE("assembly:\n" + assembly);
  EXPECT_NE(assembly.find(".amdhsa_system_sgpr_workgroup_id_x 1"),
            std::string::npos);
  EXPECT_EQ(assembly.find("workgroup_id_y"), std::string::npos);
  EXPECT_EQ(assembly.find("workgroup_id_z"), std::string::npos);
  EXPECT_NE(assembly.find(".kernarg_segment_size: 112"), std::string::npos);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 96"), 1);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 100"), 0);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 104"), 0);
  // The kernel loads alpha but not C, ldC or beta.
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 88"), 1);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 84"), 0);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 92"), 0);
  EXPECT_EQ(countLinesContaining(assembly, "s[0:1], 48"), 0);
}

static std::ptrdiff_t
indexOfLineStartingWith(const std::vector<std::string> &lines,
                        std::string_view prefix, std::ptrdiff_t from = 0) {
  for (std::ptrdiff_t i = from; i < std::ssize(lines); ++i)
    if (lines[i].rfind(prefix, 0) == 0)
      return i;
  return -1;
}

// The loop counter is the block id in s2: the guard skips the loop when the
// block has no tile, the body is emitted once, and the back edge adds the block
// count and tests again, right before the end of the program.
TEST(SkinnyGEMM, PersistentTileLoopStructure) {
  for (SkinnyGEMMConfig base :
       {config(1, 1, 16, 1, 1), config(2, 2, 16, 1, 2, 1),
        config(1, 1, 64, 4, 2, 2, 1)}) {
    std::string assembly = emitted(persistentConfig(base, 4, 8));
    SCOPED_TRACE("assembly:\n" + assembly);
    std::vector<std::string> lines = allLines(assembly);
    std::ptrdiff_t body = indexOfLineStartingWith(lines, "label_tile_loop_0:");
    std::ptrdiff_t end =
        indexOfLineStartingWith(lines, "label_tile_loop_end_1:");
    ASSERT_GE(body, 2);
    ASSERT_GT(end, body + 4);
    EXPECT_EQ(lines[body - 2].rfind("s_cmp_lt_i32 s2, ", 0), 0u);
    EXPECT_EQ(lines[body - 1], "s_cbranch_scc0 label_tile_loop_end_1");
    EXPECT_EQ(lines[end - 3].rfind("s_add_u32 s2, s2, ", 0), 0u);
    EXPECT_EQ(lines[end - 2].rfind("s_cmp_lt_i32 s2, ", 0), 0u);
    EXPECT_EQ(lines[end - 1], "s_cbranch_scc1 label_tile_loop_0");
    EXPECT_EQ(lines[end + 1], "s_endpgm");
    EXPECT_EQ(countLinesContaining(assembly, "s_endpgm"), 1);
  }
}

// Only the LDS-shared scales need the barrier, first in the body: the previous
// tile's reads must be done before this tile writes. The barrier waits for the
// loads and the LDS operations first (rocisa prints the two zero counts as
// s_waitcnt 0).
TEST(SkinnyGEMM, PersistentBarrierOnlyWhenShared) {
  std::string shared = emitted(persistentConfig(config(1, 1, 64, 4, 2, 2)));
  SCOPED_TRACE("assembly:\n" + shared);
  std::vector<std::string> lines = allLines(shared);
  std::ptrdiff_t body = indexOfLineStartingWith(lines, "label_tile_loop_0:");
  ASSERT_GE(body, 0);
  EXPECT_EQ(lines[body + 1], "s_waitcnt 0");
  EXPECT_EQ(lines[body + 2], "s_barrier");
  // The other barrier publishes the staged scales.
  EXPECT_EQ(countLinesContaining(shared, "s_barrier"), 2);

  EXPECT_EQ(
      emitted(persistentConfig(config(1, 1, 16, 1, 2, 2))).find("s_barrier"),
      std::string::npos);
  EXPECT_EQ(emitted(persistentConfig(config(1, 1, 16, 1, 1))).find("s_barrier"),
            std::string::npos);
  // The 3-D launch has no tile loop and so no per-tile barrier.
  EXPECT_EQ(
      countLinesContaining(emitted(config(1, 1, 64, 4, 2, 2)), "s_barrier"), 1);
}

// The lines of the tile loop body before its first buffer load: the barrier,
// the remap, the decode, the rotation and the bases.
static std::string tileDecode(const SkinnyGEMMConfig &config) {
  return joined(
      linesBetween(emitted(config), "label_tile_loop_0:", "buffer_load"));
}

// Every dynamic division is one v_rcp_iflag_f32. The plain order decodes kz and
// ty (2), keys the rotation on the block id outside the loop and has no remap.
// The grouped decode divides for kz, the band and tx; the reordered rotation
// divides by NI unless it is absent with one up-front entry. A power of two X
// remaps with shifts and a mask.
TEST(SkinnyGEMM, PersistentDecodeAndRotationShape) {
  SkinnyGEMMConfig ring = config(1, 1, 16, 1, 1);
  SkinnyGEMMConfig upFront = config(1, 1, 16, 1, 2, 2);
  SkinnyGEMMConfig oneEntry = config(1, 1, 16, 1, 2, 1);

  std::string plain = tileDecode(persistentConfig(ring));
  EXPECT_EQ(countLinesContaining(plain, "v_rcp_iflag_f32"), 2);
  EXPECT_EQ(countLinesContaining(plain, ", 7, s2"), 0);
  EXPECT_EQ(countLinesContaining(tileDecode(persistentConfig(upFront)),
                                 "v_rcp_iflag_f32"),
            2);

  std::string grouped = tileDecode(persistentConfig(ring, 4, 8));
  SCOPED_TRACE("body:\n" + grouped);
  EXPECT_EQ(countLinesContaining(grouped, "v_rcp_iflag_f32"), 4);
  EXPECT_EQ(countLinesContaining(grouped, ", s2, 3"), 1);
  EXPECT_EQ(countLinesContaining(grouped, ", 7, s2"), 1);
  EXPECT_EQ(countLinesContaining(tileDecode(persistentConfig(upFront, 4, 8)),
                                 "v_rcp_iflag_f32"),
            4);
  EXPECT_EQ(countLinesContaining(tileDecode(persistentConfig(oneEntry, 4, 8)),
                                 "v_rcp_iflag_f32"),
            3);
  // Only reordering by the XCDs: kz, ty and the slab rotation.
  EXPECT_EQ(countLinesContaining(tileDecode(persistentConfig(ring, 1, 8)),
                                 "v_rcp_iflag_f32"),
            3);
  // Only grouping: the same, without the remap.
  std::string groupOnly = tileDecode(persistentConfig(ring, 4, 1));
  EXPECT_EQ(countLinesContaining(groupOnly, "v_rcp_iflag_f32"), 4);
  EXPECT_EQ(countLinesContaining(groupOnly, ", 7, s2"), 0);

  // A divisor that is not a power of two uses the magic division, not a
  // dynamic one.
  std::string odd = tileDecode(persistentConfig(ring, 1, 3));
  EXPECT_EQ(countLinesContaining(odd, "v_rcp_iflag_f32"), 3);
  EXPECT_EQ(countLinesContaining(odd, "s_mul_hi_i32"), 1);
}

// The 3-D launch ignores GROUP and NUM_XCDS: the kernel
// is the one of group = 1 and numXcds = 1 under another name.
TEST(SkinnyGEMM, Grid3dIgnoresGroupAndXcds) {
  for (SkinnyGEMMConfig base :
       {config(1, 1, 16, 1, 1), config(2, 2, 16, 1, 2, 1, 1),
        config(1, 1, 64, 4, 2, 2)}) {
    SkinnyGEMMConfig reordered = base;
    reordered.group = 4;
    reordered.numXcds = 8;
    std::string plain = emitted(base);
    std::string ignored = emitted(reordered);
    for (std::string *text : {&plain, &ignored}) {
      const SkinnyGEMMConfig &owner = text == &plain ? base : reordered;
      std::string name = skinnyGEMMKernelName(owner);
      for (std::size_t at = text->find(name); at != std::string::npos;
           at = text->find(name, at))
        text->replace(at, name.size(), "gemm");
    }
    EXPECT_EQ(plain, ignored);
  }
}

// The 3-D launch groups the xcdRun = 32 / BN tiles that share scale lines on
// one XCD, after the K rotation read the physical id; only a BN that is a
// power of two below 32 regroups, and the persistent launch never does.
TEST(SkinnyGEMM, Grid3dGroupsScaleSharingTilesPerXcd) {
  // BN = 4, so G = 8 and the span is 64.
  std::string grouped = emitted(config(1, 1, 64, 4, 2, 2));
  SCOPED_TRACE("assembly:\n" + grouped);
  std::vector<std::string> lines = allLines(grouped);
  std::ptrdiff_t select = indexOfLineStartingWith(lines, "s_cselect_b32 s2, ");
  ASSERT_GE(select, 1);
  EXPECT_EQ(countLinesContaining(grouped, "s_cselect_b32 s2, "), 1);
  EXPECT_EQ(countLinesContaining(grouped, "s_cmp_lt_u32 s2, "), 1);
  EXPECT_EQ(lines[select - 1].rfind("s_cmp_lt_u32 s2, ", 0), 0u);
  EXPECT_EQ(lines[select - 2].rfind("s_and_b32 s", 0), 0u);
  EXPECT_NE(lines[select - 2].find(", -64, s"), std::string::npos);
  EXPECT_NE(lines[select - 4].find(", -64, s2"), std::string::npos);
  // The rotation key adds the physical id, so it comes before the remap.
  bool keyBeforeRemap = false;
  for (std::ptrdiff_t at = 0; at < select - 4; ++at)
    keyBeforeRemap |=
        lines[at].rfind("s_add_u32 ", 0) == 0 && lines[at].ends_with(", s2");
  EXPECT_TRUE(keyBeforeRemap);

  // BN = 32 and BN = 12 keep the physical id, as does the persistent launch.
  for (SkinnyGEMMConfig ungrouped :
       {config(1, 1, 4, 2, 1), config(1, 3, 16, 1, 1),
        persistentConfig(config(1, 1, 64, 4, 2, 2))})
    EXPECT_EQ(countLinesContaining(emitted(ungrouped), "s_cselect_b32 s2,"), 0);
}

// The near-register-limit row (BM = 6, TN = 3) still fits with
// the lane values and the tile loop state resident.
TEST(SkinnyGEMM, PersistentNearRegisterLimitConfigEmits) {
  SkinnyGEMMConfig wide = config(6, 3, 16, 1, 1);
  EXPECT_NO_THROW(emitted(persistentConfig(wide)));
  EXPECT_NO_THROW(emitted(persistentConfig(wide, 4, 8)));
}

// The beta variant keeps the kernarg layout and the segment size, so that one
// packer serves both variants; only the symbol changes.
TEST(SkinnyGEMM, BetaVariantSharesTheAbi) {
  SkinnyGEMMConfig beta = config(1, 1, 16, 2, 1);
  beta.useBeta = true;
  std::string assembly = emitted(beta);
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {".name: gemm_BM1_TN1_LANES_K16_WAVES2_UP_ITERS0_PARTIAL_STEPS0_PF1_"
       "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1",
       ".kernarg_segment_size: 112", ".group_segment_fixed_size: 0",
       ".max_flat_workgroup_size: 128"});
  expectLinesInOrder(assembly, {".name: D",
                                ".offset: 0",
                                ".name: ws",
                                ".offset: 40",
                                ".name: C",
                                ".offset: 48",
                                ".name: ldD",
                                ".offset: 56",
                                ".name: spk",
                                ".offset: 80",
                                ".name: ldC",
                                ".offset: 84",
                                ".name: alpha",
                                ".offset: 88",
                                ".name: beta",
                                ".offset: 92",
                                ".name: numWorkGroupsX",
                                ".offset: 96",
                                ".name: numWorkGroupsZ",
                                ".offset: 104"});
  EXPECT_EQ(assembly.find("hidden_"), std::string::npos);
}

// Returns the text of assembly from the first occurrence of from to the first
// later one of to.
static std::string textBetween(const std::string &assembly,
                               std::string_view from, std::string_view to) {
  std::size_t first = assembly.find(from);
  EXPECT_NE(first, std::string::npos) << from;
  std::size_t last = assembly.find(to, first);
  EXPECT_NE(last, std::string::npos) << to;
  if (first == std::string::npos || last == std::string::npos)
    return "";
  return assembly.substr(first, last - first);
}

// Returns the label that the first line starting with branch (a prefix of the
// label name included) jumps to.
static std::string labelOf(const std::string &assembly,
                           std::string_view branch) {
  std::size_t first = assembly.find(branch);
  EXPECT_NE(first, std::string::npos) << branch;
  if (first == std::string::npos)
    return "";
  std::size_t nameStart = assembly.find("label_", first);
  std::size_t nameEnd = assembly.find('\n', nameStart);
  return assembly.substr(nameStart, nameEnd - nameStart);
}

// Without useBeta nothing reads C or beta: no C load, no FMA and no beta test.
TEST(SkinnyGEMM, WithoutBetaNothingReadsC) {
  for (const SkinnyGEMMConfig &base :
       {config(1, 1, 16, 2, 1), config(2, 4, 16, 1, 1),
        config(3, 2, 16, 2, 1)}) {
    for (bool grid3d : {true, false}) {
      SkinnyGEMMConfig plain = base;
      plain.grid3d = grid3d;
      std::string assembly = emitted(plain);
      SCOPED_TRACE("assembly:\n" + assembly);
      EXPECT_EQ(countLinesContaining(assembly, "buffer_load_ushort"), 0);
      EXPECT_EQ(countLinesContaining(assembly, "v_fmac_f32"), 0);
      EXPECT_EQ(countLinesContaining(assembly, "2147483647"), 0);
    }
  }
}

// With useBeta the loads of C sit behind the beta and single-split guards, are
// issued before the first ring load, and the FMA waits for them behind the beta
// guard.
TEST(SkinnyGEMM, BetaVariantLoadsCBehindTheGuard) {
  for (bool grid3d : {true, false}) {
    SkinnyGEMMConfig beta = config(1, 1, 16, 2, 1);
    beta.useBeta = true;
    beta.grid3d = grid3d;
    std::string assembly = emitted(beta);
    SCOPED_TRACE("assembly:\n" + assembly);
    // The beta test masks the sign bit, so -0.0 counts as zero.
    expectLinesInOrder(assembly,
                       {"s_cmp_le_u32", "s_cbranch_scc0", "2147483647",
                        "s_cmp_eq_u32", "s_cbranch_scc1", "buffer_load_ushort",
                        "label_c_prefetch_skip", "buffer_load_dword"});
    std::string guarded = textBetween(
        assembly, "s_cbranch_scc1 label_c_prefetch_skip",
        labelOf(assembly, "s_cbranch_scc1 label_c_prefetch_skip") + ":");
    EXPECT_EQ(countLinesContaining(guarded, "buffer_load_ushort"), 1);
    EXPECT_EQ(countLinesContaining(assembly, "buffer_load_ushort"), 1);
    // The first load after the guard is a ring load, so C was issued first.
    std::size_t cLoad = assembly.find("buffer_load_ushort");
    std::size_t ringLoad = assembly.find("buffer_load_dwordx4");
    EXPECT_LT(cLoad, ringLoad);

    std::string consume =
        textBetween(assembly, "s_cbranch_scc1 label_no_beta", "v_fmac_f32");
    EXPECT_EQ(countLinesContaining(consume, "s_waitcnt vmcnt(0)"), 1);
    EXPECT_EQ(countLinesContaining(consume, "buffer_load"), 0);
    EXPECT_EQ(countLinesContaining(assembly, "v_fmac_f32"), 1);
  }
}

// A full tile loads C with the widest vector loads, as the store does, and an
// edge tile or an odd run per element; both orientations are emitted, and every
// path adds the beta * C of each tile element.
TEST(SkinnyGEMM, BetaVariantLoadsCInRuns) {
  SkinnyGEMMConfig even = config(2, 4, 16, 1, 1);
  even.useBeta = true;
  std::string assembly = emitted(even);
  SCOPED_TRACE("assembly:\n" + assembly);
  // Runs of 2 (one dword) and 4 (dwordx2) elements, and the masked fallback.
  EXPECT_NE(assembly.find("buffer_load_dwordx2"), std::string::npos);
  EXPECT_NE(assembly.find("buffer_load_dword "), std::string::npos);
  EXPECT_NE(assembly.find("buffer_load_ushort"), std::string::npos);
  // One FMA per tile element (BM * TN = 8) in each of the vector and the masked
  // path of each of the two orientations.
  EXPECT_EQ(countLinesContaining(assembly, "v_fmac_f32"), 2 * 2 * 8);
  // The high half of a pair is unpacked with a mask, the low half with a shift.
  EXPECT_NE(assembly.find("v_and_b32"), std::string::npos);

  SkinnyGEMMConfig wide = config(8, 2, 16, 1, 0);
  wide.useBeta = true;
  EXPECT_NE(emitted(wide).find("buffer_load_dwordx4"), std::string::npos);

  // BM = 3 and TN = 3 have no even bf16 run: per-element loads and FMAs only,
  // for each of the two orientations that the f32 workspace stores need.
  SkinnyGEMMConfig odd = config(3, 3, 16, 1, 1);
  odd.useBeta = true;
  std::string oddAssembly = emitted(odd);
  EXPECT_EQ(countLinesContaining(oddAssembly, "buffer_load_ushort"), 2 * 9);
  EXPECT_EQ(countLinesContaining(oddAssembly, "v_fmac_f32"), 2 * 9);
  EXPECT_EQ(countLinesContaining(oddAssembly, "label_c_vector"), 0);
  EXPECT_EQ(countLinesContaining(oddAssembly, "label_c_masked"), 0);
}

static bool isLabelOrComment(const std::string &line) {
  return line.empty() || line.back() == ':' || line[0] == ';' ||
         line.rfind("//", 0) == 0 || line.rfind("/*", 0) == 0;
}

// Returns the mnemonics from the kernel symbol to s_endpgm.
static std::vector<std::string> bodyMnemonics(const std::string &assembly) {
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

// The schedule, with the alpha scaling after the sum and the beta
// path (load C, widen, fmac) behind the uniform beta != 0 branch.
TEST(SplitKReduce, EmitsTheExpectedSchedule) {
  std::string assembly = emitSplitKReduce(Target::get("gfx950"));
  SCOPED_TRACE("assembly:\n" + assembly);
  // clang-format off
  std::vector<std::string> expected = {
      "s_load_dword",       "s_load_dword",   "s_load_dword",
      "s_load_dword",       "s_load_dword",   "s_load_dword",
      "s_load_dword",       "s_load_dwordx2", "s_load_dwordx2",
      "s_load_dwordx2",     "s_lshl_b32",     "v_add_u32",
      "s_waitcnt",          "v_cmp_gt_u32",   "s_sub_u32",
      "v_min_i32",          "s_lshl_b32",     "s_mul_i32",
      "s_mul_i32",          "s_add_u32",      "s_mul_i32",
      "s_lshl_b32",         "s_add_u32",      "s_add_u32",
      "s_mul_i32",          "s_lshl_b32",     "s_add_u32",
      "s_mov_b32",          "s_mov_b32",      "s_mov_b32",
      "v_lshlrev_b32",
      "s_mul_i32",          "v_add_u32",      "s_mov_b32",
      "v_mov_b32",          "s_mul_i32",      "v_add_u32",
      "buffer_load_dword",  "s_waitcnt",      "v_add_f32",
      "s_add_u32",          "s_cmp_lt_i32",   "s_cbranch_scc1",
      "v_mul_f32",          "s_and_saveexec_b64", "s_cbranch_vccz",
      "s_and_b32",          "s_cmp_eq_u32",   "s_cbranch_scc1",
      "v_lshlrev_b32",      "s_lshl_b32",     "v_add_u32",
      "s_mul_i32",          "v_add_u32",
      "buffer_load_ushort", "s_waitcnt",      "v_lshlrev_b32",
      "v_fmac_f32",         "v_lshrrev_b32",  "v_lshlrev_b32",
      "s_lshl_b32",         "v_add_u32",      "s_mul_i32",
      "v_add_u32",          "buffer_store_short",
      "s_mov_b64",          "s_endpgm"};
  // clang-format on
  EXPECT_EQ(bodyMnemonics(assembly), expected);
  EXPECT_EQ(countLinesContaining(assembly, " offen"), 3);

  expectLinesInOrder(assembly, {"s[0:1], 36",
                                "s[0:1], 32",
                                "s[0:1], 28",
                                "s[0:1], 24",
                                "s[0:1], 40",
                                "s[0:1], 44",
                                "s[0:1], 48",
                                "s[0:1], 8",
                                "s[0:1], 0",
                                "s[0:1], 16",
                                "s_waitcnt lgkmcnt(0)",
                                "v_cmp_gt_u32 vcc",
                                "buffer_load_dword",
                                "s_waitcnt vmcnt(0)",
                                "v_mul_f32",
                                "s_and_saveexec_b64",
                                "s_cbranch_vccz",
                                "s_cmp_eq_u32",
                                "buffer_load_ushort",
                                "s_waitcnt vmcnt(0)",
                                "v_fmac_f32",
                                "v_lshrrev_b32",
                                "buffer_store_short",
                                "s_mov_b64 exec",
                                "s_endpgm"});
}

// The reduce kernel declares its arguments and no hidden ones.
TEST(SplitKReduce, DeclaresTheKernargLayout) {
  std::string assembly = emitSplitKReduce(Target::get("gfx950"));
  SCOPED_TRACE("assembly:\n" + assembly);
  EXPECT_NE(assembly.find(".amdhsa_system_sgpr_workgroup_id_y 1"),
            std::string::npos);
  EXPECT_EQ(assembly.find("workgroup_id_z 1"), std::string::npos);
  expectLinesInOrder(assembly,
                     {".name: splitk_reduce", ".kernarg_segment_size: 56",
                      ".max_flat_workgroup_size: 256"});
  expectLinesInOrder(
      assembly, {".name: D",      ".offset: 0",  ".name: ws",   ".offset: 8",
                 ".name: C",      ".offset: 16", ".name: ldD",  ".offset: 24",
                 ".name: m",      ".offset: 28", ".name: n",    ".offset: 32",
                 ".name: splits", ".offset: 36", ".name: ldC",  ".offset: 40",
                 ".name: alpha",  ".offset: 44", ".name: beta", ".offset: 48"});
  EXPECT_EQ(assembly.find("hidden_"), std::string::npos);
}

// The kernargs are explicit, so every code object version emits the same
// kernel symbol and the same 112-byte segment; only the directives and the
// metadata version differ.
TEST(SkinnyGEMM, KernargsAreTheSameAtEveryCodeObjectVersion) {
  struct VersionCase {
    CodeObjectVersion version;
    std::string directive;
    std::string metadataVersion;
  };
  const std::vector<VersionCase> cases = {
      {CodeObjectVersion::v4, ".amdhsa_code_object_version 4",
       "amdhsa.version: [1, 1]"},
      {CodeObjectVersion::v5, ".amdhsa_code_object_version 5",
       "amdhsa.version: [1, 2]"},
      {CodeObjectVersion::v6, ".amdhsa_code_object_version 6",
       "amdhsa.version: [1, 2]"},
  };
  SkinnyGEMMConfig grid3d = config(1, 1, 16, 2, 1);
  for (const SkinnyGEMMConfig &gemm :
       {grid3d, persistentConfig(config(1, 1, 16, 1, 1))}) {
    for (const VersionCase &versionCase : cases) {
      KernelAbi abi;
      abi.codeObjectVersion = versionCase.version;
      std::string assembly = emitSkinnyGEMM(Target::get("gfx950"), gemm, abi);
      SCOPED_TRACE("grid3d " + std::to_string(gemm.grid3d) + " directive " +
                   versionCase.directive + "\nassembly:\n" + assembly);
      EXPECT_NE(assembly.find(".name: " + skinnyGEMMKernelName(gemm)),
                std::string::npos);
      EXPECT_NE(assembly.find(".kernarg_segment_size: 112"), std::string::npos);
      EXPECT_NE(assembly.find(versionCase.directive), std::string::npos);
      EXPECT_NE(assembly.find(versionCase.metadataVersion), std::string::npos);
      EXPECT_NE(assembly.find(".name: numWorkGroupsX"), std::string::npos);
      EXPECT_EQ(assembly.find("hidden_"), std::string::npos);
    }
  }
}

// The reduce kernel needs no counts, so it also has the same arguments at
// every code object version.
TEST(SplitKReduce, KernargsAreTheSameAtEveryCodeObjectVersion) {
  for (CodeObjectVersion version :
       {CodeObjectVersion::v4, CodeObjectVersion::v5, CodeObjectVersion::v6}) {
    KernelAbi abi;
    abi.codeObjectVersion = version;
    std::string assembly = emitSplitKReduce(Target::get("gfx950"), abi);
    SCOPED_TRACE("assembly:\n" + assembly);
    EXPECT_NE(assembly.find(".kernarg_segment_size: 56"), std::string::npos);
    EXPECT_EQ(assembly.find("hidden_"), std::string::npos);
  }
}

static constexpr const char *preloadDirective =
    ".amdhsa_user_sgpr_kernarg_preload_length";

static KernelAbi preloadAbi(int32_t maxPreloadDwords) {
  KernelAbi abi;
  abi.maxPreloadDwords = maxPreloadDwords;
  return abi;
}

// The GEMM preloads its pointers on request, with and without beta, and the
// default emits no preload directive.
TEST(SkinnyGEMM, MaxPreloadDwordsPreloadsThePointers) {
  for (bool useBeta : {false, true}) {
    SkinnyGEMMConfig gemm = config(1, 1, 16, 2, 1);
    gemm.useBeta = useBeta;
    std::string plain = emitted(gemm);
    std::string preloaded =
        emitSkinnyGEMM(Target::get("gfx950"), gemm, preloadAbi(14));
    SCOPED_TRACE("useBeta " + std::to_string(useBeta));
    EXPECT_EQ(plain.find(preloadDirective), std::string::npos);
    EXPECT_NE(preloaded.find(preloadDirective), std::string::npos);
    EXPECT_NE(plain, preloaded);
  }
}

TEST(SplitKReduce, MaxPreloadDwordsPreloadsTheArguments) {
  std::string plain = emitSplitKReduce(Target::get("gfx950"));
  std::string preloaded =
      emitSplitKReduce(Target::get("gfx950"), preloadAbi(14));
  EXPECT_EQ(plain.find(preloadDirective), std::string::npos);
  EXPECT_NE(preloaded.find(preloadDirective), std::string::npos);
  EXPECT_NE(plain, preloaded);
}

// The scaled-MFMA path.

static SkinnyGEMMConfig mfmaConfig(int32_t bm, int32_t tn, int32_t waves,
                                   int32_t prefetch, bool grid3d = true,
                                   bool useBeta = false) {
  SkinnyGEMMConfig result;
  result.mfma = true;
  result.bm = bm;
  result.tn = tn;
  result.waves = waves;
  result.prefetch = prefetch;
  result.grid3d = grid3d;
  result.useBeta = useBeta;
  return result;
}

// The suffix is the only difference to the name without mfma.
TEST(SkinnyGEMMMfma, KernelNameAppendsTheSuffix) {
  for (SkinnyGEMMConfig mfma :
       {mfmaConfig(16, 1, 4, 2), mfmaConfig(32, 3, 1, 1),
        mfmaConfig(32, 2, 2, 3, false, true)}) {
    SkinnyGEMMConfig plain = mfma;
    plain.mfma = false;
    plain.lanesK = 64;
    EXPECT_EQ(skinnyGEMMKernelName(mfma),
              skinnyGEMMKernelName(plain) + "_MFMA1");
  }
  EXPECT_EQ(skinnyGEMMKernelName(mfmaConfig(32, 1, 4, 2)),
            "gemm_BM32_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_"
            "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0_MFMA1");
}

static std::vector<SkinnyGEMMConfig> invalidMfmaConfigs() {
  std::vector<SkinnyGEMMConfig> configs(5, mfmaConfig(16, 1, 2, 1));
  configs[0].bm = 8;
  configs[1].bm = 48;
  configs[2].lanesK = 16;
  configs[3].upIters = 1;
  configs[4].prefetch = 0;
  return configs;
}

TEST(SkinnyGEMMMfma, RejectsInvalidKnobs) {
  Target target = Target::get("gfx950");
  for (const SkinnyGEMMConfig &invalid : invalidMfmaConfigs()) {
    EXPECT_THROW(skinnyGEMMKernelName(invalid), std::invalid_argument);
    EXPECT_THROW(emitSkinnyGEMM(target, invalid), std::invalid_argument);
  }
}

static SkinnyGEMMConfig partialMfmaConfig(SkinnyGEMMConfig gemm) {
  gemm.partialSteps = 1;
  return gemm;
}

TEST(SkinnyGEMMMfma, AcceptsPartialSteps) {
  SkinnyGEMMConfig partial = partialMfmaConfig(mfmaConfig(16, 1, 2, 1));
  EXPECT_NO_THROW(emitSkinnyGEMM(Target::get("gfx950"), partial));
  EXPECT_EQ(skinnyGEMMKernelName(partial),
            "gemm_BM16_TN1_LANES_K64_WAVES2_UP_ITERS0_PARTIAL_STEPS1_PF1_"
            "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0_MFMA1");
}

// The reduction slices are addressed with 16-bit offsets: 3 * 22 * 1024 bytes
// are over 64 KiB but under the 160 KiB of the target.
TEST(SkinnyGEMMMfma, RejectsConfigsOverTheLdsAndRegisterBudgets) {
  EXPECT_NE(rejectionMessage(mfmaConfig(16, 22, 4, 1))
                .find("config exceeds the LDS budget: 67584 bytes of 65536"),
            std::string::npos);
  EXPECT_NE(rejectionMessage(mfmaConfig(32, 4, 1, 4))
                .find("config exceeds the register budget: "),
            std::string::npos);
}

// The register budget rejects the largest rings, which the matrix skips.
static bool fitsTheRegisters(const SkinnyGEMMConfig &gemm) {
  try {
    emitted(gemm);
  } catch (const std::invalid_argument &error) {
    return std::string(error.what()).find("register budget") ==
           std::string::npos;
  }
  return true;
}

// Calls check with the assembly of every representative mfma configuration
// that fits the registers.
template <typename CheckFn>
static void forEachMfmaKernel(CheckFn check) {
  for (int32_t bm : {16, 32})
    for (int32_t tn : {1, 2, 3})
      for (int32_t waves : {1, 2, 4})
        for (int32_t prefetch : {1, 2, 3})
          for (bool grid3d : {true, false})
            for (bool useBeta : {false, true}) {
              SkinnyGEMMConfig gemm =
                  mfmaConfig(bm, tn, waves, prefetch, grid3d, useBeta);
              if (!fitsTheRegisters(gemm))
                continue;
              SCOPED_TRACE(skinnyGEMMKernelName(gemm));
              check(gemm, emitted(gemm));
            }
}

// The counts that the scale plan and the ring derive from the knobs: the scale
// dwords of a slot are shared by two tiles ("paired") for the wide operand
// with an even tn and the skinny operand with bm = 32.
struct MfmaPlan {
  explicit MfmaPlan(const SkinnyGEMMConfig &gemm)
      : wideTiles(gemm.tn), skinnyTiles(gemm.bm / 16),
        widePaired(gemm.tn % 2 == 0), skinnyPaired(gemm.bm == 32) {
    int32_t scales = (widePaired ? wideTiles / 2 : wideTiles) + 1;
    stepLoads = 2 * (wideTiles + skinnyTiles);
    slotLoads = scales + 2 * stepLoads;
  }
  int32_t wideTiles, skinnyTiles;
  bool widePaired, skinnyPaired;
  int32_t stepLoads, slotLoads;
};

static std::vector<std::string> kLoopBody(const std::string &assembly) {
  return linesBetween(assembly, "label_kloop_", "s_cbranch_scc1 label_kloop_");
}

// The groups of loads issued after the slot of iteration i of the arm of
// remaining iterations: the slots behind it and the refills of the earlier
// iterations.
static int32_t armGroupsAfter(int32_t prefetch, int32_t remaining, int32_t i) {
  return prefetch - 1 + std::clamp(remaining - prefetch, 0, i) - i;
}

// The straight-line remainder arms in order, the arm of 1 remaining iteration
// first: the lines between the branch that skips an arm and its end label.
static std::vector<std::vector<std::string>>
remainderArms(const std::string &assembly) {
  std::vector<std::vector<std::string>> arms;
  std::string endLabel;
  for (const std::string &line : allLines(assembly)) {
    if (endLabel.empty()) {
      if (line.rfind("s_cbranch_scc0 label_arm_end_", 0) != 0)
        continue;
      endLabel = line.substr(std::string("s_cbranch_scc0 ").size()) + ":";
      arms.emplace_back();
      continue;
    }
    if (line == endLabel) {
      endLabel.clear();
      continue;
    }
    arms.back().push_back(line);
  }
  return arms;
}

// The scale bytes that an MFMA selects for A and for B: op_sel and op_sel_hi
// are the low and the high bit.
struct ScaleBytes {
  int32_t a, b;
};

// The two flags of a modifier such as " op_sel:[x,y,0]".
static std::pair<int32_t, int32_t> modifierBits(const std::string &mfma,
                                                std::string_view key) {
  int32_t at = static_cast<int32_t>(mfma.find(key) + key.size());
  return {mfma[at] - '0', mfma[at + 2] - '0'};
}

static ScaleBytes scaleBytes(const std::string &mfma) {
  std::pair<int32_t, int32_t> low = modifierBits(mfma, " op_sel:[");
  std::pair<int32_t, int32_t> high = modifierBits(mfma, " op_sel_hi:[");
  return {low.first + 2 * high.first, low.second + 2 * high.second};
}

// The body holds the 2 steps of every tile pair of each ring slot, the arms
// those of 1 to 2 * PF - 1 iterations, and the VALU path's instructions are
// gone.
TEST(SkinnyGEMMMfma, LoopHasTheMfmaOfEverySlotAndNoDot) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    std::vector<std::string> body = kLoopBody(text);
    EXPECT_EQ(
        countLinesContaining(joined(body), "v_mfma_scale_f32_16x16x128_f8f6f4"),
        2 * gemm.tn * (gemm.bm / 16) * gemm.prefetch);
    int32_t arms = gemm.prefetch * (2 * gemm.prefetch - 1);
    EXPECT_EQ(countLinesContaining(text, "v_mfma_scale_f32_16x16x128_f8f6f4"),
              2 * gemm.tn * (gemm.bm / 16) * (gemm.prefetch + arms));
    EXPECT_EQ(text.find("v_dot2_f32_bf16"), std::string::npos);
    EXPECT_EQ(text.find("v_cvt_scalef32_pk_bf16_fp8"), std::string::npos);
  });
}

static std::set<int32_t> expectedBytes(bool paired) {
  if (paired)
    return {0, 1, 2, 3};
  return {0, 2};
}

// Paired dwords alternate the two bytes of a step with the tile, the others
// always take the first byte of the step.
TEST(SkinnyGEMMMfma, OpSelPicksTheScaleBytes) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    std::set<int32_t> bytesA;
    std::set<int32_t> bytesB;
    for (const std::string &line : kLoopBody(text)) {
      if (line.rfind("v_mfma_scale", 0) != 0)
        continue;
      ScaleBytes bytes = scaleBytes(line);
      bytesA.insert(bytes.a);
      bytesB.insert(bytes.b);
    }
    EXPECT_EQ(bytesA, expectedBytes(plan.widePaired));
    EXPECT_EQ(bytesB, expectedBytes(plan.skinnyPaired));
  });
}

// Single dwords are shifted to the tile's half once per slot, paired ones are
// not.
TEST(SkinnyGEMMMfma, SingleScalesAreShifted) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t shifts =
        (plan.widePaired ? 0 : plan.wideTiles) + (plan.skinnyPaired ? 0 : 1);
    EXPECT_EQ(countLinesContaining(joined(kLoopBody(text)), "v_lshrrev_b32"),
              shifts * gemm.prefetch);
  });
}

// Slot p waits for its step-0 loads while the other slots are in flight, then
// for its step-1 loads, capped at the counter's range.
TEST(SkinnyGEMMMfma, LoopWaitsForTheLoadsOfEachStep) {
  int32_t maxCount = Target::get("gfx950").maxVmcnt();
  forEachMfmaKernel([&](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t behind = (gemm.prefetch - 1) * plan.slotLoads;
    std::vector<std::string> expected;
    for (int32_t slot = 0; slot < gemm.prefetch; ++slot) {
      expected.push_back(vmcnt(std::min(behind + plan.stepLoads, maxCount)));
      expected.push_back(vmcnt(std::min(behind, maxCount)));
    }
    EXPECT_EQ(waitsOf(kLoopBody(text)), expected);
  });
}

// The arm of R remaining iterations waits like the loop for the loads of each
// step, with the groups that remain behind the slot of the iteration, and
// refills only the iterations that exist.
TEST(SkinnyGEMMMfma, RemainderArmsWaitForTheLoadsBehindEachSlot) {
  int32_t maxCount = Target::get("gfx950").maxVmcnt();
  forEachMfmaKernel([&](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    std::vector<std::vector<std::string>> arms = remainderArms(text);
    ASSERT_EQ(static_cast<int32_t>(arms.size()), 2 * gemm.prefetch - 1);
    for (int32_t remaining = 1; remaining < 2 * gemm.prefetch; ++remaining) {
      std::vector<std::string> expected;
      int32_t loads = 0;
      for (int32_t i = 0; i < remaining; ++i) {
        int32_t behind =
            armGroupsAfter(gemm.prefetch, remaining, i) * plan.slotLoads;
        expected.push_back(vmcnt(std::min(behind + plan.stepLoads, maxCount)));
        expected.push_back(vmcnt(std::min(behind, maxCount)));
        loads += i + gemm.prefetch < remaining;
      }
      const std::vector<std::string> &arm = arms[remaining - 1];
      EXPECT_EQ(waitsOf(arm), expected);
      EXPECT_EQ(countLinesContaining(joined(arm), "buffer_load_dword"),
                loads * plan.slotLoads);
      EXPECT_EQ(countLinesContaining(joined(arm),
                                     "v_mfma_scale_f32_16x16x128_f8f6f4"),
                2 * gemm.tn * (gemm.bm / 16) * remaining);
    }
  });
}

// Only several waves reduce through LDS, which the descriptor sizes as
// (waves - 1) * tn * (bm / 16) * 1024 bytes.
TEST(SkinnyGEMMMfma, ReducesThroughLdsOnlyWithSeveralWaves) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    int32_t tiles = gemm.tn * (gemm.bm / 16);
    int32_t lds = (gemm.waves - 1) * tiles * 1024;
    EXPECT_NE(text.find(".group_segment_fixed_size: " + std::to_string(lds)),
              std::string::npos);
    bool several = gemm.waves > 1;
    EXPECT_EQ(countLinesContaining(text, "ds_write_b128"), several ? tiles : 0);
    EXPECT_EQ(countLinesContaining(text, "ds_read_b128"),
              several ? tiles * (gemm.waves - 1) : 0);
    // The 3-D launch has the writers' and the leader's barrier, the persistent
    // one also the barrier at the top of the tile loop.
    EXPECT_EQ(countLinesContaining(text, "s_barrier"),
              !several ? 0 : (gemm.grid3d ? 2 : 3));
  });
}

TEST(SkinnyGEMMMfma, PersistentLoopStartsWithTheBarrier) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    if (gemm.grid3d)
      return;
    std::vector<std::string> lines = allLines(text);
    std::ptrdiff_t body = indexOfLineStartingWith(lines, "label_tile_loop_0:");
    ASSERT_GE(body, 0);
    EXPECT_EQ(countLinesContaining(text, "label_tile_loop_end_1:"), 1);
    EXPECT_EQ(countLinesContaining(text, "s_endpgm"), 1);
    if (gemm.waves == 1)
      return;
    EXPECT_EQ(lines[body + 1], lgkmcnt(0));
    EXPECT_EQ(lines[body + 2], "s_barrier");
  });
}

// The C elements are prefetched, four per tile, only behind the guards of a
// single split, wave 0 and a nonzero beta. The per-element loads exist in both
// orientations (the skinny-contiguous one has no vector loads, the other one
// falls back to them for a tile that is not inside the output), the vector
// loads are one dwordx2 per run of 4 rows.
TEST(SkinnyGEMMMfma, BetaPrefetchesCBehindTheGuards) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    if (!gemm.useBeta) {
      EXPECT_EQ(text.find("buffer_load_ushort"), std::string::npos);
      EXPECT_EQ(text.find("label_c_prefetch_skip"), std::string::npos);
      return;
    }
    std::vector<std::string> lines = allLines(text);
    std::ptrdiff_t first =
        indexOfLineStartingWith(lines, "s_cbranch_scc0 label_c_prefetch_skip");
    std::ptrdiff_t skip =
        indexOfLineStartingWith(lines, "label_c_prefetch_skip_");
    ASSERT_GE(first, 0);
    ASSERT_GT(skip, first);
    std::vector<std::string> guarded(lines.begin() + first,
                                     lines.begin() + skip);
    int32_t loads = 4 * plan.wideTiles * plan.skinnyTiles;
    EXPECT_EQ(countLinesContaining(joined(guarded), "buffer_load_ushort"),
              2 * loads);
    EXPECT_EQ(countLinesContaining(text, "buffer_load_ushort"), 2 * loads);
    EXPECT_EQ(countLinesContaining(joined(guarded), "buffer_load_dwordx2"),
              plan.wideTiles * plan.skinnyTiles);
    EXPECT_EQ(countLinesContaining(text, "buffer_load_dwordx2"),
              plan.wideTiles * plan.skinnyTiles);
    EXPECT_EQ(countLinesContaining(text, "s_cbranch_scc0 label_c_prefetch"), 2);
    EXPECT_EQ(countLinesContaining(text, "s_cbranch_scc1 label_c_prefetch"), 1);
  });
}

// The output is stored with one dwordx2 (bf16 D) or dwordx4 (f32 workspace) per
// run of the 4 rows that a lane holds when the wide dimension is contiguous;
// the per-element stores remain for the skinny-contiguous orientation and for
// the tiles that are not inside the output.
TEST(SkinnyGEMMMfma, StoresRunsOfRowsWithVectorStores) {
  forEachMfmaKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t runs = plan.wideTiles * plan.skinnyTiles;
    EXPECT_EQ(countLinesContaining(text, "buffer_store_dwordx2"), runs);
    EXPECT_EQ(countLinesContaining(text, "buffer_store_dwordx4"), runs);
    EXPECT_EQ(countLinesContaining(text, "buffer_store_short"), 2 * 4 * runs);
    EXPECT_EQ(countLinesContaining(text, "buffer_store_dword "), 2 * 4 * runs);
    EXPECT_NE(text.find("label_vector_"), std::string::npos);
    EXPECT_NE(text.find("label_orientation_b_"), std::string::npos);
  });
}

// The kernargs of the GEMM, and the preloading of the pointers.
TEST(SkinnyGEMMMfma, KeepsTheAbi) {
  for (SkinnyGEMMConfig gemm :
       {mfmaConfig(32, 1, 4, 2), mfmaConfig(16, 2, 1, 1, false, true)}) {
    std::string text = emitted(gemm);
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    EXPECT_NE(text.find(".name: " + skinnyGEMMKernelName(gemm)),
              std::string::npos);
    EXPECT_NE(text.find(".kernarg_segment_size: 112"), std::string::npos);
    EXPECT_NE(text.find(".name: numWorkGroupsX"), std::string::npos);
    EXPECT_EQ(text.find("hidden_"), std::string::npos);
    std::string preloaded =
        emitSkinnyGEMM(Target::get("gfx950"), gemm, preloadAbi(14));
    EXPECT_EQ(text.find(preloadDirective), std::string::npos);
    EXPECT_NE(preloaded.find(preloadDirective), std::string::npos);
  }
}

TEST(SkinnyGEMMMfma, EveryKernelAssembles) {
  Target target = Target::get("gfx950");
  forEachMfmaKernel([&](const SkinnyGEMMConfig &, const std::string &text) {
    EXPECT_FALSE(assembleToHSACO(text, target.isaName()).empty());
  });
}

// The partial tail iteration exists only with partialSteps = 1.
TEST(SkinnyGEMMMfma, TailIterationExistsOnlyWithPartialSteps) {
  for (const SkinnyGEMMConfig &gemm :
       {mfmaConfig(16, 1, 1, 1), mfmaConfig(16, 2, 2, 3, false, true),
        mfmaConfig(32, 3, 2, 2), mfmaConfig(32, 2, 1, 3, false)}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    EXPECT_EQ(emitted(gemm).find("label_tail_begin"), std::string::npos);
    EXPECT_NE(emitted(partialMfmaConfig(gemm)).find("label_tail_begin"),
              std::string::npos);
  }
}

// The tail runs both MFMA steps of slot 0 for every tile.
TEST(SkinnyGEMMMfma, TailAddsTheMfmaOfTwoSteps) {
  for (const SkinnyGEMMConfig &gemm :
       {mfmaConfig(16, 1, 1, 1), mfmaConfig(16, 2, 2, 3, false, true),
        mfmaConfig(32, 3, 2, 2), mfmaConfig(32, 2, 1, 3, false),
        mfmaConfig(16, 3, 4, 2, true, true)}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    int32_t tiles = gemm.tn * (gemm.bm / 16);
    const std::string mfma = "v_mfma_scale_f32_16x16x128_f8f6f4";
    EXPECT_EQ(countLinesContaining(emitted(partialMfmaConfig(gemm)), mfma),
              countLinesContaining(emitted(gemm), mfma) + 2 * tiles);
  }
}

TEST(SkinnyGEMMMfma, TailEdgeArmSteersTheDeadLanes) {
  expectSteeredEdgeArm(emitted(partialMfmaConfig(mfmaConfig(16, 2, 1, 2))));
}

TEST(SkinnyGEMMMfma, PartialKernelsAssemble) {
  Target target = Target::get("gfx950");
  for (const SkinnyGEMMConfig &gemm :
       {mfmaConfig(16, 1, 1, 1), mfmaConfig(16, 2, 2, 3, false, true),
        mfmaConfig(32, 3, 2, 2), mfmaConfig(32, 2, 4, 3)}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    EXPECT_FALSE(
        assembleToHSACO(emitted(partialMfmaConfig(gemm)), target.isaName())
            .empty());
  }
}

// The scale override of a partial tail iteration (the 2^0 byte 0x7F in every
// dword, printed in decimal) and the clamped byte masks of the tail data.
static void expectTailMasks(const std::string &text) {
  EXPECT_NE(text.find("2139062143"), std::string::npos);
  EXPECT_NE(text.find(", -1\n"), std::string::npos);
  EXPECT_NE(text.find("v_lshrrev_b32"), std::string::npos);
  for (std::string_view label :
       {"label_tail_edge_", "label_tail_zero_", "label_tail_chunk_end_",
        "label_tail_mask_skip_"})
    EXPECT_NE(text.find(label), std::string::npos) << label;
}

TEST(SkinnyGEMMMfma, PartialTailMasksTheDataAndOverridesTheScales) {
  for (const SkinnyGEMMConfig &gemm :
       {mfmaConfig(16, 1, 1, 1), mfmaConfig(16, 2, 2, 3, false, true),
        mfmaConfig(32, 3, 2, 2)}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    expectTailMasks(emitted(partialMfmaConfig(gemm)));
    EXPECT_EQ(emitted(gemm).find("2139062143"), std::string::npos);
  }
}

// The staged variant (lds).

static SkinnyGEMMConfig ldsConfig(int32_t bm, int32_t tn, int32_t waves,
                                  int32_t prefetch, bool grid3d = true,
                                  bool useBeta = false) {
  SkinnyGEMMConfig result =
      mfmaConfig(bm, tn, waves, prefetch, grid3d, useBeta);
  result.lds = true;
  return result;
}

// The ring of a wave: prefetch slots of 4 KiB for every tile.
static int32_t ringBytes(const SkinnyGEMMConfig &gemm) {
  return gemm.prefetch * (gemm.tn + gemm.bm / 16) * 4096;
}

// The LDS of the block: the rings of the waves, then the K reduction.
static int32_t ldsBytes(const SkinnyGEMMConfig &gemm) {
  return gemm.waves * ringBytes(gemm) +
         (gemm.waves - 1) * gemm.tn * (gemm.bm / 16) * 1024;
}

static bool fitsTheBudgets(const SkinnyGEMMConfig &gemm) {
  try {
    emitted(gemm);
  } catch (const std::invalid_argument &error) {
    return std::string(error.what()).find("budget") == std::string::npos;
  }
  return true;
}

// Calls check with the assembly of every representative lds configuration that
// fits the LDS and the registers.
template <typename CheckFn>
static void forEachLdsKernel(CheckFn check) {
  for (int32_t bm : {16, 32})
    for (int32_t tn : {1, 2, 3})
      for (int32_t waves : {1, 2, 4})
        for (int32_t prefetch : {1, 2, 4})
          for (bool grid3d : {true, false}) {
            SkinnyGEMMConfig gemm = ldsConfig(bm, tn, waves, prefetch, grid3d);
            if (!fitsTheBudgets(gemm))
              continue;
            SCOPED_TRACE(skinnyGEMMKernelName(gemm));
            check(gemm, emitted(gemm));
          }
}

TEST(SkinnyGEMMLds, KernelNameAppendsTheSuffixAfterMfma) {
  for (SkinnyGEMMConfig gemm : {ldsConfig(16, 1, 4, 2), ldsConfig(32, 3, 1, 1),
                                ldsConfig(32, 2, 2, 3, false, true)}) {
    SkinnyGEMMConfig direct = gemm;
    direct.lds = false;
    EXPECT_EQ(skinnyGEMMKernelName(gemm),
              skinnyGEMMKernelName(direct) + "_LDS1");
  }
  EXPECT_EQ(skinnyGEMMKernelName(ldsConfig(32, 1, 4, 2)),
            "gemm_BM32_TN1_LANES_K64_WAVES4_UP_ITERS0_PARTIAL_STEPS0_PF2_"
            "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0_MFMA1_LDS1");
}

TEST(SkinnyGEMMLds, RequiresMfma) {
  SkinnyGEMMConfig gemm;
  gemm.lds = true;
  EXPECT_THROW(skinnyGEMMKernelName(gemm), std::invalid_argument);
  EXPECT_THROW(emitSkinnyGEMM(Target::get("gfx950"), gemm),
               std::invalid_argument);
  EXPECT_NE(rejectionMessage(gemm).find("lds without mfma"), std::string::npos);
  EXPECT_NO_THROW(skinnyGEMMKernelName(ldsConfig(16, 1, 1, 1)));
}

// The rings of all waves and the reduction area must fit the LDS of the
// target: 4 * 4 * 16 KiB are over it, and a single wave fills it exactly with
// 10 slots of 4 tiles.
TEST(SkinnyGEMMLds, RejectsConfigsOverTheLdsBudget) {
  Target target = Target::get("gfx950");
  SkinnyGEMMConfig over = ldsConfig(32, 2, 4, 4);
  EXPECT_NE(
      rejectionMessage(over).find(
          "config exceeds the LDS budget: " + std::to_string(ldsBytes(over)) +
          " bytes of " + std::to_string(target.maxLdsSize())),
      std::string::npos);
  SkinnyGEMMConfig exact = ldsConfig(32, 2, 1, 10);
  EXPECT_EQ(ldsBytes(exact), target.maxLdsSize());
  EXPECT_NO_THROW(emitted(exact));
  EXPECT_NE(rejectionMessage(ldsConfig(32, 2, 1, 11))
                .find("config exceeds the LDS budget: "),
            std::string::npos);
}

TEST(SkinnyGEMMLds, AllocatesTheRingsThenTheReduction) {
  forEachLdsKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    EXPECT_NE(text.find(".group_segment_fixed_size: " +
                        std::to_string(ldsBytes(gemm))),
              std::string::npos);
  });
}

// Every operand load goes to LDS; only the scale dwords load into registers.
TEST(SkinnyGEMMLds, OperandLoadsGoStraightToLds) {
  forEachLdsKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::vector<std::string> body = kLoopBody(text);
    int32_t direct = 0;
    for (const std::string &line : allLines(text))
      if (line.rfind("buffer_load_dwordx4", 0) == 0 &&
          line.find(" lds") == std::string::npos)
        ++direct;
    EXPECT_EQ(direct, 0);
    EXPECT_EQ(countLinesContaining(joined(body), "buffer_load_dwordx4"),
              4 * tiles * gemm.prefetch);
    // The prologue and the loop issue PF slots each, and the arms the refills
    // of the iterations that exist: PF * (PF - 1) / 2 slots in all.
    EXPECT_EQ(
        countLinesContaining(text, "buffer_load_dwordx4"),
        4 * tiles *
            (2 * gemm.prefetch + gemm.prefetch * (gemm.prefetch - 1) / 2));
  });
}

// M0 holds the LDS base of the load: every load is preceded by its write and
// the wait state after it.
TEST(SkinnyGEMMLds, EveryLdsLoadFollowsItsM0Write) {
  forEachLdsKernel([](const SkinnyGEMMConfig &, const std::string &text) {
    std::vector<std::string> lines = allLines(text);
    int32_t loads = 0;
    for (int64_t i = 2; i < static_cast<int64_t>(lines.size()); ++i) {
      if (lines[i].rfind("buffer_load_dwordx4", 0) != 0)
        continue;
      ++loads;
      EXPECT_EQ(lines[i - 1], "s_nop 0");
      EXPECT_TRUE(lines[i - 2].rfind("s_mov_b32 m0, ", 0) == 0 ||
                  lines[i - 2].rfind("s_add_u32 m0, ", 0) == 0)
          << lines[i - 2];
    }
    EXPECT_GT(loads, 0);
  });
}

// The loop holds the MFMAs of every slot, as without lds, and reads the two
// steps of every slot from LDS: ds_read_b128 twice per tile and step.
TEST(SkinnyGEMMLds, LoopReadsEveryStepFromLds) {
  forEachLdsKernel([](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::string body = joined(kLoopBody(text));
    EXPECT_EQ(countLinesContaining(body, "v_mfma_scale_f32_16x16x128_f8f6f4"),
              2 * gemm.tn * (gemm.bm / 16) * gemm.prefetch);
    EXPECT_EQ(countLinesContaining(body, "ds_read_b128"),
              4 * tiles * gemm.prefetch);
    EXPECT_EQ(text.find("v_dot2_f32_bf16"), std::string::npos);
  });
}

// A slot waits for all its loads with the other PF - 1 slots in flight, reads
// its second step, waits for the first step's reads, then for the second's
// before the refill; the next slot's first step waits for its own loads, which
// leave the PF - 1 slots of the refill in flight behind them.
TEST(SkinnyGEMMLds, LoopWaitsForTheLoadsAndTheReads) {
  Target target = Target::get("gfx950");
  forEachLdsKernel([&](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t behind = (gemm.prefetch - 1) * plan.slotLoads;
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::vector<std::string> expected;
    for (int32_t slot = 0; slot < gemm.prefetch; ++slot) {
      expected.push_back(vmcnt(std::min(behind, target.maxVmcnt())));
      expected.push_back(lgkmcnt(std::min(2 * tiles, target.maxLgkmcnt())));
      expected.push_back(lgkmcnt(0));
      expected.push_back(
          vmcnt(std::min(behind + plan.stepLoads, target.maxVmcnt())));
    }
    EXPECT_EQ(waitsOf(kLoopBody(text)), expected);
  });
}

// The arm of R remaining iterations waits like the loop, with the groups that
// remain behind each slot, and reads the first step of the next slot only when
// there is one: the first step of its own first slot was read before.
TEST(SkinnyGEMMLds, RemainderArmsReadAheadOnlyWithinTheArm) {
  Target target = Target::get("gfx950");
  forEachLdsKernel([&](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::vector<std::vector<std::string>> arms = remainderArms(text);
    ASSERT_EQ(static_cast<int32_t>(arms.size()), 2 * gemm.prefetch - 1);
    for (int32_t remaining = 1; remaining < 2 * gemm.prefetch; ++remaining) {
      std::vector<std::string> expected;
      for (int32_t i = 0; i < remaining; ++i) {
        int32_t groups = armGroupsAfter(gemm.prefetch, remaining, i);
        bool refill = i + gemm.prefetch < remaining;
        expected.push_back(
            vmcnt(std::min(groups * plan.slotLoads, target.maxVmcnt())));
        expected.push_back(lgkmcnt(std::min(2 * tiles, target.maxLgkmcnt())));
        expected.push_back(lgkmcnt(0));
        if (i + 1 < remaining)
          expected.push_back(
              vmcnt(std::min(groups * plan.slotLoads - plan.stepLoads +
                                 (refill ? 2 * plan.stepLoads : 0),
                             target.maxVmcnt())));
      }
      const std::vector<std::string> &arm = arms[remaining - 1];
      EXPECT_EQ(waitsOf(arm), expected);
      EXPECT_EQ(countLinesContaining(joined(arm), "ds_read_b128"),
                2 * tiles * (2 * remaining - 1));
    }
  });
}

// The first step of the first slot is read before the loop, after a wait for
// its loads that leaves the other PF - 1 slots and the rest of the slot in
// flight. The loop is skipped when no iteration is left for it, and the first
// arm then consumes the slot that was read.
TEST(SkinnyGEMMLds, FirstStepIsReadBeforeTheLoop) {
  Target target = Target::get("gfx950");
  forEachLdsKernel([&](const SkinnyGEMMConfig &gemm, const std::string &text) {
    MfmaPlan plan(gemm);
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::vector<std::string> lines = allLines(text);
    std::ptrdiff_t loop = indexOfLineStartingWith(lines, "label_kloop_");
    // The loop header is aligned, which is a line before its label, and
    // follows the check that skips the loop: a compare, a move of the counter
    // and the branch.
    ASSERT_GE(loop, 2 * tiles + 5);
    EXPECT_EQ(lines[loop - 1], ".p2align 4");
    EXPECT_EQ(lines[loop - 2].rfind("s_cbranch_scc0 label_kloop_end_", 0), 0u);
    std::ptrdiff_t first = loop - 4 - 2 * tiles;
    for (std::ptrdiff_t i = first; i < loop - 4; ++i)
      EXPECT_TRUE(lines[i].rfind("ds_read_b128", 0) == 0) << lines[i];
    int32_t allowed = (gemm.prefetch - 1) * plan.slotLoads + plan.stepLoads;
    EXPECT_EQ(lines[first - 1], vmcnt(std::min(allowed, target.maxVmcnt())));
  });
}

TEST(SkinnyGEMMLds, EveryKernelAssembles) {
  Target target = Target::get("gfx950");
  forEachLdsKernel([&](const SkinnyGEMMConfig &, const std::string &text) {
    EXPECT_FALSE(assembleToHSACO(text, target.isaName()).empty());
  });
  for (const SkinnyGEMMConfig &gemm :
       {ldsConfig(16, 2, 2, 3, false, true), ldsConfig(32, 3, 1, 3, true, true),
        partialMfmaConfig(ldsConfig(16, 2, 2, 3, true, true)),
        partialMfmaConfig(ldsConfig(32, 2, 4, 1, false))}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    EXPECT_FALSE(assembleToHSACO(emitted(gemm), target.isaName()).empty());
  }
}

TEST(SkinnyGEMMLds, ReducesThroughLdsAfterTheRings) {
  SkinnyGEMMConfig gemm = ldsConfig(16, 2, 2, 2);
  std::string text = emitted(gemm);
  EXPECT_EQ(countLinesContaining(text, "s_barrier"), 2);
  EXPECT_EQ(countLinesContaining(text, "ds_write_b128"), 2);
  EXPECT_NE(
      text.find(".group_segment_fixed_size: " + std::to_string(ldsBytes(gemm))),
      std::string::npos);
}

// The tail iteration loads straight to LDS in the prologue, into a slot after
// the ring of its own, so the tail code only reads both steps and masks the
// registers, with the scale override, as without lds.
TEST(SkinnyGEMMLds, PartialTailIsLoadedInThePrologueAndMasksTheRegisters) {
  for (const SkinnyGEMMConfig &gemm :
       {ldsConfig(16, 1, 1, 1), ldsConfig(16, 2, 2, 3, false, true),
        ldsConfig(32, 3, 2, 2), ldsConfig(32, 2, 1, 3, false)}) {
    SCOPED_TRACE(skinnyGEMMKernelName(gemm));
    MfmaPlan plan(gemm);
    int32_t tiles = plan.wideTiles + plan.skinnyTiles;
    std::string full = emitted(gemm);
    std::string partial = emitted(partialMfmaConfig(gemm));
    EXPECT_EQ(full.find("label_tail_begin"), std::string::npos);
    expectTailMasks(partial);
    EXPECT_EQ(full.find("2139062143"), std::string::npos);
    EXPECT_EQ(countLinesContaining(partial, "buffer_load_dwordx4"),
              countLinesContaining(full, "buffer_load_dwordx4") + 4 * tiles);
    EXPECT_NE(partial.find(".group_segment_fixed_size: " +
                           std::to_string(ldsBytes(gemm) +
                                          gemm.waves * tiles * 4096)),
              std::string::npos);
    std::vector<std::string> tail =
        linesBetween(partial, "label_tail_begin_", "label_tail_skip_");
    ASSERT_FALSE(tail.empty());
    EXPECT_EQ(countLinesContaining(joined(tail), "buffer_load"), 0);
    EXPECT_EQ(countLinesContaining(joined(tail), "ds_read_b128"), 4 * tiles);
    EXPECT_EQ(
        countLinesContaining(joined(tail), "v_mfma_scale_f32_16x16x128_f8f6f4"),
        2 * gemm.tn * (gemm.bm / 16));
  }
}

// The tail slot counts for the LDS budget with partial steps.
TEST(SkinnyGEMMLds, PartialTailSlotCountsForTheLdsBudget) {
  SkinnyGEMMConfig exact = ldsConfig(32, 2, 1, 10);
  EXPECT_NO_THROW(emitted(exact));
  EXPECT_NE(rejectionMessage(partialMfmaConfig(exact))
                .find("config exceeds the LDS budget: "),
            std::string::npos);
  EXPECT_NO_THROW(emitted(partialMfmaConfig(ldsConfig(32, 2, 1, 9))));
}
