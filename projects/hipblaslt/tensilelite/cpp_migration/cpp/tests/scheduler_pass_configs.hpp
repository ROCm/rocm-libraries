// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared config table for the LogicalScheduler pass-pipeline snapshot tests.
//
// These specs are the self-contained scheduler configs the deleted
// test_logicalSchedulerCpp.py pass-pipeline parity sweep used (_PASS_CONFIGS =
// CONFIGS + _PASS_EXTRA_CONFIGS): representative BF16 / fp8 / fp4 gfx950
// reference shapes across single/multi-partition and PGR=0/1/2. Both the
// snapshot test and the golden generator build their schedulers from this one
// table so the two never drift.

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensile_writer/logical_scheduler.hpp"
#include "tensile_writer/logical_scheduler_passes.hpp"

namespace tw_test {

struct PassConfigSpec {
  std::string name;
  int M = 0;
  int N = 0;
  int numSubIterK = 0;
  std::pair<int, int> lrA{1, 1};
  std::pair<int, int> lrB{1, 1};
  std::pair<int, int> grA{1, 2};
  std::pair<int, int> grB{1, 2};
  std::optional<std::pair<int, int>> lrSA, lrSB, grSA, grSB;
  tw::subtile::lsched::PartitionSpec partM = 0;
  tw::subtile::lsched::PartitionSpec partN = 0;
  int pgr = 2;
};

inline tw::subtile::lsched::SchedulerConfig makePassConfig(const PassConfigSpec& s) {
  using tw::subtile::lsched::ReadGranularity;
  tw::subtile::lsched::SchedulerConfig c;
  c.numMFMATilesM = s.M;
  c.numMFMATilesN = s.N;
  c.numSubIterK = s.numSubIterK;
  c.lrA = ReadGranularity(s.lrA.first, s.lrA.second);
  c.lrB = ReadGranularity(s.lrB.first, s.lrB.second);
  c.grA = ReadGranularity(s.grA.first, s.grA.second);
  c.grB = ReadGranularity(s.grB.first, s.grB.second);
  if (s.lrSA) c.lrSA = ReadGranularity(s.lrSA->first, s.lrSA->second);
  if (s.lrSB) c.lrSB = ReadGranularity(s.lrSB->first, s.lrSB->second);
  if (s.grSA) c.grSA = ReadGranularity(s.grSA->first, s.grSA->second);
  if (s.grSB) c.grSB = ReadGranularity(s.grSB->first, s.grSB->second);
  c.partitionSizeM = s.partM;
  c.partitionSizeN = s.partN;
  c.pgr = s.pgr;
  c.post_init();
  return c;
}

inline std::vector<PassConfigSpec> passConfigs() {
  auto scaled = [](PassConfigSpec s) {
    s.lrSA = {2, 2};
    s.lrSB = {2, 2};
    s.grSA = {2, 2};
    s.grSB = {2, 2};
    return s;
  };
  std::vector<PassConfigSpec> v;
  // CONFIGS (value layer) reused for the pass-pipeline sweep.
  v.push_back({"no_scale_k1", 2, 2, 2});
  {
    PassConfigSpec s{"fp4_2x2", 8, 8, 2};
    s.partM = 4;
    s.partN = 4;
    v.push_back(scaled(s));
  }
  {
    PassConfigSpec s{"bf16_10x1", 10, 10, 2};
    s.partM = 1;
    s.partN = 10;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x384_n6", 8, 12, 2};
    s.partN = 6;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x384_n4", 8, 12, 2};
    s.partN = 4;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x384_n5", 8, 12, 2};
    s.partN = 5;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x352_n4", 8, 11, 2};
    s.partN = 4;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x352_n3", 8, 11, 2};
    s.partN = 3;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x368_n4", 4, 23, 2};
    s.partN = 4;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x368_n8", 4, 23, 2};
    s.partN = 8;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"explicit_list_N", 2, 12, 2};
    s.partN = std::vector<int>{5, 2, 5};
    v.push_back(s);
  }
  {
    PassConfigSpec s{"pgr0_single", 2, 2, 1};
    s.pgr = 0;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"pgr1_single", 2, 2, 1};
    s.pgr = 1;
    v.push_back(s);
  }
  // _PASS_EXTRA_CONFIGS — PGR=0/1/2 and the scale path beyond the value layer.
  {
    PassConfigSpec s{"fp8_8x8_pgr2", 8, 8, 2};
    s.pgr = 2;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"fp8_8x8_pgr1", 8, 8, 2};
    s.pgr = 1;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"bf16_256x384_n6_pgr1", 8, 12, 2};
    s.partN = 6;
    s.pgr = 1;
    v.push_back(s);
  }
  {
    PassConfigSpec s{"fp4_2x2_pgr1", 8, 8, 2};
    s.partM = 4;
    s.partN = 4;
    s.pgr = 1;
    v.push_back(scaled(s));
  }
  return v;
}

// print_* method -> terminal pass that must run first (matches the Python
// _PRINT_TO_PASS map). Running the terminal pass auto-runs its prerequisites.
inline const std::vector<std::string>& printMethods() {
  static const std::vector<std::string> m = {
      "print_lr",         "print_vgpr",         "print_gr",   "print_deps",
      "print_remove_deps", "print_group_lr_gr", "print_emit"};
  return m;
}

// Run the terminal pass for `method` and return its print output.
inline std::string runPrint(tw::subtile::lsched::passes::LogicalScheduler& s,
                            const std::string& method) {
  using tw::subtile::lsched::Pass;
  if (method == "print_lr") {
    s.ensure(Pass::LR);
    return s.print_lr();
  }
  if (method == "print_vgpr") {
    s.ensure(Pass::VGPR_TILES);
    return s.print_vgpr();
  }
  if (method == "print_gr") {
    s.ensure(Pass::GR);
    return s.print_gr();
  }
  if (method == "print_deps") {
    s.ensure(Pass::DEPS);
    return s.print_deps();
  }
  if (method == "print_remove_deps") {
    s.ensure(Pass::REMOVE_DEPS);
    return s.print_remove_deps();
  }
  if (method == "print_group_lr_gr") {
    s.ensure(Pass::GROUP_LR_GR);
    return s.print_group_lr_gr();
  }
  // print_emit
  s.ensure(Pass::EMIT);
  return s.print_emit();
}

}  // namespace tw_test
