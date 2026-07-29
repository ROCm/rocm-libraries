// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Golden test for the build-time-generated per-architecture geometry table
// (data/hardware/*.json -> hardware_generated.inc). Asserts the generated
// values reproduce the constants previously hardcoded in hardware.cpp.

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "origami/hardware.hpp"

using origami::hardware_t;
using arch_t = hardware_t::architecture_t;

TEST_CASE("Hardware: generated default num_xcds matches expected", "[hardware]") {
  // Expected values are derived from data/hardware/rocjitsu/*.json (for the archs
  // rocjitsu covers) and data/hardware/*.json (for the rest). gfx1250 is 8 (from
  // the rocjitsu topology; the old hardcoded value was a placeholder 1).
  const std::vector<std::pair<arch_t, size_t>> expected = {
      {arch_t::gfx90a, 1},  {arch_t::gfx942, 8},   {arch_t::gfx950, 8},
      {arch_t::gfx1200, 1}, {arch_t::gfx1201, 1},  {arch_t::gfx1100, 1},
      {arch_t::gfx1150, 1}, {arch_t::gfx1151, 1},  {arch_t::gfx1152, 1},
      {arch_t::gfx1153, 1}, {arch_t::gfx1250, 8},
  };

  for (const auto& [arch, val] : expected) {
    INFO("arch = " << hardware_t::arch_enum_to_name(arch));
    REQUIRE(hardware_t::get_default_num_xcds(arch) == val);
  }
}

TEST_CASE("Hardware: generated cache-line size matches expected", "[hardware]") {
  // Derived from rocjitsu l2_line_size where available. gfx1201 reports 256 B;
  // all others are 128 B.
  const std::vector<std::pair<arch_t, size_t>> expected = {
      {arch_t::gfx90a, 128},  {arch_t::gfx942, 128},   {arch_t::gfx950, 128},
      {arch_t::gfx1200, 128}, {arch_t::gfx1201, 256},  {arch_t::gfx1100, 128},
      {arch_t::gfx1150, 128}, {arch_t::gfx1151, 128},  {arch_t::gfx1152, 128},
      {arch_t::gfx1153, 128}, {arch_t::gfx1250, 128},
  };

  for (const auto& [arch, val] : expected) {
    INFO("arch = " << hardware_t::arch_enum_to_name(arch));
    REQUIRE(hardware_t::get_default_cache_line_bytes(arch) == val);
  }
}

TEST_CASE("Hardware: default num_xcds throws for the Count sentinel", "[hardware]") {
  REQUIRE_THROWS(hardware_t::get_default_num_xcds(arch_t::Count));
}
