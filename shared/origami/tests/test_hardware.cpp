// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "origami/hardware.hpp"

namespace {

using origami::architecture_t;
using origami::data_type_t;
using origami::family_t;
using origami::hardware_t;
using origami::variant_t;

// What the model has to know about each architecture, written out independently of the
// tables under test.  The order is the order of architecture_t.
struct expected_architecture {
  architecture_t arch;
  std::string_view name;
  family_t family;
  size_t wavefront_size;
  size_t simd_lanes;
  size_t simds_per_cu;
  size_t lds_read_bytes_per_clk;
  size_t l1_capacity;
  size_t default_num_xcds;
  bool has_mall;
  bool has_native_tf32;
};

// clang-format off
const std::vector<expected_architecture> EXPECTED_ARCHITECTURES = {
    // architecture            name       family           wave lanes simds lds  l1         xcds mall   tf32
    {architecture_t::gfx90a,  "gfx90a",  family_t::cdna2, 64,  16,   4,    128, 16 * 1024, 1,   true,  false},
    {architecture_t::gfx942,  "gfx942",  family_t::cdna3, 64,  16,   4,    128, 32 * 1024, 8,   true,  true},
    {architecture_t::gfx950,  "gfx950",  family_t::cdna4, 64,  16,   4,    256, 32 * 1024, 8,   true,  false},
    {architecture_t::gfx1200, "gfx1200", family_t::rdna4, 32,  32,   2,    128, 64 * 1024, 1,   true,  false},
    {architecture_t::gfx1201, "gfx1201", family_t::rdna4, 32,  32,   2,    128, 64 * 1024, 1,   true,  false},
    {architecture_t::gfx1170, "gfx1170", family_t::rdna4, 32,  32,   2,    128, 64 * 1024, 1,   false, false},
    {architecture_t::gfx1171, "gfx1171", family_t::rdna4, 32,  32,   2,    128, 64 * 1024, 1,   false, false},
    {architecture_t::gfx1100, "gfx1100", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   true,  false},
    {architecture_t::gfx1101, "gfx1101", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   true,  false},
    {architecture_t::gfx1150, "gfx1150", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   false, false},
    {architecture_t::gfx1151, "gfx1151", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   true,  false},
    {architecture_t::gfx1152, "gfx1152", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   false, false},
    {architecture_t::gfx1153, "gfx1153", family_t::rdna3, 32,  32,   2,    128, 32 * 1024, 1,   false, false},
    {architecture_t::gfx1250, "gfx1250", family_t::cdna5, 32,  32,   4,    256, 0,         8,   false, false},
};
// clang-format on

// A device of the given architecture built through the explicit constructor, which accepts
// every architecture value.
hardware_t make_device(architecture_t arch,
                       size_t n_cu                    = 64,
                       std::optional<int> pci_chip_id = std::nullopt) {
  return hardware_t(
      arch, n_cu, 65536, 512 * 1024, 4, 1.0, 1.0, 1.0, 4 << 20, 2.0, 4, {0, 0, 0}, pci_chip_id);
}

void require_same_constants(const hardware_t::architecture_constants& actual,
                            const hardware_t::architecture_constants& expected) {
  REQUIRE(actual.mem1_perf_ratio == expected.mem1_perf_ratio);
  REQUIRE(actual.mem2_perf_ratio == expected.mem2_perf_ratio);
  REQUIRE(actual.mem3_perf_ratio == expected.mem3_perf_ratio);
  REQUIRE(actual.parallel_mi_cu == expected.parallel_mi_cu);
  REQUIRE(actual.mem_bw_per_wg_coefficients == expected.mem_bw_per_wg_coefficients);
  REQUIRE(actual.mem_clock_ratio == expected.mem_clock_ratio);
  REQUIRE(actual.simds_per_cu == expected.simds_per_cu);
  REQUIRE(actual.l1_capacity == expected.l1_capacity);
}

// Valid matrix instructions of a datatype, sorted: their order in the library follows the
// standard library's hash table and is not part of what these tests pin down.
std::vector<std::tuple<size_t, size_t, size_t>> sorted_instructions(const hardware_t& hardware,
                                                                    data_type_t type) {
  std::vector<std::tuple<size_t, size_t, size_t>> result;
  for (const origami::dim3_t& mi : hardware.get_valid_matrix_instructions(type)) {
    result.emplace_back(mi.m, mi.n, mi.k);
  }
  std::sort(result.begin(), result.end());
  return result;
}

// The lookups stay usable in constant expressions.
static_assert(hardware_t::arch_name_to_enum("gfx950") == architecture_t::gfx950);
static_assert(hardware_t::arch_enum_to_name(architecture_t::gfx950) == "gfx950");
static_assert(origami::family_of(hardware_t::architecture_t::gfx1250) == family_t::cdna5);
static_assert(origami::resolve_variant(architecture_t::gfx1250, std::nullopt, 96) ==
              variant_t::gfx1250_6xcd);
static_assert(hardware_t::get_wavefront_size(architecture_t::gfx942) == 64);
static_assert(hardware_t::get_arch_constants(architecture_t::gfx942).parallel_mi_cu == 4);
static_assert(hardware_t::get_gfx950_arch_constants(0x75a8).mem1_perf_ratio == 8.5);
static_assert(hardware_t::get_gfx1250_arch_constants().mem1_perf_ratio == 41.93);
static_assert(hardware_t::gfx1250_wgps_select_6xcd(192));
static_assert(hardware_t::NO_MALL_AVAILABLE == 1.21875121875121875122 * 1000);

}  // namespace

TEST_CASE("Hardware: architecture names round-trip for every architecture",
          "[hardware][architecture]") {
  REQUIRE(EXPECTED_ARCHITECTURES.size() == static_cast<size_t>(architecture_t::Count));

  for (const auto& expected : EXPECTED_ARCHITECTURES) {
    INFO(expected.name);
    REQUIRE(hardware_t::arch_name_to_enum(expected.name) == expected.arch);
    REQUIRE(hardware_t::arch_enum_to_name(expected.arch) == expected.name);
    REQUIRE(origami::architecture_from_name(expected.name) == expected.arch);
    REQUIRE(origami::architecture_name(expected.arch) == expected.name);
    REQUIRE(origami::is_known_architecture(expected.arch));
  }

  SECTION("the enumeration follows the table order") {
    for (size_t i = 0; i < EXPECTED_ARCHITECTURES.size(); ++i) {
      REQUIRE(static_cast<size_t>(EXPECTED_ARCHITECTURES[i].arch) == i);
    }
  }

  SECTION("unrecognized names and values") {
    for (std::string_view name : {"", "gfx", "gfx95", "gfx9500", "GFX950", "gfx950:sramecc+"}) {
      INFO("'" << name << "'");
      REQUIRE(hardware_t::arch_name_to_enum(name) == architecture_t::Count);
    }
    REQUIRE(hardware_t::arch_enum_to_name(architecture_t::Count) == "unknown");
    REQUIRE(hardware_t::arch_enum_to_name(static_cast<architecture_t>(9999)) == "unknown");
    REQUIRE_FALSE(origami::is_known_architecture(architecture_t::Count));
  }
}

TEST_CASE("Hardware: family of every architecture", "[hardware][family]") {
  for (const auto& expected : EXPECTED_ARCHITECTURES) {
    INFO(expected.name);
    REQUIRE(origami::family_of(expected.arch) == expected.family);

    const hardware_t hardware = make_device(expected.arch);
    REQUIRE(hardware.family() == expected.family);
    REQUIRE(hardware.is_cdna5() == (expected.family == family_t::cdna5));
  }
  REQUIRE(origami::family_of(architecture_t::Count) == family_t::unknown);
  REQUIRE(origami::family_of(static_cast<architecture_t>(9999)) == family_t::unknown);

  SECTION("family names") {
    REQUIRE(origami::family_name(family_t::cdna2) == "cdna2");
    REQUIRE(origami::family_name(family_t::cdna3) == "cdna3");
    REQUIRE(origami::family_name(family_t::cdna4) == "cdna4");
    REQUIRE(origami::family_name(family_t::cdna5) == "cdna5");
    REQUIRE(origami::family_name(family_t::rdna3) == "rdna3");
    REQUIRE(origami::family_name(family_t::rdna4) == "rdna4");
    REQUIRE(origami::family_name(family_t::unknown) == "unknown");
  }

  SECTION("a family dispatches to its tag") {
    struct family_of_tag {
      family_t operator()(origami::arch::cdna2) const { return family_t::cdna2; }
      family_t operator()(origami::arch::cdna3) const { return family_t::cdna3; }
      family_t operator()(origami::arch::cdna4) const { return family_t::cdna4; }
      family_t operator()(origami::arch::cdna5) const { return family_t::cdna5; }
      family_t operator()(origami::arch::rdna3) const { return family_t::rdna3; }
      family_t operator()(origami::arch::rdna4) const { return family_t::rdna4; }
    };
    for (const auto& expected : EXPECTED_ARCHITECTURES) {
      INFO(expected.name);
      REQUIRE(origami::arch::dispatch(expected.family, family_of_tag{}) == expected.family);
      REQUIRE(origami::arch::dispatch(make_device(expected.arch), family_of_tag{}) ==
              expected.family);
    }
    REQUIRE(origami::arch::dispatch(family_t::unknown, family_of_tag{}) == family_t::cdna3);
  }
}

TEST_CASE("Hardware: traits of every architecture", "[hardware][traits]") {
  for (const auto& expected : EXPECTED_ARCHITECTURES) {
    INFO(expected.name);
    const bool is_cdna5 = expected.family == family_t::cdna5;

    REQUIRE(hardware_t::get_wavefront_size(expected.arch) == expected.wavefront_size);
    REQUIRE(hardware_t::get_simd_lanes(expected.arch) == expected.simd_lanes);
    REQUIRE(hardware_t::get_simds_per_cu(expected.arch) == expected.simds_per_cu);
    REQUIRE(hardware_t::get_valu_cycles_per_instruction(expected.arch) ==
            expected.wavefront_size / expected.simd_lanes);
    REQUIRE(hardware_t::get_has_accumulation_registers(expected.arch) == !is_cdna5);
    REQUIRE(hardware_t::get_lds_read_bytes_per_clk(expected.arch) ==
            expected.lds_read_bytes_per_clk);
    REQUIRE(hardware_t::get_l1_capacity(expected.arch) == expected.l1_capacity);
    REQUIRE(hardware_t::get_default_cache_line_bytes(expected.arch) == 128);
    REQUIRE(hardware_t::get_default_num_xcds(expected.arch) == expected.default_num_xcds);

    const hardware_t hardware = make_device(expected.arch);
    REQUIRE(hardware.wavefront_size() == expected.wavefront_size);
    REQUIRE(hardware.simd_lanes() == expected.simd_lanes);
    REQUIRE(hardware.simds_per_cu() == expected.simds_per_cu);
    REQUIRE(hardware.has_accumulation_registers() == !is_cdna5);
    REQUIRE(hardware.l1_capacity == expected.l1_capacity);
    REQUIRE(hardware.has_MALL() == expected.has_mall);
    REQUIRE(hardware.has_native_TF32() == expected.has_native_tf32);
  }

  SECTION("an unrecognized architecture") {
    const architecture_t unknown = architecture_t::Count;
    REQUIRE(hardware_t::get_wavefront_size(unknown) == 32);
    REQUIRE(hardware_t::get_simd_lanes(unknown) == 32);
    REQUIRE(hardware_t::get_simds_per_cu(unknown) == 4);
    REQUIRE(hardware_t::get_has_accumulation_registers(unknown));
    REQUIRE(hardware_t::get_lds_read_bytes_per_clk(unknown) == 128);
    REQUIRE(hardware_t::get_l1_capacity(unknown) == 32 * 1024);
    REQUIRE(hardware_t::get_default_cache_line_bytes(unknown) == 128);
    REQUIRE_THROWS_AS(hardware_t::get_default_num_xcds(unknown), std::runtime_error);

    const hardware_t hardware = make_device(unknown);
    REQUIRE_FALSE(hardware.has_MALL());
    REQUIRE_FALSE(hardware.has_native_TF32());
    REQUIRE_FALSE(hardware.is_cdna5());
  }
}

TEST_CASE("Hardware: L2 domains and dispatch order", "[hardware][traits]") {
  // gfx1250 shares one L2 between the up to 4 XCDs of an AID and dispatches from the last XCD.
  hardware_t cdna5 = make_device(architecture_t::gfx1250);
  cdna5.NUM_XCD    = 8;
  REQUIRE(cdna5.num_l2_domains() == 2);
  REQUIRE(cdna5.xcd_per_l2_domain() == 4);
  REQUIRE(cdna5.dispatch_xcd(0) == 7);
  REQUIRE(cdna5.dispatch_xcd(1) == 0);
  REQUIRE(cdna5.dispatch_xcd(8) == 7);
  cdna5.NUM_XCD = 6;
  REQUIRE(cdna5.num_l2_domains() == 2);
  REQUIRE(cdna5.xcd_per_l2_domain() == 3);
  cdna5.NUM_XCD = 3;
  REQUIRE(cdna5.num_l2_domains() == 1);
  REQUIRE(cdna5.xcd_per_l2_domain() == 3);

  // Every other architecture has one L2 per XCD and dispatches from XCD 0.
  hardware_t cdna4 = make_device(architecture_t::gfx950);
  cdna4.NUM_XCD    = 8;
  REQUIRE(cdna4.num_l2_domains() == 8);
  REQUIRE(cdna4.xcd_per_l2_domain() == 1);
  REQUIRE(cdna4.dispatch_xcd(0) == 0);
  REQUIRE(cdna4.dispatch_xcd(9) == 1);
  cdna4.NUM_XCD = 0;
  REQUIRE(cdna4.num_l2_domains() == 1);
  REQUIRE(cdna4.dispatch_xcd(5) == 0);
}

TEST_CASE("Hardware: variant resolution", "[hardware][variant]") {
  SECTION("gfx950 is told apart by PCI device id") {
    const architecture_t arch = architecture_t::gfx950;
    REQUIRE(origami::resolve_variant(arch) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, std::nullopt) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, 0x75a0) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, 0) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, 0x75a8) == variant_t::gfx950_id75a8);
    // Only the low 16 bits are the device id.
    REQUIRE(origami::resolve_variant(arch, 0x175a8) == variant_t::gfx950_id75a8);
    REQUIRE(origami::resolve_variant(arch, 0x75a80) == variant_t::standard);
    // The WGP count plays no role on gfx950.
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 96) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, 0x75a8, 192) == variant_t::gfx950_id75a8);
  }

  SECTION("gfx1250 is told apart by WGP count") {
    const architecture_t arch = architecture_t::gfx1250;
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 256) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 192) == variant_t::gfx1250_6xcd);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 96) == variant_t::gfx1250_6xcd);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 128) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 64) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 32) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, std::nullopt, 0) == variant_t::standard);
    // The PCI id plays no role on gfx1250.
    REQUIRE(origami::resolve_variant(arch, 0x75a8, 256) == variant_t::standard);
    REQUIRE(origami::resolve_variant(arch, 0x75a8, 96) == variant_t::gfx1250_6xcd);

    for (size_t n_cu : {size_t{256}, size_t{192}, size_t{96}, size_t{128}, size_t{0}}) {
      INFO(n_cu << " WGPs");
      REQUIRE(hardware_t::gfx1250_wgps_select_6xcd(n_cu) == (n_cu == 192 || n_cu == 96));
    }
  }

  SECTION("every other architecture has only the standard variant") {
    for (const auto& expected : EXPECTED_ARCHITECTURES) {
      if (expected.arch == architecture_t::gfx950 || expected.arch == architecture_t::gfx1250) {
        continue;
      }
      INFO(expected.name);
      REQUIRE(origami::resolve_variant(expected.arch, 0x75a8, 96) == variant_t::standard);
      REQUIRE(origami::resolve_variant(expected.arch, 0x75a8, 192) == variant_t::standard);
    }
    REQUIRE(origami::resolve_variant(architecture_t::Count, 0x75a8, 96) == variant_t::standard);
  }

  SECTION("variant names") {
    REQUIRE(origami::variant_name(variant_t::standard) == "standard");
    REQUIRE(origami::variant_name(variant_t::gfx950_id75a8) == "gfx950_id75a8");
    REQUIRE(origami::variant_name(variant_t::gfx1250_6xcd) == "gfx1250_6xcd");
  }
}

TEST_CASE("Hardware: every construction path stores the variant", "[hardware][variant]") {
  const auto for_arch = [](architecture_t arch, size_t n_cu, std::optional<int> pci_chip_id) {
    return hardware_t::get_hardware_for_arch(
        arch, n_cu, 65536, 512 * 1024, 4 << 20, 2'100'000, pci_chip_id);
  };
  const auto for_properties = [](const char* name, int n_cu, std::optional<int> pci_chip_id) {
    hipDeviceProp_t properties{};
    std::strncpy(properties.gcnArchName, name, sizeof(properties.gcnArchName) - 1);
    properties.multiProcessorCount = n_cu;
    properties.sharedMemPerBlock   = 65536;
    properties.regsPerBlock        = 131072;
    properties.l2CacheSize         = 4 << 20;
    properties.clockRate           = 2'100'000;
    properties.memoryClockRate     = 1'400'000;
    return hardware_t(properties, pci_chip_id);
  };

  SECTION("get_hardware_for_arch") {
    REQUIRE(for_arch(architecture_t::gfx942, 96, 0x75a8).variant() == variant_t::standard);
    REQUIRE(for_arch(architecture_t::gfx950, 256, std::nullopt).variant() == variant_t::standard);
    REQUIRE(for_arch(architecture_t::gfx950, 256, 0x75a8).variant() == variant_t::gfx950_id75a8);
    REQUIRE(for_arch(architecture_t::gfx1250, 256, std::nullopt).variant() == variant_t::standard);
    REQUIRE(for_arch(architecture_t::gfx1250, 192, std::nullopt).variant() ==
            variant_t::gfx1250_6xcd);
    REQUIRE(for_arch(architecture_t::gfx1250, 96, std::nullopt).variant() ==
            variant_t::gfx1250_6xcd);
    REQUIRE(for_arch(architecture_t::gfx1250, 128, std::nullopt).variant() == variant_t::standard);
  }

  SECTION("HIP device properties") {
    REQUIRE(for_properties("gfx950:sramecc+", 256, std::nullopt).variant() == variant_t::standard);
    REQUIRE(for_properties("gfx950:sramecc+", 256, 0x75a8).variant() == variant_t::gfx950_id75a8);
    REQUIRE(for_properties("gfx1250", 256, std::nullopt).variant() == variant_t::standard);
    REQUIRE(for_properties("gfx1250", 192, std::nullopt).variant() == variant_t::gfx1250_6xcd);
    // RDNA reports WGPs of 2 CUs each; the variant is resolved from the physical CU count.
    const hardware_t rdna4 = for_properties("gfx1201", 30, std::nullopt);
    REQUIRE(rdna4.N_CU == 60);
    REQUIRE(rdna4.variant() == variant_t::standard);
  }

  SECTION("explicit constructor and copy") {
    REQUIRE(make_device(architecture_t::gfx950, 64, 0x75a8).variant() == variant_t::gfx950_id75a8);
    REQUIRE(make_device(architecture_t::gfx950, 64).variant() == variant_t::standard);
    REQUIRE(make_device(architecture_t::gfx1250, 96).variant() == variant_t::gfx1250_6xcd);

    const hardware_t original = for_arch(architecture_t::gfx1250, 192, std::nullopt);
    const hardware_t copy(original);
    REQUIRE(copy.variant() == variant_t::gfx1250_6xcd);
    REQUIRE(copy.NUM_XCD == original.NUM_XCD);
    REQUIRE(copy.mem1_perf_ratio == original.mem1_perf_ratio);
  }
}

TEST_CASE("Hardware: constants of every architecture and variant", "[hardware][constants]") {
  using constants_t    = hardware_t::architecture_constants;
  const double no_mall = hardware_t::NO_MALL_AVAILABLE;

  // clang-format off
  const constants_t gfx90a        = {5.5,     1.21875121875121875122 * 1.2,     1.2,               4, std::make_tuple(0, 0.03, 0),                     1.5,       4, 16 * 1024};
  const constants_t gfx942        = {17,      1.21875121875121875122 * 6,       4,                 4, std::make_tuple(0, 0.015, 0),                    1.5,       4, 32 * 1024};
  const constants_t gfx950        = {17,      1.21875121875121875122 * 7,       6,                 4, std::make_tuple(-0.0000194, 0.008772, 0.007898), 1.5,       4, 32 * 1024};
  const constants_t gfx950_id75a8 = {8.5,     1.21875121875121875122 * 3.6,     2.55,              4, std::make_tuple(-0.000098, 0.02011, 0),          1.5,       4, 32 * 1024};
  const constants_t gfx1250       = {41.93,   no_mall,                          17.13 * 1.9 / 2.4, 4, std::make_tuple(-1.81e-5, 8.43e-3, -1.42e-3),    2.4 / 1.9, 4, 0};
  const constants_t gfx1100       = {7.12,    1.21875121875121875122 * 3.48,    0.732,             2, std::make_tuple(0, 0.11, 0),                     1.5,       2, 32 * 1024};
  const constants_t gfx1101       = {4.86133, 1.21875121875121875122 * 1.28305, 0.420583,          2, std::make_tuple(0, 0.13169, 0),                  1.5,       2, 32 * 1024};
  const constants_t gfx1150       = {1.497,   no_mall,                          0.0300383,         2, std::make_tuple(0, 0.18, 0),                     1.5,       2, 32 * 1024};
  const constants_t gfx1151       = {2.47,    1.21875121875121875122 * 0.93,    0.215,             2, std::make_tuple(0, 0.22, 0),                     1.5,       2, 32 * 1024};
  const constants_t gfx1152       = {0.849,   no_mall,                          0.096,             4, std::make_tuple(0, 0.13, 0),                     1.5,       2, 32 * 1024};
  const constants_t gfx1153       = {0.240,   no_mall,                          0.066,             2, std::make_tuple(0, 0.19, 0),                     1.5,       2, 32 * 1024};
  const constants_t gfx117x       = {5.74,    no_mall,                          0.464,             2, std::make_tuple(0, 0.17, 0),                     1.5,       2, 64 * 1024};
  const constants_t gfx1200       = {3.28,    1.21875121875121875122 * 1.45,    0.280,             2, std::make_tuple(0, 0.31, 0),                     1.5,       2, 64 * 1024};
  const constants_t gfx1201       = {5.74,    1.21875121875121875122 * 2.41,    0.464,             2, std::make_tuple(0, 0.17, 0),                     1.5,       2, 64 * 1024};
  const constants_t unknown       = {0,       0,                                0,                 0, std::make_tuple(0, 0, 0),                        0,         4, 32 * 1024};
  // clang-format on

  // The 6-XCD gfx1250 part, for a device of n_cu WGPs.
  const auto gfx1250_6xcd = [&](size_t n_cu) {
    const double partition = static_cast<double>(n_cu) / 96.0;
    return constants_t{14.75 * (41301.81761089743 / 41930.0) * partition,
                       no_mall,
                       8.637 * (18382.976995668938 / 19725.454545454544) * 1.65 / 2.4 * partition,
                       4,
                       std::make_tuple(-1.447e-5, 7.683e-3, -1.314e-3),
                       2.4 / 1.65,
                       4,
                       0};
  };

  SECTION("standard variants") {
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx90a), gfx90a);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx942), gfx942);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx950), gfx950);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1250), gfx1250);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1100), gfx1100);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1101), gfx1101);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1150), gfx1150);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1151), gfx1151);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1152), gfx1152);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1153), gfx1153);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1170), gfx117x);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1171), gfx117x);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1200), gfx1200);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::gfx1201), gfx1201);
  }

  SECTION("an unrecognized architecture has all-zero measured constants") {
    require_same_constants(hardware_t::get_arch_constants(architecture_t::Count), unknown);
    require_same_constants(hardware_t::get_arch_constants(architecture_t::Count, 0x75a8, 96),
                           unknown);
  }

  SECTION("gfx950 variants") {
    const architecture_t arch = architecture_t::gfx950;
    require_same_constants(hardware_t::get_gfx950_arch_constants(), gfx950);
    require_same_constants(hardware_t::get_gfx950_arch_constants(0x75a0), gfx950);
    require_same_constants(hardware_t::get_gfx950_arch_constants(0x75a8), gfx950_id75a8);
    require_same_constants(hardware_t::get_arch_constants(arch, 0x75a8), gfx950_id75a8);
    require_same_constants(hardware_t::get_arch_constants(arch, 0x175a8, 96), gfx950_id75a8);
    require_same_constants(hardware_t::get_arch_constants(arch, std::nullopt, 96), gfx950);
    require_same_constants(origami::constants_of(arch, variant_t::standard), gfx950);
    require_same_constants(origami::constants_of(arch, variant_t::gfx950_id75a8), gfx950_id75a8);
  }

  SECTION("gfx1250 variants") {
    const architecture_t arch = architecture_t::gfx1250;
    require_same_constants(hardware_t::get_gfx1250_arch_constants(), gfx1250);
    for (size_t n_cu : {size_t{256}, size_t{128}, size_t{0}}) {
      INFO(n_cu << " WGPs");
      require_same_constants(hardware_t::get_gfx1250_arch_constants(n_cu), gfx1250);
      require_same_constants(hardware_t::get_arch_constants(arch, std::nullopt, n_cu), gfx1250);
      require_same_constants(hardware_t::get_arch_constants(arch, 0x75a8, n_cu), gfx1250);
    }
    for (size_t n_cu : {size_t{192}, size_t{96}}) {
      INFO(n_cu << " WGPs");
      require_same_constants(hardware_t::get_gfx1250_arch_constants(n_cu), gfx1250_6xcd(n_cu));
      require_same_constants(hardware_t::get_arch_constants(arch, std::nullopt, n_cu),
                             gfx1250_6xcd(n_cu));
      require_same_constants(origami::constants_of(arch, variant_t::gfx1250_6xcd, n_cu),
                             gfx1250_6xcd(n_cu));
    }
  }

  SECTION("a variant of another architecture selects the standard row") {
    require_same_constants(
        origami::constants_of(architecture_t::gfx1201, variant_t::gfx950_id75a8, 96), gfx1201);
    require_same_constants(origami::constants_of(architecture_t::gfx950, variant_t::gfx1250_6xcd),
                           gfx950);
  }
}

TEST_CASE("Hardware: gfx1250 devices take their XCD count from the variant",
          "[hardware][variant]") {
  const auto device = [](size_t n_cu) {
    return hardware_t::get_hardware_for_arch(
        architecture_t::gfx1250, n_cu, 327680, 512 * 1024, 4 << 20, 2'400'000);
  };
  REQUIRE(device(256).NUM_XCD == 8);
  REQUIRE(device(192).NUM_XCD == 6);
  REQUIRE(device(96).NUM_XCD == 3);
  REQUIRE(device(128).NUM_XCD == 8);
}

TEST_CASE("Hardware: matrix instructions are shared by the architectures of a family",
          "[hardware][matrix_instructions]") {
  for (const auto& expected : EXPECTED_ARCHITECTURES) {
    INFO(expected.name);
    const auto& latencies = origami::matrix_instruction_latencies(expected.family);
    REQUIRE_FALSE(latencies.empty());

    // Every instruction of the family's table is valid on the architecture, with the table's
    // latency divided by the instructions a CU runs in parallel.
    const hardware_t hardware = make_device(expected.arch);
    for (const auto& [mi, latency] : latencies) {
      REQUIRE(hardware.get_mi_latency(mi.MI_M, mi.MI_N, mi.MI_K, mi.mi_input_type) ==
              latency / hardware.parallel_mi_cu);
    }
  }
  REQUIRE(origami::matrix_instruction_latencies(family_t::cdna2).size() == 26);
  REQUIRE(origami::matrix_instruction_latencies(family_t::cdna3).size() == 32);
  REQUIRE(origami::matrix_instruction_latencies(family_t::cdna4).size() == 50);
  REQUIRE(origami::matrix_instruction_latencies(family_t::cdna5).size() == 19);
  REQUIRE(origami::matrix_instruction_latencies(family_t::rdna3).size() == 4);
  REQUIRE(origami::matrix_instruction_latencies(family_t::rdna4).size() == 9);
  REQUIRE_THROWS_AS(origami::matrix_instruction_latencies(family_t::unknown), std::out_of_range);
  REQUIRE_THROWS_AS(
      make_device(architecture_t::Count).get_mi_latency(16, 16, 16, data_type_t::Half),
      std::out_of_range);
}

TEST_CASE("Hardware: matrix instruction lookups", "[hardware][matrix_instructions]") {
  using instructions_t = std::vector<std::tuple<size_t, size_t, size_t>>;

  SECTION("gfx90a") {
    const hardware_t hardware = make_device(architecture_t::gfx90a);
    REQUIRE(sorted_instructions(hardware, data_type_t::Double) ==
            instructions_t{{4, 4, 4}, {16, 16, 4}});
    REQUIRE(sorted_instructions(hardware, data_type_t::Float8).empty());
    // F16 32x32x8 takes 64 clocks on gfx90a, BF16 32.
    REQUIRE(hardware.get_mi_latency(32, 32, 8, data_type_t::Half) == 64 / 4);
    REQUIRE(hardware.get_mi_latency(32, 32, 8, data_type_t::BFloat16) == 32 / 4);
  }

  SECTION("gfx942") {
    const hardware_t hardware = make_device(architecture_t::gfx942);
    REQUIRE(sorted_instructions(hardware, data_type_t::XFloat32) ==
            instructions_t{{16, 16, 32}, {32, 32, 4}});
    REQUIRE(sorted_instructions(hardware, data_type_t::Float8_fnuz) ==
            instructions_t{{16, 16, 32}, {32, 32, 16}});
    REQUIRE(hardware.get_mi_latency(16, 16, 16, data_type_t::Half) == 16 / 4);
  }

  SECTION("gfx950") {
    const hardware_t hardware = make_device(architecture_t::gfx950);
    REQUIRE(sorted_instructions(hardware, data_type_t::BFloat16) ==
            instructions_t{{1, 1, 64}, {16, 16, 16}, {16, 16, 32}, {32, 32, 8}, {32, 32, 16}});
    REQUIRE(sorted_instructions(hardware, data_type_t::Float6) ==
            instructions_t{{16, 16, 128}, {32, 32, 64}});
    REQUIRE(hardware.get_mi_latency(16, 16, 4, data_type_t::Double) == 64 / 4);
    const origami::dim3_t recommended =
        hardware.get_recommended_matrix_instruction(data_type_t::BFloat16);
    REQUIRE(recommended.m == 16);
    REQUIRE(recommended.n == 16);
    REQUIRE(recommended.k == 32);
  }

  SECTION("gfx1250") {
    const hardware_t hardware = make_device(architecture_t::gfx1250);
    REQUIRE(sorted_instructions(hardware, data_type_t::Float4) ==
            instructions_t{{16, 16, 128}, {32, 16, 128}});
    REQUIRE(sorted_instructions(hardware, data_type_t::ComplexDouble) ==
            instructions_t{{16, 16, 4}});
    REQUIRE(hardware.get_mi_latency(16, 16, 32, data_type_t::XFloat32) == 8 * 3 / 4);
  }

  SECTION("RDNA3 and RDNA4") {
    for (architecture_t arch : {architecture_t::gfx1100, architecture_t::gfx1153}) {
      const hardware_t hardware = make_device(arch);
      REQUIRE(sorted_instructions(hardware, data_type_t::Int4) == instructions_t{{16, 16, 16}});
      REQUIRE(sorted_instructions(hardware, data_type_t::Float8).empty());
      REQUIRE(hardware.get_mi_latency(16, 16, 16, data_type_t::Half) == 32 / 4);
    }
    for (architecture_t arch : {architecture_t::gfx1170, architecture_t::gfx1201}) {
      const hardware_t hardware = make_device(arch);
      REQUIRE(sorted_instructions(hardware, data_type_t::Int4) ==
              instructions_t{{16, 16, 16}, {16, 16, 32}});
      REQUIRE(sorted_instructions(hardware, data_type_t::Float8) == instructions_t{{16, 16, 16}});
      REQUIRE(hardware.get_mi_latency(16, 16, 16, data_type_t::Half) == 16 / 4);
    }
  }

  SECTION("an instruction the architecture lacks costs a slow 32 clocks") {
    const hardware_t hardware = make_device(architecture_t::gfx950);
    REQUIRE(hardware.get_mi_latency(8, 8, 8, data_type_t::Half) == 32 / 4);
    const origami::dim3_t recommended =
        hardware.get_recommended_matrix_instruction(data_type_t::Int64);
    REQUIRE(recommended.m == 0);
    REQUIRE(recommended.n == 0);
    REQUIRE(recommended.k == 0);
  }
}
