// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <origami/hardware.hpp>
#include <type_traits>
#include <origami/heuristics.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

using hardware = origami::hardware_t;
using arch = hardware::architecture_t;

constexpr bool round_trip_models()
{
    constexpr std::string_view model_names[] = {
        "gfx90a", "gfx942", "gfx950", "gfx1200", "gfx1201", "gfx1170", "gfx1171",
        "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1250"};
    if(hardware::supported_architectures().size() != 14) return false;
    for(auto name : model_names)
    {
        auto model = hardware::arch_name_to_enum(name);
        if(model == arch::Count || hardware::arch_enum_to_name(model) != name) return false;
    }
    for(auto model : hardware::supported_architectures())
    {
        if(hardware::arch_name_to_enum(hardware::arch_enum_to_name(model)) != model) return false;
    }
    return true;
}

static_assert(std::is_same<arch, rocm::arch::architecture_t>::value, "Use shared identity type");
static_assert(round_trip_models(), "Model names and shared identity round trips");
static_assert(!hardware::is_supported_architecture(arch::gfx803), "Excluded model");
static_assert(!hardware::is_supported_architecture(static_cast<arch>(9999)), "Invalid identity");
static_assert(hardware::arch_enum_to_name(arch::gfx1250_strict) == "unknown", "Excluded identity");
static_assert(hardware::arch_name_to_enum("gfx942") == arch::gfx942, "Model mapping");
static_assert(hardware::arch_enum_to_name(arch::gfx942) == "gfx942", "Canonical name");
static_assert(hardware::arch_name_to_enum("gfx1250-strict") == arch::Count,
              "A known stepping does not imply model support");
static_assert(hardware::arch_name_to_enum("gfx942:xnack+") == arch::Count, "Exact identity");
static_assert(hardware::arch_name_to_enum("gfx94") == arch::Count, "No prefix matching");
static_assert(hardware::arch_enum_to_name(arch::Count) == "unknown", "Unknown model");
static_assert(hardware::known_architectures().size() == rocm::arch::architectures.size(),
              "Enumeration uses the common catalog");

int main()
{
    // Formocast overwrites the architecture stored in its binary constants.
    origami::Formocast simulator;
    for(auto identity : {arch::gfx942, arch::gfx950, arch::gfx1201})
        if(simulator.getHardwareConstants(identity).architecture != identity) return 1;

    origami::hand_optimized_kernel_key_t key{
        arch::gfx942, origami::data_type_t::BFloat16, origami::transpose_t::N,
        origami::transpose_t::N, 64, 64, 32};
    std::unordered_map<origami::hand_optimized_kernel_key_t, int,
                       origami::hand_optimized_kernel_key_hash> values;
    values.emplace(key, 42);
    auto copy = key;
    if(values.at(copy) != 42 || key.to_string().find("gfx942_") != 0) return 2;
    copy.arch = arch::gfx950;
    if(values.find(copy) != values.end()) return 3;

    try {
        hardware::get_hardware_for_arch(arch::gfx803, 1, 65536, 524288, 8388608, 1700000);
        return 4;
    } catch(const std::runtime_error&) {}
    return 0;
}
