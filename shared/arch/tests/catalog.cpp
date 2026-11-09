// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <rocm/arch/catalog.hpp>

static_assert(rocm::arch::architectures.size() >= 5, "sample processors present");
static_assert(rocm::arch::is_known("gfx942"), "processor identity");
static_assert(rocm::arch::is_known("gfx1250"), "base identity");
static_assert(rocm::arch::is_known("gfx1250-strict"), "distinct stepping identity");
static_assert(rocm::arch::is_known("gfx11-generic"), "generic identity");
static_assert(!rocm::arch::is_known("gfx942:xnack+"), "features are not identities");
static_assert(!rocm::arch::is_known("GFX942"), "case-sensitive identity");
static_assert(!rocm::arch::is_known("gfx94"), "exact lookup");
static_assert(!rocm::arch::is_known(nullptr), "null lookup");

constexpr bool identity_round_trips() {
    for (std::size_t i = 0; i < rocm::arch::architecture_ids.size(); ++i) {
        auto identity = rocm::arch::architecture_ids[i];
        if (rocm::arch::from_name(rocm::arch::name(identity)) != identity) return false;
    }
    return true;
}
static_assert(identity_round_trips(), "Every processor has a typed identity and canonical name");
static_assert(rocm::arch::from_name("gfx1250-strict") == rocm::arch::architecture_t::gfx1250_strict,
              "Stepping identity");
static_assert(rocm::arch::from_name("gfx11-generic") == rocm::arch::architecture_t::Count,
              "Generic targets are separate from processor identities");
static_assert(rocm::arch::from_name(nullptr) == rocm::arch::architecture_t::Count,
              "Invalid identity");
static_assert(rocm::arch::name(rocm::arch::architecture_t::Count) == nullptr, "Sentinel");
static_assert(rocm::arch::name(static_cast<rocm::arch::architecture_t>(-1)) == nullptr,
              "Invalid ordinal");

int main() {
    return 0;
}
