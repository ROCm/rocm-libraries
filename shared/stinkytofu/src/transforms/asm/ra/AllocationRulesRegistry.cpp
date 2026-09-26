/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace stinkytofu {
namespace {

/// Same "gfxMNS" spelling BackendRegistry uses, copied rather than shared:
/// register allocation sits below the pipeline layer, so calling into
/// BackendRegistry here would invert a layer for four lines of string building.
std::string archKey(const std::array<int, 3>& arch) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    const int stepping = (arch[2] >= 0 && arch[2] <= 15) ? arch[2] : 0;
    return "gfx" + std::to_string(arch[0]) + std::to_string(arch[1]) + kHexDigits[stepping];
}

}  // namespace

struct AllocationRulesRegistry::Registry {
    std::unordered_map<std::string, Factory> factories;
};

AllocationRulesRegistry::Registry& AllocationRulesRegistry::registry() {
    static Registry instance;
    return instance;
}

void AllocationRulesRegistry::setArch(const std::array<int, 3>& arch, Factory factory) {
    registry().factories[archKey(arch)] = std::move(factory);
}

AllocationRules AllocationRulesRegistry::forArch(const std::array<int, 3>& arch,
                                                 const AsmCapsConfig& caps) {
    const auto& factories = registry().factories;
    auto it = factories.find(archKey(arch));
    if (it == factories.end() || !it->second) return {};
    return it->second(caps);
}

bool AllocationRulesRegistry::hasArch(const std::array<int, 3>& arch) {
    const auto& factories = registry().factories;
    return factories.find(archKey(arch)) != factories.end();
}

void AllocationRulesRegistry::clearArch(const std::array<int, 3>& arch) {
    registry().factories.erase(archKey(arch));
}

std::vector<std::string> AllocationRulesRegistry::archKeys() {
    std::vector<std::string> keys;
    keys.reserve(registry().factories.size());
    for (const auto& kv : registry().factories) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Anchor declarations -- one per per-arch rules TU. Adding an architecture's
// rules? Add its anchor here, the way registerAllBackends() does for pipelines.
void anchorGfx1250AllocationRules();

// The anchors are empty and each TU installs its own factory from a static
// registrar, so calling this registers nothing directly. It earns its keep by
// naming every anchor: that reference is what makes a static archive hand over
// those TUs, and dropping it would take their registrars with it.
void AllocationRulesRegistry::registerAll() {
    anchorGfx1250AllocationRules();
}

}  // namespace stinkytofu
