/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/pipeline/BackendRegistry.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>

#include "stinkytofu/hardware/ArchHelper.hpp"

namespace stinkytofu {
struct BackendRegistry::Registry {
    std::unordered_map<std::string, ArchPipeline> pipelines;
};

BackendRegistry::Registry& BackendRegistry::getRegistry() {
    static Registry registry;
    return registry;
}

void BackendRegistry::setArchPipeline(const std::array<int, 3>& arch, ArchPipeline pipeline) {
    auto& reg = getRegistry();
    reg.pipelines[makeArchKey(arch)] = std::move(pipeline);
}

const BackendRegistry::ArchPipeline* BackendRegistry::getArchPipeline(
    const std::array<int, 3>& arch) {
    auto& reg = getRegistry();
    auto it = reg.pipelines.find(makeArchKey(arch));
    if (it != reg.pipelines.end()) {
        return &it->second;
    }
    return nullptr;
}

void BackendRegistry::clear() {
    auto& reg = getRegistry();
    reg.pipelines.clear();
}

void BackendRegistry::clearArch(const std::array<int, 3>& arch) {
    auto& reg = getRegistry();
    reg.pipelines.erase(makeArchKey(arch));
}

std::vector<std::string> BackendRegistry::getRegisteredArchKeys() {
    auto& reg = getRegistry();
    std::vector<std::string> keys;
    keys.reserve(reg.pipelines.size());
    for (const auto& kv : reg.pipelines) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::string BackendRegistry::makeArchKey(const std::array<int, 3>& arch) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    int stepping = (arch[2] >= 0 && arch[2] <= 15) ? arch[2] : 0;
    return "gfx" + std::to_string(arch[0]) + std::to_string(arch[1]) + kHexDigits[stepping];
}

bool BackendRegistry::parseArchKey(const std::string& archStr, std::array<int, 3>& out) {
    // Format: gfx<major><minor><stepping>
    //   stepping = last char, single hex digit (0-9, a-f); e.g. gfx90a → stepping=10
    //   minor    = second-to-last char, single decimal digit
    //   major    = remaining digits between "gfx" and minor
    if (archStr.size() < 6 || archStr.substr(0, 3) != "gfx") return false;
    std::string suffix = archStr.substr(3);

    char steppingChar = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix.back())));
    if (!std::isxdigit(static_cast<unsigned char>(steppingChar))) return false;
    int stepping = (steppingChar >= 'a') ? (steppingChar - 'a' + 10) : (steppingChar - '0');

    char minorChar = suffix[suffix.size() - 2];
    if (!std::isdigit(static_cast<unsigned char>(minorChar))) return false;
    int minor = minorChar - '0';

    std::string majorStr = suffix.substr(0, suffix.size() - 2);
    if (majorStr.empty()) return false;
    for (unsigned char c : majorStr)
        if (!std::isdigit(c)) return false;
    int major = std::stoi(majorStr);

    out = {major, minor, stepping};
    return true;
}

// Anchor declarations — one per backend TU. Adding a new backend? Add its anchor here.
void anchorGfx1250Backend();

void BackendRegistry::registerAllBackends() {
    anchorGfx1250Backend();
}

}  // namespace stinkytofu
