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

std::string BackendRegistry::makeArchKey(const std::array<int, 3>& arch) {
    return "gfx" + std::to_string(arch[0] * 100 + arch[1] * 10 + arch[2]);
}

}  // namespace stinkytofu
