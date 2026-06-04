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
#include "stinkytofu/pipeline/PassBuilder.hpp"

namespace stinkytofu {

void PassBuilder::registerAtExtensionPoint(PipelineExtensionPoint EP, ExtensionCallback CB) {
    callbacks_[static_cast<int>(EP)].push_back(std::move(CB));
}

void PassBuilder::applyExtensionPoint(PipelineExtensionPoint EP, PassManager& PM,
                                      StinkyAsmModule& module) const {
    auto it = callbacks_.find(static_cast<int>(EP));
    if (it == callbacks_.end()) return;
    for (const auto& cb : it->second) {
        cb(PM, module);
    }
}

struct PassBuilder::FactoryRegistry {
    std::unordered_map<std::string, std::function<std::unique_ptr<Pass>()>> factories;
    std::mutex mu;
};

PassBuilder::FactoryRegistry& PassBuilder::getFactoryRegistry() {
    static FactoryRegistry registry;
    return registry;
}

void PassBuilder::registerNamedPassFactory(const std::string& name,
                                           std::function<std::unique_ptr<Pass>()> factory) {
    auto& reg = getFactoryRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.factories[name] = std::move(factory);
}

std::unique_ptr<Pass> PassBuilder::createPassByName(const std::string& name) {
    auto& reg = getFactoryRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    auto it = reg.factories.find(name);
    if (it == reg.factories.end()) {
        return nullptr;
    }
    return it->second();
}

}  // namespace stinkytofu
