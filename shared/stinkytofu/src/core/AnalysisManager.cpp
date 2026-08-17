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
#include "stinkytofu/core/AnalysisManager.hpp"

#include <iostream>

namespace stinkytofu {
void AnalysisManager::invalidate(Function& F, const PreservedAnalyses& PA) {
    if (PA.areAllPreserved()) return;

    for (auto it = results_.begin(); it != results_.end();) {
        if (it->second->invalidate(F, PA)) {
            if (debugLogging_) {
                auto fit = factories_.find(it->first);
                if (fit != factories_.end()) debugLog("Evicting", fit->second.name);
            }
            it = results_.erase(it);
        } else {
            ++it;
        }
    }
}

void AnalysisManager::debugLog(const char* action, const char* analysisName) {
    std::cerr << "[StinkyTofu] Analysis: " << action << " " << analysisName << "\n";
}

void AnalysisManager::dumpCacheState() const {
    std::cerr << "[StinkyTofu] Analysis: Cache state: ";
    if (results_.empty()) {
        std::cerr << "empty\n";
        return;
    }
    std::cerr << results_.size() << " cached\n";
    for (const auto& [key, result] : results_) {
        auto fit = factories_.find(key);
        const char* name = (fit != factories_.end()) ? fit->second.name : "?";
        std::cerr << "[StinkyTofu] Analysis:   " << name << "\n";
    }
}

}  // namespace stinkytofu
