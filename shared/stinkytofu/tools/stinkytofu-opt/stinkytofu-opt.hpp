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
#pragma once

#include <memory>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"

using namespace stinkytofu;

// The pass catalog lives in the library (PassCatalog). This header keeps the
// driver's debug-print and feature-config helpers.
/**
 * Create default DebugPrintInstrumentation for stinkytofu-opt.
 */
std::shared_ptr<stinkytofu::PassInstrumentation> createDebugPrintInstrumentation() {
    auto streams = std::make_shared<stinkytofu::DebugOutputStreams>();
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);
    debugConfig->setPrintAfterAll(true);
    debugConfig->setDumpStreamBefore(streams->getOrCreate("before.txt"));
    debugConfig->setDumpStreamAfter(streams->getOrCreate("after.txt"));
    return std::make_shared<stinkytofu::DebugPrintInstrumentation>(std::move(debugConfig));
}

/**
 * Get default PassFeatureConfig configuration.
 */
stinkytofu::PassFeatureConfig getPassFeatureConfig() {
    stinkytofu::PassFeatureConfig config;
    config.loopConfig.unrollGemm = true;
    config.dagFeatures.distributeGlobalRead = true;
    return config;
}
