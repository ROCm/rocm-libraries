/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/**
 * @brief Configuration policy for what to wait for at barriers
 *
 * This allows fine-grained control over what memory operations
 * must complete before barriers.
 */
struct BarrierWaitPolicy {
    // What to wait for before barriers
    bool waitDSRead = true;       // Wait for LDS loads
    bool waitDSWrite = true;      // Wait for LDS stores
    bool waitGlobalRead = true;   // Wait for global memory loads
    bool waitGlobalWrite = true;  // Wait for global memory stores
    bool waitTensorLoad = true;   // Wait for tensor loads
    bool waitAtomics = true;      // Wait for atomic operations

    // Conservative mode: wait for everything
    static BarrierWaitPolicy conservative() {
        BarrierWaitPolicy policy;
        policy.waitDSRead = true;
        policy.waitDSWrite = true;
        policy.waitGlobalRead = true;
        policy.waitGlobalWrite = true;
        policy.waitTensorLoad = true;
        policy.waitAtomics = true;
        return policy;
    }

    // Minimal mode: only wait for LDS operations
    static BarrierWaitPolicy minimal() {
        BarrierWaitPolicy policy;
        policy.waitDSRead = true;
        policy.waitDSWrite = false;
        policy.waitGlobalRead = false;
        policy.waitGlobalWrite = false;
        policy.waitTensorLoad = false;
        policy.waitAtomics = false;
        return policy;
    }

    // Unroll loop mode: DS operations and tensor loads
    static BarrierWaitPolicy unrollLoop() {
        BarrierWaitPolicy policy;
        policy.waitDSRead = false;
        policy.waitDSWrite = true;
        policy.waitGlobalRead = false;
        policy.waitGlobalWrite = false;
        policy.waitTensorLoad = true;
        policy.waitAtomics = false;
        return policy;
    }

    // Custom builder pattern
    BarrierWaitPolicy& setDSRead(bool wait) {
        waitDSRead = wait;
        return *this;
    }
    BarrierWaitPolicy& setDSWrite(bool wait) {
        waitDSWrite = wait;
        return *this;
    }
    BarrierWaitPolicy& setGlobalRead(bool wait) {
        waitGlobalRead = wait;
        return *this;
    }
    BarrierWaitPolicy& setGlobalWrite(bool wait) {
        waitGlobalWrite = wait;
        return *this;
    }
    BarrierWaitPolicy& setTensorLoad(bool wait) {
        waitTensorLoad = wait;
        return *this;
    }
    BarrierWaitPolicy& setAtomics(bool wait) {
        waitAtomics = wait;
        return *this;
    }
};

/**
 * @brief Configuration policy for dependency tracking
 */
struct DependencyTrackingPolicy {
    bool trackLoadDependencies = true;   // Track load → use dependencies
    bool trackStoreDependencies = true;  // Track store → store/load ordering
    bool trackCrossBoundary = true;      // Track dependencies across loop boundaries
    bool mergeAdjacentWaitCnt = true;    // Merge adjacent waitcnt instructions

    static DependencyTrackingPolicy standard() {
        return DependencyTrackingPolicy{};
    }

    static DependencyTrackingPolicy loadsOnly() {
        DependencyTrackingPolicy policy;
        policy.trackStoreDependencies = false;
        return policy;
    }
};

/**
 * @brief Complete configuration for WaitCnt insertion
 */
struct WaitCntConfig {
    BarrierWaitPolicy barrierPolicy;
    DependencyTrackingPolicy dependencyPolicy;

    // Factory methods for common configurations
    static WaitCntConfig standard() {
        return WaitCntConfig{BarrierWaitPolicy(), DependencyTrackingPolicy::standard()};
    }

    static WaitCntConfig conservative() {
        return WaitCntConfig{BarrierWaitPolicy::conservative(),
                             DependencyTrackingPolicy::standard()};
    }

    static WaitCntConfig minimal() {
        return WaitCntConfig{BarrierWaitPolicy::minimal(), DependencyTrackingPolicy::loadsOnly()};
    }

    static WaitCntConfig unrollLoop() {
        return WaitCntConfig{BarrierWaitPolicy::unrollLoop(),
                             DependencyTrackingPolicy::loadsOnly()};
    }

    // Custom builder
    static WaitCntConfig custom() {
        return WaitCntConfig{};
    }
};

// Factory functions for different configurations
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyUnrollWaitCntPass();
std::unique_ptr<Pass> createStinkyConservativeWaitCntPass();
std::unique_ptr<Pass> createStinkyMinimalWaitCntPass();
std::unique_ptr<Pass> createStinkyCustomWaitCntPass(const WaitCntConfig& config);

}  // namespace stinkytofu
