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

#include <functional>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassInstrumentation.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
class PassContext;

//----------------------------------------------------------------------
// BasicBlock Filter Support
//----------------------------------------------------------------------

// BasicBlock filter predicate type
using BasicBlockFilter = std::function<bool(const BasicBlock&)>;

// Filter builder for easy construction of BasicBlock filters
class BasicBlockFilterBuilder {
   public:
    // Filter by label prefix
    static BasicBlockFilter byLabelPrefix(const std::string& prefix) {
        return [prefix](const BasicBlock& bb) { return bb.getLabel().rfind(prefix, 0) == 0; };
    }

    // Filter by exact label names
    static BasicBlockFilter byLabels(const std::set<std::string>& labels) {
        return [labels](const BasicBlock& bb) { return labels.count(bb.getLabel()) > 0; };
    }

    // Filter by custom predicate
    static BasicBlockFilter byPredicate(std::function<bool(const BasicBlock&)> pred) {
        return pred;
    }

    // Combine filters with AND logic
    static BasicBlockFilter combine(const BasicBlockFilter& f1, const BasicBlockFilter& f2) {
        return [f1, f2](const BasicBlock& bb) { return f1(bb) && f2(bb); };
    }

    // Exclude certain BasicBlocks (NOT logic)
    static BasicBlockFilter exclude(const BasicBlockFilter& filter) {
        return [filter](const BasicBlock& bb) { return !filter(bb); };
    }

    // Process all BasicBlocks (default)
    static BasicBlockFilter all() {
        return [](const BasicBlock&) { return true; };
    }
};

//----------------------------------------------------------------------
// Pass Infrastructure
//----------------------------------------------------------------------

// Base class for all passes.
//
// A pass operates on a Function (which contains BasicBlocks with IRLists)
// and performs either analysis or transformation.
class Pass {
   public:
    using ID = const void*;
    using PassID = ID;

   public:
    virtual ~Pass() = default;

    virtual ID getPassID() const = 0;
    virtual const char* getName() const = 0;

    virtual PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) = 0;
};

// The PassContext serves as the central state and resource manager for
// the StinkyTofu pass execution framework.
//
// It does not own a Function; the Function is passed to PassManager::run(Function&).
class PassContext {
    GemmTileConfig gemmConfig;
    PassFeatureConfig passConfig;
    AsmCapsConfig asmCapsConfig;
    uint32_t wavefrontSize = 0;  ///< Computed from gemmConfig.arch

    // Global BasicBlock filter applied to all StinkyInstPass instances.
    // By default, all BasicBlocks are processed.
    BasicBlockFilter globalBBFilter;

   public:
    PassContext() : globalBBFilter(BasicBlockFilterBuilder::all()) {}

    void setGemmTileConfig(const GemmTileConfig& config);

    const GemmTileConfig& getGemmTileConfig() const {
        return gemmConfig;
    }

    /// Get wavefront size (derived from architecture, not user-configurable)
    uint32_t getWavefrontSize() const {
        return wavefrontSize;
    }

    void setPassFeatureConfig(const PassFeatureConfig& config) {
        passConfig = config;
    }

    const PassFeatureConfig& getPassFeatureConfig() const {
        return passConfig;
    }

    void setAsmCapsConfig(const AsmCapsConfig& config) {
        asmCapsConfig = config;
    }

    const AsmCapsConfig& getAsmCapsConfig() const {
        return asmCapsConfig;
    }

    /// Set global BasicBlock filter for all StinkyInstPass instances.
    /// This filter determines which BasicBlocks should be processed by passes.
    void setBasicBlockFilter(BasicBlockFilter filter) {
        globalBBFilter = std::move(filter);
    }

    /// Check if a BasicBlock should be processed according to the global filter.
    bool shouldProcessBasicBlock(const BasicBlock& bb) const {
        return globalBBFilter(bb);
    }
};

bool isDebugOnlyEnabled(const char* TYPE);

#define DEBUG_WITH_TYPE(TYPE, X)        \
    do {                                \
        if (isDebugOnlyEnabled(TYPE)) { \
            X;                          \
        }                               \
    } while (0)

// PASS_DEBUG is used by each pass to print its internal debug information.
// A pass can be enabled for debug output by adding its DEBUG_TYPE name to
// PassManagerDebugConfig::addDebugOnly
#define PASS_DEBUG(X) DEBUG_WITH_TYPE(DEBUG_TYPE, X)

// Configuration for PassManager debug output.
//
// Users can configure which passes to print the IR before/after running,
// and where to dump the output (stdout or file).
//
// Users can also configure a global debug-only list of passes to print.
class STINKYTOFU_EXPORT PassManagerDebugConfig final {
    unsigned printAfterAll : 1;
    unsigned printBeforeAll : 1;
    unsigned dumpInitialIR : 1;
    unsigned printPassNames : 1;

    std::shared_ptr<std::ostream> dumpStreamBefore;
    std::shared_ptr<std::ostream> dumpStreamAfter;

    std::unordered_set<std::string> onlyPrintBefore;
    std::unordered_set<std::string> onlyPrintAfter;

   public:
    PassManagerDebugConfig();
    ~PassManagerDebugConfig() = default;

    void setPrintAfterAll(bool v = true);
    void setPrintBeforeAll(bool v = true);
    void setDumpInitialIR(bool v = true);
    void setPrintPassNames(bool v = true);
    void addOnlyPrintBefore(const std::string& passName);
    void addOnlyPrintAfter(const std::string& passName);
    void setDumpStreamBefore(std::shared_ptr<std::ostream> stream);
    void setDumpStreamAfter(std::shared_ptr<std::ostream> stream);

    bool shouldDumpInitialIR() const {
        return dumpInitialIR;
    }

    bool shouldPrintPassName() const {
        return printPassNames;
    }

    bool shouldPrintBefore(const std::string& passName) const;
    bool shouldPrintAfter(const std::string& passName) const;

    std::ostream& getOutputStreamInBefore() const;
    std::ostream& getOutputStreamInAfter() const;

   public:
    // Note: The debug only functions will use internal global static
    // variables, that's why they are static.
    static void addDebugOnly(const std::string& passName);
    static void clearDebugOnly();
};

// PassManager manages a list of passes to run on a module.
//
// Note: The module is using physical registers, so it is no longer in SSA form.
//
// Note: Even though StinkyInstruction is currently the only IR
//       type, there could be more IR types (levels) like MLIR in
//       the future.
class STINKYTOFU_EXPORT PassManager {
   public:
    /// Run all passes on the given Function. PassContext is used for config and analysis.
    void run(Function& F);

    // Add a pass to the list of passes to run.
    // The passes will be run in the order they are added.
    //
    // By default, the PassManager does not have any pass to run.
    void addPass(std::unique_ptr<Pass> pass) {
        passes.push_back(std::move(pass));
    }

    /// Register a PassInstrumentation callback to observe pass execution
    void addInstrumentation(std::shared_ptr<PassInstrumentation> inst) {
        instrumentations.push_back(std::move(inst));
    }

    PassManager() = default;
    ~PassManager() = default;

    PassManager(PassManager&&) noexcept = default;
    PassManager& operator=(PassManager&&) noexcept = default;

   public:
    // Set GEMM tile configuration (wavefront size automatically determined from architecture)
    void setGemmTileConfig(const GemmTileConfig& config);

    // Deprecated: Use setGemmTileConfig instead
    void setKernelConfig(std::array<int, 3> arch, uint32_t ta0, uint32_t tb0, uint32_t tm0,
                         uint32_t nGRA, uint32_t nGRB, uint32_t nGRM, uint32_t numWaves);

    // Set pass feature configuration
    void setPassFeatureConfig(const PassFeatureConfig& config);

    // Set assembler capability configuration (propagated from rocisa asmCaps)
    void setAsmCapsConfig(const AsmCapsConfig& config);

    void setBasicBlockFilter(BasicBlockFilter filter) {
        passCtx.setBasicBlockFilter(filter);
    }

    // Get access to the PassContext (for advanced usage)
    PassContext& getPassContext() {
        return passCtx;
    }

    const PassContext& getPassContext() const {
        return passCtx;
    }

    AnalysisManager& getAnalysisManager() {
        return analysisManager;
    }

   protected:
    PassContext passCtx;
    AnalysisManager analysisManager;

    // List of passes to run.
    //
    // Users can add passes to this list through 'addPass' method.
    // The passes will be run in the order they are added.
    std::vector<std::unique_ptr<Pass>> passes;

    std::vector<std::shared_ptr<PassInstrumentation>> instrumentations;
};
}  // namespace stinkytofu
