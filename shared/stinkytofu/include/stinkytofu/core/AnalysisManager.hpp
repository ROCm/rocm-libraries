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

#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace stinkytofu {
class Function;

// -----------------------------------------------------------------------
// AnalysisKey — unique identifier for an analysis pass.
//
// Each analysis declares:  static AnalysisKey Key;
// The ADDRESS of Key is the unique ID.
// -----------------------------------------------------------------------
struct alignas(8) AnalysisKey {};

// -----------------------------------------------------------------------
// PreservedAnalyses — returned by transform passes to declare what
// analyses they did NOT invalidate.
//
// Default: NOTHING preserved (everything gets evicted).
// -----------------------------------------------------------------------
class PreservedAnalyses {
    std::vector<AnalysisKey*> keys_;
    bool all_ = false;

   public:
    static PreservedAnalyses all() {
        PreservedAnalyses PA;
        PA.all_ = true;
        return PA;
    }

    static PreservedAnalyses none() {
        return {};
    }

    template <typename AnalysisT>
    PreservedAnalyses& preserve() {
        return preserve(AnalysisT::ID());
    }

    PreservedAnalyses& preserve(AnalysisKey* key) {
        keys_.push_back(key);
        return *this;
    }

    bool isPreserved(AnalysisKey* key) const {
        if (all_) return true;
        for (auto* k : keys_)
            if (k == key) return true;
        return false;
    }

    template <typename AnalysisT>
    bool isPreserved() const {
        return isPreserved(AnalysisT::ID());
    }

    bool areAllPreserved() const {
        return all_;
    }

    /// An analysis stays preserved only if both this and other preserve it.
    void intersect(const PreservedAnalyses& other) {
        if (other.all_) return;  // this & all = this (unchanged)
        if (all_) {
            // all & other = other
            all_ = false;
            keys_ = other.keys_;
            return;
        }
        // Keep only keys present in both
        std::vector<AnalysisKey*> kept;
        for (auto* k : keys_) {
            if (other.isPreserved(k)) kept.push_back(k);
        }
        keys_ = std::move(kept);
    }
};

// -----------------------------------------------------------------------
// AnalysisResultConcept / AnalysisResultModel
//
// Type-erased base and wrapper for cached analysis results.
// AnalysisManager stores unique_ptr<AnalysisResultConcept> in its cache.
// AnalysisResultModel<AnalysisT, ResultT> wraps a concrete result type
// and provides default invalidation (evict if not in PreservedAnalyses).
// -----------------------------------------------------------------------
struct AnalysisResultConcept {
    virtual ~AnalysisResultConcept() = default;

    /// Returns true if this result is invalid and should be evicted.
    virtual bool invalidate(Function& F, const PreservedAnalyses& PA) = 0;
};

/// Wraps a concrete result type. Evicted if not in PreservedAnalyses.
template <typename AnalysisT, typename ResultT>
struct AnalysisResultModel : AnalysisResultConcept {
    ResultT Result;

    explicit AnalysisResultModel(ResultT R) : Result(std::move(R)) {}

    bool invalidate(Function&, const PreservedAnalyses& PA) override {
        return !PA.isPreserved(AnalysisT::ID());
    }
};

// -----------------------------------------------------------------------
// AnalysisManager — lazy, caching manager for analysis results.
//
// Each analysis pass is a struct satisfying:
//
//   struct SomeAnalysis {
//       static AnalysisKey Key;
//       static AnalysisKey* ID() { return &Key; }
//       static const char* name() { return "SomeAnalysis"; }
//       using Result = SomeResultType;
//       static Result run(Function& F, AnalysisManager& AM);
//   };
// -----------------------------------------------------------------------
class AnalysisManager {
   public:
    AnalysisManager() = default;
    ~AnalysisManager() = default;

    AnalysisManager(const AnalysisManager&) = delete;
    AnalysisManager& operator=(const AnalysisManager&) = delete;

    AnalysisManager(AnalysisManager&&) = default;
    AnalysisManager& operator=(AnalysisManager&&) = default;

    /// Enable debug logging (cache hits, misses, evictions).
    void setDebugLogging(bool v) {
        debugLogging_ = v;
    }

    /// Register an analysis factory. Called once at pipeline setup.
    template <typename AnalysisT>
    void registerPass() {
        auto factory = [](Function& F,
                          AnalysisManager& AM) -> std::unique_ptr<AnalysisResultConcept> {
            using ModelT = AnalysisResultModel<AnalysisT, typename AnalysisT::Result>;
            return std::make_unique<ModelT>(AnalysisT::run(F, AM));
        };
        factories_[AnalysisT::ID()] = {std::move(factory), AnalysisT::name()};
    }

    /// Get or compute an analysis result (lazy, cached).
    template <typename AnalysisT>
    const typename AnalysisT::Result& getResult(Function& F) {
        auto* key = AnalysisT::ID();
        auto it = results_.find(key);
        if (it == results_.end()) {
            auto fit = factories_.find(key);
            assert(fit != factories_.end() && "Analysis not registered");
            if (debugLogging_) debugLog("Computing", fit->second.name);
            auto result = fit->second.factory(F, *this);
            auto [ins, ok] = results_.emplace(key, std::move(result));
            it = ins;
        } else if (debugLogging_) {
            debugLog("Cache hit", factories_[key].name);
        }
        using ModelT = AnalysisResultModel<AnalysisT, typename AnalysisT::Result>;
        return static_cast<ModelT&>(*it->second).Result;
    }

    /// Get cached result only. Returns nullptr if not computed.
    template <typename AnalysisT>
    const typename AnalysisT::Result* getCachedResult() const {
        auto it = results_.find(AnalysisT::ID());
        if (it == results_.end()) return nullptr;
        using ModelT = AnalysisResultModel<AnalysisT, typename AnalysisT::Result>;
        return &static_cast<const ModelT&>(*it->second).Result;
    }

    /// Evict invalidated results. Called by PM after each transform pass.
    void invalidate(Function& F, const PreservedAnalyses& PA);

    /// Clear all cached results (new Function).
    void clear() {
        results_.clear();
    }

    /// Dump currently cached analyses to stderr.
    void dumpCacheState() const;

   private:
    using Factory =
        std::function<std::unique_ptr<AnalysisResultConcept>(Function&, AnalysisManager&)>;

    struct FactoryEntry {
        Factory factory;
        const char* name;
    };

    void debugLog(const char* action, const char* analysisName);

    std::unordered_map<AnalysisKey*, FactoryEntry> factories_;
    std::unordered_map<AnalysisKey*, std::unique_ptr<AnalysisResultConcept>> results_;
    bool debugLogging_ = false;
};

}  // namespace stinkytofu
