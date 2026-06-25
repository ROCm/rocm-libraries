// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// timing.hpp -- lightweight, opt-in instrumentation for the two latency-bearing
// stages the provider owns at runtime:
//
//   * comgr compile  (.ll -> HSACO), happening in the PlanBuilder when the
//     C-JIT path or an .ll-only artifact is materialized + ensure_compiled().
//   * kernel launch   (kernarg pack + hipModuleLaunchKernel + stream sync),
//     happening in Plan::execute().
//
// Both are gated by the env var CK_DSL_TIME=1. When off there is zero overhead
// beyond a single getenv()-backed bool read (cached). The two scoped timers
// emit one stderr line each:
//
//   [ck_dsl_time] <op> compileMs=<float>
//   [ck_dsl_time] <op> launchUs=<float>
//
// Header-only so it can be used from both the runtime and the provider TUs
// without adding a link dependency.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace ck_dsl {

// Cached read of CK_DSL_TIME (1/true/on enable). Evaluated once.
inline bool timing_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CK_DSL_TIME");
        if (!v) return false;
        return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
    }();
    return on;
}

// Scoped wall-clock timer. On destruction, if timing is enabled, prints
//   [ck_dsl_time] <label> <unit_name>=<value>
// where value is the elapsed time converted by `to_unit` (ms or us).
class ScopedTimer {
   public:
    enum class Unit { Ms, Us };
    ScopedTimer(std::string label, Unit unit)
        : label_(std::move(label)), unit_(unit), enabled_(timing_enabled()) {
        if (enabled_) start_ = std::chrono::steady_clock::now();
    }
    ~ScopedTimer() {
        if (!enabled_) return;
        auto end = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start_).count();
        if (unit_ == Unit::Ms)
            std::fprintf(stderr, "[ck_dsl_time] %s compileMs=%.3f\n", label_.c_str(), ns / 1e6);
        else
            std::fprintf(stderr, "[ck_dsl_time] %s launchUs=%.3f\n", label_.c_str(), ns / 1e3);
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

   private:
    std::string label_;
    Unit unit_;
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace ck_dsl
