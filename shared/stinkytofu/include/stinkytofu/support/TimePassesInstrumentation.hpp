// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassInstrumentation.hpp"

namespace stinkytofu {

/// Measures how long each pass runs and prints an LLVM `-time-passes` style
/// report, so a slow pipeline can be attributed to individual passes.
///
/// One instance observes every PassManager of a pipeline run, the module-level
/// one included, so nested managers (module pass -> adaptor -> function pass)
/// all report into the same table. Nesting is tracked with a stack: a pass is
/// charged its inclusive time minus whatever its nested passes took, which makes
/// the self column sum to the pipeline's wall time.
class STINKYTOFU_EXPORT TimePassesInstrumentation final : public PassInstrumentation {
   public:
    void beforePass(const std::string& passName, Function&, PassContext&) override {
        enter(passName);
    }

    void afterPass(const std::string&, Function&, PassContext&) override {
        leave();
    }

    void beforeModulePass(const std::string& passName, StinkyAsmModule&, PassContext&) override {
        enter(passName);
    }

    void afterModulePass(const std::string&, StinkyAsmModule&, PassContext&) override {
        leave();
    }

    /// Write one report block to @p os, passes ordered by descending self time.
    /// @p label identifies the kernel it belongs to.
    void report(std::ostream& os, const std::string& label) const;

   private:
    using Clock = std::chrono::steady_clock;

    struct PassTime {
        double self = 0.0;   ///< seconds spent in the pass itself
        double total = 0.0;  ///< seconds including nested passes
        unsigned runs = 0;
    };

    /// A pass that has been entered but not yet left.
    struct Running {
        std::string name;
        Clock::time_point start;
        double nested = 0.0;
    };

    void enter(const std::string& passName);
    void leave();

    std::vector<Running> running;
    std::unordered_map<std::string, PassTime> passTimes;
};

/// The observer collecting times for the pipeline run in progress on this
/// thread, or null when timing is off. Pipeline builders install it on every
/// PassManager they create — see configureStandardInstrumentations().
STINKYTOFU_EXPORT std::shared_ptr<TimePassesInstrumentation> getActiveTimePasses();

/// Collects per-pass times for as long as it is in scope, then prints the report
/// to @p os. Constructing it with @p enable false costs nothing and leaves
/// getActiveTimePasses() null, so a driver can open one unconditionally.
///
/// A session opened while another is already active on the thread joins it
/// instead of starting its own, so an outer driver and Backend never print two
/// overlapping reports for the same run.
class STINKYTOFU_EXPORT TimePassesSession {
   public:
    TimePassesSession(bool enable, std::string label, std::ostream& os);
    ~TimePassesSession();

    TimePassesSession(const TimePassesSession&) = delete;
    TimePassesSession& operator=(const TimePassesSession&) = delete;

   private:
    std::shared_ptr<TimePassesInstrumentation> timer;
    std::string label;
    std::ostream& os;
};

}  // namespace stinkytofu
