// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <chrono>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace raceemulator {

/// General-purpose profiler that accumulates wall-clock time per named scope.
///
/// Designed for profiling the race-emulator, but usable in any context. Each
/// scope is identified by a string key (e.g., an instruction mnemonic like
/// "ds_load_b32", or a function name like "initializeForRun"). The profiler
/// tracks cumulative time and invocation count per key.
///
/// The profiler is disabled by default. When disabled, scopedStopwatch()
/// returns a no-op ScopedStopwatch with near-zero overhead (~1ns predicted
/// branch, no chrono calls). When enabled, each scope entry/exit costs ~50ns
/// (two chrono::now() calls + one hash map lookup).
///
/// Scopes may nest: when a new scope starts while one is already active, the
/// outer scope pauses and resumes when the inner scope ends. Time is
/// partitioned -- each instant is attributed to exactly one scope. This ensures
/// that accounted time never exceeds wall-clock time and that the report
/// percentages are meaningful.
///
/// Scopes need not follow strict LIFO destruction order. If a middle scope is
/// destroyed before an inner scope, it is silently removed (its time was
/// already flushed when it was paused). Only the most recently started scope
/// accumulates time at any given moment.
///
/// The report() method prints a table sorted by time (descending) with
/// columns for scope name, cumulative time, invocation count, average time
/// per invocation, and percentage of wall-clock time. It also shows
/// accounted vs unaccounted time to reveal uninstrumented gaps.
///
/// Example:
///   Profiler profiler(true);
///   {
///     auto sw = profiler.scopedStopwatch("myFunction");
///     // ... work ...
///   }
///   profiler.report(std::cerr);
///
/// Example with lazy key (avoids key computation when profiling is disabled):
///   auto sw = profiler.scopedStopwatch([&]() -> std::string_view {
///     return extractMnemonic(line);
///   });
///
/// Not thread-safe. For multi-threaded use, give each thread its own Profiler
/// and merge results afterwards.
class Profiler {
  struct Entry {
    double totalSeconds = 0.0;
    uint64_t count = 0;
    int depth = -1; // nesting depth at first entry (-1 = unset)
  };

public:
  /// Construct a profiler. When disabled (default), scopedStopwatch() returns
  /// no-op instances with near-zero overhead.
  explicit Profiler(bool enabled = false) : enabled(enabled) {}

  /// RAII stopwatch. Construct via Profiler::scopedStopwatch() to time a
  /// scope, or default-construct for a no-op instance (zero overhead).
  /// Nesting is supported: creating a new stopwatch pauses the current one.
  class ScopedStopwatch {
  public:
    /// No-op stopwatch -- destructor does nothing.
    ScopedStopwatch() = default;

    ~ScopedStopwatch();

    ScopedStopwatch(ScopedStopwatch &&other) noexcept;
    ScopedStopwatch &operator=(ScopedStopwatch &&other) noexcept;

    ScopedStopwatch(const ScopedStopwatch &) = delete;
    ScopedStopwatch &operator=(const ScopedStopwatch &) = delete;

    /// Active stopwatch -- pauses any current scope and starts timing.
    /// Only Profiler can supply an Entry* (Entry is private to Profiler).
    ScopedStopwatch(Profiler *profiler, Entry *entry);

  private:
    /// Remove this stopwatch from the profiler's scope list and handle
    /// time flushing / resumption of the previous scope.
    void retire();

    Profiler *profiler = nullptr;
    Entry *entry = nullptr;
    uint32_t scopeId = 0;
  };

  /// Start timing a named scope. Returns a no-op if profiling is disabled.
  /// Note: the key argument is evaluated even when disabled. For keys that
  /// are expensive to compute, use the lazy-key overload instead.
  [[nodiscard]] ScopedStopwatch scopedStopwatch(std::string_view key);

  /// Lazy-key overload: the callable is only invoked when profiling is enabled,
  /// avoiding key-computation overhead when profiling is off.
  template <typename KeyFn,
            typename = std::enable_if_t<std::is_invocable_v<KeyFn>>>
  [[nodiscard]] ScopedStopwatch scopedStopwatch(KeyFn &&keyFn) {
    if (!enabled) {
      return {};
    }
    return ScopedStopwatch(this, &entries[std::string(keyFn())]);
  }

  bool isEnabled() const { return enabled; }

  /// Print a sorted timing report.
  /// Entries below minPercentage of total time are aggregated into "other".
  void report(std::ostream &os, double minPercentage = 1.0) const;

  /// Return the report as a string.
  std::string reportStr(double minPercentage = 1.0) const;

  /// Sum of all accumulated time across all scopes.
  double getAccountedTime() const;

  /// Nesting depth at which a scope was first entered. Returns -1 if unknown.
  int getDepth(std::string_view key) const {
    auto it = entries.find(std::string(key));
    return it != entries.end() ? it->second.depth : -1;
  }

  /// Wall-clock time since profiling was enabled (or since last reset).
  double getWallTime() const;

  /// Merge another profiler's entries into this one. Sums durations and
  /// counts per key. Asserts that depths match for shared keys.
  /// Does not merge wall-clock time — the caller manages that separately.
  void merge(const Profiler &other);

  /// Clear all accumulated data and restart the wall clock.
  void reset();

private:
  using Clock = std::chrono::high_resolution_clock;

  struct ScopeFrame {
    Entry *entry;
    uint32_t id;
  };

  bool enabled = false;
  uint32_t nextScopeId = 0;
  Clock::time_point wallStart = Clock::now();
  Clock::time_point activeStart;
  std::vector<ScopeFrame> scopeStack;
  std::unordered_map<std::string, Entry> entries;
};

} // namespace raceemulator
