// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// PatternSet: a registry of compiled patterns with deterministic arbitration and
// load-time duplicate detection (plan §7). When several patterns accept the same
// graph, resolution is fixed and declaration order is NEVER consulted:
//   1. explicit priority (higher wins);
//   2. match specificity -- a strictly more-constrained pattern wins (realized
//      as a total order over constraint-like / node / edge counts, so a
//      constraint-superset always outranks its subset);
//   3. a stable id (name hash, else criterion hash), then registration index.
// Two patterns with an identical criterion (same serialized bytes, ignoring
// name/priority) collide and are flagged at load.
//
// Two entry points mirror the plan: firstMatch (cheap, for isApplicable) and
// rankedMatches (all acceptors, best-first, for selection).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <hipdnn_graph_matcher/Matcher.hpp>
#include <hipdnn_graph_matcher/Predicate.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

// One registered pattern plus its arbitration metadata.
struct PatternInfo {
    CompiledPattern pattern;
    std::string name;  // stable id source; may be empty
    int64_t priority = 0;
};

// What add() does when a criterion duplicates one already registered.
enum class DuplicatePolicy : uint8_t {
    Reject,  // add() reports failure, leaves the set unchanged (default)
    Skip     // add() silently drops the duplicate, reports it
};

struct AddResult {
    bool ok = false;         // added (or cleanly skipped)
    bool duplicate = false;  // the criterion was already present
    uint32_t index = 0;      // registration index when added
    std::string error;       // set when !ok
};

// A ranked acceptor: the registered pattern's index plus its match bindings.
struct RankedMatch {
    uint32_t index;
    MatchResult result;
};

// Match-time counters for observability (plan §10). The library only counts;
// wall-time logging is the consumer's, via the log seam below.
struct Metrics {
    uint64_t registered = 0;      // patterns currently registered
    uint64_t duplicates = 0;      // add() calls that hit a duplicate criterion
    uint64_t matchAttempts = 0;   // Matcher::match calls issued
    uint64_t matchSuccesses = 0;  // of those, that matched
    uint64_t budgetAborts = 0;    // of those, that tripped the step budget
};

// Terse log sink the library calls on registration events so an FDE can see
// load activity ("why is startup slow with a thousand drop-ins?"). No logging
// framework is imposed; the consumer owns formatting and wall-time.
using LogFn = std::function<void(std::string_view)>;

class PatternSet {
   public:
    explicit PatternSet(const PredicateRegistry& predicates = PredicateRegistry::builtin())
        : _predicates(predicates) {}

    // Registers a pattern. Duplicate criteria are handled per `policy`. Returns
    // the outcome; on Reject a duplicate yields ok=false, duplicate=true.
    AddResult add(CompiledPattern pattern, std::string name = {}, int64_t priority = 0,
                  DuplicatePolicy policy = DuplicatePolicy::Reject);

    // Sets a log sink invoked on each add() (registered / duplicate). Optional.
    void setLogSink(LogFn sink) {
        _log = std::move(sink);
    }

    // Cumulative counters (registered/duplicates from add, match* from the query
    // methods, which update them even though they are logically const).
    const Metrics& metrics() const noexcept {
        return _metrics;
    }

    size_t size() const noexcept {
        return _entries.size();
    }

    const PatternInfo& at(uint32_t index) const {
        return _entries.at(index);
    }

    // Cheapest applicability check: the first registered pattern that matches, in
    // registration order (no ranking). Returns its index, or -1 if none match.
    // Mirrors ONNX Runtime's greedy first-acceptable assign.
    int64_t firstMatch(const GraphView& graph, const MatchOptions& options = {}) const;

    // All patterns that match `graph`, ordered best-first by the arbitration
    // rule above. Deterministic regardless of registration order.
    std::vector<RankedMatch> rankedMatches(const GraphView& graph,
                                           const MatchOptions& options = {}) const;

   private:
    // Total-order comparison of two entries for arbitration (a ranks before b).
    bool ranksBefore(uint32_t a, uint32_t b) const;

    const PredicateRegistry& _predicates;
    std::vector<PatternInfo> _entries;
    std::vector<uint64_t> _stableId;                    // per entry: name/criterion hash
    std::vector<uint64_t> _criterionHash;               // per entry: dedup key
    std::vector<std::vector<uint8_t>> _criterionBytes;  // per entry: exact dedup tiebreak
    LogFn _log;
    mutable Metrics _metrics;
};

}  // namespace hipdnn::graph_matcher
