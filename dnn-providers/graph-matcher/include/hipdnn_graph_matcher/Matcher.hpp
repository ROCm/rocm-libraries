// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Matcher: runs a CompiledPattern against a live graph (via GraphView) and
// produces bindings. Stateless and allocation-light; the search is bounded
// backtracking with a step budget so a pathological graph/pattern fails closed
// instead of hanging (plan requirement 9). Phase 1: structural matching only
// (opcodes + variable-unified edges, single-node and linear chains). Constraints
// and predicates are gated in later phases and do not exist here yet.

#include <cstddef>
#include <cstdint>
#include <hipdnn_graph_matcher/CompiledPattern.hpp>
#include <hipdnn_graph_matcher/GraphView.hpp>
#include <string_view>
#include <vector>

namespace hipdnn::graph_matcher {

// The result of a match attempt. Bindings are populated only on a full match;
// a failed or aborted attempt leaves `matched == false` and empty bindings
// (all-or-nothing commit, XLA finding, plan §5.4).
struct MatchResult {
    bool matched = false;

    // Did the search abort on the step budget rather than exhaust the space?
    // (Distinguishes "no match" from "gave up".)
    bool budgetExceeded = false;

    // VarId -> bound tensor UID (index by VarId; size == pattern varCount()).
    std::vector<int64_t> varUids;

    // Pattern node index -> matched graph node index (size == pattern nodeCount()).
    std::vector<uint32_t> nodeMap;

    // VarId -> whether it was bound (an optional operand absent in the graph
    // stays unbound). size == pattern varCount() on a match.
    std::vector<bool> varBound;

    // SymId -> bound symbolic-dim value (size == pattern symCount()).
    std::vector<int64_t> symVals;

    // When MatchOptions::explain is set and the match FAILS, a short structured
    // reason for the closest near-miss: the first constraint/predicate that
    // rejected an otherwise structurally-complete placement, or a structural
    // note if no full placement was reached. Empty on success or when explain is
    // off (zero hot-path cost). Machine-consumable, not prose.
    std::string diagnostic;

    explicit operator bool() const noexcept {
        return matched;
    }

    // Bound UID for a variable, or -1 if unbound (e.g. absent optional) / no match.
    int64_t uidOf(VarId var) const noexcept {
        return (matched && var < varBound.size() && varBound[var]) ? varUids[var] : -1;
    }

    // Bound value for a symbolic dim, or -1 if unbound / no match.
    int64_t symOf(SymId sym) const noexcept {
        return (matched && sym < symVals.size()) ? symVals[sym] : -1;
    }
};

struct MatchOptions {
    // Hard cap on unify attempts before the search fails closed. Generous for
    // real patterns; a runaway backtrack trips it instead of hanging.
    size_t stepBudget = 1u << 20;

    // Record a near-miss `diagnostic` when the match fails. Off by default so
    // the hot path pays nothing; enable for "why didn't my kernel match?".
    bool explain = false;
};

class Matcher {
   public:
    // Finds the first full match of `pattern` in `graph`. Returns a MatchResult
    // whose `matched` is false if none exists or the budget was exceeded.
    // `predicates` resolves the pattern's native predicates; an unresolved
    // predicate fails the match closed.
    static MatchResult match(const CompiledPattern& pattern, const GraphView& graph,
                             const MatchOptions& options = {},
                             const PredicateRegistry& predicates = PredicateRegistry::builtin());
};

}  // namespace hipdnn::graph_matcher
