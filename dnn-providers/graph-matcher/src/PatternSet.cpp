// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <hipdnn_graph_matcher/PatternCodec.hpp>
#include <hipdnn_graph_matcher/PatternSet.hpp>
#include <tuple>

namespace hipdnn::graph_matcher {

namespace {

uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t fnv1a(std::string_view s) {
    return fnv1a(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

// Specificity as a total order: more constraint-like clauses => more specific;
// ties broken by node count then bound-edge count. A constraint-superset always
// carries strictly more clauses, so it always outranks its subset -- realizing
// "strictly more constrained wins" without a hazardous partial-order comparator.
std::tuple<size_t, size_t, size_t> specificity(const CompiledPattern& p) {
    const size_t clauses = p.constraints().size() + p.predicates().size() + p.dimBindings().size();
    size_t edges = 0;
    for (const auto& node : p.nodes()) {
        edges += node.operandEdges.size() + node.resultEdges.size();
    }
    return {clauses, p.nodeCount(), edges};
}

}  // namespace

AddResult PatternSet::add(CompiledPattern pattern, std::string name, int64_t priority,
                          DuplicatePolicy policy) {
    const std::vector<uint8_t> bytes = PatternCodec::serialize(pattern);
    const uint64_t hash = fnv1a(bytes.data(), bytes.size());

    for (size_t i = 0; i < _entries.size(); ++i) {
        if (_criterionHash[i] == hash && _criterionBytes[i] == bytes) {
            AddResult dup;
            dup.duplicate = true;
            ++_metrics.duplicates;
            if (policy == DuplicatePolicy::Skip) {
                dup.ok = true;
            } else {
                dup.error = "duplicate criterion (same as registered pattern index " +
                            std::to_string(i) + ")";
            }
            if (_log) {
                _log("graph_matcher: duplicate criterion '" + name + "' (matches index " +
                     std::to_string(i) +
                     "), policy=" + (policy == DuplicatePolicy::Skip ? "skip" : "reject"));
            }
            return dup;
        }
    }

    AddResult res;
    res.ok = true;
    res.index = static_cast<uint32_t>(_entries.size());
    _stableId.push_back(name.empty() ? hash : fnv1a(name));
    _criterionHash.push_back(hash);
    _criterionBytes.push_back(bytes);
    if (_log) {
        _log("graph_matcher: registered pattern #" + std::to_string(res.index) + " '" + name +
             "' (" + std::to_string(bytes.size()) + " criterion bytes)");
    }
    _entries.push_back(PatternInfo{std::move(pattern), std::move(name), priority});
    ++_metrics.registered;
    return res;
}

bool PatternSet::ranksBefore(uint32_t a, uint32_t b) const {
    const PatternInfo& ea = _entries[a];
    const PatternInfo& eb = _entries[b];

    // 1. explicit priority (higher first)
    if (ea.priority != eb.priority) {
        return ea.priority > eb.priority;
    }
    // 2. specificity (more specific first)
    const auto sa = specificity(ea.pattern);
    const auto sb = specificity(eb.pattern);
    if (sa != sb) {
        return sa > sb;
    }
    // 3. stable id (name hash, else criterion hash), then exact bytes, then index
    if (_stableId[a] != _stableId[b]) {
        return _stableId[a] < _stableId[b];
    }
    if (_criterionBytes[a] != _criterionBytes[b]) {
        return _criterionBytes[a] < _criterionBytes[b];
    }
    return a < b;
}

int64_t PatternSet::firstMatch(const GraphView& graph, const MatchOptions& options) const {
    for (size_t i = 0; i < _entries.size(); ++i) {
        ++_metrics.matchAttempts;
        const MatchResult r = Matcher::match(_entries[i].pattern, graph, options, _predicates);
        _metrics.budgetAborts += r.budgetExceeded ? 1 : 0;
        if (r.matched) {
            ++_metrics.matchSuccesses;
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

std::vector<RankedMatch> PatternSet::rankedMatches(const GraphView& graph,
                                                   const MatchOptions& options) const {
    std::vector<RankedMatch> matches;
    for (size_t i = 0; i < _entries.size(); ++i) {
        ++_metrics.matchAttempts;
        MatchResult r = Matcher::match(_entries[i].pattern, graph, options, _predicates);
        _metrics.budgetAborts += r.budgetExceeded ? 1 : 0;
        if (r.matched) {
            ++_metrics.matchSuccesses;
            matches.push_back(RankedMatch{static_cast<uint32_t>(i), std::move(r)});
        }
    }
    std::sort(matches.begin(), matches.end(), [this](const RankedMatch& x, const RankedMatch& y) {
        return ranksBefore(x.index, y.index);
    });
    return matches;
}

}  // namespace hipdnn::graph_matcher
